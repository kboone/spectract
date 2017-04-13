from matplotlib import pyplot as plt
import numpy as np
import sep
from astropy.table import Table, join
from astropy.io import fits
import pickle

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import erf
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.linalg import block_diag


class OpticalModelException(Exception):
    pass


class OpticalModelFitException(OpticalModelException):
    pass


class OpticalModelBoundsException(OpticalModelException):
    pass


def nmad(x, *args, **kwargs):
    return 1.4826 * np.median(
        np.abs(np.asarray(x) - np.median(x, *args, **kwargs)),
        *args, **kwargs
    )


def _calculate_transformation_components_3(x, y, z, order):
    """Calculate the components for a transformation of 3 parameters.
    """
    component_length = -1
    for parameter in [x, y, z]:
        try:
            component_length = len(parameter)
        except TypeError:
            pass

    if component_length == -1:
        components = [1.]
    else:
        components = [np.ones(component_length)]

    for iter_order in range(1, order+1):
        for start_y in range(iter_order+1):
            for start_z in range(start_y, iter_order+1):
                if component_length == -1:
                    new_component = 1.
                else:
                    new_component = np.ones(component_length)

                for i in range(iter_order):
                    if i < start_y:
                        new_component *= x
                    elif i < start_z:
                        new_component *= y
                    else:
                        new_component *= z

                components.append(new_component)

    if component_length == -1:
        components = np.array(components)
    else:
        components = np.vstack(components)

    return components


def _calculate_transformation_components_1(x, order):
    """Calculate the components for a transformation of 1 parameter.
    """
    iter_component = np.ones(len(x))
    components = [iter_component]

    for iter_order in range(order):
        iter_component = iter_component * x
        components.append(iter_component)

    components = np.vstack(components)

    return components


class IfuCcdImage():
    def __init__(self, path, optical_model, transformation=None):
        self.path = path
        self.fits_file = fits.open(path)
        self.optical_model = optical_model
        self.transformation = transformation

        self.observation_id = self.fits_file[0].header['OBSID']

        self.__arc_data = None

    def __str__(self):
        return "IfuCcdImage(%s)" % self.observation_id

    def load_image_with_transformation(self, path):
        """Load an image and transfer the transformation from this image to it.
        """
        new_image = IfuCcdImage(path, self.optical_model,
                                self.transformation)

        return new_image

    def plot(self, **kwargs):
        data = self.fits_file[0].data

        if 'vmin' not in kwargs:
            kwargs['vmin'] = np.percentile(data, 1.)
        if 'vmax' not in kwargs:
            kwargs['vmax'] = np.percentile(data, 99.)

        plt.imshow(data, **kwargs)

    def identify_arc_lines(self, arc_wavelength):
        """Identify arc line locations in the image.

        This function only needs a rough optical model to work. If called on an
        image that isn't an arc image, this function will fail horribly. Don't
        do that.

        Returns an astropy Table with the following columns:
        - wavelength
        - ccd_x
        - ccd_y
        - model_x
        - model_y
        - spaxel_i
        - spaxel_j
        - spaxel_x
        - spaxel_y
        - spaxel_number

        Note that there may be some misassociations.
        """
        data = self.fits_file[0].data

        # First, find the arc line locations using sep.
        # sep requires a specific byte order which fits files are rarely in.
        # Swap the byte order if necessary.
        if data.dtype == '>f4':
            data = data.byteswap().newbyteorder()

        # Background. We will pick up some of the slit light here, so we use a
        # big box in order to mitigate the effect of that. So long as the
        # background is smooth and gives an image with roughly zero mean we are
        # OK here since the arcs are much brighter than the continuum. Do NOT
        # use a background like this for analysis of slit data!
        background = sep.Background(data, bw=256, bh=256)

        sub_data = data - background.back()
        objects = sep.extract(sub_data, 10.0, minarea=4, filter_kernel=None)
        ccd_x = objects['x']
        ccd_y = objects['y']

        # Determine the line locations in the optical model.
        model_x_2d, model_y_2d = \
            self.optical_model.get_all_ccd_coordinates(arc_wavelength)
        num_spaxels = model_x_2d.shape[0]
        model_x = model_x_2d.flatten()
        model_y = model_y_2d.flatten()
        model_lambda = np.tile(arc_wavelength, num_spaxels)

        spaxel_x_single, spaxel_y_single = \
            self.optical_model.get_all_spaxel_xy_coordinates()
        spaxel_i_single, spaxel_j_single = \
            self.optical_model.get_all_spaxel_ij_coordinates()

        spaxel_numbers_single = sorted(self.optical_model.spaxels.keys())

        spaxel_i = np.repeat(spaxel_i_single, len(arc_wavelength))
        spaxel_j = np.repeat(spaxel_j_single, len(arc_wavelength))
        spaxel_x = np.repeat(spaxel_x_single, len(arc_wavelength))
        spaxel_y = np.repeat(spaxel_y_single, len(arc_wavelength))
        spaxel_numbers = np.repeat(spaxel_numbers_single, len(arc_wavelength))

        # Initial catalog match. I scale the y direction slightly when doing
        # matches because the true arc spacing is much larger in that
        # direction. This allows for larger offsets without failure. We run
        # this twice, applying a median X and Y shift after the first pass.
        # This helps with misidentification of lines with close neighbors.
        offset_x = 0.
        offset_y = 0.
        zz = []
        for i in range(2):
            y_scale = 3.
            kdtree_arc = KDTree(
                np.vstack([ccd_x - offset_x, (ccd_y - offset_y) / y_scale]).T
            )
            dist, matches = kdtree_arc.query(
                np.vstack([model_x, model_y / y_scale]).T
            )

            match_ccd_x = ccd_x[matches]
            match_ccd_y = ccd_y[matches]

            offset_x = np.median(match_ccd_x - model_x)
            offset_y = np.median(match_ccd_y - model_y)

            zz.append(matches)

        result = Table({
            'wavelength': model_lambda,
            'ccd_x': match_ccd_x,
            'ccd_y': match_ccd_y,
            'model_x': model_x,
            'model_y': model_y,
            'spaxel_i': spaxel_i,
            'spaxel_j': spaxel_j,
            'spaxel_x': spaxel_x,
            'spaxel_y': spaxel_y,
            'spaxel_number': spaxel_numbers,
        })

        return result

    def get_arc_data(self):
        """Return arc data at a predefined set of wavelength.

        This will only calculate the arc data the first time that it is called.
        """
        if self.__arc_data is None:
            print("TODO: put the arc lines in a proper place!")
            arc_wavelength = [
                # 3133.167,
                3261.05493164,          # GOOD
                3341.4839,
                # 3403.65209961,
                3466.19995117,          # GOOD
                # 3467.6550293,
                3610.50805664,          # GOOD
                # 3612.87304688,
                # 3650.15795898,
                # 3654.84008789,
                4046.56494141,          # GOOD
                4077.8369,              # GOOD
                # 4158.58984375,     why is this bad? Looks great but comes out
                # off by around 0.2 in Y!
                # 4200.67382812,
                # 4347.50585938,
                4358.33496094,          # GOOD
                # 4678.1489,
                4799.91210938,          # GOOD
                5085.82177734,          # GOOD

                # There is an unidentified line at 5449. Maybe 2nd order light?
                # Regardless, it throws everything off. Need to be very careful
                # that the initial model is set reasonably close or the 5460.75
                # line (which is crucial for alignment) will be lost!
                # 5449.1001,
                5460.75,

                5769.598,
                5790.663,
            ]

            self.__arc_data = self.identify_arc_lines(arc_wavelength)

        return self.__arc_data

    def align_to_arc_image(self, reference_image, verbose=False):
        """Align one arc image to another one."""

        ref_arc_data = reference_image.get_arc_data()
        arc_data = self.get_arc_data()

        join_arc_data = join(arc_data, ref_arc_data, ['spaxel_number',
                                                      'wavelength'])

        transformation = OpticalModelTransformation()

        trans_x, trans_y = transformation.fit(
            join_arc_data['ccd_x_1'],
            join_arc_data['ccd_y_1'],
            join_arc_data['ccd_x_2'],
            join_arc_data['ccd_y_2'],
            join_arc_data['spaxel_x_1'],
            join_arc_data['spaxel_y_1'],
            join_arc_data['wavelength'],
            order=3,
            verbose=verbose,
        )

        self.transformation = transformation

    def get_ccd_coordinates(self, spaxel_number, wavelength,
                            apply_transformation=True):
        """Get the CCD coordinates for a given spaxel at a given wavelength.

        If apply_transformation is True, the image transformation from the
        model CCD coordinates to image CCD coordinates will be applied.
        """

        # Calculate the model x and y coordinates.
        model_x, model_y = \
            self.optical_model.get_ccd_coordinates(spaxel_number, wavelength)

        if apply_transformation and self.transformation is not None:
            spaxel_x, spaxel_y = \
                self.optical_model.get_spaxel_xy_coordinates(spaxel_number)

            ccd_x, ccd_y = self.transform_model_to_ccd(
                model_x,
                model_y,
                spaxel_x,
                spaxel_y,
                wavelength,
            )
        else:
            ccd_x = model_x
            ccd_y = model_y

        return ccd_x, ccd_y

    def get_all_ccd_coordinates(self, wavelength, apply_transformation=True):
        """Get the CCD coordinates for all spaxels at a given wavelength.

        The coordinates will be sorted by the spaxel number to ensure a
        consistent order.

        If apply_transformation is True, the image transformation from the
        model CCD coordinates to image CCD coordinates will be applied.
        """

        # Calculate the model x and y coordinates.
        model_x, model_y = \
            self.optical_model.get_all_ccd_coordinates(wavelength)

        if apply_transformation and self.transformation is not None:
            spaxel_x, spaxel_y = \
                self.optical_model.get_all_spaxel_xy_coordinates()

            ccd_x, ccd_y = self.transform_model_to_ccd(
                model_x,
                model_y,
                spaxel_x,
                spaxel_y,
                wavelength,
            )
        else:
            ccd_x = model_x
            ccd_y = model_y

        return ccd_x, ccd_y

    def get_coordinates_for_ccd_y(self, spaxel_number, ccd_y,
                                  apply_transformation=True, precision=1e-4,
                                  verbose=False):
        """Get the wavelength and CCD x coordinate for a given spaxel at a
        given CCD y position.

        If apply_transformation is True, the image transformation from the
        model CCD coordinates to image CCD coordinates will be applied.

        This is a bit tricky... the way that we define the image transformation
        requires the wavelength to calculate the transformation. We iterate to
        get a reasonable approximation but there will be some error in the
        final result! precision sets the acceptable level of precision in
        units of pixels in CCD y.
        """
        if verbose:
            print("Finding coordinates for spaxel #%d at y=%s" %
                  (spaxel_number, ccd_y))

        max_iterations = 20
        converged = False

        old_model_x = -1.
        old_model_y = ccd_y
        old_wavelength = -1.
        old_ccd_x = -1.
        old_ccd_y = -1.

        for iteration in range(max_iterations):
            new_wavelength, new_model_x = \
                self.optical_model.get_coordinates_for_ccd_y(
                    spaxel_number, old_model_y
                )

            if not apply_transformation or self.transformation is None:
                # No transformation required, so model_y == ccd_y. We're done!
                new_ccd_x = new_model_x
                new_ccd_y = old_model_y
                converged = True
                break

            new_ccd_x, new_ccd_y = self.get_ccd_coordinates(
                spaxel_number, new_wavelength
            )

            diff = ccd_y - new_ccd_y
            new_model_y = old_model_y + diff

            max_diff = np.max(np.abs(diff))

            if verbose:
                print("Iteration %d:" % iteration)
                if np.shape(ccd_y):
                    print("  Max deviation: %9.4f" % max_diff)
                else:
                    print("  Target CCD y:   %9.4f" % ccd_y)
                    print("  Current CCD y:  %9.4f (was %9.4f)" %
                          (new_ccd_y, old_ccd_y))
                    print("  Current CCD x:  %9.4f (was %9.4f)" %
                          (new_ccd_x, old_ccd_x))
                    print("  New wavelength: %9.4f (was %9.4f)" %
                          (new_wavelength, old_wavelength))
                    print("  Model y:        %9.4f (was %9.4f)" %
                          (new_model_y, old_model_y))
                    print("  Model x:        %9.4f (was %9.4f)" %
                          (new_model_x, old_model_x))

            if max_diff < precision:
                converged = True
                break

            old_model_x = new_model_x
            old_model_y = new_model_y
            old_wavelength = new_wavelength
            old_ccd_x = new_ccd_x
            old_ccd_y = new_ccd_y

        if not converged:
            # Shouldn't get here for a reasonable precision level. If this
            # happens, rerun with verbose=True to see what is happening.
            raise OpticalModelException(
                "Unable to determine CCD positions for spaxel #%d at y=%s" %
                (spaxel_number, ccd_y)
            )

        return new_wavelength, new_ccd_x

    def scatter_ccd_coordinates(self, wavelength, *args,
                                apply_transformation=True, **kwargs):
        """Make a scatter plot of the CCD positions of all of the spaxels at a
        given wavelength.
        """
        all_x, all_y = self.get_all_ccd_coordinates(
            wavelength, apply_transformation=apply_transformation
        )
        plt.scatter(all_x, all_y, *args, **kwargs)

    def transform_ccd_to_model(self, ccd_x, ccd_y, spaxel_x, spaxel_y,
                               wavelength):
        """Transform CCD coordinates to model CCD coordinates."""

        if self.transformation is None:
            # No transformation defined
            return ccd_x, ccd_y

        model_x, model_y = self.transformation.transform(
            ccd_x,
            ccd_y,
            spaxel_x,
            spaxel_y,
            wavelength,
            reverse=True
        )

        return model_x, model_y

    def transform_model_to_ccd(self, model_x, model_y, spaxel_x, spaxel_y,
                               wavelength):
        """Transform model CCD coordinates to CCD coordinates."""

        if self.transformation is None:
            # No transformation defined
            return model_x, model_y

        ccd_x, ccd_y = self.transformation.transform(
            model_x,
            model_y,
            spaxel_x,
            spaxel_y,
            wavelength
        )

        return ccd_x, ccd_y

    def get_patch(self, spaxel_number, wavelength, width_x=4, width_y=10):
        """Return a patch of data around a given spaxel and wavelength.

        Returns:
            dx: the x distances from the target point
            dy: the y distances from the target point
            patch: the requested patch
        """
        center_x, center_y = \
            self.get_ccd_coordinates(spaxel_number, wavelength)

        min_x = int(np.around(center_x - width_x))
        max_x = int(np.around(center_x + width_x))
        min_y = int(np.around(center_y - width_y))
        max_y = int(np.around(center_y + width_y))

        x_vals = np.arange(min_x, max_x + 1)
        y_vals = np.arange(min_y, max_y + 1)
        dx_vals = x_vals - center_x
        dy_vals = y_vals - center_y

        data = self.fits_file[0].data

        patch = data[min_y:max_y + 1, min_x:max_x + 1]

        return dx_vals, dy_vals, patch

    def get_sum_patch(self, spaxel_number, wavelength, **kwargs):
        """Return a summed patch of data around a given spaxel and wavelength.

        Returns:
            dx: the x distances from the target point
            sum_patch: the patch at the target point summed in the y direction.
        """
        dx_vals, dy_vals, patch = \
            self.get_patch(spaxel_number, wavelength, **kwargs)

        sum_patch = np.sum(patch, axis=0)

        return dx_vals, sum_patch

    def fit_smooth_patch(self, spaxel_number, wavelength, width_x=2.5,
                         width_y=10, fit_background=False, psf_func=None,
                         verbose_level=1):
        """Fit for the position and width of data in a given patch around a
        smooth region of the spectrum.

        We make the following assumptions:
        - The spectrum varies smoothly and can be reasonably represented within
        the patch by f(y) = f_0 + f_1 * (y-y_0).
        - The spectrum's x position can be represented by a linear offset
        within the patch, i.e.: x(y) = x_0 + x_1 * (y-y_0).
        - The spectrum's width is constant over the patch.
        - The background is constant within the patch.

        On a continuum image with a maximum difference in the y direction of
        roughly 20 pixels these assumptions hold to one part in a thousand. Be
        careful with other situations!

        Returns a dictionary with lots of information about the fit and
        results.
        """
        if width_y > 20:
            print("WARNING: fit_smooth_patch makes approximations that aren't"
                  " valid on large patches! You requested a window of %f"
                  " pixels which is probably too big. See the doctring for"
                  " details" % width_y)

        if psf_func is None:
            psf_func = convgauss

        dx_vals, dy_vals, patch = self.get_patch(
            spaxel_number, [wavelength], width_x=width_x, width_y=width_y
        )

        # Figure out the conversion between model y coordinate to model x
        # coordinate. For a window where the amplitude slope approximation is
        # appropriate, we can get away with just a linear relation to within a
        # couple thousandths of a pixel.
        ref_x, ref_y = self.get_ccd_coordinates(
            spaxel_number, [wavelength, wavelength+10]
        )
        center_x = ref_x[0]
        center_y = ref_y[0]

        model_slope = (ref_x[1] - ref_x[0]) / (ref_y[1] - ref_y[0])
        model_dx = dy_vals * model_slope

        mesh_dx, mesh_dy = np.meshgrid(dx_vals, dy_vals)
        mesh_dx, mesh_model_dx = np.meshgrid(dx_vals, model_dx)

        fit_x = mesh_dx - mesh_model_dx
        fit_y = mesh_dy

        def model(amp, amp_slope, mu, sigma, background=0):
            fit_amp = amp + fit_y * amp_slope
            model = psf_func(fit_x, fit_amp, mu, sigma) + background

            return model

        def fit_func(params):
            return np.sum((patch - model(*params))**2)

        start_amp = np.median(np.sum(patch, axis=1))

        start_params = [start_amp, 0., 0., 1.]
        bounds = [
            (0.1*start_amp, 10*start_amp),
            (None, None),
            (-10., 10.),
            (0.2, 3.),
        ]

        if fit_background:
            start_params.append(0.)
            bounds.append((None, None))

        start_params = np.array(start_params)

        res = minimize(
            fit_func,
            start_params,
            method='L-BFGS-B',
            bounds=bounds
        )

        if not res.success:
            raise OpticalModelFitException(res.message)

        fit_params = res.x

        # fit_amp, fit_amp_slope, fit_mu, fit_sigma, fit_mean = fit_params
        # result_model = model(*fit_params)

        # no_mean_fit_params = fit_params.copy()
        # no_mean_fit_params[-1] = 0.
        # result_model_no_mean = model(*no_mean_fit_params)

        if fit_background:
            fit_amp, fit_amp_slope, fit_mu, fit_sigma, background = fit_params
        else:
            fit_amp, fit_amp_slope, fit_mu, fit_sigma = fit_params

        result_model = model(*fit_params)

        do_print = verbose_level >= 2

        if (verbose_level == 1 and
                (fit_amp < 0.2*start_amp or fit_amp > 2.0*start_amp)):
            print("WARNING: Fit amplitude out of normal bounds:")
            do_print = True

        if do_print:
            print("Fit results:")
            print("    Amplitude:  %8.2f (start: %8.2f)" %
                  (fit_amp, start_amp))
            print("    Center:     %8.2f (start: %8.2f)" % (fit_mu, 0.))
            print("    Sigma:      %8.2f (start: %8.2f)" % (fit_sigma, 1.))
            if fit_background:
                print("    Background: %8.2f (start: %8.2f)" %
                      (background, 0.))

        result = {
            'amplitude': fit_amp,
            'amplitude_slope': fit_amp_slope,
            'offset': fit_mu,
            'width': fit_sigma,

            'patch': patch,
            'model': result_model,
            'fit_x': fit_x,
            'fit_y': fit_y,
            'fit_result': res,

            'spaxel_number': spaxel_number,
            'wavelength': wavelength,

            'ccd_x': center_x,
            'ccd_y': center_y,
        }

        if fit_background:
            result['background'] = background

        return result

    def fit_smooth_patch_2d(self, ccd_x, ccd_y, full_psf, core_psf,
                            tail_psf_builder, core_psf_range=3, width_x=30,
                            width_y=10, verbose_level=1):
        """Fit for the the PSF parameters in a 2d patch on the CCD.

        We make the same assumptions as in fit_smooth_patch. Additionally, we
        assume that the CCD positions of the spaxels are already determined and
        we do not fit for them.
        """
        if width_y > 20:
            print("WARNING: fit_smooth_patch_2d makes approximations that "
                  "aren't valid on large patches! You requested a window of %f"
                  " pixels which is probably too big. See the doctring for"
                  " details" % width_y)

        # Get the patch
        data = self.fits_file[0].data

        min_x = int(np.around(ccd_x - width_x))
        max_x = int(np.around(ccd_x + width_x))
        min_y = int(np.around(ccd_y - width_y))
        max_y = int(np.around(ccd_y + width_y))

        data_max_y, data_max_x = data.shape

        min_x = np.clip(min_x, 0, data_max_x - 1)
        max_x = np.clip(max_x, 0, data_max_x - 1)
        min_y = np.clip(min_y, 0, data_max_y - 1)
        max_y = np.clip(max_y, 0, data_max_y - 1)

        x_vals = np.arange(min_x, max_x + 1)
        y_vals = np.arange(min_y, max_y + 1)

        patch = data[min_y:max_y + 1, min_x:max_x + 1]

        patch_dx, patch_dy = np.meshgrid(x_vals - ccd_x, y_vals - ccd_y)

        # Find all spaxels in the patch
        spaxel_numbers = []
        spaxel_waves = []
        spaxel_dxs = []
        spaxel_start_amps = []

        num_spaxels = 0
        for spaxel_number in self.optical_model.spaxel_numbers:
            try:
                spaxel_wave, spaxel_x = self.get_coordinates_for_ccd_y(
                    spaxel_number, y_vals
                )
            except OpticalModelBoundsException:
                continue

            spaxel_dx = np.subtract.outer(x_vals, spaxel_x).T

            if (np.min(spaxel_x) < np.min(x_vals) or np.max(spaxel_x) >
                    np.max(x_vals)):
                continue

            spaxel_idx = np.argmin(np.abs(np.median(spaxel_dx, axis=0)))
            sum_min_idx = np.clip(spaxel_idx - 2, 0, len(x_vals))
            sum_max_idx = np.clip(spaxel_idx + 3, 0, len(x_vals))
            start_amp = np.median(np.sum(patch[:, sum_min_idx:sum_max_idx],
                                         axis=1))

            spaxel_numbers.append(spaxel_number)
            spaxel_waves.append(spaxel_wave)
            spaxel_dxs.append(spaxel_dx)
            spaxel_start_amps.append(start_amp)

            num_spaxels += 1

        spaxel_numbers = np.array(spaxel_numbers)
        spaxel_waves = np.array(spaxel_waves)
        spaxel_dxs = np.array(spaxel_dxs)
        spaxel_start_amps = np.array(spaxel_start_amps)

        def spaxel_model(index, amplitude, amplitude_slope, offset, core_width,
                         tail_psf, tail_fraction):
            fit_amplitude = amplitude + patch_dy * amplitude_slope
            spaxel_dx = spaxel_dxs[index]

            model = full_psf(spaxel_dx, fit_amplitude, offset, core_psf,
                             core_width, tail_psf, tail_fraction)

            return model

        def patch_model(amplitudes, amplitude_slopes, offsets, core_widths,
                        tail_fraction, tail_alpha, tail_beta, background):
            model = np.zeros(patch.shape)

            tail_psf = tail_psf_builder(tail_alpha, tail_beta)

            for i in range(num_spaxels):
                model += spaxel_model(i, amplitudes[i], amplitude_slopes[i],
                                      offsets[i], core_widths[i], tail_psf,
                                      tail_fraction)

            model += background

            return model

        amplitude_scale = 1e4
        amplitude_slope_scale = 1e2
        offset_scale = 1e-2
        background_scale = 1e3

        def parse_parameters(params):
            amplitudes = params[:num_spaxels] * amplitude_scale
            amplitude_slopes = (params[num_spaxels:2*num_spaxels] *
                                amplitude_slope_scale)
            offsets = params[2*num_spaxels:3*num_spaxels] * offset_scale
            core_widths = params[3*num_spaxels:4*num_spaxels]
            tail_fraction, tail_alpha, tail_beta = params[4*num_spaxels:-1]
            background = params[-1] * background_scale

            return (amplitudes, amplitude_slopes, offsets, core_widths,
                    tail_fraction, tail_alpha, tail_beta, background)

        def fit_func(params):
            model = patch_model(*parse_parameters(params))
            return np.sum((patch - model)**2)

        start_params = np.hstack([
            spaxel_start_amps / amplitude_scale,
            np.zeros(num_spaxels) / amplitude_slope_scale,
            np.zeros(num_spaxels) / offset_scale,
            np.ones(num_spaxels),
            0.5,
            2.,
            2.,
            np.percentile(patch, 5) / background_scale,
        ])

        bounds = []

        # Amplitude bounds
        for i in range(num_spaxels):
            bounds.append((0., 10.*spaxel_start_amps[i] / amplitude_scale))

        # Amplitude slope bounds
        for i in range(num_spaxels):
            bounds.append((None, None))

        # Offset bounds
        for i in range(num_spaxels):
            bounds.append((-1. / offset_scale, 1. / offset_scale))

        # Core width bounds
        for i in range(num_spaxels):
            bounds.append((0.5, 3.))

        # Tail bounds
        bounds.append((0., 1.))
        bounds.append((0.2, 10))
        bounds.append((0.5, 10))

        # Background bound
        bounds.append((None, None))

        res = minimize(
            fit_func,
            start_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxfun': 100000}
        )

        if not res.success:
            # raise OpticalModelFitException(res.message)
            print(OpticalModelFitException(res.message))

        fit_params = res.x
        amplitudes, amplitude_slopes, offsets, core_widths, tail_fraction, \
            tail_alpha, tail_beta, background = parse_parameters(fit_params)

        result_model = patch_model(*parse_parameters(fit_params))

        from IPython import embed; embed()


        do_print = verbose_level >= 2

        if (verbose_level == 1 and
                (fit_amp < 0.2*start_amp or fit_amp > 2.0*start_amp)):
            print("WARNING: Fit amplitude out of normal bounds:")
            do_print = True

        if do_print:
            print("Fit results:")
            print("    Amplitude:  %8.2f (start: %8.2f)" %
                  (fit_amp, start_amp))
            print("    Center:     %8.2f (start: %8.2f)" % (fit_mu, 0.))
            print("    Sigma:      %8.2f (start: %8.2f)" % (fit_sigma, 1.))
            if fit_background:
                print("    Background: %8.2f (start: %8.2f)" %
                      (background, 0.))

        result = {
            'amplitude': fit_amp,
            'amplitude_slope': fit_amp_slope,
            'offset': fit_mu,
            'width': fit_sigma,

            'patch': patch,
            'model': result_model,
            'fit_x': fit_x,
            'fit_y': fit_y,
            'fit_result': res,

            'spaxel_number': spaxel_number,
            'wavelength': wavelength,

            'ccd_x': center_x,
            'ccd_y': center_y,
        }

        if fit_background:
            result['background'] = background

        return result



class SpaxelModelFitter():
    def __init__(self):
        self._parameters = None
        self._wavelength_scale = None

    def _calculate_model(self, params, wavelength, return_components=False):
        components = _calculate_transformation_components_1(
            wavelength, self._fit_order
        )

        model = params.dot(components)

        if return_components:
            return (model, components)
        else:
            return model

    def _calculate_target(self, params):
        model = self._calculate_model(
            params, self._fit_wavelength
        )

        # Calculate target function
        diff = (self._fit_vals - model)**2
        target = np.sum(diff[self._fit_mask]) / np.sum(self._fit_mask)

        return target

    def _calculate_gradient(self, params):
        model, components = self._calculate_model(
            params, self._fit_wavelength, True
        )

        # Calculate gradient
        norm = - 2. / np.sum(self._fit_mask)
        all_deriv = norm * components * (self._fit_vals - model)
        grad = np.sum(all_deriv[:, self._fit_mask], axis=1)

        return grad

    def _calculate_hessian(self, params):
        model, components = self._calculate_model(
            params, self._fit_wavelength, True
        )

        # Calculate Hessian
        mask = self._fit_mask
        norm = 2. / np.sum(mask)
        hess = norm * components[:, mask].dot(components[:, mask].T)

        return hess

    def _calculate_scale(self, data):
        center = np.median(data)
        scale = nmad(data)

        scaled_data = (data - center) / scale

        return scaled_data, center, scale

    def _apply_scale(self, data, center, scale):
        scaled_data = (data - center) / scale

        return scaled_data

    def _calculate_clip(self, data, clip_sigma, min_nmad=1e-8):
        """Return a mask which clips the data at a given scatter.

        We allow for a minimum value on the nmad. This ensures that nothing
        breaks when we compare data to itself.
        """
        data_median = np.median(data)
        data_nmad = nmad(data)

        if data_nmad < min_nmad:
            data_nmad = min_nmad

        clip = np.abs(data - data_median) < data_nmad * clip_sigma

        return clip

    def _calculate_num_parameters(self, order):
        """Calculate the number of parameters for a given order of
        transformation.
        """
        num_parameters = int(order) + 1

        return num_parameters

    def fit(self, wavelength, values, order):
        """Fit for a polynomial that maps wavelength to values with
        clipping.
        """
        max_iterations = 10
        initial_clip_sigma = 20.
        clip_sigma = 5.

        self._fit_order = order
        self._num_parameters = self._calculate_num_parameters(order)
        self._transformation_order = order

        scaled_wavelength, *wavelength_scale = \
            self._calculate_scale(wavelength)

        self._wavelength_scale = wavelength_scale

        self._fit_vals = values
        self._fit_wavelength = scaled_wavelength

        # Initial clip
        clip = self._calculate_clip(self._fit_vals, initial_clip_sigma)
        self._fit_mask = clip
        print("Clipped %d objects in initial clip" % np.sum(~clip))

        fit_succeeded = False

        for iteration in range(max_iterations):
            res = minimize(
                self._calculate_target,
                np.zeros(self._num_parameters),
                jac=self._calculate_gradient,
                method='BFGS',
            )

            if not res.success:
                error = ("ERROR: Transformation fit failed: \n\t%s" %
                         res.message)
                raise OpticalModelFitException(error)

            model = self._calculate_model(res.x, self._fit_wavelength)
            diff = self._fit_vals - model

            clip = self._calculate_clip(diff, clip_sigma)

            new_mask = self._fit_mask & clip

            if np.all(self._fit_mask == new_mask):
                print("No clipping required, done at iteration %d." %
                      (iteration+1))
                fit_succeeded = True
                break

            new_clip_count = np.sum(self._fit_mask != new_mask)
            print("Clipped %d objects in iteration %d" % (new_clip_count,
                                                          iteration+1))

            self._fit_mask = new_mask

        if not fit_succeeded:
            error = ("ERROR: Transformation fit did not converge after %d"
                     " iterations!" % max_iterations)
            raise OpticalModelFitException(error)

        self._parameters = res.x
        model = self._calculate_model(self._parameters, self._fit_wavelength)

        return model

    def __call__(self, wavelength):
        """Return the model at a set of wavelengths."""

        if self._wavelength_scale is None or self._parameters is None:
            raise OpticalModelException(
                'Transformation not initialized!'
            )

        # First, rescale our wavelength.
        scaled_wavelength = self._apply_scale(
            wavelength, *self._wavelength_scale
        )

        model = self._calculate_model(
            self._parameters, scaled_wavelength
        )

        return model


class OpticalModelTransformation():
    def __init__(self):
        self._parameters = None
        self._scales = None

    def _calculate_transformation(self, params, ref_x, ref_y, ref_wavelength,
                                  return_components=False):
        params_x = params[:len(params) // 2]
        params_y = params[len(params) // 2:]

        components_x = _calculate_transformation_components_3(
            ref_x, ref_y, ref_wavelength, self._transformation_order
        )
        components_y = _calculate_transformation_components_3(
            ref_x, ref_y, ref_wavelength, self._transformation_order
        )

        offset_x = params_x.dot(components_x)
        offset_y = params_y.dot(components_y)

        if return_components:
            return (offset_x, offset_y, components_x, components_y)
        else:
            return (offset_x, offset_y)

    def _calculate_target(self, params):
        offset_x, offset_y = self._calculate_transformation(
            params, self._fit_ref_x, self._fit_ref_y, self._fit_ref_wavelength
        )

        # Calculate target function
        diff = self._fit_weights * ((self._fit_delta_x + offset_x)**2 +
                                    (self._fit_delta_y + offset_y)**2)
        target = np.sum(diff[self._fit_mask]) / np.sum(self._fit_mask)

        return target

    def _calculate_gradient(self, params):
        offset_x, offset_y, components_x, components_y = \
            self._calculate_transformation(
                params, self._fit_ref_x, self._fit_ref_y,
                self._fit_ref_wavelength, True
            )

        # Calculate gradient
        norm = self._fit_weights * 2. / np.sum(self._fit_mask)
        all_deriv_x = norm * components_x * (self._fit_delta_x + offset_x)
        all_deriv_y = norm * components_y * (self._fit_delta_y + offset_y)
        grad_x = np.sum(all_deriv_x[:, self._fit_mask], axis=1)
        grad_y = np.sum(all_deriv_y[:, self._fit_mask], axis=1)
        grad = np.hstack([grad_x, grad_y])

        return grad

    def _calculate_hessian(self, params):
        offset_x, offset_y, components_x, components_y = \
            self._calculate_transformation(
                params, self._fit_ref_x, self._fit_ref_y,
                self._fit_ref_wavelength, True
            )

        # Calculate Hessian
        mask = self._fit_mask
        norm = self._fit_weights * 2. / np.sum(mask)
        hess_x = norm * components_x[:, mask].dot(components_x[:, mask].T)
        hess_y = norm * components_y[:, mask].dot(components_y[:, mask].T)
        hess = block_diag(hess_x, hess_y)

        return hess

    def _calculate_scale(self, data):
        center = np.median(data)
        scale = nmad(data)

        scaled_data = (data - center) / scale

        return scaled_data, center, scale

    def _apply_scale(self, data, center, scale):
        scaled_data = (data - center) / scale

        return scaled_data

    def _calculate_clip(self, data, wavelengths, clip_sigma, min_nmad=5e-2,
                        max_nmad=5e-1, verbose=False):
        """Return a mask which clips the data at a given scatter.

        We allow for a minimum value on the nmad. This ensures that nothing
        breaks when we compare data to itself.

        We do this by group of wavelengths and return the estimated scatter for
        each one.
        """
        data_median = np.zeros(len(data))
        data_nmad = np.zeros(len(data))

        for wavelength in np.unique(wavelengths):
            wave_cut = wavelengths == wavelength
            wave_data = data[wave_cut]

            wave_data_median = np.median(wave_data)
            wave_data_nmad = nmad(wave_data)

            if wave_data_nmad < min_nmad:
                wave_data_nmad = min_nmad

            if wave_data_nmad > max_nmad:
                wave_data_nmad = max_nmad

            data_median[wave_cut] = wave_data_median
            data_nmad[wave_cut] = wave_data_nmad

            if verbose:
                print("    Wave: %8.2f, Median: %8.4f, NMAD: %8.4f" %
                      (wavelength, wave_data_median, wave_data_nmad))

        clip = np.abs(data - data_median) < data_nmad * clip_sigma

        return clip, data_nmad

    def _calculate_clip_2d(self, data_x, data_y, wavelengths,
                           clip_sigma, verbose=False, **kwargs):
        """Calculate a clip in 2d. See _calculate_clip for details."""
        if verbose:
            print("Clip X:")
        clip_x, nmad_x = self._calculate_clip(
            data_x, wavelengths, clip_sigma, verbose=verbose, **kwargs
        )

        if verbose:
            print("Clip Y:")
        clip_y, nmad_y = self._calculate_clip(
            data_y, wavelengths, clip_sigma, verbose=verbose, **kwargs
        )

        clip = clip_x & clip_y

        return clip, nmad_x, nmad_y


    def _calculate_num_parameters(self, order):
        """Calculate the number of parameters for a given order of
        transformation.
        """
        individual_parameters = int((order+1) * (order+2) * (order+3) / 6)
        total_parameters = individual_parameters * 2

        return total_parameters

    def fit(self, target_x, target_y, orig_x, orig_y, ref_x, ref_y,
            ref_wavelength, order, verbose=False, initial_clip=True):
        """Fit for a transformation between two coordinate sets.

        orig_x and orig_y will be transformed to match target_x and target_y.
        The transformation will be done with terms up to second order in each
        of ref_x, ref_y and ref_wavelength.

        For my purposes, ref_x, ref_y and ref_wavelength are intended to be the
        i and j positions of spaxels and the target wavelength. orig/target x
        and y are intended to be CCD coordinates.
        """
        max_iterations = 10
        initial_clip_sigma = 10.
        clip_sigma = 5.

        self._num_parameters = self._calculate_num_parameters(order)
        self._transformation_order = order

        scaled_ref_x, *scales_x = self._calculate_scale(ref_x)
        scaled_ref_y, *scales_y = self._calculate_scale(ref_y)
        scaled_ref_wavelength, *scales_wavelength = \
            self._calculate_scale(ref_wavelength)

        self._scales = [scales_x, scales_y, scales_wavelength]

        self._fit_target_x = target_x
        self._fit_target_y = target_y
        self._fit_orig_x = orig_x
        self._fit_orig_y = orig_y
        self._fit_ref_x = scaled_ref_x
        self._fit_ref_y = scaled_ref_y
        self._fit_ref_wavelength = scaled_ref_wavelength

        self._fit_delta_x = self._fit_orig_x - self._fit_target_x
        self._fit_delta_y = self._fit_orig_y - self._fit_target_y

        # Initial clip
        if initial_clip:
            clip, nmad_x, nmad_y = self._calculate_clip_2d(
                self._fit_delta_x, self._fit_delta_y, ref_wavelength,
                initial_clip_sigma, verbose=verbose
            )
        else:
            clip = np.ones(len(self._fit_delta_x), dtype=bool)
            nmad_x = np.ones(len(self._fit_delta_x))
            nmad_y = np.ones(len(self._fit_delta_x))

        self._fit_mask = clip
        if verbose:
            print("Clipped %d objects in initial clip" % np.sum(~clip))

        self._fit_weights = np.ones(len(self._fit_delta_x))

        fit_succeeded = False

        for iteration in range(max_iterations):
            res = minimize(
                self._calculate_target,
                np.zeros(self._num_parameters),
                jac=self._calculate_gradient,
                method='BFGS',
            )

            if not res.success:
                error = ("ERROR: Transformation fit failed: \n\t%s" %
                         res.message)
                raise OpticalModelFitException(error)

            offset_x, offset_y = self._calculate_transformation(
                res.x, self._fit_ref_x, self._fit_ref_y,
                self._fit_ref_wavelength
            )
            diff_x = self._fit_delta_x + offset_x
            diff_y = self._fit_delta_y + offset_y

            clip, nmad_x, nmad_y = self._calculate_clip_2d(
                diff_x, diff_y, ref_wavelength, clip_sigma,
                verbose=verbose
            )

            self._fit_weights = 1 / (nmad_x**2 + nmad_y**2)

            new_mask = self._fit_mask & clip

            if np.all(self._fit_mask == new_mask):
                if verbose:
                    print("No clipping required, done at iteration %d." %
                          (iteration+1))
                fit_succeeded = True
                break

            if verbose:
                new_clip_count = np.sum(self._fit_mask != new_mask)
                print("Clipped %d objects in iteration %d" % (new_clip_count,
                                                              iteration+1))

            self._fit_mask = new_mask

        if not fit_succeeded:
            error = ("ERROR: Transformation fit did not converge after %d"
                     " iterations!" % max_iterations)
            raise OpticalModelFitException(error)

        self._parameters = res.x
        offset_x, offset_y = self._calculate_transformation(
            self._parameters, self._fit_ref_x, self._fit_ref_y,
            self._fit_ref_wavelength
        )

        trans_x = self._fit_orig_x + offset_x
        trans_y = self._fit_orig_y + offset_y

        return (trans_x, trans_y)

    def transform(self, orig_x, orig_y, ref_x, ref_y, ref_wavelength,
                  reverse=False):
        """Apply the transformation to a set of x and y coordinates."""

        if self._scales is None or self._parameters is None:
            raise OpticalModelException(
                'Transformation not initialized!'
            )

        # First, rescale our references.
        scales_x, scales_y, scales_wavelength = self._scales
        scaled_ref_x = self._apply_scale(ref_x, *scales_x)
        scaled_ref_y = self._apply_scale(ref_y, *scales_y)
        scaled_ref_wavelength = self._apply_scale(ref_wavelength,
                                                  *scales_wavelength)

        offset_x, offset_y = self._calculate_transformation(
            self._parameters, scaled_ref_x, scaled_ref_y, scaled_ref_wavelength
        )

        if reverse:
            scale = -1.
        else:
            scale = 1.

        trans_x = orig_x + scale * offset_x
        trans_y = orig_y + scale * offset_y

        return (trans_x, trans_y)


def ensure_updated(func):
    """Decorator to ensure that the SpaxelModel interpolators are updated."""
    def decorated_func(self, *args, **kwargs):
        if self._updated is False:
            self.generate_interpolators()
            self._updated = True

        return func(self, *args, **kwargs)

    return decorated_func


class SpaxelModel():
    def __init__(self, spaxel_number, spaxel_i, spaxel_j, spaxel_x, spaxel_y,
                 wavelength, ccd_x, ccd_y, width):
        self._spaxel_number = spaxel_number
        self._spaxel_i = spaxel_i
        self._spaxel_j = spaxel_j
        self._spaxel_x = spaxel_x
        self._spaxel_y = spaxel_y

        self._wavelength = wavelength
        self._ccd_x = ccd_x
        self._ccd_y = ccd_y
        self._width = width

        # Keep track of whether the interpolators are updated or not so that we
        # don't regenerate them unless necessary.
        self._updated = False
        self._interp_wave_to_x = None
        self._interp_wave_to_y = None
        self._interp_y_to_wave = None
        self._interp_y_to_x = None

    def update(self, spaxel_number=None, spaxel_i=None, spaxel_j=None,
               spaxel_x=None, spaxel_y=None, wavelength=None, ccd_x=None,
               ccd_y=None, width=None):
        """Update the model.

        Any of the internal variables can be updated with this call. It is up
        to the user to make sure that they are all the correct length and
        format for now... Make sure that wavelength, ccd_x, ccd_y and width all
        have the same shape or things will break!
        """
        if spaxel_number is not None:
            self._spaxel_number = spaxel_number
        if spaxel_i is not None:
            self._spaxel_i = spaxel_i
        if spaxel_j is not None:
            self._spaxel_j = spaxel_j
        if spaxel_x is not None:
            self._spaxel_x = spaxel_x
        if spaxel_y is not None:
            self._spaxel_y = spaxel_y

        if wavelength is not None:
            self._wavelength = wavelength
        if ccd_x is not None:
            self._ccd_x = ccd_x
        if ccd_y is not None:
            self._ccd_y = ccd_y
        if width is not None:
            self._width = width

        self._updated = False

    def generate_interpolators(self):
        self._interp_wave_to_x = InterpolatedUnivariateSpline(
            self._wavelength, self._ccd_x
        )
        self._interp_wave_to_y = InterpolatedUnivariateSpline(
            self._wavelength, self._ccd_y
        )

        # Ensure the the y-coordinates are (reverse) ordered. This should
        # always be the case unless we're overreaching on our interpolation. If
        # they aren't ordered, the InterpolatedUnivariateSpline does crazy
        # things.
        y_order = np.argsort(self._ccd_y)
        if not np.all(self._ccd_y[y_order] == self._ccd_y[::-1]):
            raise OpticalModelException(
                "CCD y coordinates are not ordered for spaxel %d!" %
                self._spaxel_number
            )

        self._interp_y_to_wave = InterpolatedUnivariateSpline(
            self._ccd_y[y_order], self._wavelength[y_order]
        )
        self._interp_y_to_x = InterpolatedUnivariateSpline(
            self._ccd_y[y_order], self._ccd_x[y_order]
        )

    @ensure_updated
    def get_ccd_coordinates(self, wavelength):
        return (self._interp_wave_to_x(wavelength),
                self._interp_wave_to_y(wavelength))

    @ensure_updated
    def get_coordinates_for_ccd_y(self, ccd_y, max_y_interpolation=20):
        """Returns the wavelength and ccd_x position as a tuple"""
        # Bounds check
        min_y = self._ccd_y[-1] - max_y_interpolation
        max_y = self._ccd_y[0] + max_y_interpolation

        ccd_y = np.asarray(ccd_y)

        if np.any((ccd_y < min_y) | (ccd_y > max_y)):
            raise OpticalModelBoundsException(
                "The model for spaxel %d is only defined for %.1f < y < %.1f" %
                (self._spaxel_number, min_y, max_y)
            )

        return (self._interp_y_to_wave(ccd_y),
                self._interp_y_to_x(ccd_y))

    @property
    def ij_coordinates(self):
        return (self._spaxel_i, self._spaxel_j)

    @property
    def xy_coordinates(self):
        return (self._spaxel_x, self._spaxel_y)

    def apply_shift(self, shift_x, shift_y):
        """Apply a shift in the x and y CCD positions to the model"""
        self._ccd_x += shift_x
        self._ccd_y += shift_y

        self._updated = False


class OpticalModel():
    def __init__(self, model_data):
        """Initialize the OpticalModel

        model_data is a dictionary-like iterable with spaxel number as the key
        and the parameters for a SpaxelModel as the values.
        """
        self.spaxels = {}

        for spaxel_number, spaxel_data in model_data.items():
            spaxel = SpaxelModel(**spaxel_data)
            self.spaxels[spaxel_number] = spaxel

    def write(self, path):
        """Write the optical model to a file"""
        with open(path, 'wb') as outfile:
            pickle.dump(self, outfile)

    @classmethod
    def read(self, path):
        """Read an optical model from a file"""
        with open(path, 'rb') as infile:
            optical_model = pickle.load(infile)

        return optical_model

    @property
    def spaxel_numbers(self):
        """Return an array of all of the spaxel numbers.

        The array will be sorted.
        """
        return sorted(self.spaxels.keys())

    def get_all_spaxel_ij_coordinates(self):
        """Return the spaxel i,j coordinates of every spaxel

        The coordinates will be sorted by the spaxel number.
        """
        all_spaxel_i = []
        all_spaxel_j = []

        for spaxel_number in self.spaxel_numbers:
            spaxel = self.spaxels[spaxel_number]
            spaxel_i, spaxel_j = spaxel.ij_coordinates
            all_spaxel_i.append(spaxel_i)
            all_spaxel_j.append(spaxel_j)

        all_spaxel_i = np.array(all_spaxel_i)
        all_spaxel_j = np.array(all_spaxel_j)

        return all_spaxel_i, all_spaxel_j

    def get_spaxel_ij_coordinates(self, spaxel_number):
        """Return the spaxel i,j coordinates of a given spaxel"""
        return self.spaxels[spaxel_number].ij_coordinates

    def get_all_spaxel_xy_coordinates(self):
        """Return the spaxel x,y coordinates of every spaxel

        The coordinates will be sorted by the spaxel number.
        """
        all_spaxel_x = []
        all_spaxel_y = []

        for spaxel_number in self.spaxel_numbers:
            spaxel = self.spaxels[spaxel_number]
            spaxel_x, spaxel_y = spaxel.xy_coordinates
            all_spaxel_x.append(spaxel_x)
            all_spaxel_y.append(spaxel_y)

        all_spaxel_x = np.array(all_spaxel_x)
        all_spaxel_y = np.array(all_spaxel_y)

        return all_spaxel_x, all_spaxel_y

    def get_spaxel_xy_coordinates(self, spaxel_number):
        """Return the spaxel x,y coordinates of a given spaxel"""
        return self.spaxels[spaxel_number].xy_coordinates

    def get_all_ccd_coordinates(self, wavelength):
        """Return the CCD coordinates of every spaxel at the given wavelength

        The coordinates will be sorted by the spaxel number to ensure a
        consistent order.
        """
        ordered_spaxel_numbers = sorted(self.spaxels.keys())

        all_x = []
        all_y = []

        for spaxel_number in ordered_spaxel_numbers:
            spaxel = self.spaxels[spaxel_number]
            x_coord, y_coord = spaxel.get_ccd_coordinates(wavelength)
            all_x.append(x_coord)
            all_y.append(y_coord)

        all_x = np.array(all_x)
        all_y = np.array(all_y)

        return all_x, all_y

    def get_ccd_coordinates(self, spaxel_number, wavelength):
        """Return the CCD coordinates of a spaxel at the given wavelength"""
        return self.spaxels[spaxel_number].get_ccd_coordinates(wavelength)

    def get_coordinates_for_ccd_y(self, spaxel_number, ccd_y):
        """Returns the wavelength and ccd_x position of a spaxel at the given
        ccd_y position."""
        return self.spaxels[spaxel_number].get_coordinates_for_ccd_y(ccd_y)

    def scatter_ccd_coordinates(self, wavelength, *args, **kwargs):
        """Make a scatter plot of the CCD positions of all of the spaxels at a
        given wavelength.
        """
        all_x, all_y = self.get_all_ccd_coordinates(wavelength)
        plt.scatter(all_x, all_y, *args, **kwargs)

    def update(self, new_spaxel_data):
        """Update the optical model.

        new_spaxel_data should be a dictionary with spaxel numbers as keys and
        SpaxelModel parameters as values. The SpaxelModel will be updated with
        whatever new information is there... make sure that everything is kept
        consistent! (eg: don't update the x and y array without updating the
        wavelength array)
        """
        for spaxel_number, iter_new_spaxel_data in new_spaxel_data.items():
            spaxel = self.spaxels[spaxel_number]
            spaxel.update(**iter_new_spaxel_data)


def convgauss(x, amp, mu, sigma):
    """Evaluate a 1d gaussian convolved with a pixel.

    - x is a 1d array of the x positions.
    - amp is the integral of the gaussian.
    - mu is the center position.
    - sigma is the standard deviation of the Gaussian.
    """
    return (
        amp*0.5*(
            erf((x + 0.5 - mu) / (np.sqrt(2) * sigma)) -
            erf((x - 0.5 - mu) / (np.sqrt(2) * sigma))
        )
    )


def convgauss_gradient(x, amp, mu, sigma, gradient_index):
    """Evaluate the gradient of a 1d gaussian convolved with a pixel.

    - x is a 1d array of the x positions.
    - amp is the integral of the gaussian.
    - mu is the center position.
    - sigma is the standard deviation of the Gaussian.
    - gradient_index is the index of the parameter to take the gradient of.
    """
    if gradient_index == 0:
        # amp. This is easy since amp is just a constant scaling.
        return convgauss(x, amp, mu, sigma) / amp
    elif gradient_index == 1:
        # mu. For this and for sigma we need to work with derivatives of the
        # error function. I worked these out analytically.
        return (
            amp * 0.5 * (-1. / (np.sqrt(2) * sigma)) * 2 / np.sqrt(np.pi) * (
                np.exp(-((x + 0.5 - mu) / (np.sqrt(2) * sigma))**2) -
                np.exp(-((x - 0.5 - mu) / (np.sqrt(2) * sigma))**2)
            )
        )
    elif gradient_index == 2:
        # sigma
        return (
            amp * 0.5 * (-1. / np.sqrt(2) / sigma**2) * 2 / np.sqrt(np.pi) * (
                (
                    (x + 0.5 - mu) *
                    np.exp(-((x + 0.5 - mu) / (np.sqrt(2) * sigma))**2)
                ) - (
                    (x - 0.5 - mu) *
                    np.exp(-((x - 0.5 - mu) / (np.sqrt(2) * sigma))**2)
                )
            )
        )

    # Shouldn't make it here... invalid index
    raise OpticalModelFitException("Invalid gradient index %d!" %
                                   gradient_index)


def multigauss(amp1, mu1, sigma1, amp2, mu2, sigma2):
    """Determine parameters of one gaussian convolved with another.

    Returns amp, mu and sigma for the convolution of the two input gaussians.
    """
    amp = amp1 * amp2
    mu = mu1 + mu2
    sigma = np.sqrt(sigma1**2 + sigma2**2)

    return amp, mu, sigma


def convgauss_2d(mesh_x, mesh_y, amp, mu_x, mu_y, sigma_x, sigma_y):
    """Evaluate a 2d gaussian convolved with a pixel.

    - mesh_x and mesh_y and a 2d grid of the x and y positions (use np.meshgrid
    to generate them).
    - amp is the 2d integral of the gaussian.
    - mu_x and mu_y are the center positions.
    - sigma_x and sigma_y are the standard deviations.
    """
    return (
        1./amp *
        convgauss(mesh_x, amp, mu_x, sigma_x) *
        convgauss(mesh_y, amp, mu_y, sigma_y)
    )


def fit_convgauss(x, y, start_mu, search_mu, start_sigma, search_sigma):
    """Fit a 1D gaussian convolved with a pixel to data"""
    fit_min_mu = start_mu - search_mu
    fit_max_mu = start_mu + search_mu
    fit_min_sigma = start_sigma - search_sigma
    fit_max_sigma = start_sigma + search_sigma

    def model(amp, mu, sigma, mean):
        model = convgauss(x, amp, mu, sigma) + mean

        return model

    def fit_func(params):
        return np.sum((y - model(*params))**2)

    def model_gradient(amp, mu, sigma, mean, gradient_index):
        # This gradient is pretty slow right now. I should probably optimize it
        # or do it in cython or something.
        if gradient_index == 3:
            # mean
            return np.ones(len(x))
        else:
            return convgauss_gradient(x, amp, mu, sigma, gradient_index)

    def fit_func_gradient(params):
        fit_model = model(*params)

        gradient = []
        for i in range(len(params)):
            gradient.append(
                np.sum(-2. * (y - fit_model) * model_gradient(*params, i))
            )

        gradient = np.array(gradient)

        return gradient

    start_amp = np.sum(y)

    start_params = np.array([start_amp, start_mu, start_sigma, 0.])
    bounds = [
        (0.1*start_amp, 10*start_amp),
        (fit_min_mu, fit_max_mu),
        (fit_min_sigma, fit_max_sigma),
        (None, None),
    ]

    res = minimize(
        fit_func,
        start_params,
        jac=fit_func_gradient,
        method='L-BFGS-B',
        bounds=bounds
    )

    if not res.success:
        raise OpticalModelFitException(res.message)

    fit_params = res.x

    fit_amp, fit_mu, fit_sigma, fit_mean = fit_params
    result_model = model(*fit_params)

    no_mean_fit_params = fit_params.copy()
    no_mean_fit_params[-1] = 0.
    result_model_no_mean = model(*no_mean_fit_params)

    do_print = False

    if fit_amp < 0.5*start_amp or fit_amp > 1.5*start_amp:
        print("WARNING: Fit amplitude out of normal bounds:")
        do_print = True

    if do_print:
        print("Fit results:")
        print("    Amplitude: %8.2f (start: %8.2f)" % (fit_amp, start_amp))
        print("    Center:    %8.2f (start: %8.2f)" % (fit_mu, start_mu))
        print("    Sigma:     %8.2f (start: %8.2f)" % (fit_sigma, start_sigma))
        print("    Mean:      %8.2f (start: %8.2f)" % (fit_mean, 0.))
        print("    Residual power fraction: %8.2f" %
              (np.sum((y - result_model)**2) / np.sum(y**2)))

    return {
        'amp': fit_amp,
        'mu': fit_mu,
        'sigma': fit_sigma,
        'mean': fit_mean,
        'model': result_model,
        'model_no_mean': result_model_no_mean,
        'x': x,
        'y': y,
        'fit_result': res,
    }

def fit_convgauss_patch(data, x, y, model_y):
    """Fit a 1D gaussian convolved with a pixel to a patch of data where the
    center varies in the y direction.
    """
    fit_min_mu = start_mu - search_mu
    fit_max_mu = start_mu + search_mu
    fit_min_sigma = start_sigma - search_sigma
    fit_max_sigma = start_sigma + search_sigma

    def model(amp, mu, sigma, mean):
        model = convgauss(x, amp, mu, sigma) + mean

        return model

    def fit_func(params):
        return np.sum((y - model(*params))**2)

    def model_gradient(amp, mu, sigma, mean, gradient_index):
        # This gradient is pretty slow right now. I should probably optimize it
        # or do it in cython or something.
        if gradient_index == 3:
            # mean
            return np.ones(len(x))
        else:
            return convgauss_gradient(x, amp, mu, sigma, gradient_index)

    def fit_func_gradient(params):
        fit_model = model(*params)

        gradient = []
        for i in range(len(params)):
            gradient.append(
                np.sum(-2. * (y - fit_model) * model_gradient(*params, i))
            )

        gradient = np.array(gradient)

        return gradient

    start_amp = np.sum(y)

    start_params = np.array([start_amp, start_mu, start_sigma, 0.])
    bounds = [
        (0.1*start_amp, 10*start_amp),
        (fit_min_mu, fit_max_mu),
        (fit_min_sigma, fit_max_sigma),
        (None, None),
    ]

    res = minimize(
        fit_func,
        start_params,
        jac=fit_func_gradient,
        method='L-BFGS-B',
        bounds=bounds
    )

    if not res.success:
        raise OpticalModelFitException(res.message)

    fit_params = res.x

    fit_amp, fit_mu, fit_sigma, fit_mean = fit_params
    result_model = model(*fit_params)

    no_mean_fit_params = fit_params.copy()
    no_mean_fit_params[-1] = 0.
    result_model_no_mean = model(*no_mean_fit_params)

    do_print = False

    if fit_amp < 0.5*start_amp or fit_amp > 1.5*start_amp:
        print("WARNING: Fit amplitude out of normal bounds:")
        do_print = True

    if do_print:
        print("Fit results:")
        print("    Amplitude: %8.2f (start: %8.2f)" % (fit_amp, start_amp))
        print("    Center:    %8.2f (start: %8.2f)" % (fit_mu, start_mu))
        print("    Sigma:     %8.2f (start: %8.2f)" % (fit_sigma, start_sigma))
        print("    Mean:      %8.2f (start: %8.2f)" % (fit_mean, 0.))
        print("    Residual power fraction: %8.2f" %
              (np.sum((y - result_model)**2) / np.sum(y**2)))

    return {
        'amp': fit_amp,
        'mu': fit_mu,
        'sigma': fit_sigma,
        'mean': fit_mean,
        'model': result_model,
        'model_no_mean': result_model_no_mean,
        'x': x,
        'y': y,
        'fit_result': res,
    }


def fit_convgauss_full_image(data, y, start_mu, search_mu, start_sigma,
                             search_sigma):
    """Fit a 1D gaussian convolved with a pixel to data"""
    fit_min_mu = int(np.around(start_mu - search_mu))
    fit_max_mu = int(np.around(start_mu + search_mu))
    fit_min_sigma = start_sigma - search_sigma
    fit_max_sigma = start_sigma + search_sigma

    model_x = slice(fit_min_mu, fit_max_mu+1)
    model_y = y
    fit_data = data[model_y, model_x].copy()

    x_vals = np.arange(fit_min_mu, fit_max_mu+1)

    def model(amp, mu, sigma, mean):
        model = convgauss(x_vals, amp, mu, sigma) + mean

        return model

    def fit_func(x):
        return np.sum((fit_data - model(*x))**2)

    use_start_mu = fit_min_mu + np.argmax(fit_data)
    start_amp = np.sum(fit_data)

    start_params = np.array([start_amp, use_start_mu, start_sigma, 0.])
    bounds = [
        (0.1*start_amp, 10*start_amp),
        (fit_min_mu, fit_max_mu),
        (fit_min_sigma, fit_max_sigma),
        (None, None),
    ]

    res = minimize(
        fit_func,
        start_params,
        method='L-BFGS-B',
        bounds=bounds
    )

    if not res.success:
        raise OpticalModelFitException(res.message)

    fit_params = res.x

    fit_amp, fit_mu, fit_sigma, fit_mean = fit_params
    result_model = model(*fit_params)

    no_mean_fit_params = fit_params.copy()
    no_mean_fit_params[-1] = 0.
    result_model_no_mean = model(*no_mean_fit_params)

    do_print = False

    if fit_amp < 0.5*start_amp or fit_amp > 1.5*start_amp:
        print("WARNING: Fit amplitude out of normal bounds:")
        do_print = True

    if do_print:
        print("Fit results:")
        print("    Amplitude: %8.2f (start: %8.2f)" % (fit_amp, start_amp))
        print("    Center:    %8.2f (start: %8.2f)" % (fit_mu, use_start_mu))
        print("    Sigma:     %8.2f (start: %8.2f)" % (fit_sigma, start_sigma))
        print("    Mean:      %8.2f (start: %8.2f)" % (fit_mean, 0.))
        print("    Residual power fraction: %8.2f" %
              (np.sum((fit_data - result_model)**2) / np.sum(fit_data**2)))

    return {
        'amp': fit_amp,
        'mu': fit_mu,
        'sigma': fit_sigma,
        'mean': fit_mean,
        'model': result_model,
        'model_no_mean': result_model_no_mean,
        'model_x': model_x,
        'model_y': model_y,
        'fit_data': fit_data,
    }


def fit_convgauss_2d(data, start_x, start_y, search_x, search_y,
                     start_sigma_x=1., start_sigma_y=1.):
    """Fit a 2D gaussian convolved with a pixel to data"""
    fit_min_x = int(np.around(start_x - search_x))
    fit_max_x = int(np.around(start_x + search_x))
    fit_min_y = int(np.around(start_y - search_y))
    fit_max_y = int(np.around(start_y + search_y))

    fit_data = data[fit_min_y:fit_max_y+1, fit_min_x:fit_max_x+1].copy()

    x_vals = np.arange(fit_min_x, fit_max_x+1)
    y_vals = np.arange(fit_min_y, fit_max_y+1)

    mesh_x, mesh_y = np.meshgrid(x_vals, y_vals)

    def model(amp, mu_x, mu_y, sigma_x, sigma_y):
        return convgauss_2d(mesh_x, mesh_y, amp, mu_x, mu_y, sigma_x,
                            sigma_y)

    def fit_func(x):
        return np.sum((fit_data - model(*x))**2)

    max_loc = np.argmax(fit_data)
    start_mu_x = mesh_x.flat[max_loc]
    start_mu_y = mesh_y.flat[max_loc]
    start_amp = np.sum(fit_data)

    start_params = np.array([start_amp, start_mu_x, start_mu_y,
                             start_sigma_x, start_sigma_y])
    bounds = [
        (0.1*start_amp, 10*start_amp),
        (fit_min_x, fit_max_x),
        (fit_min_y, fit_max_y),
        (0.5, 1.5),
        (0.5, 1.5),
    ]

    res = minimize(
        fit_func,
        start_params,
        method='L-BFGS-B',
        bounds=bounds
    )

    if not res.success:
        raise OpticalModelFitException(res.message)

    fit_params = res.x

    fit_amp, fit_mu_x, fit_mu_y, fit_sigma_x, fit_sigma_y = fit_params
    result_model = model(*fit_params)

    do_print = False

    if fit_amp < 0.5*start_amp or fit_amp > 1.5*start_amp:
        print("WARNING: Arc fit amplitude out of normal bounds:")
        do_print = True

    if do_print:
        print("Fit results:")
        print("    Amplitude: %8.2f (start: %8.2f)" % (fit_amp, start_amp))
        print("    Center X:  %8.2f (start: %8.2f)" % (fit_mu_x, start_mu_x))
        print("    Center Y:  %8.2f (start: %8.2f)" % (fit_mu_y, start_mu_y))
        print("    Sigma X:   %8.2f (start: %8.2f)" % (fit_sigma_x,
              start_sigma_x))
        print("    Sigma Y:   %8.2f (start: %8.2f)" % (fit_sigma_y,
              start_sigma_y))
        print("    Residual power fraction: %8.2f" %
              (np.sum((fit_data - result_model)**2) / np.sum(fit_data**2)))

    return {
        'amp': fit_amp,
        'x': fit_mu_x,
        'y': fit_mu_y,
        'sigma_x': fit_sigma_x,
        'sigma_y': fit_sigma_y,
        'model': result_model,
        'fit_data': fit_data,
    }
