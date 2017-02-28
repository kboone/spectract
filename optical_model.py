from matplotlib import pyplot as plt
import numpy as np
import tqdm
import sep
from astropy.table import Table
from functools import partial

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import erf
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.linalg import block_diag


class OpticalModelException(Exception):
    pass


class OpticalModelFitException(OpticalModelException):
    pass


def nmad(x, *args, **kwargs):
    return 1.4826 * np.median(
        np.abs(np.asarray(x) - np.median(x, *args, **kwargs)),
        *args, **kwargs
    )


def _calculate_order_2_transformation(c, x, y, z, return_components=False):
    """Calculate a transformation with terms up to 2nd order in 3 parameters.

    c is a list of 10 components describing the transformation

    If return_coefficients is True then the components that the c array
    multiplies is also returned (useful for calculating gradients).
    """
    components = np.vstack([
        np.ones(len(x)),
        x,
        y,
        z,
        x * x,
        y * y,
        z * z,
        x * y,
        y * z,
        z * x,
    ])

    res = c.dot(components)

    if return_components:
        return (res, components)
    else:
        return res


class OpticalModelTransformation():
    def __init__(self):
        self._parameters = None
        self._scales = None

    def _calculate_transformation(self, params, ref_x, ref_y, ref_z,
                                  return_components=False):
        params_x = params[:len(params) // 2]
        params_y = params[len(params) // 2:]

        offset_x, components_x = _calculate_order_2_transformation(
            params_x, ref_x, ref_y, ref_z, return_components=True
        )
        offset_y, components_y = _calculate_order_2_transformation(
            params_y, ref_x, ref_y, ref_z, return_components=True
        )

        if return_components:
            return (offset_x, offset_y, components_x, components_y)
        else:
            return (offset_x, offset_y)

    def _calculate_target(self, params):
        offset_x, offset_y = self._calculate_transformation(
            params, self._fit_ref_x, self._fit_ref_y, self._fit_ref_z
        )

        # Calculate target function
        diff = ((self._fit_delta_x + offset_x)**2 +
                (self._fit_delta_y + offset_y)**2)
        target = np.sum(diff[self._fit_mask]) / np.sum(self._fit_mask)

        return target

    def _calculate_gradient(self, params):
        offset_x, offset_y, components_x, components_y = \
            self._calculate_transformation(
                params, self._fit_ref_x, self._fit_ref_y, self._fit_ref_z, True
            )

        # Calculate gradient
        norm = 2. / np.sum(self._fit_mask)
        all_deriv_x = norm * components_x * (self._fit_delta_x + offset_x)
        all_deriv_y = norm * components_y * (self._fit_delta_y + offset_y)
        grad_x = np.sum(all_deriv_x[:, self._fit_mask], axis=1)
        grad_y = np.sum(all_deriv_y[:, self._fit_mask], axis=1)
        grad = np.hstack([grad_x, grad_y])

        return grad

    def _calculate_hessian(self, params):
        offset_x, offset_y, components_x, components_y = \
            self._calculate_transformation(
                params, self._fit_ref_x, self._fit_ref_y, self._fit_ref_z, True
            )

        # Calculate Hessian
        mask = self._fit_mask
        norm = 2. / np.sum(mask)
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

    def fit(self, target_x, target_y, orig_x, orig_y, ref_x, ref_y, ref_z):
        """Fit for a transformation between two coordinate sets.

        orig_x and orig_y will be transformed to match target_x and target_y.
        The transformation will be done with terms up to second order in each
        of ref_x, ref_y and ref_z.

        For my purposes, ref_x, ref_y and ref_z are intended to be the i and j
        positions of spaxels and the target wavelength. orig/target x and y are
        intended to be CCD coordinates.
        """
        max_iterations = 10
        clip_sigma = 5.

        scaled_ref_x, *scales_x = self._calculate_scale(ref_x)
        scaled_ref_y, *scales_y = self._calculate_scale(ref_y)
        scaled_ref_z, *scales_z = self._calculate_scale(ref_z)

        self._scales = [scales_x, scales_y, scales_z]

        self._fit_target_x = target_x
        self._fit_target_y = target_y
        self._fit_orig_x = orig_x
        self._fit_orig_y = orig_y
        self._fit_ref_x = scaled_ref_x
        self._fit_ref_y = scaled_ref_y
        self._fit_ref_z = scaled_ref_z

        self._fit_delta_x = self._fit_orig_x - self._fit_target_x
        self._fit_delta_y = self._fit_orig_y - self._fit_target_y

        self._fit_mask = np.ones(len(target_x), dtype=bool)

        fit_succeeded = False

        for iteration in range(max_iterations):
            res = minimize(
                self._calculate_target,
                np.zeros(20),
                jac=self._calculate_gradient,
                method='BFGS',
            )

            if not res.success:
                error = ("ERROR: Transformation fit failed: \n\t%s" %
                         res.message)
                raise OpticalModelFitException(error)

            offset_x, offset_y = self._calculate_transformation(
                res.x, self._fit_ref_x, self._fit_ref_y, self._fit_ref_z
            )
            diff_x = self._fit_delta_x + offset_x
            diff_y = self._fit_delta_y + offset_y

            median_x_diff = np.median(diff_x)
            nmad_x_diff = nmad(diff_x)
            median_y_diff = np.median(diff_y)
            nmad_y_diff = nmad(diff_y)

            # Don't break when comparing an image to itself.
            min_nmad = 1e-8
            if nmad_x_diff < min_nmad:
                nmad_x_diff = min_nmad
            if nmad_y_diff < min_nmad:
                nmad_y_diff = min_nmad

            clip = (
               ((diff_x - median_x_diff) < nmad_x_diff * clip_sigma) &
               ((diff_y - median_y_diff) < nmad_y_diff * clip_sigma)
            )

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
        offset_x, offset_y = self._calculate_transformation(
            self._parameters, self._fit_ref_x, self._fit_ref_y, self._fit_ref_z
        )

        trans_x = self._fit_orig_x + offset_x
        trans_y = self._fit_orig_y + offset_y

        return (trans_x, trans_y)

    def transform(self, orig_x, orig_y, ref_x, ref_y, ref_z):
        """Apply the transformation to a set of x and y coordinates."""

        if self._scales is None or self._parameters is None:
            raise OpticalModelException(
                'Transformation not initialized!'
            )

        # First, rescale our references.
        scales_x, scales_y, scales_z = self._scales
        scaled_ref_x = self._apply_scale(ref_x, *scales_x)
        scaled_ref_y = self._apply_scale(ref_y, *scales_y)
        scaled_ref_z = self._apply_scale(ref_z, *scales_z)

        offset_x, offset_y = self._calculate_transformation(
            self._parameters, scaled_ref_x, scaled_ref_y, scaled_ref_z
        )

        trans_x = orig_x + offset_x
        trans_y = orig_y + offset_y

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
    def __init__(self, number, i_coord, j_coord, x_mla, y_mla, wave, x_ccd,
                 y_ccd, sigma):
        self._number = number
        self._i_coord = i_coord
        self._j_coord = j_coord
        self._x_mla = x_mla
        self._y_mla = y_mla

        self._wave = wave
        self._x_ccd = x_ccd
        self._y_ccd = y_ccd
        self._sigma = sigma

        # Keep track of whether the interpolators are updated or not so that we
        # don't regenerate them unless necessary.
        self._updated = False
        self._interp_wave_to_x = None
        self._interp_wave_to_y = None
        self._interp_y_to_wave = None

    def generate_interpolators(self):
        self._interp_wave_to_x = \
            InterpolatedUnivariateSpline(self._wave, self._x_ccd)
        self._interp_wave_to_y = \
            InterpolatedUnivariateSpline(self._wave, self._y_ccd)
        self._interp_y_to_wave = \
            InterpolatedUnivariateSpline(self._y_ccd, self._wave)

    @ensure_updated
    def get_ccd_coordinates(self, wave):
        return (self._interp_wave_to_x(wave), self._interp_wave_to_y(wave))

    def get_ij_coordinates(self):
        return (self._i_coord, self._j_coord)

    def apply_shift(self, shift_x, shift_y):
        """Apply a shift in x and y to the model"""
        self._x_ccd += shift_x
        self._y_ccd += shift_y

        self._updated = False


class OpticalModel():
    def __init__(self, model_data):
        """Initialize the OpticalModel

        model_data is a dictionary-like iterable with spaxel number as the key
        and the parameters for a SpaxelModel as the values.
        """
        self._spaxels = {}

        for spaxel_number, spaxel_data in model_data.items():
            spaxel = SpaxelModel(**spaxel_data)
            self._spaxels[spaxel_number] = spaxel

    def get_ij_coordinates(self):
        """Return the i,j coordinates of every spaxel

        The coordinates will be sorted by the spaxel number to ensure a
        consistent order.
        """
        ordered_spaxel_numbers = sorted(self._spaxels.keys())

        all_i = []
        all_j = []

        for spaxel_number in ordered_spaxel_numbers:
            spaxel = self._spaxels[spaxel_number]
            i_coord, j_coord = spaxel.get_ij_coordinates()
            all_i.append(i_coord)
            all_j.append(j_coord)

        all_i = np.array(all_i)
        all_j = np.array(all_j)

        return all_i, all_j

    def get_ccd_coordinates(self, wavelength):
        """Return the CCD coordinates of every spaxel at the given wavelength

        The coordinates will be sorted by the spaxel number to ensure a
        consistent order.
        """
        ordered_spaxel_numbers = sorted(self._spaxels.keys())

        all_x = []
        all_y = []

        for spaxel_number in ordered_spaxel_numbers:
            spaxel = self._spaxels[spaxel_number]
            x_coord, y_coord = spaxel.get_ccd_coordinates(wavelength)
            all_x.append(x_coord)
            all_y.append(y_coord)

        all_x = np.array(all_x)
        all_y = np.array(all_y)

        return all_x, all_y

    def scatter_ccd_coordinates(self, wavelength, *args, **kwargs):
        """Make a scatter plot of the CCD positions of all of the spaxels at a
        given wavelength.
        """
        all_x, all_y = self.get_ccd_coordinates(wavelength)
        plt.scatter(all_x, all_y, *args, **kwargs)

    def find_global_shifts_from_arc(self, arc_data, wavelength, search_x=10,
                                    search_y=20):
        """Find the global shifts to line up with spaxel positions from an arc.

        This function will move the spaxel positions around in the x/y
        direction as a block and is intented to be used for initial alignment.
        This will only work if the target arc is the brightest thing within the
        given search box.
        """
        all_shift_x = []
        all_shift_y = []

        print("Fitting for global shifts from arc")

        for spaxel_number, spaxel in tqdm.tqdm(self._spaxels.items()):
            start_x, start_y = spaxel.get_ccd_coordinates(wavelength)

            fit_results = fit_convgauss_2d(
                arc_data, start_x, start_y, search_x, search_y,
                start_sigma_x=1., start_sigma_y=1.
            )

            fit_x = fit_results['x']
            fit_y = fit_results['y']

            shift_x = fit_x - start_x
            shift_y = fit_y - start_y

            spaxel.apply_shift(shift_x, shift_y)

            all_shift_x.append(shift_x)
            all_shift_y.append(shift_y)

        all_shift_x = np.array(all_shift_x)
        all_shift_y = np.array(all_shift_y)

        print("    X shifts: median = %5.2f, min = %5.2f, max = %5.2f" %
              (np.median(all_shift_x), np.min(all_shift_x),
               np.max(all_shift_x)))
        print("    Y shifts: median = %5.2f, min = %5.2f, max = %5.2f" %
              (np.median(all_shift_y), np.min(all_shift_y),
               np.max(all_shift_y)))

    def identify_arc_lines(self, arc_data, arc_wave):
        """Identify arc line locations based on a rough optical model.

        Returns an astropy Table with the following columns:
        - wave
        - arc_x
        - arc_y
        - model_x
        - model_y
        - model_i
        - model_j

        Note that there may be some misassociations.
        """

        # First, find the arc line locations using sep.
        # sep requires a specific byte order which fits files are rarely in.
        # Swap the byte order if necessary.
        if arc_data.dtype == '>f4':
            arc_data = arc_data.byteswap().newbyteorder()

        # Background. We will pick up some of the slit light here, so we use a
        # big box in order to mitigate the effect of that. So long as the
        # background is smooth and gives an image with roughly zero mean we are
        # OK here since the arcs are much brighter than the continuum. Do NOT
        # use a background like this for analysis of slit data!
        background = sep.Background(arc_data, bw=256, bh=256)

        sub_data = arc_data - background.back()
        objects = sep.extract(sub_data, 10.0, minarea=4)
        arc_x = objects['x']
        arc_y = objects['y']

        # Determine the line locations in the optical model.
        model_x_2d, model_y_2d = self.get_ccd_coordinates(arc_wave)
        num_spaxels = model_x_2d.shape[0]
        model_x = model_x_2d.flatten()
        model_y = model_y_2d.flatten()
        model_lambda = np.tile(arc_wave, num_spaxels)

        spaxel_i_single, spaxel_j_single = self.get_ij_coordinates()
        spaxel_i = np.repeat(spaxel_i_single, len(arc_wave))
        spaxel_j = np.repeat(spaxel_j_single, len(arc_wave))

        # Initial catalog match. I scale the y direction slightly when doing
        # matches because the true arc spacing is much larger in that
        # direction. This allows for larger offsets without failure.
        y_scale = 3.
        kdtree_arc = KDTree(np.vstack([arc_x, arc_y / y_scale]).T)
        dist, matches = kdtree_arc.query(
            np.vstack([model_x, model_y / y_scale]).T
        )

        match_arc_x = arc_x[matches]
        match_arc_y = arc_y[matches]

        result = Table({
            'wave': model_lambda,
            'arc_x': match_arc_x,
            'arc_y': match_arc_y,
            'model_x': model_x,
            'model_y': model_y,
            'spaxel_i': spaxel_i,
            'spaxel_j': spaxel_j,
        })

        return result

    def align_to_arc(self, arc_data, arc_wave):
        """Align the optical model to an arc"""

        data = self.identify_arc_lines(arc_data, arc_wave)

        trans_x, trans_y = fit_transformation(
            data['arc_x'], data['arc_y'], data['model_x'], data['model_y'],
            data['spaxel_i'], data['spaxel_j'], data['wave']
        )

        return trans_x, trans_y


def convgauss(x, amp, mu, sigma):
    return (
        amp*0.5*(
            erf((x + 0.5 - mu) / (np.sqrt(2) * sigma)) -
            erf((x - 0.5 - mu) / (np.sqrt(2) * sigma))
        )
    )


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
