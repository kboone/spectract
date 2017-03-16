import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table

from optical_model import OpticalModel, OpticalModelTransformation, \
    IfuCcdImage, OpticalModelException, fit_convgauss, OpticalModelFitException

plt.ion()


class OpticalModelBuildException(OpticalModelException):
    pass


# Set up the default optical model
slice_locations = np.genfromtxt('./slice_locations.txt', dtype=None)
optics_spaxel = slice_locations['f0']
optics_xmla = slice_locations['f2']
optics_ymla = slice_locations['f4']
optics_lambda = slice_locations['f6']
optics_xccd = slice_locations['f8'] - 1
optics_yccd = slice_locations['f10'] - 1

spaxel_coordinates = np.genfromtxt('./spaxel_coordinates.txt')

spaxel_data = {}

print("Loading analytic optical model")
for spaxel in np.unique(optics_spaxel):
    match = optics_spaxel == spaxel

    if np.sum(match) == 0:
        print("Didn't find %d" % spaxel)
        continue

    slice_xccd = optics_xccd[match]
    slice_yccd = optics_yccd[match]
    slice_xmla = optics_xmla[match]
    slice_ymla = optics_ymla[match]
    slice_lambda = optics_lambda[match]

    slice_width = np.ones(len(slice_xccd))

    match_2 = np.where((spaxel_coordinates[:, 0] == spaxel))[0][0]
    spaxel_i, spaxel_j = spaxel_coordinates[match_2, 1:]

    spaxel_data[spaxel] = {
        'spaxel_number': spaxel,
        'spaxel_i': spaxel_i.astype(int),
        'spaxel_j': spaxel_j.astype(int),
        'spaxel_x': spaxel_i,
        'spaxel_y': spaxel_j,
        'wavelength': slice_lambda,
        'ccd_x': slice_xccd,
        'ccd_y': slice_yccd,
        'width': slice_width
    }

optical_model = OpticalModel(spaxel_data)


# Align several images to each other.
paths = [
    # Ref
    ('./data/P05_139_046_010_07_B.fits', './data/P05_139_046_011_03_B.fits'),
    ('./data/P06_151_149_010_07_B.fits', './data/P06_151_149_011_03_B.fits'),
    ('./data/P07_110_148_002_07_B.fits', './data/P07_110_148_003_03_B.fits'),
    ('./data/P08_231_109_010_07_B.fits', './data/P08_231_109_011_03_B.fits'),
    ('./data/P09_184_174_010_07_B.fits', './data/P09_184_172_011_03_B.fits'),
    ('./data/P10_172_091_010_07_B.fits', './data/P10_172_091_011_03_B.fits'),
    ('./data/P11_267_082_002_07_B.fits', './data/P11_267_082_003_03_B.fits'),
    ('./data/P13_129_092_010_07_B.fits', './data/P13_129_092_011_03_B.fits'),

    # Don't use:
    # Bad fits file??
    # ('./data/P15_089_087_010_07_B.fits', './data/P15_089_087_011_03_B.fits'),
]

ref_idx = 3

ref_cont_path, ref_arc_path = paths[ref_idx]

print("Using %s as the reference arc" % ref_arc_path)

ref_arc_image = IfuCcdImage(ref_arc_path, optical_model)
ref_cont_image = IfuCcdImage(ref_cont_path, optical_model)

all_arc_data = []
arc_images = []
cont_images = []

for i, (cont_path, arc_path) in enumerate(paths):
    if i == ref_idx:
        arc_image = ref_arc_image
        cont_image = ref_cont_image
        arc_data = arc_image.get_arc_data()

        arc_data['ref_ccd_x'] = arc_data['ccd_x']
        arc_data['ref_ccd_y'] = arc_data['ccd_y']
    else:
        print("Aligning %s to %s" % (arc_path, ref_arc_path))

        arc_image = IfuCcdImage(arc_path, optical_model)
        arc_image.align_to_arc_image(ref_arc_image)

        cont_image = arc_image.load_image_with_transformation(cont_path)

        arc_data = arc_image.get_arc_data()

        # Transform the arc data into the reference coordinate frame.
        ref_ccd_x, ref_ccd_y = arc_image.transform_ccd_to_model(
            arc_data['ccd_x'], arc_data['ccd_y'], arc_data['spaxel_x'],
            arc_data['spaxel_y'], arc_data['wavelength']
        )
        arc_data['ref_ccd_x'] = ref_ccd_x
        arc_data['ref_ccd_y'] = ref_ccd_y

    arc_data['image_index'] = i

    arc_images.append(arc_image)
    cont_images.append(cont_image)
    all_arc_data.append(arc_data)

all_arc_data = np.hstack(all_arc_data)


print('Fitting for initial optical model update from arcs')
initial_model_transformation = OpticalModelTransformation()
initial_fit_x, initial_fit_y = initial_model_transformation.fit(
    all_arc_data['ref_ccd_x'],
    all_arc_data['ref_ccd_y'],
    np.zeros(len(all_arc_data)),
    np.zeros(len(all_arc_data)),
    all_arc_data['spaxel_x'],
    all_arc_data['spaxel_y'],
    all_arc_data['wavelength'],
    order=5,
    verbose=True
)


def plot_initial_model_test(spaxel_i, spaxel_j, mode=0, new_figure=True):
    """Do a test plot of the initial model.

    Modes:
        0: x
        1: y
    """
    cut = (
        (all_arc_data['spaxel_i'] == spaxel_i) &
        (all_arc_data['spaxel_j'] == spaxel_j)
    )

    cut_arc_data = all_arc_data[cut]

    if len(cut_arc_data) == 0:
        raise OpticalModelBuildException("No data found for spaxel (%d, %d)" %
                                         (spaxel_i, spaxel_j))

    spaxel_x = cut_arc_data[0]['spaxel_x']
    spaxel_y = cut_arc_data[0]['spaxel_y']

    wave = np.arange(2500, 6000)

    model_x, model_y = initial_model_transformation.transform(
        0., 0., spaxel_x, spaxel_y, wave
    )

    # Figure out what to plot
    to_scatter_x = cut_arc_data['wavelength']
    to_scatter_c = cut_arc_data['image_index']

    if mode == 0:
        model = model_x
        to_scatter_y = cut_arc_data['ref_ccd_x']
    elif mode == 1:
        model = model_y
        to_scatter_y = cut_arc_data['ref_ccd_y']

    if new_figure:
        plt.figure()

    plt.plot(wave, model, label='Initial model fit')
    plt.scatter(to_scatter_x, to_scatter_y, c=to_scatter_c, label='Arc data')
    plt.xlabel('Wavelength $\AA$')
    plt.ylabel('Pixels')
    plt.legend()


print("Applying initial optical model update from arcs")
optical_model_wavelength = np.arange(2500, 6000, 20.)

new_spaxel_data = {}

for spaxel_number in optical_model.spaxel_numbers:
    spaxel_x, spaxel_y = \
        optical_model.get_spaxel_xy_coordinates(spaxel_number)

    model_x, model_y = initial_model_transformation.transform(
        0., 0., spaxel_x, spaxel_y, optical_model_wavelength
    )

    width = np.ones(len(optical_model_wavelength))

    new_spaxel_data[spaxel_number] = {
        'wavelength': optical_model_wavelength,
        'ccd_x': model_x,
        'ccd_y': model_y,
        'width': width,
    }

optical_model.update(new_spaxel_data)


# Find sigma and x as a function of wavelength
print("Using continuum images to refine model.")

continuum_datas = []

for cont_image in cont_images:
    print("Next cont image")
    fit_wave = []

    fit_offsets = []
    fit_widths = []
    fit_amplitudes = []
    fit_amplitude_slopes = []
    fit_backgrounds = []

    fit_spaxel_xs = []
    fit_spaxel_ys = []
    fit_spaxel_is = []
    fit_spaxel_js = []

    fit_ccd_x = []
    fit_ccd_y = []

    optical_model = cont_image.optical_model

    for spaxel_number in optical_model.spaxel_numbers:
        print(spaxel_number)
        # if spaxel_number > 20:
            # continue

        for wavelength in np.arange(3200., 6000., 100.):
            try:
                patch_info = cont_image.fit_smooth_patch(
                    spaxel_number, wavelength
                )
            except OpticalModelFitException:
                # This fit sometimes fails on very low S/N data. Throw out that
                # patch when it happens.
                continue

            spaxel_i, spaxel_j = \
                optical_model.get_spaxel_ij_coordinates(spaxel_number)
            spaxel_x, spaxel_y = \
                optical_model.get_spaxel_xy_coordinates(spaxel_number)

            fit_wave.append(wavelength)
            fit_offsets.append(patch_info['offset'])
            fit_widths.append(patch_info['width'])
            fit_amplitudes.append(patch_info['amplitude'])
            fit_amplitude_slopes.append(patch_info['amplitude_slope'])
            fit_backgrounds.append(patch_info['background'])

            fit_ccd_x.append(patch_info['ccd_x'])
            fit_ccd_y.append(patch_info['ccd_y'])

            fit_spaxel_xs.append(spaxel_x)
            fit_spaxel_ys.append(spaxel_y)
            fit_spaxel_is.append(spaxel_i)
            fit_spaxel_js.append(spaxel_j)

    continuum_data = Table({
        'wavelength': fit_wave,
        'offset': fit_offsets,
        'width': fit_widths,
        'amplitude': fit_amplitudes,
        'amplitude_slope': fit_amplitude_slopes,
        'background': fit_backgrounds,

        'ccd_x': fit_ccd_x,
        'ccd_y': fit_ccd_y,

        'spaxel_x': fit_spaxel_xs,
        'spaxel_y': fit_spaxel_ys,
        'spaxel_i': fit_spaxel_is,
        'spaxel_j': fit_spaxel_js,
    })

    continuum_datas.append(continuum_data)
