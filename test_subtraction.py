import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from astropy.table import join
from collections import defaultdict
from scipy.interpolate import InterpolatedUnivariateSpline

from optical_model import OpticalModel, OpticalModelTransformation, \
    SpaxelModelFitter

plt.ion()


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

    slice_sigma = np.ones(len(slice_xccd))

    match_2 = np.where((spaxel_coordinates[:, 0] == spaxel))[0][0]
    spaxel_i, spaxel_j = spaxel_coordinates[match_2, 1:]

    spaxel_data[spaxel] = {
        'number': spaxel,
        'i_coord': spaxel_i,
        'j_coord': spaxel_j,
        'x_mla': slice_xmla,
        'y_mla': slice_ymla,
        'wave': slice_lambda,
        'x_ccd': slice_xccd,
        'y_ccd': slice_yccd,
        'sigma': slice_sigma
    }

optical_model = OpticalModel(spaxel_data)


# Load test data
arc_lines = [
    # 3252.52392578,
    3261.05493164,          # GOOD
    # 3341.4839,
    # 3403.65209961,
    3466.19995117,          # GOOD
    # 3467.6550293,
    3610.50805664,          # GOOD
    # 3612.87304688,
    # 3650.15795898,
    # 3654.84008789,
    4046.56494141,          # GOOD
    4077.8369,              # GOOD
    # 4158.58984375,     why is this bad? Looks great but comes out off by
    #                    around 0.2 in Y!
    # 4200.67382812,
    # 4347.50585938,
    4358.33496094,          # GOOD
    # 4678.1489,
    4799.91210938,          # GOOD
    5085.82177734,          # GOOD
    # 5460.75,
]

filenames = [
    # Ref
    ('./data/P08_231_109_010_07_B.fits', './data/P08_231_109_011_03_B.fits'),

    ('./data/P05_139_046_010_07_B.fits', './data/P05_139_046_011_03_B.fits'),
    ('./data/P06_151_149_010_07_B.fits', './data/P06_151_149_011_03_B.fits'),
    ('./data/P07_110_148_002_07_B.fits', './data/P07_110_148_003_03_B.fits'),
    ('./data/P09_184_174_010_07_B.fits', './data/P09_184_172_011_03_B.fits'),
    ('./data/P10_172_091_010_07_B.fits', './data/P10_172_091_011_03_B.fits'),
    ('./data/P11_267_082_002_07_B.fits', './data/P11_267_082_003_03_B.fits'),
    ('./data/P13_129_092_010_07_B.fits', './data/P13_129_092_011_03_B.fits'),

    # Don't use:
    # Bad fits file??
    # ('./data/P15_089_087_010_07_B.fits', './data/P15_089_087_011_03_B.fits'),
]

ref_idx = 0

arc_images = []
arc_datas = []

cont_images = []

for cont_filename, arc_filename in filenames:
    print(arc_filename)
    arc_file = fits.open(arc_filename)
    arc_image = arc_file[0].data
    arc_images.append(arc_image)

    arc_data = optical_model.identify_arc_lines(arc_image, arc_lines)
    arc_datas.append(arc_data)

    arc_file.close()

    cont_file = fits.open(cont_filename)
    cont_image = cont_file[0].data
    cont_images.append(cont_image)



transformations = [None]
ref_arc_data = arc_datas[ref_idx]

spaxel_xs = defaultdict(list)
spaxel_ys = defaultdict(list)
spaxel_waves = defaultdict(list)

ref_arc_data['trans_x'] = ref_arc_data['arc_x']
ref_arc_data['trans_y'] = ref_arc_data['arc_y']

for i, arc_data in enumerate(arc_datas):
    arc_data['sample_number'] = i

    if i == ref_idx:
        continue

    join_arc_data = join(ref_arc_data, arc_data, ['spaxel_i', 'spaxel_j',
                                                  'wave'])

    trans = OpticalModelTransformation()
    trans_x, trans_y = trans.fit(
        join_arc_data['arc_x_1'],
        join_arc_data['arc_y_1'],
        join_arc_data['arc_x_2'],
        join_arc_data['arc_y_2'],
        join_arc_data['spaxel_i'],
        join_arc_data['spaxel_j'],
        join_arc_data['wave'],
        order=3
    )

    from idrtools import math
    c = join_arc_data
    print(i,
          math.nmad(c['arc_x_1'] - c['arc_x_2']),
          math.nmad(c['arc_y_1'] - c['arc_y_2']),
          math.nmad(c['arc_x_1'] - trans_x),
          math.nmad(c['arc_y_1'] - trans_y))

    transformations.append(trans)

    # Transform lines that didn't find a match
    full_trans_x, full_trans_y = trans.transform(
        arc_data['arc_x'],
        arc_data['arc_y'],
        arc_data['spaxel_i'],
        arc_data['spaxel_j'],
        arc_data['wave']
    )

    for iter_trans_x, iter_trans_y, iter_arc_data in \
            zip(full_trans_x, full_trans_y, arc_data):
        spaxel_number = iter_arc_data['spaxel_number']
        spaxel_xs[spaxel_number].append(iter_trans_x)
        spaxel_ys[spaxel_number].append(iter_trans_y)
        spaxel_waves[spaxel_number].append(iter_arc_data['wave'])

    arc_data['trans_x'] = full_trans_x
    arc_data['trans_y'] = full_trans_y

# Triple transform test
# trans_x_1, trans_y_1 = transformations[0][7].transform(
    # c['arc_x_2'], c['arc_y_2'], c['spaxel_i'], c['spaxel_j'], c['wave']
# )
# trans_x_2, trans_y_2 = transformations[4][0].transform(
    # trans_x_1, trans_y_1, c['spaxel_i'], c['spaxel_j'], c['wave']
# )
# trans_x_3, trans_y_3 = transformations[7][4].transform(
    # trans_x_2, trans_y_2, c['spaxel_i'], c['spaxel_j'], c['wave']
# )


stack_data = np.hstack(arc_datas)


def get_data(spaxel_i, spaxel_j):
    cut = (
        (np.abs(stack_data['spaxel_i'] - spaxel_i) < 0.5) &
        (np.abs(stack_data['spaxel_j'] - spaxel_j) < 0.5)
    )

    return stack_data[cut]


initial_spaxel_l2xs = {}
initial_spaxel_l2ys = {}

arc_profiles = []
arc_dx = []
arc_i = []
arc_j = []

for spaxel_i in range(-7, 8):
    for spaxel_j in range(-7, 8):
        spaxel_data = get_data(spaxel_i, spaxel_j)
        if len(spaxel_data) == 0:
            continue

        wave = spaxel_data['wave']
        val = spaxel_data['trans_y']
        # orig = spaxel_data['model_y']

        # Note: need around order 5 or so to fully capture the model. However
        # we don't have enough arc lines to do that in a principled way.
        initial_model_x = SpaxelModelFitter()
        initial_model_y = SpaxelModelFitter()
        model_x_vals = initial_model_x.fit(wave, spaxel_data['trans_x'], 3)
        model_y_vals = initial_model_y.fit(wave, spaxel_data['trans_y'], 3)

        initial_spaxel_l2xs[spaxel_data[0]['spaxel_number']] = initial_model_x
        initial_spaxel_l2ys[spaxel_data[0]['spaxel_number']] = initial_model_y


        # Arc test
        test_wave = 3650.158
        arc_image = arc_images[0]
        arc_x = initial_model_x([test_wave])[0]
        arc_y = initial_model_y([test_wave])[0]

        x_min = int(arc_x) - 50
        x_max = int(arc_x) + 50
        y_min = int(arc_y) - 10
        y_max = int(arc_y) + 10
        x_win = np.arange(x_min, x_max)
        y_win = np.arange(y_min, y_max)

        dx = x_win - arc_x

        arc_profile = np.sum(arc_image[y_min:y_max, x_min:x_max], axis=0)

        if len(arc_profile) != len(dx):
            continue
        
        arc_profiles.append(arc_profile)
        arc_dx.append(dx)
        arc_i.append(spaxel_i)
        arc_j.append(spaxel_j)

arc_profiles = np.vstack(arc_profiles)
arc_dx = np.vstack(arc_dx)


# plt.imshow(cont_images[0], vmin=-10, vmax=5000)

idx = 3
# for idx in range(1, len(cont_images)):
test_im = cont_images[idx]
transformation = transformations[idx]

residual_im = test_im.copy()

for spaxel_number in initial_spaxel_l2xs.keys():
    # Fit to the continuum

    use_wave = np.arange(3500, 5200)

    x = initial_spaxel_l2xs[spaxel_number](use_wave)
    y = initial_spaxel_l2ys[spaxel_number](use_wave)

    print(spaxel_number, x[500])
    # plt.scatter(x[500], y[500])

    bla = stack_data[stack_data['spaxel_number'] == spaxel_number]

    trans_x, trans_y = transformation.transform(
        x,
        y,
        bla['spaxel_i'][0],
        bla['spaxel_j'][0],
        use_wave,
        reverse=True
    )

    order = np.argsort(trans_y)
    trans_x = trans_x[order]
    trans_y = trans_y[order]
    trans_wave = use_wave[order]
    

    y2x = InterpolatedUnivariateSpline(trans_y, trans_x)
    y2wave = InterpolatedUnivariateSpline(trans_y, trans_wave)


    # plt.plot(x, y, c='C0')
    # plt.plot(trans_x, trans_y, c='C1')

    min_y = int(np.min(trans_y))
    max_y = int(np.max(trans_y))

    from optical_model import fit_convgauss, OpticalModelFitException
    all_dx = []
    all_x = []
    all_dx_model = []
    all_sig = []
    all_wave = []
    all_y = np.arange(min_y, max_y)
    all_c = []

    all_res_x = []
    all_res_y = []
    all_res_c = []


    allxrange = np.arange(residual_im.shape[1])

    for iter_y in all_y:
        if iter_y % 100 == 0:
            print(iter_y, y2wave(iter_y))
        # print(iter_y, y2wave(iter_y))
        fit_x_start = y2x(iter_y)
        try:
            res = fit_convgauss(test_im, iter_y, fit_x_start, 5, 2, 1.5, True)

            all_dx.append(res['mu'] - fit_x_start)
            all_dx_model.append(
                res['mu'] -
                optical_model._spaxels[spaxel_number]._interp_wave_to_x(y2wave(iter_y))
            )
            all_sig.append(res['sigma'])
            all_wave.append(y2wave(iter_y))
            all_x.append(res['mu'])

            residual_im[res['model_y'], res['model_x']] -= res['model']

            all_res_x.extend(allxrange[res['model_x']] - res['mu'])
            all_res_y.extend(residual_im[res['model_y'], res['model_x']])
            all_res_c.extend(res['amp'] * np.ones(len(res['model_no_mean'])))

        except OpticalModelFitException:
            print("FAIL")
            all_dx.append(np.nan)
            all_dx_model.append(np.nan)
            all_sig.append(np.nan)
            all_wave.append(np.nan)
            all_x.append(np.nan)



    # plt.scatter(all_x, all_y)
    # plt.plot(all_wave, all_dx)
