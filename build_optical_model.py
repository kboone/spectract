import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from astropy.table import join
from collections import defaultdict

from optical_model import OpticalModel, OpticalModelTransformation

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
    3261.05493164,
    # 3341.4839,
    # 3403.65209961,
    3466.19995117,
    # 3467.6550293,
    3610.50805664,
    # 3612.87304688,
    # 3650.15795898,
    # 3654.84008789,
    4046.56494141,
    4077.8369,
    4158.58984375,
    # 4200.67382812,
    # 4347.50585938,
    4358.33496094,
    # 4678.1489,
    4799.91210938,
    5085.82177734,
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

for cont_filename, arc_filename in filenames:
    print(arc_filename)
    arc_file = fits.open(arc_filename)
    arc_image = arc_file[0].data
    arc_images.append(arc_image)

    arc_data = optical_model.identify_arc_lines(arc_image, arc_lines)
    arc_datas.append(arc_data)


transformations = []
ref_arc_data = arc_datas[ref_idx]

spaxel_xs = defaultdict(list)
spaxel_ys = defaultdict(list)
spaxel_waves = defaultdict(list)

ref_arc_data['trans_x'] = ref_arc_data['arc_x']
ref_arc_data['trans_y'] = ref_arc_data['arc_y']

for i, other_arc_data in enumerate(arc_datas):
    if i == ref_idx:
        continue

    join_arc_data = join(ref_arc_data, other_arc_data, ['spaxel_i', 'spaxel_j',
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
        2
    )

    # print(i, j,
          # math.nmad(c['arc_x_1'] - c['arc_x_2']),
          # math.nmad(c['arc_y_1'] - c['arc_y_2']),
          # math.nmad(c['arc_x_1'] - trans_x),
          # math.nmad(c['arc_y_1'] - trans_y))

    transformations.append(trans)

    # Transform lines that didn't find a match
    full_trans_x, full_trans_y = trans.transform(
        other_arc_data['arc_x'],
        other_arc_data['arc_y'],
        other_arc_data['spaxel_i'],
        other_arc_data['spaxel_j'],
        other_arc_data['wave']
    )

    for iter_trans_x, iter_trans_y, iter_arc_data in \
            zip(full_trans_x, full_trans_y, other_arc_data):
        spaxel_number = iter_arc_data['spaxel_number']
        spaxel_xs[spaxel_number].append(iter_trans_x)
        spaxel_ys[spaxel_number].append(iter_trans_y)
        spaxel_waves[spaxel_number].append(iter_arc_data['wave'])

    other_arc_data['trans_x'] = full_trans_x
    other_arc_data['trans_y'] = full_trans_y

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
