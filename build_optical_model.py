import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from astropy.table import join

from optical_model import OpticalModel, OpticalModelTransformation

from idrtools import math

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

arc_images = []
arc_datas = []

for cont_filename, arc_filename in filenames:
    print(arc_filename)
    arc_file = fits.open(arc_filename)
    arc_image = arc_file[0].data
    arc_images.append(arc_image)

    arc_data = optical_model.identify_arc_lines(arc_image, arc_lines)
    arc_datas.append(arc_data)


for i, a in enumerate(arc_datas):
    for j, b in enumerate(arc_datas):
        c = join(a, b, ['spaxel_i', 'spaxel_j', 'wave'])

        trans = OpticalModelTransformation()
        trans_x, trans_y = trans.fit(
            c['arc_x_1'], c['arc_y_1'], c['arc_x_2'], c['arc_y_2'],
            c['spaxel_i'], c['spaxel_j'], c['wave']
        )

        print(i, j, math.nmad(c['arc_x_1'] - trans_x),
              math.nmad(c['arc_y_1'] - trans_y))
