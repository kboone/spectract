import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from astropy.table import join
from collections import defaultdict
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline

from optical_model import OpticalModel, OpticalModelTransformation, \
    SpaxelModelFitter, multigauss, convgauss, OpticalModelFitException

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
    # ('./data/P08_231_109_010_07_B.fits', './data/P08_231_109_011_03_B.fits'),
    ('./data/P13_129_092_010_07_B.fits', './data/P13_129_092_011_03_B.fits'),
    # ('./data/P05_139_046_010_07_B.fits', './data/P05_139_046_011_03_B.fits'),
    # ('./data/P11_267_082_002_07_B.fits', './data/P11_267_082_003_03_B.fits'),

    ('./data/P05_139_046_010_07_B.fits', './data/P05_139_046_011_03_B.fits'),
    ('./data/P06_151_149_010_07_B.fits', './data/P06_151_149_011_03_B.fits'),
    ('./data/P07_110_148_002_07_B.fits', './data/P07_110_148_003_03_B.fits'),
    ('./data/P09_184_174_010_07_B.fits', './data/P09_184_172_011_03_B.fits'),
    ('./data/P10_172_091_010_07_B.fits', './data/P10_172_091_011_03_B.fits'),
    ('./data/P11_267_082_002_07_B.fits', './data/P11_267_082_003_03_B.fits'),
    # ('./data/P13_129_092_010_07_B.fits', './data/P13_129_092_011_03_B.fits'),

    ('./data/P08_231_109_010_07_B.fits', './data/P08_231_109_011_03_B.fits'),

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
initial_spaxel_y2xs = {}

arc_profile_datas = []
arc_dxs = []
arc_is = []
arc_js = []
arc_xs = []
arc_ys = []
arc_spax_numbers = []

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
        initial_model_yx = SpaxelModelFitter()
        model_x_vals = initial_model_x.fit(wave, spaxel_data['trans_x'], 3)
        model_y_vals = initial_model_y.fit(wave, spaxel_data['trans_y'], 3)
        model_yx_vals = initial_model_yx.fit(spaxel_data['trans_y'],
                                             spaxel_data['trans_x'], 3)

        initial_spaxel_l2xs[spaxel_data[0]['spaxel_number']] = initial_model_x
        initial_spaxel_l2ys[spaxel_data[0]['spaxel_number']] = initial_model_y
        initial_spaxel_y2xs[spaxel_data[0]['spaxel_number']] = initial_model_yx

        # Arc test
        test_wave = 3650.158
        # test_wave = 4358.3349
        arc_image = arc_images[0]
        arc_x = initial_model_x([test_wave])[0]
        arc_y = initial_model_y([test_wave])[0]

        x_min = int(arc_x) - 30
        x_max = int(arc_x) + 30
        y_min = int(arc_y) - 10
        y_max = int(arc_y) + 10
        x_win = np.arange(x_min, x_max)
        y_win = np.arange(y_min, y_max)

        dx = x_win - arc_x

        arc_profile = np.sum(arc_image[y_min:y_max, x_min:x_max], axis=0)

        if len(arc_profile) != len(dx):
            continue

        arc_profile_datas.append(arc_profile)
        arc_dxs.append(dx)
        arc_is.append(spaxel_i)
        arc_js.append(spaxel_j)
        arc_xs.append(arc_x)
        arc_ys.append(arc_y)
        arc_spax_numbers.append(spaxel_data[0]['spaxel_number'])

arc_profile_datas = np.vstack(arc_profile_datas)
arc_dxs = np.vstack(arc_dxs)
arc_is = np.array(arc_is)
arc_js = np.array(arc_js)

spax_ids = np.array(list(initial_spaxel_l2xs.keys()))
spax_xs = np.array([initial_spaxel_l2xs[i]([4000])[0] for i in spax_ids])
xorder_spax_ids = spax_ids[np.argsort(spax_xs)]

plt.figure()

bla_x = []
bla_y = []
bla_c = []
bla_scale = []

from scipy.special import erf
def convgausswide(x, amp, mu, sigma, width):
    """Evaluate a 1d gaussian convolved with a pixel.

    - x is a 1d array of the x positions.
    - amp is the integral of the gaussian.
    - mu is the center position.
    - sigma is the standard deviation of the Gaussian.
    """
    return (
        amp*0.5*(
            erf((x + width/2 - mu) / (np.sqrt(2) * sigma)) -
            erf((x - width/2 - mu) / (np.sqrt(2) * sigma))
        )
    )


for i in range(len(arc_profile_datas)):
    arc_profile_data = arc_profile_datas[i]
    arc_dx = arc_dxs[i]
    arc_i = arc_is[i]
    arc_j = arc_js[i]
    arc_x = arc_xs[i]
    arc_y = arc_ys[i]
    spax_number = arc_spax_numbers[i]

    xorder_spax_id = np.where(xorder_spax_ids == spax_number)[0]

    try:
        spax_id_l = xorder_spax_ids[xorder_spax_id - 1][0]
        spax_loc_l = initial_spaxel_y2xs[spax_id_l]([arc_y])[0] - arc_x

        spax_id_r = xorder_spax_ids[xorder_spax_id + 1][0]
        spax_loc_r = initial_spaxel_y2xs[spax_id_r]([arc_y])[0] - arc_x

        spax_id_r_2 = xorder_spax_ids[xorder_spax_id + 2][0]
        spax_loc_r_2 = initial_spaxel_y2xs[spax_id_r_2]([arc_y])[0] - arc_x
    except (KeyError, IndexError):
        continue

    start_mu = 0
    search_mu = 0.5

    start_sigma = 1.5
    search_sigma = 1.0

    fit_min_mu = start_mu - search_mu
    fit_max_mu = start_mu + search_mu
    fit_min_sigma = start_sigma - search_sigma
    fit_max_sigma = start_sigma + search_sigma

    amp_scale = 1000.

    def lorentzcorr(x):
        relamp = 0.06
        gamma = 6.
        return relamp / (np.pi * gamma * (1 + (x / gamma)**2))

    def psf(x, mu, sigma, amp_gauss, geopsf_params=None, lorentzian=True,
            pixwidth=1.):
        # From Yannick's optical model
        if geopsf_params is None:
            geopsf_params = [
                # (0.458173, 0, 0.338631),
                # (0.509488, -0.522670, 0.239250),
                # (0.509488, 0.522670, 0.239250),
                # (0.0834, 0, 0.851),
                # (0.686, -0.447, 0.01),
                # (0.686, 0.447, 0.01),
                (1., 0., 0.)
            ]

        model = np.zeros(len(x))

        for geo_amp, geo_mu, geo_sigma in geopsf_params:
            conv_amp, conv_mu, conv_sigma = multigauss(
                geo_amp, geo_mu, geo_sigma, amp_gauss, mu, sigma
            )
            # model += convgauss(x, conv_amp, conv_mu, conv_sigma)
            model += convgausswide(x, conv_amp, conv_mu, conv_sigma, pixwidth)

        # Lorentzian correction
        if lorentzian:
            model += amp_gauss * lorentzcorr(x - mu)
        # model = (
            # convgauss(x, amp_gauss, mu, sigma) +
            # amp_gauss * lorentzcorr(x - mu)
        # )
        model /= pixwidth

        return model

    def model(amp, mu, sigma, mean, mu_l, amp_l, amp_r, amp_r_2, slope):#, bkgamp, sigmabkg):
        # sigmabkg = 6.
        # bkgamp = 0.06
        return (
            psf(arc_dx, mu, sigma, amp*amp_scale) +
            psf(arc_dx, mu_l, sigma, amp_l*amp_scale) +
            psf(arc_dx, spax_loc_r, sigma, amp_r*amp_scale) +
            psf(arc_dx, spax_loc_r_2, sigma, amp_r_2*amp_scale) +
            # convgauss(arc_dx, amp * amp_scale, mu, sigma) +
            # convgauss(arc_dx, amp_l * amp_scale, spax_loc_l, sigma) +
            # convgauss(arc_dx, amp_r * amp_scale, spax_loc_r, sigma) +
            # convgauss(arc_dx, amp_r_2 * amp_scale, spax_loc_r_2, sigma) +

            # convgauss(arc_dx, bkgamp * amp_scale, 0., sigmabkg) +
            # bkgamp * amp_scale * lorentz(arc_dx, sigmabkg) +
            mean +
            slope * arc_dx
        )

    def fit_func(x):
        return np.sum((arc_profile_data - model(*x))**2)

    start_amp = np.sum(arc_profile_data)

    other_start_amp = 1000. / amp_scale

    start_params = np.array([start_amp / amp_scale, start_mu, start_sigma, 0.,
                             spax_loc_l,
                             other_start_amp,
                             other_start_amp,
                             other_start_amp,
                             0.
                             ])

                             # other_start_amp,
                             # 10])
    bounds = [
        (0.1*start_amp / amp_scale, 10*start_amp / amp_scale),
        (fit_min_mu, fit_max_mu),
        (fit_min_sigma, fit_max_sigma),
        (None, None),
        (spax_loc_l - 1, spax_loc_l + 1),
        (0 / amp_scale, 5000000 / amp_scale),
        # (-12, -6),
        (0 / amp_scale, 5000000 / amp_scale),
        # (6, 13),
        (0 / amp_scale, 5000000 / amp_scale),
        (None, None)
        # (0 / amp_scale, 5000000 / amp_scale),
        # (0.00, 0.2),
        # (3, 50),
        # (11, 25),
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

    # print(fit_params)
    # fit_amp, fit_mu, fit_sigma, fit_mean, fit_amp_l, fit_mu_l, fit_amp_r, \
        # fit_mu_r, fit_amp_r_2, fit_mu_r_2 = fit_params
    # fit_amp, fit_mu, fit_sigma, fit_mean, fit_mu_l, fit_amp_l, fit_amp_r, fit_amp_r_2, \
        # sigmaamp, sigmabkg = fit_params
    fit_amp, fit_mu, fit_sigma, fit_mean, fit_mu_l, fit_amp_l, fit_amp_r, \
        fit_amp_r_2, slope = fit_params
    # print(fit_amp, fit_amp_l, fit_mu_l, slope)#, sigmaamp, sigmabkg)
    print(fit_mu, fit_sigma)

    # if fit_amp_l > fit_amp / 10:
        # continue

    # print("Using!")
        


    use_params = fit_params.copy()
    # use_params[3] = 0.
    # use_params[-2] = 0.

    result_model = model(*use_params)

    bkgonly_params = fit_params.copy()
    bkgonly_params[:3] = 0.
    # bkgonly_params[4:] = 0.
    # use_params[3] = 0.
    # use_params[-2] = 0.
    bkgonly_model = model(*bkgonly_params)

    # print(spax_loc_l - fit_mu_l, spax_loc_r - fit_mu_r, spax_loc_r_2 -
          # fit_mu_r_2)

    color = 'C%d' % (i % 9)
    plot_y = (arc_profile_data - result_model) / (fit_amp * amp_scale)
    # plt.scatter(arc_dx, plot_y, c=color, alpha=0.2)
    # plt.plot(arc_dx, result_model, c=color)

    bla_x.extend(arc_dx - fit_mu)
    bla_y.extend(plot_y)
    bla_c.extend(np.ones(len(plot_y)) * fit_sigma)

    bla_scale.extend((arc_profile_data - bkgonly_model) / (fit_amp * amp_scale))

    no_mean_fit_params = fit_params.copy()
    no_mean_fit_params[-1] = 0.
    result_model_no_mean = model(*no_mean_fit_params)

    do_print = False

    if fit_amp < 0.5*start_amp/amp_scale or fit_amp > 1.5*start_amp/amp_scale:
        print("WARNING: Fit amplitude out of normal bounds:")
        do_print = True

bla_x = np.array(bla_x)
bla_y = np.array(bla_y)
bla_c = np.array(bla_c)
bla_scale = np.array(bla_scale)

plt.scatter(bla_x, bla_y, c=bla_c, alpha=0.2)

bin_width = 0.1
bins = np.arange(-29, 29, bin_width)
res = []
for i in bins:
    cut = (bla_x > i - bin_width / 2.) & (bla_x < i + bin_width / 2.)
    res.append(np.median(bla_y[cut]))

res = np.array(res)

# Find the median residual profile
# med_bins = 

# Find the correction spline
nmad = math.nmad(res)
correction_spline = UnivariateSpline(bins, res, w=np.ones(len(bins)) / nmad)

correction = correction_spline(bins)

# Adjust the correction spline to go to 0 gracefully. Without this it gets to
# ~1e-4 so this doesn't do too much.
zero_radius = 10.
rolloff = 3.
dist = np.abs(bins) - zero_radius
dist[dist < 0] = 0.
corr_func = np.exp(-(dist / rolloff))
adj_correction = correction * corr_func

# FINAL RESULTS
lorentz_correction = lorentzcorr(bins)
total_correction = lorentz_correction + adj_correction

total_spline = UnivariateSpline(bins, total_correction, w=np.ones(len(bins)) /
                                nmad, ext=1)

bla_residuals = bla_y - total_spline(bla_x)


bla_gaussonly = bla_scale - lorentzcorr(bla_x)

cut = (bla_x > -5) & (bla_x < 5)
cut_bla_x = bla_x[cut]
cut_bla_y = bla_y[cut]
cut_bla_c = bla_c[cut]
cut_bla_gaussonly = bla_gaussonly[cut]

def geopsf_model(x):
    geopsf_params = [
        # (x[0], 0, x[1]),
        # (x[2], -x[3], x[4]),
        # (x[5], x[6], x[7]),
        (x[0], x[1], x[2]),
        (x[3], x[4], x[5]),
    ]

    newpsf = psf(cut_bla_x, 0., cut_bla_c, 1., geopsf_params=geopsf_params,
                 lorentzian=False)

    return newpsf


def geopsf_tomin(x):
    newpsf = geopsf_model(x)

    return np.sum((newpsf - cut_bla_gaussonly)**2)


geopsf_start_params = [
    # 0.458173, 0.338631, 0.509488, 0.522670, 0.239250, 0.509488, 0.522670,
    # 0.239250
    0.9, 0.005, 0.7, 0.1, -0.5, 1.5,
]

bounds = [ (0.01, 5.), (-5, 5.), (0.01, 5.), (0.01, 5.), (-5, 5.), (0.01,
                                                                        5.),]
    # (0.01, 5.), (0.01, 5.), ]

geopsf_res = minimize( geopsf_tomin, geopsf_start_params, method='L-BFGS-B',
    bounds=bounds)


from scipy.stats import skewnorm


def convskew(x, amp, mu, sigma, skew):
    scale_x = (x - mu) / sigma
    return amp * (skewnorm.cdf(scale_x+0.5, skew) - skewnorm.cdf(scale_x-0.5,
                                                                 skew))


def widepsf_model(x):
    model = convskew(cut_bla_x, x[0], x[1], cut_bla_c*x[2], x[3])

    return model


def widepsf_tomin(x):
    print(x)
    newpsf = widepsf_model(x)

    return np.sum((newpsf - cut_bla_gaussonly)**2)


widepsf_start_params = [
    1., 0., 1., 0.
]

bounds = [
    (0.01, 5.),
    (-5., 5.),
    (0.1, 10.),
    (-5., 5.),
]

widepsf_res = minimize(
    widepsf_tomin,
    widepsf_start_params,
    method='L-BFGS-B',
    bounds=bounds
)
