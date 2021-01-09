from autofit import conf
import autolens as al
import autoastro as astro

import os

path = '{}/../../../../autolens_workspace/'.format(os.path.dirname(os.path.realpath(__file__)))

conf.instance = conf.Config(config_path=path+'config', output_path=path+'output')

import pandas as pd
import numpy as np

import autolens as al
from astropy import cosmology

import matplotlib.pyplot as plt
from pathlib import Path

M_o = 1.989e30
cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)

lens_name = np.array(['slacs0216-0813',
                      'slacs0252+0039',
                      'slacs0737+3216',
                      'slacs0912+0029',
                      'slacs0959+0410',
                      'slacs1205+4910',
                      'slacs1250+0523',
                      'slacs1402+6321',
                      'slacs1420+6019',
                      'slacs1430+4105',
                      'slacs1627-0053',
                      'slacs1630+4520',
                      'slacs2238-0754',
                      'slacs2300+0022',
                      'slacs2303+1422'])


data_path = '{}/../../../../../output/slacs_shu_shu/'.format(os.path.dirname(os.path.realpath(__file__)))
slacs_path = '{}/../../data/slacs_data_table.xlsx'.format(os.path.dirname(os.path.realpath(__file__)))
wavelength = 'F814W'
pipeline = '/pipeline_power_law_hyper__lens_bulge_disk_power_law__source_inversion__const'
model = '/pipeline_tag__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/' \
        'phase_1__lens_bulge_disk_power_law__source_inversion/phase_tag__sub_2__pos_0.50/' \
        'model.results'
fig_path = '/Users/dgmt59/Documents/Plots/slacs_shu/PL_no_shear/'

list_ = []
lens=[]
error_low = []
error_hi = []
slope_frac = []
slope_diff = []
q_diff = []
PA_diff = []
r_frac = []
slope = []

slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name
slacs = slacs.drop(['slacs0959+5100',  'slacs0959+4416'], axis=0)

## loading in date from model.results file
for i in range(len(lens_name)):
    full_data_path = Path(data_path + wavelength + '/' + lens_name[i] + pipeline + model)
    if full_data_path.is_file():
        data = pd.read_csv(full_data_path, sep='\s+', header=None, skiprows=19, nrows=7,).set_index(0)
        del data.index.name
        data[2] = data[2].str.strip('(,').astype(float)
        data[3] = data[3].str.strip(')').astype(float)
        data.columns=['param', '-error', '+error']
        list_.append(data)
        lens.append(lens_name[i])
        results = pd.concat(list_, keys=lens)
    else:
        slacs = slacs.drop([lens_name[i]], axis=0)


## calculating errors on power law slope
for i in range(len(lens)):
    lower_error = results.loc[lens[i]]['param']['slope']-results.loc[lens[i]]['-error']['slope']
    upper_error = results.loc[lens[i]]['+error']['slope']-results.loc[lens[i]]['param']['slope']
    error_low.append(lower_error)
    error_hi.append(upper_error)


for i in range(len(lens)):
    lens_galaxy = al.Galaxy(
        mass=al.mp.EllipticalPowerLaw(
            centre=(results.loc[lens[i]]['param']['centre_0'], results.loc[lens[i]]['param']['centre_1']),
            axis_ratio=results.loc[lens[i]]['param']['axis_ratio'], phi=results.loc[lens[i]]['param']['phi'],
            einstein_radius=results.loc[lens[i]]['param']['einstein_radius'], slope=results.loc[lens[i]]['param']['slope']),
            redshift=slacs['z_lens'][i]
    )

    einstein_radius = lens_galaxy.einstein_radius_in_units(unit_length='arcsec')

    gamma = results.loc[lens[i]]['param']['slope']

    gamma_diff_frac = (results.loc[lens[i]]['param']['slope'] - slacs['gamma'][i])/slacs['gamma'][i]
    gamma_diff = results.loc[lens[i]]['param']['slope'] - slacs['gamma'][i]
    axis_ratio_diff = results.loc[lens[i]]['param']['axis_ratio'] - slacs['q_SIE'][i]
    radius_diff_frac =(einstein_radius - slacs['b_SIE'][i])/slacs['b_SIE'][i]
    phi_diff = results.loc[lens[i]]['param']['phi'] - slacs['PA'][i]

    slope_frac.append(gamma_diff_frac)
    slope_diff.append(gamma_diff)
    q_diff.append(axis_ratio_diff)
    PA_diff.append(phi_diff)
    r_frac.append(radius_diff_frac)
    slope.append(gamma)

y_err = np.array([error_low, error_hi])

R_ratio = slacs['b_kpc']/slacs['R_eff']

print(np.median(slope))
print(np.mean(slope))



