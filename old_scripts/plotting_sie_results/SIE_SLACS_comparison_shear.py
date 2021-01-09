from autofit import conf
import autolens as al
import autoastro as astro

import os

path = '{}/../../../../autolens_workspace/'.format(os.path.dirname(os.path.realpath(__file__)))

conf.instance = conf.Config(config_path=path+'config', output_path=path+'output')

import pandas as pd
import numpy as np
from astropy import cosmology


import matplotlib.pyplot as plt
from pathlib import Path

M_o = 1.989e30
cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)

## setting up paths to shear and no shear results (sersic source)
data_path = '{}/../../../../../output/slacs_fresh/'.format(os.path.dirname(os.path.realpath(__file__)))
slacs_path = '{}/../../data/slacs_data_table.xlsx'.format(os.path.dirname(os.path.realpath(__file__)))
wavelength = 'F814W'
pipeline = '/pipeline_inv_hyper__lens_bulge_disk_sie__source_inversion/' \
           'pipeline_tag__with_shear__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/' \
           'phase_4__lens_bulge_disk_sie__source_inversion/phase_tag__sub_2__pos_1.00/model.results'
fig_path = '/Users/dgmt59/Documents/Plots/slacs_fresh/SIE_SLACS_comparison/shear_' + wavelength

lens_name = np.array(['slacs0216-0813',
                      'slacs0252+0039',
                      'slacs0737+3216',
                      'slacs0912+0029',
                      'slacs0959+4410',
                      'slacs1205+4910',
                      'slacs1250+0523',
                      'slacs1402+6321',
                      'slacs1420+6019',
                      'slacs1430+4105',
                      'slacs1627+0053',
                      'slacs1630+4520',
                      'slacs2238-0754',
                      'slacs2300+0022',
                      'slacs2303+1422'])

## creating variables for loading results
list_ = []
lens = []
SLACS_image = []

## loading table of slacs parameters (deleting two discarded lenses)
slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name
slacs = slacs.drop(['slacs0959+5100',  'slacs0959+4416'], axis=0)

## loading results into pandas data frames
## deleting any lenses from slacs table that don't have autolens results files
for i in range(len(lens_name)):
    full_data_path = Path(data_path + wavelength + '/' + lens_name[i] + pipeline)
    if full_data_path.is_file():
        data = pd.read_csv(full_data_path, sep='\s+', header=None, skiprows=21, nrows=9,).set_index(0)
        del data.index.name
        data[2] = data[2].str.strip('(,').astype(float)
        data[3] = data[3].str.strip(')').astype(float)
        data.columns=['param', '-error', '+error']
        list_.append(data)
        lens.append(lens_name[i])
        results = pd.concat(list_, keys=lens)
        model_image_path = data_path + wavelength + '/' + lens_name[i] + \
                           '/pipeline_inv_hyper__lens_bulge_disk_sie__source_inversion/' \
                           'pipeline_tag__with_shear__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/' \
                           'phase_4__lens_bulge_disk_sie__source_inversion/phase_tag__sub_2__pos_1.00/' \
                           'image/lens_fit/fits/fit_model_image_of_plane_1.fits'
        model_image = al.Array.from_fits(file_path=model_image_path, hdu=0,pixel_scales=0.03)
        SLACS_image.append(model_image)
    else:
        slacs = slacs.drop([lens_name[i]], axis=0)

## creating variables for einstin mass results
R_Ein = []
M_Ein = []



## creating galaxy in autolens from autolens mass profile parameters and redshift of lens galaxy
## hi and low correspond to errors on measurement
for i in range(len(lens)):
    ## creating galaxies in autolens from autolens model results
    lens_galaxy =al.Galaxy(
        mass=al.mp.EllipticalIsothermal(
            centre=(results.loc[lens[i]]['param']['centre_0'], results.loc[lens[i]]['param']['centre_1']),
            axis_ratio=results.loc[lens[i]]['param']['axis_ratio'], phi=results.loc[lens[i]]['param'][1],
            einstein_radius=results.loc[lens[i]]['param']['einstein_radius']), redshift=slacs['z_lens'][i]
    )

    einstein_radius = lens_galaxy.einstein_radius_in_units(unit_length='arcsec')
    einstein_mass = lens_galaxy.einstein_mass_in_units(redshift_object=lens_galaxy.redshift,
                                                       redshift_source=slacs['z_source'][i], cosmology=cosmo,
                                                       unit_mass='solMass'
                                                       )


    R_Ein.append(einstein_radius)
    M_Ein.append(einstein_mass)

## calcualting fractional change for plots
R_Ein_frac = (R_Ein - slacs['b_SIE'])/slacs['b_SIE']
M_Ein_frac = (M_Ein - 10**slacs['log[Me/Mo]'])/10**slacs['log[Me/Mo]']

fig1, ax = plt.subplots(figsize=(10,5))
for i in range(len(lens)):
    ax.scatter(results.loc[lens[i]]['param']['axis_ratio'], R_Ein_frac[i], label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title= 'Lens Name')
plt.axhline(y=0, color='k',)
plt.axhline(y=0.02, color='k', linestyle='--')
plt.axhline(y=-0.02, color='k', linestyle='--')
plt.xlabel('Axis ratio', size=20)
plt.ylabel(r'$\frac{R_{Ein_{autolens}}-R_{Ein_{SLACS}}}{R_{Ein_{SLACS}}}$', size=21)
plt.savefig(fig_path + '_R_Ein_as_function_of_q', bbox_inches='tight', dpi=300)

fig2, ax = plt.subplots(figsize=(10,5))
for i in range(len(lens)):
    ax.scatter(results.loc[lens[i]]['param']['axis_ratio'], M_Ein_frac[i], label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title= 'Lens Name')
plt.axhline(y=0, color='k',)
plt.axhline(y=0.02, color='k', linestyle='--')
plt.axhline(y=-0.02, color='k', linestyle='--')
plt.xlabel('Axis ratio', size=20)
plt.ylabel(r'$\frac{M_{Ein_{autolens}}-M_{Ein_{SLACS}}}{M_{Ein_{SLACS}}}$', size=21)
plt.savefig(fig_path +'_M_Ein_as_function_of_q', bbox_inches='tight', dpi=300)

error_low_q = []
error_hi_q = []

## calculating errors on q slope
for i in range(len(lens)):
    lower_error_q = results.loc[lens[i]]['param']['axis_ratio']-results.loc[lens[i]]['-error']['axis_ratio']
    upper_error_q = results.loc[lens[i]]['+error']['axis_ratio']-results.loc[lens[i]]['param']['axis_ratio']
    error_low_q.append(lower_error_q)
    error_hi_q.append(upper_error_q)

y_err_q = np.array([error_low_q, error_hi_q])

fig3, ax = plt.subplots(figsize=(8,8))
for i in range(len(lens)):
    ax.scatter(slacs['q_SIE'][i], results.loc[lens[i]]['param']['axis_ratio'],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
    ax.errorbar(slacs['q_SIE'][i], results.loc[lens[i]]['param']['axis_ratio'], yerr=np.array([y_err_q[:,i]]).T,
               color=slacs['colour'][i], elinewidth=1, fmt='none', capsize=3, label=None)
plt.xlabel(r'$q_{SLACS}$', size=14)
plt.ylabel(r'$q_{AutoLens}$', size=14)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title= 'Lens Name')
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
     ]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.savefig(fig_path + 'axis_ratio', bbox_inches='tight', dpi=300)

plt.show()
