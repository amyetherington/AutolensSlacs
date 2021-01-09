from autofit import conf
import autolens as al
import autoastro as astro

import os

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))

conf.instance = conf.Config(config_path=path+'config', output_path=path+'output')

import pandas as pd
import numpy as np
from astropy import cosmology


import matplotlib.pyplot as plt
from pathlib import Path

M_o = 1.989e30
cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)

## setting up paths to shear and no shear results (sersic source)
data_path = '{}/../../../../../output/slacs_final_shear_messed_up/F814W/'.format(os.path.dirname(os.path.realpath(__file__)))
data_path_555 = '{}/../../../../../output/slacs_final_shear_messed_up/F555W/'.format(os.path.dirname(os.path.realpath(__file__)))
slacs_path = '{}/../../data/slacs_data_table.xlsx'.format(os.path.dirname(os.path.realpath(__file__)))
pipeline = '/pipeline_inv_hyper__lens_bulge_disk_sie__source_inversion/'
no_shear = 'pipeline_tag__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/' \
           'phase_4__lens_bulge_disk_sie__source_inversion/phase_tag__sub_2__pos_1.00/model.results'

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
SLACS_image_555 = []
list__555 = []
lens_555 = []

## loading table of slacs parameters (deleting two discarded lenses)
slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name
slacs = slacs.drop(['slacs0959+5100',  'slacs0959+4416'], axis=0)


## loading F814W and F555W results into two sepearate pandas data frames
## deleting any lenses from slacs table that don't have autolens results files
for i in range(len(lens_name)):
    full_data_path = Path(data_path + lens_name[i] + pipeline + no_shear)
    full_data_path_555 = Path(data_path_555 + lens_name[i] + pipeline + no_shear)
    if full_data_path.is_file() and full_data_path_555.is_file():
        data = pd.read_csv(full_data_path, sep='\s+', header=None, skiprows=18, nrows=6,).set_index(0)
        del data.index.name
        data[2] = data[2].str.strip('(,').astype(float)
        data[3] = data[3].str.strip(')').astype(float)
        data.columns=['param', '-error', '+error']
        list_.append(data)
        lens.append(lens_name[i])
        results = pd.concat(list_, keys=lens)
        model_image_path = data_path + lens_name[i] + '/pipeline_inv_hyper__lens_bulge_disk_sie__source_inversion/' \
                                                      'pipeline_tag__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/' \
                                                      'phase_4__lens_bulge_disk_sie__source_inversion/phase_tag__sub_2__pos_1.00/' \
                                                      'image/lens_fit/fits/fit_model_image_of_plane_1.fits'
        model_image = al.Array.from_fits(file_path=model_image_path, hdu=0,
                                                                                       pixel_scales=0.03)
        SLACS_image.append(model_image)


        data_555 = pd.read_csv(full_data_path_555, sep='\s+', header=None, skiprows=18, nrows=6, ).set_index(0)
        del data_555.index.name
        data_555[2] = data_555[2].str.strip('(,').astype(float)
        data_555[3] = data_555[3].str.strip(')').astype(float)
        data_555.columns = ['param', '-error', '+error']
        list__555.append(data_555)
        lens_555.append(lens_name[i])
        results_555 = pd.concat(list__555, keys=lens_555)
        model_image_path_555 = data_path_555 + lens_name[i] + '/pipeline_inv_hyper__lens_bulge_disk_sie__source_inversion/' \
                                                      'pipeline_tag__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/' \
                                                      'phase_4__lens_bulge_disk_sie__source_inversion/phase_tag__sub_2__pos_1.00/' \
                                                      'image/lens_fit/fits/fit_model_image_of_plane_1.fits'
        model_image_555 = al.Array.from_fits(file_path=model_image_path_555, hdu=0,
                                         pixel_scales=0.03)
        SLACS_image_555.append(model_image_555)
    else:
        slacs = slacs.drop([lens_name[i]], axis=0)

## creating variables for einstin mass results
R_Ein = []
R_Ein_555 = []
M_Ein = []
M_Ein_555 = []
axis_ratio_error_low = []
axis_ratio_error_hi = []
phi_error_low = []
phi_error_hi = []
axis_ratio_error_low_555 = []
axis_ratio_error_hi_555 = []
phi_error_low_555 = []
phi_error_hi_555 = []

R_Ein_rescaled = []

## creating galaxy in autolens from autolens mass profile parameters and redshift of lens galaxy
## hi and low correspond to errors on measurement
for i in range(len(lens)):
    ## creating galaxies in autolens from autolens model results
    lens_galaxy = al.Galaxy(
        mass=al.mp.EllipticalIsothermal(
            centre=(results.loc[lens[i]]['param']['centre_0'], results.loc[lens[i]]['param']['centre_1']),
            axis_ratio=results.loc[lens[i]]['param']['axis_ratio'], phi=results.loc[lens[i]]['param']['phi'],
            einstein_radius=results.loc[lens[i]]['param']['einstein_radius']), redshift = slacs['z_lens'][i]
    )
    lens_galaxy_555 = al.Galaxy(
        mass=al.mp.EllipticalIsothermal(
            centre=(results_555.loc[lens[i]]['param']['centre_0'], results_555.loc[lens[i]]['param']['centre_1']),
            axis_ratio=results_555.loc[lens[i]]['param']['axis_ratio'], phi=results_555.loc[lens[i]]['param']['phi'],
            einstein_radius=results_555.loc[lens[i]]['param']['einstein_radius']), redshift = slacs['z_lens'][i]
    )

    ## creating grids for finding crititcal curves
    grid = al.Grid.uniform(
        shape_2d=SLACS_image[i].shape_2d, pixel_scales=0.03, sub_size=2)
    grid_555 = al.Grid.uniform(
        shape_2d=SLACS_image_555[i].shape_2d, pixel_scales=0.03, sub_size=2)

    critical_curves = lens_galaxy.critical_curves_from_grid(grid=grid)
    critical_curves_555 = lens_galaxy_555.critical_curves_from_grid(grid=grid_555)

    critical_curve_tan, critical_curve_rad = critical_curves[0], critical_curves[1]
    critical_curve_tan_555, critical_curve_rad_555 = critical_curves_555[0], critical_curves_555[1]

    ## finding area within critical curve to calculate einstein radius
    x = critical_curve_tan[:, 0]
    y = critical_curve_tan[:, 1]
    area = np.abs(0.5 * np.sum(y[:-1] * np.diff(x) - x[:-1] * np.diff(y)))
    einstein_radius = np.sqrt(area/np.pi)

    x_555 = critical_curve_tan_555[:, 0]
    y_555 = critical_curve_tan_555[:, 1]
    area_555 = np.abs(0.5 * np.sum(y_555[:-1] * np.diff(x_555) - x_555[:-1] * np.diff(y_555)))
    einstein_radius_555 = np.sqrt(area_555 / np.pi)

    ## calculating critical surface mass density to find einstein mass
    Sigma_crit = astro.util.cosmo.critical_surface_density_between_redshifts_from_redshifts_and_cosmology(
        redshift_0=slacs['z_lens'][i], redshift_1=slacs['z_source'][i], cosmology=cosmo,
        unit_length='arcsec', unit_mass='kg')

    einstein_mass = Sigma_crit * np.pi * (einstein_radius ** 2)/M_o
    einstein_mass_555 = Sigma_crit * np.pi * (einstein_radius_555 ** 2)/M_o

    ##errors on axis ratio
    lower_error_axis_ratio = results.loc[lens[i]]['param']['axis_ratio'] - results.loc[lens[i]]['-error']['axis_ratio']
    upper_error_axis_ratio = results.loc[lens[i]]['+error']['axis_ratio'] - results.loc[lens[i]]['param']['axis_ratio']
    axis_ratio_error_low.append(lower_error_axis_ratio)
    axis_ratio_error_hi.append(upper_error_axis_ratio)
    ##errors on position angle
    lower_error_phi = results.loc[lens[i]]['param']['phi'] - results.loc[lens[i]]['-error']['phi']
    upper_error_phi = results.loc[lens[i]]['+error']['phi'] - results.loc[lens[i]]['param']['phi']
    phi_error_low.append(lower_error_phi)
    phi_error_hi.append(upper_error_phi)

    ##errors on axis ratio 555
    lower_error_axis_ratio_555 = results_555.loc[lens[i]]['param']['axis_ratio'] - results_555.loc[lens[i]]['-error']['axis_ratio']
    upper_error_axis_ratio_555 = results_555.loc[lens[i]]['+error']['axis_ratio'] - results_555.loc[lens[i]]['param']['axis_ratio']
    axis_ratio_error_low_555.append(lower_error_axis_ratio_555)
    axis_ratio_error_hi_555.append(upper_error_axis_ratio_555)
    ##errors on position angle 555
    lower_error_phi_555= results_555.loc[lens[i]]['param']['phi'] - results_555.loc[lens[i]]['-error']['phi']
    upper_error_phi_555 = results_555.loc[lens[i]]['+error']['phi'] - results_555.loc[lens[i]]['param']['phi']
    phi_error_low_555.append(lower_error_phi_555)
    phi_error_hi_555.append(upper_error_phi_555)

    einstein_radius_rescaled = lens_galaxy.einstein_radius_in_units(unit_length='arcsec')\
                               *((2*np.sqrt(results.loc[lens[i]]['param']['axis_ratio']))/(1+results.loc[lens[i]]['param']['axis_ratio']))

    R_Ein.append(einstein_radius)
    R_Ein_555.append(einstein_radius_555)
    R_Ein_rescaled.append(einstein_radius_rescaled)

    M_Ein.append(einstein_mass)
    M_Ein_555.append(einstein_mass_555)

y_err_axis_ratio = np.array([axis_ratio_error_low, axis_ratio_error_hi])
y_err_phi = np.array([phi_error_low, phi_error_hi])
y_err_axis_ratio_555 = np.array([axis_ratio_error_low_555, axis_ratio_error_hi_555])
y_err_phi_555 = np.array([phi_error_low_555, phi_error_hi_555])

## calcualting fractional change for plots
R_Ein_frac = (R_Ein - slacs['b_SIE'])/slacs['b_SIE']
R_Ein_frac_555 = (R_Ein_555 - slacs['b_SIE'])/slacs['b_SIE']
M_Ein_frac = (np.log10(M_Ein) - slacs['log[Me/Mo]'])/slacs['log[Me/Mo]']
M_Ein_frac_555 = (np.log10(M_Ein) - slacs['log[Me/Mo]'])/slacs['log[Me/Mo]']

R_Ein_rescaled_frac = (R_Ein_rescaled - slacs['b_SIE'])/slacs['b_SIE']
R_Ein_autolens_frac = (R_Ein_rescaled - np.array(R_Ein))/np.array(R_Ein)

fig1, ax = plt.subplots(figsize=(10,5))
for i in range(len(lens)):
    F814W = ax.scatter(results.loc[lens[i]]['param']['axis_ratio'], R_Ein_frac[i], label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i])
    F555W = ax.scatter(results_555.loc[lens[i]]['param']['axis_ratio'], R_Ein_frac_555[i],
               marker=slacs['marker'][i], edgecolors=slacs['colour'][i], facecolors='none')
    legend1 = ax.legend([F814W, F555W], ['F814W', 'F555W'], loc='lower left', title='Wavelength')
    ax.add_artist(legend1)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
legend2 = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title= 'Lens Name')
ax.add_artist(legend1)
plt.axhline(y=0, color='k',)
plt.axhline(y=0.02, color='k', linestyle='--')
plt.axhline(y=-0.02, color='k', linestyle='--')
plt.xlabel('Axis ratio', size=20)
plt.ylabel(r'$\frac{R_{Ein_{crit}}-R_{Ein_{SLACS}}}{R_{Ein_{SLACS}}}$', size=21)


fig2, ax = plt.subplots(figsize=(10,5))
for i in range(len(lens)):
    ax.scatter(results.loc[lens[i]]['param']['axis_ratio'], R_Ein_rescaled_frac[i], label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title= 'Lens Name')
plt.axhline(y=0, color='k',)
plt.axhline(y=0.02, color='k', linestyle='--')
plt.axhline(y=-0.02, color='k', linestyle='--')
plt.xlabel('Axis ratio', size=20)
plt.ylabel(r'$\frac{R_{Ein_{rescaled}}-R_{Ein_{SLACS}}}{R_{Ein_{SLACS}}}$', size=21)

fig3, ax = plt.subplots(figsize=(10,5))
for i in range(len(lens)):
    ax.scatter(results.loc[lens[i]]['param']['axis_ratio'], R_Ein_autolens_frac[i], label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title= 'Lens Name')
plt.axhline(y=0, color='k',)
plt.axhline(y=0.02, color='k', linestyle='--')
plt.axhline(y=-0.02, color='k', linestyle='--')
plt.xlabel('Axis ratio', size=20)
plt.ylabel(r'$\frac{R_{Ein_{crit}}-R_{Ein_{autolens}}}{R_{Ein_{crit}}}$', size=21)

plt.show()

fig3, ax = plt.subplots(figsize=(5,5))
for i in range(len(lens)):
    F814W = ax.scatter(slacs['log[Me/Mo]'][i], np.array(np.log10(M_Ein[i])), label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i],)
    F555W = ax.scatter(slacs['log[Me/Mo]'][i], np.array(np.log10(M_Ein_555[i])), marker=slacs['marker'][i], edgecolors=slacs['colour'][i], facecolors='none')
legend1 = ax.legend([F814W, F555W], ['F814W', 'F555W'], loc='upper left', title='Wavelength')
ax.add_artist(legend1)
legend2 = ax.legend(loc='lower right', title= 'Lens Name')
plt.xlabel(r'$M_{Ein_{SLACS}}$', size=20)
plt.ylabel(r'$M_{Ein_{autolens}}$', size=20)
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

fig4, ax = plt.subplots(figsize=(10,5))
for i in range(len(lens)):
    F814W = ax.scatter(results.loc[lens[i]]['param']['axis_ratio'], M_Ein_frac[i], label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i])
    F555W = ax.scatter(results_555.loc[lens[i]]['param']['axis_ratio'], M_Ein_frac_555[i],
               marker=slacs['marker'][i], edgecolors=slacs['colour'][i], facecolors='none')
legend1 = ax.legend([F814W, F555W], ['F814W', 'F555W'], loc='lower left', title='Wavelength')
ax.add_artist(legend1)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
legend2 = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title= 'Lens Name')
plt.axhline(y=0, color='k',)
plt.axhline(y=0.02, color='k', linestyle='--')
plt.axhline(y=-0.02, color='k', linestyle='--')
plt.xlabel('Axis ratio', size=20)
plt.ylabel(r'$\frac{M_{Ein_{autolens}}-M_{Ein_{SLACS}}}{M_{Ein_{SLACS}}}$', size=21)

fig5, ax = plt.subplots(figsize=(5,5))
for i in range(len(lens)):
    F814W = ax.scatter(slacs['q_SIE'][i], results.loc[lens[i]]['param']['axis_ratio'], label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i],)
    ax.errorbar(slacs['q_SIE'][i], results.loc[lens[i]]['param']['axis_ratio'],
                yerr=np.array([y_err_axis_ratio[:, i]]).T,
                color=slacs['colour'][i], elinewidth=1, fmt='none', capsize=3, label=None)
    F555W= ax.scatter(slacs['q_SIE'][i], results_555.loc[lens[i]]['param']['axis_ratio'], marker=slacs['marker'][i], edgecolors=slacs['colour'][i], facecolors='none')
    ax.errorbar(slacs['q_SIE'][i], results_555.loc[lens[i]]['param']['axis_ratio'],
                yerr=np.array([y_err_axis_ratio_555[:, i]]).T,
                color=slacs['colour'][i], elinewidth=1, fmt='none', capsize=3, label=None)
legend1 = ax.legend([F814W, F555W], ['F814W', 'F555W'], loc='upper left', title='Wavelength')
ax.add_artist(legend1)
legend2 = ax.legend(loc='lower right', title= 'Lens Name')
ax.add_artist(legend1)
plt.xlabel(r'$q_{SLACS}$', size=20)
plt.ylabel(r'$q_{autolens}$', size=20)
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

fig6, ax = plt.subplots(figsize=(5,5))
for i in range(len(lens)):
    F814W = ax.scatter(slacs['PA'][i], results.loc[lens[i]]['param']['phi'], label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i],)
    ax.errorbar(slacs['PA'][i], results.loc[lens[i]]['param']['phi'],
                yerr=np.array([y_err_phi[:, i]]).T,
                color=slacs['colour'][i], elinewidth=1, fmt='none', capsize=3, label=None)
    F555W= ax.scatter(slacs['PA'][i], results_555.loc[lens[i]]['param']['phi'], marker=slacs['marker'][i], edgecolors=slacs['colour'][i], facecolors='none')
    ax.errorbar(slacs['PA'][i], results_555.loc[lens[i]]['param']['phi'],
                yerr=np.array([y_err_phi_555[:, i]]).T,
                color=slacs['colour'][i], elinewidth=1, fmt='none', capsize=3, label=None)
legend1 = ax.legend([F814W, F555W], ['F814W', 'F555W'], loc='upper left', title='Wavelength')
ax.add_artist(legend1)
legend2 = ax.legend(loc='lower right', title= 'Lens Name')
plt.xlabel(r'$\Phi_{SLACS}$', size=20)
plt.ylabel(r'$\Phi_{autolens}$', size=20)
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

plt.show()
