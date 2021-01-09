from autofit import conf

import os

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))

conf.instance = conf.Config(config_path=path+'config', output_path=path+'output')

import pandas as pd
import numpy as np
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.galaxy import galaxy as g
from autolens.model.galaxy import galaxy_model as gm
from autolens.lens import ray_tracing
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from astropy import cosmology
from autolens.array import grids
from autolens.array import scaled_array

from autolens.data.instrument import ccd
from autolens.lens import lens_data as ld

import matplotlib.pyplot as plt
from pathlib import Path

M_o = 1.989e30
cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)

## setting up paths to shear and no shear results (sersic source)
data_path = '{}/../../../../../output/slacs_final/F555W/'.format(os.path.dirname(os.path.realpath(__file__)))
slacs_path = '{}/../../data/slacs_data_table.xlsx'.format(os.path.dirname(os.path.realpath(__file__)))
pipeline = '/pipeline_inv_hyper__lens_bulge_disk_sie__source_inversion/'
no_shear = 'pipeline_tag__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/' \
           'phase_4__lens_bulge_disk_sie__source_inversion/phase_tag__sub_2__pos_1.00/model.results'
shear = 'pipeline_tag__hyper_galaxies__with_shear__bd_align_centre__disk_sersic/' \
        'phase_4__lens_bulge_disk_sie__source_sersic/phase_tag__sub_2__pos_1.00/model.results'

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

## creating varaibles for loading results
list_ = []
list_shear_ = []
lens = []
lens_shear=[]
SLACS_image = []

## loading table of slacs parameters (deleting two discarded lenses)
slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name
slacs = slacs.drop(['slacs0959+5100',  'slacs0959+4416'], axis=0)


## loading no shear results into two sepearate pandas data frames
## deleting any lenses from slacs table that don't have autolens results files
for i in range(len(lens_name)):
    full_data_path = Path(data_path + lens_name[i] + pipeline + no_shear)
    if full_data_path.is_file():
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
    else:
        slacs = slacs.drop([lens_name[i]], axis=0)

## creating variables for einstin mass results
M_Ein = []
M_Ein_error_hi = []
M_Ein_error_low = []
M_Ein_shear = []
M_Ein_error_hi_shear = []
M_Ein_error_low_shear = []
M_Ein_rescaled = []
q_fractional_change = []
q_error_hi = []
q_error_low = []
q_autolens = []
r_autolens = []
r_rescaled = []
M_ellipse = []
radius_geometric_mean = []
radius_from_area = []


## creating galaxy in autolens from autolens mass profile parameters and redshift of lens galaxy
## hi and low correspond to errors on measurement
for i in range(len(lens)):
    ## galaxies without external shear component
    lens_galaxy = al.Galaxy(
        mass=al.mp.EllipticalIsothermal(
            centre=(results.loc[lens[i]]['param']['centre_0'], results.loc[lens[i]]['param']['centre_1']),
            axis_ratio=results.loc[lens[i]]['param']['axis_ratio'], phi=results.loc[lens[i]]['param']['phi'],
            einstein_radius=results.loc[lens[i]]['param']['einstein_radius']), redshift = slacs['z_lens'][i]
    )

    lens_galaxy_slacs = al.Galaxy(
        mass=al.mp.EllipticalIsothermal(
            centre=(results.loc[lens[i]]['param']['centre_0'], results.loc[lens[i]]['param']['centre_1']),
            axis_ratio=slacs['q_SIE'][i], phi=slacs['PA'][i],
            einstein_radius=slacs['b_SIE'][i]), redshift=slacs['z_lens'][i]
    )

    lens_galaxy_hi = al.Galaxy(
        mass=al.mp.EllipticalIsothermal(
            centre=(results.loc[lens[i]]['param']['centre_0'], results.loc[lens[i]]['param']['centre_1']),
            axis_ratio=results.loc[lens[i]]['-error']['axis_ratio'],phi=results.loc[lens[i]]['param']['phi'],
            einstein_radius=results.loc[lens[i]]['+error']['einstein_radius']), redshift=slacs['z_lens'][i]
    )

    lens_galaxy_low = al.Galaxy(
        mass=al.mp.EllipticalIsothermal(
            centre=(results.loc[lens[i]]['param']['centre_0'], results.loc[lens[i]]['param']['centre_1']),
            axis_ratio=results.loc[lens[i]]['+error']['axis_ratio'],phi=results.loc[lens[i]]['param']['phi'],
            einstein_radius=results.loc[lens[i]]['-error']['einstein_radius']), redshift=slacs['z_lens'][i]
    )

    grid = al.Grid.uniform(
        shape_2d=SLACS_image[i].shape, pixel_scales=0.03, sub_size=2)

    critical_curves = lens_galaxy.critical_curves_from_grid(grid=grid)

    critical_curve_tan, critical_curve_rad = critical_curves[0], critical_curves[1]

    x = critical_curve_tan[:, 0]
    y = critical_curve_tan[:, 1]
    area = np.abs(0.5 * np.sum(y[:-1] * np.diff(x) - x[:-1] * np.diff(y)))

    Rad_area = np.sqrt(area/np.pi)


    ## calculating einstein masses
    einstein_mass = lens_galaxy.einstein_mass_in_units(
        unit_mass='solMass', redshift_source=slacs['z_source'][i], cosmology=cosmo
    )

    einstein_mass_error_hi = lens_galaxy_hi.einstein_mass_in_units(
        unit_mass='solMass', redshift_source=slacs['z_source'][i], cosmology=cosmo
    )
    einstein_mass_error_low = lens_galaxy_low.einstein_mass_in_units(
        unit_mass='solMass', redshift_source=slacs['z_source'][i], cosmology=cosmo
    )

    q_lower_error = results.loc[lens[i]]['param']['axis_ratio'] - results.loc[lens[i]]['-error']['axis_ratio']
    q_upper_error = results.loc[lens[i]]['param']['axis_ratio'] - results.loc[lens[i]]['+error']['axis_ratio']

    ## rescaling einstein masses to compare to slacs results
    einstein_mass_rescaled = einstein_mass * ((1 + results.loc[lens[i]]['param']['axis_ratio']) / 2)

    axis_ratio_autolens = results.loc[lens[i]]['param']['axis_ratio']
    einstein_radius_autolens = results.loc[lens[i]]['param']['einstein_radius']

    einstein_radius_rescaled = einstein_radius_autolens *(2*np.sqrt(axis_ratio_autolens)/(1+axis_ratio_autolens))

    mass_in_ellipse_SLACS = lens_galaxy_slacs.mass_within_ellipse_in_units(
        major_axis=slacs['b_SIE'][i]/np.sqrt(slacs['q_SIE'][i]),
        unit_mass='solMass', redshift_source=slacs['z_source'][i],
        cosmology=cosmo)

    mass_in_ellipse = lens_galaxy.mass_within_ellipse_in_units(
        major_axis=einstein_radius_autolens/ np.sqrt(axis_ratio_autolens),
        unit_mass='solMass', redshift_source=slacs['z_source'][i],
        cosmology=cosmo)

    sie = mp.EllipticalIsothermal(
        centre=(results.loc[lens[i]]['param']['centre_0'], results.loc[lens[i]]['param']['centre_1']),
        axis_ratio=slacs['q_SIE'][i], phi=slacs['PA'][i],
        einstein_radius=slacs['b_SIE'][i])


    ## appending einstein mass results
    M_Ein.append(einstein_mass)
    M_Ein_error_hi.append(einstein_mass_error_hi)
    M_Ein_error_low.append(einstein_mass_error_low)
    M_Ein_rescaled.append(einstein_mass_rescaled)
    q_autolens.append(axis_ratio_autolens)
    r_autolens.append(einstein_radius_autolens)
    r_rescaled.append(einstein_radius_rescaled)
    M_ellipse.append(mass_in_ellipse)
    radius_from_area.append(Rad_area)

    q_error_hi.append(q_upper_error)
    q_error_low.append(q_lower_error)

## calculating errors for error bars
lower_error = np.array(np.log10(M_Ein))-np.array(np.log10(M_Ein_error_low))
upper_error = np.array(np.log10(M_Ein_error_hi))-np.array(np.log10(M_Ein))
y_err = np.array([lower_error, upper_error])

q_y_err = np.array([q_error_low, q_error_hi])



## calcualting fractional change for plots
M_Ein_fractional_change = (np.log10(M_Ein) - slacs['log[Me/Mo]'])/slacs['log[Me/Mo]']
M_Ein_fractional_change_rescaled = (np.log10(M_Ein_rescaled) - slacs['log[Me/Mo]'])/slacs['log[Me/Mo]']
q_frac = (slacs['q_SIE']-q_autolens)/slacs['q_SIE']
r_frac = (r_autolens-slacs['b_SIE'])/slacs['b_SIE']
r_rescaled_frac = (r_rescaled-slacs['b_SIE'])/slacs['b_SIE']
M_ellipse_frac = (np.log10(M_ellipse) - slacs['log[Me/Mo]'])/slacs['log[Me/Mo]']
r_area_frac = (radius_from_area-slacs['b_SIE'])/slacs['b_SIE']

##plotting
fig, ax = plt.subplots()
for i in range(len(M_Ein)):
    ax.scatter(slacs['log[Me/Mo]'][i], np.log10(M_Ein[i]), label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i],)
    ax.errorbar(slacs['log[Me/Mo]'][i], np.log10(M_Ein[i]), yerr=np.array([y_err[:,i]]).T,
                color=slacs['colour'][i], elinewidth=1, fmt='none', capsize=3, label=None)
ax.legend()
plt.xlabel(r'log(M$_{Ein}$/M$_{\odot}$)$_{SLACS}$', size=14)
plt.ylabel(r'log(M$_{Ein}$/M$_{\odot}$)$_{PyAutoLens}$', size=14)
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.title('Einstein mass without shear')
plt.close()

fig2, ax = plt.subplots()
for i in range(len(M_Ein)):
    ax.scatter(results.loc[lens[i]]['param']['axis_ratio'], M_Ein_fractional_change[i], label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i])
ax.legend()
plt.axhline(y=0, color='k',)
plt.axhline(y=0.02, color='k', linestyle='--')
plt.axhline(y=-0.02, color='k', linestyle='--')
plt.xlabel('axis ratio', size=14)
plt.ylabel(r'$\frac{M_{E_{autolens}}-M_{E_{SLACS}}}{M_{E_{SLACS}}}$', size=14)
plt.title('Fractional difference in Einstein mass as a function of axis ratio')
plt.close()

fig3, ax = plt.subplots()
for i in range(len(M_Ein)):
    ax.scatter(results.loc[lens[i]]['param']['axis_ratio'], M_Ein_fractional_change_rescaled[i], label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i])
ax.legend()
plt.axhline(y=0, color='k',)
plt.axhline(y=0.02, color='k', linestyle='--')
plt.axhline(y=-0.02, color='k', linestyle='--')
plt.xlabel('axis ratio', size=14)
plt.ylabel(r'$\frac{M_{E_{autolens}}-M_{E_{SLACS}}}{M_{E_{SLACS}}}$', size=14)
plt.title('Fractional difference in rescaled Einstein mass as a function of axis ratio')
plt.close()

fig4, ax = plt.subplots()
for i in range(len(M_Ein)):
    ax.scatter(slacs['log[Me/Mo]'][i], np.log10(M_Ein_rescaled[i]), label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i],)
ax.legend()
plt.xlabel(r'log(M$_{Ein}$/M$_{\odot}$)$_{SLACS}$', size=14)
plt.ylabel(r'log(M$_{Ein}$/M$_{\odot}$)$_{PyAutoLens}$', size=14)
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.title('Einstein mass rescaled')
plt.close()

fig5, ax = plt.subplots()
for i in range(len(M_Ein)):
    ax.scatter(slacs['b_SIE'][i], results.loc[lens[i]]['param']['einstein_radius'],
               color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
   # ax.errorbar(slacs['R_Ein'][i], results.loc[lens[i]]['param']['lens_galaxies_lens_mass_einstein_radius_value'], yerr=np.array([y_err[:,i]]).T, elinewidth=1, fmt='none', capsize=3, label=None)

ax.legend()

plt.xlabel(r'b(SIE)$_{SLACS}$', size=14)
plt.ylabel(r'b(SIE)$_{PyAutoLens}$', size=14)
plt.title('Einstein radius comparison with SLACS')

lims = [
    np.min([0.9, 2]),  # min of both axes
    np.max([0.9, 2]),  # max of both axes
]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.close()

fig6, ax = plt.subplots()
for i in range(len(M_Ein)):
    ax.scatter(slacs['q_SIE'][i], results.loc[lens[i]]['param']['axis_ratio'],
               color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
  #  ax.errorbar(slacs['q_SIE'][i], results.loc[lens[i]]['param']['axis_ratio'], yerr=np.array([q_y_err[:,i]]).T, elinewidth=1, fmt='none', capsize=3, label=None)

ax.legend()

plt.xlabel(r'q$_{SLACS}$', size=14)
plt.ylabel(r'q$_{PyAutoLens}$', size=14)
plt.title('Axis ratio comparison with SLACS')

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.close()

fig7, ax = plt.subplots()
for i in range(len(M_Ein)):
    ax.scatter(slacs['PA'][i], results.loc[lens[i]]['param']['phi'],
               color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
ax.legend()
plt.xlabel(r'$\phi_{SLACS}$', size=14)
plt.ylabel(r'$\phi_{PyAutoLens}$', size=14)
plt.title('Rotation angle comparison with SLACS')

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.close()

fig8, ax = plt.subplots()
for i in range(len(M_Ein)):
    ax.scatter(results.loc[lens[i]]['param']['einstein_radius'], q_frac[i], label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i])
ax.legend()
plt.axhline(y=0, color='k',)
plt.axhline(y=0.02, color='k', linestyle='--')
plt.axhline(y=-0.02, color='k', linestyle='--')
plt.xlabel('einstein radius', size=14)
plt.ylabel(r'$\frac{M_{E_{autolens}}-M_{E_{SLACS}}}{M_{E_{SLACS}}}$', size=14)
plt.title('fractional change in axis ratio as a function of einstein radius')
plt.close()

fig9, ax = plt.subplots()
for i in range(len(M_Ein)):
    ax.scatter(results.loc[lens[i]]['param']['axis_ratio'], r_frac[i], label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i])
ax.legend()
plt.axhline(y=0, color='k',)
plt.axhline(y=0.02, color='k', linestyle='--')
plt.axhline(y=-0.02, color='k', linestyle='--')
plt.xlabel('einstein radius', size=14)
plt.ylabel(r'$\frac{R_{E_{autolens}}-R_{E_{SLACS}}}{R_{E_{SLACS}}}$', size=14)
plt.title('fractional change in radius as a function of axis ratio')
plt.close()

fig10, ax = plt.subplots()
for i in range(len(M_Ein)):
    ax.scatter(results.loc[lens[i]]['param']['axis_ratio'], r_rescaled_frac[i], label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i])
ax.legend()
plt.axhline(y=0, color='k',)
plt.axhline(y=0.02, color='k', linestyle='--')
plt.axhline(y=-0.02, color='k', linestyle='--')
plt.xlabel('einstein radius', size=14)
plt.ylabel(r'$\frac{R_{E_{autolens}}-R_{E_{SLACS}}}{R_{E_{SLACS}}}$', size=14)
plt.title('fractional change in radius rescaled as a function of axis ratio')
plt.close()

fig11, ax = plt.subplots()
for i in range(len(M_Ein)):
    ax.scatter(results.loc[lens[i]]['param']['axis_ratio'], M_ellipse_frac[i], label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i])
ax.legend()
plt.axhline(y=0, color='k',)
plt.axhline(y=0.02, color='k', linestyle='--')
plt.axhline(y=-0.02, color='k', linestyle='--')
plt.xlabel('einstein radius', size=14)
plt.ylabel(r'$\frac{R_{E_{autolens}}-R_{E_{SLACS}}}{R_{E_{SLACS}}}$', size=14)
plt.title('fractional change in mass rescaled as a function of axis ratio')
plt.close()

fig12, ax = plt.subplots(figsize=(10,4))
for i in range(len(M_Ein)):
    ax.scatter(results.loc[lens[i]]['param']['axis_ratio'], r_area_frac[i], label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.axhline(y=0, color='k',)
plt.axhline(y=0.02, color='k', linestyle='--')
plt.axhline(y=-0.02, color='k', linestyle='--')
plt.xlabel('Axis ratio', size=21)
plt.ylabel(r'$\frac{R_{Ein_{autolens}}-R_{Ein_{SLACS}}}{R_{Ein_{SLACS}}}$', size=21)
#plt.title('fractional change in radius rescaled as a function of axis ratio')
#plt.close()

plt.show()
