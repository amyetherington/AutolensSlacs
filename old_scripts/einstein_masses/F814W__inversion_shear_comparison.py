from autofit import conf

import os

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))

conf.instance = conf.Config(config_path=path+'config', output_path=path+'output')

import pandas as pd
import numpy as np
import autolens as al
import autoastro as astro
from astropy import cosmology


import matplotlib.pyplot as plt
from pathlib import Path

M_o = 1.989e30
cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)
c = 2.998e8
G = 6.674e-11
Mpc_to_m = 3.085677581e22

## setting up paths to shear and no shear results (sersic source)
data_path = '{}/../../../../../output/slacs_final/F814W/'.format(os.path.dirname(os.path.realpath(__file__)))
slacs_path = '{}/../../data/slacs_data_table.xlsx'.format(os.path.dirname(os.path.realpath(__file__)))
pipeline = '/pipeline_inv_hyper__lens_bulge_disk_sie__source_inversion/'
no_shear = 'pipeline_tag__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/' \
           'phase_4__lens_bulge_disk_sie__source_inversion/phase_tag__sub_2__pos_1.00/model.results'
shear = 'pipeline_tag__with_shear__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/' \
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

## creating varaibles for loading results
list_ = []
list_shear_ = []
lens = []
lens_shear=[]
SLACS_image = []
radius_from_area = []
radius_from_area_shear = []

## loading table of slacs parameters (deleting two discarded lenses)
slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name
slacs = slacs.drop(['slacs0959+5100',  'slacs0959+4416'], axis=0)


## loading shear and no shear results into two sepearate pandas data frames
## deleting any lenses from slacs table that don't have autolens results files
for i in range(len(lens_name)):
    full_data_path = Path(data_path + lens_name[i] + pipeline + no_shear)
    full_data_path_shear = Path(data_path + lens_name[i] + pipeline + shear)
    if full_data_path.is_file() and full_data_path_shear.is_file():
        data = pd.read_csv(full_data_path, sep='\s+', header=None, skiprows=18, nrows=6,).set_index(0)
        data_shear = pd.read_csv(full_data_path_shear, sep='\s+', header=None, skiprows=21, nrows=9, ).set_index(0)
        del data.index.name
        data[2] = data[2].str.strip('(,').astype(float)
        data[3] = data[3].str.strip(')').astype(float)
        data.columns=['param', '-error', '+error']
        list_.append(data)
        lens.append(lens_name[i])
        results = pd.concat(list_, keys=lens)
        del data_shear.index.name
        data_shear[2] = data_shear[2].str.strip('(,').astype(float)
        data_shear[3] = data_shear[3].str.strip(')').astype(float)
        data_shear.columns = ['param', '-error', '+error']
        list_shear_.append(data_shear)
        lens_shear.append(lens_name[i])
        results_shear = pd.concat(list_shear_, keys=lens_shear)
        model_image_path = data_path + lens_name[i] + '/pipeline_inv_hyper__lens_bulge_disk_sie__source_inversion/' \
                                                      'pipeline_tag__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/' \
                                                      'phase_4__lens_bulge_disk_sie__source_inversion/phase_tag__sub_2__pos_1.00/' \
                                                      'image/lens_fit/fits/fit_model_image_of_plane_1.fits'
        model_image = al.Array.from_fits(file_path=model_image_path, hdu=0, pixel_scales=0.03)

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
mass_from_crit_curv = []



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

    ## galaxies with external shear component
    lens_galaxy_shear = al.Galaxy(
        mass=al.mp.EllipticalIsothermal(
            centre=(results_shear.loc[lens[i]]['param']['centre_0'], results_shear.loc[lens[i]]['param']['centre_1']),
            axis_ratio=results_shear.loc[lens[i]]['param']['axis_ratio'], phi=results_shear.loc[lens[i]]['param'][1],
            einstein_radius=results_shear.loc[lens[i]]['param']['einstein_radius']),
        shear=al.mp.ExternalShear(
            magnitude=results_shear.loc[lens[i]]['param']['magnitude'],phi=results_shear.loc[lens[i]]['param'][8]),
        redshift=slacs['z_lens'][i]
    )

    ## calculating einstein masses
    einstein_mass = lens_galaxy.einstein_mass_in_units(
        unit_mass='solMass', redshift_source=slacs['z_source'][i], cosmology=cosmo
    )

    einstein_mass_shear = lens_galaxy_shear.einstein_mass_in_units(
        unit_mass='solMass', redshift_source=slacs['z_source'][i], cosmology=cosmo
    )

    grid = al.Grid.uniform(
        shape_2d=SLACS_image[i].shape_2d, pixel_scales=0.03, sub_size=2)

    critical_curves = lens_galaxy.critical_curves_from_grid(grid=grid)
    critical_curves_shear = lens_galaxy_shear.critical_curves_from_grid(grid=grid)

    critical_curve_tan, critical_curve_rad = critical_curves[0], critical_curves[1]
    critical_curve_tan_shear, critical_curve_rad_shear = critical_curves_shear[0], critical_curves_shear[1]

    x = critical_curve_tan[:, 0]
    y = critical_curve_tan[:, 1]
    area = np.abs(0.5 * np.sum(y[:-1] * np.diff(x) - x[:-1] * np.diff(y)))

    x_shear = critical_curve_tan_shear[:, 0]
    y_shear = critical_curve_tan_shear[:, 1]
    area_shear = np.abs(0.5 * np.sum(y_shear[:-1] * np.diff(x_shear) - x_shear[:-1] * np.diff(y_shear)))

    Rad_area = np.sqrt(area / np.pi)
    Rad_area_shear = np.sqrt(area_shear / np.pi)

    D_l = cosmo.angular_diameter_distance(slacs['z_lens'][i]).value*Mpc_to_m
    D_s = cosmo.angular_diameter_distance(slacs['z_source'][i]).value*Mpc_to_m
    D_ls = cosmo.angular_diameter_distance_z1z2(slacs['z_lens'][i], slacs['z_source'][i]).value*Mpc_to_m

    Sigma_crit = astro.util.cosmo.critical_surface_density_between_redshifts_from_redshifts_and_cosmology(
        redshift_0=slacs['z_lens'][i], redshift_1=slacs['z_source'][i], cosmology=cosmo,
        unit_length='arcsec', unit_mass='kg')

    Mass_crit_curv = Sigma_crit*np.pi*(Rad_area**2)
    


    ## rescaling einstein masses to compare to slacs results
    einstein_mass_rescaled = einstein_mass * ((1 + results.loc[lens[i]]['param']['axis_ratio']) / 2)

    ## appending einstein mass results
    M_Ein.append(einstein_mass)
    M_Ein_shear.append(einstein_mass_shear)
    M_Ein_rescaled.append(einstein_mass_rescaled)
    radius_from_area.append(Rad_area)
    radius_from_area_shear.append(Rad_area_shear)
    mass_from_crit_curv.append(Mass_crit_curv/M_o)
print(radius_from_area)

## calcualting fractional change for plots
M_Ein_fractional_change = (np.log10(M_Ein) - slacs['log[Me/Mo]'])/slacs['log[Me/Mo]']
M_Ein_fractional_change_rescaled = (np.log10(M_Ein_rescaled) - slacs['log[Me/Mo]'])/slacs['log[Me/Mo]']
r_area_frac = (radius_from_area-slacs['b_SIE'])/slacs['b_SIE']
r_area_frac_shear = (radius_from_area_shear-slacs['b_SIE'])/slacs['b_SIE']
M_Ein_crit_curv_frac = (np.log10(mass_from_crit_curv) - slacs['log[Me/Mo]'])/slacs['log[Me/Mo]']

##plotting
fig, ax = plt.subplots()
for i in range(len(M_Ein)):
    ax.scatter(slacs['log[Me/Mo]'][i], np.log10(M_Ein[i]), label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i],)
  #  ax.errorbar(slacs['log[Me/Mo]'][i], np.log10(M_Ein[i]), yerr=np.array([y_err[:,i]]).T,
   #             color=slacs['colour'][i], elinewidth=1, fmt='none', capsize=3, label=None)
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
   # ax.errorbar(slacs['R_Ein'][i], results.loc[lens[i]]['param']['lens_galaxies_lens_mass_einstein_radius_value'], yerr=np.array([y_err[:,i]]).T, elinewidth=1, fmt='none', capsize=3, label=None)

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
    ax.scatter(slacs['log[Me/Mo]'][i], np.log10(M_Ein_shear[i]), label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i],)
 #   ax.errorbar(slacs['log[Me/Mo]'][i], np.log10(M_Ein_shear[i]), yerr=np.array([y_err_shear[:, i]]).T,
 #               color=slacs['colour'][i], elinewidth=1, fmt='none', capsize=3, label=None)
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
plt.title('Einstein mass with shear')
plt.close()

fig9, ax = plt.subplots()
for i in range(len(M_Ein)):
    ax.scatter(results_shear.loc[lens[i]]['param']['einstein_radius'], results.loc[lens[i]]['param']['einstein_radius'],
               color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
   # ax.errorbar(slacs['R_Ein'][i], results.loc[lens[i]]['param']['lens_galaxies_lens_mass_einstein_radius_value'], yerr=np.array([y_err[:,i]]).T, elinewidth=1, fmt='none', capsize=3, label=None)

ax.legend()

plt.xlabel(r'R$_{Ein}(shear)$', size=14)
plt.ylabel(r'R$_{Ein}$', size=14)
plt.title('Einstein radius shear comparison')
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.close()

fig10, ax = plt.subplots()
for i in range(len(M_Ein)):
    ax.scatter(results_shear.loc[lens[i]]['param']['axis_ratio'], results.loc[lens[i]]['param']['axis_ratio'],
               color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
   # ax.errorbar(slacs['R_Ein'][i], results.loc[lens[i]]['param']['lens_galaxies_lens_mass_einstein_radius_value'], yerr=np.array([y_err[:,i]]).T, elinewidth=1, fmt='none', capsize=3, label=None)

ax.legend()

plt.xlabel(r'q$_{shear}$', size=14)
plt.ylabel('q', size=14)
plt.title('Axis ratio shear comparison')
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.close()

fig11, ax = plt.subplots()
for i in range(len(M_Ein)):
    ax.scatter(results_shear.loc[lens[i]]['param'][1], results.loc[lens[i]]['param']['phi'],
               color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
   # ax.errorbar(slacs['R_Ein'][i], results.loc[lens[i]]['param']['lens_galaxies_lens_mass_einstein_radius_value'], yerr=np.array([y_err[:,i]]).T, elinewidth=1, fmt='none', capsize=3, label=None)

ax.legend()

plt.xlabel(r'$\phi_{shear}$', size=14)
plt.ylabel(r'$\phi$', size=14)
plt.title('Position angle shear comparison')
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.close()

fig12, ax = plt.subplots()
for i in range(len(M_Ein)):
    ax.scatter(M_Ein_shear[i], M_Ein[i],
               color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
   # ax.errorbar(slacs['R_Ein'][i], results.loc[lens[i]]['param']['lens_galaxies_lens_mass_einstein_radius_value'], yerr=np.array([y_err[:,i]]).T, elinewidth=1, fmt='none', capsize=3, label=None)

ax.legend()

plt.xlabel(r'M$_{Ein}(shear)$', size=14)
plt.ylabel(r'M$_{Ein}$', size=14)
plt.title('Einstein mass shear comparison')
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.close()

fig5, ax = plt.subplots(figsize=(10,5))
for i in range(len(M_Ein)):
    ax.scatter(results.loc[lens[i]]['param']['axis_ratio'], r_area_frac[i], label=slacs.index[i], marker=slacs['marker'][i], color='c')
   # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.scatter(results_shear.loc[lens[i]]['param']['axis_ratio'], r_area_frac_shear[i], label=slacs.index[i],
               marker=slacs['marker'][i], color='orange')
plt.axhline(y=0, color='k',)
plt.axhline(y=0.02, color='k', linestyle='--')
plt.axhline(y=-0.02, color='k', linestyle='--')
plt.xlabel('axis ratio', size=14)
plt.ylabel(r'$\frac{R_{autolens}-R_{SLACS}}{R_{SLACS}}$', size=14)

#plt.close()

fig6, ax = plt.subplots(figsize=(10,5))
for i in range(len(M_Ein)):
    ax.scatter(results.loc[lens[i]]['param']['axis_ratio'], M_Ein_crit_curv_frac[i], label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.axhline(y=0, color='k',)
plt.axhline(y=0.02, color='k', linestyle='--')
plt.axhline(y=-0.02, color='k', linestyle='--')
plt.xlabel('axis ratio', size=14)
plt.ylabel(r'$\frac{M_{E_{autolens}}-M_{E_{SLACS}}}{M_{E_{SLACS}}}$', size=14)
#plt.close()

plt.show()
