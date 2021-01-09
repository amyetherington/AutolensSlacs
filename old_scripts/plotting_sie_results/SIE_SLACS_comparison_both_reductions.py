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
data_path = '{}/../../../../../output/slacs'.format(os.path.dirname(os.path.realpath(__file__)))
slacs_path = '{}/../../data/slacs_data_table.xlsx'.format(os.path.dirname(os.path.realpath(__file__)))
pipeline = '/pipeline_inv_hyper__lens_bulge_disk_sie__source_inversion/' \
           'pipeline_tag__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/' \
           'phase_4__lens_bulge_disk_sie__source_inversion/phase_tag__sub_2__pos_1.00/model.results'
fig_path = '/Users/dgmt59/Documents/Plots/error_analysis/SIE/shu_v_david_'

lens_name = np.array(['slacs0216-0813',
                      'slacs0252+0039',
                      'slacs0737+3216',
                      'slacs0912+0029',
                      'slacs0959+4410',
                      'slacs1205+4910',
                      'slacs1250+0523',
                      'slacs1402+6321',
                      'slacs1430+4105',
                      'slacs1627+0053',
                      'slacs1630+4520',
                      'slacs2238-0754',
                      'slacs2300+0022',
                      'slacs2303+1422'])

lens_name_shu = np.array(['slacs0216-0813',
                      'slacs0252+0039',
                      'slacs0737+3216',
                      'slacs0912+0029',
                      'slacs0959+0410',
                      'slacs1205+4910',
                      'slacs1250+0523',
                      'slacs1402+6321',
                      'slacs1430+4105',
                      'slacs1627-0053',
                      'slacs1630+4520',
                      'slacs2238-0754',
                      'slacs2300+0022',
                      'slacs2303+1422'])

## creating variables for loading results
list_ = []
lens = []
SLACS_image = []
list_shu = []
lens_shu = []
SLACS_image_shu = []

## loading table of slacs parameters (deleting two discarded lenses)
slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name
slacs = slacs.drop(['slacs0959+5100',  'slacs0959+4416', 'slacs1420+6019'], axis=0)

## loading results into pandas data frames
## deleting any lenses from slacs table that don't have autolens results files
for i in range(len(lens_name)):
    full_data_path = Path(data_path + '_fresh/F814W/' + lens_name[i] + pipeline)
    full_data_path_shu = Path(data_path + '_shu_shu/F814W/' + lens_name_shu[i] + pipeline)
    if full_data_path.is_file() and full_data_path_shu.is_file():
    ## Davids data
        data = pd.read_csv(full_data_path, sep='\s+', header=None, skiprows=18, nrows=6,).set_index(0)
        del data.index.name
        data[2] = data[2].str.strip('(,').astype(float)
        data[3] = data[3].str.strip(')').astype(float)
        data.columns=['param', '-error', '+error']
        list_.append(data)
        lens.append(lens_name[i])
        results = pd.concat(list_, keys=lens)
        model_image_path = data_path + '_fresh/F814W/' + lens_name[i] + \
                           '/pipeline_inv_hyper__lens_bulge_disk_sie__source_inversion/' \
                           'pipeline_tag__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/' \
                           'phase_4__lens_bulge_disk_sie__source_inversion/phase_tag__sub_2__pos_1.00/' \
                           'image/lens_fit/fits/fit_model_image_of_plane_1.fits'
        model_image = al.Array.from_fits(file_path=model_image_path, hdu=0,pixel_scales=0.03)
        SLACS_image.append(model_image)
        ## 555 data
        data_shu= pd.read_csv(full_data_path_shu, sep='\s+', header=None, skiprows=18, nrows=6, ).set_index(0)
        del data_shu.index.name
        data_shu[2] = data_shu[2].str.strip('(,').astype(float)
        data_shu[3] = data_shu[3].str.strip(')').astype(float)
        data_shu.columns = ['param', '-error', '+error']
        list_shu.append(data_shu)
        lens_shu.append(lens_name[i])
        results_shu = pd.concat(list_shu, keys=lens_shu)
        model_image_path_shu = data_path + '_shu_shu/F814W/' + lens_name_shu[i] + \
                       '/pipeline_inv_hyper__lens_bulge_disk_sie__source_inversion/' \
                           'pipeline_tag__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/' \
                           'phase_4__lens_bulge_disk_sie__source_inversion/phase_tag__sub_2__pos_1.00/' \
                               'image/lens_fit/fits/fit_model_image_of_plane_1.fits'
        model_image_shu = al.Array.from_fits(file_path=model_image_path_shu, hdu=0, pixel_scales=0.03)
        SLACS_image_shu.append(model_image_shu)
    else:
        slacs = slacs.drop([lens_name[i]], axis=0)

## creating variables for einstin mass results
R_Ein = []
M_Ein = []
R_Ein_shu = []
M_Ein_shu = []
q = []
PA = []
q_shu = []
PA_shu = []
r = []
r_shu = []

## creating galaxy in autolens from autolens mass profile parameters and redshift of lens galaxy
## hi and low correspond to errors on measurement
for i in range(len(lens)):
    ## creating galaxies in autolens from autolens model results 814 data
    lens_galaxy  = al.Galaxy(
        mass=al.mp.EllipticalIsothermal(
            centre=(results.loc[lens[i]]['param']['centre_0'], results.loc[lens[i]]['param']['centre_1']),
            axis_ratio=results.loc[lens[i]]['param']['axis_ratio'], phi=results.loc[lens[i]]['param']['phi'],
            einstein_radius=results.loc[lens[i]]['param']['einstein_radius']), redshift = slacs['z_lens'][i]
    )

    einstein_radius = lens_galaxy.einstein_radius_in_units(unit_length='arcsec')
    einstein_mass = lens_galaxy.einstein_mass_in_units(redshift_object=lens_galaxy.redshift,
                                                       redshift_source=slacs['z_source'][i], cosmology=cosmo,
                                                       unit_mass='solMass'
                                                       )

    ## 555 data
    lens_galaxy_shu = al.Galaxy(
        mass=al.mp.EllipticalIsothermal(
            centre=(results_shu.loc[lens_shu[i]]['param']['centre_0'], results_shu.loc[lens_shu[i]]['param']['centre_1']),
            axis_ratio=results_shu.loc[lens_shu[i]]['param']['axis_ratio'], phi=results_shu.loc[lens_shu[i]]['param']['phi'],
            einstein_radius=results_shu.loc[lens_shu[i]]['param']['einstein_radius']), redshift=slacs['z_lens'][i]
    )


    einstein_radius_shu = lens_galaxy_shu.einstein_radius_in_units(unit_length='arcsec')
    einstein_mass_shu = lens_galaxy_shu.einstein_mass_in_units(redshift_object=lens_galaxy_shu.redshift,
                                                       redshift_source=slacs['z_source'][i], cosmology=cosmo,
                                                       unit_mass='solMass'
                                                       )

    axis_ratio = results.loc[lens[i]]['param']['axis_ratio']
    phi = results.loc[lens[i]]['param']['phi']
    axis_ratio_shu = results_shu.loc[lens[i]]['param']['axis_ratio']
    phi_shu = results_shu.loc[lens[i]]['param']['phi']
    radius = results.loc[lens[i]]['param']['einstein_radius']
    radius_shu = results_shu.loc[lens[i]]['param']['einstein_radius']

    R_Ein.append(einstein_radius)
    M_Ein.append(einstein_mass)
    R_Ein_shu.append(einstein_radius_shu)
    M_Ein_shu.append(einstein_mass_shu)
    r.append(radius)
    r_shu.append(radius_shu)
    q.append(axis_ratio)
    PA.append(phi)
    q_shu.append(axis_ratio_shu)
    PA_shu.append(phi_shu)



## calcualting fractional change for plots
R_Ein_frac = (R_Ein - slacs['b_SIE'])/slacs['b_SIE']
M_Ein_frac = (M_Ein - 10**slacs['log[Me/Mo]'])/10**slacs['log[Me/Mo]']
R_Ein_frac_shu = (np.array(R_Ein_shu) - np.array(R_Ein))/np.array(R_Ein)
M_Ein_frac_shu = (np.array(M_Ein_shu) - np.array(M_Ein))/np.array(M_Ein)
q_diff = np.array(q_shu) - np.array(q)
PA_diff = np.abs(np.array(PA_shu) -np.array(PA))
R_frac_shu = (np.array(r_shu) - np.array(r))/np.array(r)

RMS_M_Ein = np.sqrt(np.mean(np.square(M_Ein_frac_shu)))
RMS_R_Ein = np.sqrt(np.mean(np.square(R_Ein_frac_shu)))
RMS_R = np.sqrt(np.mean(np.square(R_frac_shu)))
RMS_q = np.sqrt(np.mean(np.square(q_diff)))
RMS_PA = np.sqrt(np.mean(np.square(PA_diff)))

error_low_q = []
error_hi_q = []
error_low_phi = []
error_hi_phi = []
error_low_q_shu = []
error_hi_q_shu = []
error_low_phi_shu = []
error_hi_phi_shu = []
error_low_r = []
error_hi_r = []
error_low_r_shu = []
error_hi_r_shu = []

## calculating errors on q slope
for i in range(len(lens)):
    lower_error_q = results.loc[lens[i]]['param']['axis_ratio']-results.loc[lens[i]]['-error']['axis_ratio']
    upper_error_q = results.loc[lens[i]]['+error']['axis_ratio']-results.loc[lens[i]]['param']['axis_ratio']
    error_low_q.append(lower_error_q)
    error_hi_q.append(upper_error_q)

    lower_error_phi = results.loc[lens[i]]['param']['phi'] - results.loc[lens[i]]['-error']['phi']
    upper_error_phi = results.loc[lens[i]]['+error']['phi'] - results.loc[lens[i]]['param']['phi']
    error_low_phi.append(lower_error_phi)
    error_hi_phi.append(upper_error_phi)

    lower_error_q_shu = results_shu.loc[lens[i]]['param']['axis_ratio'] - results_shu.loc[lens[i]]['-error']['axis_ratio']
    upper_error_q_shu = results_shu.loc[lens[i]]['+error']['axis_ratio'] - results_shu.loc[lens[i]]['param']['axis_ratio']
    error_low_q_shu.append(lower_error_q_shu)
    error_hi_q_shu.append(upper_error_q_shu)

    lower_error_phi_shu = results_shu.loc[lens_shu[i]]['param']['phi'] - results_shu.loc[lens_shu[i]]['-error']['phi']
    upper_error_phi_shu = results_shu.loc[lens_shu[i]]['+error']['phi'] - results_shu.loc[lens_shu[i]]['param']['phi']
    error_low_phi_shu.append(lower_error_phi_shu)
    error_hi_phi_shu.append(upper_error_phi_shu)

    lower_error_r = (results.loc[lens[i]]['-error']['einstein_radius'] - results.loc[lens[i]]['param']['einstein_radius'])/results.loc[lens[i]]['param']['einstein_radius']
    upper_error_r = (results.loc[lens[i]]['+error']['einstein_radius'] - results.loc[lens[i]]['param']['einstein_radius'])/results.loc[lens[i]]['param']['einstein_radius']
    error_low_r.append(lower_error_r)
    error_hi_r.append(upper_error_r)

    lower_error_r_shu = np.abs((results_shu.loc[lens_shu[i]]['-error']['einstein_radius'] - results_shu.loc[lens_shu[i]]['param']['einstein_radius'])/results_shu.loc[lens_shu[i]]['param']['einstein_radius'])
    upper_error_r_shu = np.abs((results_shu.loc[lens_shu[i]]['+error']['einstein_radius'] - results_shu.loc[lens_shu[i]]['param']['einstein_radius'])/results_shu.loc[lens_shu[i]]['param']['einstein_radius'])
    error_low_r_shu.append(lower_error_r_shu)
    error_hi_r_shu.append(upper_error_r_shu)

y_err_q = np.array([error_low_q, error_hi_q])
y_err_phi = np.array([error_low_phi, error_hi_phi])
y_err_r = np.array([error_low_r, error_hi_r])
y_err_q_shu = np.array([error_low_q_shu, error_hi_q_shu])
y_err_phi_shu = np.array([error_low_phi_shu, error_hi_phi_shu])
y_err_r_shu = np.array([error_low_r_shu, error_hi_r_shu])


print('formal stat q', np.median(np.array([y_err_q, y_err_q_shu])))
print('formal stat PA', np.median(np.array([y_err_phi, y_err_phi_shu])))
print('formal stat R', np.median(np.array([y_err_r, y_err_r_shu])))

print('RMS M', RMS_M_Ein)
print('RMS R Ein', RMS_R_Ein)
print('RMS R', RMS_R)
print('RMS q', RMS_q)
print('RMS PA', RMS_PA)

fig1, ax = plt.subplots(figsize=(8,8))
for i in range(len(lens)):
    ax.scatter(np.array(M_Ein_shu)[i], np.array(M_Ein)[i],
                       color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
plt.xlabel(r'$M_{Ein}(Shu)$', size=14)
plt.ylabel(r'$M_{Ein} (David)$', size=14)
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
#plt.savefig(fig_path + '_Mass', bbox_inches='tight', dpi=300, transparent=True)

error_low_q = []
error_hi_q = []
error_low_q_shu = []
error_hi_q_shu = []
plt.close()

for i in range(len(lens)):
    lower_error_q = results.loc[lens[i]]['param']['axis_ratio']-results.loc[lens[i]]['-error']['axis_ratio']
    upper_error_q = results.loc[lens[i]]['+error']['axis_ratio']-results.loc[lens[i]]['param']['axis_ratio']
    error_low_q.append(lower_error_q)
    error_hi_q.append(upper_error_q)
    lower_error_q_shu = results_shu.loc[lens[i]]['param']['axis_ratio'] - results_shu.loc[lens[i]]['-error']['axis_ratio']
    upper_error_q_shu = results_shu.loc[lens[i]]['+error']['axis_ratio'] - results_shu.loc[lens[i]]['param']['axis_ratio']
    error_low_q_shu.append(lower_error_q_shu)
    error_hi_q_shu.append(upper_error_q_shu)

x_err_q = np.array([error_low_q, error_hi_q])
y_err_q = np.array([error_low_q_shu, error_hi_q_shu])

fig2, ax = plt.subplots(figsize=(8,8))
for i in range(len(lens)):
    ax.scatter(results_shu.loc[lens[i]]['param']['axis_ratio'], results.loc[lens[i]]['param']['axis_ratio'],
                       color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
    ax.errorbar(results_shu.loc[lens[i]]['param']['axis_ratio'], results.loc[lens[i]]['param']['axis_ratio'],
                yerr=np.array([y_err_q[:, i]]).T, xerr=np.array([x_err_q[:, i]]).T,
                color=slacs['colour'][i], elinewidth=1, fmt='none', capsize=3, label=None)
plt.xlabel(r'$q (Shu)$', size=14)
plt.ylabel(r'$q (David)$', size=14)
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
plt.close()

fig6, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6,5))
for i in range(len(lens)):
    ax1.scatter(q[i], M_Ein_frac_shu[i],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
    ax2.scatter(q[i], q_diff[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
    ax3.scatter(q[i], PA_diff[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
ax1.set_ylabel(r'$\frac{\Delta M_{Ein}}{M_{Ein}}$', size=16)
ax1.axhline(y=0, color='k', linestyle='--')
ax2.set_ylabel(r'$\Delta q$', size=14)
ax2.axhline(y=0, color='k', linestyle='--')
ax3.set_ylabel(r'$|\Delta \phi|$', size=14)
ax3.axhline(y=70.6, color='k', linestyle='--')
ax3.set_xlabel(r'$q^{AutoLens}$', size=14)
#box = fig4.position()
#fig4.position([box.x0, box.y0, box.width * 0.8, box.height])
legend = fig6.legend(loc='center right', title= 'Lens Name')
plt.tight_layout()
plt.subplots_adjust(right=0.7)
plt.savefig(fig_path + 'q_M_PA', bbox_inches='tight', dpi=300)

fig7, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,4))
for i in range(len(lens)):
    ax1.scatter(q[i], R_Ein_frac_shu[i],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
    ax2.scatter(q[i], q_diff[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
ax1.set_ylabel(r'$\frac{\Delta M_{Ein}}{M_{Ein}}$', size=16)
ax1.axhline(y=0, color='k', linestyle='--')
ax2.set_ylabel(r'$\Delta q$', size=14)
ax2.axhline(y=0, color='k', linestyle='--')
ax2.set_xlabel(r'$q$', size=14)
#box = fig4.position()
#fig4.position([box.x0, box.y0, box.width * 0.8, box.height])
legend = fig7.legend(loc='center right', title= 'Lens Name')
plt.tight_layout()
plt.subplots_adjust(right=0.7)
plt.savefig(fig_path + 'q_M', bbox_inches='tight', dpi=300)

fig8, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8,5))
for i in range(len(lens)):
    ax1.scatter(q[i], R_Ein_frac_shu[i],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
    ax2.scatter(q[i],  R_frac_shu[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
    ax3.scatter(q[i], q_diff[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
ax1.set_ylabel(r'$\frac{\Delta R_{Ein}}{R_{Ein}}$', size=16)
ax1.axhline(y=0, color='k', linestyle='--')
ax2.set_ylabel(r'$\frac{\Delta R}{R}$', size=14)
ax2.axhline(y=0, color='k', linestyle='--')
ax3.set_ylabel(r'$\Delta q$', size=14)
ax3.axhline(y=0, color='k', linestyle='--')
ax3.set_xlabel(r'$q$', size=14)
#box = fig4.position()
#fig4.position([box.x0, box.y0, box.width * 0.8, box.height])
legend = fig7.legend(loc='center right', title= 'Lens Name')
plt.tight_layout()
plt.subplots_adjust(right=0.7)
plt.savefig(fig_path + 'q_M_R', bbox_inches='tight', dpi=300)

plt.show()
