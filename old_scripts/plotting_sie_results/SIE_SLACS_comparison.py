from autofit import conf
import autolens as al
import autoastro as astro

import os

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))

conf.instance = conf.Config(config_path=path+'config', output_path=path+'output')

import pandas as pd
import numpy as np
from astropy import cosmology
from autoastro.util import cosmology_util


import matplotlib.pyplot as plt
from pathlib import Path

M_o = 1.989e30
cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)

## setting up paths to shear and no shear results (sersic source)
data_path = '{}/../../../../output/slacs_shu_shu/'.format(os.path.dirname(os.path.realpath(__file__)))
slacs_path = '{}/../../dataset/slacs_data_table.xlsx'.format(os.path.dirname(os.path.realpath(__file__)))
wavelength = 'F814W'
pipeline = '/pipeline_inv_hyper__lens_bulge_disk_sie__source_inversion/' \
           'pipeline_tag__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/' \
           'phase_4__lens_bulge_disk_sie__source_inversion/phase_tag__sub_2__pos_1.00/model.results'
fig_path = '/Users/dgmt59/Documents/Plots/error_analysis/SIE/shu_v_SLACS_'

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

## creating variables for loading results
list_ = []
lens = []
SLACS_image = []

## loading table of slacs parameters (deleting two discarded lenses)
slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name
slacs = slacs.drop(['slacs0956+5100',  'slacs0959+4416'], axis=0)

## loading results into pandas data frames
## deleting any lenses from slacs table that don't have autolens results files
for i in range(len(lens_name)):
    full_data_path = Path(data_path + wavelength + '/' + lens_name[i] + pipeline)
    if full_data_path.is_file():
        data = pd.read_csv(full_data_path, sep='\s+', header=None, skiprows=18, nrows=6,).set_index(0)
        del data.index.name
        data[2] = data[2].str.strip('(,').astype(float)
        data[3] = data[3].str.strip(')').astype(float)
        data.columns=['param', '-error', '+error']
        list_.append(data)
        lens.append(lens_name[i])
        results = pd.concat(list_, keys=lens)
        model_image_path = data_path + wavelength + '/' + lens_name[i] + \
                           '/pipeline_inv_hyper__lens_bulge_disk_sie__source_inversion/' \
                           'pipeline_tag__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/' \
                           'phase_4__lens_bulge_disk_sie__source_inversion/phase_tag__sub_2__pos_1.00/' \
                           'image/lens_fit/fits/fit_model_image_of_plane_1.fits'
        model_image = al.Array.from_fits(file_path=model_image_path, hdu=0,pixel_scales=0.03)
        SLACS_image.append(model_image)
    else:
        slacs = slacs.drop([lens_name[i]], axis=0)

## creating variables for einstin mass results
R_Ein = []
M_Ein = []
M_Ein_SLACS = []
q = []
PA = []
R_rescaled = []
R_err=[]

## creating galaxy in autolens from autolens mass profile parameters and redshift of lens galaxy
## hi and low correspond to errors on measurement
for i in range(len(lens)):
    ## creating galaxies in autolens from autolens model results
    lens_galaxy  = al.Galaxy(
        mass=al.mp.EllipticalIsothermal(
            centre=(results.loc[lens[i]]['param']['centre_0'], results.loc[lens[i]]['param']['centre_1']),
            axis_ratio=results.loc[lens[i]]['param']['axis_ratio'], phi=results.loc[lens[i]]['param']['phi'],
            einstein_radius=results.loc[lens[i]]['param']['einstein_radius']), redshift = slacs['z_lens'][i]
    )

    einstein_radius = lens_galaxy.einstein_radius_in_units(unit_length='arcsec')
    einstein_mass = lens_galaxy.einstein_mass_in_units(redshift_object=lens_galaxy.redshift,
        redshift_source=slacs['z_source'][i], cosmology=cosmo, unit_mass='solMass'
    )

    sigma_crit = cosmology_util.critical_surface_density_between_redshifts_from_redshifts_and_cosmology(
        redshift_0=slacs['z_lens'][i], redshift_1=slacs['z_source'][i], cosmology=cosmo, unit_mass='solMass'
    )


    einstein_mass_SLACS = np.pi*sigma_crit* slacs['b_SIE'][i]**2
    axis_ratio = results.loc[lens[i]]['param']['axis_ratio']
    phi = results.loc[lens[i]]['param']['phi']

    radius_rescaled = results.loc[lens[i]]['param']['einstein_radius'] * (
                2 * np.sqrt(axis_ratio) / (1 + axis_ratio))


    r_error = np.array([(np.abs((results.loc[lens[i]]['+error']['einstein_radius']-results.loc[lens[i]]['param']['einstein_radius'])
                               /results.loc[lens[i]]['param']['einstein_radius'])),
              (np.abs(results.loc[lens[i]]['-error']['einstein_radius'] - results.loc[lens[i]]['param']['einstein_radius']))
                       / results.loc[lens[i]]['param']['einstein_radius']])

    R_rescaled.append(radius_rescaled)
    R_Ein.append(einstein_radius)
    M_Ein.append(einstein_mass)
    M_Ein_SLACS.append(einstein_mass_SLACS)
    q.append(axis_ratio)
    PA.append(phi)
    R_err.append(r_error)


print('formal stat R', np.mean(R_err))

## calcualting fractional change for plots
R_Ein_frac = (R_Ein - slacs['b_SIE'])/slacs['b_SIE']
M_Ein_frac = (np.array(M_Ein) - np.array(M_Ein_SLACS))/(np.array(M_Ein_SLACS))
q_diff = np.array(q) - slacs['q_SIE']
PA_diff = np.abs(np.array(PA) - slacs['PA'])
R_frac = (np.array(R_Ein)-np.array(R_rescaled))/R_rescaled

RMS_M_Ein = np.sqrt(np.mean(np.square(M_Ein_frac)))
RMS_R_Ein = np.sqrt(np.mean(np.square(R_Ein_frac)))
RMS_q = np.sqrt(np.mean(np.square(np.array(q)-slacs['q_SIE'])))
RMS_PA = np.sqrt(np.mean(np.square(PA-slacs['PA'])))



print('RMS M', RMS_M_Ein)
print('RMS R', RMS_R_Ein)
print('RMS q', RMS_q)
print('RMS PA', RMS_PA)

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
#plt.savefig(fig_path + '_R_Ein_as_function_of_q', bbox_inches='tight', dpi=300)
plt.close()

fig2, ax = plt.subplots(figsize=(10,5))
for i in range(len(lens)):
    ax.scatter(results.loc[lens[i]]['param']['axis_ratio'], M_Ein_frac[i], label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title= 'Lens Name')
plt.axhline(y=0, color='k',)
plt.axhline(y=0.05, color='k', linestyle='--')
plt.axhline(y=-0.05, color='k', linestyle='--')
plt.xlabel('Axis ratio', size=20)
plt.ylabel(r'$\frac{M_{Ein_{autolens}}-M_{Ein_{SLACS}}}{M_{Ein_{SLACS}}}$', size=21)
#plt.savefig(fig_path +'_M_Ein_as_function_of_q', bbox_inches='tight', dpi=300)
plt.close()

error_low_q = []
error_hi_q = []
error_low_phi = []
error_hi_phi = []

## calculating errors on q
for i in range(len(lens)):
    lower_error_q = results.loc[lens[i]]['param']['axis_ratio']-results.loc[lens[i]]['-error']['axis_ratio']
    upper_error_q = results.loc[lens[i]]['+error']['axis_ratio']-results.loc[lens[i]]['param']['axis_ratio']
    error_low_q.append(lower_error_q)
    error_hi_q.append(upper_error_q)

    lower_error_phi = results.loc[lens[i]]['param']['phi'] - results.loc[lens[i]]['-error']['phi']
    upper_error_phi = results.loc[lens[i]]['+error']['phi'] - results.loc[lens[i]]['param']['phi']
    error_low_phi.append(lower_error_phi)
    error_hi_phi.append(upper_error_phi)

y_err_q = np.array([error_low_q, error_hi_q])
y_err_phi = np.array([error_low_phi, error_hi_phi])

print('formal stat q', np.mean(y_err_q))
print('formal stat PA', np.mean(y_err_phi))

fig3, ax = plt.subplots(figsize=(5,5))
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
#plt.savefig(fig_path + 'axis_ratio', bbox_inches='tight', dpi=300)
plt.close()

fig3, ax = plt.subplots(figsize=(5,5))
for i in range(len(lens)):
    ax.scatter(slacs['PA'][i], results.loc[lens[i]]['param']['phi'],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
    ax.errorbar(slacs['PA'][i], results.loc[lens[i]]['param']['phi'], yerr=np.array([y_err_phi[:,i]]).T,
               color=slacs['colour'][i], elinewidth=1, fmt='none', capsize=3, label=None)
plt.xlabel(r'$\phi_{SLACS}$', size=14)
plt.ylabel(r'$\phi_{AutoLens}$', size=14)
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
#plt.savefig(fig_path + 'phi', bbox_inches='tight', dpi=300)
plt.close()


fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
for i in range(len(lens)):
    ax1.scatter(M_Ein[i], M_Ein_SLACS[i],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
    ax2.scatter(slacs['q_SIE'][i], results.loc[lens[i]]['param']['axis_ratio'], color=slacs['colour'][i],
                marker=slacs['marker'][i])
    ax2.errorbar(slacs['q_SIE'][i], results.loc[lens[i]]['param']['axis_ratio'], yerr=np.array([y_err_q[:,i]]).T,
               color=slacs['colour'][i], elinewidth=1, fmt='none', capsize=3, label=None)
ax1.set_xlabel(r'$M_{Ein}^{SLACS}$', size=14)
ax1.set_ylabel(r'$M_{Ein}^{AutoLens}$', size=14)
ax2.set_xlabel(r'$q^{SLACS}$', size=14)
ax2.set_ylabel(r'$q^{AutoLens}$', size=14)
#box = fig4.position()
#fig4.position([box.x0, box.y0, box.width * 0.8, box.height])
legend = fig4.legend(loc='center right', title= 'Lens Name')
lims1 = [
    np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
    np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
     ]
ax1.plot(lims1, lims1, 'k--', alpha=0.75, zorder=0)
ax1.set_aspect('equal')
ax1.set_xlim(lims1)
ax1.set_ylim(lims1)
lims2 = [
    np.min([ax2.get_xlim(), ax2.get_ylim()]),  # min of both axes
    np.max([ax2.get_xlim(), ax2.get_ylim()]),  # max of both axes
     ]
ax2.plot(lims2, lims2, 'k--', alpha=0.75, zorder=0)
ax2.set_aspect('equal')
ax2.set_xlim(lims2)
ax2.set_ylim(lims2)
plt.tight_layout()
plt.subplots_adjust(right=0.8)
#plt.savefig(fig_path + 'q_and_M_SLACS_comparison', bbox_inches='tight', dpi=300)
plt.close()

fig5, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,5))
for i in range(len(lens)):
    ax1.scatter(R_Ein[i], M_Ein_frac[i],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
    ax2.scatter(R_Ein[i], q_diff[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
ax1.set_ylabel(r'$\frac{M_{Ein}^{AutoLens}-M_{Ein}^{SLACS}}{M_{Ein}^{SLACS}}$', size=14)
ax1.axhline(y=0, color='k', linestyle='--')
ax2.set_xlabel(r'$R_{Ein}^{AutoLens}$', size=14)
ax2.set_ylabel(r'$\frac{q^{AutoLens}-q^{SLACS}}{q^{SLACS}}$', size=14)
ax2.axhline(y=0, color='k', linestyle='--')
#box = fig4.position()
#fig4.position([box.x0, box.y0, box.width * 0.8, box.height])
legend = fig5.legend(loc='center right', title= 'Lens Name')
plt.tight_layout()
plt.subplots_adjust(right=0.7)
#plt.savefig(fig_path + 'R_ein_q_and_M_frac_SLACS_comparison', bbox_inches='tight', dpi=300)
plt.close()

fig6, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8,5))
for i in range(len(lens)):
    ax1.scatter(q[i], M_Ein_frac[i],
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
ax3.axhline(y=3.8, color='k', linestyle='--')
ax3.set_xlabel(r'$q^{AutoLens}$', size=14)
legend = fig6.legend(loc='center right', title= 'Lens Name')
plt.tight_layout()
plt.subplots_adjust(right=0.7)
plt.savefig(fig_path + 'q_M_PA', bbox_inches='tight', dpi=300, transparent=True)

fig7, ax = plt.subplots(figsize=(10,5))
for i in range(len(lens)):
    ax.scatter(results.loc[lens[i]]['param']['axis_ratio'], R_frac[i], label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title= 'Lens Name')
plt.axhline(y=0, color='k',)
plt.axhline(y=0.02, color='k', linestyle='--')
plt.axhline(y=-0.02, color='k', linestyle='--')
plt.xlabel('Axis ratio', size=20)
plt.ylabel(r'$\frac{R_{Ein_{autolens}}-R_{Ein_{SLACS}}}{R_{Ein_{SLACS}}}$', size=21)
#plt.savefig(fig_path + '_R_rescaled_frac_as_function_of_q', bbox_inches='tight', dpi=300)
plt.close()



plt.show()
