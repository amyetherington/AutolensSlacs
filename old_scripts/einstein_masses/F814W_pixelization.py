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

from autolens.data.instrument import ccd
from autolens.lens import lens_data as ld
import matplotlib.pyplot as plt
from pathlib import Path

M_o = 1.989e30
cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)

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

inner_mask_radii = np.array(['0.30', '0.30', '0.30', '0.50', '0.20', '0.30',
                             '0.20', '0.50', '0.50', '0.30', '0.30', '0.30', '0.30', '0.40', '0.30'])
## choose data path depending on whether fixed lens light or not
data_path = '{}/../../../../../output/slacs_fix_lens_light/F814W/'.format(os.path.dirname(os.path.realpath(__file__)))
#data_path = '{}/../../../../../output/slacs/F814W/'.format(os.path.dirname(os.path.realpath(__file__)))

slacs_path = '{}/../../data/slacs_data_table.xlsx'.format(os.path.dirname(os.path.realpath(__file__)))
pipeline = '/pipeline_inv_hyper__lens_bulge_disk_sie__source_inversion/'

## choose folder based on fixed lens light or not
no_shear = 'pipeline_tag__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre/phase_2__lens_bulge_disk_sie__source_inversion_magnification/phase_tag__sub_2__pos_1.00__cluster_0.100/model.results'
#no_shear = 'pipeline_tag__pix_voro_image__reg_adapt_bright__bd_align_centre/phase_2__lens_bulge_disk_sie__source_inversion_magnification/phase_tag__sub_2__pos_1.00__cluster_0.100/model.results'

## choose based on fix lens light or not
skiprows = 18    #slacs fix lens light
#skiprows = 46     #slacs

pixel_scales = 0.03
new_shape = (301, 301)
list_ = []
lens=[]
M_Ein = []
R_Ein = []
norm_M_Ein = []
M_Ein_test = []
M_Ein_test_Kormann = []
M_Ein_error_hi = []
M_Ein_error_low = []




slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name
slacs = slacs.drop(['slacs0959+5100',  'slacs0959+4416'], axis=0)

#image_plane_grid_stack = grids.GridStack.from_shape_pixel_scale_and_sub_size(shape_2d=(301, 301), pixel_scales=0.03,
 #                                                                     sub_size=2)



for i in range(len(lens_name)):
    full_data_path = Path(data_path + lens_name[i] + pipeline + no_shear)
    if full_data_path.is_file():
        data = pd.read_csv(full_data_path, sep='\s+', header=None, skiprows=skiprows, nrows=6,).set_index(0)
        del data.index.name
        data[2] = data[2].str.strip('(,').astype(float)
        data[3] = data[3].str.strip(')').astype(float)
        data.columns=['param', '-error', '+error']
        list_.append(data)
        lens.append(lens_name[i])
        results = pd.concat(list_, keys=lens)
    else:
        slacs = slacs.drop([lens_name[i]], axis=0)


for i in range(len(lens)):
    lens_galaxy = al.Galaxy(mass=al.mp.EllipticalIsothermal(centre=(results.loc[lens[i]]['param']['centre_0'], results.loc[lens[i]]['param']['centre_1']),
                                                    axis_ratio=results.loc[lens[i]]['param']['axis_ratio'], phi=results.loc[lens[i]]['param']['phi'],
                                                    einstein_radius=results.loc[lens[i]]['param']['einstein_radius']), redshift = slacs['z_lens'][i])

    lens_galaxy_hi = al.Galaxy(mass=al.mp.EllipticalIsothermal(centre=(results.loc[lens[i]]['param']['centre_0'], results.loc[lens[i]]['param']['centre_1']),
                                                        axis_ratio=results.loc[lens[i]]['+error']['axis_ratio'],
                                                        phi=results.loc[lens[i]]['param']['phi']),
                                                        einstein_radius=results.loc[lens[i]]['+error']['einstein_radius'], redshift=slacs['z_lens'][i])

    lens_galaxy_low = al.Galaxy(mass=al.mp.EllipticalIsothermal(centre=(results.loc[lens[i]]['param']['centre_0'], results.loc[lens[i]]['param']['centre_1']),
                                                           axis_ratio=results.loc[lens[i]]['-error']['axis_ratio'],
                                                           phi=results.loc[lens[i]]['param']['phi']),
                                                           einstein_radius=results.loc[lens[i]]['-error']['einstein_radius'], redshift=slacs['z_lens'][i])

    lens_galaxy_test = al.Galaxy(mass=al.mp.EllipticalIsothermal(axis_ratio=slacs['q_SIE'][i], phi=slacs['PA'][i], einstein_radius=slacs['b_SIE'][i]), redshift=slacs['z_lens'][i])

    lens_galaxy_test_Kormann = al.Galaxy(mass=al.mp.EllipticalIsothermalKormann(axis_ratio=slacs['q_SIE'][i], phi=slacs['PA'][i],
                                                                            einstein_radius=slacs['b_SIE'][i]), redshift=slacs['z_lens'][i])


    einstein_mass = lens_galaxy.einstein_mass_in_units(unit_mass='solMass', redshift_source=slacs['z_source'][i],
                                                       cosmology=cosmo)
    einstein_mass_error_hi = lens_galaxy_hi.einstein_mass_in_units(unit_mass='solMass', redshift_source=slacs['z_source'][i],
                                                       cosmology=cosmo)
    einstein_mass_error_low = lens_galaxy_low.einstein_mass_in_units(unit_mass='solMass', redshift_source=slacs['z_source'][i],
                                                       cosmology=cosmo)

    einstein_mass_test = lens_galaxy_test.einstein_mass_in_units(unit_mass='solMass', redshift_source=slacs['z_source'][i],
                                                                 cosmology=cosmo)
    einstein_mass_test_Kormann = lens_galaxy_test_Kormann.einstein_mass_in_units(unit_mass='solMass',
                                                                 redshift_source=slacs['z_source'][i],
                                                                 cosmology=cosmo)
    einstein_radius_test = lens_galaxy.einstein_radius_in_units()
    einstein_mass_rescaled = (einstein_mass_test_Kormann) * ((1+slacs['q_SIE'][i])/2)

    M_Ein_test.append(einstein_mass_test)
    M_Ein_test_Kormann.append(einstein_mass_test_Kormann)
    M_Ein.append(einstein_mass)
    M_Ein_error_hi.append(einstein_mass_error_hi)
    M_Ein_error_low.append(einstein_mass_error_low)
    norm_M_Ein.append(einstein_mass_rescaled)
    R_Ein.append(einstein_radius_test)

fractional_change = (np.log10(norm_M_Ein) - slacs['log[Me/Mo]'])/slacs['log[Me/Mo]']
fractional_change_Kormann = (np.log10(M_Ein_test_Kormann) - slacs['log[Me/Mo]'])/slacs['log[Me/Mo]']
fractional_change_radii = (R_Ein - slacs['b_SIE'])/slacs['b_SIE']



lower_error = np.array(np.log10(M_Ein))-np.array(np.log10(M_Ein_error_low))
upper_error = np.array(np.log10(M_Ein_error_hi))-np.array(np.log10(M_Ein))
y_err = np.array([lower_error, upper_error])


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
#plt.close()

fig2, ax = plt.subplots()

for i in range(len(M_Ein)):
    ax.scatter(slacs['b_SIE'][i], results.loc[lens[i]]['param']['einstein_radius'],
               color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
   # ax.errorbar(slacs['R_Ein'][i], results.loc[lens[i]]['param']['lens_galaxies_lens_mass_einstein_radius_value'], yerr=np.array([y_err[:,i]]).T, elinewidth=1, fmt='none', capsize=3, label=None)

ax.legend()

plt.xlabel(r'b(SIE)$_{SLACS}$', size=14)
plt.ylabel(r'b(SIE)$_{PyAutoLens}$', size=14)


lims = [
    np.min([0.9, 2]),  # min of both axes
    np.max([0.9, 2]),  # max of both axes
]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.close()

fig3, ax = plt.subplots()

for i in range(len(M_Ein)):
    ax.scatter(slacs['q_SIE'][i], results.loc[lens[i]]['param']['axis_ratio'],
               color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
   # ax.errorbar(slacs['R_Ein'][i], results.loc[lens[i]]['param']['lens_galaxies_lens_mass_einstein_radius_value'], yerr=np.array([y_err[:,i]]).T, elinewidth=1, fmt='none', capsize=3, label=None)

ax.legend()

plt.xlabel(r'q$_{SLACS}$', size=14)
plt.ylabel(r'q$_{PyAutoLens}$', size=14)


lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.close()

#plt.savefig('/Users/dgmt59/Documents/Plots/Einstein_mass_phase3_F814W.png', bbox_inches='tight', dpi=300)

fig4, ax = plt.subplots()

for i in range(len(M_Ein)):
    ax.scatter(slacs['log[Me/Mo]'][i], np.log10(M_Ein_test[i]), label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i],)
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
plt.close()

fig5, ax = plt.subplots()

for i in range(len(M_Ein)):
    ax.scatter(slacs['q_SIE'][i], fractional_change[i], label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i])
ax.legend()
plt.axhline(y=0, color='k',)
plt.axhline(y=0.002, color='k', linestyle='--')
plt.axhline(y=-0.002, color='k', linestyle='--')
plt.xlabel('axis ratio', size=14)
plt.ylabel(r'$\frac{M_{E_{autolens}}-M_{E_{SLACS}}}{M_{E_{SLACS}}}$', size=14)
plt.close()


fig6, ax = plt.subplots()

for i in range(len(M_Ein)):
    ax.scatter(slacs['q_SIE'][i], fractional_change_Kormann[i], label=slacs.index[i], marker=slacs['marker'][i], color=slacs['colour'][i])
ax.legend()
plt.axhline(y=0, color='k',)
plt.axhline(y=0.002, color='k', linestyle='--')
plt.axhline(y=-0.002, color='k', linestyle='--')
plt.xlabel('axis ratio', size=14)
plt.ylabel(r'$\frac{M_{E_{autolens Kormann}}-M_{E_{SLACS}}}{M_{E_{SLACS}}}$', size=14)
plt.close()

fig7, ax = plt.subplots()

for i in range(len(R_Ein)):
    ax.scatter(slacs['b_SIE'][i], R_Ein[i],
               color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
   # ax.errorbar(slacs['R_Ein'][i], results.loc[lens[i]]['param']['lens_galaxies_lens_mass_einstein_radius_value'], yerr=np.array([y_err[:,i]]).T, elinewidth=1, fmt='none', capsize=3, label=None)

ax.legend()

plt.xlabel(r'b$_{SLACS}$', size=14)
plt.ylabel(r'b$_{PyAutoLens}$', size=14)


lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
#plt.close()

plt.show()
