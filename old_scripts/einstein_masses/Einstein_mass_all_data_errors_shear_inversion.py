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
from autolens.data.array import grids
from autolens.data.instrument import ccd
from autolens.lens import lens_data as ld
from autolens.array import mask as msk
import matplotlib.pyplot as plt
from pathlib import Path

M_o = 1.989e30
cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)

data_path = '{}/../../../../../output/slacs/F814W/'.format(os.path.dirname(os.path.realpath(__file__)))
slacs_path = '{}/../../data/slacs_data_table.xlsx'.format(os.path.dirname(os.path.realpath(__file__)))
shear = '/pipeline_inv__lens_sersic_exp_sie_source_inversion_with_shear_fix_lens_light_pix_voro_mag_reg_const_bd_align_centre/'
phase_4 = 'phase_2_lens_sersic_exp_sie_source_inversion/settings_sub_2_pos_1.00_inner_mask_0.30_cluster_0.100/model.results'

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

pixel_scales = 0.03
new_shape = (301, 301)
list_ = []
lens=[]
M_Ein = []
M_Ein_error_hi = []
M_Ein_error_low = []
n_params = 27

slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name
slacs = slacs.drop(['slacs0959+5100',  'slacs0959+4416'], axis=0)

image_plane_grid_stack = grids.GridStack.from_shape_pixel_scale_and_sub_size(shape_2d=(301, 301), pixel_scales=0.03,
                                                                      sub_size=2)

for i in range(len(lens_name)):
    full_data_path = Path(data_path + lens_name[i] + shear + phase_4)
    if full_data_path.is_file():
        data = pd.read_csv(full_data_path, sep='\s+', header=None, skiprows=21, nrows=9,).set_index(0)
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
                                                    axis_ratio=results.loc[lens[i]]['param']['axis_ratio'], phi=results.loc[lens[i]]['param'][1],
                                                    einstein_radius=results.loc[lens[i]]['param']['einstein_radius']),
                           shear=al.mp.ExternalShear(magnitude=results.loc[lens[i]]['param']['magnitude'], phi=results.loc[lens[i]]['param'][8]),
                           redshift = slacs['z_lens'][i])

    lens_galaxy_hi = al.Galaxy(mass=al.mp.EllipticalIsothermal(centre=(results.loc[lens[i]]['param']['centre_0'], results.loc[lens[i]]['param']['centre_1']),
                                                        axis_ratio=results.loc[lens[i]]['-error']['axis_ratio'],
                                                        phi=results.loc[lens[i]]['param'][1],
                                                        einstein_radius=results.loc[lens[i]]['+error']['einstein_radius']),
                           shear=al.mp.ExternalShear(magnitude=results.loc[lens[i]]['param']['magnitude'],
                                                  phi=results.loc[lens[i]]['param'][8]), redshift=slacs['z_lens'][i])

    lens_galaxy_low = al.Galaxy(mass=al.mp.EllipticalIsothermal(centre=(results.loc[lens[i]]['param']['centre_0'], results.loc[lens[i]]['param']['centre_1']),
                                                           axis_ratio=results.loc[lens[i]]['+error']['axis_ratio'],
                                                           phi=results.loc[lens[i]]['param'][1],
                                                           einstein_radius=results.loc[lens[i]]['-error']['einstein_radius']),
                              shear=al.mp.ExternalShear(magnitude=results.loc[lens[i]]['param']['magnitude'],
                                                     phi=results.loc[lens[i]]['param'][8]), redshift=slacs['z_lens'][i])

    einstein_mass = lens_galaxy.einstein_mass_in_units(unit_mass='solMass', redshift_source=slacs['z_source'][i],
                                                       cosmology=cosmo)
    einstein_mass_error_hi = lens_galaxy_hi.einstein_mass_in_units(unit_mass='solMass', redshift_source=slacs['z_source'][i],
                                                       cosmology=cosmo)
    einstein_mass_error_low = lens_galaxy_low.einstein_mass_in_units(unit_mass='solMass', redshift_source=slacs['z_source'][i],
                                                       cosmology=cosmo)

    M_Ein.append(einstein_mass)
    M_Ein_error_hi.append(einstein_mass_error_hi)
    M_Ein_error_low.append(einstein_mass_error_low)

lower_error = np.array(np.log10(M_Ein))-np.array(np.log10(M_Ein_error_low))
upper_error = np.array(np.log10(M_Ein_error_hi))-np.array(np.log10(M_Ein))
y_err = np.array([lower_error, upper_error])

fig, ax = plt.subplots()

for i in range(len(M_Ein)):
    ax.scatter(slacs['log[Me/Mo]'][i], np.log10(M_Ein[i]), label=slacs.index[i],
               marker=slacs['marker'][i], color=slacs['colour'][i])
    ax.errorbar(slacs['log[Me/Mo]'][i], np.log10(M_Ein[i]), yerr=np.array([y_err[:,i]]).T,
                color= slacs['colour'][i], elinewidth=1, fmt='none', capsize=3, label=None)

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

fig2, ax = plt.subplots()

for i in range(len(M_Ein)):
    ax.scatter(slacs['b_SIE'][i], results.loc[lens[i]]['param']['einstein_radius'],
                  color = slacs['colour'][i],label=slacs.index[i], marker=slacs['marker'][i])
   # ax.errorbar(slacs['R_Ein'][i], results.loc[lens[i]]['param']['einstein_radius'], yerr=np.array([y_err[:,i]]).T, elinewidth=1, fmt='none', capsize=3, label=None)

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

fig3, ax = plt.subplots()

for i in range(len(M_Ein)):
    ax.scatter(slacs['q_SIE'][i], results.loc[lens[i]]['param']['axis_ratio'], color=slacs['colour'][i],
                    label=slacs.index[i], marker=slacs['marker'][i])
   # ax.errorbar(slacs['R_Ein'][i], results.loc[lens[i]]['param']['einstein_radius'], yerr=np.array([y_err[:,i]]).T, elinewidth=1, fmt='none', capsize=3, label=None)

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

plt.show()

