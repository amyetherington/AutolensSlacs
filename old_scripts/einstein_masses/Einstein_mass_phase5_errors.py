from autofit import conf

import os

path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

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

data_path = '{}/../output/slacs/slacs_555/'.format(os.path.dirname(os.path.realpath(__file__)))
slacs_path = '{}/../data/slacs_data_table.xlsx'.format(os.path.dirname(os.path.realpath(__file__)))

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
n_params = 10

slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name
slacs = slacs.drop(['slacs0959+5100', 'slacs0959+4416'], axis=0)

image_plane_grid_stack = grids.GridStack.from_shape_pixel_scale_and_sub_size(shape_2d=(301, 301), pixel_scales=0.03,
                                                                      sub_size=2)

for i in range(len(lens_name)):
    full_data_path = Path(data_path + lens_name[i] + '/double_sersic/phase_5_inversion/model.results')
    if full_data_path.is_file():
        data = pd.read_csv(full_data_path, sep='\s+', header=None, skiprows=n_params+4, nrows=n_params,).set_index(0)
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
    lens_galaxy = al.Galaxy(mass=al.mp.EllipticalIsothermal(centre=(results.loc[lens[i]]['param']['lens_mass_centre_0'], results.loc[lens[i]]['param']['lens_mass_centre_1']),
                                                    axis_ratio=results.loc[lens[i]]['param']['lens_mass_axis_ratio'], phi=results.loc[lens[i]]['param']['lens_mass_phi'],
                                                    einstein_radius=results.loc[lens[i]]['param']['lens_mass_einstein_radius']),
                           shear=al.mp.ExternalShear(magnitude=results.loc[lens[i]]['param']['lens_shear_magnitude'], phi=results.loc[lens[i]]['param']['lens_shear_phi']),
                           redshift = slacs['z_lens'][i])

    lens_galaxy_hi = al.Galaxy(mass=al.mp.EllipticalIsothermal(centre=(results.loc[lens[i]]['param']['lens_mass_centre_0'], results.loc[lens[i]]['param']['lens_mass_centre_1']),
                                                        axis_ratio=results.loc[lens[i]]['-error']['lens_mass_axis_ratio'],
                                                        phi=results.loc[lens[i]]['param']['lens_mass_phi'],
                                                        einstein_radius=results.loc[lens[i]]['+error']['lens_mass_einstein_radius']),
                           shear=al.mp.ExternalShear(magnitude=results.loc[lens[i]]['param']['lens_shear_magnitude'],
                                                  phi=results.loc[lens[i]]['param']['lens_shear_phi']),
                           redshift=slacs['z_lens'][i])

    lens_galaxy_low = al.Galaxy(mass=al.mp.EllipticalIsothermal(centre=(results.loc[lens[i]]['param']['lens_mass_centre_0'], results.loc[lens[i]]['param']['lens_mass_centre_1']),
                                                           axis_ratio=results.loc[lens[i]]['+error']['lens_mass_axis_ratio'],
                                                           phi=results.loc[lens[i]]['param']['lens_mass_phi'],
                                                           einstein_radius=results.loc[lens[i]]['-error']['lens_mass_einstein_radius']),
                              shear=al.mp.ExternalShear(magnitude=results.loc[lens[i]]['param']['lens_shear_magnitude'],
                                                     phi=results.loc[lens[i]]['param']['lens_shear_phi']),
                               redshift=slacs['z_lens'][i])

    source_galaxy = al.Galaxy(pixelization=pix.AdaptiveMagnification(shape_2d=(results.loc[lens[i]]['param']['source_pixelization_shape_0'],
                                                                           results.loc[lens[i]]['param']['source_pixelization_shape_1'])),
                             regularization=reg.Constant(results.loc[lens[i]]['param']['source_regularization_coefficients_0']),
                             redshift=slacs['z_source'][i])

    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grid_stack=image_plane_grid_stack, cosmology=cosmology.Planck15)
    tracer_error_hi = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy_hi], source_galaxies=[source_galaxy],
                                             image_plane_grid_stack=image_plane_grid_stack, cosmology=cosmology.Planck15)
    tracer_error_low = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy_low],source_galaxies=[source_galaxy],
                                                          image_plane_grid_stack=image_plane_grid_stack,cosmology=cosmology.Planck15)

    einstein_mass = tracer.physical_einstein_masses_of_planes[0]
    einstein_mass_error_hi = tracer_error_hi.physical_einstein_masses_of_planes[0]
    einstein_mass_error_low = tracer_error_low.physical_einstein_masses_of_planes[0]

    M_Ein.append(einstein_mass)
    M_Ein_error_hi.append(einstein_mass_error_hi)
    M_Ein_error_low.append(einstein_mass_error_low)

lower_error = np.array(np.log10(M_Ein))-np.array(np.log10(M_Ein_error_low))
upper_error = np.array(np.log10(M_Ein_error_hi))-np.array(np.log10(M_Ein))
y_err = np.array([lower_error, upper_error])

print(np.log10(M_Ein))
print(upper_error)

fig, ax = plt.subplots()

cm = plt.get_cmap('gist_rainbow')
ax.set_prop_cycle('color', [cm(1.*i/len(M_Ein)) for i in range(len(M_Ein))])

for i in range(len(M_Ein)):
    ax.scatter(slacs['log[Me/Mo]'][i], np.log10(M_Ein[i]), label=slacs.index[i], marker=slacs['marker'][i])
    ax.errorbar(slacs['log[Me/Mo]'][i], np.log10(M_Ein[i]), yerr=np.array([y_err[:,i]]).T, elinewidth=1, fmt='none', capsize=3, label=None)

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
plt.savefig('/Users/dgmt59/Documents/Results/log_Einstein_mass_phase5_555.png', bbox_inches='tight', dpi=300)
plt.show()
