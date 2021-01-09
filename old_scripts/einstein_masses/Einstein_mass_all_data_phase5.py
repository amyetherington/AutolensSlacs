from autofit import conf

import os

# Get the relative path to the config files and output folder in our workspace.
path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path and output path.
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

M_o = 1.989e30

data_path = '{}/../output/slacs_cosma_double_sersic_const_off/'.format(os.path.dirname(os.path.realpath(__file__)))
slacs_path = '{}/../data/slacs_data_table.xlsx'.format(os.path.dirname(os.path.realpath(__file__)))

lens_name = np.array(['slacs0216-0813',
                      'slacs0252+0039',
                      'slacs0737+3216',
                      'slacs0912+0029',
                      #'slacs0959+4410',
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
n_params = 10
M_Ein = []

slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name
slacs = slacs.drop(['slacs0959+5100',  'slacs0959+4416','slacs0959+4410'], axis=0)

for i in range(len(lens_name)):
    full_data_path = data_path + lens_name[i] + '/pipeline_light_and_source_inversion/phase_5_inversion/model.results'
    data = pd.read_csv(full_data_path, sep='\s+', header=None, skiprows=[0], nrows=n_params).set_index(0)
    del data.index.name
    data = data.T.rename(index={1:lens_name[i]})

    list_.append(data)
    results = pd.concat(list_)


image_plane_grid_stack = grids.GridStack.from_shape_pixel_scale_and_sub_size(shape_2d=(301, 301), pixel_scales=0.03,
                                                                      sub_size=2)

for i in range(len(results)):
    lens_galaxy = al.Galaxy(mass=al.mp.EllipticalIsothermal(centre=(results['lens_mass_centre_0'][i], results['lens_mass_centre_1'][i]),
                                                    axis_ratio=results['lens_mass_axis_ratio'][i], phi=results['lens_mass_phi'][i],
                                                    einstein_radius=results['lens_mass_einstein_radius'][i]),
                       shear=al.mp.ExternalShear(magnitude=results['lens_shear_magnitude'][i], phi=results['lens_shear_phi'][i]),
                        redshift = slacs['z_lens'][i])

    source_galaxy = al.Galaxy(pixelization=pix.AdaptiveMagnification(shape_2d=(results['source_pixelization_shape_0'][i],
                                                                       results['source_pixelization_shape_1'][i])),
                         regularization=reg.Constant(results['source_regularization_coefficients_0'][i]),
                         redshift=slacs['z_source'][i])

    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grid_stack=image_plane_grid_stack, cosmology=cosmology.Planck15)
    einstein_mass = tracer.einstein_masses_of_planes[0]

    M_Ein.append(einstein_mass)

fig, ax = plt.subplots()

cm = plt.get_cmap('gist_rainbow')
ax.set_color_cycle([cm(1.*i/len(slacs)) for i in range(len(slacs))])
marker = ['o', '+','v','<','s','p','*','D','h','x','8','1','2','3','4',]

for i in range(len(results)):
    ax.scatter(slacs['log[Me/Mo]'][i], np.log10(M_Ein[i]), label=slacs.index[i], marker=marker[i])

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
plt.savefig('/Users/dgmt59/Documents/Results/log_Einstein_mass_phase5.png', bbox_inches='tight', dpi=300)
plt.show()
