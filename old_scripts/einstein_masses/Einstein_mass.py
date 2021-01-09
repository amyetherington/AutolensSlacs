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

M_o = 1.989e30

data_path = '{}/../output/slacs_cosma/'.format(os.path.dirname(os.path.realpath(__file__)))
slacs_path = '{}/../data/slacs_data_table.xlsx'.format(os.path.dirname(os.path.realpath(__file__)))

#lens_name = np.array(['slacs0216-0813', 'slacs0252+0039', 'slacs0737+3216', 'slacs0912+0029', 'slacs0959+4410',
#                      'slacs0959+4416', 'slacs1011+0143', 'slacs1011+0143', 'slacs1205+4910', 'slacs1250+0523',
#                      'slacs1402+6321', 'slacs1420+6019', 'slacs1430+4105', 'slacs1627+0053', 'slacs1630+4520',
#                      'slacs2238-0754', 'slacs2300+0022', 'slacs2303+1422'])

lens_name = 'slacs1430+4105'

pixel_scales = 0.03
new_shape = (301, 301)

ccd_data = ccd.load_ccd_data_from_fits(image_path=path + '/data/slacs/' + lens_name + '/F814W_image.fits',
                                       psf_path=path+'/data/slacs/'+lens_name+'/F814W_psf.fits',
                                       noise_map_path=path+'/data/slacs/'+lens_name+'/F814W_noise_map.fits',
                                       pixel_scales=pixel_scales,
                                       resized_ccd_shape=new_shape, resized_psf_shape=(15, 15))
mask = msk.load_mask_from_fits(mask_path=path + '/data/slacs/' + lens_name + '/mask.fits', pixel_scales=pixel_scales)
mask = mask.resized_scaled_array_from_array(new_shape=new_shape)

lens_data = ld.LensData(ccd_data=ccd_data, mask=mask)

list_ = []
n_params = 17

slacs = pd.read_excel(slacs_path).set_index('lens_name')
slacs = slacs.drop(slacs.index[5])

#for i in range(len(lens_name)):

full_data_path = data_path + lens_name + '/pipeline_light_and_source_inversion/phase_5_inversion_low_position_thresh/model.results'
data = pd.read_csv(full_data_path, sep='\s+', header=None, skiprows=[0], nrows=n_params).set_index(0)
del data.index.name
data = data.T.rename(index={1:lens_name})
#    list_.append(data)

#    results = pd.concat(list_)

image_plane_grid_stack = grids.GridStack.from_shape_pixel_scale_and_sub_size(shape_2d=ccd_data.shape,
                                                                                  pixel_scales=ccd_data.pixel_scales,
                                                                                  sub_size=2)

#print(data)
#print(data.iloc[0,16])

lens_galaxy = al.Galaxy(mass=al.mp.EllipticalIsothermal(centre=(data.iloc[0,5], data.iloc[0,6]), axis_ratio=data.iloc[0,7],
                                                    phi=data.iloc[0,8], einstein_radius=data.iloc[0,9]), redshift=0.285)
source_galaxy = al.Galaxy(pixelization=pix.AdaptiveMagnification(shape_2d=(data.iloc[0,14], data.iloc[0,15])),
                         regularization=reg.Constant(data.iloc[0,16]), redshift=0.575)

#image_plane_grid_stack = grids.GridStack.from_shape_pixel_scale_and_sub_size(shape_2d=(301, 301), pixel_scales=0.03,
#                                                                      sub_size=2)

tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                             image_plane_grid_stack=lens_data.grid_stack, cosmology=cosmology.Planck15)

print(np.log10(tracer.einstein_masses_of_planes[0]))
