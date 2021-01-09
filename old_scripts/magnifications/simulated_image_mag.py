from autofit import conf

import os

# Get the relative path to the config files and output folder in our workspace.
path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path and output path.
conf.instance = conf.Config(config_path=path + 'config', output_path=path + 'output')


from autofit.tools import path_util
from astropy.io import fits
from autolens.data.instrument import ccd
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp

workspace_path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))

ccd_data_path = path_util.make_and_return_path_from_path_and_folder_names(path=workspace_path,
                                                                      folder_names=['data', 'scripts', 'magnifications'])


def simulate_ccd_data():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lens import ray_tracing

    psf = ccd.PSF.simulate_as_gaussian(shape_2d=(21, 21), sigma=0.05, pixel_scales=0.1)

    image_plane_grid_stack = grids.GridStack.grid_stack_for_simulation(shape_2d=(100, 100), pixel_scales=0.1,
                                                                       psf_shape=(21, 21))

    lens_galaxy = al.Galaxy(light=al.lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0,
                                                    sersic_index=2.0),
                           mass=al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.2))

    source_galaxy = al.Galaxy(light=al.lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0,
                                                      sersic_index=1.5))

    tracer = ray_tracing.TracerImageSourcePlanes(lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
                                                 image_plane_grid_stack=image_plane_grid_stack)

    return ccd.CCDData.simulate(array=tracer.image_plane_image_for_simulation, pixel_scales=0.1,
                               exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)


ccd_data = simulate_ccd_data()
ccd.output_ccd_data_to_fits(ccd_data=ccd_data,
                            image_path=ccd_data_path + 'image.fits',
                            noise_map_path=ccd_data_path + 'noise_map.fits',
                           psf_path=ccd_data_path + 'psf.fits', overwrite=True)
