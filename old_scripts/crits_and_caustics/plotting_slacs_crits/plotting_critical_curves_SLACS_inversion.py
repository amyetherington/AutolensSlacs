from autofit import conf

import os

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))

conf.instance = conf.Config(config_path=path+'config', output_path=path+'output')

from autolens.model.profiles import mass_profiles as mp
from autolens.array import grids
from autolens.array.util import grid_util
from autolens.lens import plane as pl
from autolens.model.galaxy import galaxy as g
from autolens.array import scaled_array
from autolens.plotters import array_plotters

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from math import pi, cos, sin


path = '{}/../../data/slacs/'.format(os.path.dirname(os.path.realpath(__file__)))
data_path = '{}/../../../../../output/slacs_final/F814W/'.format(os.path.dirname(os.path.realpath(__file__)))
pipeline = '/pipeline_inv_hyper__lens_bulge_disk_sie__source_inversion/'
no_shear = 'pipeline_tag__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/' \
           'phase_4__lens_bulge_disk_sie__source_inversion/phase_tag__sub_2__pos_1.00/model.results'
slacs_path = '{}/../../data/slacs_data_table.xlsx'.format(os.path.dirname(os.path.realpath(__file__)))

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

list_ = []
lens = []
SLACS_image = []
lens_plane_source_model_image = []

slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name
slacs = slacs.drop(['slacs0959+5100', 'slacs0959+4416'], axis=0)



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
        image_path = path + lens_name[i] +'/F814W_image.fits'
        model_image_path= data_path + lens_name[i] + '/pipeline_inv_hyper__lens_bulge_disk_sie__source_inversion/'\
           'pipeline_tag__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/' \
           'phase_4__lens_bulge_disk_sie__source_inversion/phase_tag__sub_2__pos_1.00/'\
           'image/lens_fit/fits/fit_model_image_of_plane_1.fits'
        image = al.Array.from_fits(file_path=image_path, hdu=0, pixel_scales=0.03)
        model_image = al.Array.from_fits(file_path=model_image_path, hdu=0, pixel_scales=0.03)
        SLACS_image.append(image)
        lens_plane_source_model_image.append(model_image)
    else:
        slacs = slacs.drop([lens_name[i]], axis=0)

#array_plotters.plot_array(array=image, title='slacs2303+1422')
#array_plotters.plot_array(array=model_image, title='slacs2303+1422 source model image plane')



for i in range(len(SLACS_image)):
    ## galaxies without external shear component
    grid = al.Grid.uniform(
        shape_2d=lens_plane_source_model_image[i].shape, pixel_scales=0.03, sub_size=4)

    sie = mp.EllipticalIsothermal(
            centre=(results.loc[lens[i]]['param']['centre_0'], results.loc[lens[i]]['param']['centre_1']),
            axis_ratio=results.loc[lens[i]]['param']['axis_ratio'], phi=results.loc[lens[i]]['param']['phi'],
            einstein_radius=results.loc[lens[i]]['param']['einstein_radius'])


    critical_curves = sie.critical_curves_from_grid(grid=grid)
    caustics = sie.caustics_from_grid(grid=grid)

    critical_curve_tan, critical_curve_rad = critical_curves[0], critical_curves[1]
    caustic_tan, caustic_rad = caustics[0], caustics[1]



# naming for x and y for plotting
    xcritical_tan, ycritical_tan = critical_curve_tan[:,1], critical_curve_tan[:,0]
#    xcritical_rad, ycritical_rad = critical_curve_rad_grid[:,1], critical_curve_rad_grid[:,0]
    xcaustic_tan, ycaustic_tan = caustic_tan[:,1], caustic_tan[:,0]
#    xcaustic_rad, ycaustic_rad = caustic_rad_grid[:,1], caustic_rad_grid[:,0]

    centre = np.array([[results.loc[lens[i]]['param']['centre_1'],
                        results.loc[lens[i]]['param']['centre_0']]])  #position of the center
    a = results.loc[lens[i]]['param']['einstein_radius']/ np.sqrt(results.loc[lens[i]]['param']['axis_ratio']) # major-axis
    b = results.loc[lens[i]]['param']['einstein_radius']* np.sqrt(results.loc[lens[i]]['param']['axis_ratio']) # minior-axis
    phi = results.loc[lens[i]]['param']['phi']*np.pi/180  # rotation angle




    t = np.linspace(0, 2 * pi, 100)
    Ell = np.array([a * np.cos(t), b * np.sin(t)])
    # keep the same center location
    R_rot = np.array([[cos(phi), -sin(phi)], [sin(phi), cos(phi)]])
    # 2-D rotation matrix
    Ell_rot = np.zeros((2, Ell.shape[1]))

    for j in range(Ell.shape[1]):
        Ell_rot[:, j] = np.dot(R_rot, Ell[:, j])

    slacs_centre = np.array([[results.loc[lens[i]]['param']['centre_1'],
                        results.loc[lens[i]]['param']['centre_0']]])  # position of the center
    slacs_a = slacs['b_SIE'][i]/ np.sqrt(slacs['q_SIE'][i])  # major-axis
    slacs_b = slacs['b_SIE'][i]* np.sqrt(slacs['q_SIE'][i])  # minior-axis
    slacs_phi = slacs['PA'][i] * np.pi / 180  # rotation angle

    slacs_t = np.linspace(0, 2 * pi, 100)
    slacs_Ell = np.array([slacs_a * np.cos(slacs_t), slacs_b * np.sin(slacs_t)])
    # keep the same center location
    slacs_R_rot = np.array([[cos(slacs_phi), -sin(slacs_phi)], [sin(slacs_phi), cos(slacs_phi)]])
    # 2-D rotation matrix
    slacs_Ell_rot = np.zeros((2, slacs_Ell.shape[1]))




    for k in range(slacs_Ell.shape[1]):
        slacs_Ell_rot[:, k] = np.dot(slacs_R_rot, slacs_Ell[:, k]) 



    plt.figure()
    plt.imshow(lens_plane_source_model_image[i], cmap='jet', extent=[model_image.arc_second_minima[1], model_image.arc_second_maxima[1], model_image.arc_second_minima[0], model_image.arc_second_maxima[0]])
    plt.plot(xcritical_tan,ycritical_tan, c='r', lw=1.5, zorder=200, linestyle='--')
    plt.plot(xcaustic_tan, ycaustic_tan,c='g', lw=1.5, zorder=200)
    plt.plot(centre[0,0] + slacs_Ell_rot[0, :], centre[0,1]+ slacs_Ell_rot[1,:], 'yellow')
    plt.plot(centre[0, 0] + Ell_rot[0, :], centre[0, 1] + Ell_rot[1, :], 'darkorange')



 #   plt.plot(xcritical_rad,ycritical_rad, c='r', lw=1.5, zorder=200)
 #   plt.plot(xcaustic_rad, ycaustic_rad,c='g', lw=1.5, zorder=200)


plt.show()


