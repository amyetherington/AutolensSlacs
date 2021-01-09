from autolens.model.profiles import mass_profiles as mp
from autolens.array import grids
from autolens.array.util import grid_util
from autolens.lens import plane as pl
from autolens.model.galaxy import galaxy as g

import matplotlib.pyplot as plt
import numpy as np

# setting up grid and mass profile
sie = mp.EllipticalIsothermal(
    centre=(0.0, 0.0), einstein_radius=1.4,axis_ratio=0.7, phi=40.0)

grid = al.Grid.uniform(
            shape_2d=(100, 100), pixel_scales=0.05, sub_size=1)

# loading convergence, critical curves, and caustics
kappa = sie.convergence_from_grid(grid=grid, return_in_2d=True)
critical_curves = sie.critical_curves_from_grid(grid=grid)
caustics = sie.caustics_from_grid(grid=grid)


# naming tangential and radial critical curves
critical_curve_tan, critical_curve_rad = critical_curves[0], critical_curves[1]
caustic_tan, caustic_rad = caustics[0], caustics[1]

# converting to pixel grid (incapable of plotting convergence in arcseconds -_________- )
critical_curve_tan_grid = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=critical_curve_tan, shape_2d=(100,100),
                                                                  pixel_scales=(0.05, 0.05), origin=(0, 0))
critical_curve_rad_grid = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=critical_curve_rad, shape_2d=(100,100),
                                                                  pixel_scales=(0.05, 0.05), origin=(0, 0))
caustic_tan_grid = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=caustic_tan, shape_2d=(100,100),
                                                                  pixel_scales=(0.05, 0.05), origin=(0, 0))
caustic_rad_grid = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=caustic_rad, shape_2d=(100,100),
                                                                  pixel_scales=(0.05, 0.05), origin=(0, 0))
# naming for x and y for plotting
xcritical_tan, ycritical_tan = critical_curve_tan_grid[:,1], critical_curve_tan_grid[:,0]
xcritical_rad, ycritical_rad = critical_curve_rad_grid[:,1], critical_curve_rad_grid[:,0]
xcaustic_tan, ycaustic_tan = caustic_tan_grid[:,1], caustic_tan_grid[:,0]
xcaustic_rad, ycaustic_rad = caustic_rad_grid[:,1], caustic_rad_grid[:,0]

fig1 = plt.imshow(np.log10(kappa), cmap='jet')
plt.plot(xcritical_tan,ycritical_tan, c='r', lw=1.5, zorder=200)
plt.plot(xcaustic_tan, ycaustic_tan,c='g', lw=1.5, zorder=200)
#plt.plot(xcritical_rad,ycritical_rad, c='r', lw=1.5, zorder=200)
#plt.plot(xcaustic_rad, ycaustic_rad,c='g', lw=1.5, zorder=200)


plt.show()
