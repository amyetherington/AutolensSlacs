from autofit import conf

import os

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))

conf.instance = conf.Config(config_path=path+'config', output_path=path+'output')

from autolens.data.instrument import ccd
from autolens.data.array import grids
from autolens.data.array import scaled_array
from autolens.lens import ray_tracing
from autolens.model.galaxy import galaxy as g
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.lens.plotters import ray_tracing_plotters
from autolens.data.plotters import ccd_plotters
import astropy.units as u
import scipy.interpolate as interpolate
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage import measure
from autolens.data.array.util import grid_util
from autolens.model.galaxy.util import galaxy_util


workspace_path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))

reg_grid = grids.RegularGrid.from_shape_and_pixel_scale(shape_2d=(100, 100), pixel_scales=0.05)
sub_grid = grids.SubGrid.from_shape_pixel_scale_and_sub_size(shape_2d=(100,100), pixel_scales=0.05,
                                                                      sub_size=16)
sie = mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=2, phi=0, axis_ratio=0.7)


sub_grid_2d = sub_grid.sub_grid_2d_from_sub_grid_1d(sub_grid_1d=sub_grid)
reg_grid_2d = reg_grid.grid_2d_from_grid_1d(grid_1d=reg_grid)

# loading convergence, and deflections on regular and sub grids
kappa_1d = sie.convergence_from_grid(grid=reg_grid)
kappa_1d_sub = sie.convergence_from_grid(grid=sub_grid)
alpha_1d = sie.deflections_from_grid(grid=reg_grid)
alpha_1d_sub = sie.deflections_from_grid(grid=sub_grid)

# loading stuff from new functions
kappa_1d_j = sie.convergence_from_jacobian(grid=reg_grid)
kappa_1d_j_sub = sie.convergence_from_jacobian(grid=sub_grid)
kappa1d_j_binned = sie.convergence_from_jacobian(grid=sub_grid)
grad_psi_1d = sie.deflections_via_potential_from_grid(grid=reg_grid)
grad_psi_1d_sub = sie.deflections_via_potential_from_grid(grid=sub_grid)
magnification_1d = sie.magnification_from_grid(grid=reg_grid)
magnification_1d_sub = sie.magnification_from_grid(grid=sub_grid)

# converting magnification to reg grid from sub grid
magnification_binned =sub_grid.regular_array_1d_from_binned_up_sub_array_1d(sub_array_1d=magnification_1d_sub)

# tangential critical curves, loading, and converting to grid to plot
critical_curves = sie.critical_curves_from_grid(grid=reg_grid)
critical_curves_sub = sie.critical_curves_from_grid(grid=sub_grid)
critical_curve_tan = critical_curves[0]
critical_curves_tan_pix = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=critical_curve_tan, shape_2d=(100,100),
                                                                  pixel_scales=(0.05, 0.05), origin=(0, 0))
critical_curves_tan_sub = sie.critical_curves_from_grid(grid=sub_grid)
critical_curve_tan_sub = critical_curves_sub[0]
critical_curves_pix_tan_sub = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=critical_curve_tan_sub, shape_2d=(1600,1600),
                                                                  pixel_scales=(0.003125, 0.003125), origin=(0, 0))
xcritical_tan, ycritical_tan = critical_curves_tan_pix[:,1], critical_curves_tan_pix[:,0]
xcritical_tan_sub, ycritical_tan_sub = critical_curves_pix_tan_sub[:,1], critical_curves_pix_tan_sub[:,0]
# radial critical curves
critical_curve_rad = critical_curves[1]
critical_curves_rad_pix = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=critical_curve_rad, shape_2d=(100,100),
                                                                  pixel_scales=(0.05, 0.05), origin=(0, 0))
critical_curve_rad_sub = critical_curves_sub[1]
critical_curves_rad_pix_sub = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=critical_curve_rad_sub, shape_2d=(1600,1600),
                                                                  pixel_scales=(0.003125, 0.003125), origin=(0, 0))
xcritical_rad, ycritical_rad = critical_curves_rad_pix[:,1], critical_curves_rad_pix[:,0]
xcritical_rad_sub, ycritical_rad_sub = critical_curves_rad_pix_sub[:,1], critical_curves_rad_pix_sub[:,0]

#loading tangential and radial critical curves from eigenvalues
tangential_critical_curve = sie.tangential_critical_curve_from_grid(grid=sub_grid)
radial_critical_curve = sie.radial_critical_curve_from_grid(grid=sub_grid)

radial_critical_curve_pix_sub = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=radial_critical_curve, shape_2d=(1600,1600),
                                                                  pixel_scales=(0.003125, 0.003125), origin=(0, 0))
radial_xcritical_sub, radial_ycritical_sub = radial_critical_curve_pix_sub[:,1], radial_critical_curve_pix_sub[:,0]
tangential_critical_curve_pix_sub = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=tangential_critical_curve, shape_2d=(1600,1600),
                                                                  pixel_scales=(0.003125, 0.003125), origin=(0, 0))
tangential_xcritical_sub, tangential_ycritical_sub = tangential_critical_curve_pix_sub[:,1], tangential_critical_curve_pix_sub[:,0]
tangential_caustic = sie.tangential_caustic_from_grid(grid=sub_grid)
radial_caustic = sie.radial_caustic_from_grid(grid=sub_grid)
radial_caustic_pix_sub = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=radial_caustic, shape_2d=(1600,1600),
                                                                  pixel_scales=(0.003125, 0.003125), origin=(0, 0))
radial_xcaustic_sub, radial_ycaustic_sub = radial_caustic_pix_sub[:,1], radial_caustic_pix_sub[:,0]
tangential_caustic_pix_sub = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=tangential_caustic, shape_2d=(1600,1600),
                                                                  pixel_scales=(0.003125, 0.003125), origin=(0, 0))
tangential_xcaustic_sub, tangential_ycaustic_sub = tangential_caustic_pix_sub[:,1], tangential_caustic_pix_sub[:,0]


# loading tangential and radial caustics and converting to grid for plotting
caustics = sie.caustics_from_grid(grid=reg_grid)
caustics_sub = sie.caustics_from_grid(grid=sub_grid)
caustic_tan = caustics[0]
caustic_tan_sub = caustics_sub[0]
caustic_pix_tan = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=caustic_tan, shape_2d=(100,100),
                                                                  pixel_scales=(0.05, 0.05), origin=(0, 0))
caustic_pix_tan_sub = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=caustic_tan_sub, shape_2d=(1600,1600),
                                                                  pixel_scales=(0.003125, 0.003125), origin=(0, 0))
xcaustic_tan, ycaustic_tan = caustic_pix_tan[:,1], caustic_pix_tan[:,0]
xcaustic_tan_sub, ycaustic_tan_sub = caustic_pix_tan_sub[:,1], caustic_pix_tan_sub[:,0]
caustic_rad = caustics[1]
caustic_rad_sub = caustics_sub[1]
caustic_pix_rad = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=caustic_rad, shape_2d=(100,100),
                                                                  pixel_scales=(0.05, 0.05), origin=(0, 0))
caustic_pix_rad_sub = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=caustic_rad_sub, shape_2d=(1600,1600),
                                                                  pixel_scales=(0.003125, 0.003125), origin=(0, 0))
xcaustic_rad, ycaustic_rad = caustic_pix_rad[:,1], caustic_pix_rad[:,0]
xcaustic_rad_sub, ycaustic_rad_sub = caustic_pix_rad_sub[:,1], caustic_pix_rad_sub[:,0]

print(caustic_rad)
print(radial_caustic)
print(critical_curve_rad)
print(radial_critical_curve)




# converting to 2d for plotting
kappa_2d = reg_grid.array_2d_from_array_1d(array_1d=kappa_1d)
kappa_2d_sub = sub_grid.sub_array_2d_from_sub_array_1d(sub_array_1d=kappa_1d_sub)
kappa_2d_j = reg_grid.array_2d_from_array_1d(array_1d=kappa_1d_j)
kappa_2d_sub_j = sub_grid.sub_array_2d_from_sub_array_1d(sub_array_1d=kappa_1d_j_sub)
kappa_2d_binned = sub_grid.sub_array_2d_from_sub_array_1d(sub_array_1d=kappa1d_j_binned)
alpha_x_2d = reg_grid.array_2d_from_array_1d(array_1d=alpha_1d[:, 1])
alpha_y_2d = reg_grid.array_2d_from_array_1d(array_1d=alpha_1d[:, 0])
alpha_x_2d_sub = sub_grid.sub_array_2d_from_sub_array_1d(sub_array_1d=alpha_1d_sub[:, 1])
alpha_y_2d_sub = sub_grid.sub_array_2d_from_sub_array_1d(sub_array_1d=alpha_1d_sub[:, 0])
grad_psi_x_2d = reg_grid.array_2d_from_array_1d(array_1d=grad_psi_1d[:, 1])
grad_psi_y_2d = reg_grid.array_2d_from_array_1d(array_1d=grad_psi_1d[:, 0])
grad_psi_x_2d_sub = sub_grid.sub_array_2d_from_sub_array_1d(sub_array_1d=grad_psi_1d_sub[:, 1])
grad_psi_y_2d_sub = sub_grid.sub_array_2d_from_sub_array_1d(sub_array_1d=grad_psi_1d_sub[:, 0])
magnification_2d = reg_grid.array_2d_from_array_1d(array_1d=magnification_1d)
magnification_2d_sub = sub_grid.sub_array_2d_from_sub_array_1d(sub_array_1d=magnification_1d_sub)
magnification_binned_2d = reg_grid.array_2d_from_array_1d(array_1d=magnification_binned)


# calculating differeneces
conv_diff = kappa_2d - kappa_2d_j
conv_diff_sub = kappa_2d_sub - kappa_2d_sub_j
alpha_x_diff = alpha_x_2d - grad_psi_x_2d
alpha_y_diff = alpha_y_2d - grad_psi_y_2d
alpha_x_diff_sub = alpha_x_2d_sub - grad_psi_x_2d_sub
alpha_y_diff_sub = alpha_y_2d_sub - grad_psi_y_2d_sub




def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12,4))
im1 = ax1.imshow(np.log10(kappa_2d), cmap='jet')
colorbar(im1)
ax1.set_title('convergence from calculation')

im2 = ax2.imshow(np.log10(kappa_2d_j), cmap='jet')
colorbar(im2)
ax2.set_title('convergence from jacobian')

im3 = ax3.imshow(conv_diff,cmap='jet')
colorbar(im3)
ax3.set_title('difference')

plt.tight_layout(h_pad=1)
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                    wspace=None, hspace=None)

plt.close()

fig2, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12,4))
im1 = ax1.imshow(np.log10(kappa_2d_sub), cmap='jet')
colorbar(im1)
ax1.set_title('convergence from calculation on sub grid')

im2 = ax2.imshow(np.log10(kappa_2d_sub_j), cmap='jet')
colorbar(im2)
ax2.set_title('convergence from jacobian on sub grid')

im3 = ax3.imshow(conv_diff_sub,cmap='jet')
colorbar(im3)
ax3.set_title('difference')

plt.tight_layout(h_pad=1)
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                    wspace=None, hspace=None)
plt.close()

fig3, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12,4))
im1 = ax1.imshow(alpha_x_2d, cmap='jet')
colorbar(im1)
ax1.set_title('alpha x from calculation')

im2 = ax2.imshow(grad_psi_x_2d, cmap='jet')
colorbar(im2)
ax2.set_title('alpha x from jacobian')

im3 = ax3.imshow(alpha_x_diff,cmap='jet')
colorbar(im3)
ax3.set_title('difference')

plt.tight_layout(h_pad=1)
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                    wspace=None, hspace=None)
plt.close()

fig4, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12,4))
im1 = ax1.imshow(alpha_y_2d, cmap='jet')
colorbar(im1)
ax1.set_title('alpha y from calculation')

im2 = ax2.imshow(grad_psi_y_2d, cmap='jet')
colorbar(im2)
ax2.set_title('alpha y from jacobian')

im3 = ax3.imshow(alpha_y_diff,cmap='jet')
colorbar(im3)
ax3.set_title('difference')

plt.tight_layout(h_pad=1)
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                    wspace=None, hspace=None)
plt.close()

fig5, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12,4))
im1 = ax1.imshow(alpha_x_2d_sub, cmap='jet')
colorbar(im1)
ax1.set_title('alpha x from calculation on sub grid')

im2 = ax2.imshow(grad_psi_x_2d_sub, cmap='jet')
colorbar(im2)
ax2.set_title('alpha x from jacobian on sub grid')

im3 = ax3.imshow(alpha_x_diff_sub,cmap='jet')
colorbar(im3)
ax3.set_title('difference')

plt.tight_layout(h_pad=1)
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                    wspace=None, hspace=None)
plt.close()

fig6, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12,4))
im1 = ax1.imshow(alpha_y_2d_sub, cmap='jet')
colorbar(im1)
ax1.set_title('alpha y from calculation on sub grid')

im2 = ax2.imshow(grad_psi_y_2d_sub, cmap='jet')
colorbar(im2)
ax2.set_title('alpha y from jacobian on sub grid')

im3 = ax3.imshow(alpha_y_diff_sub,cmap='jet')
colorbar(im3)
ax3.set_title('difference')

plt.tight_layout(h_pad=1)
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                    wspace=None, hspace=None)
#plt.close()

fig7, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(np.log10(np.abs(magnification_2d)), cmap='viridis')
ax1.set_title('magnification on reg grid')
colorbar(im1)
im2 = ax2.imshow(np.log10(np.abs(magnification_2d_sub)), cmap='viridis')
ax2.set_title('magnification on sub grid')
colorbar(im2)
#plt.close()


fig8, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(np.log10(kappa_2d), cmap='viridis')
ax1.plot(xcritical_tan,ycritical_tan, c='r', lw=1.5, zorder=200)
ax1.plot(xcaustic_tan, ycaustic_tan,c='g', lw=1.5, zorder=200)
ax1.plot(xcritical_rad,ycritical_rad, c='r', lw=1.5, zorder=200)
ax1.plot(xcaustic_rad, ycaustic_rad,c='g', lw=1.5, zorder=200)
ax1.set_title('crtitcal curves and caustics from reg grid')
colorbar(im1)
im2 = ax2.imshow(np.log10(kappa_2d_sub), cmap='viridis')
ax2.plot(xcritical_tan_sub,ycritical_tan_sub, c='r', lw=1.5, zorder=200)
ax2.plot(xcaustic_tan_sub, ycaustic_tan_sub,c='g', lw=1.5, zorder=200)
ax2.plot(xcritical_rad_sub,ycritical_rad_sub, c='r', lw=1.5, zorder=200)
ax2.plot(xcaustic_rad_sub, ycaustic_rad_sub,c='g', lw=1.5, zorder=200)
ax2.set_title('critical curves and caustics from sub grid')
colorbar(im2)
#plt.close()

fig9, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(np.log10(kappa_2d_sub), cmap='viridis')
ax1.plot(xcritical_tan_sub, ycritical_tan_sub, c='r', lw=1.5, zorder=200)
ax1.plot(xcaustic_tan_sub, ycaustic_tan_sub,c='g', lw=1.5, zorder=200)
ax1.plot(xcritical_rad_sub, ycritical_rad_sub, c='r', lw=1.5, zorder=200)
ax1.plot(xcaustic_rad_sub, ycaustic_rad_sub,c='g', lw=1.5, zorder=200)
ax1.set_title('crtitcal curves from magnification')
colorbar(im1)
im2 = ax2.imshow(np.log10(kappa_2d_sub), cmap='viridis')
ax2.plot(tangential_xcritical_sub,tangential_ycritical_sub, c='r', lw=1.5, zorder=200)
ax2.plot(tangential_xcaustic_sub,tangential_ycaustic_sub, c='g', lw=1.5, zorder=200)
ax2.plot(radial_xcritical_sub,radial_ycritical_sub, c='r', lw=1.5, zorder=200)
ax2.plot(radial_xcaustic_sub,radial_ycaustic_sub, c='g', lw=1.5, zorder=200)
ax2.set_title('critical curves from eigenvalues')
colorbar(im2)


fig10, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(np.log10(np.abs(magnification_2d)), cmap='viridis')
ax1.set_title('magnification on reg grid')
colorbar(im1)
im2 = ax2.imshow(np.log10(np.abs(magnification_binned_2d)), cmap='viridis')
ax2.set_title('magnification on reg grid binned from sub grid')
colorbar(im2)
#plt.close()

fig11, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12,4))
im1 = ax1.imshow(np.log10(kappa_2d_j), cmap='jet')
colorbar(im1)
ax1.set_title('convergence on reg grid')

im2 = ax2.imshow(np.log10(kappa_2d_sub_j), cmap='jet')
colorbar(im2)
ax2.set_title('convergence on sub grid')

im3 = ax3.imshow(np.log10(kappa_2d_binned),cmap='jet')
colorbar(im3)
ax3.set_title('convergence binned to reg grid')
plt.close()

plt.show()
