from autofit import conf

import os

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))

conf.instance = conf.Config(config_path=path+'config', output_path=path+'output')

from autolens.data.instrument import ccd
from autolens.data.array import grids
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

# generating grids and galxies from autolens
reg_grid = grids.RegularGrid.from_shape_and_pixel_scale(shape_2d=(100, 100), pixel_scales=0.05)
sub_grid = grids.SubGrid.from_shape_pixel_scale_and_sub_size(shape_2d=(100,100), pixel_scales=0.05,
                                                                      sub_size=16)
sub_grid_2d = sub_grid.sub_grid_2d_from_sub_grid_1d(sub_grid_1d=sub_grid)
lens_galaxy = al.Galaxy(mass=al.mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=2,
                                                   axis_ratio=0.7, phi=35.0), redshift=0.5)
#lens_galaxy = al.Galaxy(mass=al.mp.SphericalIsothermal(centre=(0,0), einstein_radius=2))
source_galaxy = al.Galaxy(light=al.lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0,
                                                        intensity=1.0, effective_radius=1.0, sersic_index=2.5),
                         redshift=1)
reg_grid_2d = reg_grid.grid_2d_from_grid_1d(grid_1d=reg_grid)
sub_grid_2d = sub_grid.sub_grid_2d_from_sub_grid_1d(sub_grid_1d=sub_grid)
reg_grid_pix = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=reg_grid, shape_2d=(100,100),
                                                                 pixel_scales=(0.05, 0.05), origin=(0, 0))
reg_grid_pix_2d = reg_grid.grid_2d_from_grid_1d(grid_1d=reg_grid_pix)
sub_grid_pix = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=sub_grid, shape_2d=(1600,1600),
                                                          pixel_scales=(0.003125,0.003125), origin=(0,0))
sub_grid_pix_2d = sub_grid.sub_grid_2d_from_sub_grid_1d(sub_grid_1d=sub_grid_pix)

# loading convergence, potential, and deflections on regular and sub grids
kappa_1d = lens_galaxy.convergence_from_grid(grid=reg_grid)
kappa_2d = reg_grid.array_2d_from_array_1d(array_1d=kappa_1d)
kappa_1d_sub = lens_galaxy.convergence_from_grid(grid=sub_grid)
kappa_1d_binned = sub_grid.regular_array_1d_from_binned_up_sub_array_1d(sub_array_1d=kappa_1d_sub)
kappa_2d_binned = reg_grid.array_2d_from_array_1d(array_1d=kappa_1d_binned)
kappa_2d_sub = sub_grid.sub_array_2d_from_sub_array_1d(sub_array_1d=kappa_1d_sub)
psi_1d = lens_galaxy.potential_from_grid(grid=reg_grid)
psi_2d = reg_grid.array_2d_from_array_1d(array_1d=psi_1d)
psi_1d_sub = lens_galaxy.potential_from_grid(grid=sub_grid)
psi_2d_sub = sub_grid.sub_array_2d_from_sub_array_1d(sub_array_1d=psi_1d_sub)
alpha_1d = lens_galaxy.deflections_from_grid(grid=reg_grid)
alpha_1d_sub = lens_galaxy.deflections_from_grid(grid=sub_grid)
alphax_2d = reg_grid.array_2d_from_array_1d(array_1d=alpha_1d[:, 1])
alphay_2d = reg_grid.array_2d_from_array_1d(array_1d=alpha_1d[:, 0])
alphax_2d_sub = sub_grid.sub_array_2d_from_sub_array_1d(sub_array_1d=alpha_1d_sub[:, 1])
alphay_2d_sub = sub_grid.sub_array_2d_from_sub_array_1d(sub_array_1d=alpha_1d_sub[:, 0])


# jacobian matrix from deflection angles loaded from autolens
deflect_A11 = 1 - np.gradient(alphax_2d, reg_grid_2d[0,:,1], axis=1)
deflect_A12 = - np.gradient(alphax_2d, reg_grid_2d[:,1,0], axis=0)
deflect_A21 = - np.gradient(alphay_2d, reg_grid_2d[0,:,1], axis=1)
deflect_A22 = 1 - np.gradient(alphay_2d, reg_grid_2d[:,1,0], axis=0)

jacobian = np.array([[deflect_A11, deflect_A12],[deflect_A21, deflect_A22]])

deflect_detA = deflect_A11 * deflect_A22 - deflect_A12 * deflect_A21
deflect_mag = 1 / deflect_detA

deflect_ka = 1 - 0.5 * (deflect_A11 + deflect_A22)
deflect_ga1 = 0.5 * (deflect_A22 - deflect_A11)
deflect_ga2 = -0.5 * (deflect_A12 + deflect_A21)
deflect_ga = (deflect_ga1 ** 2 + deflect_ga2 ** 2) ** 0.5

deflect_lambda_t = 1 - deflect_ka - deflect_ga
deflect_lambda_r = 1 - deflect_ka + deflect_ga


# jacobian matrix from deflection angles loaded from autolens on subgrid
deflect_A11_sub = 1 - np.gradient(alphax_2d_sub, sub_grid_2d[0,:,1], axis=1)
deflect_A12_sub = - np.gradient(alphax_2d_sub, sub_grid_2d[:,1,0], axis=0)
deflect_A21_sub = - np.gradient(alphay_2d_sub, sub_grid_2d[0,:,1], axis=1)
deflect_A22_sub = 1 - np.gradient(alphay_2d_sub, sub_grid_2d[:,1,0], axis=0)

jacobian_sub = np.array([[deflect_A11_sub, deflect_A12_sub],[deflect_A21_sub, deflect_A22_sub]])

deflect_detA_sub = deflect_A11_sub * deflect_A22_sub - deflect_A12_sub * deflect_A21_sub
deflect_mag_sub = 1 / deflect_detA_sub

detA_1d = sub_grid.sub_array_1d_from_sub_array_2d(sub_array_2d=deflect_detA_sub)
detA_1d_binned = sub_grid.regular_array_1d_from_binned_up_sub_array_1d(sub_array_1d=detA_1d)
detA_2d_binned = reg_grid.array_2d_from_array_1d(array_1d=detA_1d_binned)

mag_1d = sub_grid.sub_array_1d_from_sub_array_2d(sub_array_2d=deflect_mag_sub)
mag_binned = sub_grid.regular_array_1d_from_binned_up_sub_array_1d(sub_array_1d=mag_1d)
mag_binned_2d = reg_grid.array_2d_from_array_1d(array_1d=mag_binned)

deflect_ka_sub = 1 - 0.5 * (deflect_A11_sub + deflect_A22_sub)
deflect_ga1_sub = 0.5 * (deflect_A22_sub - deflect_A11_sub)
deflect_ga2_sub = -0.5 * (deflect_A12_sub + deflect_A21_sub)
deflect_ga_sub = (deflect_ga1_sub ** 2 + deflect_ga2_sub ** 2) ** 0.5

deflect_lambda_t_sub = 1 - deflect_ka_sub - deflect_ga_sub
deflect_lambda_r_sub = 1 - deflect_ka_sub + deflect_ga_sub

# jacobian matrix and magnification from deflection angles calculated from grad psi
alpha_x = np.gradient(psi_2d, reg_grid_2d[0,:,1], axis=1)
alpha_y = np.gradient(psi_2d, reg_grid_2d[:,1,0], axis=0)

alpha_x_sub = np.gradient(psi_2d_sub, sub_grid_2d[0,:,1], axis=1)
alpha_y_sub = np.gradient(psi_2d_sub, sub_grid_2d[:,1,0], axis=0)

print(alpha_x_sub)

A11 = 1 - np.gradient(alpha_x, reg_grid_2d[0,:,1], axis=1)
A12 = - np.gradient(alpha_x, reg_grid_2d[:,1,0], axis=0)
A21 = - np.gradient(alpha_y, reg_grid_2d[0,:,1], axis=1)
A22 = 1 - np.gradient(alpha_y, reg_grid_2d[:,1,0], axis=0)

detA = A11 * A22 - A12 * A21
mag = 1 / detA



ka = 1 - 0.5 * (A11 + A22)
ga1 = 0.5 * (A22 - A11)
ga2 = -0.5 * (A12 + A21)
ga = (ga1 ** 2 + ga2 ** 2) ** 0.5

lambda_t = 1 - ka - ga
lambda_r = 1 - ka + ga




# sprting out colorbars for subfigures
def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


# magnifications and convergence calculated from grad psi
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12,4))
im1 = ax1.imshow(np.log10(kappa_2d), cmap='jet')
colorbar(im1)
#ax1.contour(reg_grid_pix_2d[:,:,1], reg_grid_pix_2d[:,:,0], detA, levels=(0,), colors='r', linewidths=1.5, zorder=200)
ax1.contour(reg_grid_pix_2d[:,:,1], reg_grid_pix_2d[:,:,0], lambda_t, levels=(0,), colors='b', linewidths=2.5, zorder=200)

critical_curves_indices = measure.find_contours(detA, 0)
Ncrit = len(critical_curves_indices)
critical_curves = []
caustics = []

for jj in np.arange(Ncrit):
    critical_curves.append(critical_curves_indices[jj])
    print(critical_curves)
    xcritical, ycritical = critical_curves[jj].T
    pixel_coord = np.stack((xcritical, ycritical), axis=-1)
    print(pixel_coord)

    new_grid = grid_util.grid_pixels_1d_to_grid_arcsec_1d(grid_pixels_1d=pixel_coord, shape_2d=(100,100),
                                                           pixel_scales=(0.05, 0.05), origin=(0.0, 0.0))

    crit_deflect = lens_galaxy.deflections_from_grid(grid=new_grid)
    ycaustic_arc = new_grid[:,0] - crit_deflect[:,0]
    xcaustic_arc = new_grid[:,1] - crit_deflect[:,1]
    caustics_arc = np.stack((ycaustic_arc, xcaustic_arc), axis=-1)
    caustic_grid = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=caustics_arc, shape_2d=(100,100),
                                                                  pixel_scales=(0.05, 0.05), origin=(0, 0))
    xcaustic = caustic_grid[:,1]
    ycaustic = caustic_grid[:,0]

    ax1.plot(ycaustic, xcaustic, c='g', lw=1.5, zorder=250)
    ax1.plot(ycritical, xcritical,c='r', lw=1.5, zorder=200)

ax1.set_title('convergence')

im2 = ax2.imshow(np.log10(np.abs(mag)), cmap='viridis')
colorbar(im2)
ax2.set_title('magnification')

im3 = ax3.imshow(np.log10(ka),cmap='jet')
colorbar(im3)
ax3.set_title('convergence from deflection angles')
plt.tight_layout(h_pad=1)
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                    wspace=None, hspace=None)
fig.savefig('magnification_from_potential.png')
#plt.show()

# magnifications and convergence as calculated from autolens deflection angles
fig2, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12,4))
im1 = ax1.imshow(np.log10(kappa_2d), cmap='jet')
colorbar(im1)
#ax1.contour(reg_grid_pix_2d[:,:,1], reg_grid_pix_2d[:,:,0], deflect_detA, levels=(0,), colors='r', linewidths=1.5, zorder=200)
#ax1.contour(reg_grid_pix_2d[:,:,1], reg_grid_pix_2d[:,:,0], deflect_lambda_t, levels=(0,), colors='b', linewidths=2.5, zorder=50)
ax1.set_title('convergence')

critical_curves_indices_deflect = measure.find_contours(deflect_detA, 0)
Ncrit_deflect = len(critical_curves_indices_deflect)
critical_curves_deflect = []
caustics_deflect = []

for jj in np.arange(Ncrit_deflect):
    critical_curves_deflect.append(critical_curves_indices_deflect[jj])
    xcritical_deflect, ycritical_deflect = critical_curves_deflect[jj].T
    pixel_coord_deflect = np.stack((xcritical_deflect, ycritical_deflect), axis=-1)

    new_grid_deflect = grid_util.grid_pixels_1d_to_grid_arcsec_1d(grid_pixels_1d=pixel_coord_deflect, shape_2d=(100,100),
                                                           pixel_scales=(0.05, 0.05), origin=(0.0, 0.0))
    deflect_crit_deflect = lens_galaxy.deflections_from_grid(grid=new_grid_deflect)
    ycaustic_arc_deflect = new_grid_deflect[:,0] - deflect_crit_deflect[:,0]
    xcaustic_arc_deflect = new_grid_deflect[:,1] - deflect_crit_deflect[:,1]
    caustics_arc_deflect = np.stack((ycaustic_arc_deflect, xcaustic_arc_deflect), axis=-1)
    caustic_grid_deflect = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=caustics_arc_deflect, shape_2d=(100,100),
                                                                  pixel_scales=(0.05, 0.05), origin=(0, 0))
    xcaustic_deflect = caustic_grid_deflect[:,1]
    ycaustic_deflect = caustic_grid_deflect[:,0]

    caustics_deflect.append(np.stack((xcaustic_deflect, ycaustic_deflect), axis=-1))

    ax1.plot(ycaustic_deflect, xcaustic_deflect, c='g', lw=1.5, zorder=250)
    ax1.plot(ycritical_deflect, xcritical_deflect, c='r', lw=1.5, zorder=250)

im2 = ax2.imshow(np.log10(np.abs(deflect_mag)), cmap='viridis')
colorbar(im2)
ax2.set_title('magnification')

im3 = ax3.imshow(np.log10(deflect_ka),cmap='jet')
colorbar(im3)
ax3.set_title('convergence from deflection angles')

plt.tight_layout(h_pad=1)

fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                    wspace=None, hspace=None)
plt.close()
#fig2.savefig('magnification_from_deflection_angles.png')

# magnifications and convergence on subgrod
fig3, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12,4))
im1 = ax1.imshow(np.log10(kappa_2d_sub), cmap='jet')
colorbar(im1)
#ax1.contour(sub_grid_pix_2d[:,:,1], sub_grid_pix_2d[:,:,0], deflect_detA_sub, levels=(0,), colors='r', linewidths=1.5, zorder=200)
#ax1.contour(sub_grid_pix_2d[:,:,1], sub_grid_pix_2d[:,:,0], deflect_lambda_t_sub, levels=(0,), colors='b', linewidths=2.5, zorder=50)
ax1.set_title('convergence')

critical_curves_indices_sub = measure.find_contours(deflect_detA_sub, 0)
Ncrit_sub = len(critical_curves_indices_sub)
critical_curves_sub = []
caustics_sub = []

for jj in np.arange(Ncrit_sub):
    critical_curves_sub.append(critical_curves_indices_sub[jj])
    xcritical_sub, ycritical_sub = critical_curves_sub[jj].T
    pixel_coord_sub = np.stack((xcritical_sub, ycritical_sub), axis=-1)

    new_grid_sub = grid_util.grid_pixels_1d_to_grid_arcsec_1d(grid_pixels_1d=pixel_coord_sub, shape_2d=(200,200),
                                                           pixel_scales=(0.025, 0.025), origin=(0.0, 0.0))
    crit_deflect_sub = lens_galaxy.deflections_from_grid(grid=new_grid_sub)
    ycaustic_arc_sub = new_grid_sub[:,0] - crit_deflect_sub[:,0]
    xcaustic_arc_sub = new_grid_sub[:,1] - crit_deflect_sub[:,1]
    caustics_arc_sub = np.stack((ycaustic_arc_sub, xcaustic_arc_sub), axis=-1)
    caustic_grid_sub = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=caustics_arc_sub, shape_2d=(200,200),
                                                                  pixel_scales=(0.025, 0.025), origin=(0, 0))
    xcaustic_sub = caustic_grid_sub[:,1]
    ycaustic_sub = caustic_grid_sub[:,0]

    ax1.plot(ycaustic_sub, xcaustic_sub, c='g', lw=1.5, zorder=250)
    ax1.plot(ycritical_sub, xcritical_sub, c='r', lw=1.5, zorder=250)

im2 = ax2.imshow(np.log10(np.abs(deflect_mag_sub)), cmap='viridis')
colorbar(im2)
ax2.set_title('magnification')

im3 = ax3.imshow(np.log10(deflect_ka_sub),cmap='jet')
colorbar(im3)
ax3.set_title('convergence from deflection angles')

plt.tight_layout(h_pad=1)

fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                    wspace=None, hspace=None)
plt.close()
#fig3.savefig('magnification_from_deflection_angles_sub_grid.png')

# calculating and plotting difference between magnifications calculated from deflection angles and grad psi
mag_diff = deflect_mag-mag

fig4 = plt.figure(4)
plt.imshow(mag_diff, cmap='viridis')
plt.colorbar()
plt.title('magnitude difference')
plt.close()
#fig4.savefig('magnification_difference.png')

# calculating and plotting difference between convergence via calculation and via jacobian
deflect_conv_diff = kappa_2d - deflect_ka
conv_diff = kappa_2d - ka

fig5, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(deflect_conv_diff, cmap='jet', vmin=-4, vmax=4)
ax1.set_title('from deflection angles')
colorbar(im1)
im2 = ax2.imshow(conv_diff, cmap='jet', vmin=-4, vmax=4)
ax2.set_title('from potential')
colorbar(im2)
plt.close()
#fig5.savefig('convergence_difference.png')

# calculating and plotting difference between deflection angles from autolens and grad psi
diff_alphax = alpha_x-alphax_2d
diff_alphay = alpha_y-alphay_2d

fig6, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(diff_alphax, cmap='viridis')
ax1.set_title('x alpha difference ')
colorbar(im1)
im2 = ax2.imshow(diff_alphay, cmap='viridis')
ax2.set_title('y alpha difference')
colorbar(im2)
plt.close()

fig7, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12,4))
im1 = ax1.imshow(np.log10(np.abs(deflect_mag)), cmap='viridis')
ax1.set_title('magnification on reg grid ')
colorbar(im1)
im2 = ax2.imshow(np.log10(np.abs(deflect_mag_sub)), cmap='viridis')
ax2.set_title('magnification on sub grid')
colorbar(im2)
im3 = ax3.imshow(np.log10(np.abs(mag_binned_2d)), cmap='viridis')
ax3.set_title('magnification on binned sub grid')
colorbar(im3)

# image comparing caustics and critical curves on re grid, sub grid, and binned reg grid

fig8, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12,4))
im1 = ax1.imshow(np.log10(kappa_2d), cmap='jet')
colorbar(im1)
#ax1.contour(reg_grid_pix_2d[:,:,1], reg_grid_pix_2d[:,:,0], deflect_detA, levels=(0,), colors='r', linewidths=1.5, zorder=200)
#ax1.contour(reg_grid_pix_2d[:,:,1], reg_grid_pix_2d[:,:,0], deflect_lambda_t, levels=(0,), colors='b', linewidths=2.5, zorder=50)
ax1.set_title('regular grid')

critical_curves_indices_deflect = measure.find_contours(deflect_detA, 0)
Ncrit_deflect = len(critical_curves_indices_deflect)
critical_curves_deflect = []
caustics_deflect = []

for jj in np.arange(Ncrit_deflect):
    critical_curves_deflect.append(critical_curves_indices_deflect[jj])
    xcritical_deflect, ycritical_deflect = critical_curves_deflect[jj].T
    pixel_coord_deflect = np.stack((xcritical_deflect, ycritical_deflect), axis=-1)
    new_grid_deflect = grid_util.grid_pixels_1d_to_grid_arcsec_1d(grid_pixels_1d=pixel_coord_deflect, shape_2d=(100,100),
                                                           pixel_scales=(0.05, 0.05), origin=(0.0, 0.0))
    deflect_crit_deflect = lens_galaxy.deflections_from_grid(grid=new_grid_deflect)
    ycaustic_arc_deflect = new_grid_deflect[:,0] - deflect_crit_deflect[:,0]
    xcaustic_arc_deflect = new_grid_deflect[:,1] - deflect_crit_deflect[:,1]
    caustics_arc_deflect = np.stack((ycaustic_arc_deflect, xcaustic_arc_deflect), axis=-1)
    caustic_grid_deflect = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=caustics_arc_deflect, shape_2d=(100,100),
                                                                  pixel_scales=(0.05, 0.05), origin=(0, 0))
    xcaustic_deflect = caustic_grid_deflect[:,1]
    ycaustic_deflect = caustic_grid_deflect[:,0]

    caustics_deflect.append(np.stack((xcaustic_deflect, ycaustic_deflect), axis=-1))

    ax1.plot(ycaustic_deflect, xcaustic_deflect, c='g', lw=1.5, zorder=250)
    ax1.plot(ycritical_deflect, xcritical_deflect, c='r', lw=1.5, zorder=250)

im2 = ax2.imshow(np.log10(kappa_2d_sub), cmap='jet')
colorbar(im2)
ax2.contour(sub_grid_pix_2d[:,:,1], sub_grid_pix_2d[:,:,0], deflect_detA_sub, levels=(0,), colors='r', linewidths=1.5, zorder=200)
ax2.contour(sub_grid_pix_2d[:,:,1], sub_grid_pix_2d[:,:,0], deflect_lambda_t_sub, levels=(0,), colors='b', linewidths=2.5, zorder=50)
ax2.set_title('sub grid')

critical_curves_indices_sub = measure.find_contours(deflect_detA_sub, 0)
Ncrit_sub = len(critical_curves_indices_sub)
critical_curves_sub = []
caustics_sub = []

for jj in np.arange(Ncrit_sub):
    critical_curves_sub.append(critical_curves_indices_sub[jj])
    xcritical_sub, ycritical_sub = critical_curves_sub[jj].T
    pixel_coord_sub = np.stack((xcritical_sub, ycritical_sub), axis=-1)
    new_grid_sub = grid_util.grid_pixels_1d_to_grid_arcsec_1d(grid_pixels_1d=pixel_coord_sub, shape_2d=(1600,1600),
                                                           pixel_scales=(0.003125, 0.003125), origin=(0.0, 0.0))
    crit_deflect_sub = lens_galaxy.deflections_from_grid(grid=new_grid_sub)
    ycaustic_arc_sub = new_grid_sub[:,0] - crit_deflect_sub[:,0]
    xcaustic_arc_sub = new_grid_sub[:,1] - crit_deflect_sub[:,1]
    caustics_arc_sub = np.stack((ycaustic_arc_sub, xcaustic_arc_sub), axis=-1)
    caustic_grid_sub = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=caustics_arc_sub, shape_2d=(1600,1600),
                                                                  pixel_scales=(0.003125, 0.003125), origin=(0, 0))
    xcaustic_sub = caustic_grid_sub[:,1]
    ycaustic_sub = caustic_grid_sub[:,0]

    ax2.plot(ycaustic_sub, xcaustic_sub, c='g', lw=1.5, zorder=250)
    ax2.plot(ycritical_sub, xcritical_sub, c='r', lw=1.5, zorder=250)

im3 = ax3.imshow(np.log10(kappa_2d_binned), cmap='jet')
colorbar(im3)
#ax3.contour(sub_grid_pix_2d[:,:,1], sub_grid_pix_2d[:,:,0], deflect_detA_sub, levels=(0,), colors='r', linewidths=1.5, zorder=200)
#ax3.contour(sub_grid_pix_2d[:,:,1], sub_grid_pix_2d[:,:,0], deflect_lambda_t_sub, levels=(0,), colors='b', linewidths=2.5, zorder=50)
ax3.set_title('reg grid binned from sub grid')

critical_curves_indices_binned = measure.find_contours(detA_2d_binned, 0)
Ncrit_binned = len(critical_curves_indices_binned)
critical_curves_binned = []
caustics_binned = []

for jj in np.arange(Ncrit_binned):
    critical_curves_binned.append(critical_curves_indices_binned[jj])
    xcritical_binned, ycritical_binned = critical_curves_binned[jj].T
    pixel_coord_binned = np.stack((xcritical_binned, ycritical_binned), axis=-1)
    new_grid_binned = grid_util.grid_pixels_1d_to_grid_arcsec_1d(grid_pixels_1d=pixel_coord_binned, shape_2d=(100,100),
                                                           pixel_scales=(0.05, 0.05), origin=(0.0, 0.0))
    crit_deflect_binned = lens_galaxy.deflections_from_grid(grid=new_grid_binned)
    ycaustic_arc_binned = new_grid_binned[:,0] - crit_deflect_binned[:,0]
    xcaustic_arc_binned = new_grid_binned[:,1] - crit_deflect_binned[:,1]
    caustics_arc_binned = np.stack((ycaustic_arc_binned, xcaustic_arc_binned), axis=-1)
    caustic_grid_binned = grid_util.grid_arcsec_1d_to_grid_pixels_1d(grid_arcsec_1d=caustics_arc_binned, shape_2d=(100,100),
                                                                  pixel_scales=(0.05, 0.05), origin=(0, 0))
    xcaustic_binned = caustic_grid_binned[:,1]
    ycaustic_binned = caustic_grid_binned[:,0]

    ax3.plot(ycaustic_binned, xcaustic_binned, c='g', lw=1.5, zorder=250)
    ax3.plot(ycritical_binned, xcritical_binned, c='r', lw=1.5, zorder=250)

#plt.show()
