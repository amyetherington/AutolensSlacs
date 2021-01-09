import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from autolens.model.profiles import mass_profiles as mp
from autolens.array import grids
from autolens.model.galaxy import galaxy as g
from autolens.plotters import array_plotters
from scipy import ndimage
import scipy.ndimage.filters

reg_grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=1)
reg_grid_2d = reg_grid.grid_2d_from_grid_1d(grid_1d=reg_grid)

grid = al.Grid.uniform(shape_2d=(1,1), pixel_scales=1.0, sub_size=1)
grid = grid + 5.0

lens_galaxy = al.Galaxy(mass=al.mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0,
                                                   axis_ratio=0.5, phi=0.0), redshift=0.5)
lens_galaxy_Kormann = al.Galaxy(mass=al.mp.EllipticalIsothermalKormann(centre=(0.0, 0.0), einstein_radius=2.0,
                                                   axis_ratio=0.5, phi=0.0), redshift=0.5)

kappa = lens_galaxy.convergence_from_grid(grid=reg_grid, return_in_2d=True)
alpha_Keeton = lens_galaxy.deflections_from_grid(grid=reg_grid, return_in_2d=True)

kappa_Kormann = lens_galaxy_Kormann.convergence_from_grid(grid=reg_grid, return_in_2d=True, return_binned=True)
kappa_deflections_Kormann = lens_galaxy_Kormann.convergence_via_jacobian_from_grid(grid=reg_grid, return_in_2d=True)
#array_plotters.plot_array(array=kappa_Kormann, norm_max=1.2)



psi_Kormann = lens_galaxy_Kormann.potential_from_grid(grid=reg_grid, return_in_2d=True)
alpha_Kormann = lens_galaxy_Kormann.deflections_from_grid(grid=reg_grid, return_in_2d=True)
#alphay_Kormann = alpha_Kormann[:, 1]
#alphax_Kormann = alpha_Kormann[:, 0]

psi_Keeton = lens_galaxy.potential_from_grid(grid=reg_grid, return_in_2d=True)

alpha_psi_Kormann = lens_galaxy_Kormann.deflections_via_potential_from_grid(grid=reg_grid, return_in_2d=True)
#alphay_psi_Kormann = alpha_psi_Kormann[:, 1]
#alphax_psi_Kormann = alpha_psi_Kormann[:, 0]

# pick a value for the lens axis ~[0,1)
axis_ratio = 0.4

#AutoLens/ Keeton's equations for calcualting angles
def deflections_MD(x, y, q):
    x_component = np.arctan((np.sqrt(1 - q**2)*x)/(np.sqrt((x**2)*(q**2) + y**2)))
    y_component = np.arctanh((np.sqrt(1 - q**2)*y)/(np.sqrt((x**2)*(q**2) + y**2)))

    return x_component, y_component

#SLACS/ Kormann's equations for calculating deflection angles
def deflections_P(x, y, q):
    x_component = np.arcsinh((np.sqrt(1 - q ** 2) * x) / (q * np.sqrt((q ** 2) * (x ** 2) + y ** 2)))
    y_component = np.arcsin((np.sqrt(1 - q ** 2) * y) / (np.sqrt((x ** 2) * (q ** 2) + y ** 2)))
    return x_component, y_component

def convergence_MD(x, y, q):
    radii = np.sqrt(np.add(np.square(x), np.square(np.divide(y, q))))
    surface_density = radii**-1
    return surface_density

def convergence_P(x, y, q):
    return np.sqrt(q)/(2*q*np.sqrt(x**2+(y**2/q**2)))

def elliptical_P(x, y, q):
    return np.sqrt(x ** 2 * q ** 2 + y ** 2)

def potential_MD(x, y, alpha):
    return x*alpha[0] + y*alpha[1]

def potential_P(x, y, q):
    f_prime = np.sqrt(1 - q**2)
    sin_phi = x / np.sqrt(q**2*y**2+x**2)
    cos_phi = y / np.sqrt(q**2*y**2+x**2)
    return (np.sqrt(q)/f_prime)*(x*np.arcsin(f_prime*sin_phi)+y*np.arcsinh((f_prime/q)*cos_phi))

def convergence__grad_alpha(alpha, grid):
    grad_alpha_x = 1-np.gradient(alpha[0], grid[0, :, 1], axis=0)
    grad_alpha_y = 1-np.gradient(alpha[1], grid[:, 0, 0], axis=1)

    return 1-0.5*(grad_alpha_x+grad_alpha_y)

def convergence__for_grad_psi_deflections(alpha, grid):
    a_11 = 1-np.gradient(alpha[:,:,1], grid[0, :, 1], axis=1)
    a_22 = 1-np.gradient(alpha[:,:,0], grid[:, 0, 0], axis=0)

    return 1-0.5*(a_11+a_22)

def convergence_laplace(psi):
    return scipy.ndimage.filters.laplace(psi)

alpha_MD = deflections_MD(x=reg_grid_2d[0,:,1,None], y=reg_grid_2d[None,:,1,0], q=axis_ratio)
alpha_P = deflections_P(x=reg_grid_2d[0,:,1,None], y=reg_grid_2d[None,:,1,0], q=axis_ratio)
kappa_MD = convergence_MD(x=reg_grid_2d[0,:,1,None], y=reg_grid_2d[None,:,1,0], q=axis_ratio)
kappa_P = convergence_P(x=reg_grid_2d[0,:,1,None], y=reg_grid_2d[None,:,1,0], q=axis_ratio)
psi_P =potential_P(x=reg_grid_2d[0,:,1,None], y=reg_grid_2d[None,:,1,0], q=axis_ratio)
psi_MD = potential_MD(x=reg_grid_2d[0,:,1,None], y=reg_grid_2d[None,:,1,0], alpha=alpha_MD)
convergence_from_deflection_angles = convergence__grad_alpha(alpha=alpha_P, grid=reg_grid_2d)
convergence_from_laplace = convergence_laplace(psi=psi_P)
psi_elliptical_P = elliptical_P(x=reg_grid_2d[0,:,1,None], y=reg_grid_2d[None,:,1,0], q=axis_ratio)

alpha_x_P = np.gradient(psi_elliptical_P, reg_grid_2d[0,:,1], axis=0)
alpha_y_P = np.gradient(psi_elliptical_P, reg_grid_2d[:,0,0], axis=1)
alpha_elliptical_P = alpha_x_P, alpha_y_P
convergence_from_elliptical_deflection_angles = convergence__grad_alpha(alpha=alpha_elliptical_P, grid=reg_grid_2d)

alpha_x = np.gradient(psi_P, reg_grid_2d[0,:,1], axis=0)
alpha_y = np.gradient(psi_P, reg_grid_2d[:,0,0], axis=1)
alpha = alpha_x, alpha_y





convergence_from_grad_deflection_angles = convergence__grad_alpha(alpha=alpha, grid=reg_grid_2d)

Kappa_from_deflections_Kormann = convergence__for_grad_psi_deflections(alpha=alpha_psi_Kormann, grid=reg_grid_2d)

grad_alpha_x = 1-np.gradient(alpha_P[0], reg_grid_2d[0, :, 1], axis=0)
grad_alpha_y = 1-np.gradient(alpha_P[1], reg_grid_2d[:, 0, 0], axis=1)
convergence_from_analytic_deflection_angles = 1-0.5*(grad_alpha_x+grad_alpha_y)

def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

fig1, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(alpha_MD[0], cmap='jet')
ax1.set_title('Keeton (x component)')
colorbar(im1)
im2 = ax2.imshow(alpha_P[0], cmap='jet')
ax2.set_title('Kormann (x component)')
colorbar(im2)
plt.close()

fig2, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(alpha_MD[1], cmap='jet')
ax1.set_title('Keeton (y component)')
colorbar(im1)
im2 = ax2.imshow(alpha_P[1], cmap='jet')
ax2.set_title('Kormann (y component)')
colorbar(im2)
plt.close()

fig3, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(alpha_MD[0]-alpha_P[0], cmap='jet')
ax1.set_title('Difference (x component)')
colorbar(im1)
im2 = ax2.imshow(alpha_MD[1]-alpha_P[1], cmap='jet')
ax2.set_title('Difference (y component)')
colorbar(im2)
plt.close()

fig5, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(np.log10(kappa_MD), cmap='jet')
ax1.set_title('Convergence (Keeton)')
colorbar(im1)
im2 = ax2.imshow(np.log10(kappa_P), cmap='jet')
ax2.set_title('Convergence (Kormann)')
colorbar(im2)
plt.close()

fig6, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(kappa_MD, cmap='jet')
ax1.set_title('Convergence')
colorbar(im1)
im2 = ax2.imshow(kappa, cmap='jet')
ax2.set_title('Convergence (autolens)')
colorbar(im2)
plt.close()

fig7, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12,4))
im1 = ax1.imshow(np.log10(psi_MD), cmap='jet')
ax1.set_title('Potential (Keeton)')
colorbar(im1)
im2 = ax2.imshow(np.log10(psi_P), cmap='jet')
ax2.set_title('Potential (Kormann)')
colorbar(im2)
im3 = ax3.imshow(np.log10(psi_elliptical_P), cmap='jet')
ax3.set_title('Potential (elliptical)')
colorbar(im3)
#plt.close()

fig9, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(alpha_P[0], cmap='jet')
ax1.set_title('Analytic (x component)')
colorbar(im1)
im2 = ax2.imshow(alpha_x, cmap='jet')
ax2.set_title('Grad Psi (x component)')
colorbar(im2)
plt.close()

fig10, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(alpha_P[1], cmap='jet')
ax1.set_title('Analytic (y component)')
colorbar(im1)
im2 = ax2.imshow(alpha_y, cmap='jet')
ax2.set_title('Grad Psi (y component)')
colorbar(im2)
plt.close()

fig11, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12,4))
im1 = ax1.imshow(np.log10(kappa_MD), cmap='jet')
ax1.set_title('Convergence (Keeton)')
colorbar(im1)
im2 = ax2.imshow(np.log10(convergence_from_deflection_angles), cmap='jet')
ax2.set_title('Convergence from analytic deflection angles')
colorbar(im2)
im3 = ax3.imshow(np.log10(convergence_from_elliptical_deflection_angles), cmap='jet')
ax3.set_title('Convergence from grad psi deflection angles')
colorbar(im3)
plt.close()

fig12, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(convergence_from_grad_deflection_angles, cmap='jet')
ax1.set_title('Convergence from grad psi')
colorbar(im1)
im2 = ax2.imshow(convergence_from_analytic_deflection_angles, cmap='jet')
ax2.set_title('Convergence from analytic deflection angles')
colorbar(im2)
plt.close()

fig13, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(alpha_P[0], cmap='jet')
ax1.set_title('Analytic (x component)')
colorbar(im1)
im2 = ax2.imshow(alpha_elliptical_P[0], cmap='jet')
ax2.set_title('Grad Psi (x component)')
colorbar(im2)
plt.close()

fig14, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(alpha_P[1], cmap='jet')
ax1.set_title('Analytic (y component)')
colorbar(im1)
im2 = ax2.imshow(alpha_elliptical_P[1], cmap='jet')
ax2.set_title('Grad Psi (y component)')
colorbar(im2)
plt.close()

fig15, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(np.log10(kappa_deflections_Kormann), cmap='jet')
ax1.set_title('Convergence (from deflections)')
colorbar(im1)
im2 = ax2.imshow(np.log10(kappa_Kormann), cmap='jet')
ax2.set_title('Convergence (Kormann equation)')
colorbar(im2)
#plt.close()

fig16, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(alpha_Keeton[:,:,0], cmap='jet')
ax1.set_title('Analytic (y component)')
colorbar(im1)
im2 = ax2.imshow(alpha_Keeton[:,:,1], cmap='jet')
ax2.set_title('Analytic (x component)')
colorbar(im2)


fig16, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(alpha_Kormann[:,:,0], cmap='jet')
ax1.set_title('Analytic (y component)')
colorbar(im1)
im2 = ax2.imshow(alpha_Kormann[:,:,1], cmap='jet')
ax2.set_title('Analytic (x component)')
colorbar(im2)
#plt.close()

fig17, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(alpha_psi_Kormann[:,:,0], cmap='jet')
ax1.set_title('Grad Psi (y component)')
colorbar(im1)
im2 = ax2.imshow(alpha_psi_Kormann[:,:,1], cmap='jet')
ax2.set_title('Grad Psi (x component)')
colorbar(im2)
#plt.close()

fig18, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(np.log10(psi_Kormann), cmap='jet')
ax1.set_title('Potential (Kormann)')
colorbar(im1)
im2 = ax2.imshow(np.log10(psi_Keeton), cmap='jet')
ax2.set_title('Potential(Keeton')
colorbar(im2)

fig19, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(np.log10(kappa_deflections_Kormann), cmap='jet')
ax1.set_title('Convergence (from deflections in autolens)')
colorbar(im1)
im2 = ax2.imshow(np.log10(Kappa_from_deflections_Kormann), cmap='jet')
ax2.set_title('Convergence (from deflections from potential)')
colorbar(im2)
#plt.close()

fig20, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))
im1 = ax1.imshow(np.log10(kappa), cmap='jet')
ax1.set_title('Convergence (from grid)')
colorbar(im1)
im2 = ax2.imshow(np.log10(kappa_), cmap='jet')
ax2.set_title('Convergence (from deflections)')
colorbar(im2)

plt.show()
