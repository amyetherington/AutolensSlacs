from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import astropy.cosmology as cosmo
import astropy.units as u
from skimage import measure


import os

for i in np.arange(10):
    plt.close()


def alpha_map_fourier(kappa, x, padFac):
    xlen, ylen = kappa.shape
    xpad, ypad = xlen * padFac, ylen * padFac
    Lx = x[-1, 0, 0] - x[0, 0, 0]
    Ly = x[0, -1, 1] - x[0, 0, 1]
    # round to power of 2 to speed up FFT
    xpad = np.int(2 ** (np.ceil(np.log2(xpad))))
    ypad = np.int(2 ** (np.ceil(np.log2(ypad))))
    kappa_ft = np.fft.fft2(kappa, s=[xpad, ypad])
    Lxpad, Lypad = Lx * xpad / xlen, Ly * ypad / ylen
    # make a k-space grid
    kxgrid, kygrid = np.meshgrid(np.fft.fftfreq(xpad), np.fft.fftfreq(ypad), indexing='ij')
    kxgrid *= 2 * np.pi * xpad / Lxpad
    kygrid *= 2 * np.pi * ypad / Lypad
    alphaX_kfac = 2j * kxgrid / (kxgrid ** 2 + kygrid ** 2)
    alphaY_kfac = 2j * kygrid / (kxgrid ** 2 + kygrid ** 2)
    # [0,0] component mucked up by dividing by k^2, and subject to mass-sheet degeneracy
    alphaX_kfac[0, 0], alphaY_kfac[0, 0] = 0, 0
    alphaX_ft = alphaX_kfac * kappa_ft
    alphaY_ft = alphaY_kfac * kappa_ft
    alphaX = np.fft.ifft2(alphaX_ft)[:xlen, :ylen]
    alphaY = np.fft.ifft2(alphaY_ft)[:xlen, :ylen]
    # return as Nx x Ny x 2 array
    alpha = np.zeros(x.shape)
    alpha[:, :, 0], alpha[:, :, 1] = alphaX, alphaY
    return -alpha


path = '{}/'.format(os.path.dirname(os.path.realpath(__file__)))
kappa_data = np.load(path + '/Group_00006.npz')
kappa = kappa_data['kappa']
xphys, yphys = kappa_data['xs'] * u.Mpc, kappa_data['ys'] * u.Mpc  # physical dimensions in lens plane
zLens, zSource = kappa_data['zLens'], kappa_data['zSource']
hfov = kappa_data['xs'].max()  # half the field of view in Mpc
xi0 = 0.01 * u.Mpc  # unit length in lens plane
Lrays = 1.0 * u.Mpc  # area over which to calculate magnifications, etc.
Nrays = 1024

print("x, y", xphys, yphys)


# Dimensionless lens plane coordinates
xs, ys = xphys / xi0, yphys / xi0


# Array of 2D dimensionless coordinates
xsgrid, ysgrid = np.meshgrid(xs, ys, indexing='ij')
x = np.zeros((len(xs), len(ys), 2))
x[:, :, 0] = xsgrid
x[:, :, 1] = ysgrid


# alpha should be deflection angles on a regular grid (calculated however)
alpha = alpha_map_fourier(kappa, x, padFac=4.0)
xtest = x

alphaxinterp = interpolate.RectBivariateSpline(xtest[:, 0, 0], xtest[0, :, 1], alpha[:, :, 0])
alphayinterp = interpolate.RectBivariateSpline(xtest[:, 0, 0], xtest[0, :, 1], alpha[:, :, 1])

# Higher resolution grid onto which the deflection angles are interpolated
xrays, yrays = np.linspace(-0.5 * (Lrays / xi0).to(''), 0.5 * (Lrays / xi0).to(''), Nrays), np.linspace(
    -0.5 * (Lrays / xi0).to(''), 0.5 * (Lrays / xi0).to(''), Nrays)
xraysgrid, yraysgrid = np.meshgrid(xrays, yrays, indexing='ij')
dxray, dyray = xrays[1] - xrays[0], yrays[1] - yrays[0]
hfov_hires = ((xrays * xi0).max()).value  # half the field of view in Mpc

alphax = alphaxinterp(xrays, yrays)
alphay = alphayinterp(xrays, yrays)




# Jacobian matrix elements
A11 = 1 - np.gradient(alphax, dxray, axis=0)
A12 = - np.gradient(alphax, dyray, axis=1)
A21 = - np.gradient(alphay, dxray, axis=0)
A22 = 1 - np.gradient(alphay, dyray, axis=1)

detA = A11 * A22 - A12 * A21
mag = 1 / detA

# Convergence and shear from Jacobian matrix
ka = 1 - 0.5 * (A11 + A22)
ga1 = 0.5 * (A22 - A11)
ga2 = -0.5 * (A12 + A21)
ga = (ga1 ** 2 + ga2 ** 2) ** 0.5

# magnification = 1 / (lambda_t x lambda_r), with tangential critical curves lambda_t=0
lambda_t = 1 - ka - ga
lambda_r = 1 - ka + ga


############################
# Make a plot
############################

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(131)

cax = ax.imshow(np.log10(kappa).T, extent=[-hfov, hfov, -hfov, hfov], vmin=np.log10(0.13), vmax=np.log10(8),
                cmap='jet_r', origin='lower')
# Plot critical curves
ax.contour(xraysgrid * xi0, yraysgrid * xi0, detA, levels=(0,), colors='r', linewidths=1.5, zorder=200)
ax.contour(xraysgrid * xi0, yraysgrid * xi0, lambda_t, levels=(0,), colors='k', linewidths=2.5, zorder=50)

critical_curve_indices = measure.find_contours(detA, 0)

Ncrit = len(critical_curve_indices)
critical_curves = []
caustics = []
for jj in np.arange(Ncrit):
    # find_contours is in index number coordinates, convert to dimensionless lens plane coordinates
    critical_curves.append(((critical_curve_indices[jj] / Nrays) - 0.5) * Lrays / xi0)
    xcritical, ycritical = critical_curves[jj].T
    xcaustic = xcritical - alphaxinterp(xcritical, ycritical, grid=False)
    ycaustic = ycritical - alphayinterp(xcritical, ycritical, grid=False)
    caustic = np.zeros((len(xcaustic), len(xcaustic), 2))

    caustic[:, :, 0] = xcaustic
    caustic[:, :, 1] = ycaustic
    caustics.append(caustic)





    ax.plot(xcaustic * xi0, ycaustic * xi0, c='g', lw=1.5, zorder=250)
    ax.plot(xcaustic * xi0, ycaustic * xi0, c='k', lw=2, zorder=100)


fig.colorbar(cax, ax=ax)

ax2 = fig.add_subplot(132)
cax2 = ax2.imshow(np.log10(np.abs(mag)).T, extent=[-hfov_hires, hfov_hires, -hfov_hires, hfov_hires], vmin=np.log10(0.1),
           vmax=np.log10(100), cmap='viridis', origin='lower')
fig.colorbar(cax2, ax=ax2)

ax3 = fig.add_subplot(133)
cax3 = ax3.imshow(np.log10(ka).T, extent=[-hfov_hires, hfov_hires, -hfov_hires, hfov_hires], vmin=np.log10(0.13),
           vmax=np.log10(8), cmap='jet_r', origin='lower')
ax3.contour(xraysgrid * xi0, yraysgrid * xi0, detA, levels=(0,), colors='r', linewidths=1.5, zorder=200)
ax3.contour(xraysgrid * xi0, yraysgrid * xi0, lambda_t, levels=(0,), colors='k', linewidths=2.5, zorder=50)
fig.colorbar(cax3, ax=ax3)

ax.set_title('convergence')
ax.set_xlim(-0.2, 0.2)
ax.set_ylim(-0.2, 0.2)

ax2.set_title('magnification')
ax2.set_xlim(-0.2, 0.2)
ax2.set_ylim(-0.2, 0.2)

ax3.set_title('convergence from deflection angles')
ax3.set_xlim(-0.2, 0.2)
ax3.set_ylim(-0.2, 0.2)

fig.subplots_adjust(left=0.05, bottom=0.05, right=0.99, top=0.97,
                    wspace=None, hspace=None)
fig.savefig('example.pdf')

plt.show()
