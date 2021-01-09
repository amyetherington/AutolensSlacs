import numpy as np
import matplotlib.pyplot as plt

def sigma_gamma_isothermal(S, theta_r):
    return np.divide(2*np.sqrt(theta_r), S*(1-theta_r))

positions_ratio = np.arange(0, 1, 0.001)

signal_to_noise = 30

uncertainty = sigma_gamma_isothermal(S=signal_to_noise, theta_r=positions_ratio)

plt.plot(positions_ratio, uncertainty)
plt.yscale('log')
plt.show()
