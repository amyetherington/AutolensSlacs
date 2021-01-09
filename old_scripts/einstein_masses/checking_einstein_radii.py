from autofit import conf

import os

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))

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

from autolens.data.instrument import ccd
from autolens.lens import lens_data as ld

import matplotlib.pyplot as plt
from pathlib import Path

M_o = 1.989e30
cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)

## setting up paths to shear and no shear results (sersic source)
slacs_path = '{}/../../data/slacs_data_table.xlsx'.format(os.path.dirname(os.path.realpath(__file__)))

## loading table of slacs parameters (deleting two discarded lenses)
slacs = pd.read_excel(slacs_path, index_col=0)


d_A = cosmo.angular_diameter_distance(slacs['z_lens'])
theta_radian = slacs['b_SIE'] * np.pi / 180 / 3600
distance_kpc = d_A * theta_radian * 1000

b_rad = slacs['b_kpc']/(1000*d_A.value)
b_SIE_from_kpc = b_rad*180*3600/(np.pi)

print(b_SIE_from_kpc-slacs['b_SIE'])
