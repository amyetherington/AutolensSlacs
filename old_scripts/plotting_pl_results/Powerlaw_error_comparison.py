import os

import pandas as pd
import numpy as np

from astropy import cosmology

import matplotlib.pyplot as plt
from pathlib import Path

M_o = 1.989e30
cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)

lens_name = np.array(['slacs0216-0813',
                      'slacs0252+0039',
                      'slacs0737+3216',
                      'slacs0912+0029',
                      'slacs0959+4410',
                      'slacs1205+4910',
                      'slacs1250+0523',
                      'slacs1402+6321',
                      'slacs1430+4105',
                      'slacs1627+0053',
                      'slacs1630+4520',
                      'slacs2238-0754',
                      'slacs2300+0022',
                      'slacs2303+1422'])

inner_mask_radii = np.array(['0.30', '0.30', '0.30', '0.50', '0.20', '0.30',
                             '0.20', '0.50', '0.50', '0.30', '0.30', '0.30', '0.30', '0.40', '0.30'])

slacs_path = '{}/../../data/slacs_data_table.xlsx'.format(os.path.dirname(os.path.realpath(__file__)))
data_path = '{}/../../../../../output/slacs_fresh/F814W/'.format(os.path.dirname(os.path.realpath(__file__)))
data_path_2 = '{}/../../../../../output/slacs_final/F814W/'.format(os.path.dirname(os.path.realpath(__file__)))
pipeline = '/pipeline_power_law_hyper__lens_bulge_disk_power_law__source_inversion__const'
model = '/pipeline_tag__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/'\
          'phase_1__lens_bulge_disk_power_law__source_inversion/'\
          'phase_tag__sub_2__pos_0.50/model.results'
fig_path = '/Users/dgmt59/Documents/Plots/slacs_final/Power_law_wavelength_comparison/no_shear'


list_ = []
lens=[]
list__2 = []
lens_2 = []

error_low = []
error_hi = []
error_low_2 = []
error_hi_2 = []
error_low_mag = []
error_hi_mag = []
error_low_2_mag = []
error_hi_2_mag = []

slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name
slacs = slacs.drop(['slacs0959+5100',  'slacs0959+4416', 'slacs1420+6019'], axis=0)

for i in range(len(lens_name)):
    full_data_path = Path(data_path + lens_name[i] + pipeline + model)
    full_data_path_2 = Path(data_path_2 + lens_name[i] + pipeline + model )
    if full_data_path_2.is_file() and full_data_path.is_file():
        data = pd.read_csv(full_data_path, sep='\s+', header=None, skiprows=19, nrows=7, ).set_index(0)
        del data.index.name
        data[2] = data[2].str.strip('(,').astype(float)
        data[3] = data[3].str.strip(')').astype(float)
        data.columns = ['param', '-error', '+error']
        list_.append(data)
        lens.append(lens_name[i])
        results = pd.concat(list_, keys=lens)
        ## shear F555W results
        data_2 = pd.read_csv(full_data_path_2, sep='\s+', header=None, skiprows=19, nrows=7, ).set_index(0)
        del data_2.index.name
        data_2[2] = data_2[2].str.strip('(,').astype(float)
        data_2[3] = data_2[3].str.strip(')').astype(float)
        data_2.columns = ['param', '-error', '+error']
        list__2.append(data_2)
        lens_2.append(lens_name[i])
        results_2 = pd.concat(list__2, keys=lens_2)    
    else:
        slacs = slacs.drop([lens_name[i]], axis=0)

for i in range(len(lens)):
    ## shear F814W results
    lower_error = results.loc[lens[i]]['param']['slope'] - results.loc[lens[i]]['-error']['slope']
    upper_error = results.loc[lens[i]]['+error']['slope'] - results.loc[lens[i]]['param']['slope']
    error_low.append(lower_error)
    error_hi.append(upper_error)
    ## shear F555W results
    lower_error_2 = results_2.loc[lens[i]]['param']['slope'] - results_2.loc[lens[i]]['-error']['slope']
    upper_error_2 = results_2.loc[lens[i]]['+error']['slope'] - results_2.loc[lens[i]]['param']['slope']
    error_low_2.append(lower_error_2)
    error_hi_2.append(upper_error_2)
    ## shear F814W results
 #   lower_error_mag = results.loc[lens[i]]['param']['magnitude'] - results.loc[lens[i]]['-error']['magnitude']
 #   upper_error_mag = results.loc[lens[i]]['+error']['magnitude'] - results.loc[lens[i]]['param']['magnitude']
 #   error_low_mag.append(lower_error_mag)
 #   error_hi_mag.append(upper_error_mag)
    ## shear F555W results
 #   lower_error_2_mag = results_2.loc[lens[i]]['param']['magnitude'] - results_2.loc[lens[i]]['-error']['magnitude']
 #   upper_error_2_mag = results_2.loc[lens[i]]['+error']['magnitude'] - results_2.loc[lens[i]]['param']['magnitude']
 #   error_low_2_mag.append(lower_error_2_mag)
 #   error_hi_2_mag.append(upper_error_2_mag)

y_err = np.array([error_low, error_hi])
y_err_2 = np.array([error_low_2, error_hi_2])
#y_err_mag = np.array([error_low_mag, error_hi_mag])
#y_err_2_mag = np.array([error_low_2_mag, error_hi_2_mag])

fig1, ax = plt.subplots(figsize=(8,8))
for i in range(len(lens)):
   ax.scatter(results_2.loc[lens[i]]['param']['slope'], results.loc[lens[i]]['param']['slope'],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
   ax.errorbar(results_2.loc[lens[i]]['param']['slope'], results.loc[lens[i]]['param']['slope'],
               yerr=np.array([y_err[:,i]]).T, xerr=np.array([y_err_2[:,i]]).T, color=slacs['colour'][i],
               elinewidth=1, fmt='none', capsize=3, label=None)

plt.xlabel(r'$\gamma$', size=14)
plt.ylabel(r'$\gamma (re-run)$', size=14)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title= 'Lens Name')
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
     ]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.savefig(fig_path, bbox_inches='tight', dpi=300, transparent=True)

plt.show()

fig2, ax = plt.subplots(figsize=(8,8))
for i in range(len(lens)):
   ax.scatter(results_2.loc[lens[i]]['param']['magnitude'], results.loc[lens[i]]['param']['magnitude'],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
   ax.errorbar(results_2.loc[lens[i]]['param']['magnitude'], results.loc[lens[i]]['param']['magnitude'],
               yerr=np.array([y_err_mag[:,i]]).T, xerr=np.array([y_err_2_mag[:,i]]).T, color=slacs['colour'][i],
               elinewidth=1, fmt='none', capsize=3, label=None)

plt.xlabel(r'$\gamma_{shear}$', size=14)
plt.ylabel(r'\gamma_{shear} (re-run)$', size=14)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title= 'Lens Name')
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
     ]
ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
plt.savefig(fig_path + 'shear_mag', bbox_inches='tight', dpi=300, transparent=True)

plt.show()

