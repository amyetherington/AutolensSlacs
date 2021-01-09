import os

import pandas as pd
import numpy as np

from astropy import cosmology

import matplotlib.pyplot as plt
from pathlib import Path

M_o = 1.989e30
cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)

slacs_path = '{}/../../data/slacs_data_table.xlsx'.format(os.path.dirname(os.path.realpath(__file__)))
data_path = '{}/../../../../../output/slacs'.format(os.path.dirname(os.path.realpath(__file__)))
pipeline = '/pipeline_power_law_hyper__lens_bulge_disk_power_law__source_inversion__const' \
          '/pipeline_tag__with_shear__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/'\
          'phase_1__lens_bulge_disk_power_law__source_inversion/'\
          'phase_tag__sub_2__pos_0.50/model.results'
fig_path = '/Users/dgmt59/Documents/Plots/slacs_final/Power_law_wavelength_comparison/no_shear'

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

lens_name_shu = np.array(['slacs0216-0813',
                      'slacs0252+0039',
                      'slacs0737+3216',
                      'slacs0912+0029',
                      'slacs0959+0410',
                      'slacs1205+4910',
                      'slacs1250+0523',
                      'slacs1402+6321',
                      'slacs1430+4105',
                      'slacs1627-0053',
                      'slacs1630+4520',
                      'slacs2238-0754',
                      'slacs2300+0022',
                      'slacs2303+1422'])

## creating variables for loading results
list_ = []
lens = []
SLACS_image = []
list_shu = []
lens_shu = []
SLACS_image_shu = []

slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name
slacs = slacs.drop(['slacs0959+5100',  'slacs0959+4416', 'slacs1420+6019'], axis=0)

for i in range(len(lens_name)):
    full_data_path = Path(data_path + '_fresh/F814W/' + lens_name[i] + pipeline)
    full_data_path_shu = Path(data_path + '_shu_shu/F814W/' + lens_name_shu[i] + pipeline)
    if full_data_path.is_file() and full_data_path_shu.is_file():
        data = pd.read_csv(full_data_path, sep='\s+', header=None, skiprows=22, nrows=10, ).set_index(0)
        del data.index.name
        data[2] = data[2].str.strip('(,').astype(float)
        data[3] = data[3].str.strip(')').astype(float)
        data.columns = ['param', '-error', '+error']
        list_.append(data)
        lens.append(lens_name[i])
        results = pd.concat(list_, keys=lens)
        ## shu results
        data_shu = pd.read_csv(full_data_path_shu, sep='\s+', header=None, skiprows=22, nrows=10, ).set_index(0)
        del data_shu.index.name
        data_shu[2] = data_shu[2].str.strip('(,').astype(float)
        data_shu[3] = data_shu[3].str.strip(')').astype(float)
        data_shu.columns = ['param', '-error', '+error']
        list_shu.append(data_shu)
        lens_shu.append(lens_name_shu[i])
        results_shu = pd.concat(list_shu, keys=lens_shu)    
    else:
        slacs = slacs.drop([lens_name[i]], axis=0)


error_low = []
error_hi = []
error_low_shu = []
error_hi_shu = []
error_low_mag = []
error_hi_mag = []
error_low_mag_shu = []
error_hi_mag_shu = []

q = []
PA = []
q_shu = []
PA_shu = []
gamma = []
gamma_shu = []
mag = []
mag_shu = []

for i in range(len(lens)):
    ## david results
    lower_error = results.loc[lens[i]]['param']['slope'] - results.loc[lens[i]]['-error']['slope']
    upper_error = results.loc[lens[i]]['+error']['slope'] - results.loc[lens[i]]['param']['slope']
    error_low.append(lower_error)
    error_hi.append(upper_error)
    ## shu results
    lower_error_shu = results_shu.loc[lens_shu[i]]['param']['slope'] - results_shu.loc[lens_shu[i]]['-error']['slope']
    upper_error_shu = results_shu.loc[lens_shu[i]]['+error']['slope'] - results_shu.loc[lens_shu[i]]['param']['slope']
    error_low_shu.append(lower_error_shu)
    error_hi_shu.append(upper_error_shu)

    # david
    lower_error_mag = results.loc[lens[i]]['param']['magnitude'] - results.loc[lens[i]]['-error']['magnitude']
    upper_error_mag = results.loc[lens[i]]['+error']['magnitude'] - results.loc[lens[i]]['param']['magnitude']
    error_low_mag.append(lower_error_mag)
    error_hi_mag.append(upper_error_mag)
    ## shu
    lower_error_mag_shu = results_shu.loc[lens_shu[i]]['param']['magnitude'] - results_shu.loc[lens_shu[i]]['-error']['magnitude']
    upper_error_mag_shu = results_shu.loc[lens_shu[i]]['+error']['magnitude'] - results_shu.loc[lens_shu[i]]['param']['magnitude']
    error_low_mag_shu.append(lower_error_mag_shu)
    error_hi_mag_shu.append(upper_error_mag_shu)

    axis_ratio = results.loc[lens[i]]['param']['axis_ratio']
    phi = results.loc[lens[i]]['param'][2]
    axis_ratio_shu = results_shu.loc[lens_shu[i]]['param']['axis_ratio']
    phi_shu = results_shu.loc[lens_shu[i]]['param'][2]
    pl = results.loc[lens[i]]['param']['slope']
    pl_shu = results_shu.loc[lens_shu[i]]['param']['slope']
    magnitude = results.loc[lens[i]]['param']['magnitude']
    magnitude_shu = results_shu.loc[lens_shu[i]]['param']['magnitude']

    q.append(axis_ratio)
    PA.append(phi)
    q_shu.append(axis_ratio_shu)
    PA_shu.append(phi_shu)
    gamma.append(pl)
    gamma_shu.append(pl_shu)
    mag.append(magnitude)
    mag_shu.append(magnitude_shu)

y_err = np.array([error_low, error_hi])
y_err_shu = np.array([error_low_shu, error_hi_shu])
y_err_mag = np.array([error_low_mag, error_hi_mag])
y_err_mag_shu = np.array([error_low_mag_shu, error_hi_mag_shu])

q_diff = np.abs(np.array(q) - np.array(q_shu))
PA_diff = np.abs(np.array(PA) -np.array(PA_shu))
gamma_diff = np.abs(np.array(gamma) -np.array(gamma_shu))

RMS_gamma = np.sqrt(np.mean(np.square(np.array(gamma)-np.array(gamma_shu))))
print(RMS_gamma)

fig1, ax = plt.subplots(figsize=(8,8))
for i in range(len(lens)):
   ax.scatter(results_shu.loc[lens_shu[i]]['param']['slope'], results.loc[lens[i]]['param']['slope'],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
   ax.errorbar(results_shu.loc[lens_shu[i]]['param']['slope'], results.loc[lens[i]]['param']['slope'],
               yerr=np.array([y_err[:,i]]).T, xerr=np.array([y_err_shu[:,i]]).T, color=slacs['colour'][i],
               elinewidth=1, fmt='none', capsize=3, label=None)

plt.xlabel(r'$\gamma_{Shu}$', size=14)
plt.ylabel(r'$\gamma_{David}$', size=14)
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
plt.savefig(fig_path, bbox_inches='tight', dpi=300)

fig1, ax = plt.subplots(figsize=(8,8))
for i in range(len(lens)):
   ax.scatter(results_shu.loc[lens_shu[i]]['param']['magnitude'], results.loc[lens[i]]['param']['magnitude'],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
   ax.errorbar(results_shu.loc[lens_shu[i]]['param']['magnitude'], results.loc[lens[i]]['param']['magnitude'],
               yerr=np.array([y_err_mag[:,i]]).T, xerr=np.array([y_err_mag_shu[:,i]]).T, color=slacs['colour'][i],
               elinewidth=1, fmt='none', capsize=3, label=None)

plt.xlabel(r'$shear_{Shu}$', size=14)
plt.ylabel(r'$shear_{David}$', size=14)
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
plt.savefig(fig_path, bbox_inches='tight', dpi=300)


fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(7,4))
for i in range(len(lens)):
    ax1.scatter(q[i], gamma_diff[i],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
    ax2.scatter(q[i], q_diff[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
    ax3.scatter(q[i], PA_diff[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
ax1.set_ylabel(r'$|\Delta \gamma|$', size=14)

ax2.set_ylabel(r'$|\Delta q|$', size=14)
ax3.set_ylabel(r'$|\Delta \phi|$', size=14)
ax3.set_xlabel(r'$q^{AutoLens}$', size=12)
#box = fig4.position()
#fig4.position([box.x0, box.y0, box.width * 0.8, box.height])
legend = fig2.legend(loc='center right', title= 'Lens Name')
plt.tight_layout()
plt.subplots_adjust(right=0.7)
plt.savefig(fig_path + 'q_M_PA_SLACS_comparison', bbox_inches='tight', dpi=300)




plt.show()

