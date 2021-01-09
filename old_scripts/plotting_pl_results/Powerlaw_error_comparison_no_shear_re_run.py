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
          '/pipeline_tag__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/'\
          'phase_1__lens_bulge_disk_power_law__source_inversion/'\
          'phase_tag__sub_2__pos_0.50/model.results'
fig_path = '/Users/dgmt59/Documents/Plots/error_analysis/PL/david_v_david_'

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


## creating variables for loading results
list_ = []
lens = []
SLACS_image = []
list_2 = []
lens_2 = []
SLACS_image_2 = []

slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name
slacs = slacs.drop(['slacs0959+5100',  'slacs0959+4416', 'slacs1420+6019'], axis=0)

for i in range(len(lens_name)):
    full_data_path = Path(data_path + '_final/F814W/' + lens_name[i] + pipeline)
    full_data_path_2 = Path(data_path + '_fresh/F814W/' + lens_name[i] + pipeline)
    if full_data_path.is_file() and full_data_path_2.is_file():
        data = pd.read_csv(full_data_path, sep='\s+', header=None, skiprows=19, nrows=7, ).set_index(0)
        del data.index.name
        data[2] = data[2].str.strip('(,').astype(float)
        data[3] = data[3].str.strip(')').astype(float)
        data.columns = ['param', '-error', '+error']
        list_.append(data)
        lens.append(lens_name[i])
        results = pd.concat(list_, keys=lens)
        ## shu results
        data_2 = pd.read_csv(full_data_path_2, sep='\s+', header=None, skiprows=19, nrows=7, ).set_index(0)
        del data_2.index.name
        data_2[2] = data_2[2].str.strip('(,').astype(float)
        data_2[3] = data_2[3].str.strip(')').astype(float)
        data_2.columns = ['param', '-error', '+error']
        list_2.append(data_2)
        lens_2.append(lens_name[i])
        results_2 = pd.concat(list_2, keys=lens_2)    
    else:
        slacs = slacs.drop([lens_name[i]], axis=0)


error_low = []
error_hi = []
error_low_2 = []
error_hi_2 = []
error_low_frac = []
error_hi_frac = []
error_low_2_frac = []
error_hi_2_frac = []
error_low_q = []
error_hi_q = []
error_low_q_2 = []
error_hi_q_2 = []
error_low_PA = []
error_hi_PA = []
error_low_PA_2 = []
error_hi_PA_2 = []
error_low_R = []
error_hi_R = []
error_low_R_2 = []
error_hi_R_2 = []
error_low_Radius = []
error_hi_Radius = []
error_low_Radius_2 = []
error_hi_Radius_2 = []

q = []
PA = []
q_2 = []
PA_2 = []
gamma = []
gamma_2 = []
r = []
r_2 = []


for i in range(len(lens)):
    ## david results
    lower_error = results.loc[lens[i]]['param']['slope'] - results.loc[lens[i]]['-error']['slope']
    upper_error = results.loc[lens[i]]['+error']['slope'] - results.loc[lens[i]]['param']['slope']
    error_low.append(lower_error)
    error_hi.append(upper_error)
    ## shu results
    lower_error_2 = results_2.loc[lens_2[i]]['param']['slope'] - results_2.loc[lens_2[i]]['-error']['slope']
    upper_error_2 = results_2.loc[lens_2[i]]['+error']['slope'] - results_2.loc[lens_2[i]]['param']['slope']
    error_low_2.append(lower_error_2)
    error_hi_2.append(upper_error_2)

    ## david results
    lower_error_frac = (results.loc[lens[i]]['param']['slope'] - results.loc[lens[i]]['-error']['slope'])/results.loc[lens[i]]['param']['slope']
    upper_error_frac = (results.loc[lens[i]]['+error']['slope'] - results.loc[lens[i]]['param']['slope'])/results.loc[lens[i]]['param']['slope']
    error_low_frac.append(lower_error_frac)
    error_hi_frac.append(upper_error_frac)
    ## shu results
    lower_error_2_frac = (results_2.loc[lens_2[i]]['param']['slope'] - results_2.loc[lens_2[i]]['-error']['slope'])/results_2.loc[lens_2[i]]['param']['slope']
    upper_error_2_frac = (results_2.loc[lens_2[i]]['+error']['slope'] - results_2.loc[lens_2[i]]['param']['slope'])/results_2.loc[lens_2[i]]['param']['slope']
    error_low_2_frac.append(lower_error_2_frac)
    error_hi_2_frac.append(upper_error_2_frac)

    ## david results
    lower_error_q = results.loc[lens[i]]['param']['axis_ratio'] - results.loc[lens[i]]['-error']['axis_ratio']
    upper_error_q = results.loc[lens[i]]['+error']['axis_ratio'] - results.loc[lens[i]]['param']['axis_ratio']
    error_low_q.append(lower_error_q)
    error_hi_q.append(upper_error_q)
    ## shu results
    lower_error_q_2 = results_2.loc[lens_2[i]]['param']['axis_ratio'] - results_2.loc[lens_2[i]]['-error']['axis_ratio']
    upper_error_q_2 = results_2.loc[lens_2[i]]['+error']['axis_ratio'] - results_2.loc[lens_2[i]]['param']['axis_ratio']
    error_low_q_2.append(lower_error_q_2)
    error_hi_q_2.append(upper_error_q_2)

    ## david results
    lower_error_PA = results.loc[lens[i]]['param']['phi'] - results.loc[lens[i]]['-error']['phi']
    upper_error_PA = results.loc[lens[i]]['+error']['phi'] - results.loc[lens[i]]['param']['phi']
    error_low_PA.append(lower_error_PA)
    error_hi_PA.append(upper_error_PA)
    ## shu results
    lower_error_PA_2 = results_2.loc[lens_2[i]]['param']['phi'] - results_2.loc[lens_2[i]]['-error']['phi']
    upper_error_PA_2 = results_2.loc[lens_2[i]]['+error']['phi'] - results_2.loc[lens_2[i]]['param']['phi']
    error_low_PA_2.append(lower_error_PA_2)
    error_hi_PA_2.append(upper_error_PA_2)


    axis_ratio = results.loc[lens[i]]['param']['axis_ratio']
    phi = results.loc[lens[i]]['param']['phi']
    axis_ratio_2 = results_2.loc[lens_2[i]]['param']['axis_ratio']
    phi_2 = results_2.loc[lens_2[i]]['param']['phi']
    pl = results.loc[lens[i]]['param']['slope']
    pl_2 = results_2.loc[lens_2[i]]['param']['slope']
    radius = results.loc[lens[i]]['param']['einstein_radius']
    radius_2 = results_2.loc[lens_2[i]]['param']['einstein_radius']

    ## david results
    lower_error_R = (results.loc[lens[i]]['param']['einstein_radius'] - results.loc[lens[i]]['-error']['einstein_radius'])/radius
    upper_error_R = (results.loc[lens[i]]['+error']['einstein_radius'] - results.loc[lens[i]]['param']['einstein_radius'])/radius
    error_low_R.append(lower_error_R)
    error_hi_R.append(upper_error_R)
    ## shu results
    lower_error_R_2 = (results_2.loc[lens_2[i]]['param']['einstein_radius'] - results_2.loc[lens_2[i]]['-error']['einstein_radius'])/radius_2
    upper_error_R_2 = (results_2.loc[lens_2[i]]['+error']['einstein_radius'] - results_2.loc[lens_2[i]]['param']['einstein_radius'])/radius_2
    error_low_R_2.append(lower_error_R_2)
    error_hi_R_2.append(upper_error_R_2)

    ## david results
    lower_error_Radius = (results.loc[lens[i]]['param']['einstein_radius'] - results.loc[lens[i]]['-error'][
        'einstein_radius'])
    upper_error_Radius = (results.loc[lens[i]]['+error']['einstein_radius'] - results.loc[lens[i]]['param'][
        'einstein_radius'])
    error_low_Radius.append(lower_error_Radius)
    error_hi_Radius.append(upper_error_Radius)
    ## shu results
    lower_error_Radius_2 = (results_2.loc[lens_2[i]]['param']['einstein_radius'] -
                         results_2.loc[lens_2[i]]['-error']['einstein_radius'])
    upper_error_R_adiusshu = (results_2.loc[lens_2[i]]['+error']['einstein_radius'] -
                         results_2.loc[lens_2[i]]['param']['einstein_radius'])
    error_low_Radius_2.append(lower_error_Radius_2)
    error_hi_Radius_2.append(upper_error_R_adiusshu)

    q.append(axis_ratio)
    PA.append(phi)
    q_2.append(axis_ratio_2)
    PA_2.append(phi_2)
    gamma.append(pl)
    gamma_2.append(pl_2)
    r.append(radius)
    r_2.append(radius_2)

err = np.array([error_low, error_hi])
err_2 = np.array([error_low_2, error_hi_2])
err_frac = np.array([error_low_frac, error_hi_frac])
err_2_frac = np.array([error_low_2_frac, error_hi_2_frac])
err_q = np.array([error_low_q, error_hi_q])
err_q_2 = np.array([error_low_q_2, error_hi_q_2])
err_PA = np.array([error_low_PA, error_hi_PA])
err_PA_2 = np.array([error_low_PA_2, error_hi_PA_2])
err_R_frac = np.array([error_low_R, error_hi_R])
err_R_2_frac = np.array([error_low_R_2, error_hi_R_2])
err_R= np.array([error_low_Radius, error_hi_Radius])
err_R_2 = np.array([error_low_Radius_2, error_hi_Radius_2])

q_diff = np.array(q) - np.array(q_2)
PA_diff = np.abs(np.array(PA) - np.array(PA_2))
gamma_diff = np.array(gamma) - np.array(gamma_2)
gamma_frac_diff = (np.array(gamma) - np.array(gamma_2))/gamma_2
r_frac_diff = (np.array(r)-np.array(r_2))/np.array(r_2)

RMS_gamma = np.sqrt(np.mean(np.square(gamma_diff)))
RMS_gamma_frac = np.sqrt(np.mean(np.square(gamma_frac_diff)))
RMS_PA = np.sqrt(np.mean(np.square(PA_diff)))
RMS_q = np.sqrt(np.mean(np.square(q_diff)))
RMS_r = np.sqrt(np.mean(np.square(r_frac_diff)))


print('formal stat error slope', np.median(np.array([err, err_2])))
print('formal stat frac error slope', np.median(np.array([err_frac, err_2_frac])))
print('formal stat error q', np.median(np.array([err_q, err_q_2])))
print('formal stat error PA', np.median(np.array([err_PA, err_PA_2])))
print('formal stat error R', np.median(np.array([err_R, err_R_2])))
print('SLACS error', np.median(slacs['gamma_err']))
print('RMS gamma', RMS_gamma)
print('RMS gamma frac', RMS_gamma_frac)
print('RMS q', RMS_q)
print('RMS PA', RMS_PA)
print('RMS R', RMS_r)

fig1, ax = plt.subplots(figsize=(8,8))
for i in range(len(lens)):
   ax.scatter(results_2.loc[lens_2[i]]['param']['slope'], results.loc[lens[i]]['param']['slope'],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
   ax.errorbar(results_2.loc[lens_2[i]]['param']['slope'], results.loc[lens[i]]['param']['slope'],
               yerr=np.array([err[:,i]]).T, xerr=np.array([err_2[:,i]]).T, color=slacs['colour'][i],
               elinewidth=1, fmt='none', capsize=3, label=None)

plt.xlabel(r'$\gamma_{re-run}$', size=14)
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
plt.savefig(fig_path + 'slopes' , bbox_inches='tight', dpi=300)

fig2, ax = plt.subplots(figsize=(8,8))
for i in range(len(lens)):
   ax.scatter(r_2[i], r[i],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
   ax.errorbar(r_2[i], r[i],
               yerr=np.array([err_R[:,i]]).T, xerr=np.array([err_R_2[:,i]]).T, color=slacs['colour'][i],
               elinewidth=1, fmt='none', capsize=3, label=None)

plt.xlabel(r'$R_{re-run}$', size=14)
plt.ylabel(r'$R_{David}$', size=14)
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
plt.savefig(fig_path + 'radii', bbox_inches='tight', dpi=300)



fig3, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(7,5))
for i in range(len(lens)):
    ax1.scatter(q[i], gamma_diff[i],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
    ax2.scatter(q[i], r_frac_diff[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
    ax3.scatter(q[i], q_diff[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
    ax4.scatter(q[i], PA_diff[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
ax1.set_ylabel(r'$\Delta \gamma$', size=14)
ax1.axhline(y=0, color='k', linestyle='--')
ax2.set_ylabel(r'$\frac{\Delta R}{R}$', size=14)
ax2.axhline(y=0, color='k', linestyle='--')
ax3.set_ylabel(r'$\Delta q$', size=14)
ax3.axhline(y=0, color='k', linestyle='--')
ax4.set_ylabel(r'$|\Delta \phi|$', size=14)
ax4.set_xlabel(r'$q^{AutoLens}$', size=12)
#box = fig4.position()
#fig4.position([box.x0, box.y0, box.width * 0.8, box.height])
legend = fig3.legend(loc='center right', title= 'Lens Name')
plt.tight_layout()
plt.subplots_adjust(right=0.7)
plt.savefig(fig_path + 'gamma_R_q_PA', bbox_inches='tight', dpi=300)


plt.show()

