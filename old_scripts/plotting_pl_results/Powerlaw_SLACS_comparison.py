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


data_path = '{}/../../../../../output/slacs_shu_shu/'.format(os.path.dirname(os.path.realpath(__file__)))
slacs_path = '{}/../../data/slacs_data_table.xlsx'.format(os.path.dirname(os.path.realpath(__file__)))
wavelength = 'F814W'
pipeline = '/pipeline_power_law_hyper__lens_bulge_disk_power_law__source_inversion__const'
model = '/pipeline_tag__with_shear__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/' \
        'phase_1__lens_bulge_disk_power_law__source_inversion/phase_tag__sub_2__pos_0.50/' \
        'model.results'
fig_path = '/Users/dgmt59/Documents/Plots/slacs_shu/Power_law_SLACS_comparison/shear_' + wavelength

list_ = []
lens=[]
error_low = []
error_hi = []
gamma_frac = []
error_low_mag = []
error_hi_mag = []

slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name
slacs = slacs.drop(['slacs0959+5100',  'slacs0959+4416', 'slacs1420+6019'], axis=0)

## loading in date from model.results file
for i in range(len(lens_name)):
    full_data_path = Path(data_path + wavelength + '/' + lens_name[i] + pipeline + model)
    if full_data_path.is_file():
        data = pd.read_csv(full_data_path, sep='\s+', header=None, skiprows=22, nrows=10,).set_index(0)
        del data.index.name
        data[2] = data[2].str.strip('(,').astype(float)
        data[3] = data[3].str.strip(')').astype(float)
        data.columns=['param', '-error', '+error']
        list_.append(data)
        lens.append(lens_name[i])
        results = pd.concat(list_, keys=lens)
    else:
        slacs = slacs.drop([lens_name[i]], axis=0)

## calculating errors on power law slope
for i in range(len(lens)):
    lower_error = results.loc[lens[i]]['param']['slope']-results.loc[lens[i]]['-error']['slope']
    upper_error = results.loc[lens[i]]['+error']['slope']-results.loc[lens[i]]['param']['slope']
    error_low.append(lower_error)
    error_hi.append(upper_error)

y_err = np.array([error_low, error_hi])

## plotting
fig1, ax = plt.subplots(figsize=(8,8))
for i in range(len(lens)):
   ax.scatter(slacs['gamma'][i], results.loc[lens[i]]['param']['slope'],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
   ax.errorbar(slacs['gamma'][i], results.loc[lens[i]]['param']['slope'], yerr=np.array([y_err[:,i]]).T, xerr=slacs['gamma_err'][i],
               color=slacs['colour'][i], elinewidth=1, fmt='none', capsize=3, label=None)
plt.xlabel(r'$\gamma_{SLACS}$', size=14)
plt.ylabel(r'$\gamma_{AutoLens}$', size=14)
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

R_ratio = slacs['b_kpc']/slacs['R_eff']

for i in range(len(lens)):
    gamma_diff = (results.loc[lens[i]]['param']['slope'] - slacs['gamma'][i])/slacs['gamma'][i]
    gamma_frac.append(gamma_diff)

fig2, ax = plt.subplots(figsize=(8,5))
for i in range(len(lens)):
    ax.scatter(R_ratio[i], results.loc[lens[i]]['param']['slope'], label=slacs.index[i],
                       marker=slacs['marker'][i], color=slacs['colour'][i])
    ax.errorbar(R_ratio[i], results.loc[lens[i]]['param']['slope'], yerr=np.array([y_err[:, i]]).T,
                color=slacs['colour'][i], elinewidth=1, fmt='none', capsize=3, label=None)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Lens Name')
plt.xlabel(r'$R_{Ein}/R_{eff}$', size=20)
plt.ylabel(r'$\gamma$', size=20)
plt.savefig(fig_path + '_R_ratio', bbox_inches='tight', dpi=300)

fig2, ax = plt.subplots(figsize=(8,5))
for i in range(len(lens)):
    ax.scatter(R_ratio[i], gamma_frac[i], label=slacs.index[i],
                       marker=slacs['marker'][i], color=slacs['colour'][i])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Lens Name')
plt.xlabel(r'$R_{Ein}/R_{eff}$', size=20)
plt.ylabel(r'$\frac{\gamma_{AutoLens}-\gamma_{SLACS}}{\gamma_{SLACS}}$', size=20)
plt.savefig(fig_path + '_R_ratio', bbox_inches='tight', dpi=300, transparent=True)

fig3, ax = plt.subplots(figsize=(8,5))
for i in range(len(lens)):
    ax.scatter(slacs['sigma'][i], results.loc[lens[i]]['param']['slope'], label=slacs.index[i],
                       marker=slacs['marker'][i], color=slacs['colour'][i])
    ax.errorbar(slacs['sigma'][i], results.loc[lens[i]]['param']['slope'], yerr=np.array([y_err[:, i]]).T,
                x_err=slacs['sigma_err'][i],
                color=slacs['colour'][i], elinewidth=1, fmt='none', capsize=3, label=None)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Lens Name')
plt.xlabel(r'$\sigma_{e/2}$', size=20)
plt.ylabel(r'$\gamma$', size=20)
plt.savefig(fig_path + '_sigma', bbox_inches='tight', dpi=300, transparent=True)

fig4, ax = plt.subplots(figsize=(8,5))
for i in range(len(lens)):
    ax.scatter(slacs['sigma'][i], gamma_frac[i], label=slacs.index[i],
                       marker=slacs['marker'][i], color=slacs['colour'][i])
    ax.errorbar(slacs['sigma'][i], gamma_frac[i], x_err=slacs['sigma_err'][i],
                color=slacs['colour'][i], elinewidth=1, fmt='none', capsize=3, label=None)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Lens Name')
plt.xlabel(r'$\sigma_{e/2}$', size=20)
plt.ylabel(r'$\frac{\gamma_{AutoLens}-\gamma_{SLACS}}{\gamma_{SLACS}}$', size=20)
plt.savefig(fig_path + 'gamma_frac_sigma', bbox_inches='tight', dpi=300)

fig5, ax = plt.subplots(figsize=(8,5))
for i in range(len(lens)):
    AL = ax.scatter(slacs['sigma'][i], results.loc[lens[i]]['param']['slope'], label=slacs.index[i],
                       marker=slacs['marker'][i], color='cyan')
    ax.errorbar(slacs['sigma'][i], results.loc[lens[i]]['param']['slope'], yerr=np.array([y_err[:, i]]).T ,
                x_err=slacs['sigma_err'][i],
                color='cyan', elinewidth=1, fmt='none', capsize=3, label=None)
    SLACS = ax.scatter(slacs['sigma'][i], slacs['gamma'][i],
               marker=slacs['marker'][i], color='orange')
    ax.errorbar(slacs['sigma'][i], slacs['gamma'][i], yerr=slacs['gamma_err'][i],
                x_err=slacs['sigma_err'][i],
                color='orange', elinewidth=1, fmt='none', capsize=3, label=None)
    legend1 = ax.legend([AL, SLACS], ['AutoLens', 'SLACS'], loc='lower right')
    ax.add_artist(legend1)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Lens Name')
plt.xlabel(r'$\sigma_{e/2}$', size=20)
plt.ylabel(r'$\gamma$', size=20)
plt.savefig(fig_path + 'gamma_sigma', bbox_inches='tight', dpi=300, transparent=True)

fig6, ax = plt.subplots(figsize=(8,5))
for i in range(len(lens)):
    ax.scatter(slacs['sigma'][i], slacs['gamma'][i], label=slacs.index[i],
                       marker=slacs['marker'][i], color=slacs['colour'][i])
    ax.errorbar(slacs['sigma'][i], slacs['gamma'][i], yerr=slacs['gamma_err'][i],
                x_err=slacs['sigma_err'][i],
                color=slacs['colour'][i], elinewidth=1, fmt='none', capsize=3, label=None)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Lens Name')
plt.xlabel(r'$\sigma_{e/2}$', size=20)
plt.ylabel(r'$\gamma$', size=20)
plt.savefig(fig_path + '_sigma_SLACS', bbox_inches='tight', dpi=300, transparent=True)

## calculating errors on shear magnitude
for i in range(len(lens)):
    lower_error_mag = results.loc[lens[i]]['param']['magnitude']-results.loc[lens[i]]['-error']['magnitude']
    upper_error_mag = results.loc[lens[i]]['+error']['magnitude']-results.loc[lens[i]]['param']['magnitude']
    error_low_mag.append(lower_error_mag)
    error_hi_mag.append(upper_error_mag)
y_err_mag = np.array([error_low_mag, error_hi_mag])

fig7, ax = plt.subplots(figsize=(8,5))
for i in range(len(lens)):
    ax.scatter(slacs['sigma'][i], results.loc[lens[i]]['param']['magnitude'], label=slacs.index[i],
                       marker=slacs['marker'][i], color=slacs['colour'][i])
    ax.errorbar(slacs['sigma'][i], results.loc[lens[i]]['param']['magnitude'], yerr=np.array([y_err_mag[:, i]]).T,
                x_err=slacs['sigma_err'][i],
                color=slacs['colour'][i], elinewidth=1, fmt='none', capsize=3, label=None)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Lens Name')
plt.xlabel(r'$\sigma_{e/2}$', size=20)
plt.ylabel(r'shear magnitude', size=20)
plt.savefig(fig_path + '_sigma_mag', bbox_inches='tight', dpi=300, transparent=True)

fig8, ax = plt.subplots(figsize=(8,5))
for i in range(len(lens)):
    ax.scatter(slacs['R_eff'][i], results.loc[lens[i]]['param']['slope'], label=slacs.index[i],
                       marker=slacs['marker'][i], color=slacs['colour'][i])
    ax.errorbar(slacs['R_eff'][i], results.loc[lens[i]]['param']['slope'], yerr=np.array([y_err[:, i]]).T,
                color=slacs['colour'][i], elinewidth=1, fmt='none', capsize=3, label=None)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Lens Name')
plt.xlabel(r'$R_{eff}$', size=20)
plt.ylabel(r'$\gamma$', size=20)
plt.savefig(fig_path + '_R_eff', bbox_inches='tight', dpi=300)

plt.show()

