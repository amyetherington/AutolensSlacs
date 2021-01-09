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
                      'slacs1420+6019',
                      'slacs1430+4105',
                      'slacs1627+0053',
                      'slacs1630+4520',
                      'slacs2238-0754',
                      'slacs2300+0022',
                      'slacs2303+1422'])


data_path = '{}/../../../../../output/slacs_final/'.format(os.path.dirname(os.path.realpath(__file__)))
slacs_path = '{}/../../data/slacs_data_table.xlsx'.format(os.path.dirname(os.path.realpath(__file__)))
wavelength = 'F814W'
pipeline = '/pipeline_power_law_hyper__lens_bulge_disk_power_law__source_inversion__const'
model = '/pipeline_tag__with_shear__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/' \
        'phase_2__lens_bulge_power_law__refine_source_inversion/phase_tag__sub_2__pos_0.50/' \
        'model.results'
fig_path = '/Users/dgmt59/Documents/Plots/slacs_final/Power_law_SLACS_comparison/refined_shear_' + wavelength

list_ = []
lens=[]
error_low = []
error_hi = []

slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name
slacs = slacs.drop(['slacs0959+5100',  'slacs0959+4416'], axis=0)

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
   ax.errorbar(slacs['gamma'][i], results.loc[lens[i]]['param']['slope'], yerr=np.array([y_err[:,i]]).T,
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

plt.show()

