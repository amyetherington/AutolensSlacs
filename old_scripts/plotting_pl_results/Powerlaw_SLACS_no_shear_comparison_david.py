from autofit import conf
import autolens as al
import autoastro as astro

import os

path = '{}/../../../../autolens_workspace/'.format(os.path.dirname(os.path.realpath(__file__)))

conf.instance = conf.Config(config_path=path+'config', output_path=path+'output')

import pandas as pd
import numpy as np

import autolens as al
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


data_path = '{}/../../../../../output/slacs_1/'.format(os.path.dirname(os.path.realpath(__file__)))
slacs_path = '{}/../../data/slacs_data_table.xlsx'.format(os.path.dirname(os.path.realpath(__file__)))
wavelength = 'F814W'
pipeline = '/pipeline_power_law_hyper__lens_bulge_disk_power_law__source_inversion__const'
model = '/pipeline_tag__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/' \
        'phase_1__lens_bulge_disk_power_law__source_inversion/phase_tag__sub_2__pos_0.50/' \
        'model.results'
fig_path = '/Users/dgmt59/Documents/Plots/slacs_fresh/PL_no_shear/'

list_ = []
lens=[]
error_low = []
error_hi = []
slope_frac = []
slope_diff = []
q_diff = []
PA_diff = []
r_frac = []
slope = []

slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name
#slacs = slacs.drop(['slacs0959+5100',  'slacs0959+4416', 'slacs2300+0022'], axis=0)

## loading in date from model.results file
for i in range(len(lens_name)):
    full_data_path = Path(data_path + wavelength + '/' + lens_name[i] + pipeline + model)
    if full_data_path.is_file():
        data = pd.read_csv(full_data_path, sep='\s+', header=None, skiprows=19, nrows=7,).set_index(0)
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


for i in range(len(lens)):
    lens_galaxy = al.Galaxy(
        mass=al.mp.EllipticalPowerLaw(
            centre=(results.loc[lens[i]]['param']['centre_0'], results.loc[lens[i]]['param']['centre_1']),
            axis_ratio=results.loc[lens[i]]['param']['axis_ratio'], phi=results.loc[lens[i]]['param']['phi'],
            einstein_radius=results.loc[lens[i]]['param']['einstein_radius'], slope=results.loc[lens[i]]['param']['slope']),
            redshift=slacs['z_lens'][i]
    )

    einstein_radius = lens_galaxy.einstein_radius_in_units(unit_length='arcsec')

    gamma = results.loc[lens[i]]['param']['slope']

    gamma_diff_frac = (results.loc[lens[i]]['param']['slope'] - slacs['gamma'][i])/slacs['gamma'][i]
    gamma_diff = results.loc[lens[i]]['param']['slope'] - slacs['gamma'][i]
    axis_ratio_diff = results.loc[lens[i]]['param']['axis_ratio'] - slacs['q_SIE'][i]
    radius_diff_frac =(einstein_radius - slacs['b_SIE'][i])/slacs['b_SIE'][i]
    phi_diff = results.loc[lens[i]]['param']['phi'] - slacs['PA'][i]


    slope_frac.append(gamma_diff_frac)
    slope_diff.append(gamma_diff)
    q_diff.append(axis_ratio_diff)
    PA_diff.append(phi_diff)
    r_frac.append(radius_diff_frac)
    slope.append(gamma)

y_err = np.array([error_low, error_hi])

print(y_err)
R_ratio = slacs['b_kpc']/slacs['R_eff']

print(np.median(slope))
print(np.mean(slope))

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
plt.savefig(fig_path + 'slopes', bbox_inches='tight', dpi=300)

fig2, ax = plt.subplots(figsize=(8,8))
for i in range(len(lens)):
   ax.scatter(slacs['gamma'][i], results.loc[lens[i]]['param']['slope'],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
   ax.errorbar(slacs['gamma'][i], results.loc[lens[i]]['param']['slope'], yerr=0.23, xerr=slacs['gamma_err'][i],
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
plt.savefig(fig_path + 'slopes_with_estimated_error', bbox_inches='tight', dpi=300)


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
#plt.savefig(fig_path + 'gamma_as_function_of_R_ratio', bbox_inches='tight', dpi=300)
#plt.close()

fig3, ax = plt.subplots(figsize=(8,5))
for i in range(len(lens)):
    ax.scatter(R_ratio[i], slope_frac[i], label=slacs.index[i],
                       marker=slacs['marker'][i], color=slacs['colour'][i])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Lens Name')
plt.xlabel(r'$R_{Ein}/R_{eff}$', size=20)
plt.ylabel(r'$\frac{\gamma_{AutoLens}-\gamma_{SLACS}}{\gamma_{SLACS}}$', size=20)
#plt.savefig(fig_path + '_gamma_frac_R_ratio', bbox_inches='tight', dpi=300)
plt.close()

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
#plt.savefig(fig_path + '_sigma', bbox_inches='tight', dpi=300)
plt.close()

fig4, ax = plt.subplots(figsize=(8,5))
for i in range(len(lens)):
    ax.scatter(slacs['sigma'][i], slope_frac[i], label=slacs.index[i],
                       marker=slacs['marker'][i], color=slacs['colour'][i])
    ax.errorbar(slacs['sigma'][i], slope_frac[i], x_err=slacs['sigma_err'][i],
                color=slacs['colour'][i], elinewidth=1, fmt='none', capsize=3, label=None)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Lens Name')
plt.xlabel(r'$\sigma_{e/2}$', size=20)
plt.ylabel(r'$\frac{\gamma_{AutoLens}-\gamma_{SLACS}}{\gamma_{SLACS}}$', size=20)
#plt.savefig(fig_path + 'gamma_frac_sigma', bbox_inches='tight', dpi=300)
plt.close()

fig3, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(7,4))
for i in range(len(lens)):
    ax1.scatter(R_ratio[i], slope_diff[i],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
    ax2.scatter(R_ratio[i], r_frac[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
    ax3.scatter(R_ratio[i], q_diff[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
    ax4.scatter(R_ratio[i], PA_diff[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
ax1.set_ylabel(r'$\Delta \gamma$', size=14)
ax2.set_ylabel(r'$\frac{\Delta R_{Ein}}{R_{Ein}}$', size=14)
ax3.set_ylabel(r'$\Delta q$', size=14)
ax4.set_ylabel(r'$|\Delta \phi|$', size=14)
ax4.set_xlabel(r'$\frac{R_{Ein}}{R_{eff}}$', size=12)
#box = fig4.position()
#fig4.position([box.x0, box.y0, box.width * 0.8, box.height])
legend = fig3.legend(loc='center right', title= 'Lens Name')
plt.tight_layout()
plt.subplots_adjust(right=0.7)
plt.savefig(fig_path + 'gamma_R_q_PA_SLACS_comparison', bbox_inches='tight', dpi=300)

plt.show()

