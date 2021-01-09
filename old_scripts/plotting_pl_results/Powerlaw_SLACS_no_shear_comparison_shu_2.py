from autofit import conf
import autolens as al
import autoastro as astro

import os


import pandas as pd
import numpy as np
import scipy.optimize as opt

import autolens as al
from astropy import cosmology

import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.cm as cm
from matplotlib import markers

M_o = 1.989e30
cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)

lens_name = np.array(['slacs0008-0004',
                      'slacs0330-0020',
                      'slacs0903+4116',
                      'slacs0959+0410',
                      'slacs1029+0420',
                      'slacs1153+4612',
                      'slacs1402+6321',
                      'slacs1451-0239',
                      'slacs2300+0022',
                      'slacs0029-0055',
                      'slacs0728+3835',
                      'slacs0912+0029',
                      'slacs0959+4416',
                      'slacs1032+5322',
                      'slacs1205+4910',
                      'slacs1416+5136',
                      'slacs1525+3327',
                      'slacs2303+1422',
                      'slacs0157-0056',
                      'slacs0737+3216',
                      'slacs0936+0913',
                      'slacs1016+3859',
                      'slacs1103+5322',
                      'slacs1213+6708',
                      'slacs1420+6019',
                      'slacs1627-0053',
                      'slacs0216-0813',
                      'slacs0822+2652',
                      'slacs0946+1006',
                  #    'slacs1020+1122',
                      'slacs1142+1001',
                      'slacs1218+0830',
                      'slacs1430+4105',
                      'slacs1630+4520',
                      'slacs0252+0039',
                      'slacs0841+3824',
                      'slacs0956+5100',
                      'slacs1023+4230',
                      'slacs1143-0144',
                      'slacs1250+0523',
                 #     'slacs1432+6317',
                      'slacs2238-0754',
                      'slacs2341+0000'])


data_path = '{}/../../../../output/slacs_shu_2/'.format(os.path.dirname(os.path.realpath(__file__)))
slacs_path = '{}/../../dataset/slacs_data_table.xlsx'.format(os.path.dirname(os.path.realpath(__file__)))
pipeline = '/pipeline_mass__power_law/general/' \
           'source__sersic__no_shear__lens_light_centre_(0.00,0.00)__lens_mass_centre_(0.00,0.00)__fix_lens_light/' \
           'light__bulge_disk__align_bulge_disk_centre__disk_exp/mass__power_law__no_shear/phase_1__lens_power_law__source/' \
           'phase_tag__sub_2__pos_0.30/model.results'
fig_path = '/Users/dgmt59/Documents/Plots/slacs_shu_full_2/'

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
r_ein = []
q = []

slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name
slacs = slacs.drop(['slacs1432+6317'], axis=0)


## loading in date from model.results file
for i in range(len(lens_name)):
    full_data_path = Path(data_path + '/' + lens_name[i] + pipeline)
    print(full_data_path)
    if full_data_path.is_file():
        data = pd.read_csv(full_data_path, sep='\s+', header=None, skiprows=33, nrows=8,).set_index(0)
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
    axis_ratio = results.loc[lens[i]]['param']['axis_ratio']

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
    r_ein.append(einstein_radius)
    q.append(axis_ratio)

y_err = np.array([error_low, error_hi])

R_ratio = slacs['b_SIE']/slacs['R_eff']

RMS_slope = np.sqrt(np.mean(np.square(slope_diff)))
RMS_slope_frac = np.sqrt(np.mean(np.square(slope_frac)))

print(RMS_slope_frac)

print(np.median(slope))
print(np.mean(slope))
print(np.median(slacs['gamma']))
print(np.sum(slacs['gamma']*slacs['gamma_err'])/np.sum(slacs['gamma_err']))

print(slacs['colour'])

## plotting
fig1, ax = plt.subplots(figsize=(8,8))
for i in range(len(lens)):
    print(slacs['colour'][lens[i]])
    ax.scatter(slacs['gamma'][lens[i]], results.loc[lens[i]]['param']['slope'],
              color=slacs['colour'][lens[i]], marker=slacs['marker'][lens[i]], label=lens[i])
    ax.errorbar(slacs['gamma'][lens[i]], results.loc[lens[i]]['param']['slope'],
               color=slacs['colour'][lens[i]], yerr=np.array([y_err[:,i]]).T,
               xerr=slacs['gamma_err'][lens[i]], elinewidth=1, fmt='none', capsize=3, label=None)
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
plt.savefig(fig_path + 'slopes_shu!', bbox_inches='tight', dpi=300)

plt.show()

fig2, ax = plt.subplots(figsize=(8,8))
for i in range(len(lens)):
   ax.scatter(slacs['gamma'][lens[i]], results.loc[lens[i]]['param']['slope'],label=lens[i])
   ax.errorbar(slacs['gamma'][lens[i]], results.loc[lens[i]]['param']['slope'], yerr=0.23, xerr=slacs['gamma_err'][i],
               elinewidth=1, fmt='none', capsize=3, label=None)
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
#plt.savefig(fig_path + 'slopes_with_estimated_error', bbox_inches='tight', dpi=300, transparent=True)
plt.close()

fig2, ax = plt.subplots(figsize=(8,5))
for i in range(len(lens)):
    ax.scatter(R_ratio[i], results.loc[lens[i]]['param']['slope'], label=lens[i])
    ax.errorbar(R_ratio[i], results.loc[lens[i]]['param']['slope'], yerr=np.array([y_err[:, i]]).T, elinewidth=1, fmt='none', capsize=3, label=None)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Lens Name')
plt.xlabel(r'$R_{Ein}/R_{eff}$', size=20)
plt.ylabel(r'$\gamma$', size=20)
plt.savefig(fig_path + 'gamma_as_function_of_R_ratio', bbox_inches='tight', dpi=300)
#plt.close()

fig3, ax = plt.subplots(figsize=(8,5))
for i in range(len(lens)):
    ax.scatter(R_ratio[i], slope_frac[i], label=lens[i])
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Lens Name')
plt.xlabel(r'$R_{Ein}/R_{eff}$', size=20)
plt.ylabel(r'$\frac{\gamma_{AutoLens}-\gamma_{SLACS}}{\gamma_{SLACS}}$', size=20)
plt.savefig(fig_path + '_gamma_frac_R_ratio', bbox_inches='tight', dpi=300)
#plt.close()

fig3, ax = plt.subplots(figsize=(8,5))
for i in range(len(lens)):
    ax.scatter(slacs['sigma'][i], results.loc[lens[i]]['param']['slope'], label=lens[i])
    ax.errorbar(slacs['sigma'][i], results.loc[lens[i]]['param']['slope'], yerr=np.array([y_err[:, i]]).T,
                x_err=slacs['sigma_err'][lens[i]], elinewidth=1, fmt='none', capsize=3, label=None)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Lens Name')
plt.xlabel(r'$\sigma_{e/2}$', size=20)
plt.ylabel(r'$\gamma$', size=20)
#plt.savefig(fig_path + '_sigma', bbox_inches='tight', dpi=300)
plt.close()

fig4, ax = plt.subplots(figsize=(8,5))
for i in range(len(lens)):
    ax.scatter(slacs['sigma'][lens[i]], slope_frac[i], label=lens[i])
    ax.errorbar(slacs['sigma'][lens[i]], slope_frac[i], x_err=slacs['sigma_err'][lens[i]],
                elinewidth=1, fmt='none', capsize=3, label=None)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Lens Name')
plt.xlabel(r'$\sigma_{e/2}$', size=20)
plt.ylabel(r'$\frac{\gamma_{AutoLens}-\gamma_{SLACS}}{\gamma_{SLACS}}$', size=20)
#plt.savefig(fig_path + 'gamma_frac_sigma', bbox_inches='tight', dpi=300)
plt.close()

fig3, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8,5))
for i in range(len(lens)):
    ax1.scatter(R_ratio[i], slope_diff[i], label=lens[i])
    ax2.scatter(R_ratio[i], r_frac[i])
    ax3.scatter(R_ratio[i], q_diff[i])
ax1.set_ylabel(r'$\Delta \gamma$', size=14)
ax2.set_ylabel(r'$\frac{\Delta R_{Ein}}{R_{Ein}}$', size=14)
ax3.set_ylabel(r'$\Delta q$', size=14)
ax3.set_xlabel(r'$\frac{R_{Ein}}{R_{eff}}$', size=12)
#box = fig4.position()
#fig4.position([box.x0, box.y0, box.width * 0.8, box.height])
legend = fig3.legend(loc='center right', title= 'Lens Name')
plt.tight_layout()
plt.subplots_adjust(right=0.7)
#plt.savefig(fig_path + 'r_ratio_gamma_R_q_SLACS_comparison', bbox_inches='tight', dpi=300)
plt.close()

fig4, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8,5))
for i in range(len(lens)):
    ax1.scatter(R_ratio[i], slope[i], label=lens[i])
    ax2.scatter(R_ratio[i], r_ein[i])
    ax3.scatter(R_ratio[i], q[i])
ax1.set_ylabel(r'$\gamma$', size=14)
ax2.set_ylabel(r'$R_{Ein}$', size=14)
ax3.set_ylabel(r'$q$', size=14)
ax3.set_xlabel(r'$\frac{R_{Ein}}{R_{eff}}$', size=12)
#box = fig4.position()
#fig4.position([box.x0, box.y0, box.width * 0.8, box.height])
legend = fig4.legend(loc='center right', title= 'Lens Name')
plt.tight_layout()
plt.subplots_adjust(right=0.7)
plt.savefig(fig_path + 'gamma_R_q_SLACS_comparison', bbox_inches='tight', dpi=300)

fig5, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8,5))
for i in range(len(lens)):
    ax1.scatter(q[i], slope_diff[i], label=lens[i])
    ax2.scatter(q[i], r_frac[i])
    ax3.scatter(q[i], q_diff[i])
ax1.set_ylabel(r'$\Delta \gamma$', size=14)
ax2.set_ylabel(r'$\frac{\Delta R_{Ein}}{R_{Ein}}$', size=14)
ax3.set_ylabel(r'$\Delta q$', size=14)
ax3.set_xlabel(r'$q$', size=12)
#box = fig4.position()
#fig4.position([box.x0, box.y0, box.width * 0.8, box.height])
legend = fig5.legend(loc='center right', title= 'Lens Name')
plt.tight_layout()
plt.subplots_adjust(right=0.7)
#plt.savefig(fig_path + 'r_ratio_gamma_R_q_SLACS_comparison', bbox_inches='tight', dpi=300)
plt.show()
stop
gamma_diff_error = np.array(np.sqrt(np.square(0.23)+np.square(slacs['gamma_err'])))

def func(r, m, c):
    return m*r+c

x0 = np.array([10, 0])

fit, fit_cov = opt.curve_fit(func,  R_ratio, slope_diff, x0, method='lm', sigma=gamma_diff_error)
uncert = np.sqrt(np.diag(fit_cov))
best_fit = func(R_ratio, fit[0], fit[1])

print(fit)
print(uncert)

print(gamma_diff_error)

error_hi_slope_diff = []
error_low_slope_diff = []

for i in range(len(lens)):
    lower_error_slope_diff = slope_diff[i]-gamma_diff_error[i]
    upper_error_slope_diff = slope_diff[i]+gamma_diff_error[i]
    error_low_slope_diff.append(lower_error_slope_diff)
    error_hi_slope_diff.append(upper_error_slope_diff)

y_err_gamma_diff = np.array([error_low_slope_diff, error_hi_slope_diff])

print(y_err_gamma_diff)

fig5, ax = plt.subplots(figsize=(8,5))
for i in range(len(lens)):
    ax.scatter(R_ratio[i], slope_diff[i], label=slacs.index[i],
                       marker=slacs['marker'][i], color=slacs['colour'][i])
    ax.errorbar(R_ratio[i], slope_diff[i], y_err=np.array([y_err_gamma_diff[:, i]]).T,
                color=slacs['colour'][i], elinewidth=1, fmt='none', capsize=3, label=None)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Lens Name')
ax.plot(R_ratio, best_fit, color='grey', linestyle='--')
plt.xlabel(r'$R_{Ein}/R_{eff}$', size=20)
plt.ylabel(r'$\gamma_{AutoLens}-\gamma_{SLACS}$', size=20)
#plt.savefig(fig_path + '_gamma_frac_R_ratio', bbox_inches='tight', dpi=300)
#plt.close()




plt.show()

