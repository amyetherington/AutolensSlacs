from autofit import conf
import os

path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))

conf.instance = conf.Config(config_path=path+'config', output_path=path+'output')



import autolens as al

import pandas as pd
import numpy as np

from astropy import cosmology

import matplotlib.pyplot as plt
from pathlib import Path

M_o = 1.989e30
cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)

slacs_path = '{}/../../dataset/slacs_data_table.xlsx'.format(os.path.dirname(os.path.realpath(__file__)))
data_path = '{}/../../../../output/slacs'.format(os.path.dirname(os.path.realpath(__file__)))
pipeline = '/pipeline_power_law_hyper__lens_bulge_disk_power_law__source_inversion__const' \
          '/pipeline_tag__fix_lens_light__pix_voro_image__reg_adapt_bright__bd_align_centre__disk_sersic/'\
          'phase_1__lens_bulge_disk_power_law__source_inversion/'\
          'phase_tag__sub_2__pos_0.50/model.results'
fig_path = '/Users/dgmt59/Documents/Plots/error_analysis/PL/shu_v_david_'

lens_name = np.array(['slacs0216-0813',
                      'slacs0252+0039',
                      'slacs0737+3216',
                      'slacs0912+0029',
                      'slacs0959+4410',
                      'slacs1205+4910',
                      'slacs1250+0523',
                      'slacs1402+6321',
                      'slacs1430+4105',
               #       'slacs1627+0053',
                      'slacs1630+4520',
                      'slacs2238-0754',
               #       'slacs2300+0022',
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
              #        'slacs1627-0053',
                      'slacs1630+4520',
                      'slacs2238-0754',
                #      'slacs2300+0022',
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
slacs = slacs.drop(['slacs0959+5100',  'slacs0959+4416', 'slacs1420+6019', 'slacs1627+0053', 'slacs2300+0022'], axis=0)

for i in range(len(lens_name)):
    full_data_path = Path(data_path + '_fresh/F814W/' + lens_name[i] + pipeline)
    full_data_path_shu = Path(data_path + '_shu_shu/F814W/' + lens_name_shu[i] + pipeline)
    if full_data_path.is_file() and full_data_path_shu.is_file():
        data = pd.read_csv(full_data_path, sep='\s+', header=None, skiprows=19, nrows=7, ).set_index(0)
        del data.index.name
        data[2] = data[2].str.strip('(,').astype(float)
        data[3] = data[3].str.strip(')').astype(float)
        data.columns = ['param', '-error', '+error']
        list_.append(data)
        lens.append(lens_name[i])
        results = pd.concat(list_, keys=lens)
        ## shu results
        data_shu = pd.read_csv(full_data_path_shu, sep='\s+', header=None, skiprows=19, nrows=7, ).set_index(0)
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
error_low_frac = []
error_hi_frac = []
error_low_shu_frac = []
error_hi_shu_frac = []
error_low_q = []
error_hi_q = []
error_low_q_shu = []
error_hi_q_shu = []
error_low_PA = []
error_hi_PA = []
error_low_PA_shu = []
error_hi_PA_shu = []
error_low_R = []
error_hi_R = []
error_low_R_shu = []
error_hi_R_shu = []
error_low_Radius = []
error_hi_Radius = []
error_low_Radius_shu = []
error_hi_Radius_shu = []


q = []
PA = []
q_shu = []
PA_shu = []
gamma = []
gamma_shu = []
r_ein = []
r_ein_shu = []
r = []
r_shu = []
m_ein = []
m_ein_shu = []

for i in range(len(lens)):
    lens_galaxy = al.Galaxy(
        mass=al.mp.EllipticalPowerLaw(
            centre=(results.loc[lens[i]]['param']['centre_0'], results.loc[lens[i]]['param']['centre_1']),
            axis_ratio=results.loc[lens[i]]['param']['axis_ratio'], phi=results.loc[lens[i]]['param']['phi'],
            einstein_radius=results.loc[lens[i]]['param']['einstein_radius'],
            slope=results.loc[lens[i]]['param']['slope']),
        redshift=slacs['z_lens'][i]
    )

    radius_ein = lens_galaxy.einstein_radius_in_units(unit_length='arcsec')
    mass_ein = lens_galaxy.einstein_mass_in_units(redshift_object=lens_galaxy.redshift,
                                                       redshift_source=slacs['z_source'][i], cosmology=cosmo,
                                                       unit_mass='solMass'
                                                       )

    lens_galaxy_shu = al.Galaxy(
        mass=al.mp.EllipticalPowerLaw(
            centre=(results_shu.loc[lens_shu[i]]['param']['centre_0'], results_shu.loc[lens_shu[i]]['param']['centre_1']),
            axis_ratio=results_shu.loc[lens_shu[i]]['param']['axis_ratio'], phi=results_shu.loc[lens_shu[i]]['param']['phi'],
            einstein_radius=results_shu.loc[lens_shu[i]]['param']['einstein_radius'],
            slope=results_shu.loc[lens_shu[i]]['param']['slope']),
        redshift=slacs['z_lens'][i]
    )

    radius_ein_shu = lens_galaxy_shu.einstein_radius_in_units(unit_length='arcsec')
    mass_ein_shu = lens_galaxy_shu.einstein_mass_in_units(redshift_object=lens_galaxy.redshift,
                                                       redshift_source=slacs['z_source'][i], cosmology=cosmo,
                                                       unit_mass='solMass'
                                                       )


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

    ## david results
    lower_error_frac = (results.loc[lens[i]]['param']['slope'] - results.loc[lens[i]]['-error']['slope'])/results.loc[lens[i]]['param']['slope']
    upper_error_frac = (results.loc[lens[i]]['+error']['slope'] - results.loc[lens[i]]['param']['slope'])/results.loc[lens[i]]['param']['slope']
    error_low_frac.append(lower_error_frac)
    error_hi_frac.append(upper_error_frac)
    ## shu results
    lower_error_shu_frac = (results_shu.loc[lens_shu[i]]['param']['slope'] - results_shu.loc[lens_shu[i]]['-error']['slope'])/results_shu.loc[lens_shu[i]]['param']['slope']
    upper_error_shu_frac = (results_shu.loc[lens_shu[i]]['+error']['slope'] - results_shu.loc[lens_shu[i]]['param']['slope'])/results_shu.loc[lens_shu[i]]['param']['slope']
    error_low_shu_frac.append(lower_error_shu_frac)
    error_hi_shu_frac.append(upper_error_shu_frac)

    ## david results
    lower_error_q = results.loc[lens[i]]['param']['axis_ratio'] - results.loc[lens[i]]['-error']['axis_ratio']
    upper_error_q = results.loc[lens[i]]['+error']['axis_ratio'] - results.loc[lens[i]]['param']['axis_ratio']
    error_low_q.append(lower_error_q)
    error_hi_q.append(upper_error_q)
    ## shu results
    lower_error_q_shu = results_shu.loc[lens_shu[i]]['param']['axis_ratio'] - results_shu.loc[lens_shu[i]]['-error']['axis_ratio']
    upper_error_q_shu = results_shu.loc[lens_shu[i]]['+error']['axis_ratio'] - results_shu.loc[lens_shu[i]]['param']['axis_ratio']
    error_low_q_shu.append(lower_error_q_shu)
    error_hi_q_shu.append(upper_error_q_shu)

    ## david results
    lower_error_PA = results.loc[lens[i]]['param']['phi'] - results.loc[lens[i]]['-error']['phi']
    upper_error_PA = results.loc[lens[i]]['+error']['phi'] - results.loc[lens[i]]['param']['phi']
    error_low_PA.append(lower_error_PA)
    error_hi_PA.append(upper_error_PA)
    ## shu results
    lower_error_PA_shu = results_shu.loc[lens_shu[i]]['param']['phi'] - results_shu.loc[lens_shu[i]]['-error']['phi']
    upper_error_PA_shu = results_shu.loc[lens_shu[i]]['+error']['phi'] - results_shu.loc[lens_shu[i]]['param']['phi']
    error_low_PA_shu.append(lower_error_PA_shu)
    error_hi_PA_shu.append(upper_error_PA_shu)


    axis_ratio = results.loc[lens[i]]['param']['axis_ratio']
    phi = results.loc[lens[i]]['param']['phi']
    axis_ratio_shu = results_shu.loc[lens_shu[i]]['param']['axis_ratio']
    phi_shu = results_shu.loc[lens_shu[i]]['param']['phi']
    pl = results.loc[lens[i]]['param']['slope']
    pl_shu = results_shu.loc[lens_shu[i]]['param']['slope']
    radius_shu = results_shu.loc[lens_shu[i]]['param']['einstein_radius']
    radius = results.loc[lens[i]]['param']['einstein_radius']

    ## david results
    lower_error_R = (results.loc[lens[i]]['param']['einstein_radius'] - results.loc[lens[i]]['-error']['einstein_radius'])/radius
    upper_error_R = (results.loc[lens[i]]['+error']['einstein_radius'] - results.loc[lens[i]]['param']['einstein_radius'])/radius
    error_low_R.append(lower_error_R)
    error_hi_R.append(upper_error_R)
    ## shu results
    lower_error_R_shu = (results_shu.loc[lens_shu[i]]['param']['einstein_radius'] - results_shu.loc[lens_shu[i]]['-error']['einstein_radius'])/radius_shu
    upper_error_R_shu = (results_shu.loc[lens_shu[i]]['+error']['einstein_radius'] - results_shu.loc[lens_shu[i]]['param']['einstein_radius'])/radius_shu
    error_low_R_shu.append(lower_error_R_shu)
    error_hi_R_shu.append(upper_error_R_shu)

    ## david results
    lower_error_Radius = (results.loc[lens[i]]['param']['einstein_radius'] - results.loc[lens[i]]['-error'][
        'einstein_radius'])
    upper_error_Radius = (results.loc[lens[i]]['+error']['einstein_radius'] - results.loc[lens[i]]['param'][
        'einstein_radius'])
    error_low_Radius.append(lower_error_Radius)
    error_hi_Radius.append(upper_error_Radius)
    ## shu results
    lower_error_Radius_shu = (results_shu.loc[lens_shu[i]]['param']['einstein_radius'] -
                         results_shu.loc[lens_shu[i]]['-error']['einstein_radius'])
    upper_error_R_adiusshu = (results_shu.loc[lens_shu[i]]['+error']['einstein_radius'] -
                         results_shu.loc[lens_shu[i]]['param']['einstein_radius'])
    error_low_Radius_shu.append(lower_error_Radius_shu)
    error_hi_Radius_shu.append(upper_error_R_adiusshu)

    q.append(axis_ratio)
    PA.append(phi)
    q_shu.append(axis_ratio_shu)
    PA_shu.append(phi_shu)
    gamma.append(pl)
    gamma_shu.append(pl_shu)
    r.append(radius)
    r_shu.append(radius_shu)
    r_ein.append(radius_ein)
    r_ein_shu.append(radius_ein_shu)
    m_ein.append(mass_ein)
    m_ein_shu.append(mass_ein_shu)



err = np.array([error_low, error_hi])
err_shu = np.array([error_low_shu, error_hi_shu])
err_frac = np.array([error_low_frac, error_hi_frac])
err_shu_frac = np.array([error_low_shu_frac, error_hi_shu_frac])
err_q = np.array([error_low_q, error_hi_q])
err_q_shu = np.array([error_low_q_shu, error_hi_q_shu])
err_PA = np.array([error_low_PA, error_hi_PA])
err_PA_shu = np.array([error_low_PA_shu, error_hi_PA_shu])
err_R_frac = np.array([error_low_R, error_hi_R])
err_R_shu_frac = np.array([error_low_R_shu, error_hi_R_shu])
err_R= np.array([error_low_Radius, error_hi_Radius])
err_R_shu = np.array([error_low_Radius_shu, error_hi_Radius_shu])

q_diff = np.array(q) - np.array(q_shu)
PA_diff = np.abs(np.array(PA) - np.array(PA_shu))
gamma_diff = np.array(gamma) - np.array(gamma_shu)
gamma_frac_diff = (np.array(gamma) - np.array(gamma_shu))/gamma_shu
r_frac_diff = (np.array(r)-np.array(r_shu))/np.array(r_shu)
r_ein_frac_diff = (np.array(r_ein)-np.array(r_ein_shu))/np.array(r_ein_shu)
m_ein_frac_diff = (np.array(m_ein)-np.array(m_ein_shu))/np.array(m_ein_shu)

r_shu_diff = (np.array(r_shu)-np.array(r_ein_shu))/np.array(r_ein_shu)
r_harvey_diff = (np.array(r)-np.array(r_ein))/np.array(r_ein)

r_harvey_diff_rescale = (np.array(np.array(r)*(np.sqrt(np.array(q)*(3-np.array(gamma))) / (1 + np.array(q))))-np.array(r_ein))/np.array(r_ein)

RMS_gamma = np.sqrt(np.mean(np.square(gamma_diff)))
RMS_gamma_frac = np.sqrt(np.mean(np.square(gamma_frac_diff)))
RMS_PA = np.sqrt(np.mean(np.square(PA_diff)))
RMS_q = np.sqrt(np.mean(np.square(q_diff)))
RMS_r = np.sqrt(np.mean(np.square(r_frac_diff)))
RMS_r_ein = np.sqrt(np.mean(np.square(r_ein_frac_diff)))
RMS_m_ein = np.sqrt(np.mean(np.square(m_ein_frac_diff)))

print('formal stat error slope', np.median(np.array([err, err_shu])))
print('formal stat frac error slope', np.median(np.array([err_frac, err_shu_frac])))
print('formal stat error q', np.median(np.array([err_q, err_q_shu])))
print('formal stat error PA', np.median(np.array([err_PA, err_PA_shu])))
print('formal stat error R', np.median(np.array([err_R, err_R_shu])))
print('SLACS error', np.median(slacs['gamma_err']))
print('RMS gamma', RMS_gamma)
print('RMS gamma frac', RMS_gamma_frac)
print('RMS q', RMS_q)
print('RMS PA', RMS_PA)
print('RMS R', RMS_r)
print('RMS R Ein', RMS_r_ein)
print('RMS R Ein', RMS_m_ein)

fig1, ax = plt.subplots(figsize=(8,8))
for i in range(len(lens)):
   ax.scatter(results_shu.loc[lens_shu[i]]['param']['slope'], results.loc[lens[i]]['param']['slope'],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
   ax.errorbar(results_shu.loc[lens_shu[i]]['param']['slope'], results.loc[lens[i]]['param']['slope'],
               yerr=np.array([err[:,i]]).T, xerr=np.array([err_shu[:,i]]).T, color=slacs['colour'][i],
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
plt.savefig(fig_path + 'slopes' , bbox_inches='tight', dpi=300, transparent=True)
plt.close()

fig2, ax = plt.subplots(figsize=(8,8))
for i in range(len(lens)):
   ax.scatter(results_shu.loc[lens_shu[i]]['param']['slope'], results.loc[lens[i]]['param']['slope'],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
   ax.errorbar(results_shu.loc[lens_shu[i]]['param']['slope'], results.loc[lens[i]]['param']['slope'],
               yerr=0.23, xerr=0.23, color=slacs['colour'][i],
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
plt.savefig(fig_path + 'slopes_with_estimated_errors' , bbox_inches='tight', dpi=300, transparent=True)
plt.close()

fig2, ax = plt.subplots(figsize=(8,8))
for i in range(len(lens)):
   ax.scatter(r_shu[i], r[i],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
   ax.errorbar(r_shu[i], r[i],
               yerr=np.array([err_R[:,i]]).T, xerr=np.array([err_R_shu[:,i]]).T, color=slacs['colour'][i],
               elinewidth=1, fmt='none', capsize=3, label=None)

plt.xlabel(r'$R_{Shu}$', size=14)
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
plt.close()


fig3, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8,5))
for i in range(len(lens)):
    ax1.scatter(q[i], gamma_diff[i],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
    ax2.scatter(q[i], r_ein_frac_diff[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
    ax3.scatter(q[i], q_diff[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
ax1.set_ylabel(r'$\Delta \gamma$', size=14)
ax1.axhline(y=0, color='k', linestyle='--')
ax2.set_ylabel(r'$\frac{\Delta R_{Ein}}{R_{Ein}}$', size=14)
ax2.axhline(y=0, color='k', linestyle='--')
ax3.set_ylabel(r'$\Delta q$', size=14)
ax3.axhline(y=0, color='k', linestyle='--')
ax3.set_xlabel(r'$q^{AutoLens}$', size=12)
#box = fig4.position()
#fig4.position([box.x0, box.y0, box.width * 0.8, box.height])
legend = fig3.legend(loc='center right', title= 'Lens Name')
plt.tight_layout()
plt.subplots_adjust(right=0.7)
plt.savefig(fig_path + 'gamma_REin_q', bbox_inches='tight', dpi=300)
plt.close()

fig4, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8,6))
for i in range(len(lens)):
    ax1.scatter(q[i], gamma_diff[i],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
    ax2.scatter(q[i], r_frac_diff[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
    ax3.scatter(q[i], q_diff[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
ax1.set_ylabel(r'$\Delta \gamma$', size=14)
ax1.axhline(y=0, color='k', linestyle='--')
ax2.set_ylabel(r'$\frac{\Delta R}{R}$', size=14)
ax2.axhline(y=0, color='k', linestyle='--')
ax3.set_ylabel(r'$\Delta q$', size=14)
ax3.axhline(y=0, color='k', linestyle='--')
ax3.set_xlabel(r'$q^{AutoLens}$', size=12)
#box = fig4.position()
#fig4.position([box.x0, box.y0, box.width * 0.8, box.height])
legend = fig4.legend(loc='center right', title= 'Lens Name')
plt.tight_layout()
plt.subplots_adjust(right=0.7)
plt.savefig(fig_path + 'gamma_R_q', bbox_inches='tight', dpi=300)
plt.close()

fig5, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8,5))
for i in range(len(lens)):
    ax1.scatter(q[i], gamma_diff[i],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
    ax2.scatter(q[i], m_ein_frac_diff[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
    ax3.scatter(q[i], q_diff[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
ax1.set_ylabel(r'$\Delta \gamma$', size=14)
ax1.axhline(y=0, color='k', linestyle='--')
ax2.set_ylabel(r'$\frac{\Delta M_{Ein}}{M_{Ein}}$', size=14)
ax2.axhline(y=0, color='k', linestyle='--')
ax3.set_ylabel(r'$\Delta q$', size=14)
ax3.axhline(y=0, color='k', linestyle='--')
ax3.set_xlabel(r'$q^{AutoLens}$', size=12)
#box = fig4.position()
#fig4.position([box.x0, box.y0, box.width * 0.8, box.height])
legend = fig5.legend(loc='center right', title= 'Lens Name')
plt.tight_layout()
plt.subplots_adjust(right=0.7)
plt.savefig(fig_path + 'gamma_MEin_q', bbox_inches='tight', dpi=300)


fig6, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,4))
for i in range(len(lens)):
    ax1.scatter(q[i], r_harvey_diff[i],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
    ax2.scatter(q[i], r_harvey_diff_rescale[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
ax1.set_ylabel(r'$\frac{R-R_{Ein}}{R_{Ein}}$', size=14)
ax1.axhline(y=0, color='k', linestyle='--')
ax2.set_ylabel(r'$\frac{R-R_{Ein}}{R_{Ein}}(rescale)$', size=14)
ax2.axhline(y=0, color='k', linestyle='--')
ax2.set_xlabel(r'$q^{AutoLens}$', size=12)
#box = fig4.position()
#fig4.position([box.x0, box.y0, box.width * 0.8, box.height])
legend = fig6.legend(loc='center right', title= 'Lens Name')
plt.tight_layout()
plt.subplots_adjust(right=0.7)
plt.savefig(fig_path + 'R difference', bbox_inches='tight', dpi=300)
plt.close()

fig7, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(8,5))
for i in range(len(lens)):
    ax1.scatter(q[i], gamma_diff[i],
                     color=slacs['colour'][i], label=slacs.index[i], marker=slacs['marker'][i])
    ax2.scatter(q[i], r_ein_frac_diff[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
    ax3.scatter(q[i], r_frac_diff[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
    ax4.scatter(q[i], q_diff[i], color=slacs['colour'][i],
                marker=slacs['marker'][i])
ax1.set_ylabel(r'$\Delta \gamma$', size=14)
ax1.axhline(y=0, color='k', linestyle='--')
ax2.set_ylabel(r'$\frac{\Delta R_{Ein}}{R_{Ein}}$', size=14)
ax2.axhline(y=0, color='k', linestyle='--')
ax3.set_ylabel(r'$\frac{\Delta R}{R}$', size=14)
ax3.axhline(y=0, color='k', linestyle='--')
ax4.set_ylabel(r'$\Delta q$', size=14)
ax4.axhline(y=0, color='k', linestyle='--')
ax4.set_xlabel(r'$q^{AutoLens}$', size=12)
legend = fig7.legend(loc='center right', title= 'Lens Name')
plt.tight_layout()
plt.subplots_adjust(right=0.7)
plt.savefig(fig_path + 'gamma_REin_R_q', bbox_inches='tight', dpi=300)

plt.show()

