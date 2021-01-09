# %%
"""
__SLaM (Source, Light and Mass)__

This SLaM pipeline runner loads a strong lens dataset and analyses it using a SLaM lens modeling pipeline.

__THIS RUNNER__

Using two source pipelines, a light pipeline and a mass pipeline this runner fits _Imaging_ of a strong lens system
where in the final phase of the pipeline:

 - The lens galaxy's _LightProfile_'s are modeled as an _EllipticalSersic_ + _EllipticalExponential_, representing
   a bulge + disk model.
 - The lens galaxy's stellar _MassProfile_ is fitted using the _EllipticalSersic_ + EllipticalExponential of the
    _LightProfile_, where it is converted to a stellar mass distribution via constant mass-to-light ratios.
 - The lens galaxy's _MassProfile_ is modeled as an _EllipticalPowerLaw_.
 - The source galaxy's _LightProfile_ is modeled using an _Inversion_.

This runner uses the SLaM pipelines:

 'slam/with_lens_light/source__sersic.py'.
 'slam/with_lens_light/source___inversion.py'.
 'slam/with_lens_light/light__bulge_disk.py'.
 'slam/with_lens_light/mass__total.py'.

Check them out for a detailed description of the analysis!
"""

# %%
"""Use the WORKSPACE environment variable to determine the path to the autolens workspace."""

# %%
import os
import time
import sys
import json
import random

from autoconf import conf
import autofit as af


# %%
""" AUTOLENS + DATA SETUP """

# %%

cosma_path = "/cosma7/data/dp004/dc-ethe1"

dataset_label = "slacs_shu"

workspace_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))


cosma_dataset_path = f"{cosma_path}/dataset/{dataset_label}"

cosma_output_path = f"{cosma_path}/output"

conf.instance.push(new_path=f"{workspace_path}/cosma/config", output_path=cosma_output_path)

# %%
"""Specify the dataset type, label and name, which we use to determine the path we load the data from."""
import autolens as al

cosma_array_id = int(sys.argv[1])


data_name = []
data_name.append("")  # Task number beings at 1, so keep index 0 blank

data_name.append("slacs0008-0004")  # Index 1
data_name.append("slacs0330-0020")  # Index 2
data_name.append("slacs0903+4116")  # Index 3
data_name.append("slacs0959+0410")  # Index 4
data_name.append("slacs1029+0420")  # Index 5
data_name.append("slacs1153+4612")  # Index 6
data_name.append("slacs1402+6321")  # Index 7
data_name.append("slacs1451-0239")  # Index 8
data_name.append("slacs2300+0022")  # Index 9
data_name.append("slacs0029-0055")  # Index 10
data_name.append("slacs0728+3835")  # Index 11
data_name.append("slacs0912+0029")  # Index 12
data_name.append("slacs0959+4416")  # Index 13
data_name.append("slacs1032+5322")  # Index 14
data_name.append("slacs1205+4910")  # Index 15
data_name.append("slacs1416+5136")  # Index 16
data_name.append("slacs1525+3327")  # Index 17
data_name.append("slacs2303+1422")  # Index 18
data_name.append("slacs0157-0056")  # Index 19
data_name.append("slacs0737+3216")  # Index 20
data_name.append("slacs0936+0913")  # Index 21
data_name.append("slacs1016+3859")  # Index 22
data_name.append("slacs1103+5322")  # Index 23
data_name.append("slacs1213+6708")  # Index 24
data_name.append("slacs1420+6019")  # Index 25
data_name.append("slacs1627-0053")  # Index 26
data_name.append("slacs0216-0813")  # Index 27
data_name.append("slacs0822+2652")  # Index 28
data_name.append("slacs0946+1006")  # Index 29
data_name.append("slacs1020+1122")  # Index 30
data_name.append("slacs1142+1001")  # Index 31
data_name.append("slacs1218+0830")  # Index 32
data_name.append("slacs1430+4105")  # Index 33
data_name.append("slacs1630+4520")  # Index 34
data_name.append("slacs0252+0039")  # Index 35
data_name.append("slacs0841+3824")  # Index 36
data_name.append("slacs0956+5100")  # Index 37
data_name.append("slacs1023+4230")  # Index 38
data_name.append("slacs1143-0144")  # Index 39
data_name.append("slacs1250+0523")  # Index 40
data_name.append("slacs1432+6317")  # Index 41
data_name.append("slacs2238-0754")  # Index 42
data_name.append("slacs2341+0000")  # Index 43


data_name = data_name[cosma_array_id]

pixel_scales = 0.05
# %%
"""
Create the path where the dataset will be loaded from, which in this case is
'/autolens_workspace/dataset/imaging/light_bulge_disk__mass_mlr_nfw__source_parametric'
"""

# %%
dataset_path = f"{cosma_dataset_path}/{data_name}"

with open(f"{dataset_path}/info.json") as json_file:
    info = json.load(json_file)

# %%
imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/F814W_image.fits",
    psf_path=f"{dataset_path}/F814W_psf.fits",
    noise_map_path=f"{dataset_path}/noise_map_scaled_new.fits",
    pixel_scales=pixel_scales,
    positions_path=f"{dataset_path}/positions.dat",
    name=data_name,
)


if data_name in ["slacs0912+0029", "slacs0216-0813"]:
    mask = al.Mask2D.circular(
        shape_2d=imaging.shape_2d, pixel_scales=pixel_scales, centre=(0.0, 0.0), radius=4.0
    )
else:
    mask = al.Mask2D.circular(
         shape_2d=imaging.shape_2d, pixel_scales=pixel_scales, centre=(0.0, 0.0), radius=3.5
    )


try:
    imaging.psf = al.preprocess.array_with_new_shape(
        array=imaging.psf, new_shape=(21, 21)
    )
    yay = True
except OSError:
    yay = False

for i in range(100):

    if not yay:

        try:
            imaging.psf = al.preprocess.array_with_new_shape(
                array=imaging.psf, new_shape=(21, 21)
            )
            yay = True
        except OSError:
            yay = False

if not yay:
    raise IOError("Nope")

# %%
settings_masked_imaging = al.SettingsMaskedImaging(grid_class=al.Grid)

# %%
settings_lens = al.SettingsLens(
    positions_threshold=0.7,
    auto_positions_factor=3.0,
    auto_positions_minimum_threshold=0.2,
)

settings = al.SettingsPhaseImaging(
    settings_masked_imaging=settings_masked_imaging, settings_lens=settings_lens
)

# %%
hyper = al.SetupHyper(
    hyper_galaxies_lens=True,
    hyper_image_sky=False,
    hyper_background_noise=False,
    evidence_tolerance=20.0,
)

setup_light = al.SetupLightParametric(
    bulge_prior_model=al.lp.EllipticalSersic,
    disk_prior_model=al.lp.EllipticalExponential,
    align_bulge_disk_centre=True,
    align_bulge_disk_elliptical_comps=False,
)

setup_mass = al.SetupMassTotal(mass_prior_model=al.mp.EllipticalIsothermal, with_shear=True, mass_centre=(0.0, 0.0))

setup_source = al.SetupSourceParametric()

pipeline_source_parametric = al.SLaMPipelineSourceParametric(
    setup_light=setup_light, setup_mass=setup_mass, setup_source=setup_source
)

setup_source = al.SetupSourceInversion(
    pixelization_prior_model=al.pix.VoronoiBrightnessImage,
    regularization_prior_model=al.reg.AdaptiveBrightness,
)

pipeline_source_inversion = al.SLaMPipelineSourceInversion(setup_source=setup_source)


# %%
setup_light = al.SetupLightParametric(
    align_bulge_disk_centre=True,
    align_bulge_disk_elliptical_comps=False,

)

pipeline_light = al.SLaMPipelineLightParametric(setup_light=setup_light)

setup_mass = al.SetupMassTotal(mass_prior_model=al.mp.EllipticalPowerLaw, with_shear=True)

pipeline_mass = al.SLaMPipelineMass(setup_mass=setup_mass)


slam = al.SLaM(
    path_prefix=f"{dataset_label}_hyper_gal/{data_name}",
    redshift_lens=info["redshift_lens"],
    redshift_source=info["redshift_source"],
    setup_hyper=hyper,
    pipeline_source_parametric=pipeline_source_parametric,
    pipeline_source_inversion=pipeline_source_inversion,
    pipeline_light_parametric=pipeline_light,
    pipeline_mass=pipeline_mass,
)


# %%
from autolens_slacs.slam.with_lens_light.pipelines import source__parametric
from autolens_slacs.slam.with_lens_light.pipelines import source__inversion
from autolens_slacs.slam.with_lens_light.pipelines import light__parametric
from autolens_slacs.slam.with_lens_light.pipelines import mass__total

source__sersic = source__parametric.make_pipeline(slam=slam, settings=settings)
source__inversion = source__inversion.make_pipeline(slam=slam, settings=settings)
light__bulge_disk = light__parametric.make_pipeline(slam=slam, settings=settings)
mass__total = mass__total.make_pipeline(slam=slam, settings=settings)

pipeline = source__sersic # + source__inversion + light__bulge_disk + mass__total

pipeline.run(dataset=imaging, mask=mask, info=info)
