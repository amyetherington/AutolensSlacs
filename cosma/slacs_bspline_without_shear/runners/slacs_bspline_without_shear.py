import os
import time
import json
import random
import sys

from autoconf import conf
import autofit as af

# %%
"""Setup the path to the autolens_slacs, using a relative directory name."""


cosma_path = "/cosma7/data/dp004/dc-ethe1"

dataset_label = "slacs_shu"

workspace_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))


cosma_dataset_path = f"{cosma_path}/dataset/{dataset_label}"

cosma_output_path = f"{cosma_path}/output"

conf.instance.push(new_path=f"{workspace_path}/cosma/config", output_path=cosma_output_path)

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

dataset_path = f"{cosma_dataset_path}/{data_name}"

with open(f"{dataset_path}/info.json") as json_file:
    info = json.load(json_file)

# %%
"""Using the dataset path, load the data (image, noise map, PSF) as an imaging object from .fits files."""

# %%
imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/F814W_image_bsplines.fits",
    psf_path=f"{dataset_path}/F814W_psf.fits",
    noise_map_path=f"{dataset_path}/noise_map_scaled_new.fits",
    pixel_scales=pixel_scales,
    positions_path=f"{dataset_path}/positions.dat",
    name=data_name,
)

if data_name in ["slacs0912+0029", "slacs0216-0813"]:
    mask = al.Mask2D.circular(
        shape_2d=imaging.shape_2d, pixel_scales=pixel_scales, centre=(0.0, 0.0), radius=4.5
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

settings_masked_imaging = al.SettingsMaskedImaging(grid_class=al.Grid)

settings_lens = al.SettingsLens(
    positions_threshold=0.7,
    auto_positions_factor=3.0,
    auto_positions_minimum_threshold=0.2,
)

settings = al.SettingsPhaseImaging(
    settings_masked_imaging=settings_masked_imaging, settings_lens=settings_lens
)


# %%
"""
__PIPELINE SETUP__

Transdimensional pipelines used the _SetupPipeline_ object to customize the analysis performed by the pipeline,
for example if a shear was included in the mass model and the model used for the source galaxy.

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong 
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own setup object 
which is equivalent to the _SetupPipeline_ object, customizing the analysis in that pipeline. Each pipeline therefore
has its own _SetupMass_, _SetupLight_ and _SetupSource_ object.

The _Setup_ used in earlier pipelines determine the model used in later pipelines. For example, if the _Source_ 
pipeline is given a _Pixelization_ and _Regularization_, than this _Inversion_ will be used in the subsequent _SLaMPipelineLight_ and 
Mass pipelines. The assumptions regarding the lens light chosen by the _Light_ object are carried forward to the 
_Mass_  pipeline.

The _Setup_ again tags the path structure of every pipeline in a unique way, such than combinations of different
SLaM pipelines can be used to fit lenses with different models. If the earlier pipelines are identical (e.g. they use
the same _SLaMPipelineSource_) they will reuse those results before branching off to fit different models in the _SLaMPipelineLight_ 
and / or _SLaMPipelineMass_ pipelines. 
"""

# %%
"""
__HYPER SETUP__

The _SetupHyper_ determines which hyper-mode features are used during the model-fit as is used identically to the
hyper pipeline examples.

The _SetupHyper_ object has a new input available, 'hyper_fixed_after_source', which fixes the hyper-parameters to
the values computed by the hyper-phase at the end of the Source pipeline. By fixing the hyper-parameter values in the
_SLaMPipelineLight_ and _SLaMPipelineMass_ pipelines, model comparison can be performed in a consistent fashion.
"""

# %%
hyper = al.SetupHyper(
    hyper_galaxies_lens=False,
    hyper_image_sky=False,
    hyper_background_noise=False,
    evidence_tolerance=20.0,
)

# %%
"""
__SLaMPipelineSourceParametric__

The parametric source pipeline aims to initialize a robust model for the source galaxy using _LightProfile_ objects. 

_SLaMPipelineSourceParametric_ determines the source model used by the parametric source pipeline. A full description of all 
options can be found ? and ?.

By default, this assumes an _EllipticalIsothermal_ profile for the lens galaxy's mass. Our experience with lens 
modeling has shown they are the simpliest models that provide a good fit to the majority of strong lenses.

For this runner the _SLaMPipelineSourceParametric_ customizes:

 - If there is an _ExternalShear_ in the mass model or not.
"""

setup_mass = al.SetupMassTotal(mass_prior_model=al.mp.EllipticalIsothermal, with_shear=False, mass_centre=(0.0, 0.0))
setup_source = al.SetupSourceParametric()

pipeline_source_parametric = al.SLaMPipelineSourceParametric(
    setup_mass=setup_mass, setup_source=setup_source
)

# %%
"""
__SLaMPipelineSourceInversion__

The Source inversion pipeline aims to initialize a robust model for the source galaxy using an _Inversion_.

_SLaMPipelineSourceInversion_ determines the _Inversion_ used by the inversion source pipeline. A full description of all 
options can be found ? and ?.

By default, this again assumes _EllipticalIsothermal_ profile for the lens galaxy's mass model.

For this runner the _SLaMPipelineSourceInversion_ customizes:

 - The _Pixelization_ used by the _Inversion_ of this pipeline.
 - The _Regularization_ scheme used by of this pipeline.

The _SLaMPipelineSourceInversion_ use's the _SetupMass_ of the _SLaMPipelineSourceParametric_.

The _SLaMPipelineSourceInversion_ determines the source model used in the _SLaMPipelineLight_ and _SLaMPipelineMass_ pipelines, which in this
example therefore both use an _Inversion_.
"""

setup_source = al.SetupSourceInversion(
    pixelization_prior_model=al.pix.VoronoiBrightnessImage, regularization_prior_model=al.reg.AdaptiveBrightness
)

pipeline_source_inversion = al.SLaMPipelineSourceInversion(setup_source=setup_source)

# %%
"""
__SLaMPipelineMassTotal__

The _SLaMPipelineMassTotal_ pipeline fits the model for the lens galaxy's total mass distribution. 

A full description of all options can be found ? and ?.

The model used to represent the lens galaxy's mass is input into _SLaMPipelineMassTotal_ and this runner uses the default of an 
_EllipticalPowerLaw_ in this example.

For this runner the _SLaMPipelineMass_ customizes:

 - The _MassProfile_ fitted by the pipeline.
 - If there is an _ExternalShear_ in the mass model or not.
"""

setup_mass = al.SetupMassTotal(mass_prior_model=al.mp.EllipticalPowerLaw, with_shear=False)

pipeline_mass = al.SLaMPipelineMass(setup_mass=setup_mass)

# %%
"""
__SLaM__

We combine all of the above _SLaM_ pipelines into a _SLaM_ object.

The _SLaM_ object contains a number of methods used in the make_pipeline functions which are used to compose the model 
based on the input values. It also handles pipeline tagging and path structure.
"""

slam = al.SLaM(
    path_prefix=f"{dataset_label}_bspline_without_shear/{data_name}",
    redshift_lens=info["redshift_lens"],
    redshift_source=info["redshift_source"],
    setup_hyper=hyper,
    pipeline_source_parametric=pipeline_source_parametric,
    pipeline_source_inversion=pipeline_source_inversion,
    pipeline_mass=pipeline_mass,
)

# %%
"""
__PIPELINE CREATION__

We import and make pipelines as per usual, albeit we'll now be doing this for multiple pipelines!

We then add the pipelines together and run this summed pipeline, which runs each individual pipeline back-to-back.
"""

# %%
from autolens_slacs.slam.no_lens_light.pipelines import source__parametric
from autolens_slacs.slam.no_lens_light.pipelines import source__inversion
from autolens_slacs.slam.no_lens_light.pipelines import mass__total

source__sersic = source__parametric.make_pipeline(slam=slam, settings=settings)
source__inversion = source__inversion.make_pipeline(slam=slam, settings=settings)
mass__total = mass__total.make_pipeline(slam=slam, settings=settings)

pipeline = source__sersic + source__inversion + mass__total

pipeline.run(dataset=imaging, mask=mask, info=info)
