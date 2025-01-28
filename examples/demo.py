"""
Analyses with the HCP LANGUAGE Task Dataset
===========================================

The demo uses LANGUAGE task data from a sample of 30 subjects in the
Human Connectome Project (HCP) Young Adult dataset to showcase the
processing pipeline. This includes first-level processing through to a
variety of fROI-based analyses, such as effect size, spatial
correlations and spatial overlap estimations.

The language localizer task in the HCP involves two conditions: a story
condition, where participants listen to brief auditory stories followed
by a comprehension question, and a math condition, where participants
solve arithmetic problems. These tasks are designed to activate distinct
regions of the brain, with the story condition engaging the language
network and the math condition serving as a non-linguistic control. fMRI
data collected during these tasks allow researchers to identify brain
regions specifically involved in language processing.

Prerequisites
=============

Before running the demo locally, please configure your AWS credentials
to access the HCP dataset. Follow these steps:

1. Refer to the `HCP wiki
   guide <https://wiki.humanconnectome.org/docs/How%20To%20Connect%20to%20Connectome%20Data%20via%20AWS.html>`__
   for instructions on obtaining AWS credentials for accessing the
   dataset.

2. Configure and store your credentials in the ``~/.aws/credentials``
   file. You can find detailed instructions in the `AWS CLI user
   guide <https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html>`__.

"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

subjects = [
    "211417", "164030", "480141", "248238", "214221", "381038", "117021", 
    "671855", "352738", "180836", "677968", "200917", "715647", "107018", 
    "937160", "349244", "214625", "286347", "715041", "749058", "614439", 
    "250932", "145834", "872158", "164636", "932554", "118528", "737960", 
    "187547", "110613"
]
subjects_heldout = [
    '996782', '995174', '994273', '993675', '992774', '992673', '991267',
    '990366', '989987', '987983'
]
    

from funROI.datasets import hcp
hcp.fetch_language_data("./data", subjects + subjects_heldout)


######################################################################
# First Level Modeling
# ====================
# 


######################################################################
# The first-level model in fMRI processing is designed to analyze
# individual subject data by modeling the relationship between
# task-related experimental conditions and the observed brain activity, by
# constructing a General Linear Model (GLM) for each voxel to estimate
# condition-specific effects and identify brain regions activated by the
# task.
# 
# The funROI toolbox wraps Nilearn’s first-level modeling, supporting
# event-related and block designs, customizable hemodynamic response
# functions, confound regression, and statistical contrasts. Below, we
# demonstrate how to configure and run a first-level model using funROI.
# 

import funROI
funROI.set_bids_data_folder('./data/bids')
funROI.set_bids_preprocessed_folder('./data/bids') # using HCP preprocessed data
funROI.set_bids_deriv_folder('./data/bids/derivatives')

from funROI.first_level.nilearn import run_first_level
run_first_level(
    task = 'LANGUAGE',
    subjects = subjects + subjects_heldout,
    space = 'MNINonLinear',
    contrasts = [
        ('story', {'story': 1}),
        ('math', {'math': 1}),
        ('story-math', {'story': 1, 'math': -1}),
    ],
    slice_time_ref = 0
)


######################################################################
# Generate Parcels for the Language System
# ========================================
# 


######################################################################
# In this part, we will demonstrate how to generate parcels for the
# language system using the 30-subject sample. We will focus on the
# story-math contrast to isolate regions of the brain involved in language
# processing. These group-level parcels will later serve as spatial
# constraints for defining subject-specific functional regions of
# interest.
# 

funROI.set_analysis_output_folder("./data/analysis")

from funROI.analysis import ParcelsGenerator
parcels_generator = ParcelsGenerator(
    parcels_name="Language",
    smoothing_kernel_size=8,
    overlap_thr_vox=0.05,
    min_voxel_size=100,
    overlap_thr_roi=0.8
)
parcels_generator.add_subjects(
    subjects=subjects,
    task="LANGUAGE",
    contrasts=["story-math"],
    p_threshold_type="none",
    p_threshold_value=0.05,
)
parcels = parcels_generator.run(return_results=True)


######################################################################
# Let’s take a look at the parcels generated using a sample of 30 subjects
# for the language system. The code snippet below plots the parcels on the
# brain surface for better visualization:
# 

from nilearn.plotting import plot_surf_roi
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import vol_to_surf
fsaverage = fetch_surf_fsaverage('fsaverage5')

surf_data = {
    "left": vol_to_surf(parcels, fsaverage.pial_left, interpolation='nearest', radius=0),
    "right": vol_to_surf(parcels, fsaverage.pial_right, interpolation='nearest', radius=0),
}

views = ["lateral", "medial"]
hemispheres = ["left", "right"]

for hemi in hemispheres:
    for view in views:
        plot_surf_roi(
            surf_mesh=getattr(fsaverage, f"pial_{hemi}"),
            roi_map=surf_data[hemi],
            hemi=hemi,
            view=view,
            bg_on_data=True,
            bg_map=getattr(fsaverage, f"sulc_{hemi}"),
            darkness=0.5,
            cmap="gist_rainbow",
            avg_method='max',
            title=f"{hemi.capitalize()} Hemisphere - {view.capitalize()} View"
        )
        plt.savefig(f"./outputs/parcels_{hemi}_{view}.png")


######################################################################
# Rendered results:
# 
# .. image:: parcels_left_lateral.png
#   :width: 45%
# .. image:: parcels_right_lateral.png
#   :width: 45%
# .. image:: parcels_left_medial.png
#   :width: 45%
# .. image:: parcels_right_medial.png
#   :width: 45%


######################################################################
# Analysis: Effect Sizes
# ======================
# 


######################################################################
# Effect size estimation is a critical step in fROI analysis, as it
# provides a quantitative measure of the strength of the neural response
# to specific contrasts or conditions.
# 
# In this section, we will demonstrate how to estimate effect sizes for
# story and math conditions within subjects’ language fROIs. We will use
# the language system parcel generated in the previous section. For each
# parcel, the language fROI is defined as the top 10% of voxels responding
# to the story-math contrast. The effect sizes of the language system will
# be evaluated in the defined fROIs.
# 
# The analyses below will be done using an heldout set of 10 subjects
# independent from subjects for generating language parcels.
# 

from funROI.analysis import EffectEstimator
froi = funROI.FROIConfig(
    task="LANGUAGE",
    contrasts=["story-math"],
    threshold_type="percent",
    threshold_value=0.1,
    parcels="./data/analysis/parcels/Language/Language_0000.nii.gz",
)
effect_estimator = EffectEstimator(subjects=subjects_heldout, froi=froi)
df_summary, df_detail = effect_estimator.run(
    task="LANGUAGE", effects=["story", "math"], return_results=True)

######################################################################
# Visualize:
#

plt.figure(figsize=(3,5))
data = df_summary.groupby(["subject", "effect"]).mean().reset_index()
sns.barplot(data=data, y="size", x="effect", hue="effect", errorbar="se")
plt.ylabel("Effect Size")
plt.xlabel("Effect")
plt.savefig("./outputs/effect_size.png", dpi=300, bbox_inches="tight")


######################################################################
# As we examine the effect sizes for the subjects’ language system, we
# observe the expected higher responsiveness to story compared to math.
# This confirms the validity of our approach.
#
# Rendered results:
#
# .. image:: effect_size.png
#   :height: 300px
# 
# More interesting questions can be explored by applying the language
# localizer to evaluate response magnitude for other conditions in other
# task runs!
# 


######################################################################
# Analysis: Spatial Correlation Across Conditions
# ===============================================
# 


######################################################################
# Spatial correlation provides a valuable metric for assessing the
# similarity of activation patterns across different conditions or runs.
# This analysis can be performed on parcels or fROIs, allowing researchers
# to evaluate the consistency of functional responses in specific regions
# of the brain.
# 
# In the context of the HCP dataset, which includes only two runs for the
# language localizer task, we cannot fully utilize fROIs for spatial
# correlation due to the lack of sufficient runs (at least three are
# required). However, we can demonstrate spatial correlation between the
# story and math conditions using the previously defined parcels. When
# datasets with more runs are available, this approach can be extended to
# fROIs for a more refined analysis of spatial similarity.
# 

from funROI.analysis import SpatialCorrelationEstimator
spcorr_estimator = SpatialCorrelationEstimator(
    subjects=subjects_heldout,
    froi="./data/analysis/parcels/Language/Language_0000.nii.gz"
)
df_math, _ = spcorr_estimator.run(
    task1='LANGUAGE', effect1='math', task2='LANGUAGE', effect2='math',
    return_results=True
)
df_story, _ = spcorr_estimator.run(
    task1='LANGUAGE', effect1='story', task2='LANGUAGE', effect2='story',
    return_results=True
)
df_between, _ = spcorr_estimator.run(
    task1='LANGUAGE', effect1='story', task2='LANGUAGE', effect2='math',
    return_results=True
)

######################################################################
# Visualize:
#

df_math['Type'] = 'Math-Math'
df_story['Type'] = 'Story-Story'
df_between['Type'] = 'Story-Math'
data = pd.concat(
    [df_between, df_math, df_story]
).groupby(["subject", "Type"]).mean().reset_index()

plt.figure(figsize=(5,5))
sns.barplot(data=data, y="fisher_z", x="Type", hue="Type", errorbar="se")   
plt.ylabel("Fishers Z Correlation")
plt.xlabel("Comparison Type")
plt.savefig("./outputs/spatial_correlation.png", dpi=300, bbox_inches="tight")


######################################################################
# Rendered results:
# 
# .. image:: spatial_correlation.png
#   :height: 300px


######################################################################
# Analysis: Overlap Between fROIs
# ===============================
# 


######################################################################
# Assessing the spatial overlap between parcels or fROIs can be useful in
# various ways, guiding comparisons across tasks and conditions. In this
# section, we showcase an example of computing the degree of overlap
# between the language system defined using the localizer, as stated above
# (10% top voxels), across subjects and within subjects using different
# runs.
# 

from funROI.analysis import OverlapEstimator
overlap_estimator = OverlapEstimator()

data = []
for i, subject1 in enumerate(subjects_heldout):
    df, _ = overlap_estimator.run(
        froi1=froi, froi2=froi, subject1=subject1, subject2=subject1,
        return_results=True
    )
    data.append(df[df['froi1'] == df['froi2']])

    subject2 = subjects_heldout[(i+1) % len(subjects_heldout)]
    df, _ = overlap_estimator.run(
        froi1=froi, froi2=froi, subject1=subject1, subject2=subject2,
        return_results=True
    )
    data.append(df[df['froi1'] == df['froi2']])

data = pd.concat(data)

######################################################################
# Visualize:
#

data.loc[data['subject1'] == data['subject2'], 'Type'] = 'Within'
data.loc[data['subject1'] != data['subject2'], 'Type'] = 'Between'
data_mean = data.groupby(["subject1", "subject2", "Type"]).mean().reset_index()
plt.figure(figsize=(3,5))
sns.barplot(data=data_mean, y="overlap", x="Type", hue="Type", errorbar="se")
plt.ylabel("Overlap")
plt.xlabel("Comparison Type")
plt.savefig("./outputs/overlap.png", dpi=300, bbox_inches="tight")


######################################################################
# The results visualized below illustrate the spatial overlap results:
#
# .. image:: overlap.png
#   :height: 300px
# 
# They demonstrate that the language system defined within the same subject
# using different runs has a higher overlap compared to the overlap between
# different subjects - suggesting that the precise location of the language
# system may vary across subjects!
# 