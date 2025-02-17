"""
Analyses with the HCP LANGUAGE Task Dataset
===========================================

The demo uses LANGUAGE task data in the Human Connectome Project (HCP) Young 
Adult dataset to showcase the processing pipeline. This includes first-level 
processing through to a variety of fROI-based analyses, such as effect size, 
spatial correlations and spatial overlap estimations.

The language localizer task in the HCP involves two conditions: a story 
condition, where participants listen to brief auditory stories followed by a 
comprehension question, and a math condition, where participants solve 
arithmetic problems. These tasks are designed to activate distinct regions of 
the brain, with the story condition engaging the language network and the math 
condition serving as a non-linguistic control. fMRI data collected during these 
tasks allow researchers to identify brain regions specifically involved in 
language processing.


Prerequisites
=============

Before running the demo locally, please configure your AWS credentials to 
access the HCP dataset. Follow these steps:

1. Refer to the `HCP wiki
   guide <https://wiki.humanconnectome.org/docs/How%20To%20Connect%20to%20Connectome%20Data%20via%20AWS.html>`__
   for instructions on obtaining AWS credentials for accessing the
   dataset.

2. Configure and store your credentials in the ``~/.aws/credentials``
   file. You can find detailed instructions in the `AWS CLI user
   guide <https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html>`__.

3. Run the following code to download the data

"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from nilearn.plotting import plot_surf_roi
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.surface import vol_to_surf

subjects1 = ['100206', '100307', '100408', '100610', '101006', '101107',
   '101309', '101410', '101915', '102008', '102109', '102311', '102513',
   '102614', '102715', '102816', '103010', '103111', '103212', '103414',
   '103515', '103818', '104012', '104416', '104820', '105014', '105115',
   '105216', '105620', '105923', '106016', '106319', '106521', '106824',
   '107018', '107321', '107422', '107725', '108020', '108121', '108222',
   '108323', '108525', '108828', '109123', '109325', '109830', '110007',
   '110411', '110613']
subjects2 = ['111009', '111211', '111312', '111413', '111514', '111716',
   '112112', '112314', '112516', '112920', '113215', '113316', '113619',
   '113922', '114116', '114217', '114318', '114419', '114621', '114823']
    

from funROI.datasets import hcp
hcp.fetch_data("./data", task='LANGUAGE', subjects=subjects1+subjects2)

######################################################################
# The following code snippet allows visualizing data on brain surface for 
# later section:

fsaverage = fetch_surf_fsaverage('fsaverage5')


def plot_surf(data, label=None,
             views=["lateral", "medial"],
             hemispheres=["left", "right"]):
   surf_data = {
       "left": vol_to_surf(data, fsaverage.pial_left,
                           interpolation='nearest', radius=0),
       "right": vol_to_surf(data, fsaverage.pial_right,
                            interpolation='nearest', radius=0),
   }


   for hemi in hemispheres:
       for view in views:
           title = f"{hemi.capitalize()} Hemisphere - {view.capitalize()} View"
           if label:
               title = f"{label}: {title}"
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
               title=title
           )


######################################################################
# First Level Modeling
# ====================
# 


######################################################################
# The first-level model in fMRI processing is designed to analyze individual 
# subject data by modeling the relationship between task-related experimental 
# conditions and the observed brain activity, by constructing a General Linear 
# Model (GLM) for each voxel to estimate condition-specific effects and 
# identify brain regions activated by the task.
# 
# The funROI toolbox wraps Nilearn’s first-level modeling, supporting 
# event-related and block designs, customizable hemodynamic response functions, 
# confound regression, and statistical contrasts. Below, we demonstrate how to 
# configure and run a first-level model using funROI. For a list of 
# customizable options for first level modeling, please refer to Nilearn 
# `first_level_from_bids <https://nilearn.github.io/dev/modules/generated/nilearn.glm.first_level.first_level_from_bids.html>`__.
# These options are supported for `run_first_level <https://funroi.readthedocs.io/en/latest/api/funROI.first_level.html#funROI.first_level.nilearn.run_first_level>`__ 
# using the toolbox. 
# 
# In our analysis pipeline, we will run first-level modeling on two distinct 
# samples:
# 
# 1. N = 50 subjects for parcel generation
# 2. N = 20 subjects for subsequent analyses
#
# funROI is designed to operate on and generate data that is compliant with 
# BIDS, the Brain Imaging Data Structure standard. For more information and 
# additional resources on BIDS, please visit the `official website <https://bids.neuroimaging.io/>`__.

import funROI
funROI.set_bids_data_folder('./data/bids')
funROI.set_bids_preprocessed_folder('./data/bids')
funROI.set_bids_deriv_folder('./data/bids/derivatives')
funROI.set_analysis_output_folder("./data/analysis")

from funROI.first_level.nilearn import run_first_level
run_first_level(
   task = 'LANGUAGE',
   subjects = subjects1 + subjects2,
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
# In this demo, we’re focusing on parcel generation using the group-constrained, 
# subject-specific (GcSS) approach introduced by Fedorenko (2010). The process 
# starts by generating a probabilistic map by overlaying individual activation 
# maps obtained from a localizer contrast. We then apply a watershed algorithm 
# to segment this probabilistic map into parcels (brain masks). These parcels 
# will later be used as spatial constraints for defining individual fROIs. When 
# generating the parcels with the toolbox, you can customize the spatial 
# smoothing applied to the probabilistic map, the minimum voxel probability 
# required for inclusion in parcels, and the parcel-level thresholding 
# parameters—including the minimum voxel size and the minimum proportion of 
# subjects that must have a significant voxel with respect to the localizer 
# contrast within that parcel.
# 
# Refer to the `ParcelsGenerator <https://funroi.readthedocs.io/en/latest/api/funROI.analysis.html#funROI.analysis.ParcelsGenerator>`__ 
# documentation for more details.
# 
# In this part, we will demonstrate how to generate parcels for the language 
# system using a sample of 50 subjects. We will focus on the story-math 
# contrast to isolate regions of the brain involved in language processing. 
#

from funROI.analysis import ParcelsGenerator
parcels_generator = ParcelsGenerator(
   parcels_name="Language",
   smoothing_kernel_size=10,
   overlap_thr_vox=0.03,
   min_voxel_size=100,
   overlap_thr_roi=0.8
)
parcels_generator.add_subjects(
   subjects=subjects1,
   task="LANGUAGE",
   contrasts=["story-math"],
   p_threshold_type="none",
   p_threshold_value=0.05,
)
parcels = parcels_generator.run()


######################################################################
# Let's take a look at the parcels generated:
#

plot_surf(parcels, views=["lateral", "medial"], hemispheres=["left", "right"])


######################################################################
# Rendered results:
# 
# .. image:: parcels.png
#   :width: 80%


######################################################################
# Analysis: Define Language fROIs for Individual Subjects
# ======================
# 

######################################################################
# For each parcel generated in the previous step, we now define 
# subject-specific functional ROIs (fROIs) for the language system. Within each 
# parcel, the language fROI for a subject is defined as the top 10% of voxels 
# that exhibit the strongest story-math contrast.
#
# If you proceed directly to the analyses with fROI definition configuration 
# provided, the subject fROIs are automatically defined and saved in the BIDS 
# derivatives. You can customize the definition of the fROI by specifying the 
# proportion of the parcel size (e.g., 10%), a fixed number of voxels, or a 
# significance threshold with respect to a given p-value. The following example 
# demonstrates how to manually generate and inspect fROIs.
#
# Let’s inspect the first three subjects in sample 2 to look at their language fROIs:
#

from funROI.analysis import FROIGenerator
froi = funROI.FROIConfig(
   task="LANGUAGE",
   contrasts=["story-math"],
   threshold_type="percent",
   threshold_value=0.1,
   parcels="./data/analysis/parcels/Language/Language_0000.nii.gz",
)
froi_generator = FROIGenerator(subjects2[:3], froi)
froi_imgs = froi_generator.run()
for subject_label, froi_img in froi_imgs:
   plot_surf(froi_img, label=subject_label)


######################################################################
# Rendered results:
# 
# .. image:: frois.png
#   :width: 100%


######################################################################
# Analysis: Effect Sizes
# ======================
# 


######################################################################
# Effect size estimation is a critical step in fROI analysis, as it provides a 
# quantitative measure of the strength of the neural response to specific 
# contrasts or conditions.
#
# In this section, we will demonstrate how to estimate effect sizes for story 
# and math conditions within subjects’ language fROIs. We will use the language 
# system parcel generated in the previous section. The effect sizes of the 
# language system will be evaluated in the defined fROIs.
#
# We are using the 20-subject independent samples that we used for defining 
# parcels, as described previously.
# 

from funROI.analysis import EffectEstimator
effect_estimator = EffectEstimator(subjects=subjects2, froi=froi)
df_summary, df_detail = effect_estimator.run(
   task="LANGUAGE", effects=["story", "math"])

######################################################################
# Visualize:
#

plt.figure(figsize=(2,5))
data = df_summary.groupby(["subject", "effect"]).mean().reset_index()
sns.barplot(data=data, y="size", x="effect", hue="effect", errorbar="se",
           order=["story", "math"])
sns.stripplot(data=data, y="size", x="effect", dodge=False, alpha=0.5,
             jitter=True, order=["story", "math"], color='black')
plt.ylabel("Effect Size")
plt.xlabel("Effect")


######################################################################
# As we examine the effect sizes for the subjects' language system, we observe 
# the expected higher responsiveness to story compared to math. This confirms 
# the validity of our approach.
#
# Rendered results:
#
# .. image:: effect_size_viz.png
#   :height: 300px
# 
# More interesting questions can be explored by applying the language localizer 
# to evaluate response magnitude for other conditions in other task runs!
# 


######################################################################
# Analysis: Spatial Correlation Across Conditions
# ===============================================
# 


######################################################################
# Spatial correlation provides a valuable metric for assessing the similarity 
# of within-subject activation patterns across different conditions or runs. 
# This analysis can be performed on parcels or fROIs, allowing researchers to 
# evaluate the consistency of functional responses in specific regions of the 
# brain.
#
# Here, we will estimate the spatial correlation between  the story and math 
# conditions within each language parcel (using the parcels defined in the 
# “Generate Parcels for the Language System” section). The toolbox also allows 
# comparing activation patterns in individual fROIs (defined with separate data
# ).
# 

from funROI.analysis import SpatialCorrelationEstimator
spcorr_estimator = SpatialCorrelationEstimator(
   subjects=subjects2,
   froi="./data/analysis/parcels/Language/Language_0000.nii.gz"
)
df_math, _ = spcorr_estimator.run(
   task1='LANGUAGE', effect1='math', task2='LANGUAGE', effect2='math',
)
df_story, _ = spcorr_estimator.run(
   task1='LANGUAGE', effect1='story', task2='LANGUAGE', effect2='story',
)
df_between, _ = spcorr_estimator.run(
   task1='LANGUAGE', effect1='story', task2='LANGUAGE', effect2='math',
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


plt.figure(figsize=(4,5))
sns.barplot(data=data, y="fisher_z", x="Type", hue="Type", errorbar="se",
           order=["Story-Story",  "Math-Math", "Story-Math"])
sns.stripplot(data=data, y="fisher_z", x="Type", dodge=False, alpha=0.5,
             jitter=True, order=["Story-Story",  "Math-Math", "Story-Math"],
             color='black')
plt.ylabel("Fishers Z Correlation")
plt.xlabel("Comparison Type")


######################################################################
# Rendered results:
# 
# .. image:: spatial_correlation_viz.png
#   :height: 300px


######################################################################
# Analysis: Overlap Between fROIs
# ===============================
# 


######################################################################
# Assessing the spatial overlap between parcels or fROIs can be useful in 
# various ways, guiding comparisons across tasks and conditions. In this 
# section, we showcase an example of computing the degree of overlap between 
# the language system defined using the localizer, as stated above (10% top 
# voxels), across subjects and within subjects using different runs.
# 
# We support two overlap measurements: Dice coefficient and overlap coefficient. 
# The default is the overlap coefficient, which is what we will use now.
#

from funROI.analysis import OverlapEstimator
overlap_estimator = OverlapEstimator()
data = []
for i, subject1 in enumerate(subjects2):
   df, _ = overlap_estimator.run(
       froi1=froi, froi2=froi, subject1=subject1, subject2=subject1,
       run1='01', run2='02'
   )
   data.append(df[df['froi1'] == df['froi2']])


   subject2 = subjects2[(i+1) % len(subjects2)]
   df, _ = overlap_estimator.run(
       froi1=froi, froi2=froi, subject1=subject1, subject2=subject2,
       run1='01', run2='02'
   )
   data.append(df[df['froi1'] == df['froi2']])


data = pd.concat(data)

######################################################################
# The results visualized below illustrate the spatial overlap results:
#

data.loc[data['subject1'] == data['subject2'], 'Type'] = 'Within'
data.loc[data['subject1'] != data['subject2'], 'Type'] = 'Between'
data_mean = data.groupby(["subject1", "subject2", "Type"]).mean().reset_index()
plt.figure(figsize=(2,5))
sns.barplot(data=data_mean, y="overlap", x="Type", hue="Type", errorbar="se")
sns.stripplot(data=data_mean, y="overlap", x="Type", dodge=False,
             alpha=0.5, jitter=True, color='black')
plt.ylabel("Overlap")
plt.xlabel("Comparison Type")


######################################################################
# The results visualized below illustrate the spatial overlap results:
#
# .. image:: overlap_viz.png
#   :height: 300px
# 
# They demonstrate that the within-subject definitions are more consistent 
# compared to across-subject definitions!
#
# Let's see some example fROIs by leveraging the fROI generator:  
#
# Below shows the fROI defined for the same subject across different runs:

for subject, run in zip(['111211', '111211', '111312', '111312'],
                       ['01', '02', '01', '02']):
   froi_generator = FROIGenerator([subject], froi, run_label=run)
   froi_imgs = froi_generator.run()
   subject_label, froi_img = froi_imgs[0]
   plot_surf(froi_img, label=subject_label+'-'+run)

######################################################################
# Rendered results:
# 
# .. image:: same_sub_froi.png
#   :width: 80%

######################################################################
# fROI defined for different subjects:

for subject, run in zip(['111514', '111716', '114823', '111009'],
                       ['01', '02', '01', '02']):
   froi_generator = FROIGenerator([subject], froi, run_label=run)
   froi_imgs = froi_generator.run()
   subject_label, froi_img = froi_imgs[0]
   plot_surf(froi_img, label=subject_label+'-'+run)

######################################################################
# Rendered results:
# 
# .. image:: diff_sub_froi.png
#   :width: 80%