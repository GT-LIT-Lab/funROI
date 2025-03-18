funROI
========================

|docs|

.. |docs| image:: https://readthedocs.org/projects/funroi/badge
    :alt: Documentation Status
    :target: https://funroi.readthedocs.io/en/latest/?badge=latest

The **funROI** (FUNctional Region Of Interest) toolbox is designed to provide robust analytic methods for fMRI data analyses that accommodate inter-subject variability in the precise locations of functional activations. Unlike conventional voxel-wise group analyses, this toolbox implements the subject-specific **functional localization** approach, which does not assume strict voxel correspondence across individuals (see, e.g., Saxe et al, 2006; Fedorenko et al, 2010).

.. image:: doc/source/funROI-collage.png
   :width: 800px
   :align: center

Features
--------

- **Parcel generation:** generates parcels (brain masks) based on individual activation maps, which can serve as a spatial constraint for subsequent subject-level analyses. (This step can be skipped if you already have parcels of interest).

- **fROI definition:** defines functional regions of interest (fROIs) by selecting a subset of functionally responsive voxels within predefined parcels.

- **Effect estimation:** extracts average effect sizes for each subject-specific fROI.

- **Spatial correlation estimation:** quantifies the similarity of within-subject activation patterns across conditions (within either a parcel or an fROI).

- **Spatial overlap estimation:** calculates the overlap between parcels and/or fROIs from different subjects or definitions.

Installation
------------
Install funROI via pip:

.. code-block:: bash

   pip install funROI

Usage
-------------
For more details and examples, please refer to the full documentation at:
https://funroi.readthedocs.io/en/latest/

Citation
--------
If you use funROI in your work, please cite it as follows:

   Gao, R., & Ivanova, A. A. (2025). *funROI: A Python package for functional ROI analyses of fMRI data* (Version 1.0.0). Figshare. https://doi.org/10.6084/m9.figshare.28120967

Acknowledgements
----------------

This toolbox implements the parcel definition, fROI definition, and fROI effect size estimation methods described in `Fedorenko et al. (2010) <https://pmc.ncbi.nlm.nih.gov/articles/PMC2934923/>`_. It builds heavily on the `spm_ss <https://github.com/alfnie/spm_ss>`_ toolbox, which provides a Matlab-based implementation for fROI analyses. We thank Alfonso Nieto-Castañon and Ev Fedorenko for developing these methods. 
