.. funROI documentation master file, created by
   sphinx-quickstart on Sat Nov  2 15:54:19 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

funROI Documentation
====================

The **funROI** Toolbox is designed to provide robust analytic methods for fMRI data that accommodate inter-subject 
variability in the precise locations of functional activations. Unlike conventional voxel-wise group analyses, 
this toolbox implement the subject-specific functional localization approach introduced in `Fedorenko et al. (2010) <https://pmc.ncbi.nlm.nih.gov/articles/PMC2934923/>`_,
which does not assume strict voxel correspondence across individuals. 
By leveraging each participant's activation maps, the toolbox enables analyses that enhance the sensitivity and resolution of fMRI data interpretations at the individual level.

The **funROI** Toolbox supports several types of analyses:

1. **Parcel generation**: generates functional parcels (brain masks) based on individual activation maps, allowing for subject-specific localization of functional regions without assuming voxel-wise correspondence across participants.

2. **fROI definition**: defines functionally localized Regions of Interest (fROIs) by selecting active voxels within predefined parcels.

3. **Effect estimation**: extracts effect sizes for each subject-specific fROI.

4. **Spatial correlation estimation**: quantifies the similarity of within-subject activation patterns across conditions.

5. **Spatial overlap estimation**: calculates the overlap between parcels and/or fROIs from different subjects or definitions.

The **funROI** Toolbox is compliant with the Brain Imaging Data Structure (BIDS), a standardized framework for organizing and sharing neuroimaging data. BIDS ensures consistency across datasets, 
facilitating data sharing and reproducibility in neuroimaging research. For more details on BIDS, visit `their official webiste <https://bids.neuroimaging.io>`_.

For more assistance with using the toolbox, see the demo at the end of the page on how to use the toolbox. The demo will guide you step by step through setting up an analysis, running different types of subject-specific analyses, and interpreting the output.

How to Cite
-----------

.. warning::
   This section is under construction! The citation information will be updated soon!

If you use **funROI** in your research or other work, please cite it using the following information:

.. note::
   The software is archived on Figshare. For additional citation details or alternative citation formats, please visit the
   `Figshare page <https://doi.org/10.6084/m9.figshare.28120967>`_.

If you need an example reference in a common format, you might consider:

    Gao, R., & Ivanova, A. (2025). funROI: A Python package for ROI-level analyses of functional MRI data
    (Version 1.0) [Software]. Figshare. https://doi.org/10.6084/m9.figshare.28120967

For more details on how to cite the package or to download citation metadata, please visit our
`Figshare repository <https://doi.org/10.6084/m9.figshare.28120967>`_.

Thank you for acknowledging our work!


.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules/     


Demo
-----------------

.. grid::

  .. grid-item-card::
    :link: auto_examples/demo.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 0

      .. grid-item::
        :columns: 12 4 4 4

        .. image:: auto_examples/parcels_left_lateral.png
          :width: 200px

      .. grid-item::
        :columns: 12 8 8 8

        .. div:: sd-font-weight-bold

          Analyses with the HCP LANGUAGE Task Dataset

        Explore how to use the `funROI` package for running analyses on the 
        Human Connectome Project (HCP) dataset.
        