.. funROI documentation master file, created by
   sphinx-quickstart on Sat Nov  2 15:54:19 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

funROI Documentation
====================

The **funROI** (FUNctional Region Of Interest) toolbox is designed to provide robust analytic methods for fMRI data analyses that accommodate inter-subject variability in the precise locations of functional activations. Unlike conventional voxel-wise group analyses, this toolbox implements the subject-specific **functional localization** approach, which does not assume strict voxel correspondence across individuals (see, e.g., Saxe et al, 2006; Fedorenko et al, 2010). 

The **funROI** Toolbox supports several types of analyses:

1. **Parcel generation:** generates parcels (brain masks) based on individual activation maps, which can serve as a spatial constraint for subsequent subject-level analyses. (This step can be skipped if you already have parcels of interest).

2. **fROI definition:** defines functional regions of interest (fROIs) by selecting a subset of functionally responsive voxels within predefined parcels.

3. **Effect estimation:** extracts average effect sizes for each subject-specific fROI.

4. **Spatial correlation estimation:** quantifies the similarity of within-subject activation patterns across conditions (within either a parcel or an fROI).

5. **Spatial overlap estimation:** calculates the overlap between parcels and/or fROIs from different subjects or definitions.

The **funROI** Toolbox is compliant with the Brain Imaging Data Structure (BIDS), a standardized framework for organizing and sharing neuroimaging data. For more details on BIDS, visit `their official webiste <https://bids.neuroimaging.io>`_.

The toolbox builds on top of the *nilearn* package for analyzing fMRI data and follows the same object-oriented code design principles.

For more information, see the demo at the end of the page on how to use the toolbox. The demo will guide you step by step through setting up an analysis, running different types of subject-specific analyses, and interpreting the output. Full toolbox documentation can be accessed `here <https://funroi.readthedocs.io/en/latest/api/modules.html>`_.


Using independent data
-----------------------

To prevent double dipping (Kriegeskorte et al, 2009), the toolbox implements strict data separation principles for different analyses in the pipeline. Specifically, the data used to define fROIs must be independent from the data used to estimate effect sizes or spatial correlation for those fROIs. In practice, this is done by using different runs to define fROIs and to conduct subsequent analyses on those fROIs.

We also recommend that the data used to define parcels should come from different subjects than the subjects used for fROI analyses that use those parcels, although this constraint is not implemented in the toolbox.


Pre-existing parcels
---------------------

Instead of deriving the parcels yourself (a procedure that requires data from a relatively large number of subjects), you can use already existing anatomical or functionally derived parcels. Some sample parcels for the language network, multiple demand network, and high-level visual areas are available `here <https://www.evlab.mit.edu/resources-all/download-parcels>`_ courtesy of Evlab and Kanwisher Lab at MIT.

Acknowledgements
----------------

This toolbox implements the parcel definition, fROI definition, and fROI effect size estimation methods described in `Fedorenko et al. (2010) <https://pmc.ncbi.nlm.nih.gov/articles/PMC2934923/>`_. It builds heavily on the `spm_ss <https://github.com/alfnie/spm_ss>`_ toolbox, which provides a Matlab-based implementation for fROI analyses. We thank Alfonso Nieto-Castañon and Ev Fedorenko for developing these methods. 


How to Cite
-----------

If you use **funROI** in your research or other work, please cite it using the following information:

.. note::
   The software is archived on Figshare. For additional citation details or alternative citation formats, please visit the
   `Figshare page <https://doi.org/10.6084/m9.figshare.28120967>`_.

If you need an example reference in a common format, you might consider:

    Gao, R., & Ivanova, A. A. (2025). funROI: A Python package for functional ROI analyses of fMRI data
    (Version 1.0) [Software]. Figshare. https://doi.org/10.6084/m9.figshare.28120967

For more details on how to cite the package or to download citation metadata, please visit our
`Figshare repository <https://doi.org/10.6084/m9.figshare.28120967>`_.

Thank you for acknowledging our work!


References
----------

- Fedorenko, E., Hsieh, P. J., Nieto-Castañón, A., Whitfield-Gabrieli, S., & Kanwisher, N. (2010). New method for fMRI investigations of language: defining ROIs functionally in individual subjects. Journal of neurophysiology, 104(2), 1177-1194.

- Kriegeskorte, N., Simmons, W. K., Bellgowan, P. S., & Baker, C. I. (2009). Circular analysis in systems neuroscience: the dangers of double dipping. Nature neuroscience, 12(5), 535-540.

- Saxe, R., Brett, M., & Kanwisher, N. (2006). Divide and conquer: a defense of functional localizers. Neuroimage, 30(4), 1088-1096.



Demo
-----------------

.. grid::

  .. grid-item-card::
    :link: examples/demo.html
    :link-type: url
    :columns: 12 12 12 12
    :class-card: sd-shadow-sm
    :margin: 2 2 auto auto

    .. grid::
      :gutter: 3
      :margin: 0
      :padding: 0

      .. grid-item::
        :columns: 12 3 3 3

        .. image:: examples/parcels_left.png
          :width: 200px

      .. grid-item::
        :columns: 12 9 9 9

        .. div:: sd-font-weight-bold

          Functional ROI analyses of the HCP LANGUAGE Task Dataset

        Explore how to use the `funROI` package by running analyses on the 
        Human Connectome Project (HCP) dataset.
        

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules/     
