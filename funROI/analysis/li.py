from ..froi import FROIConfig, _get_froi_data
import nibabel as nib
from .utils import AnalysisSaver
import numpy as np
from typing import List, Tuple, Optional
import pandas as pd


class LateralityIndexAnalyzer(AnalysisSaver):
    """
    A class to compute the laterality index (LI).
    """

    def __init__(
        self,
        subjects: List[str],
        froi: FROIConfig,
    ):
        """
        Initialize the LateralityIndexAnalyzer.

        :param subjects: List of subject labels.
        :type subjects: List[str]
        :param froi: The fROI configuration.
        :type froi: FROIConfig
        """
        self.subjects = subjects
        self.froi = froi

        self._type = "laterality"
        self._data_summary = None
        self._data_detail = None
        

    def run(self, save: Optional[bool] = True) -> Optional[List[Tuple[str, float]]]:
        """
        Run the analysis to compute the laterality index for each subject.

        :param save: Whether to save the results. If True, the results will be
            saved in a CSV file. Default is True.
        :type save: Optional[bool]
        :return: List of tuples containing subject label and computed LI.
        :rtype: Optional[List[Tuple[str, float]]]
        """
        _data = []
        for subject in self.subjects:
            froi_img = _get_froi_data(
                subject=subject,
                config=self.froi,
                return_nifti=True, 
                run_label='all'
            )
            if froi_img is None:
                print(f"fROI data not found for subject {subject}. Skipping...")
                continue
            _data.append([subject] + list(self._run(froi_img)))    

        self._data_summary = pd.DataFrame(
            _data, 
            columns=["subject", "n_left", "n_right", "laterality_index"])

        if save:
            new_li_info = pd.DataFrame(
                { "froi": [self.froi] }
            )
            self._save(new_li_info)

        return self._data_summary


    @staticmethod
    def _run(img: nib.Nifti1Image) -> Tuple[int, int, float]:
        """
        Compute the laterality index from the provided image.
        :param img: Nifti image containing the fROI data.
        :type img: nib.Nifti1Image
        :return: Tuple containing the number of left, right voxels and the LI.
        :rtype: Tuple[int, int, float]
        """
        data = img.get_fdata()
        affine = img.affine
        nonzero_vox = np.argwhere(data > 0)
        world_coords = nib.affines.apply_affine(affine, nonzero_vox)
        n_left = np.sum(world_coords[:, 0] < 0)
        n_right = np.sum(world_coords[:, 0] > 0)
        if (n_left + n_right) > 0:
            li = (n_left - n_right) / (n_left + n_right)
        else:
            li = np.nan
        return n_left, n_right, li
