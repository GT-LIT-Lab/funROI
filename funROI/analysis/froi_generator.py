from typing import List
from ..froi import _get_froi_all
from nibabel.nifti1 import Nifti1Image
from ..utils import get_froi_path, FROIConfig
from nilearn.image import load_img


class FROIGenerator:
    @staticmethod
    def generate(subject: str, froi: FROIConfig) -> Nifti1Image:
        """
        Get the froi maps for a FROI setup.

        Parameters
        ----------
        subject : str
            The subject ID.
        froi : FROIConfig
            The fROI configuration.

        Returns
        -------
        Nifti1Image
            The fROI map.
        """
        _get_froi_all(subject, froi)
        froi_path = get_froi_path(
            subject,
            froi.task,
            "all",
            froi.contrasts,
            froi.conjunction_type,
            froi.threshold_type,
            froi.threshold_value,
            froi.parcels,
        )
        return load_img(froi_path)
