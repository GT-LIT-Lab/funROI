import os
import json
from typing import Union, Tuple, Optional
from nibabel.nifti1 import Nifti1Image
from nilearn.image import load_img, math_img
from pathlib import Path
import numpy as np
from . import get_analysis_output_folder
from .utils import ensure_paths

_get_parcels_folder = lambda: get_analysis_output_folder() / "parcels"


class ParcelsConfig(dict):
    @ensure_paths("parcels_path", "labels_path")
    def __init__(
        self,
        parcels_path: Union[str, Path],
        labels_path: Optional[Union[str, Path]] = None,
    ):
        self.parcels_path = parcels_path
        self.labels_path = labels_path
        dict.__init__(self, parcels_path=parcels_path, labels_path=labels_path)

    def __repr__(self):
        return (
            f"ParcelsConfig(parcels_path={self.parcels_path}, "
            f"labels_path={self.labels_path})"
        )

    def __eq__(self, other):
        if not isinstance(other, ParcelsConfig):
            return False
        return (
            self.parcels_path == other.parcels_path
            and self.labels_path == other.labels_path
        )


def get_parcels(
    parcels: Union[str, ParcelsConfig]
) -> Tuple[Nifti1Image, dict]:
    """
    Get parcels image and labels.
    """
    if isinstance(parcels, str):
        parcels_img, label_dict = _get_saved_parcels(parcels)
        if parcels_img is None:
            parcels_img, label_dict = _get_external_parcels(
                ParcelsConfig(parcels_path=parcels)
            )
    else:
        parcels_img, label_dict = _get_external_parcels(parcels)

    return parcels_img, label_dict


def _get_saved_parcels(parcels_label: str) -> Tuple[Nifti1Image, dict]:
    """
    Get parcels image and labels from a saved parcels file.
    """
    parcels_path = (
        _get_parcels_folder() / f"parcels-{parcels_label}_mask.nii.gz"
    )
    parcels_labels_path = None
    return _get_external_parcels(
        ParcelsConfig(
            parcels_path=parcels_path, labels_path=parcels_labels_path
        )
    )


def _get_external_parcels(parcels: ParcelsConfig) -> Tuple[Nifti1Image, dict]:
    """
    Get parcels image and labels from externally specified paths.
    """
    if parcels.parcels_path is None or not parcels.parcels_path.exists():
        return None, None

    parcels_img = load_img(parcels.parcels_path)
    parcels_img = math_img("np.round(img)", img=parcels_img)

    if parcels.labels_path is not None and parcels.labels_path.exists():
        if parcels.labels_path.name.endswith("json"):
            # If JSON file, label dict is a dictionary from numerical labels to
            # label names
            label_dict = json.load(open(parcels.labels_path))
            label_dict = {int(k): v for k, v in label_dict.items()}
        elif parcels.labels_path.name.endswith("txt"):
            # If txt file, one label name per line
            label_dict = {}
            with open(parcels.labels_path, "r") as f:
                for i, line in enumerate(f):
                    label_dict[i + 1] = line.strip()
    else:
        # Default: no text labels
        label_dict = {}
        for label in np.unique(parcels_img.get_fdata()):
            if label != 0:
                label_dict[int(label)] = int(label)
    return parcels_img, label_dict
