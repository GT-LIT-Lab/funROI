import os
import json
from typing import Union, Tuple, Optional
from nibabel.nifti1 import Nifti1Image
from nilearn.image import load_img, math_img
from pathlib import Path
import numpy as np
from . import get_analysis_output_folder
from .utils import ensure_paths
import pandas as pd

_get_parcels_folder = lambda: get_analysis_output_folder() / "parcels"


class ParcelsConfig(dict):
    """
    Configuration for parcels.

    :param parcels_path: Path to the parcels image.
    :type parcels_path: Union[str, Path]
    :param labels_path: Path to the labels file. The labels file can be a JSON
        file mapping numerical labels to label names, or a text file with one
        label name per line.
    :type labels_path: Optional[Union[str, Path]]
    """

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

    @staticmethod
    def from_analysis_output(
        name: str,
        smoothing_kernel_size: int,
        overlap_thr_vox: float,
        overlap_thr_roi: float,
        min_voxel_size: int,
        use_spm_smooth: bool = True,
    ):
        """
        Create a ParcelsConfig object from the analysis output folder.
        """
        parcels_path = (
            _get_parcels_folder()
            / f"parcels-{name}"
            / f"parcels-{name}_sm-{smoothing_kernel_size}_spmsmooth-{use_spm_smooth}_voxthres-{overlap_thr_vox}_roithres-{overlap_thr_roi}_sz-{min_voxel_size}.nii.gz"
        )
        if os.path.exists(
            _get_parcels_folder()
            / f"parcels-{name}"
            / f"parcels-{name}_sm-{smoothing_kernel_size}_spmsmooth-{use_spm_smooth}_voxthres-{overlap_thr_vox}_roithres-{overlap_thr_roi}_sz-{min_voxel_size}.json"
        ):
            labels_path = (
                _get_parcels_folder()
                / f"parcels-{name}"
                / f"parcels-{name}_sm-{smoothing_kernel_size}_spmsmooth-{use_spm_smooth}_voxthres-{overlap_thr_vox}_roithres-{overlap_thr_roi}_sz-{min_voxel_size}.json"
            )
        else:
            labels_path = None
        return ParcelsConfig(parcels_path, labels_path)


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


def label_parcel(
    parcels_img: Nifti1Image, label_dict: dict, label: int
) -> Tuple[Nifti1Image, str]:
    """
    Label a parcel.
    """
    if label not in label_dict:
        raise ValueError(f"Label {label} not found in label dictionary.")
    label_name = label_dict[label]
    return math_img("img == {}".format(label), img=parcels_img), label_name


def merge_parcels(
    parcels_img: Nifti1Image,
    label_dict: dict,
    label1: Union[int, str],
    label2: Union[int, str],
    new_label: Optional[str] = None,
) -> Tuple[Nifti1Image, dict]:
    """
    Merge two parcels.
    """

    if new_label in label_dict.values():
        raise ValueError(
            f"New label {new_label} already exists in label dictionary."
        )

    if isinstance(label1, str):
        label1 = {v: k for k, v in label_dict.items()}[label1]
    if isinstance(label2, str):
        label2 = {v: k for k, v in label_dict.items()}[label2]
    parcels_data = _merge_parcels(parcels_img.get_fdata(), label1, label2)
    parcels_img = Nifti1Image(
        parcels_data, parcels_img.affine, parcels_img.header
    )

    label_dict.pop(label1, None)
    label_dict.pop(label2, None)
    if new_label:
        label_dict[new_label] = new_label

    return parcels_img, label_dict


def _merge_parcels(data: np.ndarray, x: int, y: int) -> np.ndarray:
    if len(data.shape) != 3:
        raise ValueError("Data must be 3D.")
    if x == y:
        return data

    neighbors26 = np.zeros((26, data.shape[0], data.shape[1], data.shape[2]))
    ni = 0
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                neighbors26[ni] = np.roll(data, dx, axis=0)
                neighbors26[ni] = np.roll(neighbors26[ni], dy, axis=1)
                neighbors26[ni] = np.roll(neighbors26[ni], dz, axis=2)
                ni += 1

    mask = (
        np.all(np.isin(neighbors26, [0, x, y]), axis=0)
        & np.any(neighbors26 == x, axis=0)
        & np.any(neighbors26 == y, axis=0)
    )
    data[mask] = x
    data[data == y] = x

    return data


def save_parcels(parcels_img: Nifti1Image, label_dict: dict, name: str):
    """
    Save parcels image and labels.
    """
    parcels_path = _get_parcels_folder() / f"{name}.nii.gz"
    parcels_labels_path = _get_parcels_folder() / f"{name}.json"
    parcels_img.to_filename(parcels_path)
    with open(parcels_labels_path, "w") as f:
        json.dump(label_dict, f)
