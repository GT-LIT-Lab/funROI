import json
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from nibabel.nifti1 import Nifti1Image
from nilearn.image import load_img, math_img
from nilearn.surface import SurfaceImage

from ._surface import (
    SURFACE_HEMIS,
    SURFACE_PARTS,
    flatten_image_data,
    get_surface_data_parts,
    is_surface_image,
    load_surface_image,
    save_surface_image,
    surface_image_from_flat,
)
from .settings import get_analysis_output_folder
from .utils import ensure_paths

_get_parcels_folder = lambda: get_analysis_output_folder() / "parcels"


class ParcelsConfig(dict):
    """
    Configuration for volumetric parcels.

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
        parcels_path: Optional[Union[str, Path]],
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
            / (
                f"parcels-{name}_sm-{smoothing_kernel_size}"
                f"_spmsmooth-{use_spm_smooth}_voxthres-{overlap_thr_vox}"
                f"_roithres-{overlap_thr_roi}_sz-{min_voxel_size}.nii.gz"
            )
        )
        if not os.path.exists(parcels_path):
            raise FileNotFoundError(f"Parcels file not found: {parcels_path}")
        labels_candidate = (
            _get_parcels_folder()
            / f"parcels-{name}"
            / (
                f"parcels-{name}_sm-{smoothing_kernel_size}"
                f"_spmsmooth-{use_spm_smooth}_voxthres-{overlap_thr_vox}"
                f"_roithres-{overlap_thr_roi}_sz-{min_voxel_size}.json"
            )
        )
        labels_path = labels_candidate if labels_candidate.exists() else None
        return ParcelsConfig(parcels_path, labels_path)


class SurfaceParcelsConfig(dict):
    """
    Configuration for paired hemisphere surface parcels.
    """

    @ensure_paths("labels_path")
    def __init__(
        self,
        parcels_paths: Mapping[str, Union[str, Path]],
        mesh_paths: Mapping[str, Union[str, Path]],
        labels_path: Optional[Union[str, Path]] = None,
        space: Optional[str] = None,
    ):
        self.parcels_paths = {
            hemi: Path(parcels_paths[hemi]) for hemi in SURFACE_HEMIS
        }
        self.mesh_paths = {
            hemi: Path(mesh_paths[hemi]) for hemi in SURFACE_HEMIS
        }
        self.labels_path = labels_path
        self.space = space
        dict.__init__(
            self,
            parcels_paths=self.parcels_paths,
            mesh_paths=self.mesh_paths,
            labels_path=labels_path,
            space=space,
        )

    def __repr__(self):
        return (
            "SurfaceParcelsConfig("
            f"parcels_paths={self.parcels_paths}, "
            f"mesh_paths={self.mesh_paths}, "
            f"labels_path={self.labels_path}, "
            f"space={self.space})"
        )

    def __eq__(self, other):
        if not isinstance(other, SurfaceParcelsConfig):
            return False
        return (
            self.parcels_paths == other.parcels_paths
            and self.mesh_paths == other.mesh_paths
            and self.labels_path == other.labels_path
            and self.space == other.space
        )

    @staticmethod
    def from_analysis_output(
        name: str,
        smoothing_kernel_size: Union[int, float],
        overlap_thr_vox: float,
        overlap_thr_roi: float,
        min_voxel_size: int,
        space: str = "fsLR32k",
    ):
        base = _get_parcels_folder() / f"parcels-{name}"
        stem = (
            f"parcels-{name}_space-{space}_sm-{smoothing_kernel_size}"
            f"_voxthres-{overlap_thr_vox}_roithres-{overlap_thr_roi}"
            f"_sz-{min_voxel_size}"
        )
        parcels_paths = {
            hemi: base / f"{stem}_hemi-{hemi}.func.gii"
            for hemi in SURFACE_HEMIS
        }
        missing = [path for path in parcels_paths.values() if not path.exists()]
        if len(missing) != 0:
            raise FileNotFoundError(
                f"Surface parcels files not found: {missing}"
            )

        config_path = base / f"parcels-{name}_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Surface parcels config not found: {config_path}"
            )
        with open(config_path, "r") as f:
            config = json.load(f)
        mesh_paths = config.get("mesh_paths", {})
        if not all(hemi in mesh_paths for hemi in SURFACE_HEMIS):
            raise ValueError(
                "Surface parcels config does not include both hemisphere meshes."
            )

        labels_path = base / f"{stem}.json"
        if not labels_path.exists():
            labels_path = None

        return SurfaceParcelsConfig(
            parcels_paths=parcels_paths,
            mesh_paths=mesh_paths,
            labels_path=labels_path,
            space=space,
        )


def is_no_parcels(
    parcels: Optional[Union[str, ParcelsConfig, SurfaceParcelsConfig]]
) -> bool:
    """
    Return True when the configuration explicitly requests parcel-free fROIs.
    """
    if parcels is None:
        return True
    if isinstance(parcels, str):
        return parcels.lower() == "none"
    if isinstance(parcels, SurfaceParcelsConfig):
        return False
    return parcels.parcels_path is None


def _load_label_dict(labels_path, data: np.ndarray) -> dict:
    if labels_path is not None and Path(labels_path).exists():
        labels_path = Path(labels_path)
        if labels_path.name.endswith("json"):
            label_dict = json.load(open(labels_path))
            return {int(k): v for k, v in label_dict.items()}
        if labels_path.name.endswith("txt"):
            label_dict = {}
            with open(labels_path, "r") as f:
                for i, line in enumerate(f):
                    label_dict[i + 1] = line.strip()
            return label_dict

    label_dict = {}
    for label in np.unique(data):
        if label != 0:
            label_dict[int(label)] = int(label)
    return label_dict


def get_parcels(
    parcels: Union[str, ParcelsConfig, SurfaceParcelsConfig]
) -> Tuple[Optional[Union[Nifti1Image, SurfaceImage]], Optional[dict]]:
    """
    Get parcels image and labels.
    """
    if is_no_parcels(parcels):
        return None, None

    if isinstance(parcels, str):
        parcels_img, label_dict = _get_saved_parcels(parcels)
        if parcels_img is None:
            parcels_img, label_dict = _get_external_parcels(
                ParcelsConfig(parcels_path=parcels)
            )
    elif isinstance(parcels, SurfaceParcelsConfig):
        parcels_img, label_dict = _get_surface_parcels(parcels)
    else:
        parcels_img, label_dict = _get_external_parcels(parcels)

    return parcels_img, label_dict


def _get_saved_parcels(parcels_label: str) -> Tuple[Optional[Nifti1Image], Optional[dict]]:
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


def _get_external_parcels(
    parcels: ParcelsConfig,
) -> Tuple[Optional[Nifti1Image], Optional[dict]]:
    """
    Get parcels image and labels from externally specified paths.
    """
    if parcels.parcels_path is None or not parcels.parcels_path.exists():
        return None, None

    parcels_img = load_img(parcels.parcels_path)
    parcels_img = math_img("np.round(img)", img=parcels_img)
    label_dict = _load_label_dict(
        parcels.labels_path, parcels_img.get_fdata().reshape(-1)
    )
    return parcels_img, label_dict


def _get_surface_parcels(
    parcels: SurfaceParcelsConfig,
) -> Tuple[Optional[SurfaceImage], Optional[dict]]:
    if not all(path.exists() for path in parcels.parcels_paths.values()):
        return None, None
    if not all(path.exists() for path in parcels.mesh_paths.values()):
        return None, None

    parcels_img = load_surface_image(
        parcels.parcels_paths, parcels.mesh_paths
    )
    parts = get_surface_data_parts(parcels_img)
    parcels_img = SurfaceImage(
        mesh=parcels_img.mesh,
        data={
            SURFACE_PARTS[hemi]: np.round(parts[hemi]).astype(np.float32)
            for hemi in SURFACE_HEMIS
        },
    )
    label_dict = _load_label_dict(
        parcels.labels_path, flatten_image_data(parcels_img)
    )
    return parcels_img, label_dict


def label_parcel(
    parcels_img: Union[Nifti1Image, SurfaceImage], label_dict: dict, label: int
) -> Tuple[Union[Nifti1Image, SurfaceImage], str]:
    """
    Label a parcel.
    """
    if label not in label_dict:
        raise ValueError(f"Label {label} not found in label dictionary.")
    label_name = label_dict[label]
    if is_surface_image(parcels_img):
        parts = get_surface_data_parts(parcels_img)
        mask_img = SurfaceImage(
            mesh=parcels_img.mesh,
            data={
                SURFACE_PARTS[hemi]: (
                    np.round(parts[hemi]) == label
                ).astype(np.float32)
                for hemi in SURFACE_HEMIS
            },
        )
        return mask_img, label_name
    return math_img(f"img == {label}", img=parcels_img), label_name


def merge_parcels(
    parcels_img: Union[Nifti1Image, SurfaceImage],
    label_dict: dict,
    label1: Union[int, str],
    label2: Union[int, str],
    new_label: Optional[str] = None,
) -> Tuple[Union[Nifti1Image, SurfaceImage], dict]:
    """
    Merge two parcels.
    """
    if is_surface_image(parcels_img):
        raise NotImplementedError(
            "merge_parcels does not currently support surface parcels."
        )

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


def save_parcels(
    parcels_img: Union[Nifti1Image, SurfaceImage], label_dict: dict, name: str
):
    """
    Save parcels image and labels.
    """
    base = _get_parcels_folder()
    base.mkdir(parents=True, exist_ok=True)
    if is_surface_image(parcels_img):
        save_surface_image(
            parcels_img,
            {
                hemi: base / f"{name}_hemi-{hemi}.func.gii"
                for hemi in SURFACE_HEMIS
            },
        )
        parcels_labels_path = base / f"{name}.json"
        with open(parcels_labels_path, "w") as f:
            json.dump(label_dict, f)
        return

    parcels_path = base / f"{name}.nii.gz"
    parcels_labels_path = base / f"{name}.json"
    parcels_img.to_filename(parcels_path)
    with open(parcels_labels_path, "w") as f:
        json.dump(label_dict, f)
