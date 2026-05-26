from pathlib import Path
import re
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from nilearn.image import load_img, new_img_like
from statsmodels.stats.multitest import fdrcorrection

from ._registry import get_or_create_record_id
from ._surface import (
    SURFACE_HEMIS,
    flatten_image_data,
    is_surface_image,
    load_surface_image,
    save_surface_image,
    surface_image_from_flat,
)
from .contrast import (
    _get_contrast_data,
    _get_contrast_path,
    _get_contrast_runs,
    _get_surface_contrast_path,
)
from .first_level.nilearn import _find_surface_mesh_paths
from .parcels import (
    ParcelsConfig,
    SurfaceParcelsConfig,
    get_parcels,
    is_no_parcels,
)
from .settings import (
    get_bids_data_folder,
    get_bids_deriv_folder,
    get_bids_preprocessed_folder,
    get_bids_preprocessed_folder_relative,
)
from .utils import _get_orthogonalized_run_labels, validate_arguments


class FROIConfig(dict):
    """
    Functional region of interest (fROI) configuration.

    :param task: The task label.
    :type task: str
    :param contrasts: List of contrast labels.
    :type contrasts: List[str]
    :param threshold_type: The threshold type.
        Options are 'none', 'bonferroni', 'fdr', 'n', 'percent'.
    :type threshold_type: str
    :param threshold_value: The threshold value.
    :type threshold_value: float
    :param parcels: The parcels configuration. If a string is provided, it is
        assumed to be the path to the parcels file.
    :type parcels: Union[str, ParcelsConfig, SurfaceParcelsConfig]
    :param conjunction_type: The conjunction type if multiple contrasts are
        provided. Options are 'min', 'max', 'sum', 'prod', 'and', 'or', or None.
    :type conjunction_type: str, optional
    """

    @validate_arguments(
        threshold_type={"none", "bonferroni", "fdr", "n", "percent"},
        conjunction_type={"min", "max", "sum", "prod", "and", "or", None},
    )
    def __init__(
        self,
        task: str,
        contrasts: List[str],
        threshold_type: str,
        threshold_value: float,
        parcels: Union[
            str, ParcelsConfig, SurfaceParcelsConfig, None
        ],
        conjunction_type: Optional[str] = None,
    ):
        if threshold_value < 0:
            raise ValueError("Threshold value must be non-negative.")
        self.task = task
        self.contrasts = contrasts
        self.conjunction_type = conjunction_type
        self.threshold_type = threshold_type
        self.threshold_value = threshold_value

        if not isinstance(parcels, (ParcelsConfig, SurfaceParcelsConfig)):
            if is_no_parcels(parcels):
                parcels = ParcelsConfig(None)
            else:
                parcels = ParcelsConfig(parcels)
        self.parcels = parcels

        dict.__init__(
            self,
            task=task,
            contrasts=contrasts,
            conjunction_type=conjunction_type,
            threshold_type=threshold_type,
            threshold_value=threshold_value,
            parcels=parcels,
        )

    def __repr__(self):
        return (
            f"FROIConfig(task={self.task}, "
            f"contrasts={self.contrasts}, "
            f"conjunction_type={self.conjunction_type}, "
            f"threshold_type={self.threshold_type}, "
            f"threshold_value={self.threshold_value}, "
            f"parcels={self.parcels})"
        )

    def __eq__(self, other):
        return isinstance(other, FROIConfig) and (
            self.task == other.task
            and self.contrasts == other.contrasts
            and self.conjunction_type == other.conjunction_type
            and self.threshold_type == other.threshold_type
            and self.threshold_value == other.threshold_value
            and self.parcels == other.parcels
        )


_get_subject_froi_folder = lambda subject, task: (
    get_bids_deriv_folder() / f"first_level_{task}" / f"sub-{subject}" / "froi"
)
_get_froi_info_path = lambda subject, task: (
    _get_subject_froi_folder(subject, task)
    / f"sub-{subject}_task-{task}_frois.csv"
)


def _serialize_parcels_reference(parcels):
    if not isinstance(parcels, (ParcelsConfig, SurfaceParcelsConfig)):
        parcels = ParcelsConfig(parcels)

    if is_no_parcels(parcels):
        return None, None
    if isinstance(parcels, SurfaceParcelsConfig):
        return {
            "kind": "surface",
            "parcels_paths": parcels.parcels_paths,
            "mesh_paths": parcels.mesh_paths,
            "space": parcels.space,
        }, parcels.labels_path
    return parcels.parcels_path, parcels.labels_path


def _build_froi_registry_record(config: FROIConfig) -> dict:
    parcels_value, labels_value = _serialize_parcels_reference(config.parcels)
    return {
        "contrasts": str(sorted(config.contrasts)),
        "conjunction_type": config.conjunction_type,
        "threshold_type": config.threshold_type,
        "threshold_value": config.threshold_value,
        "parcels": parcels_value,
        "labels": labels_value,
    }


def _get_froi_record_id(
    subject: str, config: FROIConfig, create: bool = False
) -> int:
    return get_or_create_record_id(
        _get_froi_info_path(subject, config.task),
        _build_froi_registry_record(config),
        create=create,
    )


def _get_froi_stem(
    subject: str,
    run_label: str,
    config: FROIConfig,
    create: Optional[bool] = False,
) -> Path:
    record_id = _get_froi_record_id(subject, config, create=create)
    record_label = f"{int(record_id):04d}"
    return (
        _get_subject_froi_folder(subject, config.task)
        / (
            f"sub-{subject}_task-{config.task}_run-{run_label}"
            f"_froi-{record_label}"
        )
    )


def _get_froi_path(
    subject: str,
    run_label: str,
    config: FROIConfig,
    create: Optional[bool] = False,
) -> Path:
    stem = _get_froi_stem(subject, run_label, config, create=create)
    return stem.with_name(f"{stem.name}_mask.nii.gz")


def _get_surface_froi_paths(
    subject: str,
    run_label: str,
    config: FROIConfig,
    create: Optional[bool] = False,
) -> dict:
    stem = _get_froi_stem(subject, run_label, config, create=create)
    return {
        hemi: stem.with_name(f"{stem.name}_hemi-{hemi}_mask.func.gii")
        for hemi in SURFACE_HEMIS
    }


def _get_derivatives_root() -> Path:
    try:
        bids_data_folder = Path(get_bids_data_folder())
        derivatives_folder = get_bids_preprocessed_folder_relative()
        if derivatives_folder == ".":
            return bids_data_folder
        return bids_data_folder / derivatives_folder
    except (ValueError, RuntimeError):
        return Path(get_bids_preprocessed_folder())


def _infer_surface_mesh_paths(subject: str) -> dict:
    derivatives_root = _get_derivatives_root()
    anat_dir = derivatives_root / f"sub-{subject}" / "anat"
    if not anat_dir.exists():
        raise FileNotFoundError(
            f"No anatomical directory found for subject {subject}."
        )

    space_matches = []
    for path in anat_dir.glob(f"sub-{subject}_hemi-L_*.surf.gii"):
        match = re.search(r"_space-([^_]+)_", path.name)
        if match is not None:
            space_matches.append(match.group(1))
    for space in sorted(set(space_matches)):
        mesh_paths = _find_surface_mesh_paths(derivatives_root, subject, space)
        if mesh_paths is not None:
            return mesh_paths
    raise FileNotFoundError(
        f"Could not infer surface meshes for subject {subject}."
    )


def _get_surface_mesh_paths_for_froi(subject: str, config: FROIConfig) -> dict:
    if isinstance(config.parcels, SurfaceParcelsConfig):
        return config.parcels.mesh_paths
    return _infer_surface_mesh_paths(subject)


def _load_surface_contrast_image(
    subject: str,
    task: str,
    run_label: str,
    contrast: str,
    image_type: str,
    mesh_paths: dict,
):
    data_paths = {
        hemi: _get_surface_contrast_path(
            subject, task, run_label, contrast, image_type, hemi
        )
        for hemi in SURFACE_HEMIS
    }
    if not all(path.exists() for path in data_paths.values()):
        return None
    return load_surface_image(data_paths, mesh_paths)


def _get_froi_runs(subject: str, config: FROIConfig):
    runs = None
    for contrast in config.contrasts:
        runs_i = _get_contrast_runs(subject, config.task, contrast)
        if runs is None:
            runs = runs_i
        else:
            runs = list(set(runs) & set(runs_i))
    return sorted(runs)


@validate_arguments(
    group={1, 2}, orthogonalization={"all-but-one", "odd-even"}
)
def _get_orthogonalized_froi_data(
    subject: str,
    config: FROIConfig,
    group: int,
    orthogonalization: Optional[str] = "all-but-one",
) -> Tuple[np.ndarray, List[str]]:
    """
    Get the orthogonalized froi data.

    :return: The froi masks, shape (n_runs, n_voxels) and the run labels.
        If any of the froi masks is not found, return None, None.
    :rtype: Tuple[np.ndarray, List[str]]
    """
    runs = _get_froi_runs(subject, config)
    if len(runs) == 0:
        return None, None
    labels = _get_orthogonalized_run_labels(runs, group, orthogonalization)

    data = []
    for label in labels:
        froi_data = _get_froi_data(subject, config, label)
        if froi_data is None:
            froi_data = _create_froi(subject, config, label)
            if froi_data is None:
                return None, None
        data.append(froi_data.flatten())
    return np.array(data), labels


def _get_froi_data(
    subject: str,
    config: FROIConfig,
    run_label: str,
    return_nifti: bool = False,
) -> np.ndarray:
    """
    Get the froi data by run label.

    :return: The froi mask, shape (n_voxels,). If the froi mask is not
        found, return None.
    :rtype: np.ndarray
    """
    froi_path = _get_froi_path(subject, run_label, config)
    if froi_path.exists():
        if return_nifti:
            return load_img(froi_path)
        return load_img(froi_path).get_fdata().flatten()

    surface_paths = _get_surface_froi_paths(subject, run_label, config)
    if all(path.exists() for path in surface_paths.values()):
        mesh_paths = _get_surface_mesh_paths_for_froi(subject, config)
        img = load_surface_image(surface_paths, mesh_paths)
        if return_nifti:
            return img
        return flatten_image_data(img)

    data = _create_froi(subject, config, run_label, return_nifti=return_nifti)
    if data is None:
        return None
    return data


def _create_froi(
    subject: str,
    config: FROIConfig,
    run_label: str,
    return_nifti: bool = False,
) -> np.ndarray:
    """
    Create and save a fROI mask. The fROI labels are based on the parcels.
    Numeric labels not included in the label dictionary are not included, if
    an external label file is provided.

    :return: The froi mask, shape (n_voxels,). If any contrast data
        is not found, return None.
    :rtype: np.ndarray
    """
    parcels_img, parcel_labels = get_parcels(config.parcels)
    parcels_ref = None

    data = []
    for contrast in config.contrasts:
        data_i = _get_contrast_data(
            subject, config.task, run_label, contrast, "p"
        )
        if data_i is None:
            return None
        if parcels_img is None and parcels_ref is None:
            contrast_pth = _get_contrast_path(
                subject, config.task, run_label, contrast, "p"
            )
            if contrast_pth.exists():
                parcels_ref = load_img(contrast_pth)
            else:
                mesh_paths = _get_surface_mesh_paths_for_froi(subject, config)
                parcels_ref = _load_surface_contrast_image(
                    subject,
                    config.task,
                    run_label,
                    contrast,
                    "p",
                    mesh_paths,
                )
                if parcels_ref is None:
                    return None
        data.append(data_i[None, ...])
    data = np.array(data)

    reference_img = parcels_img if parcels_img is not None else parcels_ref
    if reference_img is None:
        return None

    if parcels_img is None:
        froi_mask = _create_p_map_mask(
            data,
            config.conjunction_type,
            config.threshold_type,
            config.threshold_value,
        ).squeeze()
    else:
        parcel_data = flatten_image_data(parcels_img)
        froi_mask = np.zeros_like(parcel_data)
        for label in parcel_labels.keys():
            froi_mask_i = (
                _create_p_map_mask(
                    data,
                    config.conjunction_type,
                    config.threshold_type,
                    config.threshold_value,
                    parcel_data == label,
                )
                .squeeze()
                .astype(bool)
            )
            froi_mask[froi_mask_i] = label

    if is_surface_image(reference_img):
        froi_img = surface_image_from_flat(froi_mask, reference_img)
        froi_paths = _get_surface_froi_paths(
            subject, run_label, config, create=True
        )
        next(iter(froi_paths.values())).parent.mkdir(
            parents=True, exist_ok=True
        )
        save_surface_image(froi_img, froi_paths)
        if return_nifti:
            return froi_img
        return flatten_image_data(froi_img)

    froi_path = _get_froi_path(subject, run_label, config, create=True)
    froi_path.parent.mkdir(parents=True, exist_ok=True)
    froi_img = new_img_like(reference_img, froi_mask.reshape(reference_img.shape))
    froi_img.to_filename(froi_path)

    if return_nifti:
        return froi_img

    return froi_mask.flatten()


@validate_arguments(
    threshold_type={"n", "percent", "fdr", "bonferroni", "none"},
)
def _threshold_p_map(
    data: np.ndarray, threshold_type: str, threshold_value: float
) -> np.ndarray:
    """
    Extract voxels from a p-map image based on a threshold. p-value correction
    is applied along the voxel dimension.

    :param data: The p-map data, shape (n_runs, n_voxels).
    :type data: np.ndarray
    :param threshold_type: The threshold type.
        Options are 'n', 'percent', 'fdr', 'bonferroni', or 'none'.
    :type threshold_type: str
    :param threshold_value: The threshold value.
    :type threshold_value: float

    :return: The froi mask, shape (n_runs, n_voxels).
    :rtype: np.ndarray
    """
    data = np.moveaxis(data, -1, 0)
    froi_mask = np.zeros_like(data)

    if threshold_type == "n":
        pvals_sorted = np.sort(data, axis=0)
        threshold = pvals_sorted[threshold_value - 1]
        froi_mask[data <= threshold] = 1

    elif "percent" in threshold_type:
        pvals_sorted = np.sort(data, axis=0)
        n = np.floor(
            threshold_value * np.sum(~np.isnan(data), axis=0, keepdims=True)
        ).astype(int)
        threshold = np.take_along_axis(pvals_sorted, n - 1, axis=0)
        froi_mask[data <= threshold] = 1
    elif threshold_type == "fdr":
        for i in range(data.shape[-1]):
            pvals = data[:, i]
            mask = fdrcorrection(pvals, alpha=threshold_value)[0]
            froi_mask[:, i] = mask
    elif threshold_type == "bonferroni":
        froi_mask.flat[data.flatten() < (threshold_value / data.shape[0])] = 1
    else:
        froi_mask.flat[data.flatten() < threshold_value] = 1

    froi_mask = np.moveaxis(froi_mask, 0, -1)
    return froi_mask


@validate_arguments(
    conjunction_type={"min", "max", "sum", "prod", "and", "or", None},
    threshold_type={"n", "percent", "fdr", "bonferroni", "none"},
)
def _create_p_map_mask(
    data: np.ndarray,
    conjunction_type: str,
    threshold_type: str,
    threshold_value: float,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create a mask based on a p-map data.

    :param data: The p-map data, shape (n_contrast, n_runs, n_voxels).
    :type data: np.ndarray
    :param conjunction_type: The conjunction type.
        Options are 'min', 'max', 'sum', 'prod', 'and', or 'or'.
    :type conjunction_type: str
    :param threshold_type: The threshold type.
        Options are 'n', 'percent', 'fdr', 'bonferroni', or 'none'.
    :type threshold_type: str
    :param threshold_value: The threshold value.
    :type threshold_value: float
    :param mask: The explicit mask to be applied before thresholding.
    :type mask: np.ndarray, shape (n_voxels), optional

    :return: The froi masks, shape (n_runs, n_voxels).
    :rtype: np.ndarray
    """
    assert (
        data.ndim == 3
    ), "data should have shape (n_contrast, n_runs, n_voxels)"

    if mask is not None:
        data = data.astype(float)
        data[np.isnan(data)] = np.inf
        data[:, :, mask == 0] = np.nan

    if conjunction_type in ["min", "max", "sum", "prod"]:
        combined_data = eval(f"np.{conjunction_type}(data, axis=-3)")
        froi_mask = _threshold_p_map(
            combined_data, threshold_type, threshold_value
        )
    else:
        thresholded_data = _threshold_p_map(
            data, threshold_type, threshold_value
        )
        if conjunction_type == "and":
            froi_mask = np.all(thresholded_data, axis=-3)
        else:
            froi_mask = np.any(thresholded_data, axis=-3)

    return froi_mask
