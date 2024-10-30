import os
from .utils import get_froi_path, get_runs, get_subject_froi_folder, FROIConfig
from .parcels import get_parcels
from .contrast import (
    _get_contrast_all,
    _get_contrast_orth,
    _get_contrast_run,
    _create_p_map_mask,
)
from typing import Optional
from nibabel.nifti1 import Nifti1Image
import numpy as np
from nilearn.image import load_img, new_img_like


FROIConfig.__new__.__defaults__ = FROIConfig(
    task="",
    contrasts=[],
    conjunction_type="and",
    threshold_type="none",
    threshold_value=None,
    parcels=None,
)


def _get_froi_all(subject: str, froi: FROIConfig) -> np.ndarray:
    """
    Get the all-run froi maps.
    Returns the froi maps, shape (1, n_voxels).
    """
    os.makedirs(get_subject_froi_folder(subject, froi.task), exist_ok=True)
    froi_path = get_froi_path(
        subject,
        froi.task,
        "all",
        froi.contrasts,
        froi.conjunction_type,
        froi.threshold_type,
        froi.threshold_value,
        froi.parcels,
        create=True,
    )
    if os.path.exists(froi_path):
        img = load_img(froi_path)
        return img.get_fdata().flatten()[np.newaxis, :]

    parcels_img, _ = get_parcels(froi.parcels)
    assert parcels_img is not None, "Parcels image not found."

    p_maps = []
    for contrast in froi.contrasts:
        p_maps.append(_get_contrast_all(subject, froi.task, contrast, "p"))
    p_maps = np.stack(p_maps, axis=0)

    parcels_data = parcels_img.get_fdata()
    froi_mask = _generate_froi_mask(
        p_maps,
        parcels_data.flatten(),
        froi.conjunction_type,
        froi.threshold_type,
        froi.threshold_value,
    )

    # save the fROI mask
    froi_img = new_img_like(
        parcels_img,
        froi_mask[0].reshape(parcels_data.shape).astype(np.float32),
    )
    froi_img.to_filename(froi_path)

    return froi_mask


def _get_froi_orth(subject: str, froi: FROIConfig) -> np.ndarray:
    """
    Get the orthogonal-run froi maps.
    Returns the froi maps, shape (n_runs, n_voxels).
    """
    return _get_froi_run(subject, froi, orthogonal=True)


def _get_froi_run(
    subject: str,
    froi: FROIConfig,
    orthogonal: Optional[bool] = False,
) -> np.ndarray:
    """
    Get the orthogonal-run froi maps.
    Returns the froi maps, shape (n_runs, n_voxels).
    """
    os.makedirs(get_subject_froi_folder(subject, froi.task), exist_ok=True)
    runs = get_runs(subject, froi.task)

    data = []
    for run in runs:
        file_path = get_froi_path(
            subject,
            froi.task,
            run if not orthogonal else f"orth{run}",
            froi.contrasts,
            froi.conjunction_type,
            froi.threshold_type,
            froi.threshold_value,
            froi.parcels,
            create=True,
        )
        if not os.path.exists(file_path):
            # If any of the froi masks are missing, redo the process
            data = []
            break
        img = load_img(file_path)
        data.append(img.get_fdata().flatten())

    if data:
        return np.stack(data, axis=0)

    parcels_img, _ = get_parcels(froi.parcels)
    if parcels_img is None:
        raise ValueError("Parcels image not found.")

    if orthogonal:
        p_maps = []
        for contrast in froi.contrasts:
            p_maps.append(
                _get_contrast_orth(subject, froi.task, contrast, "p")
            )
        p_maps = np.stack(p_maps, axis=0)
    else:
        p_maps = []
        for contrast in froi.contrasts:
            p_maps.append(_get_contrast_run(subject, froi.task, contrast, "p"))
        p_maps = np.stack(p_maps, axis=0)
    parcels_data = parcels_img.get_fdata()
    froi_mask = _generate_froi_mask(
        p_maps,
        parcels_data.flatten(),
        froi.conjunction_type,
        froi.threshold_type,
        froi.threshold_value,
    )

    # save the fROI mask
    for run_i, run in enumerate(runs):
        froi_path = get_froi_path(
            subject,
            froi.task,
            run if not orthogonal else f"orth{run}",
            froi.contrasts,
            froi.conjunction_type,
            froi.threshold_type,
            froi.threshold_value,
            froi.parcels,
            create=True,
        )
        froi_img = new_img_like(
            parcels_img,
            froi_mask[run_i].reshape(parcels_data.shape).astype(np.float32),
        )
        froi_img.to_filename(froi_path)

    return froi_mask


def _generate_froi_mask(
    p_maps: np.ndarray,
    parcels: np.ndarray,
    conjunction_type: str,
    threshold_type: str,
    threshold_value: float,
) -> np.ndarray:
    """
    Generate the binary activation mask using p maps.

    Parameters
    ----------
    p_maps : np.ndarray, shape (n_contrast, n_runs, n_voxels)
        The p-map data.
    parcels : np.ndarray, shape (n_voxels)
        The parcel data.
    conjunction_type : str
        The conjunction type.
        Options are 'min', 'max', 'sum', 'prod', 'and', or 'or'.
    threshold_type : str
        The threshold type.
        Options are 'n', 'percent', 'fdr', 'bonferroni', or 'none'.
    threshold_value : float
        The threshold value. If p-value thresholding is used, the threshold
        value corresponds to the alpha level.

    Returns
    -------
    froi_mask : np.ndarray, shape (n_runs, n_voxels)
        The fROI masks.
    """
    assert (
        len(p_maps.shape) == 3
    ), "p_maps should have shape (n_contrast, n_runs, n_voxels)"
    assert len(parcels.shape) == 1, "parcels should have shape (n_voxels)"

    parcels = np.round(parcels).astype(int)
    labels = np.unique(parcels)
    labels = labels[(labels != 0) & (~np.isnan(labels))]

    # Expand parcel labels to a new axis
    parcels_expanded = (parcels == labels[:, None]).astype(float)
    parcels_expanded[parcels_expanded == 0] = np.nan

    p_maps_expanded = (
        p_maps[np.newaxis, :, :, :]
        * parcels_expanded[:, np.newaxis, np.newaxis, :]
    )  # n_labels x n_contrast x n_runs x n_voxels

    # in mask, NaN p_maps are set to inf
    p_maps_expanded[
        (np.isnan(p_maps_expanded))
        & (~np.isnan(parcels_expanded[:, np.newaxis, np.newaxis, :]))
    ] = np.inf

    froi_mask = []
    for i in range(p_maps_expanded.shape[0]):
        froi_mask.append(
            _create_p_map_mask(
                p_maps_expanded[i],
                conjunction_type,
                threshold_type,
                threshold_value,
            )
        )
    froi_mask = np.stack(froi_mask, axis=0)

    # Put labels back as values
    froi_mask_argmax = np.argmax(froi_mask, axis=0)
    froi_mask = labels[froi_mask_argmax] * (np.sum(froi_mask, axis=0) != 0)

    return froi_mask
