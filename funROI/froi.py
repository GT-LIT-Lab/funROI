import os
from .utils import (
    get_froi_path,
    get_runs,
    get_parcels,
    get_subject_froi_folder,
)
from .contrast import (
    get_contrasts_all_multi_task,
    get_contrasts_orth_single_task,
    get_contrasts_runs_single_task,
)
from typing import List, Union, Tuple, Optional, Dict
from nibabel.nifti1 import Nifti1Image
import numpy as np
from .localizer import (
    get_localizer_info,
    create_p_map_mask,
    register_localizer,
)
from nilearn.image import load_img, new_img_like
from collections import namedtuple
import logging

LOGGER = logging.getLogger(__name__)

FROI = namedtuple(
    "FROI",
    [
        "localizer",
        "contrasts",
        "conjunctionType",
        "thresholdType",
        "thresholdValue",
        "parcels",
    ],
)

FROI.__new__.__defaults__ = (None,) * len(FROI._fields)


def get_frois(
    subjects: List[str], task_frois: List[Tuple[str, List[FROI]]]
) -> Dict[str, Dict[Tuple[str, str], Nifti1Image]]:
    """
    Get the froi maps for tasks for a subject.

    Parameters
    ----------
    subject : str
        The subject ID.
    task_frois : List[Tuple[str, List[FROI]]]
        List of (task, frois) tuples.

    Returns
    -------
    Dict[str, Dict[Tuple[str, str], np.ndarray]]
        The froi maps for the tasks,
        Each key is a subject ID, and each value is a dictionary with
        (task, localizer) tuple as key and froi map as value.
    """
    data = {}
    for subject in subjects:
        data[subject] = {}
        _ = _get_frois_all_multi_task(subject, task_frois)
        for task, frois in task_frois:
            for froi in frois:
                data[subject][(task, froi.localizer)] = load_img(
                    get_froi_path(
                        subject,
                        task,
                        "all",
                        froi.localizer,
                        froi.thresholdType,
                        froi.thresholdValue,
                        froi.parcels,
                    )
                )
    return data


def _fill_froi_info(subject: str, task: str, froi: FROI) -> FROI:
    """
    Fill in the missing fields in the FROI namedtuple.

    Parameters
    ----------
    subject : str
        The subject ID.
    task : str
        The task name.
    froi : FROI
        The FROI namedtuple.

    Returns
    -------
    FROI
        The FROI namedtuple with filled in fields.
    """
    localizer_info = get_localizer_info(subject, task, froi.localizer)
    if froi.contrasts is None or froi.conjunctionType is None:
        assert localizer_info is not None, (
            f"Localizer {froi.localizer} not found for subject {subject}. "
            "Please provide contrasts and conjunctionType."
        )
        froi = froi._replace(
            contrasts=localizer_info["contrasts"],
            conjunctionType=localizer_info["conjunctionType"],
        )
    elif localizer_info is None:
        register_localizer(
            [subject],
            froi.localizer,
            task,
            froi.contrasts,
            froi.conjunctionType,
        )

    return froi


def _get_frois_all_multi_task(
    subject: str,
    task_frois: List[Tuple[str, List[FROI]]],
) -> np.ndarray:
    """
    Get the all-run froi maps for tasks for a subject.

    Parameters
    ----------
    subject : str
        The subject ID.
    task_frois : List[Tuple[str, List[FROI]]]
        List of (task, frois) tuples.

    Returns
    -------
    np.ndarray
        The froi maps for the tasks,
        shape (n_total_frois, 1, n_voxels).
    """
    os.makedirs(get_subject_froi_folder(subject), exist_ok=True)
    data = []
    for task, frois in task_frois:
        for froi in frois:
            froi = _fill_froi_info(subject, task, froi)
            froi_path = get_froi_path(
                subject,
                task,
                "all",
                froi.localizer,
                froi.thresholdType,
                froi.thresholdValue,
                froi.parcels,
            )
            if os.path.exists(froi_path):
                img = load_img(froi_path)
                data.append(img.get_fdata().flatten()[np.newaxis, :])
                continue

            parcels_img = get_parcels(froi.parcels)
            assert parcels_img is not None, "Parcels image not found."

            p_maps = get_contrasts_all_multi_task(
                subject, [(task, froi.contrasts)], "p"
            )
            parcels_data = parcels_img.get_fdata()
            froi_mask = _generate_froi_mask(
                p_maps,
                parcels_data.flatten(),
                froi.conjunctionType,
                froi.thresholdType,
                froi.thresholdValue,
            )
            data.append(froi_mask)

            # save the froi mask
            froi_img = new_img_like(
                parcels_img, froi_mask[0].reshape(parcels_data.shape)
            )
            froi_img.to_filename(froi_path)

    return np.stack(data, axis=0)


def _get_frois_orth_single_task(
    subject: str, task: str, frois: List[FROI]
) -> np.ndarray:
    """
    Get the orthogonal-run froi maps for tasks for a subject.

    Parameters
    ----------
    subject : str
        The subject ID.
    task : str
        The task name.
    frois : List[FROI]
        List of frois.

    Returns
    -------
    np.ndarray
        The froi maps for the tasks,
        shape (n_frois, n_runs, n_voxels).
    """
    return _get_frois_runs_single_task(subject, task, frois, orthogonal=True)


def _get_frois_runs_single_task(
    subject: str,
    task: str,
    frois: List[FROI],
    orthogonal: Optional[bool] = False,
) -> np.ndarray:
    """
    Get the orthogonal-run froi maps for tasks for a subject.

    Parameters
    ----------
    subject : str
        The subject ID.
    task : str
        The task name.
    frois : List[FROI]
        List of frois.
    orthogonal : Optional[bool]
        Whether to use orthogonal runs.

    Returns
    -------
    np.ndarray
        The froi maps for the tasks,
        shape (n_frois, n_runs, n_voxels).
    """
    os.makedirs(get_subject_froi_folder(subject), exist_ok=True)
    runs = get_runs(subject, task)

    data = []
    for froi in frois:
        froi = _fill_froi_info(subject, task, froi)
        data_froi = []
        for run in runs:
            file_path = get_froi_path(
                subject,
                task,
                run if not orthogonal else f"orth{run}",
                froi.localizer,
                froi.thresholdType,
                froi.thresholdValue,
                froi.parcels,
            )
            if not os.path.exists(file_path):
                # If any of the froi masks are missing, redo the process
                data_froi = []
                break
            img = load_img(file_path)
            data_froi.append(img.get_fdata().flatten())

        if data_froi:
            data.append(np.stack(data_froi, axis=0))
            continue

        parcels_img = get_parcels(froi.parcels)
        assert parcels_img is not None, "Parcels image not found."

        if orthogonal:
            p_maps = get_contrasts_orth_single_task(
                subject, task, froi.contrasts, "p"
            )
        else:
            p_maps = get_contrasts_runs_single_task(
                subject, task, froi.contrasts, "p"
            )
        parcels_data = parcels_img.get_fdata()
        froi_mask = _generate_froi_mask(
            p_maps,
            parcels_data.flatten(),
            froi.conjunctionType,
            froi.thresholdType,
            froi.thresholdValue,
        )
        data.append(froi_mask)

        # save the froi mask
        for run_i, run in enumerate(runs):
            froi_path = get_froi_path(
                subject,
                task,
                run if not orthogonal else f"orth{run}",
                froi.localizer,
                froi.thresholdType,
                froi.thresholdValue,
                froi.parcels,
            )
            froi_img = new_img_like(
                parcels_img, froi_mask[run_i].reshape(parcels_data.shape)
            )
            froi_img.to_filename(froi_path)

    return np.stack(data, axis=0)


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
            create_p_map_mask(
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


def flatten_task_frois_labels(
    task_frois: List[Tuple[str, List[FROI]]]
) -> Tuple[List[str], List[str]]:
    """
    Flatten the task frois labels for tasks.

    Parameters
    ----------
    task_frois : List[Tuple[str, List[FROI]]]
        List of (task, frois) tuples.

    Returns
    -------
    Tuple[List[str], List[str]]
        The task frois labels for tasks
    """
    task_labels = []
    froi_labels = []
    for task, frois in task_frois:
        task_labels.extend([task] * len(frois))
        froi_labels.extend([froi.localizer for froi in frois])
    return task_labels, froi_labels
