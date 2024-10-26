import os
import ast
from typing import List, Optional, Union
import pandas as pd
from .utils import (
    get_subject_localizer_folder,
    get_localizer_info_path,
)
from .contrast import get_contrasts_orth_single_task
import numpy as np
from statsmodels.stats.multitest import fdrcorrection
import warnings
from collections import namedtuple
from .contrast import get_contrast_info

Localizer = namedtuple(
    "Localizer",
    ["localizer", "contrasts", "conjunctionType"],
)
Localizer.__new__.__defaults__ = (None,) * len(Localizer._fields)


def register_localizer(
    subjects: List[str],
    localizer_name: str,
    task: str,
    contrasts: List[str],
    conjunction_type: str,
):
    """
    Register a localizer for subjects.

    Parameters
    ----------
    subjects : List[str]
        List of subject IDs.
    localizer_name : str
        Localizer name.
    task : str
        Task label.
    contrasts : List[str]
        List of contrast names.
    conjunction_type : str
        The conjunction type.
        Options are 'min', 'max', 'sum', 'prod', 'and', or 'or'.
    """
    for subject in subjects:
        os.makedirs(get_subject_localizer_folder(subject), exist_ok=True)
        localizer_info_path = get_localizer_info_path(subject, task)
        if not os.path.exists(localizer_info_path):
            localizer_info = pd.DataFrame(
                columns=["localizer", "contrasts", "conjunctionType"]
            )
            matched = localizer_info[
                localizer_info["localizer"] == localizer_name
            ]
            if not matched.empty:
                raise ValueError(
                    (
                        f"Localizer {localizer_name} already exists for "
                        "subject {subject}."
                    )
                )
        else:
            localizer_info = pd.read_csv(localizer_info_path)

        localizer_info = pd.concat(
            [
                localizer_info,
                pd.DataFrame(
                    {
                        "localizer": [localizer_name],
                        "contrasts": [contrasts],
                        "conjunctionType": [conjunction_type],
                    }
                ),
            ],
            ignore_index=True,
        )

        localizer_info.to_csv(localizer_info_path, index=False)


def get_localizer_info(subject: str, task: str, localizer: str) -> dict:
    contrast = get_contrast_info(subject, task, localizer)
    if contrast is not None:
        return {
            "localizer": localizer,
            "contrasts": [localizer],
            "conjunctionType": "and",
        }

    localizer_info_path = get_localizer_info_path(subject, task)
    if not os.path.exists(localizer_info_path):
        return None

    localizer_info = pd.read_csv(localizer_info_path)
    localizer_info = localizer_info[localizer_info["localizer"] == localizer]
    localizer_info["contrasts"] = localizer_info["contrasts"].apply(
        ast.literal_eval
    )
    if localizer_info.empty:
        return None

    if len(localizer_info) > 1:
        warnings.warn(
            (
                f"Multiple localizers with the same name {localizer} found for"
                f" subject {subject}, task {task}. Using the first one."
            )
        )
    return localizer_info.to_dict(orient="records")[0]


def get_localizers(
    subjects: List[str],
    task: str,
    localizer: Union[str, Localizer],
    threshold_type: str,
    threshold_value: float,
) -> np.ndarray:
    """
    Get the localizer masks for subjects.
    Currently, data across subjects are assumed to be in the same space.

    Parameters
    ----------
    subjects : List[str]
        List of subject IDs.
    localizer : str
        Localizer name.
    threshold_type : str
        The threshold type.
        Options are 'n', 'percent', 'fdr', 'bonferroni', or 'none'.
    threshold_value : float
        The threshold value.

    Returns
    -------
    masks : List[np.ndarray]
        The localizer masks for each subject. Shape (n_runs, n_voxels).
    """
    data = []
    for subject in subjects:
        localizer_info = get_localizer_info(subject, task, localizer)
        if isinstance(localizer, Localizer):
            if localizer_info is None:
                register_localizer(
                    [subject],
                    localizer.localizer,
                    task,
                    localizer.contrasts,
                    localizer.conjunctionType,
                )
                localizer_info = localizer._asdict()
        assert (
            localizer_info is not None
        ), f"Localizer {localizer} not found for subject {subject}."

        p_maps = get_contrasts_orth_single_task(
            subject, task, localizer_info["contrasts"], "p"
        )
        masks = create_p_map_mask(
            p_maps,
            localizer_info["conjunctionType"],
            threshold_type,
            threshold_value,
        )
        data.append(masks)
    return data


def threshold_p_map(
    data: np.ndarray, threshold_type: str, threshold_value: float
) -> np.ndarray:
    """
    Extract voxels from a p-map image based on a threshold

    Parameters
    ----------
    data : np.ndarray
        The p-map data, shape (*, n_voxels).
    threshold_type : str
        The threshold type.
        Options are 'n', 'percent', 'fdr', 'bonferroni', or 'none'.
    threshold_value : float
        The threshold value.
    """
    assert threshold_type in [
        "n",
        "percent",
        "fdr",
        "bonferroni",
        "none",
    ], f"Unknown threshold type: {threshold_type}"

    data = np.moveaxis(data, -1, 0)
    froi_mask = np.zeros_like(data)

    if threshold_type == "n":
        pvals_sorted = np.sort(data, axis=0)
        threshold = pvals_sorted[threshold_value - 1]
        froi_mask[data <= threshold] = 1

    # All-tie-inclusive thresholding
    elif "percent" in threshold_type:
        pvals_sorted = np.sort(data, axis=0)
        n = np.floor(
            threshold_value * np.sum(~np.isnan(data), axis=0, keepdims=True)
        ).astype(int)
        threshold = np.take_along_axis(pvals_sorted, n - 1, axis=0)
        froi_mask[data <= threshold] = 1

    elif threshold_type == "fdr":
        pvals = data.flatten()
        _, pvals_fdr = fdrcorrection(pvals)
        froi_mask.flat[pvals_fdr < threshold_value] = 1
    elif threshold_type == "bonferroni":
        froi_mask.flat[data.flatten() < threshold_value / data.size] = 1
    elif threshold_type == "none":
        froi_mask.flat[data.flatten() < threshold_value] = 1
    else:
        raise ValueError(f"Unknown threshold type: {threshold_type}")

    froi_mask = np.moveaxis(froi_mask, 0, -1)
    return froi_mask


def create_p_map_mask(
    data: np.ndarray,
    conjunction_type: str,
    threshold_type: str,
    threshold_value: float,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create a mask based on a p-map image

    Parameters
    ----------
    data : np.ndarray, shape (n_contrast, n_runs, n_voxels)
        The p-map data.
    conjunction_type : str
        The conjunction type.
        Options are 'min', 'max', 'sum', 'prod', 'and', or 'or'.
    threshold_type : str
        The threshold type.
        Options are 'n', 'percent', 'fdr', 'bonferroni', or 'none'.
    threshold_value : float
        The threshold value.
    mask : np.ndarray, shape (n_voxels), optional
        The explicit mask to be applied before thresholding.

    Returns
    -------
    froi_mask : np.ndarray, shape (n_runs, n_voxels)
        The froi masks.
    """
    assert conjunction_type in [
        "min",
        "max",
        "sum",
        "prod",
        "and",
        "or",
    ], f"Unknown conjunction type: {conjunction_type}"
    assert threshold_type in [
        "n",
        "percent",
        "fdr",
        "bonferroni",
        "none",
    ], f"Unknown threshold type: {threshold_type}"
    assert (
        data.ndim == 3
    ), "data should have shape (n_contrast, n_runs, n_voxels)"

    if mask is not None:
        data[:, :, mask] = np.nan

    if conjunction_type in ["min", "max", "sum", "prod"]:
        combined_data = eval(f"np.{conjunction_type}(data, axis=-3)")
        froi_mask = threshold_p_map(
            combined_data, threshold_type, threshold_value
        )
    elif conjunction_type in ["and", "or"]:
        thresholded_data = threshold_p_map(
            data, threshold_type, threshold_value
        )
        if conjunction_type == "and":
            froi_mask = np.all(thresholded_data, axis=-3)
        elif conjunction_type == "or":
            froi_mask = np.any(thresholded_data, axis=-3)
        else:
            raise ValueError(
                f"Conjunction type {conjunction_type} not supported"
            )
    else:
        raise ValueError(f"Conjunction type {conjunction_type} not supported")

    return froi_mask
