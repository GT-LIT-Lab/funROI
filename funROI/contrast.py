from .utils import (
    get_contrast_info_path,
    get_contrast_path,
    get_dof_path,
    get_runs,
    validate_arguments,
    get_design_matrix,
)
from nilearn.glm import compute_fixed_effects
from nilearn.image import load_img
from scipy.stats import t as t_dist
from typing import List, Optional
import os
import pandas as pd
import ast
import warnings
from nibabel.nifti1 import Nifti1Image
import numpy as np
from statsmodels.stats.multitest import fdrcorrection
import scipy


def register_contrast(
    subject: str, task: str, contrast_name: str, contrast_vector: List[float]
):
    """
    Register a contrast for a subject and task.

    Parameters
    ----------
    subject : str
        Subject ID.
    task : str
        Task name.
    contrast_name : str
        Contrast name.
    contrast_vector : List[float]
        Contrast vector, defining the contrast in terms of the design matrix.
    """
    contrast_info_path = get_contrast_info_path(subject, task)
    if not os.path.exists(contrast_info_path):
        contrast_info = pd.DataFrame(columns=["contrast", "vector"])
        matched = contrast_info[contrast_info["contrast"] == contrast_name]
        if not matched.empty:
            raise ValueError(
                f"Contrast {contrast_name} already exists for subject {subject}, task {task}."
            )
    else:
        contrast_info = pd.read_csv(contrast_info_path)

    contrast_info = pd.concat(
        [
            contrast_info,
            pd.DataFrame(
                {"contrast": [contrast_name], "vector": [contrast_vector]}
            ),
        ],
        ignore_index=True,
    )

    contrast_info.to_csv(contrast_info_path, index=False)


def get_contrast_info(subject: str, task: str, contrast: str) -> dict:
    """
    Get the contrast information for a subject and task.

    Parameters
    ----------
    subject : str
        Subject ID.
    task : str
        Task name.
    contrast : str
        Contrast name.
    """
    contrast_info_path = get_contrast_info_path(subject, task)
    if not os.path.exists(contrast_info_path):
        return None

    contrast_info = pd.read_csv(contrast_info_path)
    contrast_info = contrast_info[contrast_info["contrast"] == contrast]
    if contrast_info.empty:
        return None

    contrast_info["vector"] = contrast_info["vector"].apply(ast.literal_eval)

    if len(contrast_info) > 1:
        warnings.warn(
            f"Multiple contrasts with the same name {contrast} found for "
            f"subject {subject}, task {task}. Using the first one."
        )
    return contrast_info.to_dict(orient="records")[0]


@validate_arguments(type={"effect", "variance", "t", "z", "p"})
def _get_contrast_all(
    subject: str, task: str, contrast: str, type: str
) -> np.ndarray:
    """
    Get the all-run contrast maps for tasks and contrasts for a subject.
    Returns the contrast maps, shape (1, n_voxels).
    """
    contrast_path = get_contrast_path(subject, task, "all", contrast, type)
    if not os.path.exists(contrast_path):
        img = _create_contrast(subject, task, "all", contrast, type)
    else:
        img = load_img(contrast_path)
    flattened_data = img.get_fdata().flatten()
    return flattened_data[np.newaxis, :]


@validate_arguments(type={"effect", "variance", "t", "z", "p"})
def _get_contrast_orth(subject: str, task: str, contrast: str, type: str):
    """
    Get the orthogonal-run contrast maps for a single task and contrast for a
    subject. Returns the contrast maps, shape (n_runs, n_voxels).
    """
    data = []
    runs = get_runs(subject, task)
    for run in runs:
        contrast_path = get_contrast_path(
            subject, task, f"orth{run}", contrast, type
        )
        if not os.path.exists(contrast_path):
            img = _create_contrast(subject, task, f"orth{run}", contrast, type)
        else:
            img = load_img(contrast_path)
        flattened_data = img.get_fdata().flatten()
        data.append(flattened_data)
    return np.stack(data, axis=0)


@validate_arguments(type={"effect", "variance", "t", "z", "p"})
def _get_contrast_run(subject: str, task: str, contrast: str, type: str):
    """
    Get the run-wise contrast maps for a single task and contrast for a
    subject. Returns the contrast maps, shape (n_runs, n_voxels).
    """
    data = []
    runs = get_runs(subject, task)
    for run in runs:
        contrast_path = get_contrast_path(subject, task, run, contrast, type)
        if not os.path.exists(contrast_path):
            img = _create_contrast(subject, task, run, contrast, type)
        else:
            img = load_img(contrast_path)
        flattened_data = img.get_fdata().flatten()
        data.append(flattened_data)
    return np.stack(data, axis=0)


@validate_arguments(type={"effect", "variance", "t", "z", "p"})
def _create_contrast(
    subject: str, task: str, run_label: str, contrast: str, type: str
) -> Nifti1Image:
    """
    Create a contrast map with a registered defintion.
    Returns the created contrast image.
    """
    contrast_info = get_contrast_info(subject, task, contrast)
    if contrast_info is None:
        raise ValueError(
            f"Info for {contrast} not found for subject {subject}, task {task}."
        )

    if run_label.isdigit():
        raise ValueError(
            "Except for fixed effects, run-wise contrast maps need to be "
            "created using the first level model. using Nilearn first level "
            "model in use, please create the contrasts first."
        )
    elif "orth" in run_label:
        contrast_img = _create_contrast_orth(
            subject,
            task,
            int(run_label.replace("orth", "")),
            contrast,
            type,
        )
    elif "all" in run_label:
        contrast_img = _create_contrast_multi_run(
            subject,
            task,
            get_runs(subject, task),
            contrast,
            type,
        )

    contrast_path = get_contrast_path(subject, task, run_label, contrast, type)
    contrast_img.to_filename(contrast_path)
    return contrast_img


@validate_arguments(type={"effect", "variance", "t", "z", "p"})
def _create_contrast_orth(
    subject: str,
    task: str,
    exclude_run_label: str,
    contrast_name: str,
    type: str,
) -> Nifti1Image:
    """
    Create a contrast map with a registered defintion for all-but-one runs.
    Returns the created contrast image.
    """
    runs = get_runs(subject, task)
    runs.remove(exclude_run_label)
    return _create_contrast_multi_run(subject, task, runs, contrast_name, type)


@validate_arguments(type={"effect", "variance", "t", "z", "p"})
def _create_contrast_multi_run(
    subject: str,
    task: str,
    runs: List[int],
    contrast_name: str,
    type: str,
) -> Nifti1Image:
    """
    Create a contrast map with a registered defintion for multiple runs.
    Returns the created contrast image.
    """
    effect_runs = []
    variance_runs = []
    dof_runs = []
    for run_i in runs:
        effect_map_path = get_contrast_path(
            subject, task, run_i, contrast_name, "effect"
        )
        variance_map_path = get_contrast_path(
            subject, task, run_i, contrast_name, "variance"
        )
        if not os.path.exists(effect_map_path) or not os.path.exists(
            variance_map_path
        ):
            raise ValueError(
                f"Contrast maps not found for subject {subject}, task {task}, "
                f"run {run_i}. If you are using Nilearn first level model, "
                "please create the contrasts first."
            )
        else:
            effect_map = load_img(effect_map_path)
            variance_map = load_img(variance_map_path)
        effect_runs.append(effect_map)
        variance_runs.append(variance_map)

        dof_path = get_dof_path(subject, task)
        dofs = pd.read_csv(dof_path)
        dof = dofs[(dofs["task"] == task) & (dofs["run"] == run_i)][
            "dof"
        ].values[0]
        dof_runs.append(dof)

    contrast_map, variance_map, t_map, z_map = compute_fixed_effects(
        effect_runs, variance_runs, dofs=dof_runs, return_z_score=True
    )
    if type == "effect":
        return contrast_map
    elif type == "variance":
        return variance_map
    elif type == "t":
        return t_map
    elif type == "z":
        return z_map
    elif type == "p":
        p_values = 1 - t_dist.cdf(t_map.get_fdata(), dof_runs[0])
        return Nifti1Image(p_values, affine=t_map.affine, header=t_map.header)


@validate_arguments(
    threshold_type={"n", "percent", "fdr", "bonferroni", "none"},
)
def _threshold_p_map(
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

    froi_mask = np.moveaxis(froi_mask, 0, -1)
    return froi_mask


@validate_arguments(
    conjunction_type={"min", "max", "sum", "prod", "and", "or"},
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
    assert (
        data.ndim == 3
    ), "data should have shape (n_contrast, n_runs, n_voxels)"

    if mask is not None:
        data[:, :, mask] = np.nan

    if conjunction_type in ["min", "max", "sum", "prod"]:
        combined_data = eval(f"np.{conjunction_type}(data, axis=-3)")
        froi_mask = _threshold_p_map(
            combined_data, threshold_type, threshold_value
        )
    elif conjunction_type in ["and", "or"]:
        thresholded_data = _threshold_p_map(
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


def _check_orthogonal(
    subject: str,
    task_1: str,
    contrasts_1: List[str],
    task_2: str,
    contrasts_2: List[str],
):
    """
    Check if two set of contrasts are orthogonal.
    """
    if task_1 != task_2:
        return True

    X = get_design_matrix(subject, task_1).values
    _, singular_values, Vt = scipy.linalg.svd(X, full_matrices=False)
    tol = max(X.shape) * np.finfo(float).eps * max(np.abs(singular_values))
    rank = np.sum(singular_values > tol)
    XpX = Vt[:rank].T @ np.diag(singular_values[:rank] ** 2) @ Vt[:rank]

    for contrast_1 in contrasts_1:
        for contrast_2 in contrasts_2:
            vector1 = np.array(
                get_contrast_info(subject, task_1, contrast_1)["vector"]
            )
            vector2 = np.array(
                get_contrast_info(subject, task_2, contrast_2)["vector"]
            )
            c = np.stack([vector1, vector2], axis=0)
            if np.abs(c @ XpX @ c.T)[0, 1] >= tol:
                return False
    return True
