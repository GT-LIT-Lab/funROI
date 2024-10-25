from .utils import (
    get_contrast_info_path,
    get_contrast_path,
    get_dof_path,
    get_runs,
)
from nilearn.glm import compute_fixed_effects
from nilearn.image import load_img
from scipy.stats import t as t_dist
from typing import List, Union, Optional, Tuple
import os
import pandas as pd
import ast
import warnings
from nibabel.nifti1 import Nifti1Image
import numpy as np


def register_contrast(
    subject: str, task: str, contrast_name: str, contrast_vector: List[float]
):
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
    contrast_info_path = get_contrast_info_path(subject, task)
    if not os.path.exists(contrast_info_path):
        return None

    contrast_info = pd.read_csv(contrast_info_path)
    contrast_info = contrast_info[contrast_info["contrast"] == contrast]
    contrast_info["vector"] = contrast_info["vector"].apply(ast.literal_eval)
    if contrast_info.empty:
        return None

    if len(contrast_info) > 1:
        warnings.warn(
            f"Multiple contrasts with the same name {contrast} found for "
            f"subject {subject}, task {task}. Using the first one."
        )
    return contrast_info.to_dict(orient="records")[0]


def get_contrasts_all_multi_task(
    subject: str, task_contrasts: List[Tuple[str, List[str]]], type: str
) -> np.ndarray:
    """
    Get the all-run contrast maps for tasks and contrasts for a subject.

    Parameters
    ----------
    subject : str
        Subject ID.
    task_contrasts : List[Tuple[str, List[str]]]
        List of (task, contrasts) tuples.
    type : str
        Type of the contrast map. Can be 'effect', 'variance', 't', 'z', or 'p'.

    Returns
    -------
    np.ndarray
        Contrast maps for the tasks and contrasts,
        shape (n_total_contrasts, 1, n_voxels).
    """
    data = []
    for task, contrasts in task_contrasts:
        for contrast in contrasts:
            contrast_path = get_contrast_path(
                subject, task, "all", contrast, type
            )
            if not os.path.exists(contrast_path):
                img = create_contrast(subject, task, "all", contrast, type)
            else:
                img = load_img(contrast_path)
            flattened_data = img.get_fdata().flatten()
            data.append(flattened_data[np.newaxis, :])
    return np.stack(data, axis=0)


def get_contrasts_orth_single_task(
    subject: str, task: str, contrasts: List[str], type: str
):
    """
    Get the orthogonal-run contrast maps for a single task and contrast for a
    subject.

    Parameters
    ----------
    subject : str
        Subject ID.
    task : str
        Task name.
    contrasts : List[str]
        List of contrast names.
    type : str
        Type of the contrast map. Can be 'effect', 'variance', 't', 'z', or
        'p'.

    Returns
    -------
    np.ndarray
        Contrast maps for the task and contrasts,
        shape (n_contrasts, n_runs, n_voxels).
    """
    data = []
    for contrast in contrasts:
        data_contrast = []
        runs = get_runs(subject, task)
        for run in runs:
            contrast_path = get_contrast_path(
                subject, task, f"orth{run}", contrast, type
            )
            if not os.path.exists(contrast_path):
                img = create_contrast(
                    subject, task, f"orth{run}", contrast, type
                )
            else:
                img = load_img(contrast_path)
            flattened_data = img.get_fdata().flatten()
            data_contrast.append(flattened_data)
        data.append(np.stack(data_contrast, axis=0))
    return np.stack(data, axis=0)


def get_contrasts_runs_single_task(
    subject: str, task: str, contrasts: List[str], type: str
):
    """
    Get the run-wise contrast maps for a single task and contrast for a
    subject.

    Parameters
    ----------
    subject : str
        Subject ID.
    task : str
        Task name.
    contrasts : List[str]
        List of contrast names.
    type : str
        Type of the contrast map. Can be 'effect', 'variance', 't', 'z', or
        'p'.

    Returns
    -------
    np.ndarray
        Contrast maps for the task and contrasts,
        shape (n_contrasts, n_runs, n_voxels).
    """
    data = []
    for contrast in contrasts:
        data_contrast = []
        runs = get_runs(subject, task)
        for run in runs:
            contrast_path = get_contrast_path(
                subject, task, run, contrast, type
            )
            if not os.path.exists(contrast_path):
                img = create_contrast(subject, task, run, contrast, type)
            else:
                img = load_img(contrast_path)
            flattened_data = img.get_fdata().flatten()
            data_contrast.append(flattened_data)
        data.append(np.stack(data_contrast, axis=0))
    return np.stack(data, axis=0)


def create_contrast(
    subject: str, task: str, run_label: str, contrast: str, type: str
) -> Nifti1Image:
    contrast_info = get_contrast_info(subject, task, contrast)
    assert (
        contrast_info is not None
    ), f"Contrast {contrast} not found for subject {subject}, task {task}."

    if run_label.isdigit():
        raise ValueError(
            (
                "Except for fixed effects, run-wise contrast maps need to be "
                "created using the first level model. using Nilearn first "
                "level model in use, please create the contrasts first."
            )
        )
    elif "orth" in run_label:
        contrast_img = _create_contrast_orthogonal(
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


def _create_contrast_orthogonal(
    subject: str,
    task: str,
    exclude_run_label: str,
    contrast_name: str,
    type: str,
) -> Nifti1Image:
    runs = get_runs(subject, task)
    runs.remove(exclude_run_label)
    return _create_contrast_multi_run(subject, task, runs, contrast_name, type)


def _create_contrast_multi_run(
    subject: str,
    task: str,
    runs: List[int],
    contrast_name: str,
    type: str,
) -> Nifti1Image:
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
                (
                    f"Contrast maps not found for subject {subject}, task "
                    f"{task}, run {run_i}. If you are using Nilearn first "
                    "level model, please create the contrasts first."
                )
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
    if type == "p":
        p_values = 1 - t_dist.cdf(t_map.get_fdata(), dof_runs[0])
        return Nifti1Image(p_values, affine=t_map.affine, header=t_map.header)
    else:
        raise ValueError(f"Map type {type} not supported")


def flatten_task_contrasts(
    task_contrasts: List[Tuple[str, List[str]]]
) -> Tuple[List[str], List[str]]:
    tasks = []
    contrasts = []
    for task, task_contrast in task_contrasts:
        tasks.extend([task] * len(task_contrast))
        contrasts.extend(task_contrast)
    return tasks, contrasts
