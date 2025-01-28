from . import get_bids_deriv_folder
from nilearn.glm import compute_fixed_effects
from nilearn.image import load_img, new_img_like
from nilearn.glm import expression_to_contrast_vector
from scipy.stats import t as t_dist
from typing import List, Tuple, Optional, Union
import os
import pandas as pd
import ast
import warnings
from nibabel.nifti1 import Nifti1Image
import numpy as np
from statsmodels.stats.multitest import fdrcorrection
import scipy
from .utils import validate_arguments, _get_orthogonalized_run_labels
import glob
import re
import logging


_get_contrast_folder = lambda subject, task: (
    get_bids_deriv_folder()
    / f"first_level_{task}"
    / f"sub-{subject}"
    / "contrasts"
)
_get_contrast_info_path = lambda subject, task: (
    _get_contrast_folder(subject, task)
    / f"sub-{subject}_task-{task}_contrasts.csv"
)
_get_contrast_path = lambda subject, task, run_label, contrast, type: (
    _get_contrast_folder(subject, task)
    / f"sub-{subject}_task-{task}_run-{run_label}_contrast-{contrast}_{type}.nii.gz"
)

_get_model_folder = lambda subject, task: (
    get_bids_deriv_folder()
    / f"first_level_{task}"
    / f"sub-{subject}"
    / "models"
)
_get_dof_path_bids = lambda subject, task: (
    _get_model_folder(subject, task) / f"sub-{subject}_task-{task}_dof.csv"
)
_get_design_matrix_path = lambda subject, task: (
    _get_model_folder(subject, task)
    / f"sub-{subject}_task-{task}_design-matrix.csv"
)


def _get_contrast_vector(
    subject: str, task: str, contrast: str
) -> List[float]:
    contrast_info_path = _get_contrast_info_path(subject, task)
    if not contrast_info_path.exists():
        return None

    contrast_info = pd.read_csv(contrast_info_path)
    contrast_info = contrast_info[contrast_info["contrast"] == contrast]
    if contrast_info.empty:
        return None

    contrast_info["vector"] = contrast_info["vector"].apply(ast.literal_eval)
    return contrast_info.to_dict(orient="records")[0]["vector"]


def _get_contrast_runs(subject: str, task: str, contrast: str) -> List[str]:
    """
    Search for all numerically labeled runs for a contrast.
    """
    base_pattern = _get_contrast_path(subject, task, "*", contrast, "*")
    matched_files = glob.glob(str(base_pattern))
    run_numbers = []
    run_pattern = re.compile(r"run-(\d+)")
    for file_path in matched_files:
        match = run_pattern.search(file_path)
        if match:
            run_number = match.group(1)
            if run_number not in run_numbers:
                run_numbers.append(run_number)
    return sorted(run_numbers)


def _get_contrast_runs_by_group(
    subject: str, task: str, contrast: str, run_label: str
) -> List[str]:
    """
    Search for all numerically labeled runs for a contrast.
    """
    runs = _get_contrast_runs(subject, task, contrast)
    if run_label == "all":
        return runs
    elif run_label == "odd":
        return [run for run in runs if int(run) % 2 == 1]
    elif run_label == "even":
        return [run for run in runs if int(run) % 2 == 0]
    elif "orth" in run_label:
        return [run for run in runs if run not in run_label[5:]]
    else:
        return [run_label]


@validate_arguments(
    group={1, 2},
    type={"effect", "t", "variance", "p"},
    orthogonalization={"all-but-one", "odd-even"},
)
def _get_orthogonalized_contrast_data(
    subject: str,
    task: str,
    contrast: str,
    group: int,
    type: str,
    orthogonalization: Optional[str] = "all-but-one",
) -> Tuple[np.ndarray, List[str]]:
    """
    Get the orthogonalized contrast data.

    :return: The contrast maps, shape (n_runs, n_voxels) and the run labels.
        If any of the contrast maps is not found, return None, None.
    :rtype: Tuple[np.ndarray, List[str]]
    """
    runs = _get_contrast_runs(subject, task, contrast)
    labels = _get_orthogonalized_run_labels(runs, group, orthogonalization)

    data = []
    for label in labels:
        dat = _get_contrast_data(subject, task, label, contrast, type)
        if dat is None:
            return None, None
        data.append(dat)
    return np.array(data), labels


@validate_arguments(type={"effect", "t", "variance", "p"})
def _get_contrast_data(
    subject: str, task: str, run_label: str, contrast: str, type: str
) -> np.ndarray:
    """
    Get the contrast data by run label.

    :return: The contrast map, shape (n_voxels,). If the contrast map is not
        found, return None.
    :rtype: np.ndarray
    """
    contrast_img_path = _get_contrast_path(
        subject, task, run_label, contrast, type
    )
    if not contrast_img_path.exists():
        return None
    dat = load_img(contrast_img_path).get_fdata().flatten()
    if type == "p":
        dat[dat == 0] = np.nan
    return dat


def _get_design_matrix(subject: str, task: str) -> pd.DataFrame:
    """
    Get the design matrix for a subject and task.
    """
    design_matrix_path = _get_design_matrix_path(subject, task)
    if not design_matrix_path.exists():
        return None
    return pd.read_csv(design_matrix_path, index_col=0)


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

    design_matrix = _get_design_matrix(subject, task_1)
    if design_matrix is None:
        raise ValueError(
            f"Design matrix not found for {subject} and {task_1}. "
            "Cannot check orthogonality."
        )
    X = design_matrix.values
    _, singular_values, Vt = scipy.linalg.svd(X, full_matrices=False)
    tol = max(X.shape) * np.finfo(float).eps * max(np.abs(singular_values))
    rank = np.sum(singular_values > tol)
    XpX = Vt[:rank].T @ np.diag(singular_values[:rank] ** 2) @ Vt[:rank]

    for contrast_1 in contrasts_1:
        for contrast_2 in contrasts_2:
            vector1 = _get_contrast_vector(subject, task_1, contrast_1)
            if vector1 is None:
                raise ValueError(
                    f"Contrast vector not found for {subject} {task_1} "
                    f"{contrast_1}. Cannot check orthogonality."
                )
            vector2 = _get_contrast_vector(subject, task_2, contrast_2)
            if vector2 is None:
                raise ValueError(
                    f"Contrast vector not found for {subject} {task_2} "
                    f"{contrast_2}. Cannot check orthogonality."
                )
            c = np.stack([vector1, vector2], axis=0)
            if np.abs(c @ XpX @ c.T)[0, 1] >= tol:
                return False
    return True
