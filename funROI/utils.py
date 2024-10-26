from . import get_bids_deriv_folder
import os
from nibabel.nifti1 import Nifti1Image
from nilearn.image import load_img, math_img
from typing import List, Optional, Union, Tuple
import pandas as pd
import warnings
import numpy as np
import ast
import re
import glob
from collections import namedtuple
import json


#### Path setting
# Contrast
def get_subject_contrast_folder(subject: str) -> str:
    return os.path.join(get_bids_deriv_folder(), f"sub-{subject}", "contrasts")


def get_contrast_info_path(subject: str, task: str) -> str:
    return os.path.join(
        get_subject_contrast_folder(subject),
        f"sub-{subject}_task-{task}_contrasts.csv",
    )


def get_contrast_path(
    subject: str, task: str, run_label: str, contrast: str, type: str
) -> str:
    return os.path.join(
        get_subject_contrast_folder(subject),
        f"sub-{subject}_task-{task}_run-{run_label}_contrast-{contrast}_{type}.nii.gz",
    )


# Localizer
def get_subject_localizer_folder(subject: str) -> str:
    return os.path.join(
        get_bids_deriv_folder(), f"sub-{subject}", "localizers"
    )


def get_localizer_info_path(subject: str, task: str) -> str:
    return os.path.join(
        get_subject_localizer_folder(subject),
        f"sub-{subject}_task-{task}_localizers.csv",
    )


def get_localizer_path(
    subject: str,
    task: str,
    run_label: str,
    localizer: str,
    threshold_type: str,
    threshold_value: float,
) -> str:
    return os.path.join(
        get_subject_localizer_folder(subject),
        f"sub-{subject}_task-{task}_run-{run_label}_localizer-{localizer}_thresholdType-{threshold_type}_thresholdValue-{threshold_value}_mask.nii.gz",
    )


# froi
def get_subject_froi_folder(subject: str) -> str:
    return os.path.join(get_bids_deriv_folder(), f"sub-{subject}", "froi")


def get_froi_path(
    subject: str,
    task: str,
    run_label: str,
    localizer: str,
    threshold_type: str,
    threshold_value: float,
    parcels: Nifti1Image,
) -> str:
    return os.path.join(
        get_subject_froi_folder(subject),
        f"sub-{subject}_task-{task}_run-{run_label}_localizer-{localizer}_thresholdType-{threshold_type}_thresholdValue_{threshold_value}_parcels-{parcels}_mask.nii.gz",
    )


# Models
def get_subject_model_folder(subject: str) -> str:
    return os.path.join(get_bids_deriv_folder(), f"sub-{subject}", "models")


def get_dof_path(subject: str, task: str) -> str:
    return os.path.join(
        get_subject_model_folder(subject), f"sub-{subject}_task-{task}_dof.csv"
    )


def get_design_matrix_path(subject: str, task: str, run: int) -> str:
    return os.path.join(
        get_subject_model_folder(subject),
        f"sub-{subject}_task-{task}_run-{run}_design-matrix.csv",
    )


# Parcels
def get_parcels_folder() -> str:
    return os.path.join(get_bids_deriv_folder(), "parcels")


def get_parcels_labels_path(parcel_name: str) -> str:
    return os.path.join(get_parcels_folder(), f"{parcel_name}_labels.json")


def get_parcels_path(parcel_name: str) -> str:
    return os.path.join(get_parcels_folder(), f"{parcel_name}.nii.gz")


def get_parcels_config_path(parcel_name: str) -> str:
    return os.path.join(get_parcels_folder(), f"{parcel_name}.json")


#### Utilities
def get_design_matrix(subject: str, task: str, run: int):
    design_matrix_path = get_design_matrix_path(subject, task, run)
    assert os.path.exists(
        design_matrix_path
    ), f"Design matrix not found for sub-{subject}, task {task}, run {run}."
    return pd.read_csv(design_matrix_path)


def get_runs(subject: str, task: str) -> List[int]:
    template_contrast_path = get_contrast_path(subject, task, "*", "*", "p")
    potential_paths = glob.glob(template_contrast_path)
    pattern = re.compile(
        rf".*{template_contrast_path.replace('run-*', 'run-[0-9]+').replace('contrast-*', 'contrast-.*')}"
    )
    run_paths = [path for path in potential_paths if pattern.match(path)]
    unique_runs = list(
        set(
            [
                int(re.search(r"run-(\d+)", run_path).group(1))
                for run_path in run_paths
            ]
        )
    )
    return unique_runs


def get_parcels(parcels: str) -> Optional[Nifti1Image]:
    parcels_path = get_parcels_path(parcels)
    if not os.path.exists(parcels_path):
        return None
    parcels_img = load_img(parcels_path)
    parcels_img = math_img("np.round(img)", img=parcels_img)
    return parcels_img


def get_parcels_labels(parcels: str) -> Tuple[Nifti1Image, dict]:
    parcels_img = get_parcels(parcels)
    parcels_labels_path = get_parcels_labels_path(parcels)
    if os.path.exists(parcels_labels_path):
        label_dict = json.load(open(parcels_labels_path))
    else:
        label_dict = {}
        for label in np.unique(parcels_img.get_fdata()):
            if label != 0:
                label_dict[int(label)] = int(label)
    return parcels_img, label_dict

def harmonic_mean(data):
    data = np.array(data).flatten()
    return len(data) / np.sum(1 / data)