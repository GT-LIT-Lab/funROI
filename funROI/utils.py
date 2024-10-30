from . import get_bids_deriv_folder, get_analysis_output_folder
import os
from typing import List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import re
import glob
from functools import wraps
from collections import namedtuple

ParcelsConfig = namedtuple(
    "ParcelConfig",
    ["parcels_path", "labels_path"],
)
ParcelsConfig.__new__.__defaults__ = (None,) * len(ParcelsConfig._fields)

FROIConfig = namedtuple(
    "FROIConfig",
    [
        "task",
        "contrasts",
        "conjunction_type",
        "threshold_type",
        "threshold_value",
        "parcels",
    ],
)


#### Path setting
def get_subject_contrast_folder(subject: str, task: str) -> str:
    return os.path.join(
        get_bids_deriv_folder(),
        f"first_level_{task}",
        f"sub-{subject}",
        "contrasts",
    )


def get_contrast_info_path(subject: str, task: str) -> str:
    return os.path.join(
        get_subject_contrast_folder(subject, task),
        f"sub-{subject}_task-{task}_contrasts.csv",
    )


def get_contrast_path(
    subject: str, task: str, run_label: str, contrast: str, type: str
) -> str:
    return os.path.join(
        get_subject_contrast_folder(subject, task),
        f"sub-{subject}_task-{task}_run-{run_label}_contrast-{contrast}_{type}.nii.gz",
    )


def get_subject_froi_folder(subject: str, task: str) -> str:
    return os.path.join(
        get_bids_deriv_folder(),
        f"first_level_{task}",
        f"sub-{subject}",
        "froi",
    )


def get_froi_info_path(subject: str, task: str) -> str:
    return os.path.join(
        get_subject_froi_folder(subject, task),
        f"sub-{subject}_task-{task}_frois.csv",
    )


def get_froi_path(
    subject: str,
    task: str,
    run_label: str,
    contrasts: List[str],
    conjunction_type: str,
    threshold_type: str,
    threshold_value: float,
    parcels: Union[str, ParcelsConfig],
    create: Optional[bool] = False,
) -> str:
    if isinstance(parcels, ParcelsConfig):
        parcels = parcels.parcels_path
    contrasts = str(sorted(contrasts))

    info_path = get_froi_info_path(subject, task)
    if not os.path.exists(info_path):
        id = 0
        if create:
            os.makedirs(get_subject_froi_folder(subject, task), exist_ok=True)
            frois = pd.DataFrame(
                {
                    "id": [id],
                    "contrasts": [contrasts],
                    "conjunction_type": [conjunction_type],
                    "threshold_type": [threshold_type],
                    "threshold_value": [threshold_value],
                    "parcels": [parcels],
                }
            )
            frois.to_csv(info_path, index=False)
    else:
        frois = pd.read_csv(info_path)
        # check match
        frois_matched = frois[
            (frois["contrasts"] == contrasts)
            & (frois["conjunction_type"] == conjunction_type)
            & (frois["threshold_type"] == threshold_type)
            & (frois["threshold_value"] == threshold_value)
            & (frois["parcels"] == parcels)
        ]
        if len(frois_matched) == 0:
            id = frois["id"].max() + 1
            if create:
                frois_new = pd.DataFrame(
                    {
                        "id": [id],
                        "contrasts": [contrasts],
                        "conjunction_type": [conjunction_type],
                        "threshold_type": [threshold_type],
                        "threshold_value": [threshold_value],
                        "parcels": [parcels],
                    }
                )
                frois = pd.concat([frois, frois_new], ignore_index=True)
                frois.to_csv(info_path, index=False)
        else:
            id = frois_matched["id"].values[0]

    id = int(id)
    return os.path.join(
        get_subject_froi_folder(subject, task),
        f"sub-{subject}_task_{task}_run-{run_label}_froi-{id:04d}"
        "_mask.nii.gz",
    )


def get_subject_model_folder(subject: str, task: str) -> str:
    return os.path.join(
        get_bids_deriv_folder(),
        f"first_level_{task}",
        f"sub-{subject}",
        "models",
    )


def get_dof_path(subject: str, task: str) -> str:
    return os.path.join(
        get_subject_model_folder(subject, task),
        f"sub-{subject}_task-{task}_dof.csv",
    )


def get_design_matrix_path(subject: str, task: str) -> str:
    return os.path.join(
        get_subject_model_folder(subject, task),
        f"sub-{subject}_task-{task}_design-matrix.csv",
    )


def get_parcels_folder() -> str:
    return os.path.join(get_analysis_output_folder(), "parcels")


def get_parcels_labels_path(parcel_name: str) -> str:
    return os.path.join(
        get_parcels_folder(), f"parcels-{parcel_name}_labels.json"
    )


def get_parcels_path(parcel_name: str) -> str:
    return os.path.join(get_parcels_folder(), f"parcels-{parcel_name}.nii.gz")


def get_parcels_config_path(parcel_name: str) -> str:
    return os.path.join(
        get_parcels_folder(), f"parcels-{parcel_name}_config.json"
    )


def get_effect_estimation_folder() -> str:
    return os.path.join(get_analysis_output_folder(), "effect")


def get_next_effect_estimation_paths() -> Tuple[str, str]:
    # get_effect_estimation_folder() / "effect-<next number: %04d>.csv"
    effect_estimation_folder = get_effect_estimation_folder()
    effect_estimation_files = glob.glob(
        os.path.join(effect_estimation_folder, "effect-*.csv")
    )
    if len(effect_estimation_files) == 0:
        next_number = 0
    else:
        next_number = (
            max(
                [
                    int(re.search(r"effect-(\d+).csv", file).group(1))
                    for file in effect_estimation_files
                ]
            )
            + 1
        )
    return os.path.join(
        effect_estimation_folder, f"effect-{next_number:04d}.csv"
    ), os.path.join(
        effect_estimation_folder, f"effect-{next_number:04d}_config.json"
    )


def get_spatial_correlation_estimation_folder() -> str:
    return os.path.join(get_analysis_output_folder(), "spatial_correlation")


def get_next_spatial_correlation_estimation_path() -> Tuple[str, str]:
    # get_spatial_correlation_estimation_folder() / "spcorr-<next number: %04d>.csv"
    spcorr_estimation_folder = get_spatial_correlation_estimation_folder()
    spcorr_estimation_files = glob.glob(
        os.path.join(spcorr_estimation_folder, "spcorr-*.csv")
    )
    if len(spcorr_estimation_files) == 0:
        next_number = 0
    else:
        next_number = (
            max(
                [
                    int(re.search(r"spcorr-(\d+).csv", file).group(1))
                    for file in spcorr_estimation_files
                ]
            )
            + 1
        )
    return os.path.join(
        spcorr_estimation_folder, f"spcorr-{next_number:04d}.csv"
    ), os.path.join(
        spcorr_estimation_folder, f"spcorr-{next_number:04d}_config.json"
    )


def get_overlap_estimation_folder() -> str:
    return os.path.join(get_analysis_output_folder(), "overlap")


def get_next_overlap_estimation_path() -> Tuple[str, str]:
    # get_overlap_estimation_folder() / "overlap-<next number: %04d>.csv"
    overlap_estimation_folder = get_overlap_estimation_folder()
    overlap_estimation_files = glob.glob(
        os.path.join(overlap_estimation_folder, "overlap-*.csv")
    )
    if len(overlap_estimation_files) == 0:
        next_number = 0
    else:
        next_number = (
            max(
                [
                    int(re.search(r"overlap-(\d+)\.csv$", file).group(1))
                    for file in overlap_estimation_files
                    if re.search(r"overlap-(\d+)\.csv$", file)
                ]
            )
            + 1
        )
    return os.path.join(
        overlap_estimation_folder, f"overlap-{next_number:04d}.csv"
    ), os.path.join(
        overlap_estimation_folder, f"overlap-{next_number:04d}_froi.csv"
    )


def get_design_matrix(subject: str, task: str):
    design_matrix_path = get_design_matrix_path(subject, task)
    assert os.path.exists(
        design_matrix_path
    ), f"Design matrix not found for sub-{subject}, task {task}."
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


def harmonic_mean(data):
    data = np.array(data).flatten()
    return len(data) / np.sum(1 / data)


def validate_arguments(**valid_options):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for arg_name, valid_vals in valid_options.items():
                if arg_name in kwargs and kwargs[arg_name] not in valid_vals:
                    raise ValueError(
                        f"Invalid {arg_name}: '{kwargs[arg_name]}'. "
                        f"Supported options are: {valid_vals}"
                    )
            return func(*args, **kwargs)

        return wrapper

    return decorator
