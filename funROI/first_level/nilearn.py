from .. import (
    get_bids_data_folder,
    get_bids_deriv_folder,
    get_bids_preprocessed_folder,
    get_bids_preprocessed_folder_relative,
)

from ..utils import (
    get_subject_model_folder,
    get_subject_contrast_folder,
    get_contrast_path,
    get_dof_path,
    get_contrast_info_path,
    get_design_matrix_path,
    get_runs,
    get_design_matrix,
)

from ..contrast import register_contrast

import os
import pickle
from typing import List, Optional, Union
from nilearn.image import new_img_like, load_img
from nilearn.glm import (
    compute_contrast,
    SimpleRegressionResults,
    expression_to_contrast_vector,
)
from nilearn.glm.first_level import (
    first_level_from_bids,
    make_first_level_design_matrix,
    run_glm,
)
from nibabel.nifti1 import Nifti1Image
from nilearn.interfaces.bids import get_bids_files
import numpy as np
import pandas as pd
import re


#### Nilearn first level
def run_first_level(
    subjects: List[str],
    tasks: List[str],
    space: str,
    res: Optional[int] = None,
    confound_labels: Optional[List[str]] = [],
    **args,
):
    """
    Run first level analysis on BIDS preprocessed data.

    Parameters
    ----------
    subjects : list of str
        List of subject labels.
    tasks : list of str
        List of task labels.
    space : str
        Space of the images.
    res : int, optional
        Resolution of the images.
    confound_labels : list of str, optional
        List of confound labels to include in the design matrix.
    **args
        Additional arguments to pass to make_first_level_design_matrix.
    """
    os.makedirs(get_bids_deriv_folder(), exist_ok=True)
    dofs = {}
    for task in tasks:
        # TODO: If more are added, change the input to get a filter dict
        img_filters = []
        if res is not None:
            img_filters.append(("res", str(res)))

        (models, models_run_imgs, models_events, models_confounds) = (
            first_level_from_bids(
                get_bids_data_folder(),
                task,
                sub_labels=subjects,
                space_label=space,
                derivatives_folder=get_bids_preprocessed_folder_relative(),
                img_filters=img_filters,
                slice_time_ref=None,
            )
        )

        for subject_i in range(len(models)):
            model, imgs, events, confounds = (
                models[subject_i],
                models_run_imgs[subject_i],
                models_events[subject_i],
                models_confounds[subject_i],
            )
            subject_label = model.subject_label
            os.makedirs(
                get_subject_model_folder(subject_label, task), exist_ok=True
            )
            os.makedirs(
                get_subject_contrast_folder(subject_label, task), exist_ok=True
            )

            for run_i in range(len(events)):
                events_i = events[run_i]
                imgs_i = load_img(imgs[run_i])
                frame_times = np.arange(imgs_i.shape[-1]) * model.t_r
                if model.slice_time_ref is not None:
                    frame_times = (
                        frame_times + model.slice_time_ref * model.t_r
                    )
                design_matrix = make_first_level_design_matrix(
                    frame_times=frame_times, events=events_i, **args
                )

                # Add confounds to design matrix
                confounds_i = confounds[run_i]
                for confound_label in confound_labels:
                    confound = confounds_i[confound_label]
                    # Replace NaNs with mean value
                    confound = np.where(
                        np.isnan(confound), np.nanmean(confound), confound
                    )
                    design_matrix[confound_label] = confound

                ses_label = re.search(r"ses-(\w+)_", imgs[run_i]).group(1)
                task_label = re.search(r"task-(\w+)_", imgs[run_i]).group(1)
                run_label = re.search(r"run-(\d+)_", imgs[run_i]).group(1)
                filters = [
                    ("ses", ses_label),
                    ("task", task_label),
                    ("run", run_label),
                    ("space", space),
                    ("desc", "brain"),
                ]
                if res is not None:
                    filters.append(("res", str(res)))

                mask_img = load_img(
                    get_bids_files(
                        get_bids_preprocessed_folder(),
                        sub_label=subject_label,
                        modality_folder="func",
                        filters=filters,
                        file_tag="mask",
                        file_type="nii.gz",
                    )[0]
                )
                mask_img_data = mask_img.get_fdata()
                imgs_i_data = imgs_i.get_fdata()

                # Brain masking
                imgs_i_data[mask_img_data == 0] = np.nan

                labels, estimates = run_glm(
                    imgs_i_data.reshape(-1, imgs_i_data.shape[-1]).T,
                    design_matrix.values,
                )
                for key, value in estimates.items():
                    if not isinstance(value, SimpleRegressionResults):
                        estimates[key] = SimpleRegressionResults(value)

                # Dump labels and estimates
                labels_path, estimates_path = get_first_level_model_paths(
                    subject_label, task, run_i + 1
                )
                with open(labels_path, "wb") as f:
                    pickle.dump(labels, f)
                with open(estimates_path, "wb") as f:
                    pickle.dump(estimates, f)

                # Save design matrix
                design_matrix_path = get_design_matrix_path(
                    subject_label, task
                )
                design_matrix["run"] = run_i + 1
                design_matrix.reset_index(inplace=True, names=["frame_times"])
                design_matrix.set_index(["run", "frame_times"], inplace=True)
                if not os.path.exists(design_matrix_path):
                    design_matrix.to_csv(design_matrix_path, index=True)
                else:
                    design_matrix_prev = pd.read_csv(
                        design_matrix_path, index_col=["run", "frame_times"]
                    )
                    design_matrix = pd.concat(
                        [design_matrix_prev, design_matrix], axis=0
                    )
                    design_matrix.to_csv(design_matrix_path, index=True)

                vars = design_matrix.columns
                for var_i, var in enumerate(vars):
                    var_array = np.zeros((len(vars)))
                    var_array[var_i] = 1

                    maps_i = {}

                    contrast = compute_contrast(
                        labels, estimates, var_array, stat_type="t"
                    )
                    maps_i["effect"] = contrast.effect_size().reshape(
                        imgs_i_data.shape[:-1]
                    )
                    maps_i["variance"] = contrast.effect_variance().reshape(
                        imgs_i_data.shape[:-1]
                    )
                    maps_i["t"] = contrast.stat().reshape(
                        imgs_i_data.shape[:-1]
                    )
                    maps_i["z"] = contrast.z_score().reshape(
                        imgs_i_data.shape[:-1]
                    )
                    maps_i["p"] = contrast.p_value().reshape(
                        imgs_i_data.shape[:-1]
                    )

                    # Export contrast maps
                    for map_type, map_data in maps_i.items():
                        new_img_like(imgs_i, map_data).to_filename(
                            get_contrast_path(
                                subject_label, task, run_i + 1, var, map_type
                            )
                        )

                    if run_i == 0:
                        register_contrast_and_create_maps(
                            subject_label, task, var, var_array
                        )

                # Compute degrees of freedom
                dof = imgs_i_data.shape[-1] - len(vars)
                if subject_label not in dofs:
                    dofs[subject_label] = pd.DataFrame(
                        columns=["task", "run", "dof"]
                    )
                dofs[subject_label] = pd.concat(
                    [
                        dofs[subject_label],
                        pd.DataFrame(
                            {"task": [task], "run": [run_i + 1], "dof": [dof]}
                        ),
                    ]
                )

    for subject, df in dofs.items():
        df.to_csv(get_dof_path(subject, task), index=False)


def get_first_level_model_paths(subject: str, task: str, run: int):
    labels_path = os.path.join(
        get_subject_model_folder(subject, task),
        f"sub-{subject}_task-{task}_run-{run}_model-labels.pkl",
    )
    estimates_path = os.path.join(
        get_subject_model_folder(subject, task),
        f"sub-{subject}_task-{task}_run-{run}_model-estimates.pkl",
    )
    return labels_path, estimates_path


def register_contrast_and_create_maps(
    subject: str,
    task: str,
    contrast_name: str,
    contrast_def: Union[List[float], str],
):
    """
    Register a contrast and create maps for all runs.

    Parameters
    ----------
    subject : str
        Subject label.
    task : str
        Task label.
    contrast_name : str
        Contrast name.
    contrast_def : list of float or str
        Contrast definition. If str, it should be a valid expression for the
        design, e.g. 'A - B'.
    """
    if isinstance(contrast_def, str):
        design_matrix = get_design_matrix(subject, task, "all")
        contrast_def = expression_to_contrast_vector(
            contrast_def, design_matrix.columns
        )
    register_contrast(subject, task, contrast_name, contrast_def)
    runs = get_runs(subject, task)
    for run in runs:
        for map_type in ["effect", "t", "z", "p", "variance"]:
            _create_contrast_single_run(
                subject,
                task,
                run,
                contrast_def,
                map_type,
            )


def get_first_level_model(subject: str, task: str, run: int):
    labels_path, estimates_path = get_first_level_model_paths(
        subject, task, run
    )
    assert os.path.exists(labels_path) and os.path.exists(
        estimates_path
    ), f"Model files not found for subject {subject}, task {task}, run {run}."
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)
    with open(estimates_path, "rb") as f:
        estimates = pickle.load(f)
    return labels, estimates


def _create_contrast_single_run(
    subject: str,
    task: str,
    run_label: int,
    contrast_vector: List[float],
    type: str,
) -> Nifti1Image:
    labels, estimates = get_first_level_model(subject, task, run_label)

    run0 = get_runs(subject, task)[0]
    contrast_info_path = get_contrast_info_path(subject, task)
    contrast_info = pd.read_csv(contrast_info_path)
    contrast0 = contrast_info["contrast"][0]
    ref_img = load_img(get_contrast_path(subject, task, run0, contrast0, type))

    contrast = compute_contrast(
        labels, estimates, contrast_vector, stat_type="t"
    )
    if type == "t":
        return new_img_like(
            ref_img, contrast.stat().reshape(ref_img.get_fdata().shape)
        )
    elif type == "p":
        return new_img_like(
            ref_img, contrast.p_value().reshape(ref_img.get_fdata().shape)
        )
    elif type == "z":
        return new_img_like(
            ref_img, contrast.z_score().reshape(ref_img.get_fdata().shape)
        )
    elif type == "effect":
        return new_img_like(
            ref_img, contrast.effect_size().reshape(ref_img.get_fdata().shape)
        )
    elif type == "variance":
        return new_img_like(
            ref_img,
            contrast.effect_variance().reshape(ref_img.get_fdata().shape),
        )
    else:
        raise ValueError(f"Type {type} not recognized.")
