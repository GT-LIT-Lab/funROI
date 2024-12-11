from typing import List, Optional, Union, Tuple
import os
import numpy as np
import pandas as pd
import re
from nilearn.interfaces.bids import get_bids_files
from nibabel.nifti1 import Nifti1Image
from .. import (
    get_bids_deriv_folder,
    get_bids_data_folder,
    get_bids_preprocessed_folder_relative,
    get_bids_preprocessed_folder,
)
from nilearn.glm.first_level import (
    first_level_from_bids,
    make_first_level_design_matrix,
    run_glm,
    mean_scaling,
)
from nilearn.glm import (
    compute_contrast,
    SimpleRegressionResults,
    expression_to_contrast_vector,
)
from ..contrast import (
    _get_contrast_folder,
    _get_design_matrix_path,
    _get_contrast_path,
    _get_design_matrix,
    _get_contrast_runs,
    _get_contrast_info_path,
)
from nilearn.image import load_img, new_img_like
import pickle
from .utils import _register_contrast


_get_model_folder = lambda subject, task: (
    get_bids_deriv_folder()
    / f"first_level_{task}"
    / f"sub-{subject}"
    / "models"
)

_get_model_labels_path = lambda subject, task: (
    _get_model_folder(subject, task)
    / f"sub-{subject}_task-{task}_model-labels.pkl"
)
_get_model_estimates_path = lambda subject, task: (
    _get_model_folder(subject, task)
    / f"sub-{subject}_task-{task}_model-estimates.pkl"
)


def run_first_level(
    subjects: List[str],
    tasks: List[str],
    space: str,
    data_filter: Optional[dict] = {},
    confound_labels: Optional[List[str]] = [],
    design_matrix_args: Optional[dict] = {},
    other_contrasts: Optional[List[Union[str, Tuple[str, str]]]] = [],
):
    """
    Run first-level analysis for a list of subjects. The preprocessed dataset 
    should be organized in BIDS format. The data and preprocessed data folders
    should be specified before running this function using `set_bids_data_folder`
    and `set_bids_preprocessed_folder`.

    :param subjects: List of subject labels.
    :type subjects: List[str]
    :param tasks: List of task labels.
    :type tasks: List[str]
    :param space: The space name of the data.
    :type space: str
    :param data_filter: Additional data filter, e.g. the resolution associated
        with the space.
    :type data_filter: Optional[dict]
    :param confound_labels: List of confound labels.
    :type confound_labels: Optional[List[str]]
    :param design_matrix_args: Design matrix arguments to be passed to the
        Nilearn `make_first_level_design_matrix` function.
    :type design_matrix_args: Optional[dict]
    :param other_contrasts: List of contrast definitions. Each contrast is 
        either a string, indicating a formula, or a tuple of two strings,
        indicating the contrast name and the formula.
    :type other_contrasts: Optional[List[Union[str, Tuple[str, str]]]]
    """
    os.makedirs(get_bids_deriv_folder(), exist_ok=True)
    dofs = {}
    for task in tasks:
        (models, models_run_imgs, models_events, models_confounds) = (
            first_level_from_bids(
                get_bids_data_folder(),
                task,
                sub_labels=subjects,
                space_label=space,
                derivatives_folder=get_bids_preprocessed_folder_relative(),
                img_filters=data_filter,
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
            subject = model.subject_label
            os.makedirs(
                _get_model_folder(subject, task),
                exist_ok=True,
            )
            os.makedirs(
                _get_contrast_folder(subject, task),
                exist_ok=True,
            )

            design_matrices = []
            data = []
            for run_i in range(len(events)):
                events_i = events[run_i]
                imgs_i = load_img(imgs[run_i])
                frame_times = np.arange(imgs_i.shape[-1]) * model.t_r
                if model.slice_time_ref is not None:
                    frame_times = (
                        frame_times + model.slice_time_ref * model.t_r
                    )
                design_matrix = make_first_level_design_matrix(
                    frame_times=frame_times,
                    events=events_i,
                    **design_matrix_args,
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

                # Change column with prefix 'run_{i}
                orig_cols = design_matrix.columns
                design_matrix.columns = [
                    f"run_{run_i}_{col}" for col in design_matrix.columns
                ]

                design_matrices.append(design_matrix)

                filters = [
                    ("space", space),
                    ("desc", "brain"),
                ]
                for search_label in ["ses", "task", "run"]:
                    search_res = re.search(
                        rf"{search_label}-(\w+)_", imgs[run_i]
                    )
                    if search_res:
                        filters.append((search_label, search_res.group(1)))
                filters.extend(data_filter)

                mask_img = load_img(
                    get_bids_files(
                        get_bids_preprocessed_folder(),
                        sub_label=subject,
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
                data.append(imgs_i_data)

            print(f"Saving design matrix for subject {subject}")
            design_matrices = pd.concat(design_matrices, axis=0)
            design_matrices = design_matrices.fillna(0)
            design_matrix_path = _get_design_matrix_path(subject, task)
            design_matrices.to_csv(design_matrix_path, index=False)
            data_long = np.concatenate(data, axis=-1)

            print(f"Running GLM for subject {subject}")

            Y = data_long.reshape(-1, data_long.shape[-1]).T
            Y, _ = mean_scaling(Y, axis=(0))

            labels, estimates = run_glm(
                Y,
                design_matrices.values,
                n_jobs=-1,
                random_state=0,
            )
            for key, value in estimates.items():
                if not isinstance(value, SimpleRegressionResults):
                    estimates[key] = SimpleRegressionResults(value)

            # Dump labels and estimates
            print(f"Saving model files for subject {subject}")
            labels_path, estimates_path = _get_model_labels_path(
                subject, task
            ), _get_model_estimates_path(subject, task)
            with open(labels_path, "wb") as f:
                pickle.dump(labels, f)
            with open(estimates_path, "wb") as f:
                pickle.dump(estimates, f)

            ref_img = Nifti1Image(
                imgs_i.get_fdata()[:, :, :, 0], imgs_i.affine, imgs_i.header
            )

            run_label_masks = {}
            run_label_masks["all"] = np.ones((len(events * len(orig_cols))))
            run_label_masks["all"] = run_label_masks["all"] / len(events)
            odd_mask = np.zeros((len(events)))
            odd_mask[::2] = 1
            run_label_masks["odd"] = np.repeat(odd_mask, len(orig_cols))
            run_label_masks["even"] = ~run_label_masks["odd"].astype(bool)

            run_label_masks["odd"] = (
                run_label_masks["odd"]
                / np.sum(run_label_masks["odd"])
                * len(orig_cols)
            )
            run_label_masks["even"] = (
                run_label_masks["even"]
                / np.sum(run_label_masks["even"])
                * len(orig_cols)
            )
            for run_i in range(len(events)):
                run_i_label = str(run_i + 1)
                run_label_masks[run_i_label] = np.zeros((len(events)))
                run_label_masks[run_i_label][run_i] = 1
                run_label_masks[run_i_label] = np.repeat(
                    run_label_masks[run_i_label], len(orig_cols)
                )

                orth_i_label = f"orth{run_i_label}"
                run_label_masks[orth_i_label] = ~run_label_masks[
                    run_i_label
                ].astype(bool)

                run_label_masks[orth_i_label] = (
                    run_label_masks[orth_i_label]
                    / np.sum(run_label_masks[orth_i_label])
                    * len(orig_cols)
                )

            print(f"Creating main contrasts for subject {subject}")
            for orig_col_i, orig_col in enumerate(orig_cols):
                if 'drift' in orig_col or 'derivative' in orig_col:
                    continue
                cont = False
                for confound_label in confound_labels:
                    if confound_label in orig_col:
                        cont = True
                        break
                if cont:
                    continue
                
                for run_label, mask in run_label_masks.items():
                    if run_label == "all":
                        con_name = orig_col
                    else:
                        con_name = f"{run_label}_{orig_col}"
                    con_def = np.zeros(len(orig_cols))
                    con_def[orig_col_i] = 1
                    con_def = np.tile(con_def, len(events))
                    con_def = con_def * mask

                    _register_contrast(subject, task, con_name, con_def)
                    _create_contrast(
                        subject,
                        task,
                        run_label,
                        orig_col,
                        con_def,
                        labels,
                        estimates,
                        ref_img,
                    )

            print(f"Creating additional contrasts for subject {subject}")
            for contrast in other_contrasts:
                if isinstance(contrast, tuple):
                    contrast_name, contrast = contrast
                else:
                    contrast_name = contrast
                    
                contrast_vector = expression_to_contrast_vector(
                    contrast, orig_cols
                )
                for run_label, mask in run_label_masks.items():
                    if run_label == "all":
                        con_name = contrast_name
                    else:
                        con_name = f"{run_label}_{contrast_name}"
                    con_def = np.tile(contrast_vector, len(events))
                    con_def = con_def * mask
                    _register_contrast(subject, task, con_name, con_def)
                    _create_contrast(
                        subject,
                        task,
                        run_label,
                        contrast_name,
                        con_def,
                        labels,
                        estimates,
                        ref_img,
                    )

def add_contrasts(
    subjects: List[str],
    tasks: List[str],
    contrasts: Optional[List[Union[str, Tuple[str, str]]]] = [],
):
    """
    Post-hoc addition of contrasts to existing first-level models.

    :param subjects: List of subject labels.
    :type subjects: List[str]
    :param tasks: List of task labels.
    :type tasks: List[str]
    :param contrasts: List of contrast definitions. Each contrast is 
        either a string, indicating a formula, or a tuple of two strings,
        indicating the contrast name and the formula.
    :type contrasts: Optional[List[Union[str, Tuple[str, str]]]]
    """
    for task in tasks:
        for subject in subjects:
            # Load model
            labels, estimates = _get_first_level_model(subject, task)
            design_matrix = _get_design_matrix(subject, task)
            orig_cols_run = design_matrix.columns
            run_ns = orig_cols_run.str.extract(r"run_(\d+)_")[0].unique()
            # remove run_*_ prefix
            orig_cols = list(set(re.sub(r"run_\d+_", "", col) for col in orig_cols_run))
            for contrast in contrasts:
                if isinstance(contrast, tuple):
                    contrast_name, contrast = contrast
                else:
                    contrast_name = contrast
                contrast_vector = expression_to_contrast_vector(
                    contrast, orig_cols
                )
                contrast_vector_run = np.zeros(len(orig_cols_run))
                for i, var in enumerate(orig_cols_run):
                    var_name = re.sub(r"run_\d+_", "", var)
                    if var_name not in orig_cols:
                        contrast_vector_run[i] = 0 
                    else:
                        contrast_vector_run[i] = contrast_vector[orig_cols.index(var_name)]
                contrast_vector_run = contrast_vector_run / len(run_ns)
                _register_contrast(subject, task, contrast_name, contrast_vector_run)
                _create_contrast(
                    subject,
                    task,
                    "all",
                    contrast_name,
                    contrast_vector_run,
                    labels,
                    estimates,
                )

                for run_n in run_ns:
                    run_label = int(run_n) + 1 # awkward stuff, # TODO: fix fun label problem in design matrix
                    run_mask = orig_cols_run.str.startswith(f"run_{run_n}_")
                    contrast_vector_run_masked = contrast_vector_run.copy()
                    contrast_vector_run_masked[~run_mask] = 0
                    _create_contrast(
                        subject,
                        task,
                        run_label,
                        contrast_name,
                        contrast_vector_run_masked,
                        labels,
                        estimates,
                    )

                    # TODO: odd-even

                    run_mask_orth = ~run_mask
                    contrast_vector_run_masked_orth = contrast_vector_run.copy()
                    contrast_vector_run_masked_orth[run_mask_orth] = 0
                    _create_contrast(
                        subject,
                        task,
                        f"orth{run_label}",
                        contrast_name,
                        contrast_vector_run_masked_orth,
                        labels,
                        estimates,
                    )


def _get_first_level_model(subject: str, task: str):
    labels_path, estimates_path = _get_model_labels_path(
        subject, task
    ), _get_model_estimates_path(subject, task)
    assert os.path.exists(labels_path) and os.path.exists(
        estimates_path
    ), f"Model files not found for subject {subject}, task {task}."
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)
    with open(estimates_path, "rb") as f:
        estimates = pickle.load(f)
    return labels, estimates


def _create_contrast(
    subject: str,
    task: str,
    run_label: str,
    contrast_name: str,
    contrast_def: List[float],
    labels: Optional[List[str]] = None,
    estimates: Optional[dict] = None,
    ref_img: Optional[Nifti1Image] = None,
):
    # Get first run to get reference image
    if labels is None or estimates is None:
        labels, estimates = _get_first_level_model(subject, task)
    if ref_img is None:
        contrast_info_path = _get_contrast_info_path(subject, task)
        contrast_info = pd.read_csv(contrast_info_path)
        contrast0 = contrast_info["contrast"][0]
        run0 = _get_contrast_runs(subject, task, contrast0)[0]
        ref_img = load_img(
            _get_contrast_path(subject, task, run0, contrast0, "effect")
        )

    maps = {}
    contrast = compute_contrast(labels, estimates, contrast_def, stat_type="t")
    maps["t"] = new_img_like(
        ref_img, contrast.stat().reshape(ref_img.get_fdata().shape)
    )
    maps["p"] = new_img_like(
        ref_img, contrast.p_value().reshape(ref_img.get_fdata().shape)
    )
    # maps["z"] = new_img_like(
    #     ref_img, contrast.z_score().reshape(ref_img.get_fdata().shape)
    # )
    maps["effect"] = new_img_like(
        ref_img, contrast.effect_size().reshape(ref_img.get_fdata().shape)
    )
    # maps["variance"] = new_img_like(
    #     ref_img, contrast.effect_variance().reshape(ref_img.get_fdata().shape)
    # )

    for map_type, map_data in maps.items():
        map_data.to_filename(
            _get_contrast_path(
                subject, task, run_label, contrast_name, map_type
            )
        )
