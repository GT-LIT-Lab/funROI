from typing import List, Optional, Union, Tuple, Dict
import numpy as np
import pandas as pd
from .. import (
    get_bids_data_folder,
    get_bids_preprocessed_folder_relative,
    get_bids_preprocessed_folder,
)
from ..contrast import (
    _get_contrast_path,
    _get_design_matrix_path,
    _get_contrast_folder,
)
from nilearn.glm.first_level import (
    first_level_from_bids,
    make_first_level_design_matrix,
)
from nilearn.glm import expression_to_contrast_vector
from nilearn.image import load_img, new_img_like
from .utils import _register_contrast
import os

IMAGE_SUFFIXES = {
    "z_score": "z",
    "effect_size": "effect",
    "effect_variance": "variance",
    "stat": "t",
    "p_value": "p",
}


def _compute_contrast(sub, task, run, model, contrast_name, contrast_vector):
    contrast_imgs = model.compute_contrast(
        np.array(contrast_vector), stat_type="t", output_type="all"
    )
    for image_type, suffix in IMAGE_SUFFIXES.items():
        contrast_imgs[image_type].to_filename(
            _get_contrast_path(sub, task, run, contrast_name, suffix)
        )


def run_first_level(
    task: str,
    subjects: Optional[List[str]] = None,
    space: Optional[str] = None,
    data_filter: Optional[List[Tuple[str, str]]] = [],
    contrasts: Optional[List[Tuple[str, Dict[str, float]]]] = [],
    orthogs: Optional[List[str]] = ["all-but-one", "odd-even"],
    **kwargs,
):
    """
    Run first-level analysis for a list of subjects.

    :param task: The task label.
    :type task: str
    :param subjects: List of subject labels. If None, all subjects are included.
    :type subjects: Optional[List[str]]
    :param space: The space name of the data. If None, the data is assumed to
        be in the native space.
    :type space: Optional[str]
    :param data_filter: Additional data filter, e.g. the resolution associated
        with the space. See Nilearn `get_bids_files` documentation for more
        information.
    :type data_filter: Optional[List[Tuple[str, str]]]
    :param contrasts: List of contrast definitions. Each contrast is a tuple
        of the contrast name and the contrast expression. The contrast
        expression is defined by a dictionary of regressor names and their
        weights.
    :type contrasts: Optional[List[Tuple[str, Dict[str, float]]]
    :param orthogs: List of orthogonalization strategies. For each group,
        contrast images are also generated for corresponding run labels.
        Supported strategies are 'all-but-one' and 'odd-even'. Default is both.
    :type orthogs: Optional[List[str]]
    :param kwargs: Additional keyword arguments for the first-level analysis.
        See Nilearn `first_level_from_bids` documentation for more information.
    :type kwargs: Dict
    """
    try:
        bids_data_folder = get_bids_data_folder()
    except ValueError:
        raise ValueError(
            "The output directory is not set. The default output directory "
            "cannot be inferred from the BIDS data folder."
        )

    try:  # when only preprocessed data is available
        bids_data_folder = get_bids_data_folder()
        derivatives_folder = get_bids_preprocessed_folder_relative()
    except ValueError:
        bids_data_folder = get_bids_preprocessed_folder()
        derivatives_folder = "."

    (models, models_run_imgs, models_events, models_confounds) = (
        first_level_from_bids(
            bids_data_folder,
            task,
            sub_labels=subjects,
            space_label=space,
            derivatives_folder=derivatives_folder,
            img_filters=data_filter,
            **kwargs,
        )
    )

    for sub_i in range(len(models)):
        model = models[sub_i]
        subject = model.subject_label
        run_imgs = [
            load_img(os.path.realpath(img)) for img in models_run_imgs[sub_i]
        ]
        events = models_events[sub_i]
        confounds = models_confounds[sub_i]

        contrasts_folder = _get_contrast_folder(subject, task)
        contrasts_folder.mkdir(parents=True, exist_ok=True)

        run_img_grand = np.concat(
            [img.get_fdata() for img in run_imgs], axis=-1
        )
        run_img_grand = new_img_like(run_imgs[0], run_img_grand)
        design_matrices = []
        for run_i in range(1, len(run_imgs) + 1):
            events_i = events[run_i - 1]
            imgs_i = run_imgs[run_i - 1]
            frame_times = np.arange(imgs_i.shape[-1]) * model.t_r
            design_matrix = make_first_level_design_matrix(
                frame_times=frame_times,
                events=events_i,
                hrf_model=model.hrf_model,
                drift_model=model.drift_model,
                high_pass=model.high_pass,
                drift_order=model.drift_order,
                fir_delays=model.fir_delays,
                min_onset=model.min_onset,
            )

            # Add confounds
            if confounds is not None:
                design_matrix = pd.concat(
                    [
                        design_matrix.reset_index(drop=True),
                        confounds[run_i - 1].reset_index(drop=True),
                    ],
                    axis=1,
                )

            # Prefix all columns with run_i
            design_matrix.columns = [
                f"run-{run_i:02d}_{col}" for col in design_matrix.columns
            ]
            design_matrices.append(design_matrix)

        design_matrix = pd.concat(design_matrices, axis=0)
        design_matrix = design_matrix.fillna(0)
        design_matrix_path = _get_design_matrix_path(subject, task)
        design_matrix_path.parent.mkdir(parents=True, exist_ok=True)
        design_matrix.to_csv(design_matrix_path)

        contrasts_ = contrasts.copy()
        for con_i, (contrast_name, contrast_expr) in enumerate(contrasts_):
            contrast_expr_by_run = {
                f"run-{run_i:02d}_{reg_name}": v / len(run_imgs)
                for reg_name, v in contrast_expr.items()
                for run_i in range(1, len(run_imgs) + 1)
            }
            for reg_name in contrast_expr_by_run.keys():
                if reg_name not in design_matrix.columns:
                    raise ValueError(
                        f"For contrast '{contrast_name}', "
                        f"invalid regressor name: '{reg_name}'."
                    )
            contrasts_[con_i] = (contrast_name, contrast_expr_by_run)

        for con_i, (contrast_name, contrast_expr) in enumerate(contrasts_):
            contrast_vector = [
                contrast_expr.get(col, 0) for col in design_matrix.columns
            ]
            _register_contrast(subject, task, contrast_name, contrast_vector)
            contrasts_[con_i] = (contrast_name, contrast_vector)

        model.fit(run_img_grand, design_matrices=design_matrix)
        for con_i, (contrast_name, contrast_vector) in enumerate(contrasts_):
            # Compute single-run contrasts
            for run_i in range(1, len(run_imgs) + 1):
                run_contrast_vector = [
                    (
                        v * len(run_imgs)
                        if label.startswith(f"run-{run_i:02d}_")
                        else 0
                    )
                    for label, v in zip(design_matrix.columns, contrast_vector)
                ]
                _compute_contrast(
                    subject,
                    task,
                    f"{run_i:02d}",
                    model,
                    contrast_name,
                    run_contrast_vector,
                )

            # Compute all-run contrasts
            _compute_contrast(
                subject,
                task,
                "all",
                model,
                contrast_name,
                contrast_vector,
            )

            # Compute orthogonalized contrasts
            if "odd-even" in orthogs:
                for rem, run_label in zip([1, 0], ["odd", "even"]):
                    orthog_contrast_expr = [
                        (
                            (
                                v
                                if int(label.split("_")[0].split("-")[1]) % 2
                                == rem
                                else 0
                            )
                            * (len(run_imgs))
                            / (
                                len(run_imgs) // 2
                                if rem == 0
                                else len(run_imgs) - len(run_imgs) // 2
                            )
                        )
                        for label, v in zip(
                            design_matrix.columns, contrast_vector
                        )
                    ]
                    _compute_contrast(
                        subject,
                        task,
                        run_label,
                        model,
                        contrast_name,
                        orthog_contrast_expr,
                    )

            if "all-but-one" in orthogs:
                for run_i in range(1, len(run_imgs) + 1):
                    orthog_contrast_expr = [
                        (
                            v * len(run_imgs) / (len(run_imgs) - 1)
                            if not label.startswith(f"run-{run_i:02d}_")
                            else 0
                        )
                        for label, v in zip(
                            design_matrix.columns, contrast_vector
                        )
                    ]
                    _compute_contrast(
                        subject,
                        task,
                        f"orth{run_i:02d}",
                        model,
                        contrast_name,
                        orthog_contrast_expr,
                    )
