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

IMAGE_SUFFIXES = {
    "z_score": "z",
    "effect_size": "effect",
    "effect_variance": "variance",
    "stat": "t",
    "p_value": "p",
}


def _compute_contrast(
    sub, task, run, model, contrast_name, contrast_vector, contrasts_folder
):
    contrast_imgs = model.compute_contrast(
        np.array(contrast_vector), stat_type="t", output_type="all"
    )
    for image_type, suffix in IMAGE_SUFFIXES.items():
        contrast_imgs[image_type].to_filename(
            _get_contrast_path(
                contrasts_folder, sub, task, run, contrast_name, suffix
            )
        )


def run_first_level(
    task: str,
    subjects: Optional[List[str]] = None,
    space: Optional[str] = None,
    data_filter: Optional[List[Tuple[str, str]]] = [],
    contrasts: Optional[List[Tuple[str, Union[str, Dict[str, float]]]]] = [],
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
        expression can be a formula string or a dictionary of regressor names
        and their weights.
    :type contrasts: Optional[List[Tuple[str, Union[str, Dict[str, float]]]]
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
            slice_time_ref=None,
            **kwargs,
        )
    )

    for sub_i in range(len(models)):
        model = models[sub_i]
        subject = model.subject_label
        run_imgs = [load_img(img) for img in models_run_imgs[sub_i]]
        events = models_events[sub_i]
        confounds = models_confounds[sub_i]

        contrasts_folder = _get_contrast_folder(subject, task)
        contrasts_folder.mkdir(parents=True, exist_ok=True)

        run_img_grand = np.concat(
            [img.get_fdata() for img in run_imgs], axis=-1
        )
        run_img_grand = new_img_like(run_imgs[0], run_img_grand)
        design_matrices = []
        design_matrix_cols = []
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
                    [design_matrix, confounds[run_i - 1]], axis=1
                )

            for col in design_matrix.columns:
                if col not in design_matrix_cols:
                    design_matrix_cols.append(col)

            # Prefix all columns with run_i
            design_matrix.columns = [
                f"run-{run_i}_{col}" for col in design_matrix.columns
            ]
            design_matrices.append(design_matrix)

        design_matrix = pd.concat(design_matrices, axis=0)
        design_matrix = design_matrix.fillna(0)
        design_matrix.to_csv(_get_design_matrix_path(subject, task))

        for con_i, (contrast_name, contrast_expr) in enumerate(contrasts):
            if isinstance(contrast_expr, str):
                contrast_vector = expression_to_contrast_vector(
                    contrast_expr, design_matrix_cols
                )
                contrast_expr = dict(zip(design_matrix_cols, contrast_vector))
                contrasts[con_i] = (contrast_name, contrast_expr)
            else:
                for reg_name in contrast_expr.keys():
                    if reg_name not in design_matrix_cols:
                        raise ValueError(
                            f"For contrast '{contrast_name}', "
                            f"invalid regressor name: '{reg_name}'."
                        )

        for con_i, (contrast_name, contrast_expr) in enumerate(contrasts):
            contrast_vector = [
                contrast_expr.get(col, 0) for col in design_matrix_cols
            ]
            _register_contrast(subject, task, contrast_name, contrast_vector)

        model.fit(run_img_grand, design_matrices=design_matrix)
        for con_i, (contrast_name, contrast_expr) in enumerate(contrasts):
            # Compute single-run contrasts
            all_run_contrast_expr = []
            for run_i in range(1, len(run_imgs) + 1):
                run_contrast_expr = len(all_run_contrast_expr) * [0]
                col_i = len(all_run_contrast_expr)
                while col_i < len(
                    design_matrix.columns
                ) and design_matrix.columns[col_i].startswith(f"run_{run_i}_"):
                    run_contrast_expr.append(
                        contrast_expr.get(
                            design_matrix.columns[col_i].replace(
                                f"run_{run_i}_", ""
                            ),
                            0,
                        )
                    )
                    col_i += 1
                all_run_contrast_expr.extend(
                    run_contrast_expr[len(all_run_contrast_expr) :]
                )
                _compute_contrast(
                    subject,
                    task,
                    f"{run_i:02d}",
                    model,
                    contrast_name,
                    run_contrast_expr,
                    contrasts_folder,
                )

            # Compute all-run contrasts
            _compute_contrast(
                subject,
                task,
                "all",
                model,
                contrast_name,
                all_run_contrast_expr,
                contrasts_folder,
            )

            # Compute orthogonalized contrasts
            if "odd-even" in orthogs:
                for rem, run_label in zip([0, 1], ["odd", "even"]):
                    orthog_contrast_expr = [
                        v if int(label.split("_")[1]) % 2 == rem else 0
                        for label, v in zip(
                            design_matrix.columns, all_run_contrast_expr
                        )
                    ]
                    _compute_contrast(
                        subject,
                        task,
                        run_label,
                        model,
                        contrast_name,
                        orthog_contrast_expr,
                        contrasts_folder,
                    )

            if "all-but-one" in orthogs:
                for run_i in range(1, len(run_imgs) + 1):
                    orthog_contrast_expr = [
                        v if label.startswith(f"run_{run_i}_") else 0
                        for label, v in zip(
                            design_matrix.columns, all_run_contrast_expr
                        )
                    ]
                    _compute_contrast(
                        subject,
                        task,
                        f"orth{run_i:02d}",
                        model,
                        contrast_name,
                        orthog_contrast_expr,
                        contrasts_folder,
                    )
