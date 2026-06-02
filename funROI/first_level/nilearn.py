import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm import expression_to_contrast_vector
from nilearn.glm.first_level import (
    first_level_from_bids,
    make_first_level_design_matrix,
)
from nilearn.image import load_img, new_img_like
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.surface import SurfaceImage
from scipy import sparse

from .._surface import (
    SURFACE_HEMIS,
    SURFACE_PARTS,
    is_surface_image,
    load_surface_image,
    load_surface_numeric_data,
    save_surface_image,
)
from ..contrast import (
    _get_contrast_folder,
    _get_contrast_path,
    _get_design_matrix_path,
    _get_residuals_path,
    _get_run_group_info_path,
    _get_surface_contrast_path,
    _get_surface_residuals_path,
)
from ..settings import (
    get_bids_data_folder,
    get_bids_preprocessed_folder,
    get_bids_preprocessed_folder_relative,
)
from .utils import _register_contrast


IMAGE_SUFFIXES = {
    "z_score": "z",
    "effect_size": "effect",
    "effect_variance": "variance",
    "stat": "t",
    "p_value": "p",
}


def _compute_contrast_safely(model, contrast_vector):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Contrast for run \d+ is null\.",
            category=UserWarning,
        )
        return model.compute_contrast(
            contrast_vector, stat_type="t", output_type="all"
        )


def _compute_contrast_volume(
    sub, task, run, model, contrast_name, contrast_vector
):
    contrast_imgs = _compute_contrast_safely(model, np.array(contrast_vector))
    for image_type, suffix in IMAGE_SUFFIXES.items():
        contrast_imgs[image_type].to_filename(
            _get_contrast_path(sub, task, run, contrast_name, suffix)
        )


def _save_image(img, volume_path: Path, surface_paths: Dict[str, Path]) -> None:
    if is_surface_image(img):
        save_surface_image(img, surface_paths)
        return
    img.to_filename(volume_path)


def _compute_contrast_surface(
    sub, task, run, model, contrast_name, contrast_vector
):
    contrast_imgs = _compute_contrast_safely(model, contrast_vector)
    for image_type, suffix in IMAGE_SUFFIXES.items():
        _save_image(
            contrast_imgs[image_type],
            _get_contrast_path(sub, task, run, contrast_name, suffix),
            {
                hemi: _get_surface_contrast_path(
                    sub, task, run, contrast_name, suffix, hemi
                )
                for hemi in SURFACE_HEMIS
            },
        )


def _validate_run_groups(
    run_groups: Optional[Dict[str, List[int]]], n_runs: int
) -> Dict[str, List[str]]:
    if run_groups is None:
        return {}

    reserved_labels = {"all", "odd", "even"}
    normalized_groups = {}
    for label, run_ids in run_groups.items():
        if label in reserved_labels or label.isdigit() or label.startswith("orth"):
            raise ValueError(
                f"Invalid custom run group name '{label}'. "
                "Custom run groups cannot reuse built-in run labels."
            )
        if len(run_ids) == 0:
            raise ValueError(
                f"Custom run group '{label}' must include at least one run."
            )
        if len(set(run_ids)) != len(run_ids):
            raise ValueError(
                f"Custom run group '{label}' contains duplicate run ids."
            )
        if min(run_ids) < 1 or max(run_ids) > n_runs:
            raise ValueError(
                f"Custom run group '{label}' includes an invalid run id. "
                f"Expected 1-indexed run ids between 1 and {n_runs}."
            )
        normalized_groups[label] = [f"{run_id:02d}" for run_id in run_ids]
    return normalized_groups


def _build_run_group_summary(
    run_labels: List[str],
    orthogs: Optional[List[str]],
    custom_run_groups: Dict[str, List[str]],
) -> pd.DataFrame:
    records = [
        {
            "run_label": run_label,
            "runs": [run_label],
            "n_runs": 1,
            "group_type": "single-run",
        }
        for run_label in run_labels
    ]
    records.append(
        {
            "run_label": "all",
            "runs": run_labels,
            "n_runs": len(run_labels),
            "group_type": "builtin",
        }
    )

    if len(run_labels) > 1 and orthogs is not None:
        if "odd-even" in orthogs:
            for group_label, rem in [("odd", 1), ("even", 0)]:
                runs = [
                    run_label
                    for run_label in run_labels
                    if int(run_label) % 2 == rem
                ]
                if len(runs) != 0:
                    records.append(
                        {
                            "run_label": group_label,
                            "runs": runs,
                            "n_runs": len(runs),
                            "group_type": "builtin",
                        }
                    )
        if "all-but-one" in orthogs:
            for run_label in run_labels:
                records.append(
                    {
                        "run_label": f"orth{run_label}",
                        "runs": [
                            other_run
                            for other_run in run_labels
                            if other_run != run_label
                        ],
                        "n_runs": len(run_labels) - 1,
                        "group_type": "builtin",
                    }
                )

    for group_label, runs in custom_run_groups.items():
        records.append(
            {
                "run_label": group_label,
                "runs": runs,
                "n_runs": len(runs),
                "group_type": "custom",
            }
        )
    return pd.DataFrame(records)


def _mesh_laplacian(mesh) -> sparse.csr_matrix:
    coordinates = np.asarray(mesh.coordinates, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Surface smoothing expects triangular faces.")

    edge_pairs = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    )
    edge_pairs = np.sort(edge_pairs, axis=1)
    edge_pairs = np.unique(edge_pairs, axis=0)
    edge_lengths = np.linalg.norm(
        coordinates[edge_pairs[:, 0]] - coordinates[edge_pairs[:, 1]], axis=1
    )
    edge_lengths[edge_lengths == 0] = 1.0
    weights = 1.0 / (edge_lengths**2)

    adjacency = sparse.csr_matrix(
        (weights, (edge_pairs[:, 0], edge_pairs[:, 1])),
        shape=(len(coordinates), len(coordinates)),
    )
    adjacency = adjacency + adjacency.T
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return sparse.diags(degree) - adjacency


def _smooth_surface_array(
    data: np.ndarray, mesh, fwhm: Optional[float]
) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if fwhm is None or fwhm <= 0:
        return data.copy()

    sigma = float(fwhm) / np.sqrt(8.0 * np.log(2.0))
    if sigma == 0:
        return data.copy()

    laplacian = _mesh_laplacian(mesh)
    max_degree = float(laplacian.diagonal().max(initial=0.0))
    if max_degree == 0:
        return data.copy()

    total_time = (sigma**2) / 2.0
    n_steps = max(1, int(np.ceil(total_time * max_degree / 0.49)))
    step_size = total_time / n_steps

    smoothed = data.astype(np.float64, copy=True)
    for _ in range(n_steps):
        smoothed = smoothed - step_size * laplacian.dot(smoothed)
    return smoothed.astype(np.float32)


def _uses_surface_space(
    derivatives_root: Path, task: str, space: Optional[str]
) -> bool:
    if space is None:
        return False
    for path in derivatives_root.glob("sub-*/func/*.func.gii"):
        name = path.name
        if (
            f"task-{task}" in name
            and "hemi-L" in name
            and f"space-{space}" in name
            and name.endswith("_bold.func.gii")
        ):
            return True
    return False


def _parse_bids_entities(pathlike) -> Optional[Dict[str, str]]:
    filename = Path(pathlike).name
    pattern = (
        r"sub-(?P<subject>[^_]+)_task-(?P<task>[^_]+)"
        r"(?:_run-(?P<run>[^_]+))?"
        r"(?:_acq-(?P<acq>[^_]+))?"
    )
    match = re.search(pattern, filename)
    if match is None:
        return None
    return match.groupdict()


def _find_surface_run_paths(
    derivatives_root: Path,
    subject: str,
    task: str,
    run_label: Optional[str],
    acquisition: Optional[str],
    space: str,
) -> Optional[Dict[str, Path]]:
    func_dir = derivatives_root / f"sub-{subject}" / "func"
    surface_paths = {}
    for hemi in SURFACE_HEMIS:
        matches = []
        for path in func_dir.glob("*.func.gii"):
            name = path.name
            if f"sub-{subject}" not in name or f"task-{task}" not in name:
                continue
            if run_label and f"run-{run_label}" not in name:
                continue
            if acquisition and f"acq-{acquisition}" not in name:
                continue
            if f"hemi-{hemi}" not in name or f"space-{space}" not in name:
                continue
            if not name.endswith("_bold.func.gii"):
                continue
            matches.append(str(path))
        matches = sorted(matches)
        if len(matches) == 0:
            return None
        surface_paths[hemi] = Path(matches[0])
    return surface_paths


def _find_surface_mesh_paths(
    derivatives_root: Path, subject: str, space: str
) -> Optional[Dict[str, Path]]:
    anat_dir = derivatives_root / f"sub-{subject}" / "anat"
    mesh_paths = {}
    for hemi in SURFACE_HEMIS:
        candidates = [
            anat_dir
            / f"sub-{subject}_hemi-{hemi}_space-{space}_midthickness.surf.gii",
            anat_dir / f"sub-{subject}_hemi-{hemi}_space-{space}_pial.surf.gii",
            anat_dir / f"sub-{subject}_hemi-{hemi}_midthickness.surf.gii",
            anat_dir / f"sub-{subject}_hemi-{hemi}_pial.surf.gii",
        ]
        for candidate in candidates:
            if candidate.exists():
                mesh_paths[hemi] = candidate
                break
        else:
            return None
    return mesh_paths


def _infer_bootstrap_volume_space(
    derivatives_root: Path, task: str
) -> Optional[str]:
    spaces = []
    for path in derivatives_root.glob("sub-*/func/*_bold.nii*"):
        name = path.name
        if f"task-{task}" not in name:
            continue
        match = re.search(r"_space-([^_]+)", name)
        if match is not None:
            spaces.append(match.group(1))

    unique_spaces = set(spaces)
    for preferred_space in ["MNINonLinear", "MNI152NLin2009cAsym"]:
        if preferred_space in unique_spaces:
            return preferred_space
    if len(unique_spaces) > 0:
        return sorted(unique_spaces)[0]
    return None


def _load_surface_run_image(
    surface_run_paths: Dict[str, Path], mesh_paths: Dict[str, Path]
) -> SurfaceImage:
    return load_surface_image(surface_run_paths, mesh_paths)


def _load_run_image(run_spec, space: Optional[str], derivatives_root: Path):
    if is_surface_image(run_spec) or hasattr(run_spec, "get_fdata"):
        return run_spec

    run_path = Path(os.path.realpath(run_spec))
    if space is not None:
        entities = _parse_bids_entities(run_path)
        if entities is not None:
            surface_run_paths = _find_surface_run_paths(
                derivatives_root,
                entities["subject"],
                entities["task"],
                entities["run"],
                entities["acq"],
                space,
            )
            mesh_paths = _find_surface_mesh_paths(
                derivatives_root, entities["subject"], space
            )
            if surface_run_paths is not None and mesh_paths is not None:
                return _load_surface_run_image(surface_run_paths, mesh_paths)
    return load_img(run_path)


def _zero_run_contrast_vectors(run_contrast_vectors):
    return [np.zeros_like(vector) for vector in run_contrast_vectors]


def _run_subject_volume_first_level(
    subject,
    task,
    model,
    run_imgs,
    events,
    confounds,
    contrasts,
    orthogs,
    run_groups,
):
    if confounds is not None and not isinstance(confounds, list):
        confounds = [confounds]

    contrasts_folder = _get_contrast_folder(subject, task)
    contrasts_folder.mkdir(parents=True, exist_ok=True)

    run_img_grand = np.concatenate(
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

        if confounds is not None:
            design_matrix = pd.concat(
                [
                    design_matrix.reset_index(drop=True),
                    confounds[run_i - 1].reset_index(drop=True),
                ],
                axis=1,
            )

        design_matrix.columns = [
            f"run-{run_i:02d}_{col}" for col in design_matrix.columns
        ]
        design_matrices.append(design_matrix)

    design_matrix = pd.concat(design_matrices, axis=0)
    design_matrix = design_matrix.fillna(0)
    design_matrix_path = _get_design_matrix_path(subject, task)
    design_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    design_matrix.to_csv(design_matrix_path, index=False)

    run_labels = [f"{run_i:02d}" for run_i in range(1, len(run_imgs) + 1)]
    custom_run_groups = _validate_run_groups(run_groups, len(run_imgs))
    run_group_summary = _build_run_group_summary(
        run_labels, orthogs, custom_run_groups
    )
    run_group_summary.to_csv(_get_run_group_info_path(subject, task), index=False)

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
    model.residuals[0].to_filename(_get_residuals_path(subject, task))

    for contrast_name, contrast_vector in contrasts_:
        for run_i in range(1, len(run_imgs) + 1):
            run_contrast_vector = [
                (
                    v * len(run_imgs)
                    if label.startswith(f"run-{run_i:02d}_")
                    else 0
                )
                for label, v in zip(design_matrix.columns, contrast_vector)
            ]
            _compute_contrast_volume(
                subject,
                task,
                f"{run_i:02d}",
                model,
                contrast_name,
                run_contrast_vector,
            )

        _compute_contrast_volume(
            subject,
            task,
            "all",
            model,
            contrast_name,
            contrast_vector,
        )

        for group_label, group_runs in custom_run_groups.items():
            group_contrast_vector = [
                (
                    v * len(run_imgs) / len(group_runs)
                    if label.split("_")[0].replace("run-", "") in group_runs
                    else 0
                )
                for label, v in zip(design_matrix.columns, contrast_vector)
            ]
            _compute_contrast_volume(
                subject,
                task,
                group_label,
                model,
                contrast_name,
                group_contrast_vector,
            )

        if len(run_imgs) == 1:
            continue

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
                    for label, v in zip(design_matrix.columns, contrast_vector)
                ]
                _compute_contrast_volume(
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
                    for label, v in zip(design_matrix.columns, contrast_vector)
                ]
                _compute_contrast_volume(
                    subject,
                    task,
                    f"orth{run_i:02d}",
                    model,
                    contrast_name,
                    orthog_contrast_expr,
                )


def _run_subject_surface_first_level(
    subject,
    task,
    model,
    run_specs,
    derivatives_root,
    surface_space,
    events,
    confounds,
    contrasts,
    orthogs,
    run_groups,
):
    run_imgs = [
        _load_run_image(run_img, surface_space, derivatives_root)
        for run_img in run_specs
    ]

    if confounds is not None and not isinstance(confounds, list):
        confounds = [confounds]

    contrasts_folder = _get_contrast_folder(subject, task)
    contrasts_folder.mkdir(parents=True, exist_ok=True)

    run_design_matrices = []
    prefixed_design_matrices = []
    for run_i, (events_i, imgs_i) in enumerate(
        zip(events, run_imgs, strict=False), start=1
    ):
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

        if confounds is not None:
            design_matrix = pd.concat(
                [
                    design_matrix.reset_index(drop=True),
                    confounds[run_i - 1].reset_index(drop=True),
                ],
                axis=1,
            )

        run_design_matrices.append(design_matrix)

        design_matrix_prefixed = design_matrix.copy()
        design_matrix_prefixed.columns = [
            f"run-{run_i:02d}_{col}" for col in design_matrix.columns
        ]
        prefixed_design_matrices.append(design_matrix_prefixed)

    design_matrix = pd.concat(prefixed_design_matrices, axis=0).fillna(0)
    design_matrix_path = _get_design_matrix_path(subject, task)
    design_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    design_matrix.to_csv(design_matrix_path, index=False)

    run_labels = [f"{run_i:02d}" for run_i in range(1, len(run_imgs) + 1)]
    custom_run_groups = _validate_run_groups(run_groups, len(run_imgs))
    run_group_summary = _build_run_group_summary(
        run_labels, orthogs, custom_run_groups
    )
    run_group_summary.to_csv(_get_run_group_info_path(subject, task), index=False)

    contrasts_ = []
    for contrast_name, contrast_expr in contrasts.copy():
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
        contrast_vector = [
            contrast_expr_by_run.get(col, 0) for col in design_matrix.columns
        ]
        _register_contrast(subject, task, contrast_name, contrast_vector)

        run_contrast_vectors = []
        for run_design_matrix in run_design_matrices:
            run_contrast_vectors.append(
                np.array(
                    [
                        contrast_expr.get(column, 0)
                        for column in run_design_matrix.columns
                    ],
                    dtype=float,
                )
            )
        contrasts_.append((contrast_name, run_contrast_vectors))

    model.fit(run_imgs, design_matrices=run_design_matrices)
    _save_image(
        model.residuals[0],
        _get_residuals_path(subject, task),
        {
            hemi: _get_surface_residuals_path(subject, task, hemi)
            for hemi in SURFACE_HEMIS
        },
    )

    for contrast_name, run_contrast_vectors in contrasts_:
        for run_i in range(1, len(run_imgs) + 1):
            single_run_contrast = _zero_run_contrast_vectors(
                run_contrast_vectors
            )
            single_run_contrast[run_i - 1] = run_contrast_vectors[run_i - 1]
            _compute_contrast_surface(
                subject,
                task,
                f"{run_i:02d}",
                model,
                contrast_name,
                single_run_contrast,
            )

        _compute_contrast_surface(
            subject,
            task,
            "all",
            model,
            contrast_name,
            run_contrast_vectors,
        )

        for group_label, group_runs in custom_run_groups.items():
            group_contrast = _zero_run_contrast_vectors(run_contrast_vectors)
            for run_label in group_runs:
                group_index = int(run_label) - 1
                group_contrast[group_index] = run_contrast_vectors[group_index]
            _compute_contrast_surface(
                subject,
                task,
                group_label,
                model,
                contrast_name,
                group_contrast,
            )

        if len(run_imgs) == 1:
            continue

        if "odd-even" in orthogs:
            for rem, run_label in zip([1, 0], ["odd", "even"]):
                orthog_contrast = _zero_run_contrast_vectors(
                    run_contrast_vectors
                )
                for run_i in range(1, len(run_imgs) + 1):
                    if run_i % 2 == rem:
                        orthog_contrast[run_i - 1] = run_contrast_vectors[
                            run_i - 1
                        ]
                _compute_contrast_surface(
                    subject,
                    task,
                    run_label,
                    model,
                    contrast_name,
                    orthog_contrast,
                )

        if "all-but-one" in orthogs:
            for run_i in range(1, len(run_imgs) + 1):
                orthog_contrast = _zero_run_contrast_vectors(
                    run_contrast_vectors
                )
                for other_run_i in range(1, len(run_imgs) + 1):
                    if other_run_i != run_i:
                        orthog_contrast[other_run_i - 1] = (
                            run_contrast_vectors[other_run_i - 1]
                        )
                _compute_contrast_surface(
                    subject,
                    task,
                    f"orth{run_i:02d}",
                    model,
                    contrast_name,
                    orthog_contrast,
                )


def run_first_level(
    task: str,
    subjects: Optional[List[str]] = None,
    space: Optional[str] = None,
    data_filter: Optional[List[Tuple[str, str]]] = [],
    contrasts: Optional[List[Tuple[str, Dict[str, float]]]] = [],
    orthogs: Optional[List[str]] = ["all-but-one", "odd-even"],
    run_groups: Optional[Dict[str, List[int]]] = None,
    fd_threshold: Optional[float] = None,
    std_dvars_threshold: Optional[float] = None,
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
    :param run_groups: Optional custom run groups to compute alongside the
        built-in run labels. Keys are group names and values are 1-indexed run
        ids.
    :type run_groups: Optional[Dict[str, List[int]]]
    :param fd_threshold: Threshold for framewise displacement (FD) to be used
        for confound generation.
    :type fd_threshold: Optional[float]
    :param std_dvars_threshold: Threshold for standard deviation of DVARS to be
        used for confound generation.
    :type std_dvars_threshold: Optional[float]
    :param kwargs: Additional keyword arguments for the first-level analysis.
        See Nilearn `first_level_from_bids` documentation for more information.
    :type kwargs: Dict
    """
    try:
        bids_data_folder = get_bids_data_folder()
    except (ValueError, RuntimeError):
        raise ValueError(
            "The output directory is not set. The default output directory "
            "cannot be inferred from the BIDS data folder."
        )

    try:
        bids_data_folder = get_bids_data_folder()
        derivatives_folder = get_bids_preprocessed_folder_relative()
    except (ValueError, RuntimeError):
        bids_data_folder = get_bids_preprocessed_folder()
        derivatives_folder = "."

    bids_data_folder = Path(bids_data_folder)
    derivatives_root = (
        bids_data_folder
        if derivatives_folder == "."
        else bids_data_folder / derivatives_folder
    )

    surface_requested = _uses_surface_space(derivatives_root, task, space)
    bootstrap_space = space
    if surface_requested:
        bootstrap_space = _infer_bootstrap_volume_space(
            derivatives_root, task
        )
        if bootstrap_space is None:
            raise ValueError(
                f"Surface data were found for space '{space}', but no matching "
                "preprocessed volume BOLD files were found to bootstrap the "
                "first-level model metadata. Expected at least one "
                "`*_bold.nii*` file for the same task in the derivatives "
                "folder."
            )

    (
        models,
        models_run_imgs,
        models_events,
        models_confounds,
    ) = first_level_from_bids(
        bids_data_folder,
        task,
        sub_labels=subjects,
        space_label=bootstrap_space,
        derivatives_folder=derivatives_folder,
        img_filters=data_filter,
        minimize_memory=False,
        **kwargs,
    )

    if fd_threshold is not None or std_dvars_threshold is not None:
        if fd_threshold is None:
            fd_threshold = np.inf
        if std_dvars_threshold is None:
            std_dvars_threshold = np.inf
        for sub_i in range(len(models)):
            imgs = models_run_imgs[sub_i]
            confounds = models_confounds[sub_i]
            confounds, masks = load_confounds(
                imgs,
                fd_threshold=fd_threshold,
                std_dvars_threshold=std_dvars_threshold,
                strategy=["scrub"],
            )
            if not isinstance(confounds, list):
                models_confounds[sub_i] = [confounds]
                masks = [masks]
            n_runs = len(models_confounds[sub_i])
            for run_i in range(n_runs):
                confounds_i = models_confounds[sub_i][run_i]
                masks_i = masks[run_i]
                if masks_i is not None:
                    outlier_indexes = set(confounds_i.index) - set(masks_i)
                else:
                    outlier_indexes = {}
                for outlier_index in outlier_indexes:
                    models_confounds[sub_i][run_i][
                        f"outlier_index_{outlier_index}"
                    ] = (np.arange(len(confounds_i)) == outlier_index).astype(
                        int
                    )

    for sub_i in range(len(models)):
        model = models[sub_i]
        subject = model.subject_label
        run_specs = models_run_imgs[sub_i]
        events = models_events[sub_i]
        confounds = models_confounds[sub_i]

        if surface_requested:
            _run_subject_surface_first_level(
                subject,
                task,
                model,
                run_specs,
                derivatives_root,
                space,
                events,
                confounds,
                contrasts,
                orthogs,
                run_groups,
            )
        else:
            run_imgs = [
                load_img(os.path.realpath(img)) for img in run_specs
            ]
            _run_subject_volume_first_level(
                subject,
                task,
                model,
                run_imgs,
                events,
                confounds,
                contrasts,
                orthogs,
                run_groups,
            )
