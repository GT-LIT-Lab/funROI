from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import hashlib
import json
import re
import warnings

import numpy as np
import pandas as pd
from nilearn import glm, image, maskers, signal
from nilearn.image import load_img
from nilearn.surface import SurfaceImage

from .._surface import (
    SURFACE_HEMIS,
    SURFACE_PARTS,
    flatten_image_data,
    is_surface_image,
    load_surface_image,
    save_surface_image,
)
from ..first_level.nilearn import _find_surface_mesh_paths, _smooth_surface_array
from ..froi import FROIConfig, _get_froi_data
from ..parcels import ParcelsConfig, get_parcels, is_no_parcels
from ..settings import (
    get_bids_data_folder,
    get_bids_deriv_folder,
    get_bids_preprocessed_folder,
)
from .utils import AnalysisSaver


FC_CLEAN_CONFIG = {
    "clean_surf": False,
    "mask_suffix": "_desc-brain_mask.nii.gz",
    # Match the CLiMB-style postfprep volume cleaner, which does not smooth the
    # mask in its active code path.
    "mask_fwhm": None,
    "target_affine": None,
    "confounds_regex": (
        r"^(global_signal|framewise_displacement|trans_[xyz]|rot_[xyz]|"
        r"a_comp_cor_0[0-4]|cosine.*)$"
    ),
    "fd_threshold": 0.5,
    "volume_fwhm": 4,
    "surface_fwhm": 4,
    "standardize": True,
    "detrend": True,
    "regress_out_task": True,
    "task_conditions": None,
    "regress_task_conditions": None,
    "concat_conditions_across_runs": False,
    "low_pass": 0.1,
    "high_pass": 0.01,
    "min_T": 50,
}


RUN_RE = re.compile(r"_run-([A-Za-z0-9]+)_")
SPACE_RE = re.compile(r"_space-([A-Za-z0-9]+)_")
SES_RE = re.compile(r"_ses-([A-Za-z0-9]+)_")
ENTITY_RE = re.compile(r"(^|_)([A-Za-z0-9]+)-([^_]+)")
FC_PREPROC_DERIV_NAME = "fconn_preproc"
FC_PREPROC_CONFIG_KEYS = (
    "clean_surf",
    "mask_suffix",
    "mask_fwhm",
    "target_affine",
    "confounds_regex",
    "fd_threshold",
    "volume_fwhm",
    "surface_fwhm",
    "standardize",
    "detrend",
    "regress_out_task",
    "regress_task_conditions",
    "low_pass",
    "high_pass",
)


def _normalize_standardize_arg(standardize):
    if standardize is True:
        return "zscore_sample"
    if standardize is False:
        return None
    return standardize


def _masker_fit_transform_with_explicit_mask(masker, func_img, confounds):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"\[NiftiMasker\.fit\] Generation of a mask has been requested "
                r"\(imgs != None\) while a mask was given at masker creation\. "
                r"Given mask will be used\."
            ),
            category=UserWarning,
        )
        return masker.fit_transform(func_img, confounds=confounds)


def _normalize_task_conditions(
    task_conditions: Optional[Union[str, List[str], Tuple[str, ...]]]
) -> Optional[List[str]]:
    if task_conditions is None:
        return None
    if isinstance(task_conditions, str):
        return [task_conditions]
    return list(task_conditions)


def _frame_times_from_sidecar(
    sidecar: Dict,
    TR: float,
    start_time: float,
    n_timepoints: int,
) -> np.ndarray:
    slice_time_ref = sidecar.get("SliceTimingCorrected")
    if slice_time_ref is True:
        slice_time_ref = 0.5
    elif slice_time_ref is None:
        slice_time_ref = 0.0
    else:
        slice_time_ref = float(slice_time_ref)
    return np.arange(n_timepoints) * TR + start_time + (slice_time_ref * TR)


def _parse_bids_entities(path: Path) -> Dict[str, str]:
    stem = path.name
    for suffix in (".nii.gz", ".func.gii", ".tsv", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    entities = {}
    for match in ENTITY_RE.finditer(stem):
        entities[match.group(2)] = match.group(3)
    return entities


def _find_matching_bids_file(
    directory: Path,
    reference_file: Path,
    suffix: str,
    *,
    required_entities: Optional[Tuple[str, ...]] = None,
) -> Optional[Path]:
    if required_entities is None:
        required_entities = ("sub", "ses", "task", "acq", "ce", "dir", "rec", "run")

    reference_entities = _parse_bids_entities(reference_file)
    reference_entities = {
        key: reference_entities[key]
        for key in required_entities
        if key in reference_entities
    }

    matches = []
    for candidate in sorted(directory.glob(f"*{suffix}")):
        candidate_entities = _parse_bids_entities(candidate)
        if all(
            candidate_entities.get(key) == value
            for key, value in reference_entities.items()
        ):
            matches.append(candidate)

    if len(matches) == 0:
        return None
    return matches[0]


def _make_json_serializable(value):
    if isinstance(value, dict):
        return {
            str(key): _make_json_serializable(val)
            for key, val in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_make_json_serializable(val) for val in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _build_fc_preproc_config(clean_config: Dict) -> Dict:
    return {
        key: _make_json_serializable(clean_config[key])
        for key in FC_PREPROC_CONFIG_KEYS
        if key in clean_config
    }


def _fc_preproc_config_hash(preproc_config: Dict) -> str:
    payload = json.dumps(preproc_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _get_fc_preproc_root() -> Path:
    return get_bids_deriv_folder() / FC_PREPROC_DERIV_NAME


def _get_fc_preproc_record_paths(record: Dict, preproc_config: Dict) -> Dict[str, Path]:
    config_hash = _fc_preproc_config_hash(preproc_config)
    if "func_file" in record:
        source_file = record["func_file"]
    else:
        source_file = record["func_files"]["L"]

    relative_parent = source_file.relative_to(get_bids_preprocessed_folder()).parent
    output_dir = _get_fc_preproc_root() / relative_parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if "func_file" in record:
        output_name = source_file.name.replace(
            "_desc-preproc_bold.nii.gz",
            f"_desc-fcpreproc-{config_hash}_bold.nii.gz",
        )
        img_path = output_dir / output_name
        meta_path = output_dir / output_name.replace(".nii.gz", ".json")
        return {"img": img_path, "meta": meta_path}

    paths = {}
    for hemi, source_path in record["func_files"].items():
        output_name = source_path.name.replace(
            "_desc-preproc_bold.func.gii",
            f"_desc-fcpreproc-{config_hash}_bold.func.gii",
        )
        paths[hemi] = output_dir / output_name
    meta_path = output_dir / paths["L"].name.replace(".func.gii", ".json")
    paths["meta"] = meta_path
    return paths


def _get_img_n_timepoints(img) -> int:
    if is_surface_image(img):
        return int(np.asarray(img.data.parts["left"]).shape[-1])
    data = img.get_fdata()
    if data.ndim < 4:
        return 1
    return int(data.shape[-1])


def _subset_cleaned_img(img, selected_frames: np.ndarray):
    if is_surface_image(img):
        return SurfaceImage(
            mesh=img.mesh,
            data={
                part_name: np.asarray(img.data.parts[part_name])[:, selected_frames]
                for part_name in img.data.parts
            },
        )

    data = img.get_fdata()
    if data.ndim != 4:
        raise ValueError("Cleaned functional image must be 4D.")
    return image.new_img_like(
        img,
        data[..., selected_frames],
        copy_header=True,
    )


def _load_cached_preprocessed_run(record: Dict, preproc_config: Dict):
    cache_paths = _get_fc_preproc_record_paths(record, preproc_config)
    if "img" in cache_paths:
        if not cache_paths["img"].exists():
            return None
        return {
            "cleaned_img": load_img(cache_paths["img"]),
            "cache_paths": cache_paths,
            **record,
        }

    if not all(cache_paths[hemi].exists() for hemi in SURFACE_HEMIS):
        return None
    return {
        "cleaned_img": load_surface_image(
            {hemi: cache_paths[hemi] for hemi in SURFACE_HEMIS},
            record["mesh_paths"],
        ),
        "cache_paths": cache_paths,
        **record,
    }


def _save_cached_preprocessed_run(cleaned_img, record: Dict, preproc_config: Dict):
    cache_paths = _get_fc_preproc_record_paths(record, preproc_config)
    metadata = {
        "source_files": _make_json_serializable(
            record.get("func_file", record.get("func_files"))
        ),
        "confounds_file": _make_json_serializable(record["confounds_file"]),
        "events_file": _make_json_serializable(record["events_file"]),
        "run_label": record["run_label"],
        "session_label": record["session_label"],
        "TR": record["TR"],
        "StartTime": record["StartTime"],
        "preproc_config": preproc_config,
        "config_hash": _fc_preproc_config_hash(preproc_config),
    }

    if "img" in cache_paths:
        cleaned_img.to_filename(cache_paths["img"])
    else:
        save_surface_image(
            cleaned_img,
            {hemi: cache_paths[hemi] for hemi in SURFACE_HEMIS},
        )

    with open(cache_paths["meta"], "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    return {
        "cleaned_img": cleaned_img,
        "cache_paths": cache_paths,
        **record,
    }


def preprocess_bold_for_fc(
    subjects: Union[str, List[str]],
    task: str,
    *,
    session: Optional[str] = None,
    space: Optional[str] = None,
    clean_surf: bool = False,
    mask_suffix: str = FC_CLEAN_CONFIG["mask_suffix"],
    mask_fwhm: Optional[float] = FC_CLEAN_CONFIG["mask_fwhm"],
    target_affine: Optional[List[List[float]]] = None,
    confounds_regex: str = FC_CLEAN_CONFIG["confounds_regex"],
    fd_threshold: Optional[float] = FC_CLEAN_CONFIG["fd_threshold"],
    volume_fwhm: Optional[float] = FC_CLEAN_CONFIG["volume_fwhm"],
    surface_fwhm: Optional[float] = FC_CLEAN_CONFIG["surface_fwhm"],
    standardize: Union[bool, str, None] = FC_CLEAN_CONFIG["standardize"],
    detrend: bool = FC_CLEAN_CONFIG["detrend"],
    regress_out_task: bool = FC_CLEAN_CONFIG["regress_out_task"],
    regress_task_conditions: Optional[
        Union[str, List[str], Tuple[str, ...]]
    ] = None,
    low_pass: Optional[float] = FC_CLEAN_CONFIG["low_pass"],
    high_pass: Optional[float] = FC_CLEAN_CONFIG["high_pass"],
    min_T: int = FC_CLEAN_CONFIG["min_T"],
    overwrite: bool = False,
) -> pd.DataFrame:
    if isinstance(subjects, str):
        subjects = [subjects]

    clean_config = {
        "clean_surf": clean_surf,
        "mask_suffix": mask_suffix,
        "mask_fwhm": mask_fwhm,
        "target_affine": target_affine,
        "confounds_regex": confounds_regex,
        "fd_threshold": fd_threshold,
        "volume_fwhm": volume_fwhm,
        "surface_fwhm": surface_fwhm,
        "standardize": _normalize_standardize_arg(standardize),
        "detrend": detrend,
        "regress_out_task": regress_out_task,
        "regress_task_conditions": _normalize_task_conditions(
            regress_task_conditions
        ),
        "task_conditions": None,
        "low_pass": low_pass,
        "high_pass": high_pass,
        "min_T": min_T,
    }
    preproc_config = _build_fc_preproc_config(clean_config)

    manifest_rows = []
    for subject in subjects:
        prepared_runs = load_preprocessed_bold_for_fc(
            subject,
            task,
            session=session,
            space=space,
            clean_surf=clean_surf,
            mask_suffix=mask_suffix,
            mask_fwhm=mask_fwhm,
            target_affine=target_affine,
            confounds_regex=confounds_regex,
            fd_threshold=fd_threshold,
            volume_fwhm=volume_fwhm,
            surface_fwhm=surface_fwhm,
            standardize=standardize,
            detrend=detrend,
            regress_out_task=regress_out_task,
            regress_task_conditions=regress_task_conditions,
            low_pass=low_pass,
            high_pass=high_pass,
            min_T=min_T,
            create_if_missing=True,
            overwrite=overwrite,
        )
        for prepared in prepared_runs:
            row = {
                "subject": subject,
                "task": task,
                "run_label": prepared["run_label"],
                "session_label": prepared["session_label"],
                "config_hash": _fc_preproc_config_hash(preproc_config),
            }
            row.update(
                {
                    f"cache_{key}": str(path)
                    for key, path in prepared["cache_paths"].items()
                }
            )
            manifest_rows.append(row)
    return pd.DataFrame(manifest_rows)


def load_preprocessed_bold_for_fc(
    subject: str,
    task: str,
    *,
    session: Optional[str] = None,
    space: Optional[str] = None,
    clean_surf: bool = False,
    mask_suffix: str = FC_CLEAN_CONFIG["mask_suffix"],
    mask_fwhm: Optional[float] = FC_CLEAN_CONFIG["mask_fwhm"],
    target_affine: Optional[List[List[float]]] = None,
    confounds_regex: str = FC_CLEAN_CONFIG["confounds_regex"],
    fd_threshold: Optional[float] = FC_CLEAN_CONFIG["fd_threshold"],
    volume_fwhm: Optional[float] = FC_CLEAN_CONFIG["volume_fwhm"],
    surface_fwhm: Optional[float] = FC_CLEAN_CONFIG["surface_fwhm"],
    standardize: Union[bool, str, None] = FC_CLEAN_CONFIG["standardize"],
    detrend: bool = FC_CLEAN_CONFIG["detrend"],
    regress_out_task: bool = FC_CLEAN_CONFIG["regress_out_task"],
    regress_task_conditions: Optional[
        Union[str, List[str], Tuple[str, ...]]
    ] = None,
    low_pass: Optional[float] = FC_CLEAN_CONFIG["low_pass"],
    high_pass: Optional[float] = FC_CLEAN_CONFIG["high_pass"],
    min_T: int = FC_CLEAN_CONFIG["min_T"],
    create_if_missing: bool = False,
    overwrite: bool = False,
) -> List[Dict]:
    clean_config = {
        "clean_surf": clean_surf,
        "mask_suffix": mask_suffix,
        "mask_fwhm": mask_fwhm,
        "target_affine": target_affine,
        "confounds_regex": confounds_regex,
        "fd_threshold": fd_threshold,
        "volume_fwhm": volume_fwhm,
        "surface_fwhm": surface_fwhm,
        "standardize": _normalize_standardize_arg(standardize),
        "detrend": detrend,
        "regress_out_task": regress_out_task,
        "regress_task_conditions": _normalize_task_conditions(
            regress_task_conditions
        ),
        "task_conditions": None,
        "low_pass": low_pass,
        "high_pass": high_pass,
        "min_T": min_T,
    }
    preproc_config = _build_fc_preproc_config(clean_config)

    if clean_surf:
        run_records = FunctionalConnectivityEstimator._find_preprocessed_surface_runs(
            subject, task, session, space
        )
    else:
        run_records = FunctionalConnectivityEstimator._find_preprocessed_runs(
            subject, task, session, space, mask_suffix
        )

    prepared_runs = []
    for record in run_records:
        prepared = None
        if not overwrite:
            prepared = _load_cached_preprocessed_run(record, preproc_config)

        if prepared is None and create_if_missing:
            if clean_surf:
                cleaned_img = FunctionalConnectivityEstimator._clean_surface_run(
                    record, clean_config
                )
            else:
                cleaned_img = FunctionalConnectivityEstimator._clean_run(
                    record, clean_config
                )
            if cleaned_img is None:
                continue
            prepared = _save_cached_preprocessed_run(
                cleaned_img, record, preproc_config
            )

        if prepared is not None:
            prepared_runs.append(prepared)
    return prepared_runs


class FunctionalConnectivityEstimator(AnalysisSaver):
    """
    Estimate within-subject functional connectivity from cleaned
    preprocessed BOLD runs.
    """

    def __init__(
        self,
        subjects: List[str],
        froi1: Union[FROIConfig, str, ParcelsConfig],
        froi2: Union[FROIConfig, str, ParcelsConfig],
    ):
        self.subjects = subjects
        self.froi1 = froi1
        self.froi2 = froi2
        self._type = "fconn"
        self._data_summary = None
        self._data_detail = None

    def run(
        self,
        task: Optional[str] = None,
        froi1_run_label: Optional[str] = None,
        froi2_run_label: Optional[str] = None,
        session: Optional[str] = None,
        space: Optional[str] = None,
        clean_surf: bool = False,
        mask_suffix: str = FC_CLEAN_CONFIG["mask_suffix"],
        mask_fwhm: Optional[float] = FC_CLEAN_CONFIG["mask_fwhm"],
        target_affine: Optional[List[List[float]]] = None,
        confounds_regex: str = FC_CLEAN_CONFIG["confounds_regex"],
        fd_threshold: Optional[float] = 0.5,
        volume_fwhm: Optional[float] = 4,
        surface_fwhm: Optional[float] = 4,
        standardize: Union[bool, str, None] = True,
        detrend: bool = True,
        regress_out_task: bool = True,
        task_conditions: Optional[Union[str, List[str], Tuple[str, ...]]] = None,
        regress_task_conditions: Optional[
            Union[str, List[str], Tuple[str, ...]]
        ] = None,
        concat_conditions_across_runs: bool = False,
        low_pass: Optional[float] = 0.1,
        high_pass: Optional[float] = 0.01,
        min_T: int = 50,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the functional connectivity analysis for all subjects.
        """
        self.froi1_run_label = froi1_run_label
        self.froi2_run_label = froi2_run_label

        task = self._resolve_task(task, self.froi1, self.froi2)
        clean_config = {
            "clean_surf": clean_surf,
            "mask_suffix": mask_suffix,
            "mask_fwhm": mask_fwhm,
            "target_affine": target_affine,
            "confounds_regex": confounds_regex,
            "fd_threshold": fd_threshold,
            "volume_fwhm": volume_fwhm,
            "surface_fwhm": surface_fwhm,
            "standardize": _normalize_standardize_arg(standardize),
            "detrend": detrend,
            "regress_out_task": regress_out_task,
            "task_conditions": _normalize_task_conditions(task_conditions),
            "regress_task_conditions": _normalize_task_conditions(
                regress_task_conditions
            ),
            "concat_conditions_across_runs": concat_conditions_across_runs,
            "low_pass": low_pass,
            "high_pass": high_pass,
            "min_T": min_T,
        }

        data_summary = []
        data_detail = []
        for subject in self.subjects:
            froi1_labels, froi1_name, froi1_has_explicit_parcels, froi1_img = (
                self._resolve_mask_data(self.froi1, subject, froi1_run_label)
            )
            froi2_labels, froi2_name, froi2_has_explicit_parcels, froi2_img = (
                self._resolve_mask_data(self.froi2, subject, froi2_run_label)
            )

            subject_clean_config = dict(clean_config)
            subject_clean_config["clean_surf"] = (
                subject_clean_config["clean_surf"]
                or is_surface_image(froi1_img)
                or is_surface_image(froi2_img)
            )
            if subject_clean_config["clean_surf"] and not (
                is_surface_image(froi1_img) and is_surface_image(froi2_img)
            ):
                raise ValueError(
                    "Surface functional connectivity requires both ROI inputs to "
                    "be surface-based."
                )

            (
                cleaned_imgs,
                cleaned_run_labels,
                cleaned_session_labels,
            ) = self._get_cleaned_imgs_by_run(
                subject, task, session, space, subject_clean_config
            )
            df_summary, df_detail = self._run(cleaned_imgs, froi1_img, froi2_img)

            df_detail["bold_run"] = df_detail["run"].apply(
                lambda idx: cleaned_run_labels[idx]
            )
            df_detail["bold_session"] = df_detail["run"].apply(
                lambda idx: cleaned_session_labels[idx]
            )
            df_detail = df_detail.drop(columns=["run"])

            if froi1_labels is not None:
                df_summary["froi1"] = df_summary["froi1"].apply(
                    lambda x: froi1_labels[x]
                )
                df_detail["froi1"] = df_detail["froi1"].apply(
                    lambda x: froi1_labels[x]
                )
            elif not froi1_has_explicit_parcels:
                df_summary = df_summary.drop(columns=["froi1"])
                df_detail = df_detail.drop(columns=["froi1"])

            if froi2_labels is not None:
                df_summary["froi2"] = df_summary["froi2"].apply(
                    lambda x: froi2_labels[x]
                )
                df_detail["froi2"] = df_detail["froi2"].apply(
                    lambda x: froi2_labels[x]
                )
            elif not froi2_has_explicit_parcels:
                df_summary = df_summary.drop(columns=["froi2"])
                df_detail = df_detail.drop(columns=["froi2"])

            if froi1_name == "parcel":
                df_summary = df_summary.rename(columns={"froi1": "parcel1"})
                df_detail = df_detail.rename(columns={"froi1": "parcel1"})
            if froi2_name == "parcel":
                df_summary = df_summary.rename(columns={"froi2": "parcel2"})
                df_detail = df_detail.rename(columns={"froi2": "parcel2"})

            df_summary["subject"] = subject
            df_detail["subject"] = subject
            data_summary.append(df_summary)
            data_detail.append(df_detail)

        self._data_summary = pd.concat(data_summary, ignore_index=True)
        self._data_detail = pd.concat(data_detail, ignore_index=True)
        new_info = pd.DataFrame(
            {
                "task": [task],
                "session": [session],
                "space": [space],
                "froi1": [self.froi1],
                "froi2": [self.froi2],
                "customized_froi1_run": [froi1_run_label],
                "customized_froi2_run": [froi2_run_label],
                "clean_config": [clean_config],
            }
        )
        self._save(new_info)
        return self._data_summary, self._data_detail

    @staticmethod
    def _resolve_task(
        task: Optional[str],
        froi1: Union[FROIConfig, str, ParcelsConfig],
        froi2: Union[FROIConfig, str, ParcelsConfig],
    ) -> str:
        if task is not None:
            return task

        tasks = []
        if isinstance(froi1, FROIConfig):
            tasks.append(froi1.task)
        if isinstance(froi2, FROIConfig):
            tasks.append(froi2.task)
        tasks = sorted(set(tasks))
        if len(tasks) == 1:
            return tasks[0]
        raise ValueError(
            "Task could not be inferred. Please specify the task whose cleaned "
            "preprocessed data should be used."
        )

    @staticmethod
    def _resolve_mask_data(
        froi: Union[FROIConfig, str, ParcelsConfig],
        subject: str,
        run_label: Optional[str],
    ):
        is_parcels = not isinstance(froi, FROIConfig)
        parcels_config = froi.parcels if isinstance(froi, FROIConfig) else froi
        _, labels = get_parcels(parcels_config)

        if is_parcels:
            parcels_img, labels = get_parcels(froi)
            if parcels_img is None:
                raise ValueError("Parcels image not found.")
            return labels, "parcel", True, parcels_img

        if run_label is None:
            run_label = "all"
        froi_img = _get_froi_data(subject, froi, run_label, return_nifti=True)
        if froi_img is None:
            raise ValueError(
                f"Data not found for subject {subject}, fROI {froi}, run {run_label}."
            )
        return labels, "froi", not is_no_parcels(froi.parcels), froi_img

    @staticmethod
    def _get_cleaned_imgs_by_run(
        subject: str,
        task: str,
        session: Optional[str],
        space: Optional[str],
        config: Dict,
    ) -> Tuple[List, List[str], List[Optional[str]]]:
        prepared_runs = load_preprocessed_bold_for_fc(
            subject,
            task,
            session=session,
            space=space,
            clean_surf=config["clean_surf"],
            mask_suffix=config["mask_suffix"],
            mask_fwhm=config["mask_fwhm"],
            target_affine=config["target_affine"],
            confounds_regex=config["confounds_regex"],
            fd_threshold=config["fd_threshold"],
            volume_fwhm=config["volume_fwhm"],
            surface_fwhm=config["surface_fwhm"],
            standardize=config["standardize"],
            detrend=config["detrend"],
            regress_out_task=config["regress_out_task"],
            regress_task_conditions=config["regress_task_conditions"],
            low_pass=config["low_pass"],
            high_pass=config["high_pass"],
            min_T=config["min_T"],
            create_if_missing=True,
        )
        prepared_runs = FunctionalConnectivityEstimator._filter_preprocessed_runs_by_min_t(
            prepared_runs,
            config["min_T"],
        )

        selected_runs = FunctionalConnectivityEstimator._select_preprocessed_runs(
            prepared_runs,
            config["task_conditions"],
            config["concat_conditions_across_runs"],
        )
        if len(selected_runs) == 0:
            if config["task_conditions"] is not None:
                raise ValueError(
                    "No usable runs remained for subject "
                    f"{subject} and task {task} after selecting task_conditions "
                    f"{config['task_conditions']}."
                )
            raise ValueError(
                f"No usable preprocessed runs found for subject {subject} and task {task}."
            )

        cleaned_imgs = [prepared["cleaned_img"] for prepared in selected_runs]
        cleaned_run_labels = [prepared["run_label"] for prepared in selected_runs]
        cleaned_session_labels = [
            prepared["session_label"] for prepared in selected_runs
        ]
        return cleaned_imgs, cleaned_run_labels, cleaned_session_labels

    @staticmethod
    def _select_preprocessed_runs(
        prepared_runs: List[Dict],
        task_conditions: Optional[List[str]],
        concat_conditions_across_runs: bool,
    ) -> List[Dict]:
        if task_conditions is None:
            return prepared_runs

        selected_runs = []
        for prepared in prepared_runs:
            cleaned_img = prepared["cleaned_img"]
            n_timepoints = _get_img_n_timepoints(cleaned_img)
            selected_frames = FunctionalConnectivityEstimator._select_task_condition_frames(
                prepared,
                n_timepoints,
                task_conditions,
            )
            if selected_frames is None or not np.any(selected_frames):
                continue

            selected_img = _subset_cleaned_img(cleaned_img, selected_frames)
            selected_runs.append(
                {
                    **prepared,
                    "cleaned_img": selected_img,
                }
            )

        if concat_conditions_across_runs:
            if len(selected_runs) == 0:
                return []
            return [FunctionalConnectivityEstimator._concatenate_prepared_runs(selected_runs)]

        return selected_runs

    @staticmethod
    def _filter_preprocessed_runs_by_min_t(
        prepared_runs: List[Dict],
        min_T: int,
    ) -> List[Dict]:
        filtered_runs = []
        for prepared in prepared_runs:
            n_timepoints = _get_img_n_timepoints(prepared["cleaned_img"])
            if n_timepoints < min_T:
                source_name = prepared.get(
                    "func_file",
                    prepared.get("func_files", {}).get("L"),
                )
                if source_name is not None:
                    warnings.warn(
                        f"Skipping {Path(source_name).name}: insufficient timepoints."
                    )
                continue
            filtered_runs.append(prepared)
        return filtered_runs

    @staticmethod
    def _concatenate_prepared_runs(prepared_runs: List[Dict]) -> Dict:
        first = prepared_runs[0]
        cleaned_imgs = [prepared["cleaned_img"] for prepared in prepared_runs]

        if is_surface_image(cleaned_imgs[0]):
            concatenated_img = SurfaceImage(
                mesh=cleaned_imgs[0].mesh,
                data={
                    part_name: np.concatenate(
                        [
                            np.asarray(img.data.parts[part_name])
                            for img in cleaned_imgs
                        ],
                        axis=-1,
                    )
                    for part_name in cleaned_imgs[0].data.parts
                },
            )
        else:
            concatenated_img = image.concat_imgs(
                cleaned_imgs, auto_resample=True
            )

        run_labels = [prepared["run_label"] for prepared in prepared_runs]
        session_labels = [
            prepared["session_label"]
            for prepared in prepared_runs
            if prepared["session_label"] is not None
        ]
        session_label = None
        if len(session_labels) == 1:
            session_label = session_labels[0]
        elif len(session_labels) > 1:
            session_label = "+".join(session_labels)

        return {
            **first,
            "cleaned_img": concatenated_img,
            "run_label": "concatenated:" + "+".join(run_labels),
            "session_label": session_label,
        }

    @staticmethod
    def _find_preprocessed_runs(
        subject: str,
        task: str,
        session: Optional[str],
        space: Optional[str],
        mask_suffix: str,
    ) -> List[Dict]:
        preproc_root = get_bids_preprocessed_folder()
        bids_root = None
        try:
            bids_root = get_bids_data_folder()
        except RuntimeError:
            pass

        subject_dir = preproc_root / f"sub-{subject}"
        func_dirs = FunctionalConnectivityEstimator._get_subject_func_dirs(
            subject_dir, session
        )

        func_files = []
        missing_dirs = []
        for func_dir, _ in func_dirs:
            if not func_dir.exists():
                missing_dirs.append(func_dir)
                continue
            func_files.extend(
                sorted(func_dir.glob(f"*task-{task}_*desc-preproc_bold.nii.gz"))
            )
        if len(func_files) == 0 and len(missing_dirs) == len(func_dirs):
            raise ValueError(f"Functional directory not found: {missing_dirs[0]}")
        if len(func_files) == 0:
            raise ValueError(
                f"No preprocessed BOLD runs found for subject {subject} and task {task}."
            )

        spaces_found = sorted(
            {
                match.group(1)
                for path in func_files
                for match in [SPACE_RE.search(path.name)]
                if match is not None
            }
        )
        if space is None and len(spaces_found) > 1:
            raise ValueError(
                "Multiple spaces were found for the requested task. Please specify `space`."
            )

        run_records = []
        for func_file in func_files:
            file_space_match = SPACE_RE.search(func_file.name)
            file_space = (
                file_space_match.group(1) if file_space_match is not None else None
            )
            if space is not None and file_space != space:
                continue

            session_match = SES_RE.search(func_file.name)
            file_session = (
                session_match.group(1) if session_match is not None else None
            )
            run_match = RUN_RE.search(func_file.name)
            run_label = run_match.group(1) if run_match is not None else "01"
            confounds_file = _find_matching_bids_file(
                func_file.parent,
                func_file,
                "_desc-confounds_timeseries.tsv",
            )
            if confounds_file is None or not confounds_file.exists():
                warnings.warn(
                    f"Confounds file not found for {func_file.name}, skipping."
                )
                continue

            if bids_root is None:
                events_file = None
            else:
                if file_session is None:
                    events_dir = bids_root / f"sub-{subject}" / "func"
                else:
                    events_dir = bids_root / f"sub-{subject}" / f"ses-{file_session}" / "func"
                events_file = _find_matching_bids_file(
                    events_dir,
                    func_file,
                    "_events.tsv",
                )

            anat_dir = FunctionalConnectivityEstimator._find_anat_dir(
                preproc_root, subject, file_session
            )
            mask_file, used_functional_mask = (
                FunctionalConnectivityEstimator._find_mask_file(
                    anat_dir,
                    func_file,
                    subject,
                    file_session,
                    file_space,
                    mask_suffix,
                )
            )
            if mask_file is None:
                warnings.warn(
                    f"No matching mask found for {func_file.name}, skipping."
                )
                continue
            if used_functional_mask:
                warnings.warn(
                    f"Anatomical mask not found for {func_file.name}; using "
                    f"functional mask {mask_file.name} instead."
                )

            sidecar_path = FunctionalConnectivityEstimator._find_bold_sidecar(
                func_file
            )
            if not sidecar_path.exists():
                warnings.warn(
                    f"JSON sidecar not found for {func_file.name}, skipping."
                )
                continue
            with open(sidecar_path, "r") as f:
                sidecar = json.load(f)
            TR = sidecar.get("RepetitionTime")
            if TR is None:
                warnings.warn(
                    f"RepetitionTime missing for {func_file.name}, skipping."
                )
                continue

            run_records.append(
                {
                    "func_file": func_file,
                    "confounds_file": confounds_file,
                    "events_file": events_file,
                    "mask_file": mask_file,
                    "TR": TR,
                    "StartTime": sidecar.get("StartTime", 0),
                    "sidecar": sidecar,
                    "run_label": run_label,
                    "session_label": file_session,
                }
            )

        return run_records

    @staticmethod
    def _find_preprocessed_surface_runs(
        subject: str,
        task: str,
        session: Optional[str],
        space: Optional[str],
    ) -> List[Dict]:
        preproc_root = get_bids_preprocessed_folder()
        bids_root = None
        try:
            bids_root = get_bids_data_folder()
        except RuntimeError:
            pass

        subject_dir = preproc_root / f"sub-{subject}"
        func_dirs = FunctionalConnectivityEstimator._get_subject_func_dirs(
            subject_dir, session
        )

        func_files = []
        missing_dirs = []
        for func_dir, _ in func_dirs:
            if not func_dir.exists():
                missing_dirs.append(func_dir)
                continue
            func_files.extend(
                sorted(
                    func_dir.glob(
                        f"*task-{task}_*hemi-L*_desc-preproc_bold.func.gii"
                    )
                )
            )
        if len(func_files) == 0 and len(missing_dirs) == len(func_dirs):
            raise ValueError(f"Functional directory not found: {missing_dirs[0]}")
        if len(func_files) == 0:
            raise ValueError(
                f"No surface preprocessed BOLD runs found for subject {subject} and task {task}."
            )

        spaces_found = sorted(
            {
                match.group(1)
                for path in func_files
                for match in [SPACE_RE.search(path.name)]
                if match is not None
            }
        )
        if space is None and len(spaces_found) > 1:
            raise ValueError(
                "Multiple spaces were found for the requested task. Please specify `space`."
            )

        run_records = []
        for left_file in func_files:
            file_space_match = SPACE_RE.search(left_file.name)
            file_space = (
                file_space_match.group(1)
                if file_space_match is not None
                else None
            )
            if file_space is None:
                warnings.warn(
                    f"Space entity missing for {left_file.name}, skipping."
                )
                continue
            if space is not None and file_space != space:
                continue

            right_file = Path(str(left_file).replace("_hemi-L_", "_hemi-R_"))
            if not right_file.exists():
                warnings.warn(
                    f"Right hemisphere file not found for {left_file.name}, skipping."
                )
                continue

            session_match = SES_RE.search(left_file.name)
            file_session = (
                session_match.group(1)
                if session_match is not None
                else None
            )
            run_match = RUN_RE.search(left_file.name)
            run_label = run_match.group(1) if run_match is not None else "01"
            confounds_file = _find_matching_bids_file(
                left_file.parent,
                left_file,
                "_desc-confounds_timeseries.tsv",
            )
            if confounds_file is None or not confounds_file.exists():
                warnings.warn(
                    f"Confounds file not found for {left_file.name}, skipping."
                )
                continue

            if bids_root is None:
                events_file = None
            else:
                if file_session is None:
                    events_dir = bids_root / f"sub-{subject}" / "func"
                else:
                    events_dir = (
                        bids_root
                        / f"sub-{subject}"
                        / f"ses-{file_session}"
                        / "func"
                    )
                events_file = _find_matching_bids_file(
                    events_dir,
                    left_file,
                    "_events.tsv",
                )

            mesh_paths = _find_surface_mesh_paths(
                preproc_root, subject, file_space
            )
            if mesh_paths is None:
                warnings.warn(
                    f"No matching surface meshes found for {left_file.name}, skipping."
                )
                continue

            sidecar_path = Path(str(left_file).replace(".func.gii", ".json"))
            if not sidecar_path.exists():
                warnings.warn(
                    f"JSON sidecar not found for {left_file.name}, skipping."
                )
                continue
            with open(sidecar_path, "r") as f:
                sidecar = json.load(f)
            TR = sidecar.get("RepetitionTime")
            if TR is None:
                warnings.warn(
                    f"RepetitionTime missing for {left_file.name}, skipping."
                )
                continue

            run_records.append(
                {
                    "func_files": {"L": left_file, "R": right_file},
                    "mesh_paths": mesh_paths,
                    "confounds_file": confounds_file,
                    "events_file": events_file,
                    "TR": TR,
                    "StartTime": sidecar.get("StartTime", 0),
                    "sidecar": sidecar,
                    "run_label": run_label,
                    "session_label": file_session,
                }
            )

        return run_records

    @staticmethod
    def _get_subject_func_dirs(
        subject_dir: Path, session: Optional[str]
    ) -> List[Tuple[Path, Optional[str]]]:
        if session is not None:
            return [(subject_dir / f"ses-{session}" / "func", session)]

        func_dirs = []
        root_func_dir = subject_dir / "func"
        if root_func_dir.exists():
            func_dirs.append((root_func_dir, None))

        for session_dir in sorted(subject_dir.glob("ses-*")):
            session_label = session_dir.name.replace("ses-", "", 1)
            func_dirs.append((session_dir / "func", session_label))

        if len(func_dirs) == 0:
            func_dirs.append((root_func_dir, None))
        return func_dirs

    @staticmethod
    def _find_anat_dir(
        preproc_root: Path, subject: str, session: Optional[str]
    ) -> Path:
        if session is not None:
            anat_dir = preproc_root / f"sub-{subject}" / f"ses-{session}" / "anat"
            if anat_dir.exists():
                return anat_dir
        anat_dir = preproc_root / f"sub-{subject}" / "anat"
        if anat_dir.exists():
            return anat_dir
        raise FileNotFoundError(
            f"No anatomical directory found for subject {subject}."
        )

    @staticmethod
    def _find_mask_file(
        anat_dir: Path,
        func_file: Path,
        subject: str,
        session: Optional[str],
        space: Optional[str],
        mask_suffix: str,
    ) -> Tuple[Optional[Path], bool]:
        ses_str = f"_ses-{session}" if session is not None else ""
        patterns = []
        if space in {None, "T1w"}:
            patterns.append(f"sub-{subject}{ses_str}{mask_suffix}")
        if space is not None and space != "T1w":
            patterns.append(f"sub-{subject}{ses_str}_space-{space}{mask_suffix}")
            patterns.append(f"sub-{subject}{ses_str}_space-{space}_res-*{mask_suffix}")

        for pattern in patterns:
            matches = sorted(anat_dir.glob(pattern))
            if len(matches) != 0:
                return matches[0], False

        func_mask_name = func_file.name.replace(
            "_desc-preproc_bold.nii.gz", mask_suffix
        )
        func_mask_file = func_file.parent / func_mask_name
        if func_mask_file.exists():
            return func_mask_file, True

        return None, False

    @staticmethod
    def _find_bold_sidecar(func_file: Path) -> Path:
        exact_match = func_file.parent / func_file.name.replace(".nii.gz", ".json")
        if exact_match.exists():
            return exact_match

        candidate = _find_matching_bids_file(
            func_file.parent,
            func_file,
            "_bold.json",
            required_entities=(
                "sub",
                "ses",
                "task",
                "acq",
                "ce",
                "dir",
                "rec",
                "run",
                "space",
            ),
        )
        if candidate is not None:
            return candidate

        return exact_match

    @staticmethod
    def _clean_run(record: Dict, config: Dict):
        confounds_raw = pd.read_csv(record["confounds_file"], sep="\t")
        confounds = confounds_raw.filter(regex=config["confounds_regex"])

        extra_confounds = []
        if (
            "framewise_displacement" in confounds_raw.columns
            and config["fd_threshold"] is not None
        ):
            fd_values = confounds_raw["framewise_displacement"].fillna(0)
            for idx, is_outlier in enumerate(fd_values > config["fd_threshold"]):
                if is_outlier:
                    outlier = np.zeros(len(confounds_raw))
                    outlier[idx] = 1
                    extra_confounds.append(
                        pd.DataFrame({f"fd_outlier_{idx:04d}": outlier})
                    )

        if (
            config["regress_out_task"]
            and record["events_file"] is not None
            and Path(record["events_file"]).exists()
        ):
            extra_confounds.extend(
                FunctionalConnectivityEstimator._build_task_regressors(
                    record["events_file"],
                    record["sidecar"],
                    record["TR"],
                    record["StartTime"],
                    len(confounds_raw),
                    task_conditions=config["regress_task_conditions"],
                )
            )

        all_confounds = [confounds] + extra_confounds
        confounds = pd.concat(all_confounds, axis=1).fillna(0)

        mask_img = load_img(record["mask_file"])
        mask_img = image.new_img_like(
            mask_img, image.get_data(mask_img).astype(np.float32)
        )
        if config["mask_fwhm"] is not None:
            mask_img = image.smooth_img(mask_img, fwhm=config["mask_fwhm"])
        if config["target_affine"] is not None:
            mask_img = image.resample_img(
                mask_img,
                target_affine=np.asarray(config["target_affine"]),
                interpolation="linear",
                copy_header=True,
                force_resample=True,
            )
            mask_img = image.math_img("img > 0", img=mask_img)
            mask_img = image.crop_img(mask_img, copy_header=True)

        func_img = load_img(record["func_file"])
        if len(func_img.shape) <= 3 or func_img.shape[3] < config["min_T"]:
            warnings.warn(
                f"Skipping {record['func_file'].name}: insufficient timepoints."
            )
            return None
        if config["target_affine"] is not None:
            func_img = image.resample_to_img(
                func_img, mask_img, copy_header=True, force_resample=True
            )

        masker = maskers.NiftiMasker(
            mask_img=mask_img,
            standardize=config["standardize"],
            detrend=config["detrend"],
            t_r=record["TR"],
            smoothing_fwhm=config["volume_fwhm"],
            low_pass=config["low_pass"],
            high_pass=config["high_pass"],
        )
        cleaned_data = _masker_fit_transform_with_explicit_mask(
            masker,
            func_img,
            confounds,
        )
        return masker.inverse_transform(cleaned_data)

    @staticmethod
    def _clean_surface_run(record: Dict, config: Dict):
        confounds_raw = pd.read_csv(record["confounds_file"], sep="\t")
        confounds = confounds_raw.filter(regex=config["confounds_regex"])

        extra_confounds = []
        if (
            "framewise_displacement" in confounds_raw.columns
            and config["fd_threshold"] is not None
        ):
            fd_values = confounds_raw["framewise_displacement"].fillna(0)
            for idx, is_outlier in enumerate(fd_values > config["fd_threshold"]):
                if is_outlier:
                    outlier = np.zeros(len(confounds_raw))
                    outlier[idx] = 1
                    extra_confounds.append(
                        pd.DataFrame({f"fd_outlier_{idx:04d}": outlier})
                    )

        if (
            config["regress_out_task"]
            and record["events_file"] is not None
            and Path(record["events_file"]).exists()
        ):
            extra_confounds.extend(
                FunctionalConnectivityEstimator._build_task_regressors(
                    record["events_file"],
                    record["sidecar"],
                    record["TR"],
                    record["StartTime"],
                    len(confounds_raw),
                    task_conditions=config["regress_task_conditions"],
                )
            )

        all_confounds = [confounds] + extra_confounds
        confounds = pd.concat(all_confounds, axis=1).fillna(0)
        confounds_arg = confounds if confounds.shape[1] != 0 else None

        func_img = load_surface_image(record["func_files"], record["mesh_paths"])
        cleaned_parts = {}
        n_timepoints = None
        for hemi in SURFACE_HEMIS:
            part_name = SURFACE_PARTS[hemi]
            hemi_data = np.asarray(
                func_img.data.parts[part_name], dtype=np.float32
            )
            if hemi_data.ndim == 1:
                hemi_data = hemi_data[:, None]
            if n_timepoints is None:
                n_timepoints = hemi_data.shape[-1]
            elif hemi_data.shape[-1] != n_timepoints:
                raise ValueError(
                    "Surface hemispheres must share the same number of timepoints."
                )
            if n_timepoints < config["min_T"]:
                warnings.warn(
                    "Skipping surface run: insufficient timepoints."
                )
                return None

            hemi_data = _smooth_surface_array(
                hemi_data,
                func_img.mesh.parts[part_name],
                config["surface_fwhm"],
            )
            cleaned = signal.clean(
                hemi_data.T,
                confounds=confounds_arg,
                detrend=config["detrend"],
                standardize=config["standardize"],
                low_pass=config["low_pass"],
                high_pass=config["high_pass"],
                t_r=record["TR"],
            )
            cleaned_parts[part_name] = cleaned.T.astype(np.float32)

        return SurfaceImage(
            mesh=func_img.mesh,
            data=cleaned_parts,
        )

    @staticmethod
    def _build_task_regressors(
        events_file: Path,
        sidecar: Dict,
        TR: float,
        start_time: float,
        n_timepoints: int,
        task_conditions: Optional[List[str]] = None,
    ) -> List[pd.DataFrame]:
        events = pd.read_csv(events_file, sep="\t")
        if "trial_type" not in events.columns:
            events["trial_type"] = "dummy"
        if task_conditions is not None:
            events = events[events["trial_type"].isin(task_conditions)].copy()
            if events.shape[0] == 0:
                return []
            events["trial_type"] = "selected_task"
        events = events[["trial_type", "onset", "duration"]]
        dummies = pd.get_dummies(
            events["trial_type"], prefix="trial_type", prefix_sep="."
        )
        events = pd.concat([dummies, events[["onset", "duration"]]], axis=1)
        trial_cols = [col for col in events.columns if col.startswith("trial_type")]

        frame_times = _frame_times_from_sidecar(
            sidecar, TR, start_time, n_timepoints
        )

        regressors = []
        for trial_col in trial_cols:
            regressor, _ = glm.first_level.compute_regressor(
                (
                    events["onset"],
                    events["duration"],
                    events[trial_col],
                ),
                "glover + derivative",
                frame_times,
            )
            if regressor.shape[1] == 1:
                column_names = [trial_col]
            else:
                column_names = [trial_col, f"{trial_col}_derivative"]
            regressors.append(pd.DataFrame(regressor, columns=column_names))
        return regressors

    @staticmethod
    def _select_task_condition_frames(
        record: Dict,
        n_timepoints: int,
        task_conditions: Optional[List[str]],
    ) -> Optional[np.ndarray]:
        if task_conditions is None:
            return None
        events_file = record["events_file"]
        if events_file is None or not Path(events_file).exists():
            warnings.warn(
                "Skipping run because task condition selection was requested "
                "but no events file is available."
            )
            return None

        events = pd.read_csv(events_file, sep="\t")
        if "trial_type" not in events.columns:
            warnings.warn(
                f"Skipping {Path(events_file).name}: events file does not "
                "contain a trial_type column."
            )
            return None

        events = events[events["trial_type"].isin(task_conditions)].copy()
        if events.shape[0] == 0:
            warnings.warn(
                f"Skipping {Path(events_file).name}: none of the requested "
                "task conditions were found."
            )
            return None

        frame_times = _frame_times_from_sidecar(
            record["sidecar"],
            record["TR"],
            record["StartTime"],
            n_timepoints,
        )
        frame_starts = frame_times - (record["TR"] / 2.0)
        frame_ends = frame_times + (record["TR"] / 2.0)
        selected = np.zeros(n_timepoints, dtype=bool)

        for _, event in events.iterrows():
            onset = float(event["onset"])
            duration = float(event["duration"])
            event_end = onset + max(duration, 0.0)
            selected |= (frame_starts <= event_end) & (frame_ends > onset)

        if not np.any(selected):
            warnings.warn(
                f"Skipping {Path(events_file).name}: none of the requested "
                "task conditions overlapped any sampled frames."
            )
            return None
        return selected

    @staticmethod
    def _run(cleaned_imgs: List, froi1_img, froi2_img) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if len(cleaned_imgs) == 0:
            raise ValueError("No cleaned runs available for connectivity analysis.")

        detail_rows = []
        resampled_masks = {}
        for run_idx, cleaned_img in enumerate(cleaned_imgs):
            if is_surface_image(cleaned_img):
                if not (
                    is_surface_image(froi1_img) and is_surface_image(froi2_img)
                ):
                    raise ValueError(
                        "Surface connectivity requires surface ROI masks."
                    )
                time_series = np.concatenate(
                    [
                        np.asarray(
                            cleaned_img.data.parts[SURFACE_PARTS[hemi]]
                        )
                        for hemi in SURFACE_HEMIS
                    ],
                    axis=0,
                ).T
                froi1_masks = flatten_image_data(froi1_img)
                froi2_masks = flatten_image_data(froi2_img)
                if (
                    time_series.shape[1] != froi1_masks.size
                    or time_series.shape[1] != froi2_masks.size
                ):
                    raise ValueError(
                        "Surface ROI masks must match the cleaned data vertex count."
                    )
            else:
                cleaned_data = cleaned_img.get_fdata()
                if cleaned_data.ndim != 4:
                    raise ValueError("Cleaned functional image must be 4D.")
                time_series = cleaned_data.reshape(
                    (-1, cleaned_data.shape[-1])
                ).T

                cache_key = (
                    cleaned_img.shape[:3],
                    tuple(cleaned_img.affine.flatten()),
                )
                if cache_key not in resampled_masks:
                    froi1_resampled = image.resample_to_img(
                        froi1_img,
                        image.index_img(cleaned_img, 0),
                        interpolation="nearest",
                        copy_header=True,
                        force_resample=True,
                    ).get_fdata().flatten()
                    froi2_resampled = image.resample_to_img(
                        froi2_img,
                        image.index_img(cleaned_img, 0),
                        interpolation="nearest",
                        copy_header=True,
                        force_resample=True,
                    ).get_fdata().flatten()
                    resampled_masks[cache_key] = (
                        froi1_resampled,
                        froi2_resampled,
                    )
                froi1_masks, froi2_masks = resampled_masks[cache_key]

            froi1_labels = np.unique(froi1_masks)
            froi1_labels = froi1_labels[
                ~np.isnan(froi1_labels) & (froi1_labels != 0)
            ]
            froi2_labels = np.unique(froi2_masks)
            froi2_labels = froi2_labels[
                ~np.isnan(froi2_labels) & (froi2_labels != 0)
            ]

            froi1_roi_masks = np.stack(
                [(froi1_masks == label) for label in froi1_labels], axis=0
            )
            froi2_roi_masks = np.stack(
                [(froi2_masks == label) for label in froi2_labels], axis=0
            )

            ts1 = np.array(
                [np.nanmean(time_series[:, mask], axis=1) for mask in froi1_roi_masks]
            )
            ts2 = np.array(
                [np.nanmean(time_series[:, mask], axis=1) for mask in froi2_roi_masks]
            )

            for i, label1 in enumerate(froi1_labels):
                for j, label2 in enumerate(froi2_labels):
                    valid = ~np.isnan(ts1[i]) & ~np.isnan(ts2[j])
                    if np.sum(valid) < 2:
                        fisher_z = np.nan
                    else:
                        corr = np.corrcoef(ts1[i, valid], ts2[j, valid])[0, 1]
                        corr = np.clip(
                            corr,
                            -1 + np.finfo(float).eps,
                            1 - np.finfo(float).eps,
                        )
                        fisher_z = np.arctanh(corr)
                    detail_rows.append(
                        {
                            "run": run_idx,
                            "froi1": label1,
                            "froi2": label2,
                            "fisher_z": fisher_z,
                        }
                    )

        df_detail = pd.DataFrame(detail_rows)
        df_summary = (
            df_detail.groupby(["froi1", "froi2"])
            .agg({"fisher_z": "mean"})
            .reset_index()
        )
        return df_summary, df_detail
