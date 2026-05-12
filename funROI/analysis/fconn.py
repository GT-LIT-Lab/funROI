from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import json
import re
import warnings

import numpy as np
import pandas as pd
from nilearn import glm, image, maskers
from nilearn.image import load_img

from ..froi import FROIConfig, _get_froi_data
from ..parcels import ParcelsConfig, get_parcels, is_no_parcels
from ..settings import get_bids_data_folder, get_bids_preprocessed_folder
from .utils import AnalysisSaver


FC_CLEAN_CONFIG = {
    "clean_surf": False,
    "mask_suffix": "_desc-brain_mask.nii.gz",
    "mask_fwhm": 2,
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
    "low_pass": 0.1,
    "high_pass": 0.01,
    "min_T": 50,
}


RUN_RE = re.compile(r"_run-([A-Za-z0-9]+)_")
SPACE_RE = re.compile(r"_space-([A-Za-z0-9]+)_")
SES_RE = re.compile(r"_ses-([A-Za-z0-9]+)_")


class FunctionalConnectivityEstimator(AnalysisSaver):
    """
    Estimate functional connectivity from cleaned preprocessed BOLD runs.
    """

    def __init__(self):
        self._type = "fconn"
        self._data_summary = None
        self._data_detail = None

    def run(
        self,
        froi1: Union[FROIConfig, str, ParcelsConfig],
        froi2: Union[FROIConfig, str, ParcelsConfig],
        subject1: Optional[str] = None,
        subject2: Optional[str] = None,
        task: Optional[str] = None,
        run1: Optional[str] = None,
        run2: Optional[str] = None,
        session: Optional[str] = None,
        space: Optional[str] = None,
        clean_surf: bool = False,
        mask_suffix: str = "_desc-brain_mask.nii.gz",
        mask_fwhm: Optional[float] = 2,
        target_affine: Optional[List[List[float]]] = None,
        confounds_regex: str = FC_CLEAN_CONFIG["confounds_regex"],
        fd_threshold: Optional[float] = 0.5,
        volume_fwhm: Optional[float] = 4,
        surface_fwhm: Optional[float] = 4,
        standardize: bool = True,
        detrend: bool = True,
        regress_out_task: bool = True,
        low_pass: Optional[float] = 0.1,
        high_pass: Optional[float] = 0.01,
        min_T: int = 50,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the functional connectivity analysis.
        """
        self.froi1 = froi1
        self.froi2 = froi2
        self.run1 = run1
        self.run2 = run2

        subject = self._resolve_subject(subject1, subject2)
        task = self._resolve_task(task, froi1, froi2)
        clean_config = {
            "clean_surf": clean_surf,
            "mask_suffix": mask_suffix,
            "mask_fwhm": mask_fwhm,
            "target_affine": target_affine,
            "confounds_regex": confounds_regex,
            "fd_threshold": fd_threshold,
            "volume_fwhm": volume_fwhm,
            "surface_fwhm": surface_fwhm,
            "standardize": standardize,
            "detrend": detrend,
            "regress_out_task": regress_out_task,
            "low_pass": low_pass,
            "high_pass": high_pass,
            "min_T": min_T,
        }

        froi1_labels, froi1_name, froi1_has_explicit_parcels, froi1_img = (
            self._resolve_mask_data(froi1, subject, run1)
        )
        froi2_labels, froi2_name, froi2_has_explicit_parcels, froi2_img = (
            self._resolve_mask_data(froi2, subject, run2)
        )

        cleaned_imgs, cleaned_run_labels = self._get_cleaned_imgs_by_run(
            subject, task, session, space, clean_config
        )
        df_summary, df_detail = self._run(cleaned_imgs, froi1_img, froi2_img)

        df_detail["bold_run"] = df_detail["run"].apply(
            lambda idx: cleaned_run_labels[idx]
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

        self._data_summary = df_summary
        self._data_detail = df_detail
        new_info = pd.DataFrame(
            {
                "task": [task],
                "session": [session],
                "space": [space],
                "froi1": [self.froi1],
                "froi2": [self.froi2],
                "customized_run1": [run1],
                "customized_run2": [run2],
                "clean_config": [clean_config],
            }
        )
        self._save(new_info)
        return self._data_summary, self._data_detail

    @staticmethod
    def _resolve_subject(
        subject1: Optional[str], subject2: Optional[str]
    ) -> str:
        if subject1 is None and subject2 is None:
            raise ValueError(
                "A subject label is required for functional connectivity."
            )
        if subject1 is None:
            subject1 = subject2
        if subject2 is None:
            subject2 = subject1
        if subject1 != subject2:
            raise ValueError(
                "Functional connectivity must be computed within a single subject."
            )
        return subject1

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
    ) -> Tuple[List, List[str]]:
        if config["clean_surf"]:
            raise NotImplementedError(
                "Surface-based functional connectivity cleaning is not implemented."
            )

        run_records = FunctionalConnectivityEstimator._find_preprocessed_runs(
            subject, task, session, space, config["mask_suffix"]
        )
        cleaned_imgs = []
        cleaned_run_labels = []
        for record in run_records:
            cleaned_img = FunctionalConnectivityEstimator._clean_run(record, config)
            if cleaned_img is None:
                continue
            cleaned_imgs.append(cleaned_img)
            cleaned_run_labels.append(record["run_label"])

        if len(cleaned_imgs) == 0:
            raise ValueError(
                f"No usable preprocessed runs found for subject {subject} and task {task}."
            )
        return cleaned_imgs, cleaned_run_labels

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
        if session is not None:
            func_dir = subject_dir / f"ses-{session}" / "func"
        else:
            session_dirs = sorted(subject_dir.glob("ses-*"))
            if len(session_dirs) > 1:
                raise ValueError(
                    "Multiple sessions were found. Please specify `session`."
                )
            if len(session_dirs) == 1:
                func_dir = session_dirs[0] / "func"
            else:
                func_dir = subject_dir / "func"

        if not func_dir.exists():
            raise ValueError(f"Functional directory not found: {func_dir}")

        func_files = sorted(func_dir.glob(f"*task-{task}_*desc-preproc_bold.nii.gz"))
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
            prefix = re.sub(
                r"_space-[A-Za-z0-9]+", "", func_file.name
            ).replace("_desc-preproc_bold.nii.gz", "")

            confounds_file = func_file.parent / f"{prefix}_desc-confounds_timeseries.tsv"
            if not confounds_file.exists():
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
                candidate = events_dir / f"{prefix}_events.tsv"
                events_file = candidate if candidate.exists() else None

            anat_dir = FunctionalConnectivityEstimator._find_anat_dir(
                preproc_root, subject, file_session
            )
            mask_file = FunctionalConnectivityEstimator._find_mask_file(
                anat_dir, subject, file_session, file_space, mask_suffix
            )
            if mask_file is None:
                warnings.warn(
                    f"No matching mask found for {func_file.name}, skipping."
                )
                continue

            sidecar_path = Path(str(func_file).replace(".nii.gz", ".json"))
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
                }
            )

        return run_records

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
        subject: str,
        session: Optional[str],
        space: Optional[str],
        mask_suffix: str,
    ) -> Optional[Path]:
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
                return matches[0]
        return None

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
        cleaned_data = masker.fit_transform(func_img, confounds=confounds)
        return masker.inverse_transform(cleaned_data)

    @staticmethod
    def _build_task_regressors(
        events_file: Path,
        sidecar: Dict,
        TR: float,
        start_time: float,
        n_timepoints: int,
    ) -> List[pd.DataFrame]:
        events = pd.read_csv(events_file, sep="\t")
        if "trial_type" not in events.columns:
            events["trial_type"] = "dummy"
        events = events[["trial_type", "onset", "duration"]]
        dummies = pd.get_dummies(
            events["trial_type"], prefix="trial_type", prefix_sep="."
        )
        events = pd.concat([dummies, events[["onset", "duration"]]], axis=1)
        trial_cols = [col for col in events.columns if col.startswith("trial_type")]

        slice_time_ref = sidecar.get("SliceTimingCorrected")
        if slice_time_ref is True:
            slice_time_ref = 0.5
        elif slice_time_ref is None:
            slice_time_ref = 0.0
        else:
            slice_time_ref = float(slice_time_ref)
        frame_times = (
            np.arange(n_timepoints) * TR + start_time + (slice_time_ref * TR)
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
    def _run(cleaned_imgs: List, froi1_img, froi2_img) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if len(cleaned_imgs) == 0:
            raise ValueError("No cleaned runs available for connectivity analysis.")

        detail_rows = []
        resampled_masks = {}
        for run_idx, cleaned_img in enumerate(cleaned_imgs):
            cleaned_data = cleaned_img.get_fdata()
            if cleaned_data.ndim != 4:
                raise ValueError("Cleaned functional image must be 4D.")
            time_series = cleaned_data.reshape((-1, cleaned_data.shape[-1])).T

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
                resampled_masks[cache_key] = (froi1_resampled, froi2_resampled)
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
