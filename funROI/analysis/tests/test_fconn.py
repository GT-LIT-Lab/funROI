import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from nibabel.nifti1 import Nifti1Image
from nilearn.surface import InMemoryMesh, SurfaceImage

import funROI.analysis.fconn as fconn_mod
from funROI.analysis.tests.utils import DummyFROI


def _img_from_flat(flat: np.ndarray) -> Nifti1Image:
    data = np.asarray(flat, dtype=np.float32).reshape((1, 1, -1))
    return Nifti1Image(data, np.eye(4))


def _bold_img(time_by_voxel: np.ndarray) -> Nifti1Image:
    data = np.asarray(time_by_voxel, dtype=np.float32).T.reshape((1, 1, -1, time_by_voxel.shape[0]))
    return Nifti1Image(data, np.eye(4))


def _surface_mesh(offset: float = 0.0) -> InMemoryMesh:
    coordinates = np.array(
        [
            [0.0 + offset, 0.0, 0.0],
            [1.0 + offset, 0.0, 0.0],
            [0.0 + offset, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return InMemoryMesh(coordinates, faces)


def _surface_img(left, right) -> SurfaceImage:
    return SurfaceImage(
        mesh={"left": _surface_mesh(), "right": _surface_mesh(2.0)},
        data={
            "left": np.asarray(left, dtype=np.float32),
            "right": np.asarray(right, dtype=np.float32),
        },
    )


def test_fconn_run_basic_properties():
    cleaned_imgs = [
        _bold_img(
            np.array(
                [
                    [1.0, 2.0, 10.0, 20.0],
                    [2.0, 3.0, 20.0, 30.0],
                    [3.0, 4.0, 30.0, 40.0],
                ]
            )
        )
    ]
    froi1 = _img_from_flat(np.array([1.0, 1.0, 0.0, 0.0]))
    froi2 = _img_from_flat(np.array([0.0, 0.0, 2.0, 2.0]))

    summary, detail = fconn_mod.FunctionalConnectivityEstimator._run(
        cleaned_imgs, froi1, froi2
    )

    assert list(summary.columns) == ["froi1", "froi2", "fisher_z"]
    assert list(detail.columns) == ["run", "froi1", "froi2", "fisher_z"]
    assert summary.shape[0] == 1
    assert np.isfinite(summary["fisher_z"].iloc[0])


def test_fconn_run_requires_single_subject():
    est = fconn_mod.FunctionalConnectivityEstimator(["S1", "S2"], "P1", "P2")
    assert est.subjects == ["S1", "S2"]


def test_normalize_standardize_arg_matches_nilearn_015_values():
    assert fconn_mod._normalize_standardize_arg(True) == "zscore_sample"
    assert fconn_mod._normalize_standardize_arg(False) is None
    assert (
        fconn_mod._normalize_standardize_arg("zscore")
        == "zscore"
    )
    assert fconn_mod._normalize_standardize_arg(None) is None


def test_fconn_run_parcels_and_froi(monkeypatch):
    est = fconn_mod.FunctionalConnectivityEstimator(
        ["S1"], "P1", DummyFROI(task="TASK")
    )
    monkeypatch.setattr(
        fconn_mod.FunctionalConnectivityEstimator,
        "_save",
        lambda self, info: None,
    )
    monkeypatch.setattr(fconn_mod, "FROIConfig", DummyFROI, raising=False)

    parcels_img = _img_from_flat(np.array([1.0, 1.0, 0.0, 0.0]))

    def fake_get_parcels(arg):
        if arg == "P1":
            return parcels_img, {1.0: "ParcelA"}
        if arg == "dummy_parcels":
            return None, {2.0: "ROI"}
        raise AssertionError(f"unexpected parcels arg {arg}")

    monkeypatch.setattr(fconn_mod, "get_parcels", fake_get_parcels)
    monkeypatch.setattr(
        fconn_mod,
        "_get_froi_data",
        lambda subject, cfg, run_label, return_nifti=False: _img_from_flat(
            np.array([0.0, 0.0, 2.0, 2.0])
        ),
    )
    monkeypatch.setattr(
        fconn_mod.FunctionalConnectivityEstimator,
        "_get_cleaned_imgs_by_run",
        staticmethod(lambda subject, task, session, space, config: (
            [_bold_img(np.array([[1.0, 2.0, 10.0, 20.0], [2.0, 3.0, 20.0, 30.0]]))],
            ["01"],
            ["A"],
        )),
    )

    summary, detail = est.run(task="TASK", froi2_run_label="all")

    assert "parcel1" in summary.columns
    assert "froi2" in summary.columns
    assert "bold_run" in detail.columns
    assert "bold_session" in detail.columns
    assert set(summary["parcel1"]) == {"ParcelA"}
    assert set(summary["froi2"]) == {"ROI"}
    assert set(detail["subject"]) == {"S1"}
    assert set(detail["bold_session"]) == {"A"}


def test_fconn_run_omits_froi_column_when_parcelless(monkeypatch):
    est = fconn_mod.FunctionalConnectivityEstimator(
        ["S1"],
        DummyFROI(task="TASK", parcels="none"),
        DummyFROI(task="TASK", parcels="none"),
    )
    monkeypatch.setattr(
        fconn_mod.FunctionalConnectivityEstimator,
        "_save",
        lambda self, info: None,
    )
    monkeypatch.setattr(fconn_mod, "FROIConfig", DummyFROI, raising=False)
    monkeypatch.setattr(fconn_mod, "get_parcels", lambda arg: (None, None))
    monkeypatch.setattr(
        fconn_mod,
        "_get_froi_data",
        lambda subject, cfg, run_label, return_nifti=False: _img_from_flat(
            np.array([1.0, 0.0, 0.0, 0.0])
        ),
    )
    monkeypatch.setattr(
        fconn_mod.FunctionalConnectivityEstimator,
        "_get_cleaned_imgs_by_run",
        staticmethod(lambda subject, task, session, space, config: (
            [_bold_img(np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]))],
            ["01"],
            [None],
        )),
    )

    summary, detail = est.run()

    assert "froi1" not in summary.columns
    assert "froi2" not in summary.columns
    assert "froi1" not in detail.columns
    assert "froi2" not in detail.columns


def test_fconn_run_passes_cleaning_overrides(monkeypatch):
    est = fconn_mod.FunctionalConnectivityEstimator(["S1"], "P1", "P1")
    monkeypatch.setattr(
        fconn_mod.FunctionalConnectivityEstimator,
        "_save",
        lambda self, info: None,
    )
    monkeypatch.setattr(fconn_mod, "get_parcels", lambda arg: (_img_from_flat(np.array([1.0, 0.0])), {1.0: "A"}))

    captured = {}

    def fake_get_cleaned(subject, task, session, space, config):
        captured.update(config)
        return ([_bold_img(np.array([[1.0, 2.0], [2.0, 3.0]]))], ["01"], [None])

    monkeypatch.setattr(
        fconn_mod.FunctionalConnectivityEstimator,
        "_get_cleaned_imgs_by_run",
        staticmethod(fake_get_cleaned),
    )

    est.run(
        task="TASK",
        volume_fwhm=6,
        low_pass=0.2,
        regress_out_task=False,
        task_conditions=["story", "math"],
        regress_task_conditions=["story"],
        concat_conditions_across_runs=True,
    )

    assert captured["volume_fwhm"] == 6
    assert captured["low_pass"] == 0.2
    assert captured["regress_out_task"] is False
    assert captured["task_conditions"] == ["story", "math"]
    assert captured["regress_task_conditions"] == ["story"]
    assert captured["concat_conditions_across_runs"] is True


def test_preprocess_and_load_preprocessed_bold_for_fc_uses_cache(
    monkeypatch, tmp_path
):
    preproc_root = tmp_path / "preproc"
    deriv_root = tmp_path / "derivatives"
    func_dir = preproc_root / "sub-S1" / "func"
    func_dir.mkdir(parents=True)

    func_file = (
        func_dir
        / "sub-S1_task-LANGUAGE_run-1_space-MNINonLinear_desc-preproc_bold.nii.gz"
    )
    func_file.write_bytes(b"")
    confounds_file = (
        func_dir / "sub-S1_task-LANGUAGE_run-1_desc-confounds_timeseries.tsv"
    )
    confounds_file.write_text("framewise_displacement\n0.0\n", encoding="utf-8")
    mask_file = (
        func_dir
        / "sub-S1_task-LANGUAGE_run-1_space-MNINonLinear_desc-brain_mask.nii.gz"
    )
    mask_file.write_bytes(b"")

    record = {
        "func_file": func_file,
        "confounds_file": confounds_file,
        "events_file": None,
        "mask_file": mask_file,
        "TR": 1.0,
        "StartTime": 0.0,
        "sidecar": {"RepetitionTime": 1.0},
        "run_label": "01",
        "session_label": None,
    }

    monkeypatch.setattr(
        fconn_mod,
        "get_bids_preprocessed_folder",
        lambda: preproc_root,
    )
    monkeypatch.setattr(
        fconn_mod,
        "get_bids_deriv_folder",
        lambda: deriv_root,
    )
    monkeypatch.setattr(
        fconn_mod.FunctionalConnectivityEstimator,
        "_find_preprocessed_runs",
        staticmethod(lambda *args, **kwargs: [record]),
    )

    clean_calls = {"count": 0}

    def fake_clean(record_arg, config_arg):
        clean_calls["count"] += 1
        return _bold_img(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]))

    monkeypatch.setattr(
        fconn_mod.FunctionalConnectivityEstimator,
        "_clean_run",
        staticmethod(fake_clean),
    )

    manifest = fconn_mod.preprocess_bold_for_fc(["S1"], task="LANGUAGE")

    assert manifest.shape[0] == 1
    assert clean_calls["count"] == 1

    monkeypatch.setattr(
        fconn_mod.FunctionalConnectivityEstimator,
        "_clean_run",
        staticmethod(
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("cache should be used")
            )
        ),
    )

    prepared = fconn_mod.load_preprocessed_bold_for_fc("S1", task="LANGUAGE")

    assert len(prepared) == 1
    assert prepared[0]["run_label"] == "01"
    assert prepared[0]["cleaned_img"].get_fdata().shape[-1] == 3


def test_build_task_regressors_returns_canonical_and_derivative(monkeypatch, tmp_path):
    events_file = tmp_path / "events.tsv"
    pd.DataFrame(
        {
            "trial_type": ["story", "math"],
            "onset": [0.0, 10.0],
            "duration": [5.0, 5.0],
        }
    ).to_csv(events_file, sep="\t", index=False)

    monkeypatch.setattr(
        fconn_mod.glm.first_level,
        "compute_regressor",
        lambda *args, **kwargs: (
            np.ones((4, 2), dtype=float),
            None,
        ),
    )

    regs = fconn_mod.FunctionalConnectivityEstimator._build_task_regressors(
        events_file=events_file,
        sidecar={"SliceTimingCorrected": True},
        TR=2.0,
        start_time=0.0,
        n_timepoints=4,
    )

    assert len(regs) == 2
    assert list(regs[0].columns) == [
        "trial_type.math",
        "trial_type.math_derivative",
    ] or list(regs[0].columns) == [
        "trial_type.story",
        "trial_type.story_derivative",
    ]


def test_build_task_regressors_can_select_and_concatenate_conditions(
    monkeypatch, tmp_path
):
    events_file = tmp_path / "events.tsv"
    pd.DataFrame(
        {
            "trial_type": ["story", "math", "fix"],
            "onset": [0.0, 10.0, 20.0],
            "duration": [5.0, 5.0, 5.0],
        }
    ).to_csv(events_file, sep="\t", index=False)

    monkeypatch.setattr(
        fconn_mod.glm.first_level,
        "compute_regressor",
        lambda *args, **kwargs: (
            np.ones((4, 2), dtype=float),
            None,
        ),
    )

    regs = fconn_mod.FunctionalConnectivityEstimator._build_task_regressors(
        events_file=events_file,
        sidecar={"SliceTimingCorrected": True},
        TR=2.0,
        start_time=0.0,
        n_timepoints=4,
        task_conditions=["story", "math"],
    )

    assert len(regs) == 1
    assert list(regs[0].columns) == [
        "trial_type.selected_task",
        "trial_type.selected_task_derivative",
    ]


def test_build_task_regressors_returns_empty_when_selected_conditions_missing(
    monkeypatch, tmp_path
):
    events_file = tmp_path / "events.tsv"
    pd.DataFrame(
        {
            "trial_type": ["story"],
            "onset": [0.0],
            "duration": [5.0],
        }
    ).to_csv(events_file, sep="\t", index=False)

    monkeypatch.setattr(
        fconn_mod.glm.first_level,
        "compute_regressor",
        lambda *args, **kwargs: (
            np.ones((4, 2), dtype=float),
            None,
        ),
    )

    regs = fconn_mod.FunctionalConnectivityEstimator._build_task_regressors(
        events_file=events_file,
        sidecar={"SliceTimingCorrected": True},
        TR=2.0,
        start_time=0.0,
        n_timepoints=4,
        task_conditions=["math"],
    )

    assert regs == []


def test_select_task_condition_frames_concatenates_selected_events(tmp_path):
    events_file = tmp_path / "events.tsv"
    pd.DataFrame(
        {
            "trial_type": ["story", "fix", "math"],
            "onset": [0.0, 4.0, 8.0],
            "duration": [2.0, 2.0, 2.0],
        }
    ).to_csv(events_file, sep="\t", index=False)

    record = {
        "events_file": events_file,
        "sidecar": {"SliceTimingCorrected": False},
        "TR": 2.0,
        "StartTime": 0.0,
    }

    selected = fconn_mod.FunctionalConnectivityEstimator._select_task_condition_frames(
        record,
        n_timepoints=6,
        task_conditions=["story", "math"],
    )

    assert selected.tolist() == [True, True, False, False, True, True]


def test_select_task_condition_frames_returns_none_when_events_missing(tmp_path):
    record = {
        "events_file": tmp_path / "missing.tsv",
        "sidecar": {"SliceTimingCorrected": False},
        "TR": 2.0,
        "StartTime": 0.0,
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        selected = fconn_mod.FunctionalConnectivityEstimator._select_task_condition_frames(
            record,
            n_timepoints=6,
            task_conditions=["story"],
        )

    assert selected is None
    assert any("no events file is available" in str(w.message) for w in caught)


def test_filter_preprocessed_runs_applies_min_t_to_full_runs(
    monkeypatch,
):
    prepared_runs = [
        {
            "cleaned_img": _bold_img(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])),
            "run_label": "01",
            "session_label": None,
            "func_file": Path("run-01.nii.gz"),
            "events_file": Path("run-01_events.tsv"),
            "sidecar": {"SliceTimingCorrected": False},
            "TR": 1.0,
            "StartTime": 0.0,
        },
        {
            "cleaned_img": _bold_img(np.array([[4.0, 5.0, 6.0]])),
            "run_label": "02",
            "session_label": None,
            "func_file": Path("run-02.nii.gz"),
            "events_file": Path("run-02_events.tsv"),
            "sidecar": {"SliceTimingCorrected": False},
            "TR": 1.0,
            "StartTime": 0.0,
        },
    ]

    selected_runs = fconn_mod.FunctionalConnectivityEstimator._filter_preprocessed_runs_by_min_t(
        prepared_runs,
        min_T=2,
    )

    assert len(selected_runs) == 1
    assert selected_runs[0]["run_label"] == "01"


def test_select_preprocessed_runs_does_not_reapply_min_t_after_condition_selection(
    monkeypatch,
):
    prepared_runs = [
        {
            "cleaned_img": _bold_img(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])),
            "run_label": "01",
            "session_label": None,
            "func_file": Path("run-01.nii.gz"),
            "events_file": Path("run-01_events.tsv"),
            "sidecar": {"SliceTimingCorrected": False},
            "TR": 1.0,
            "StartTime": 0.0,
        },
        {
            "cleaned_img": _bold_img(np.array([[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]])),
            "run_label": "02",
            "session_label": None,
            "func_file": Path("run-02.nii.gz"),
            "events_file": Path("run-02_events.tsv"),
            "sidecar": {"SliceTimingCorrected": False},
            "TR": 1.0,
            "StartTime": 0.0,
        },
    ]

    monkeypatch.setattr(
        fconn_mod.FunctionalConnectivityEstimator,
        "_select_task_condition_frames",
        staticmethod(
            lambda record, n_timepoints, task_conditions: np.array(
                [True, True, False]
            )
        ),
    )

    selected_runs = fconn_mod.FunctionalConnectivityEstimator._select_preprocessed_runs(
        prepared_runs,
        ["story"],
        concat_conditions_across_runs=False,
    )

    assert len(selected_runs) == 2
    assert selected_runs[0]["cleaned_img"].get_fdata().shape[-1] == 2
    assert selected_runs[1]["cleaned_img"].get_fdata().shape[-1] == 2


def test_select_preprocessed_runs_can_concatenate_across_runs(monkeypatch):
    prepared_runs = [
        {
            "cleaned_img": _bold_img(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])),
            "run_label": "01",
            "session_label": None,
            "func_file": Path("run-01.nii.gz"),
            "events_file": Path("run-01_events.tsv"),
            "sidecar": {"SliceTimingCorrected": False},
            "TR": 1.0,
            "StartTime": 0.0,
        },
        {
            "cleaned_img": _bold_img(np.array([[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]])),
            "run_label": "02",
            "session_label": None,
            "func_file": Path("run-02.nii.gz"),
            "events_file": Path("run-02_events.tsv"),
            "sidecar": {"SliceTimingCorrected": False},
            "TR": 1.0,
            "StartTime": 0.0,
        },
    ]

    monkeypatch.setattr(
        fconn_mod.FunctionalConnectivityEstimator,
        "_select_task_condition_frames",
        staticmethod(
            lambda record, n_timepoints, task_conditions: np.array(
                [True, True, False]
            )
        ),
    )

    selected_runs = fconn_mod.FunctionalConnectivityEstimator._select_preprocessed_runs(
        prepared_runs,
        ["story"],
        concat_conditions_across_runs=True,
    )

    assert len(selected_runs) == 1
    assert selected_runs[0]["cleaned_img"].get_fdata().shape[-1] == 4
    assert selected_runs[0]["run_label"] == "concatenated:01+02"


def test_fconn_run_surface_properties():
    cleaned_imgs = [
        _surface_img(
            left=np.array(
                [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [0.0, 1.0, 2.0]]
            ),
            right=np.array(
                [[10.0, 20.0, 30.0], [20.0, 30.0, 40.0], [5.0, 6.0, 7.0]]
            ),
        )
    ]
    froi1 = _surface_img([1.0, 1.0, 0.0], [0.0, 0.0, 0.0])
    froi2 = _surface_img([0.0, 0.0, 0.0], [2.0, 2.0, 0.0])

    summary, detail = fconn_mod.FunctionalConnectivityEstimator._run(
        cleaned_imgs, froi1, froi2
    )

    assert list(summary.columns) == ["froi1", "froi2", "fisher_z"]
    assert list(detail.columns) == ["run", "froi1", "froi2", "fisher_z"]
    assert summary.shape[0] == 1
    assert np.isfinite(summary["fisher_z"].iloc[0])


def test_fconn_run_rejects_mixed_surface_and_volume_inputs(monkeypatch):
    est = fconn_mod.FunctionalConnectivityEstimator(["S1"], "surface", "volume")
    monkeypatch.setattr(
        fconn_mod.FunctionalConnectivityEstimator,
        "_save",
        lambda self, info: None,
    )

    def fake_get_parcels(arg):
        if arg == "surface":
            return _surface_img([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]), {1.0: "S"}
        if arg == "volume":
            return _img_from_flat(np.array([1.0, 0.0])), {1.0: "V"}
        raise AssertionError(f"unexpected parcels arg {arg}")

    monkeypatch.setattr(fconn_mod, "get_parcels", fake_get_parcels)

    with pytest.raises(ValueError, match="both ROI inputs to be surface-based"):
        est.run(task="TASK")


def test_find_preprocessed_runs_falls_back_to_functional_mask(monkeypatch, tmp_path):
    bids_root = tmp_path / "bids"
    subject = "S1"
    func_dir = bids_root / f"sub-{subject}" / "func"
    anat_dir = bids_root / f"sub-{subject}" / "anat"
    func_dir.mkdir(parents=True)
    anat_dir.mkdir(parents=True)

    func_file = (
        func_dir
        / f"sub-{subject}_task-LANGUAGE_run-1_space-MNINonLinear_desc-preproc_bold.nii.gz"
    )
    func_file.write_bytes(b"")
    (func_dir / f"sub-{subject}_task-LANGUAGE_run-1_desc-confounds_timeseries.tsv").write_text(
        "framewise_displacement\n0.0\n",
        encoding="utf-8",
    )
    sidecar = (
        func_dir
        / f"sub-{subject}_task-LANGUAGE_run-1_space-MNINonLinear_bold.json"
    )
    sidecar.write_text(
        json.dumps({"RepetitionTime": 0.72}),
        encoding="utf-8",
    )
    func_mask = (
        func_dir
        / f"sub-{subject}_task-LANGUAGE_run-1_space-MNINonLinear_desc-brain_mask.nii.gz"
    )
    func_mask.write_bytes(b"")

    monkeypatch.setattr(
        fconn_mod,
        "get_bids_preprocessed_folder",
        lambda: bids_root,
    )
    monkeypatch.setattr(
        fconn_mod,
        "get_bids_data_folder",
        lambda: bids_root,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_records = fconn_mod.FunctionalConnectivityEstimator._find_preprocessed_runs(
            subject=subject,
            task="LANGUAGE",
            session=None,
            space="MNINonLinear",
            mask_suffix="_desc-brain_mask.nii.gz",
        )

    assert len(run_records) == 1
    assert run_records[0]["mask_file"] == func_mask
    assert run_records[0]["sidecar"] == {"RepetitionTime": 0.72}
    assert any("using functional mask" in str(w.message) for w in caught)


def test_find_preprocessed_runs_matches_confounds_without_space_or_res_entities(
    monkeypatch, tmp_path
):
    bids_root = tmp_path / "bids"
    subject = "S1"
    session_label = "01"
    func_dir = bids_root / f"sub-{subject}" / f"ses-{session_label}" / "func"
    anat_dir = bids_root / f"sub-{subject}" / f"ses-{session_label}" / "anat"
    func_dir.mkdir(parents=True)
    anat_dir.mkdir(parents=True)

    func_file = (
        func_dir
        / (
            f"sub-{subject}_ses-{session_label}_task-LANGUAGE_run-1_"
            "space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
        )
    )
    func_file.write_bytes(b"")
    confounds_file = (
        func_dir
        / f"sub-{subject}_ses-{session_label}_task-LANGUAGE_run-1_desc-confounds_timeseries.tsv"
    )
    confounds_file.write_text(
        "framewise_displacement\n0.0\n",
        encoding="utf-8",
    )
    events_file = (
        func_dir
        / f"sub-{subject}_ses-{session_label}_task-LANGUAGE_run-1_events.tsv"
    )
    events_file.write_text(
        "onset\tduration\ttrial_type\n0\t1\tstory\n",
        encoding="utf-8",
    )
    (
        func_dir
        / (
            f"sub-{subject}_ses-{session_label}_task-LANGUAGE_run-1_"
            "space-MNI152NLin2009cAsym_bold.json"
        )
    ).write_text(json.dumps({"RepetitionTime": 0.72}), encoding="utf-8")
    (
        func_dir
        / (
            f"sub-{subject}_ses-{session_label}_task-LANGUAGE_run-1_"
            "space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
        )
    ).write_bytes(b"")

    monkeypatch.setattr(
        fconn_mod,
        "get_bids_preprocessed_folder",
        lambda: bids_root,
    )
    monkeypatch.setattr(
        fconn_mod,
        "get_bids_data_folder",
        lambda: bids_root,
    )

    run_records = fconn_mod.FunctionalConnectivityEstimator._find_preprocessed_runs(
        subject=subject,
        task="LANGUAGE",
        session=session_label,
        space="MNI152NLin2009cAsym",
        mask_suffix="_desc-brain_mask.nii.gz",
    )

    assert len(run_records) == 1
    assert run_records[0]["confounds_file"] == confounds_file
    assert run_records[0]["events_file"] == events_file


def test_find_preprocessed_runs_uses_all_sessions_when_session_none(
    monkeypatch, tmp_path
):
    bids_root = tmp_path / "bids"
    subject = "S1"
    for session_label in ("01", "02"):
        func_dir = bids_root / f"sub-{subject}" / f"ses-{session_label}" / "func"
        anat_dir = bids_root / f"sub-{subject}" / f"ses-{session_label}" / "anat"
        func_dir.mkdir(parents=True)
        anat_dir.mkdir(parents=True)

        func_file = (
            func_dir
            / (
                f"sub-{subject}_ses-{session_label}_task-LANGUAGE_run-1_"
                "space-MNINonLinear_desc-preproc_bold.nii.gz"
            )
        )
        func_file.write_bytes(b"")
        (
            func_dir
            / f"sub-{subject}_ses-{session_label}_task-LANGUAGE_run-1_desc-confounds_timeseries.tsv"
        ).write_text("framewise_displacement\n0.0\n", encoding="utf-8")
        (
            func_dir
            / f"sub-{subject}_ses-{session_label}_task-LANGUAGE_run-1_space-MNINonLinear_bold.json"
        ).write_text(json.dumps({"RepetitionTime": 0.72}), encoding="utf-8")
        (
            func_dir
            / f"sub-{subject}_ses-{session_label}_task-LANGUAGE_run-1_space-MNINonLinear_desc-brain_mask.nii.gz"
        ).write_bytes(b"")

    monkeypatch.setattr(
        fconn_mod,
        "get_bids_preprocessed_folder",
        lambda: bids_root,
    )
    monkeypatch.setattr(
        fconn_mod,
        "get_bids_data_folder",
        lambda: bids_root,
    )

    run_records = fconn_mod.FunctionalConnectivityEstimator._find_preprocessed_runs(
        subject=subject,
        task="LANGUAGE",
        session=None,
        space="MNINonLinear",
        mask_suffix="_desc-brain_mask.nii.gz",
    )

    assert len(run_records) == 2
    assert [record["session_label"] for record in run_records] == ["01", "02"]


def test_find_preprocessed_surface_runs_uses_all_sessions_when_session_none(
    monkeypatch, tmp_path
):
    bids_root = tmp_path / "bids"
    subject = "S1"
    for session_label in ("01", "02"):
        func_dir = bids_root / f"sub-{subject}" / f"ses-{session_label}" / "func"
        anat_dir = bids_root / f"sub-{subject}" / "anat"
        func_dir.mkdir(parents=True)
        anat_dir.mkdir(parents=True, exist_ok=True)

        left_file = (
            func_dir
            / (
                f"sub-{subject}_ses-{session_label}_task-LANGUAGE_run-1_hemi-L_"
                "space-fsLR32k_desc-preproc_bold.func.gii"
            )
        )
        left_file.write_bytes(b"")
        right_file = Path(str(left_file).replace("_hemi-L_", "_hemi-R_"))
        right_file.write_bytes(b"")
        Path(str(left_file).replace(".func.gii", ".json")).write_text(
            json.dumps({"RepetitionTime": 0.72}),
            encoding="utf-8",
        )
        Path(str(right_file).replace(".func.gii", ".json")).write_text(
            json.dumps({"RepetitionTime": 0.72}),
            encoding="utf-8",
        )
        (
            func_dir
            / f"sub-{subject}_ses-{session_label}_task-LANGUAGE_run-1_desc-confounds_timeseries.tsv"
        ).write_text("framewise_displacement\n0.0\n", encoding="utf-8")

    monkeypatch.setattr(
        fconn_mod,
        "get_bids_preprocessed_folder",
        lambda: bids_root,
    )
    monkeypatch.setattr(
        fconn_mod,
        "get_bids_data_folder",
        lambda: bids_root,
    )
    monkeypatch.setattr(
        fconn_mod,
        "_find_surface_mesh_paths",
        lambda preproc_root, subject, space: {
            "L": preproc_root / f"sub-{subject}" / "anat" / f"sub-{subject}_hemi-L_space-{space}_midthickness.surf.gii",
            "R": preproc_root / f"sub-{subject}" / "anat" / f"sub-{subject}_hemi-R_space-{space}_midthickness.surf.gii",
        },
    )

    run_records = fconn_mod.FunctionalConnectivityEstimator._find_preprocessed_surface_runs(
        subject=subject,
        task="LANGUAGE",
        session=None,
        space="fsLR32k",
    )

    assert len(run_records) == 2
    assert [record["session_label"] for record in run_records] == ["01", "02"]
