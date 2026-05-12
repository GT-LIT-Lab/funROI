import numpy as np
import pandas as pd
import pytest
from nibabel.nifti1 import Nifti1Image

import funROI.analysis.fconn as fconn_mod
from funROI.analysis.tests.utils import DummyFROI


def _img_from_flat(flat: np.ndarray) -> Nifti1Image:
    data = np.asarray(flat, dtype=np.float32).reshape((1, 1, -1))
    return Nifti1Image(data, np.eye(4))


def _bold_img(time_by_voxel: np.ndarray) -> Nifti1Image:
    data = np.asarray(time_by_voxel, dtype=np.float32).T.reshape((1, 1, -1, time_by_voxel.shape[0]))
    return Nifti1Image(data, np.eye(4))


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
    est = fconn_mod.FunctionalConnectivityEstimator()

    with pytest.raises(ValueError, match="within a single subject"):
        est.run("P1", "P2", subject1="S1", subject2="S2", task="TASK")


def test_fconn_run_parcels_and_froi(monkeypatch):
    est = fconn_mod.FunctionalConnectivityEstimator()
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
        )),
    )

    summary, detail = est.run(
        "P1", DummyFROI(task="TASK"), subject1="S1", task="TASK", run2="all"
    )

    assert "parcel1" in summary.columns
    assert "froi2" in summary.columns
    assert "bold_run" in detail.columns
    assert set(summary["parcel1"]) == {"ParcelA"}
    assert set(summary["froi2"]) == {"ROI"}
    assert set(detail["subject"]) == {"S1"}


def test_fconn_run_omits_froi_column_when_parcelless(monkeypatch):
    est = fconn_mod.FunctionalConnectivityEstimator()
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
        )),
    )

    summary, detail = est.run(
        DummyFROI(task="TASK", parcels="none"),
        DummyFROI(task="TASK", parcels="none"),
        subject1="S1",
    )

    assert "froi1" not in summary.columns
    assert "froi2" not in summary.columns
    assert "froi1" not in detail.columns
    assert "froi2" not in detail.columns


def test_fconn_run_passes_cleaning_overrides(monkeypatch):
    est = fconn_mod.FunctionalConnectivityEstimator()
    monkeypatch.setattr(
        fconn_mod.FunctionalConnectivityEstimator,
        "_save",
        lambda self, info: None,
    )
    monkeypatch.setattr(fconn_mod, "get_parcels", lambda arg: (_img_from_flat(np.array([1.0, 0.0])), {1.0: "A"}))

    captured = {}

    def fake_get_cleaned(subject, task, session, space, config):
        captured.update(config)
        return ([_bold_img(np.array([[1.0, 2.0], [2.0, 3.0]]))], ["01"])

    monkeypatch.setattr(
        fconn_mod.FunctionalConnectivityEstimator,
        "_get_cleaned_imgs_by_run",
        staticmethod(fake_get_cleaned),
    )

    est.run(
        "P1",
        "P1",
        subject1="S1",
        task="TASK",
        volume_fwhm=6,
        low_pass=0.2,
        regress_out_task=False,
    )

    assert captured["volume_fwhm"] == 6
    assert captured["low_pass"] == 0.2
    assert captured["regress_out_task"] is False


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
