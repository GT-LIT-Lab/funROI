import numpy as np
import pandas as pd
import pytest
from nibabel.nifti1 import Nifti1Image

import funROI.analysis.overlap as overlap_mod
from funROI.analysis.tests.utils import DummyFROI

def _img_from_flat(flat: np.ndarray) -> Nifti1Image:
    # make it a small 3D image; flatten order doesn't matter because code flattens
    data = np.asarray(flat, dtype=np.float32).reshape((1, 1, -1))
    return Nifti1Image(data, np.eye(4))


def test_overlap_run_validates_shapes():
    a = np.zeros((2, 5))
    b = np.zeros((5,))
    with pytest.raises(ValueError, match="froi2_masks must have shape"):
        overlap_mod.OverlapEstimator._run(a, b, "overlap")

    with pytest.raises(ValueError, match="froi1_masks must have shape"):
        overlap_mod.OverlapEstimator._run(np.zeros((5,)), np.zeros((1, 5)), "overlap")

    with pytest.raises(ValueError, match="must have the same shape"):
        overlap_mod.OverlapEstimator._run(np.zeros((1, 5)), np.zeros((2, 5)), "overlap")


def test_overlap_run_overlap_and_dice_values():
    # 1 run, 6 voxels
    # froi1 labels: 1 occupies vox 0,1,2; 2 occupies vox 3
    # froi2 labels: 10 occupies vox 1,2,3; 20 occupies vox 4
    froi1 = np.array([[1, 1, 1, 2, 0, 0]], dtype=float)
    froi2 = np.array([[0,10,10,10,20, 0]], dtype=float)

    # For (1,10): intersection = vox 1,2 => 2
    # sizes: froi1 label1 size=3, froi2 label10 size=3
    # overlap (min denom) = 2/min(3,3)=2/3
    # dice = 2*2/(3+3)=4/6=2/3
    # For (2,10): intersection = vox 3 => 1; sizes 1 and 3
    # overlap = 1/min(1,3)=1
    # dice = 2/(1+3)=0.5
    df_sum_o, df_det_o = overlap_mod.OverlapEstimator._run(froi1, froi2, "overlap")
    df_sum_d, df_det_d = overlap_mod.OverlapEstimator._run(froi1, froi2, "dice")

    # check expected pairs exist
    s_o = df_sum_o.set_index(["froi1", "froi2"])["overlap"].to_dict()
    s_d = df_sum_d.set_index(["froi1", "froi2"])["overlap"].to_dict()

    assert s_o[(1.0, 10.0)] == pytest.approx(2/3)
    assert s_d[(1.0, 10.0)] == pytest.approx(2/3)

    assert s_o[(2.0, 10.0)] == pytest.approx(1.0)
    assert s_d[(2.0, 10.0)] == pytest.approx(0.5)

    # detail has 1 run, 2x2 label pairs => 4 rows
    assert df_det_o.shape[0] == 4
    assert set(df_det_o.columns) == {"run", "froi1", "froi2", "overlap"}


def test_overlap_estimator_run_parcels_vs_parcels(monkeypatch):
    est = overlap_mod.OverlapEstimator(kind="overlap")
    monkeypatch.setattr(overlap_mod.OverlapEstimator, "_save", lambda self, info: None)

    # parcels labels mapping
    parcels1 = _img_from_flat(np.array([1,1,0,0], dtype=float))
    parcels2 = _img_from_flat(np.array([2,0,2,0], dtype=float))

    def fake_get_parcels(arg):
        if arg == "P1":
            return parcels1, {1: "A"}
        if arg == "P2":
            return parcels2, {2: "B"}
        raise AssertionError("unexpected parcels key")

    monkeypatch.setattr(overlap_mod, "get_parcels", fake_get_parcels)

    summary, detail = est.run("P1", "P2")

    # should rename froi columns to parcel columns
    assert "parcel1" in summary.columns and "parcel2" in summary.columns
    assert "parcel1" in detail.columns and "parcel2" in detail.columns

    # run labels for parcels-parcels not added (code adds run1/run2 only if not both parcels)
    assert "run1" not in detail.columns
    assert "run2" not in detail.columns

    # check mapping applied
    assert set(summary["parcel1"]) == {"A"}
    assert set(summary["parcel2"]) == {"B"}


def test_overlap_estimator_run_parcels_vs_froi_requires_subject(monkeypatch):
    est = overlap_mod.OverlapEstimator(kind="overlap")
    monkeypatch.setattr(overlap_mod.OverlapEstimator, "_save", lambda self, info: None)

    # CRITICAL: make isinstance(DummyFROI(), FROIConfig) True inside overlap_mod
    monkeypatch.setattr(overlap_mod, "FROIConfig", DummyFROI, raising=False)

    parcels_img = _img_from_flat(np.array([1, 1, 0, 0], dtype=float))

    def fake_get_parcels(x):
        if x == "P1":
            return parcels_img, {1: "A"}
        if x == "dummy_parcels":
            # overlap.py calls get_parcels() even for FROI just to fetch labels;
            # img can be None and that's fine as long as it's not treated as parcels.
            return None, {1: "L"}
        raise AssertionError(f"unexpected get_parcels arg: {x}")

    monkeypatch.setattr(overlap_mod, "get_parcels", fake_get_parcels)

    froi2 = DummyFROI()

    with pytest.raises(ValueError, match="Subject label 2 is required for fROIs"):
        est.run("P1", froi2, subject2=None)

    monkeypatch.setattr(
        overlap_mod, "_get_froi_data",
        lambda subject, cfg, run: np.array([1, 0, 1, 0], dtype=float),
    )

    summary, detail = est.run("P1", froi2, subject2="S2", run2="all")

    assert "parcel1" in summary.columns
    assert "froi2" in summary.columns
    assert set(detail["run1"]) == {"parcels"}
    assert set(detail["run2"]) == {"all"}


def test_overlap_estimator_run_froi_vs_froi_orthogonal_uses_all(monkeypatch):
    est = overlap_mod.OverlapEstimator(kind="dice")
    monkeypatch.setattr(overlap_mod.OverlapEstimator, "_save", lambda self, info: None)

    monkeypatch.setattr(overlap_mod, "FROIConfig", DummyFROI, raising=False)

    froi1 = DummyFROI(task="T", contrasts=["a"], parcels="parc1")
    froi2 = DummyFROI(task="T", contrasts=["b"], parcels="parc2")

    def fake_get_parcels(p):
        # called with froi.parcels, so "parc1"/"parc2"
        return _img_from_flat(np.array([1, 0, 0, 0], dtype=float)), {1: "L"}

    monkeypatch.setattr(overlap_mod, "get_parcels", fake_get_parcels)

    monkeypatch.setattr(overlap_mod, "_check_orthogonal", lambda *a, **k: True)

    def fake_get_froi_data(subject, cfg, run):
        assert run == "all"
        return (
            np.array([1, 1, 0, 0], dtype=float)
            if subject == "S1"
            else np.array([1, 0, 1, 0], dtype=float)
        )

    monkeypatch.setattr(overlap_mod, "_get_froi_data", fake_get_froi_data)
    monkeypatch.setattr(
        overlap_mod,
        "_get_orthogonalized_froi_data",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    summary, detail = est.run(froi1, froi2, subject1="S1", subject2="S2")

    assert "subject1" in summary.columns and "subject2" in summary.columns
    assert set(detail["run1"]) == {"all"}
    assert set(detail["run2"]) == {"all"}


def test_overlap_estimator_run_froi_vs_froi_nonorthogonal_uses_orthogonalized(monkeypatch):
    est = overlap_mod.OverlapEstimator(kind="overlap", orthogonalization="odd-even")
    monkeypatch.setattr(overlap_mod.OverlapEstimator, "_save", lambda self, info: None)

    # Treat DummyFROI as FROIConfig inside overlap_mod
    monkeypatch.setattr(overlap_mod, "FROIConfig", DummyFROI, raising=False)

    froi1 = DummyFROI(task="T", contrasts=["a"], parcels="parc1")
    froi2 = DummyFROI(task="T", contrasts=["b"], parcels="parc2")

    # Labels preload (not important here, just needed)
    monkeypatch.setattr(
        overlap_mod,
        "get_parcels",
        lambda p: (_img_from_flat(np.array([1, 0, 0, 0], dtype=float)), {1: "L"}),
    )

    # Same subject -> okorth depends on _check_orthogonal
    monkeypatch.setattr(overlap_mod, "_check_orthogonal", lambda *a, **k: False)

    # Must NOT be used in non-orthogonal branch
    monkeypatch.setattr(
        overlap_mod,
        "_get_froi_data",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    froi1_data = np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=float)
    froi2_data = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    def fake_get_orth(subject, cfg, group, orth):
        assert subject == "S1"
        assert orth == "odd-even"
        if group == 1:
            return froi1_data, ["odd", "even"]
        else:
            return froi2_data, ["even", "odd"]

    monkeypatch.setattr(overlap_mod, "_get_orthogonalized_froi_data", fake_get_orth)

    summary, detail = est.run(froi1, froi2, subject1="S1", subject2="S1")

    assert set(detail["run1"]) == {"odd", "even"}
    assert set(detail["run2"]) == {"even", "odd"}


def test_overlap_estimator_run_custom_runs(monkeypatch):
    est = overlap_mod.OverlapEstimator(kind="overlap")
    monkeypatch.setattr(overlap_mod.OverlapEstimator, "_save", lambda self, info: None)

    monkeypatch.setattr(overlap_mod, "FROIConfig", DummyFROI, raising=False)

    froi1 = DummyFROI(task="T", contrasts=["a"], parcels="parc1")
    froi2 = DummyFROI(task="T", contrasts=["b"], parcels="parc2")

    monkeypatch.setattr(
        overlap_mod, "get_parcels",
        lambda p: (_img_from_flat(np.array([1, 0, 0, 0], dtype=float)), {1: "L"}),
    )

    calls = []

    def fake_get_froi_data(subject, cfg, run):
        calls.append((subject, run))
        return np.array([1, 0, 0, 0], dtype=float)

    monkeypatch.setattr(overlap_mod, "_get_froi_data", fake_get_froi_data)

    summary, detail = est.run(
        froi1, froi2,
        subject1="S1", subject2="S2",
        run1="01", run2="02",
    )

    assert ("S1", "01") in calls
    assert ("S2", "02") in calls
    assert set(detail["run1"]) == {"01"}
    assert set(detail["run2"]) == {"02"}
