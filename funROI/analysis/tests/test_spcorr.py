import numpy as np
import pandas as pd
import pytest

import funROI.analysis.spcorr as spcorr_mod
from funROI.analysis.tests.utils import DummyFROI


def _img_from_flat(flat, shape=(2, 2, 1)):
    import nibabel as nib
    data = np.array(flat, dtype=float).reshape(shape)
    return nib.Nifti1Image(data, affine=np.eye(4))


def test_spcorr_init_parcels_missing_raises(monkeypatch):
    # parcels mode: get_parcels returns None -> should raise in __init__
    monkeypatch.setattr(spcorr_mod, "get_parcels", lambda x: (None, None))
    with pytest.raises(ValueError, match="Specified as parcels, but no parcels found"):
        spcorr_mod.SpatialCorrelationEstimator(subjects=["S1"], froi="P1")


def test_spcorr_run_parcels_requires_both_run1_run2(monkeypatch):
    # parcels mode: run1 xor run2 should raise
    monkeypatch.setattr(spcorr_mod, "get_parcels", lambda x: (_img_from_flat([1, 1, 0, 0]), {1: "A"}))
    est = spcorr_mod.SpatialCorrelationEstimator(subjects=["S1"], froi="P1")
    monkeypatch.setattr(spcorr_mod.SpatialCorrelationEstimator, "_save", lambda self, info: None)

    with pytest.raises(ValueError, match="Some but not all necessary run labels are specified"):
        est.run(task1="T1", effect1="e1", task2="T2", effect2="e2", run1="01", run2=None)

    with pytest.raises(ValueError, match="Some but not all necessary run labels are specified"):
        est.run(task1="T1", effect1="e1", task2="T2", effect2="e2", run1=None, run2="02")


def test_spcorr_run_froi_requires_all_three_runs_if_any(monkeypatch):
    # FROI mode: if any of (run_froi, run1, run2) is specified, all 3 must be.
    monkeypatch.setattr(spcorr_mod, "FROIConfig", DummyFROI, raising=False)
    monkeypatch.setattr(spcorr_mod, "get_parcels", lambda x: (_img_from_flat([1, 0, 0, 0]), {1: "L"}))
    est = spcorr_mod.SpatialCorrelationEstimator(subjects=["S1"], froi=DummyFROI())
    monkeypatch.setattr(spcorr_mod.SpatialCorrelationEstimator, "_save", lambda self, info: None)

    with pytest.raises(ValueError, match="Some but not all necessary run labels are specified"):
        est.run("T1", "e1", "T2", "e2", run_froi="all", run1=None, run2=None)

    with pytest.raises(ValueError, match="Some but not all necessary run labels are specified"):
        est.run("T1", "e1", "T2", "e2", run_froi=None, run1="01", run2="02")

    with pytest.raises(ValueError, match="Some but not all necessary run labels are specified"):
        est.run("T1", "e1", "T2", "e2", run_froi="all", run1="01", run2=None)


def test_spcorr_run_all_three_nonorthogonal_raises(monkeypatch):
    # If froi-effect1, froi-effect2, and effect1-effect2 are all non-orthogonal, must raise.
    monkeypatch.setattr(spcorr_mod, "FROIConfig", DummyFROI, raising=False)
    monkeypatch.setattr(spcorr_mod, "get_parcels", lambda x: (_img_from_flat([1, 0, 0, 0]), {1: "L"}))
    monkeypatch.setattr(spcorr_mod, "_check_orthogonal", lambda *a, **k: False)
    est = spcorr_mod.SpatialCorrelationEstimator(subjects=["S1"], froi=DummyFROI())
    monkeypatch.setattr(spcorr_mod.SpatialCorrelationEstimator, "_save", lambda self, info: None)

    with pytest.raises(ValueError, match="all non-orthogonal to each other"):
        est.run("T1", "e1", "T2", "e2")  # no run labels => auto-orth check triggers error


def test_spcorr_run_orthogonal_uses_all(monkeypatch):
    # When all orthogonal -> group_froi=group1=group2=0 => uses _get_froi_data(all) + _get_contrast_data(all)
    monkeypatch.setattr(spcorr_mod, "FROIConfig", DummyFROI, raising=False)
    monkeypatch.setattr(spcorr_mod, "get_parcels", lambda x: (_img_from_flat([1, 0, 0, 0]), {1: "L"}))
    monkeypatch.setattr(spcorr_mod, "_check_orthogonal", lambda *a, **k: True)

    calls = {"froi": [], "c": []}

    def fake_get_froi_data(subject, cfg, run_label):
        calls["froi"].append((subject, run_label))
        return np.array([1, 1, 0, 0], dtype=float)

    def fake_get_contrast_data(subject, task, run_label, contrast, typ):
        assert typ == "effect"
        calls["c"].append((subject, task, run_label, contrast))
        # Give two different patterns so correlation is non-trivial but defined.
        if contrast == "e1":
            return np.array([1, 2, 0, 0], dtype=float)
        return np.array([2, 1, 0, 0], dtype=float)

    monkeypatch.setattr(spcorr_mod, "_get_froi_data", fake_get_froi_data)
    monkeypatch.setattr(spcorr_mod, "_get_contrast_data", fake_get_contrast_data)
    monkeypatch.setattr(spcorr_mod.SpatialCorrelationEstimator, "_save", lambda self, info: None)

    est = spcorr_mod.SpatialCorrelationEstimator(subjects=["S1"], froi=DummyFROI(task="LOC", contrasts=["c1"]))
    summary, detail = est.run("T1", "e1", "T2", "e2")

    assert ("S1", "all") in calls["froi"]
    assert ("S1", "T1", "all", "e1") in calls["c"]
    assert ("S1", "T2", "all", "e2") in calls["c"]

    assert set(summary.columns) >= {"froi", "fisher_z", "subject"}
    assert set(detail.columns) >= {"froi", "fisher_z", "subject", "froi_run", "effect1_run", "effect2_run"}
    assert np.isfinite(summary["fisher_z"]).all()


def test_spcorr_run_effects_nonorthogonal_triggers_asymmetric_fix_all_but_one(monkeypatch):
    """
    Condition for asymmetric fix:
      orthogonalization == "all-but-one"
      not okorth_effects
      okorth_froi_effect1 and okorth_froi_effect2
    => should call _get_orthogonalized_contrast_data twice more with groups (2 for effect1, 1 for effect2)
    """
    monkeypatch.setattr(spcorr_mod, "FROIConfig", DummyFROI, raising=False)
    monkeypatch.setattr(spcorr_mod, "get_parcels", lambda x: (_img_from_flat([1, 0, 0, 0]), {1: "L"}))

    def fake_check_orth(subject, task_a, cons_a, task_b, cons_b):
        # Froi-task is "LOC" for DummyFROI default
        froi_task = "LOC"
        if task_b == froi_task or task_a == froi_task:
            return True  # froi orthogonal to both effects
        return False     # effects non-orthogonal to each other

    monkeypatch.setattr(spcorr_mod, "_check_orthogonal", fake_check_orth)

    # froi uses all-run data
    monkeypatch.setattr(spcorr_mod, "_get_froi_data", lambda *a, **k: np.array([1, 1, 0, 0], dtype=float))

    # base orthogonalized contrast data for group1/group2 (from _get_orthogonalized_group => 0,1,2)
    call_log = []

    def fake_get_orth_contrast(subject, task, contrast, group, typ, orth):
        call_log.append((contrast, group))
        assert typ == "effect"
        # return n_runs=2, n_vox=4
        if contrast == "e1":
            dat = np.array([[1, 2, 0, 0], [1, 2, 0, 0]], dtype=float)
            labels = ["odd", "even"] if group == 1 else ["even", "odd"]
            return dat, labels
        else:
            dat = np.array([[2, 1, 0, 0], [2, 1, 0, 0]], dtype=float)
            labels = ["even", "odd"] if group == 2 else ["odd", "even"]
            return dat, labels

    monkeypatch.setattr(spcorr_mod, "_get_orthogonalized_contrast_data", fake_get_orth_contrast)
    monkeypatch.setattr(spcorr_mod.SpatialCorrelationEstimator, "_save", lambda self, info: None)

    est = spcorr_mod.SpatialCorrelationEstimator(
        subjects=["S1"], froi=DummyFROI(task="LOC", contrasts=["c1"]), orthogonalization="all-but-one"
    )
    est.run("T1", "e1", "T2", "e2")

    # Must include the "asymmetric fix" extra calls: (e1,2) and (e2,1)
    assert ("e1", 2) in call_log
    assert ("e2", 1) in call_log


def test_spcorr__run_basic_properties():
    # Directly test _run: one label region, fisher z finite, expected columns.
    effect1 = np.array([[1, 2, 0, 0]], dtype=float)
    effect2 = np.array([[2, 1, 0, 0]], dtype=float)
    froi = np.array([[1, 1, 0, 0]], dtype=float)

    summary, detail = spcorr_mod.SpatialCorrelationEstimator._run(effect1, effect2, froi)

    assert list(summary.columns) == ["froi", "fisher_z"]
    assert list(detail.columns) == ["froi", "run", "fisher_z"]
    assert summary.shape[0] == 1
    assert np.isfinite(summary["fisher_z"].iloc[0])
