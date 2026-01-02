import numpy as np
import pytest

import funROI.analysis.effect as effect_mod
from funROI.analysis.tests.utils import DummyFROI

def test_effect_run_validates_shapes():
    eff = np.zeros((2, 5))
    froi = np.zeros((5,))  # wrong dim

    with pytest.raises(ValueError, match=r"froi_masks should have shape"):
        effect_mod.EffectEstimator._run(eff, froi, fill_na_with_zero=True)

    with pytest.raises(ValueError, match=r"effect_data should have shape"):
        effect_mod.EffectEstimator._run(np.zeros((5,)), np.zeros((1, 5)), True)

    with pytest.raises(ValueError, match=r"should have the same shape"):
        effect_mod.EffectEstimator._run(np.zeros((2, 5)), np.zeros((2, 6)), True)


def test_effect_run_computes_detail_and_summary_fill_na_with_zero_true():
    # 2 runs, 6 voxels
    # froi labels: [1,1,2,2,0,0] (same for both runs)
    froi = np.array(
        [
            [1, 1, 2, 2, 0, 0],
            [1, 1, 2, 2, 0, 0],
        ],
        dtype=float,
    )

    eff = np.array(
        [
            [1.0, np.nan, 3.0, 5.0, 9.0, 10.0],
            [2.0, 4.0,   np.nan, 6.0, 9.0, 10.0],
        ],
        dtype=float,
    )

    # non_nan_voxels keeps voxels where froi != 0 in any run => first 4
    # fill_na_with_zero: eff nan -> 0
    # label1 voxels (0,1): run0 mean = mean([1,0]) = 0.5; run1 mean = mean([2,4]) = 3
    # label2 voxels (2,3): run0 mean = mean([3,5]) = 4; run1 mean = mean([0,6]) = 3
    df_sum, df_det = effect_mod.EffectEstimator._run(eff, froi, fill_na_with_zero=True)

    # detail: 2 labels * 2 runs = 4 rows
    assert df_det.shape[0] == 4
    assert set(df_det.columns) == {"froi", "run", "n_voxels", "size"}

    # expected sizes per label/run
    # df_det has rows ordered by repeat(labels) then tile(runs)
    # but label ordering comes from np.unique(froi_masks) => [0,1,2] typically, then filtered.
    # We assert by grouping.
    got = df_det.set_index(["froi", "run"])["size"].to_dict()
    assert got[(1.0, 0)] == pytest.approx(0.5)
    assert got[(1.0, 1)] == pytest.approx(3.0)
    assert got[(2.0, 0)] == pytest.approx(4.0)
    assert got[(2.0, 1)] == pytest.approx(3.0)

    # summary mean across runs
    s = df_sum.set_index("froi")["size"].to_dict()
    assert s[1.0] == pytest.approx((0.5 + 3.0) / 2)
    assert s[2.0] == pytest.approx((4.0 + 3.0) / 2)


def test_effect_run_fill_na_with_zero_false_ignores_nan():
    froi = np.array([[1, 1, 0], [1, 1, 0]], dtype=float)
    eff = np.array([[1.0, np.nan, 9.0], [2.0, 4.0, 9.0]], dtype=float)

    # Keep first 2 voxels; do NOT fill nan -> mean ignores nan:
    # run0 label1 mean = mean([1]) = 1
    # run1 label1 mean = mean([2,4]) = 3
    df_sum, df_det = effect_mod.EffectEstimator._run(eff, froi, fill_na_with_zero=False)
    got = df_det.set_index(["froi", "run"])["size"].to_dict()
    assert got[(1.0, 0)] == pytest.approx(1.0)
    assert got[(1.0, 1)] == pytest.approx(3.0)


def test_effect_estimator_run_orthogonal_path(monkeypatch):
    """
    okorth True => uses _get_contrast_data(..., run='all') and _get_froi_data(..., run='all')
    """
    subjects = ["S1"]
    froi = DummyFROI()

    # parcels labels mapping
    monkeypatch.setattr(effect_mod, "get_parcels", lambda parcels: (None, {1.0: "L1", 2.0: "L2"}))

    # all effects orthogonal
    monkeypatch.setattr(effect_mod, "_check_orthogonal", lambda *a, **k: True)

    # froi_all exists: 1 run-label "all" (EffectEstimator makes it [None,:] later)
    froi_mask = np.array([1, 1, 2, 2], dtype=float)
    monkeypatch.setattr(effect_mod, "_get_froi_data", lambda subject, config, run_label: froi_mask)

    # effect data for 'all'
    eff = np.array([10.0, 20.0, 30.0, 50.0], dtype=float)
    monkeypatch.setattr(effect_mod, "_get_contrast_data", lambda subject, task, run_label, contrast, typ: eff)

    # should not be called in orthogonal path
    monkeypatch.setattr(effect_mod, "_get_orthogonalized_contrast_data", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")))
    monkeypatch.setattr(effect_mod, "_get_orthogonalized_froi_data", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")))

    # patch _save
    monkeypatch.setattr(effect_mod.EffectEstimator, "_save", lambda self, info: None)

    est = effect_mod.EffectEstimator(subjects=subjects, froi=froi, fill_na_with_zero=True)
    summary, detail = est.run(task="TASKX", effects=["eff1"])

    # label mapping applied
    assert set(summary["froi"]) == {"L1", "L2"}
    assert set(detail["froi"]) == {"L1", "L2"}

    # effect column populated
    assert summary["effect"].unique().tolist() == ["eff1"]
    assert detail["effect"].unique().tolist() == ["eff1"]

    # run labels resolved (only one run index 0 -> "all")
    assert set(detail["effect_run"]) == {"all"}
    assert set(detail["froi_run"]) == {"all"}

    # sizes:
    # L1 mean([10,20])=15; L2 mean([30,50])=40
    s = summary.set_index("froi")["size"].to_dict()
    assert s["L1"] == pytest.approx(15.0)
    assert s["L2"] == pytest.approx(40.0)


def test_effect_estimator_run_non_orthogonal_path(monkeypatch):
    """
    okorth False => uses orthogonalized contrast + froi data, with run label mapping.
    """
    subjects = ["S1"]
    froi = DummyFROI()

    monkeypatch.setattr(effect_mod, "get_parcels", lambda parcels: (None, {1.0: "L1"}))
    monkeypatch.setattr(effect_mod, "_check_orthogonal", lambda *a, **k: False)

    # froi_all still required to decide skip; can be minimal non-None
    monkeypatch.setattr(effect_mod, "_get_froi_data", lambda subject, config, run_label: np.array([1, 0, 0, 0], dtype=float))

    # orthogonalized froi group=1: returns (n_runs=2, n_voxels=4), labels
    froi_orth = np.array(
        [
            [1, 0, 0, 0],  # run label "odd"
            [1, 0, 0, 0],  # run label "even"
        ],
        dtype=float,
    )
    monkeypatch.setattr(effect_mod, "_get_orthogonalized_froi_data", lambda *a, **k: (froi_orth, ["odd", "even"]))

    # orthogonalized contrast group=2: effect data and labels
    eff_orth = np.array(
        [
            [10, 20, 30, 40],  # run label "even"
            [1, 2, 3, 4],      # run label "odd"
        ],
        dtype=float,
    )
    monkeypatch.setattr(effect_mod, "_get_orthogonalized_contrast_data", lambda *a, **k: (eff_orth, ["even", "odd"]))

    monkeypatch.setattr(effect_mod.EffectEstimator, "_save", lambda self, info: None)

    est = effect_mod.EffectEstimator(subjects=subjects, froi=froi, fill_na_with_zero=True, orthogonalization="odd-even")
    summary, detail = est.run(task="TASKX", effects=["eff1"])

    # run mapping: df_detail has "run" indices 0/1 which map to labels lists
    assert set(detail["effect_run"]) == {"even", "odd"}
    assert set(detail["froi_run"]) == {"odd", "even"}

    # Only label 1 exists and occupies voxel0 -> size equals effect voxel0 per run
    # run index 0 => effect_run "even" => voxel0 = 10
    # run index 1 => effect_run "odd"  => voxel0 = 1
    got = detail.set_index(["effect_run"])["size"].to_dict()
    assert got["even"] == pytest.approx(10.0)
    assert got["odd"] == pytest.approx(1.0)

    # summary is mean across runs: (10 + 1)/2
    assert summary["size"].iloc[0] == pytest.approx(5.5)


def test_effect_estimator_run_customized_runs_requires_both_labels(monkeypatch):
    froi = DummyFROI()
    monkeypatch.setattr(effect_mod, "get_parcels", lambda parcels: (None, {1.0: "L1"}))

    est = effect_mod.EffectEstimator(subjects=["S1"], froi=froi)

    with pytest.raises(ValueError, match="must both be specified"):
        est.run(task="TASKX", effects=["eff1"], effect_run_label="01")

    est2 = effect_mod.EffectEstimator(subjects=["S1"], froi=froi, froi_run_label="all")
    with pytest.raises(ValueError, match="must both be specified"):
        est2.run(task="TASKX", effects=["eff1"], effect_run_label=None)
