from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from nibabel.nifti1 import Nifti1Image

import funROI
import funROI.froi as froi_mod
from funROI.parcels import ParcelsConfig


@pytest.fixture(autouse=True)
def _reset_and_set_deriv(tmp_path):
    funROI.reset_settings()
    funROI.set_bids_deriv_folder(tmp_path / "derivatives")
    yield
    funROI.reset_settings()


def _nii(path: Path, data: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4))
    img.to_filename(path)
    return path


def test_get_froi_path_creates_and_reuses_and_increments_id(tmp_path):
    subject = "100307"
    task = "LANGUAGE"
    run = "01"

    # parcels file must exist for ParcelsConfig (used in froi info matching)
    parcels_path = _nii(tmp_path / "parcels.nii.gz", np.zeros((2, 2, 2)))

    cfg1 = froi_mod.FROIConfig(
        task=task,
        contrasts=["c1"],
        threshold_type="none",
        threshold_value=0.05,
        parcels=ParcelsConfig(parcels_path),
        conjunction_type=None,
    )
    cfg2 = froi_mod.FROIConfig(
        task=task,
        contrasts=["c2"],  # different => should create new id
        threshold_type="none",
        threshold_value=0.05,
        parcels=ParcelsConfig(parcels_path),
        conjunction_type=None,
    )

    # Create first entry (id 0000)
    p1 = froi_mod._get_froi_path(subject, run, cfg1, create=True)
    assert p1.name.endswith("_froi-0000_mask.nii.gz")

    info_path = froi_mod._get_froi_info_path(subject, task)
    assert info_path.exists()
    df = pd.read_csv(info_path)
    assert df.shape[0] == 1
    assert int(df.loc[0, "id"]) == 0

    # Same config should reuse id 0000 (no new row)
    p1b = froi_mod._get_froi_path(subject, run, cfg1, create=True)
    assert p1b.name.endswith("_froi-0000_mask.nii.gz")
    df2 = pd.read_csv(info_path)
    assert df2.shape[0] == 1

    # Different config should create id 0001
    p2 = froi_mod._get_froi_path(subject, run, cfg2, create=True)
    assert p2.name.endswith("_froi-0001_mask.nii.gz")
    df3 = pd.read_csv(info_path)
    assert df3.shape[0] == 2
    assert set(df3["id"].astype(int).tolist()) == {0, 1}


def test_create_froi_with_real_parcels_labels_produces_labeled_mask(tmp_path, monkeypatch):
    """
    Branch: parcels exist -> froi mask is integer-labeled by parcels.
    """
    subject = "100307"
    task = "LANGUAGE"
    run = "01"
    contrast = "c1"

    # parcels image with labels: 1 occupies first half, 2 occupies second half
    parcels_data = np.zeros((2, 2, 2), dtype=float)
    parcels_data.flat[:4] = 1
    parcels_data.flat[4:] = 2
    parcels_path = _nii(tmp_path / "parcels.nii.gz", parcels_data)

    # label dict only includes labels 1 and 2
    labels_json = tmp_path / "parcels.json"
    labels_json.write_text('{"1": "A", "2": "B"}')

    cfg = froi_mod.FROIConfig(
        task=task,
        contrasts=[contrast],
        threshold_type="none",
        threshold_value=0.05,
        parcels=ParcelsConfig(parcels_path, labels_json),
        conjunction_type=None,
    )

    # Tell froi code what runs exist
    monkeypatch.setattr(froi_mod, "_get_contrast_runs", lambda sub, task, con: [run])

    # Provide p-values: label 1 significant, label 2 not
    pvals = np.ones(parcels_data.size, dtype=float) * 0.5
    pvals[parcels_data.flatten() == 1] = 0.001

    monkeypatch.setattr(
        froi_mod,
        "_get_contrast_data",
        lambda sub, task_, run_, con, suf: pvals if suf == "p" else None,
    )

    # Create froi; it should save under bids-derivatives
    mask = froi_mod._create_froi(subject, cfg, run, return_nifti=False)
    assert mask is not None
    # In label-1 voxels -> value 1, in label-2 voxels -> 0
    assert set(np.unique(mask)) <= {0.0, 1.0, 2.0}
    assert np.all(mask[parcels_data.flatten() == 1] == 1)
    assert np.all(mask[parcels_data.flatten() == 2] == 0)

    # File exists
    froi_path = froi_mod._get_froi_path(subject, run, cfg)
    assert froi_path.exists()


def test_create_froi_without_parcels_uses_contrast_as_reference_and_binary_mask(tmp_path, monkeypatch):
    """
    Branch: parcels missing -> parcels_ref is loaded from contrast path; mask is boolean (0/1).
    """
    subject = "100307"
    task = "LANGUAGE"
    run = "01"
    contrast = "c1"

    # parcels path does not exist -> get_parcels returns (None, None)
    missing_parcels = tmp_path / "does_not_exist.nii.gz"
    cfg = froi_mod.FROIConfig(
        task=task,
        contrasts=[contrast],
        threshold_type="none",
        threshold_value=0.05,
        parcels=ParcelsConfig(missing_parcels),
        conjunction_type=None,
    )

    monkeypatch.setattr(froi_mod, "_get_contrast_runs", lambda sub, task_, con: [run])

    # Create a contrast p-map nifti to serve as parcels_ref
    # shape must match expected mask reshape
    ref_data = np.zeros((2, 2, 2), dtype=float)
    ref_path = _nii(tmp_path / "ref_p.nii.gz", ref_data)

    monkeypatch.setattr(
        froi_mod,
        "_get_contrast_path",
        lambda sub, task_, run_, con, suf: ref_path,
    )

    # p-values for 8 voxels: first 3 significant
    pvals = np.array([0.001, 0.002, 0.003, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=float)
    monkeypatch.setattr(
        froi_mod,
        "_get_contrast_data",
        lambda sub, task_, run_, con, suf: pvals if suf == "p" else None,
    )

    mask = froi_mod._create_froi(subject, cfg, run, return_nifti=False)
    assert mask is not None
    assert set(np.unique(mask)) <= {0.0, 1.0}
    assert mask[:3].sum() == 3
    assert mask[3:].sum() == 0

    froi_path = froi_mod._get_froi_path(subject, run, cfg)
    assert froi_path.exists()


def _basic_cfg(tmp_path, task="LANGUAGE"):
    parcels_path = _nii(tmp_path / "parcels.nii.gz", np.zeros((2, 2, 2)))
    return froi_mod.FROIConfig(
        task=task,
        contrasts=["c1"],
        threshold_type="none",
        threshold_value=0.05,
        parcels=ParcelsConfig(parcels_path),
        conjunction_type=None,
    )


def test_get_orthogonalized_froi_data_returns_none_when_no_runs(tmp_path, monkeypatch):
    cfg = _basic_cfg(tmp_path)

    monkeypatch.setattr(froi_mod, "_get_froi_runs", lambda subject, config: [])
    out = froi_mod._get_orthogonalized_froi_data("100307", cfg, group=1, orthogonalization="all-but-one")
    assert out == (None, None)


def test_get_orthogonalized_froi_data_creates_missing_and_stacks(tmp_path, monkeypatch):
    cfg = _basic_cfg(tmp_path)

    # Pretend we have two runs overall
    monkeypatch.setattr(froi_mod, "_get_froi_runs", lambda subject, config: ["01", "02"])

    # Deterministic labels returned by orthogonalization helper
    labels = ["orth01", "orth02"]
    monkeypatch.setattr(
        froi_mod,
        "_get_orthogonalized_run_labels",
        lambda runs, group, orth: labels,
    )

    # First label missing -> _get_froi_data returns None -> should call _create_froi
    a = np.arange(8, dtype=float)  # 8 voxels
    b = np.ones(8, dtype=float)

    calls = {"create": 0}

    def fake_get(subject, config, label, return_nifti=False):
        if label == "orth01":
            return None
        return b

    def fake_create(subject, config, label, return_nifti=False):
        calls["create"] += 1
        assert label == "orth01"
        return a

    monkeypatch.setattr(froi_mod, "_get_froi_data", fake_get)
    monkeypatch.setattr(froi_mod, "_create_froi", fake_create)

    data, out_labels = froi_mod._get_orthogonalized_froi_data("100307", cfg, group=1, orthogonalization="all-but-one")

    assert out_labels == labels
    assert data.shape == (2, 8)
    assert np.allclose(data[0], a)
    assert np.allclose(data[1], b)
    assert calls["create"] == 1


def test_get_orthogonalized_froi_data_returns_none_if_create_fails(tmp_path, monkeypatch):
    cfg = _basic_cfg(tmp_path)

    monkeypatch.setattr(froi_mod, "_get_froi_runs", lambda subject, config: ["01"])
    monkeypatch.setattr(
        froi_mod, "_get_orthogonalized_run_labels", lambda runs, group, orth: ["orth01"]
    )

    monkeypatch.setattr(froi_mod, "_get_froi_data", lambda *a, **k: None)
    monkeypatch.setattr(froi_mod, "_create_froi", lambda *a, **k: None)

    assert froi_mod._get_orthogonalized_froi_data("100307", cfg, group=1, orthogonalization="odd-even") == (None, None)


def test_get_froi_data_loads_existing_mask(tmp_path, monkeypatch):
    # configure deriv folder so _get_froi_path writes under tmp
    funROI.reset_settings()
    funROI.set_bids_deriv_folder(tmp_path / "derivatives")

    cfg = _basic_cfg(tmp_path)
    subject, run = "100307", "01"

    # Create the fROI file at the path that _get_froi_path expects (create=True writes registry)
    froi_path = froi_mod._get_froi_path(subject, run, cfg, create=True)
    data = np.zeros((2, 2, 2), dtype=float)
    data.flat[:3] = 1
    _nii(froi_path, data)

    # Should load and flatten
    out = froi_mod._get_froi_data(subject, cfg, run, return_nifti=False)
    assert out.shape == (8,)
    assert out.sum() == 3

    # Should return nifti if requested
    img = froi_mod._get_froi_data(subject, cfg, run, return_nifti=True)
    assert img.shape == (2, 2, 2)

    funROI.reset_settings()


def test_get_froi_data_creates_when_missing(tmp_path, monkeypatch):
    funROI.reset_settings()
    funROI.set_bids_deriv_folder(tmp_path / "derivatives")

    cfg = _basic_cfg(tmp_path)
    subject, run = "100307", "01"

    # Force path to be "missing" by pointing it somewhere without writing a file
    missing_path = tmp_path / "derivatives" / "missing.nii.gz"
    monkeypatch.setattr(froi_mod, "_get_froi_path", lambda *a, **k: missing_path)

    created = np.arange(8, dtype=float)

    monkeypatch.setattr(froi_mod, "_create_froi", lambda *a, **k: created)

    out = froi_mod._get_froi_data(subject, cfg, run, return_nifti=False)
    assert np.allclose(out, created)

    funROI.reset_settings()


def test_threshold_p_map_none_threshold(tmp_path):
    # data shape: (n_runs, n_voxels)
    data = np.array(
        [
            [0.01, 0.20, 0.03, 0.50],
            [0.02, 0.10, 0.04, 0.60],
        ],
        dtype=float,
    )
    mask = froi_mod._threshold_p_map(data, threshold_type="none", threshold_value=0.05)
    assert mask.shape == data.shape
    assert np.array_equal(mask, (data < 0.05).astype(float))


def test_threshold_p_map_bonferroni(tmp_path):
    data = np.array(
        [
            [0.01, 0.20, 0.03, 0.50],
            [0.02, 0.10, 0.04, 0.60],
        ],
        dtype=float,
    )
    mask = froi_mod._threshold_p_map(data, threshold_type="bonferroni", threshold_value=0.05)
    expected = (data < (0.05 / 4)).astype(float)
    assert np.array_equal(mask, expected)


def test_threshold_p_map_n_selects_top_n_per_voxel_across_runs(tmp_path):
    # For each voxel, choose the best (smallest) p-value across runs when n=1
    data = np.array(
        [
            [0.30, 0.01, 0.20],
            [0.10, 0.02, 0.15],
            [0.03, 0.04, 0.25],
        ],
        dtype=float,
    )  # shape (3 runs, 3 voxels)

    mask = froi_mod._threshold_p_map(data, threshold_type="n", threshold_value=1)
    assert mask.shape == data.shape

    expected = np.zeros_like(data)
    expected[0, 1] = 1
    expected[1, 1] = 1
    expected[2, 0] = 1
    assert np.array_equal(mask, expected)

def test_create_p_map_mask_conjunction_min():
    """
    conjunction_type='min':
    combine across contrasts using min, then threshold.
    """
    # data shape: (n_contrast, n_runs, n_voxels)
    data = np.array(
        [
            # contrast 1
            [
                [0.01, 0.20, 0.03],  # run 1
                [0.02, 0.10, 0.04],  # run 2
            ],
            # contrast 2
            [
                [0.05, 0.30, 0.02],  # run 1
                [0.01, 0.40, 0.06],  # run 2
            ],
        ],
        dtype=float,
    )

    mask = froi_mod._create_p_map_mask(
        data,
        conjunction_type="min",
        threshold_type="none",
        threshold_value=0.05,
    )

    expected = np.array(
        [
            [1, 0, 1],  # run1
            [1, 0, 1],  # run2
        ],
        dtype=float,
    )

    assert mask.shape == (2, 3)
    assert np.array_equal(mask, expected)


def test_create_p_map_mask_conjunction_and():
    """
    conjunction_type='and':
    voxel survives only if it passes threshold for ALL contrasts.
    """
    data = np.array(
        [
            # contrast 1
            [
                [0.01, 0.20, 0.03],  # run 1
                [0.02, 0.10, 0.04],  # run 2
            ],
            # contrast 2
            [
                [0.02, 0.30, 0.01],  # run 1
                [0.04, 0.40, 0.02],  # run 2
            ],
        ],
        dtype=float,
    )

    mask = froi_mod._create_p_map_mask(
        data,
        conjunction_type="and",
        threshold_type="none",
        threshold_value=0.05,
    )

    expected = np.array(
        [
            [1, 0, 1],
            [1, 0, 1],
        ],
        dtype=float,
    )

    assert mask.shape == (2, 3)
    assert np.array_equal(mask, expected)
