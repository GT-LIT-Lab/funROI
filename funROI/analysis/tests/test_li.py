import numpy as np
import pandas as pd
import nibabel as nib

import funROI.analysis.li as li_mod


def _make_img_with_affine(data: np.ndarray, affine: np.ndarray) -> nib.Nifti1Image:
    return nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)


def test_li_run_counts_left_right_and_li():
    # 1D along x, so it's easy to reason.
    # We'll make 5 voxels along x: indices 0..4
    data = np.zeros((5, 1, 1), dtype=float)
    # Activate voxels at x=1,2,3 (three active)
    data[1, 0, 0] = 1
    data[2, 0, 0] = 1
    data[3, 0, 0] = 1

    # Affine: world_x = voxel_x - 2
    # So world_x for x=1 -> -1 (left), x=2 -> 0 (neither), x=3 -> +1 (right)
    affine = np.array(
        [
            [1, 0, 0, -2],
            [0, 1, 0,  0],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ],
        dtype=float,
    )
    img = _make_img_with_affine(data, affine)

    n_left, n_right, li = li_mod.LateralityIndexAnalyzer._run(img)

    assert n_left == 1
    assert n_right == 1
    assert li == 0.0  # (1-1)/(1+1)


def test_li_run_nan_when_no_nonzero_voxels():
    data = np.zeros((3, 3, 3), dtype=float)
    affine = np.eye(4)
    img = _make_img_with_affine(data, affine)

    n_left, n_right, li = li_mod.LateralityIndexAnalyzer._run(img)

    assert n_left == 0
    assert n_right == 0
    assert np.isnan(li)


def test_analyzer_run_skips_missing_subjects_and_saves_when_requested(monkeypatch):
    # Make a tiny image with one left and one right voxel in world coords
    data = np.zeros((5, 1, 1), dtype=float)
    data[1, 0, 0] = 1  # world_x=-1
    data[3, 0, 0] = 1  # world_x=+1
    affine = np.array(
        [
            [1, 0, 0, -2],
            [0, 1, 0,  0],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ],
        dtype=float,
    )
    img = _make_img_with_affine(data, affine)

    # Patch _get_froi_data: one subject returns img, another returns None
    def fake_get_froi_data(subject, config, return_nifti, run_label):
        assert return_nifti is True
        assert run_label == "all"
        if subject == "S1":
            return img
        return None

    monkeypatch.setattr(li_mod, "_get_froi_data", fake_get_froi_data)

    # Patch _save to observe calls without writing files
    saved = {"called": 0, "arg": None}

    def fake_save(self, info_df):
        saved["called"] += 1
        saved["arg"] = info_df

    monkeypatch.setattr(li_mod.LateralityIndexAnalyzer, "_save", fake_save, raising=True)

    analyzer = li_mod.LateralityIndexAnalyzer(subjects=["S1", "S2"], froi="dummy_froi_config")

    df = analyzer.run(save=True)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1  # S2 skipped
    assert df.loc[0, "subject"] == "S1"
    assert df.loc[0, "n_left"] == 1
    assert df.loc[0, "n_right"] == 1
    assert df.loc[0, "laterality_index"] == 0.0

    assert saved["called"] == 1
    assert list(saved["arg"].columns) == ["froi"]


def test_analyzer_run_does_not_save_when_save_false(monkeypatch):
    data = np.zeros((3, 1, 1), dtype=float)
    data[0, 0, 0] = 1
    affine = np.eye(4)
    img = _make_img_with_affine(data, affine)

    monkeypatch.setattr(
        li_mod,
        "_get_froi_data",
        lambda subject, config, return_nifti, run_label: img,
    )

    saved = {"called": 0}
    monkeypatch.setattr(
        li_mod.LateralityIndexAnalyzer,
        "_save",
        lambda self, info_df: saved.__setitem__("called", saved["called"] + 1),
        raising=True,
    )

    analyzer = li_mod.LateralityIndexAnalyzer(subjects=["S1"], froi="dummy")
    df = analyzer.run(save=False)

    assert df.shape[0] == 1
    assert saved["called"] == 0
