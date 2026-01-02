import numpy as np
import pandas as pd
import pytest

import funROI
import funROI.analysis.froi_gen as froi_gen_mod
from funROI.analysis.tests.utils import DummyFROIConfig


def _img_from_flat(flat, shape=(2, 2, 1)):
    import nibabel as nib
    data = np.asarray(flat, dtype=float).reshape(shape)
    return nib.Nifti1Image(data, affine=np.eye(4))

@pytest.fixture
def tmp_settings(tmp_path):
    # Use package IO, as requested.
    out = tmp_path / "analysis_out"
    funROI.set_analysis_output_folder(out)
    yield out
    funROI.reset_settings()


def test_froi_generator_run_loads_existing_and_saves(tmp_settings, tmp_path, monkeypatch):
    cfg = DummyFROIConfig(task="T")

    # Create a "first-level" froi file somewhere, and force _get_froi_path to point to it.
    froi_src = tmp_path / "src_froi.nii.gz"
    _img_from_flat([1, 2, 0, 0]).to_filename(froi_src)

    monkeypatch.setattr(froi_gen_mod, "_get_froi_path", lambda subject, run_label, config: froi_src)
    monkeypatch.setattr(froi_gen_mod, "_create_froi", lambda *a, **k: None)  # should not be called

    gen = froi_gen_mod.FROIGenerator(subjects=["S1"], froi=cfg, run_label="all")
    out = gen.run(save=True)

    assert out is not None and len(out) == 1
    assert out[0][0] == "S1"

    # Should have saved into analysis output folder under froi_T/...
    froi_info = tmp_settings / "froi_T" / "froi_info.csv"
    assert froi_info.exists()

    info_df = pd.read_csv(froi_info)
    assert len(info_df) == 1

    saved_img_path = tmp_settings / "froi_T" / "froi_0000" / "sub-S1_run-all_froi.nii.gz"
    assert saved_img_path.exists()


def test_froi_generator_run_calls_create_when_missing(tmp_settings, tmp_path, monkeypatch):
    cfg = DummyFROIConfig(task="T")

    froi_src = tmp_path / "src_froi_missing_then_created.nii.gz"

    # First call: path does not exist. _create_froi should create it.
    monkeypatch.setattr(froi_gen_mod, "_get_froi_path", lambda subject, run_label, config: froi_src)

    def fake_create(subject, config, run_label):
        _img_from_flat([1, 0, 0, 0]).to_filename(froi_src)

    monkeypatch.setattr(froi_gen_mod, "_create_froi", fake_create)

    gen = froi_gen_mod.FROIGenerator(subjects=["S1"], froi=cfg, run_label="all")
    out = gen.run(save=False)

    assert out is not None and len(out) == 1
    assert froi_src.exists()


def test_get_analysis_froi_path_id_stable_and_increments(tmp_settings):
    cfg1 = DummyFROIConfig(task="T", contrasts=["a"], threshold_type="none", threshold_value=0.0, parcels="P")
    cfg1_same = DummyFROIConfig(task="T", contrasts=["a"], threshold_type="none", threshold_value=0.0, parcels="P")
    cfg2 = DummyFROIConfig(task="T", contrasts=["b"], threshold_type="none", threshold_value=0.0, parcels="P")

    # Create paths (and thus info rows)
    p1 = froi_gen_mod.FROIGenerator._get_analysis_froi_path("S1", "all", cfg1, create=True)
    p1b = froi_gen_mod.FROIGenerator._get_analysis_froi_path("S2", "all", cfg1_same, create=True)
    p2 = froi_gen_mod.FROIGenerator._get_analysis_froi_path("S1", "all", cfg2, create=True)

    # Same config should map to same id folder
    assert "froi_0000" in str(p1)
    assert "froi_0000" in str(p1b)

    # Different config should be new id
    assert "froi_0001" in str(p2)

    info_path = tmp_settings / "froi_T" / "froi_info.csv"
    df = pd.read_csv(info_path)
    assert set(df["id"].astype(int).tolist()) == {0, 1}


def test_froi_generator_select_label_not_found_raises(tmp_settings, monkeypatch):
    cfg = DummyFROIConfig(task="T", parcels="P")

    # Pretend we already ran and have one image in memory
    gen = froi_gen_mod.FROIGenerator(subjects=["S1"], froi=cfg, run_label="all")
    gen.subjects = ["S1"]
    gen._data = [_img_from_flat([1, 1, 0, 0])]

    # parcels labels do not include requested label
    monkeypatch.setattr(froi_gen_mod, "get_parcels", lambda p: (_img_from_flat([0, 0, 0, 0]), {1: "A"}))

    with pytest.raises(ValueError, match="Label .* not found in parcels labels"):
        gen.select(froi_label="NOT_A_REAL_LABEL", return_results=False)


@pytest.mark.xfail(reason="Bug in select(): overwrites list `data` with ndarray then calls data.append(...)")
def test_froi_generator_select_returns_results_when_valid_label(tmp_settings, monkeypatch):
    cfg = DummyFROIConfig(task="T", parcels="P")
    gen = froi_gen_mod.FROIGenerator(subjects=["S1"], froi=cfg, run_label="all")
    gen.subjects = ["S1"]
    gen._data = [_img_from_flat([1, 2, 0, 0])]

    # Label mapping includes 1 -> "A"
    monkeypatch.setattr(froi_gen_mod, "get_parcels", lambda p: (_img_from_flat([0, 0, 0, 0]), {1: "A"}))

    results = gen.select(froi_label=1, return_results=True)
    assert results is not None
    assert results[0][0] == "S1"
