import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import nibabel as nib

import funROI
import funROI.analysis.parcels_gen as parcels_gen_mod
from funROI.analysis.tests.utils import DummyFROIConfig


def _img(shape=(2, 2, 1), affine=None, fill=None):
    if affine is None:
        affine = np.eye(4)
    if fill is None:
        data = np.zeros(shape, dtype=float)
    else:
        data = np.asarray(fill, dtype=float).reshape(shape)
    return nib.Nifti1Image(data, affine)


@pytest.fixture
def tmp_settings(tmp_path):
    out = tmp_path / "analysis_out"
    funROI.set_analysis_output_folder(out)
    yield out
    funROI.reset_settings()


def test_add_subjects_rejects_redundant_subjects(monkeypatch, tmp_settings):
    monkeypatch.setattr(parcels_gen_mod, "FROIConfig", DummyFROIConfig, raising=False)

    gen = parcels_gen_mod.ParcelsGenerator("P")
    # Pretend we already added S1 once
    gen.configs = [{"subjects": ["S1"], "froi": DummyFROIConfig("T", ["c"], "none", 0.05, "and", None)}]

    with pytest.raises(ValueError, match="already added"):
        gen.add_subjects(
            subjects=["S1", "S2"],
            task="T",
            contrasts=["c"],
            p_threshold_type="none",
            p_threshold_value=0.05,
            conjunction_type="and",
        )


def test_add_subjects_sets_shape_affine_and_collects_data(monkeypatch, tmp_settings, tmp_path):
    monkeypatch.setattr(parcels_gen_mod, "FROIConfig", DummyFROIConfig, raising=False)

    # 3 runs => "orth<run>" behavior
    monkeypatch.setattr(parcels_gen_mod, "_get_froi_runs", lambda subject, froi: ["01", "02", "03"])

    created = []

    def fake_get_froi_path(subject, run, froi):
        return tmp_path / f"sub-{subject}_run-{run}.nii.gz"

    def fake_create_froi(subject, froi, run):
        created.append((subject, run))
        # create file with known data: one voxel active for run string
        data = np.array([1, 0, 0, 0], dtype=float).reshape((2, 2, 1))
        _img(shape=(2, 2, 1), affine=np.eye(4), fill=data).to_filename(fake_get_froi_path(subject, run, froi))

    def fake_load_img(pth):
        # nibabel load is fine
        return nib.load(str(pth))

    monkeypatch.setattr(parcels_gen_mod, "_get_froi_path", fake_get_froi_path)
    monkeypatch.setattr(parcels_gen_mod, "_create_froi", fake_create_froi)
    monkeypatch.setattr(parcels_gen_mod, "load_img", fake_load_img)

    gen = parcels_gen_mod.ParcelsGenerator("P")
    gen.add_subjects(
        subjects=["S1"],
        task="T",
        contrasts=["c1"],
        p_threshold_type="none",
        p_threshold_value=0.05,
        conjunction_type="and",
    )

    # For 3 runs, run label should be "orth<run>"
    assert created == [("S1", "orth01"), ("S1", "orth02"), ("S1", "orth03")]

    assert gen.img_shape == (2, 2, 1)
    assert np.allclose(gen.img_affine, np.eye(4))
    assert len(gen._data) == 1
    subj_data = gen._data[0]
    assert subj_data.shape == (3, 4)  # (n_runs, n_vox)
    assert len(gen.configs) == 1
    assert gen.configs[0]["subjects"] == ["S1"]


def test_add_subjects_two_runs_uses_other_run_not_orth(monkeypatch, tmp_settings, tmp_path):
    monkeypatch.setattr(parcels_gen_mod, "FROIConfig", DummyFROIConfig, raising=False)

    # exactly 2 runs: code swaps to "the other run" (weird but test it)
    monkeypatch.setattr(parcels_gen_mod, "_get_froi_runs", lambda subject, froi: ["01", "02"])

    calls = []

    def fake_get_froi_path(subject, run, froi):
        return tmp_path / f"sub-{subject}_run-{run}.nii.gz"

    def fake_create_froi(subject, froi, run):
        calls.append(run)
        _img(shape=(2, 2, 1), affine=np.eye(4)).to_filename(fake_get_froi_path(subject, run, froi))

    monkeypatch.setattr(parcels_gen_mod, "_get_froi_path", fake_get_froi_path)
    monkeypatch.setattr(parcels_gen_mod, "_create_froi", fake_create_froi)
    monkeypatch.setattr(parcels_gen_mod, "load_img", lambda p: nib.load(str(p)))

    gen = parcels_gen_mod.ParcelsGenerator("P")
    gen.add_subjects(
        subjects=["S1"],
        task="T",
        contrasts=["c1"],
        p_threshold_type="none",
        p_threshold_value=0.05,
        conjunction_type="and",
    )

    # When iterating run="01" -> uses "02"; run="02" -> uses "01"
    assert calls == ["02", "01"]


def test_add_subjects_raises_on_shape_affine_mismatch(monkeypatch, tmp_settings, tmp_path):
    monkeypatch.setattr(parcels_gen_mod, "FROIConfig", DummyFROIConfig, raising=False)
    monkeypatch.setattr(parcels_gen_mod, "_get_froi_runs", lambda subject, froi: ["01"])

    def fake_get_froi_path(subject, run, froi):
        return tmp_path / f"sub-{subject}_run-{run}.nii.gz"

    def fake_create_froi(subject, froi, run):
        # S1: 2x2x1; S2: 3x1x1
        if subject == "S1":
            img = _img(shape=(2, 2, 1), affine=np.eye(4))
        else:
            img = _img(shape=(3, 1, 1), affine=np.eye(4))
        img.to_filename(fake_get_froi_path(subject, run, froi))

    monkeypatch.setattr(parcels_gen_mod, "_get_froi_path", fake_get_froi_path)
    monkeypatch.setattr(parcels_gen_mod, "_create_froi", fake_create_froi)
    monkeypatch.setattr(parcels_gen_mod, "load_img", lambda p: nib.load(str(p)))

    gen = parcels_gen_mod.ParcelsGenerator("P")
    with pytest.raises(ValueError, match="same shape and affine"):
        gen.add_subjects(
            subjects=["S1", "S2"],
            task="T",
            contrasts=["c1"],
            p_threshold_type="none",
            p_threshold_value=0.05,
            conjunction_type="and",
        )


def test_add_subjects_raises_if_no_runs(monkeypatch, tmp_settings):
    monkeypatch.setattr(parcels_gen_mod, "FROIConfig", DummyFROIConfig, raising=False)
    monkeypatch.setattr(parcels_gen_mod, "_get_froi_runs", lambda subject, froi: [])

    gen = parcels_gen_mod.ParcelsGenerator("P")
    with pytest.raises(ValueError, match="No data found"):
        gen.add_subjects(
            subjects=["S1"],
            task="T",
            contrasts=["c1"],
            p_threshold_type="none",
            p_threshold_value=0.05,
            conjunction_type="and",
        )


def test_run_calls_internal_run_and_saves(monkeypatch, tmp_settings):
    gen = parcels_gen_mod.ParcelsGenerator(
        "P",
        smoothing_kernel_size=8,
        overlap_thr_vox=0.25,
        use_spm_smooth=True,
    )

    # Pretend data already loaded: list of per-subject arrays (n_runs, n_vox)
    gen._data = [
        np.array([[1, 0, 0, 0], [1, 0, 0, 0]], dtype=float),
        np.array([[0, 1, 0, 0], [0, 1, 0, 0]], dtype=float),
    ]
    gen.img_shape = (2, 2, 1)
    gen.img_affine = np.eye(4)

    # Make _run deterministic: overlap_map and parcels labels
    overlap_map = np.ones((2, 2, 1), dtype=float) * 0.5
    parcels = np.array([1, 1, 0, 2], dtype=float).reshape((2, 2, 1))

    monkeypatch.setattr(
        parcels_gen_mod.ParcelsGenerator,
        "_run",
        classmethod(lambda cls, *a, **k: (overlap_map, parcels)),
    )

    # Avoid filesystem annoyance from repeated saves? No: test that it writes.
    out_img = gen.run()
    assert isinstance(out_img, nib.Nifti1Image)

    # Outputs should exist
    base = tmp_settings / "parcels" / "parcels-P"
    assert (base / "parcels-P_config.json").exists()
    assert (base / "parcels-P_overlap.nii.gz").exists()

    info_csv = base / "parcels-P_sm-8_spmsmooth-True_voxthres-0.25_info.csv"
    assert info_csv.exists()
    df = pd.read_csv(info_csv)
    assert set(df.columns) == {"id", "size", "roi_overlap"}

    parcels_nii = base / "parcels-P_sm-8_spmsmooth-True_voxthres-0.25_roithres-0_sz-0.nii.gz"
    assert parcels_nii.exists()


def test_run_applies_filter_when_thresholds_nonzero(monkeypatch, tmp_settings):
    gen = parcels_gen_mod.ParcelsGenerator("P", min_voxel_size=2, overlap_thr_roi=0.9)
    gen._data = [np.ones((1, 4), dtype=float)]
    gen.img_shape = (2, 2, 1)
    gen.img_affine = np.eye(4)

    # parcels with ids 1 and 2
    overlap_map = np.ones((2, 2, 1), dtype=float)
    parcels = np.array([1, 1, 2, 0], dtype=float).reshape((2, 2, 1))

    # parcel_info: parcel 1 passes size=2 overlap=1.0; parcel 2 fails overlap and size
    parcel_info = pd.DataFrame(
        {"id": [1, 2], "size": [2, 1], "roi_overlap": [1.0, 0.1]}
    )

    def fake_run(*a, **k):
        return overlap_map, parcels

    monkeypatch.setattr(parcels_gen_mod.ParcelsGenerator, "_run", classmethod(lambda cls, *a, **k: fake_run()))
    # Force parcel_info to be created as above by intercepting after run:
    def fake_save(self):
        # do nothing during first save so we can inject parcel_info before filtering
        return

    monkeypatch.setattr(parcels_gen_mod.ParcelsGenerator, "_save", fake_save, raising=True)

    # Run once to generate parcels/parcels_info
    gen.overlap_map, gen.parcels = overlap_map, parcels
    gen.parcel_info = parcel_info

    # Now call run(): should call _filter because thresholds nonzero
    # We need real _save back to avoid crash after filtering; monkeypatch per-call:
    monkeypatch.setattr(parcels_gen_mod.ParcelsGenerator, "_save", parcels_gen_mod.ParcelsGenerator._save, raising=True)

    out = gen.run()
    data = out.get_fdata()
    # parcel 2 should be removed (set to 0)
    assert np.all(data.reshape(-1) != 2)


def test_filter_runtime_error_when_no_parcels(tmp_settings):
    gen = parcels_gen_mod.ParcelsGenerator("P")
    with pytest.raises(RuntimeError, match="Run the parcels generation first"):
        gen.filter(overlap_thr_roi=0.5, min_voxel_size=10)


def test_filter_warns_and_noop_when_thresholds_not_stricter(monkeypatch, tmp_settings):
    gen = parcels_gen_mod.ParcelsGenerator("P", overlap_thr_roi=0.5, min_voxel_size=10)
    gen.parcels = np.array([1, 0, 0, 0], dtype=float).reshape((2, 2, 1))
    gen.img_affine = np.eye(4)
    gen._data = [np.ones((1, 4), dtype=float)]

    # intercept _save to ensure called (or not)
    saves = []
    monkeypatch.setattr(parcels_gen_mod.ParcelsGenerator, "_save", lambda self: saves.append(True))

    with pytest.warns(UserWarning):
        gen.filter(overlap_thr_roi=0.4, min_voxel_size=5)  # both <= current -> no filtering
    # Should not have saved because no filtering applied (code only saves when applying)
    assert saves == []


def test_filter_applies_and_updates_thresholds(monkeypatch, tmp_settings):
    gen = parcels_gen_mod.ParcelsGenerator("P", overlap_thr_roi=0.0, min_voxel_size=0)
    gen.parcels = np.array([1, 1, 2, 0], dtype=float).reshape((2, 2, 1))
    gen.img_affine = np.eye(4)
    gen._data = [np.ones((1, 4), dtype=float)]

    # patch _filter to remove parcel 2
    monkeypatch.setattr(
        parcels_gen_mod.ParcelsGenerator,
        "_filter",
        classmethod(lambda cls, parcels, parcel_info, overlap_thr_roi, min_voxel_size: np.where(parcels == 2, 0, parcels)),
    )
    monkeypatch.setattr(parcels_gen_mod.ParcelsGenerator, "_save", lambda self: None)

    out = gen.filter(overlap_thr_roi=0.9, min_voxel_size=2)
    assert isinstance(out, nib.Nifti1Image)
    assert gen.overlap_thr_roi == 0.9
    assert gen.min_voxel_size == 2
    assert 2 not in out.get_fdata().reshape(-1)


def test_run_internal_uses_spm_smooth_branch(monkeypatch):
    # Pin that _run calls _smooth_array when use_spm_smooth True
    called = {"smooth": 0, "nilearn": 0, "watershed": 0}

    def fake_smooth_array(arr, affine, fwhm):
        called["smooth"] += 1
        return arr  # no-op

    def fake_smooth_img(img, fwhm):
        called["nilearn"] += 1
        return img

    def fake_watershed(A):
        called["watershed"] += 1
        return np.zeros_like(A)

    monkeypatch.setattr(parcels_gen_mod.ParcelsGenerator, "_smooth_array", classmethod(lambda cls, arr, affine, fwhm: fake_smooth_array(arr, affine, fwhm)))
    monkeypatch.setattr(parcels_gen_mod, "smooth_img", fake_smooth_img)
    monkeypatch.setattr(parcels_gen_mod.ParcelsGenerator, "_watershed", classmethod(lambda cls, A: fake_watershed(A)))

    masks = [np.array([1, 0, 0, 0], dtype=float), np.array([1, 0, 0, 0], dtype=float)]
    overlap, parcels = parcels_gen_mod.ParcelsGenerator._run(
        binary_masks=masks,
        img_shape=(2, 2, 1),
        img_affine=np.eye(4),
        smoothing_kernel_size=8,
        overlap_thr_vox=0.1,
        use_spm_smooth=True,
    )
    assert called["smooth"] == 1
    assert called["nilearn"] == 0
    assert called["watershed"] == 1
    assert overlap.shape == (2, 2, 1)
    assert parcels.shape == (2, 2, 1)


def test_run_internal_uses_nilearn_branch(monkeypatch):
    called = {"smooth": 0, "nilearn": 0, "watershed": 0}

    def fake_smooth_array(arr, affine, fwhm):
        called["smooth"] += 1
        return arr

    class FakeSmoothed:
        def __init__(self, img):
            self._img = img
        def get_fdata(self):
            called["nilearn"] += 1
            return self._img.get_fdata()

    def fake_smooth_img(img, fwhm):
        return FakeSmoothed(img)

    def fake_watershed(A):
        called["watershed"] += 1
        return np.zeros_like(A)

    monkeypatch.setattr(parcels_gen_mod.ParcelsGenerator, "_smooth_array", classmethod(lambda cls, arr, affine, fwhm: fake_smooth_array(arr, affine, fwhm)))
    monkeypatch.setattr(parcels_gen_mod, "smooth_img", fake_smooth_img)
    monkeypatch.setattr(parcels_gen_mod.ParcelsGenerator, "_watershed", classmethod(lambda cls, A: fake_watershed(A)))

    masks = [np.array([1, 0, 0, 0], dtype=float), np.array([1, 0, 0, 0], dtype=float)]
    overlap, parcels = parcels_gen_mod.ParcelsGenerator._run(
        binary_masks=masks,
        img_shape=(2, 2, 1),
        img_affine=np.eye(4),
        smoothing_kernel_size=8,
        overlap_thr_vox=0.1,
        use_spm_smooth=False,
    )
    assert called["smooth"] == 0
    assert called["nilearn"] == 1  # get_fdata called once
    assert called["watershed"] == 1


def test_filter_function_filters_by_overlap_and_size():
    parcels = np.array([1, 1, 2, 0], dtype=int).reshape((2, 2, 1))
    info = pd.DataFrame(
        {"id": [1, 2], "size": [2, 1], "roi_overlap": [0.9, 0.05]}
    )
    out = parcels_gen_mod.ParcelsGenerator._filter(
        parcels=parcels,
        parcel_info=info,
        overlap_thr_roi=0.1,
        min_voxel_size=2,
    )
    # parcel 1 stays, parcel 2 removed
    assert 1 in np.unique(out)
    assert 2 not in np.unique(out)


def test_harmonic_mean_matches_definition():
    x = np.array([1.0, 2.0, 4.0])
    hm = parcels_gen_mod.ParcelsGenerator._harmonic_mean(x)
    # harmonic mean = n / sum(1/x)
    expected = len(x) / np.sum(1 / x)
    assert np.isclose(hm, expected)


def test_save_writes_expected_files(tmp_settings):
    gen = parcels_gen_mod.ParcelsGenerator("P", smoothing_kernel_size=8, overlap_thr_vox=0.1, use_spm_smooth=True)
    gen.configs = [{"subjects": ["S1"], "froi": {"dummy": True}}]
    gen.overlap_map = np.ones((2, 2, 1), dtype=float)
    gen.parcels = np.array([1, 0, 0, 2], dtype=float).reshape((2, 2, 1))
    gen.img_affine = np.eye(4)
    gen.parcel_info = pd.DataFrame({"id": [1, 2], "size": [1, 1], "roi_overlap": [1.0, 1.0]})

    gen._save()

    base = tmp_settings / "parcels" / "parcels-P"
    cfg = base / "parcels-P_config.json"
    assert cfg.exists()
    with open(cfg, "r") as f:
        js = json.load(f)
    assert "configs" in js

    assert (base / "parcels-P_overlap.nii.gz").exists()
    assert (base / "parcels-P_sm-8_spmsmooth-True_voxthres-0.1_info.csv").exists()
    assert (base / "parcels-P_sm-8_spmsmooth-True_voxthres-0.1_roithres-0_sz-0.nii.gz").exists()
