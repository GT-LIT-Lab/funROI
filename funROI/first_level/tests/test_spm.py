from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import h5py
import funROI.first_level.spm as spm_mod
from funROI.first_level.tests.utils import FakeImg


def _write_hdf5_spm_mat(spm_mat_path: Path, *, X: np.ndarray, colnames: list[str],
                        contrasts: dict[str, np.ndarray], erdf: int = 120):
    spm_mat_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(spm_mat_path, "w") as f:
        spm = f.create_group("SPM")
        xX = spm.create_group("xX")
        xCon = spm.create_group("xCon")

        xX.create_dataset("erdf", data=np.array([[erdf]], dtype=np.float64))
        f.create_dataset("/SPM/xX/X", data=X.astype(np.float64))

        def put_string(ds_name: str, s: str):
            # MATLAB-ish char array: Nx1 of uint16 char codes
            arr = np.array([[ord(c)] for c in s], dtype=np.uint16)
            f.create_dataset(ds_name, data=arr)

        name_refs = []
        for i, nm in enumerate(colnames):
            ds_name = f"/SPM/_name_{i}"
            put_string(ds_name, nm)
            name_refs.append([f[ds_name].ref])  # <-- wrap in list to make (n,1)

        f.create_dataset(
            "/SPM/xX/name",
            data=np.array(name_refs, dtype=h5py.ref_dtype),
        )

        con_name_refs = []
        con_vec_refs = []
        for i, (cname, cvec) in enumerate(contrasts.items(), start=1):
            ds_n = f"/SPM/_conname_{i}"
            put_string(ds_n, cname)
            con_name_refs.append([f[ds_n].ref])  # <-- (n,1)

            ds_v = f"/SPM/_convec_{i}"
            f.create_dataset(ds_v, data=cvec.astype(np.float64))
            con_vec_refs.append([f[ds_v].ref])  

        f.create_dataset(
            "/SPM/xCon/name",
            data=np.array(con_name_refs, dtype=h5py.ref_dtype),
        )
        f.create_dataset(
            "/SPM/xCon/c",
            data=np.array(con_vec_refs, dtype=h5py.ref_dtype),
        )


def test_migrate_first_level_from_spm_raises_when_spm_mat_missing(tmp_path, monkeypatch):
    spm_dir = tmp_path / "spm"
    spm_dir.mkdir(parents=True, exist_ok=True)

    # Patch output folders to land in tmp_path and avoid any global config
    monkeypatch.setattr(spm_mod, "_get_model_folder", lambda sub, task: tmp_path / "model" / sub / task)
    monkeypatch.setattr(spm_mod, "_get_contrast_folder", lambda sub, task: tmp_path / "con" / sub / task)

    with pytest.raises(FileNotFoundError, match=r"'SPM\.mat' not found"):
        spm_mod.migrate_first_level_from_spm(spm_dir=spm_dir, subject="100307", task="LANGUAGE")


def test_migrate_first_level_from_spm_writes_design_and_contrasts(tmp_path, monkeypatch):
    subject = "100307"
    task = "LANGUAGE"
    spm_dir = tmp_path / "spm"
    spm_dir.mkdir(parents=True, exist_ok=True)

    # Patch all output path helpers to write under tmp_path
    out_root = tmp_path / "out"
    out_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(spm_mod, "_get_model_folder", lambda sub, task: out_root / "model" / sub / task)
    monkeypatch.setattr(spm_mod, "_get_contrast_folder", lambda sub, task: out_root / "contrasts" / sub / task)
    monkeypatch.setattr(spm_mod, "_get_design_matrix_path", lambda sub, task: out_root / "design" / f"sub-{sub}_task-{task}_design.csv")

    def fake_contrast_path(sub, task, run_label, contrast_label, suffix):
        return out_root / "contrasts" / sub / task / f"run-{run_label}_{contrast_label}_{suffix}.nii.gz"

    monkeypatch.setattr(spm_mod, "_get_contrast_path", fake_contrast_path)

    # Capture contrast registration
    registered = []

    def fake_register(sub, task, con_name, con_vec):
        registered.append((sub, task, con_name, con_vec))

    monkeypatch.setattr(spm_mod, "_register_contrast", fake_register)

    colnames = ["Intercept", "RegA", "RegB"]
    X = np.array(
        [
            [1, 1, 1, 1],   # Intercept
            [0, 1, 0, 1],   # RegA
            [1, 0, 1, 0],   # RegB
        ],
        dtype=np.float64,
    )

    contrasts = {
        "SESSION1_CONA": np.array([[0, 1, -1]], dtype=np.float64),
        "ORTH_TO_SESSION1_CONA": np.array([[0, 1, -1]], dtype=np.float64),
        "ODD_CONA": np.array([[0, 1, -1]], dtype=np.float64),
        "EVEN_CONA": np.array([[0, 1, -1]], dtype=np.float64),
        "CONA": np.array([[0, 1, -1]], dtype=np.float64), 
    }

    spm_mat_path = spm_dir / "SPM.mat"
    _write_hdf5_spm_mat(spm_mat_path, X=X, colnames=colnames, contrasts=contrasts, erdf=120)

    def fake_load_img(path):
        path = str(path)
        if "spmT_" in path:
            t_data = np.array([[[0.0, 2.0]]], dtype=np.float32)  # includes a 0
            return FakeImg(t_data)
        elif "con_" in path:
            eff_data = np.array([[[1.0, 1.0]]], dtype=np.float32)
            return FakeImg(eff_data)
        raise FileNotFoundError(path)
    monkeypatch.setattr(spm_mod, "load_img", fake_load_img)

    def fake_new_img_like(ref_img, data):
        return FakeImg(np.asarray(data))
    monkeypatch.setattr(spm_mod, "new_img_like", fake_new_img_like)

    spm_mod.migrate_first_level_from_spm(spm_dir=spm_dir, subject=subject, task=task)

    dm_path = out_root / "design" / f"sub-{subject}_task-{task}_design.csv"
    assert dm_path.exists()
    dm = pd.read_csv(dm_path)
    assert list(dm.columns) == colnames
    assert dm.shape == (4, 3)

    assert {x[2] for x in registered} == set(contrasts.keys())
    for _, _, _, vec in registered:
        assert len(vec) == 3

    expected = [
        ("1", "CONA"),      # SESSION1_CONA -> run_label "1", contrast_label "CONA"
        ("orth1", "CONA"),  # ORTH_TO_SESSION1_CONA -> "orth1"
        ("odd", "CONA"),    # ODD_CONA
        ("even", "CONA"),   # EVEN_CONA
        ("all", "CONA"),    # CONA
    ]
    for run_label, con_label in expected:
        for suffix in ["effect", "t", "p"]:
            p = out_root / "contrasts" / subject / task / f"run-{run_label}_{con_label}_{suffix}.nii.gz"
            assert p.exists(), f"missing {p}"
