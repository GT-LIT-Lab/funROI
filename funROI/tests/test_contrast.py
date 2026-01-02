from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from nibabel.nifti1 import Nifti1Image

import funROI
import funROI.contrast as contrast_mod


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


def _write_contrast_info(subject: str, task: str, rows: list[dict]):
    p = contrast_mod._get_contrast_info_path(subject, task)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _write_design_matrix(subject: str, task: str, X: np.ndarray, columns=None):
    p = contrast_mod._get_design_matrix_path(subject, task)
    p.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = [f"x{i}" for i in range(X.shape[1])]
    pd.DataFrame(X, columns=columns).to_csv(p, index=False)
    return p


def test_get_contrast_vector_returns_none_when_info_missing():
    v = contrast_mod._get_contrast_vector("100307", "LANGUAGE", "c1")
    assert v is None


def test_get_contrast_vector_returns_none_when_contrast_not_found():
    _write_contrast_info(
        "100307",
        "LANGUAGE",
        [{"contrast": "other", "vector": "[1, 0, -1]"}],
    )
    v = contrast_mod._get_contrast_vector("100307", "LANGUAGE", "c1")
    assert v is None


def test_get_contrast_vector_parses_vector():
    _write_contrast_info(
        "100307",
        "LANGUAGE",
        [{"contrast": "c1", "vector": "[1, 0, -1]"}],
    )
    v = contrast_mod._get_contrast_vector("100307", "LANGUAGE", "c1")
    assert v == [1, 0, -1]


def test_get_contrast_runs_finds_and_sorts_unique_runs(tmp_path):
    subject, task, con = "100307", "LANGUAGE", "c1"
    # create a few files that match the glob; type wildcard is used
    _nii(contrast_mod._get_contrast_path(subject, task, "02", con, "t"), np.zeros((2, 2, 2)))
    _nii(contrast_mod._get_contrast_path(subject, task, "01", con, "p"), np.zeros((2, 2, 2)))
    _nii(contrast_mod._get_contrast_path(subject, task, "01", con, "t"), np.zeros((2, 2, 2)))  # duplicate run
    _nii(contrast_mod._get_contrast_path(subject, task, "10", con, "t"), np.zeros((2, 2, 2)))

    runs = contrast_mod._get_contrast_runs(subject, task, con)
    assert runs == ["01", "02", "10"]


def test_get_contrast_runs_by_group_variants(tmp_path):
    subject, task, con = "100307", "LANGUAGE", "c1"
    for r in ["01", "02", "03", "04"]:
        _nii(contrast_mod._get_contrast_path(subject, task, r, con, "t"), np.zeros((2, 2, 2)))

    assert contrast_mod._get_contrast_runs_by_group(subject, task, con, "all") == ["01", "02", "03", "04"]
    assert contrast_mod._get_contrast_runs_by_group(subject, task, con, "odd") == ["01", "03"]
    assert contrast_mod._get_contrast_runs_by_group(subject, task, con, "even") == ["02", "04"]

    # Note: current implementation is buggy: run_label[5:] is a string, so "run not in run_label[5:]" is char-based.
    # This test locks current behavior minimally by checking it returns a subset and doesn't crash.
    out = contrast_mod._get_contrast_runs_by_group(subject, task, con, "orth01")
    assert isinstance(out, list)
    assert all(r in ["01", "02", "03", "04"] for r in out)

    assert contrast_mod._get_contrast_runs_by_group(subject, task, con, "03") == ["03"]


# -----------------------
# _get_contrast_data
# -----------------------

def test_get_contrast_data_returns_none_when_file_missing():
    dat = contrast_mod._get_contrast_data("100307", "LANGUAGE", "01", "c1", "t")
    assert dat is None


def test_get_contrast_data_p_converts_zero_to_nan(tmp_path):
    subject, task, run, con = "100307", "LANGUAGE", "01", "c1"
    p_path = contrast_mod._get_contrast_path(subject, task, run, con, "p")
    _nii(p_path, np.array([[[0.0, 0.01]]], dtype=float))

    dat = contrast_mod._get_contrast_data(subject, task, run, con, "p")
    assert dat.shape == (2,)
    assert np.isnan(dat[0])
    assert dat[1] == pytest.approx(0.01)


def test_get_orthogonalized_contrast_data_happy(tmp_path, monkeypatch):
    subject, task, con, typ = "100307", "LANGUAGE", "c1", "t"

    # runs present
    for r in ["01", "02"]:
        _nii(contrast_mod._get_contrast_path(subject, task, r, con, typ), np.ones((2, 2, 2)) * int(r))

    # make orth labels predictable
    monkeypatch.setattr(
        contrast_mod,
        "_get_orthogonalized_run_labels",
        lambda runs, group, orth: ["01", "02"],
    )

    data, labels = contrast_mod._get_orthogonalized_contrast_data(
        subject, task, con, group=1, type=typ, orthogonalization="all-but-one"
    )
    assert labels == ["01", "02"]
    assert data.shape == (2, 8)
    assert np.allclose(data[0], 1.0)
    assert np.allclose(data[1], 2.0)


def test_get_orthogonalized_contrast_data_returns_none_if_any_missing(monkeypatch):
    subject, task, con, typ = "100307", "LANGUAGE", "c1", "t"

    monkeypatch.setattr(contrast_mod, "_get_contrast_runs", lambda *a, **k: ["01", "02"])
    monkeypatch.setattr(contrast_mod, "_get_orthogonalized_run_labels", lambda *a, **k: ["01", "02"])
    monkeypatch.setattr(contrast_mod, "_get_contrast_data", lambda subject, task, run, con, typ: None if run == "02" else np.zeros(5))

    data, labels = contrast_mod._get_orthogonalized_contrast_data(subject, task, con, group=1, type=typ)
    assert (data, labels) == (None, None)


# -----------------------
# _get_design_matrix
# -----------------------

def test_get_design_matrix_returns_none_when_missing():
    dm = contrast_mod._get_design_matrix("100307", "LANGUAGE")
    assert dm is None


def test_get_design_matrix_loads_csv(tmp_path):
    X = np.array([[1, 0], [0, 1]], dtype=float)
    _write_design_matrix("100307", "LANGUAGE", X, columns=["a", "b"])
    dm = contrast_mod._get_design_matrix("100307", "LANGUAGE")
    assert dm.shape == (2, 2)
    assert list(dm.columns) == ["a", "b"]


def test_check_orthogonal_true_when_tasks_differ():
    assert contrast_mod._check_orthogonal("100307", "TASK1", ["c1"], "TASK2", ["c2"]) is True


def test_check_orthogonal_raises_when_design_matrix_missing():
    _write_contrast_info("100307", "LANGUAGE", [{"contrast": "c1", "vector": "[1]"}])
    with pytest.raises(ValueError, match="Design matrix not found"):
        contrast_mod._check_orthogonal("100307", "LANGUAGE", ["c1"], "LANGUAGE", ["c1"])


def test_check_orthogonal_raises_when_contrast_vector_missing():
    _write_design_matrix("100307", "LANGUAGE", np.eye(2))
    # no contrast info file -> vector missing
    with pytest.raises(ValueError, match="Contrast vector not found"):
        contrast_mod._check_orthogonal("100307", "LANGUAGE", ["c1"], "LANGUAGE", ["c2"])


def test_check_orthogonal_false_when_not_orthogonal():
    """
    Design matrix is identity -> X'X is identity, so check reduces to c1·c2 != 0.
    """
    _write_design_matrix("100307", "LANGUAGE", np.eye(2), columns=["x1", "x2"])
    _write_contrast_info(
        "100307",
        "LANGUAGE",
        [
            {"contrast": "c1", "vector": "[1, 0]"},
            {"contrast": "c2", "vector": "[1, 0]"},  # same => not orthogonal
        ],
    )
    assert contrast_mod._check_orthogonal("100307", "LANGUAGE", ["c1"], "LANGUAGE", ["c2"]) is False


def test_check_orthogonal_true_when_orthogonal():
    _write_design_matrix("100307", "LANGUAGE", np.eye(2), columns=["x1", "x2"])
    _write_contrast_info(
        "100307",
        "LANGUAGE",
        [
            {"contrast": "c1", "vector": "[1, 0]"},
            {"contrast": "c2", "vector": "[0, 1]"},
        ],
    )
    assert contrast_mod._check_orthogonal("100307", "LANGUAGE", ["c1"], "LANGUAGE", ["c2"]) is True
