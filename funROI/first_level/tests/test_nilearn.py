# The test mocks nilearn functions to test that run_first_level to avoids
# the costly operations of fitting models and writing images...

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import funROI.first_level.nilearn as nl
from funROI.first_level.tests.utils import FakeImg

class FakeResidual:
    def to_filename(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake-residuals")

class FakeFirstLevelModel:
    def __init__(self, subject_label="100307", t_r=0.72):
        self.subject_label = subject_label
        self.t_r = t_r

        self.hrf_model = "glover"
        self.drift_model = "cosine"
        self.high_pass = 0.01
        self.drift_order = 1
        self.fir_delays = None
        self.min_onset = 0

        self.residuals = [FakeResidual()]

        self.fit_called = False
        self.fit_args = None

    def fit(self, run_img_grand, design_matrices):
        self.fit_called = True
        self.fit_args = (run_img_grand, design_matrices)
        return self

    def compute_contrast(self, contrast_vec, stat_type="t", output_type="all"):
        # Return the dict keyed by IMAGE_SUFFIXES
        return {k: FakeImg() for k in nl.IMAGE_SUFFIXES.keys()}


def _mk_paths(monkeypatch, tmp_path: Path):
    """
    Patch all path getters used by nilearn.py so everything writes under tmp_path.
    """
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)

    def p(*parts):
        return out.joinpath(*parts)

    contrast_folder = lambda s, t: p("contrasts", f"sub-{s}", f"task-{t}")
    contrast_path   = lambda s, t, r, c, x: contrast_folder(s, t) / f"run-{r}_{c}_{x}.nii.gz"
    design_matrix_path = lambda s, t: p("design", f"sub-{s}_task-{t}_design.csv")
    residuals_path     = lambda s, t: p("resid",  f"sub-{s}_task-{t}_residuals.nii.gz")

    monkeypatch.setattr(nl, "_get_contrast_folder", contrast_folder)
    monkeypatch.setattr(nl, "_get_contrast_path", contrast_path)
    monkeypatch.setattr(nl, "_get_design_matrix_path", design_matrix_path)
    monkeypatch.setattr(nl, "_get_residuals_path", residuals_path)

    return out

@pytest.mark.parametrize(
    "n_runs, runs_expected",
    [
        (1, ["01", "all"]),
        (2, ["01", "02", "all", "odd", "even", "orth01", "orth02"]),
    ],
)
def test_run_first_level_writes_design_residuals_and_contrasts(tmp_path, monkeypatch, n_runs, runs_expected):
    out = _mk_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(nl, "get_bids_data_folder", lambda: tmp_path / "bids")
    monkeypatch.setattr(nl, "get_bids_preprocessed_folder_relative", lambda: "derivatives")
    monkeypatch.setattr(nl, "get_bids_preprocessed_folder", lambda: tmp_path / "derivatives")

    model = FakeFirstLevelModel(subject_label="100307", t_r=0.72)

    run_img_paths = [f"fake_run_{i:02d}.nii.gz" for i in range(1, n_runs + 1)]
    events = [pd.DataFrame({"trial_type": ["math"], "onset": [0.0], "duration": [1.0]}) for _ in range(n_runs)]
    confounds = [pd.DataFrame({"conf1": [0, 0, 0, 0]}) for _ in range(n_runs)]
    if n_runs == 1: confounds = confounds[0]

    def fake_first_level_from_bids(*args, **kwargs):
        return (
            [model],
            [run_img_paths],  # per-subject list of run paths
            [events],         # per-subject list of per-run event DFs
            [confounds],      # per-subject list of per-run confound DFs
        )
    monkeypatch.setattr(nl, "first_level_from_bids", fake_first_level_from_bids)

    def fake_load_img(_):  # each run has 4 TRs
        return FakeImg(np.zeros((2, 2, 2, 4), dtype=float))

    def fake_new_img_like(_ref, data):
        return FakeImg(np.asarray(data))

    monkeypatch.setattr(nl, "load_img", fake_load_img)
    monkeypatch.setattr(nl, "new_img_like", fake_new_img_like)

    def fake_make_first_level_design_matrix(*, frame_times, events, **kwargs):
        n = len(frame_times)
        return pd.DataFrame({"math": np.ones(n), "story": np.zeros(n)})
    monkeypatch.setattr(nl, "make_first_level_design_matrix", fake_make_first_level_design_matrix)

    registered = []
    def fake_register(sub, task, con_name, con_vec):
        registered.append((sub, task, con_name, list(con_vec)))
    monkeypatch.setattr(nl, "_register_contrast", fake_register)

    nl.run_first_level(
        task="LANGUAGE",
        subjects=["100307"],
        contrasts=[("math_gt_story", {"math": 1.0, "story": -1.0})],
        orthogs=["odd-even", "all-but-one"],
    )

    dm_path = out / "design" / "sub-100307_task-LANGUAGE_design.csv"
    assert dm_path.exists()
    dm = pd.read_csv(dm_path)

    # n_runs x 4 TRs
    assert dm.shape[0] == 4 * n_runs
    for i in range(1, n_runs + 1):
        assert any(c.startswith(f"run-{i:02d}_") for c in dm.columns)

    resid_path = out / "resid" / "sub-100307_task-LANGUAGE_residuals.nii.gz"
    assert resid_path.exists()

    assert len(registered) == 1
    assert registered[0][2] == "math_gt_story"
    assert len(registered[0][3]) == len(dm.columns)

    suffixes = list(nl.IMAGE_SUFFIXES.values())
    for run in runs_expected:
        for suf in suffixes:
            p = (
                out
                / "contrasts"
                / "sub-100307"
                / "task-LANGUAGE"
                / f"run-{run}_math_gt_story_{suf}.nii.gz"
            )
            assert p.exists(), f"missing {p}"

def test_run_first_level_raises_on_invalid_regressor(tmp_path, monkeypatch):
    """
    If a contrast references a regressor not in the design matrix columns, it should raise ValueError.
    """
    _mk_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(np, "concat", np.concatenate, raising=False)

    monkeypatch.setattr(nl, "get_bids_data_folder", lambda: tmp_path / "bids")
    monkeypatch.setattr(nl, "get_bids_preprocessed_folder_relative", lambda: "derivatives")
    monkeypatch.setattr(nl, "get_bids_preprocessed_folder", lambda: tmp_path / "derivatives")

    model = FakeFirstLevelModel(subject_label="100307", t_r=0.72)

    def fake_first_level_from_bids(*args, **kwargs):
        return (
            [model],
            [["fake_run_01.nii.gz"]],
            [[pd.DataFrame({"trial_type": ["math"], "onset": [0.0], "duration": [1.0]})]],
            [[pd.DataFrame({"conf1": [0, 0, 0, 0]})]],
        )

    monkeypatch.setattr(nl, "first_level_from_bids", fake_first_level_from_bids)
    monkeypatch.setattr(nl, "load_img", lambda _: FakeImg(np.zeros((2, 2, 2, 4))))
    monkeypatch.setattr(nl, "new_img_like", lambda ref, data: FakeImg(np.asarray(data)))

    def fake_make_first_level_design_matrix(*, frame_times, events, **kwargs):
        return pd.DataFrame({"math": np.ones(len(frame_times))})

    monkeypatch.setattr(nl, "make_first_level_design_matrix", fake_make_first_level_design_matrix)
    monkeypatch.setattr(nl, "_register_contrast", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="invalid regressor name"):
        nl.run_first_level(
            task="LANGUAGE",
            subjects=["100307"],
            contrasts=[("bad_con", {"NOT_IN_DM": 1.0})],
        )


def test_run_first_level_raises_when_bids_folder_not_set(monkeypatch):
    monkeypatch.setattr(
        nl,
        "get_bids_data_folder",
        lambda: (_ for _ in ()).throw(ValueError("no bids")),
    )

    with pytest.raises(
        ValueError,
        match="The output directory is not set.*cannot be inferred",
    ):
        nl.run_first_level(task="LANGUAGE", subjects=["100307"])


def test_run_first_level_uses_preprocessed_folder_when_relative_derivatives_missing(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(nl, "get_bids_data_folder", lambda: tmp_path / "bids")
    monkeypatch.setattr(
        nl,
        "get_bids_preprocessed_folder_relative",
        lambda: (_ for _ in ()).throw(ValueError("no relative derivatives")),
    )
    monkeypatch.setattr(
        nl,
        "get_bids_preprocessed_folder",
        lambda: tmp_path / "derivatives",
    )

    called = {}
    def fake_first_level_from_bids(
        bids_folder, task, *, derivatives_folder=None, **kwargs
    ):
        called["bids_folder"] = bids_folder
        called["derivatives_folder"] = derivatives_folder
        raise RuntimeError("stop")  # abort early on purpose

    monkeypatch.setattr(nl, "first_level_from_bids", fake_first_level_from_bids)

    with pytest.raises(RuntimeError, match="stop"):
        nl.run_first_level(task="LANGUAGE", subjects=["100307"])

    assert called["bids_folder"] == tmp_path / "derivatives"
    assert called["derivatives_folder"] == "."
