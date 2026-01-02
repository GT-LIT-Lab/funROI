from pathlib import Path
import os
import pytest
import funROI

@pytest.fixture(autouse=True)
def _reset():
    """
    Ensure singleton state does not leak across tests.
    """
    funROI.reset_settings()
    yield
    funROI.reset_settings()


def test_getters_raise_when_not_set():
    with pytest.raises(RuntimeError, match="bids_data_folder not set"):
        funROI.get_bids_data_folder()

    with pytest.raises(RuntimeError, match="bids_deriv_folder not set"):
        funROI.get_bids_deriv_folder()

    with pytest.raises(RuntimeError, match="bids_preprocessed_folder not set"):
        funROI.get_bids_preprocessed_folder()

    with pytest.raises(RuntimeError, match="analysis_output_folder not set"):
        funROI.get_analysis_output_folder()


def test_set_and_get_bids_folders(tmp_path):
    bids = tmp_path / "bids"
    deriv = tmp_path / "derivatives"
    preproc = tmp_path / "preproc"

    funROI.set_bids_data_folder(bids)
    funROI.set_bids_deriv_folder(deriv)
    funROI.set_bids_preprocessed_folder(preproc)

    assert funROI.get_bids_data_folder() == bids
    assert funROI.get_bids_deriv_folder() == deriv
    assert funROI.get_bids_preprocessed_folder() == preproc

    # ensure_paths should coerce to Path
    assert isinstance(funROI.get_bids_data_folder(), Path)
    assert isinstance(funROI.get_bids_deriv_folder(), Path)
    assert isinstance(funROI.get_bids_preprocessed_folder(), Path)


def test_get_bids_preprocessed_folder_relative(tmp_path):
    bids = tmp_path / "bids"
    preproc = bids / "derivatives" / "fmriprep"

    funROI.set_bids_data_folder(bids)
    funROI.set_bids_preprocessed_folder(preproc)

    rel = funROI.get_bids_preprocessed_folder_relative()
    assert rel == os.path.relpath(preproc, start=bids)


def test_get_bids_preprocessed_folder_relative_requires_both(tmp_path):
    bids = tmp_path / "bids"
    preproc = tmp_path / "preproc"

    funROI.set_bids_data_folder(bids)
    with pytest.raises(RuntimeError, match="bids_preprocessed_folder not set"):
        funROI.get_bids_preprocessed_folder_relative()

    funROI.reset_settings()
    funROI.set_bids_preprocessed_folder(preproc)
    with pytest.raises(RuntimeError, match="bids_data_folder not set"):
        funROI.get_bids_preprocessed_folder_relative()


def test_set_analysis_output_folder_creates_directory(tmp_path):
    out = tmp_path / "analysis_out"
    assert not out.exists()

    funROI.set_analysis_output_folder(out)

    assert out.exists()
    assert out.is_dir()
    assert funROI.get_analysis_output_folder() == out


def test_reset_settings_clears_state(tmp_path):
    funROI.set_bids_data_folder(tmp_path)
    funROI.set_bids_deriv_folder(tmp_path)
    funROI.set_bids_preprocessed_folder(tmp_path)
    funROI.set_analysis_output_folder(tmp_path)

    funROI.reset_settings()

    with pytest.raises(RuntimeError):
        funROI.get_bids_data_folder()
    with pytest.raises(RuntimeError):
        funROI.get_bids_deriv_folder()
    with pytest.raises(RuntimeError):
        funROI.get_bids_preprocessed_folder()
    with pytest.raises(RuntimeError):
        funROI.get_analysis_output_folder()
