import pandas as pd
import pytest

import funROI
from funROI.analysis.utils import AnalysisSaver


@pytest.fixture(autouse=True)
def _reset_and_set_output(tmp_path):
    funROI.reset_settings()
    funROI.set_analysis_output_folder(tmp_path / "analysis_out")
    yield
    funROI.reset_settings()


class DummyAnalysis(AnalysisSaver):
    def __init__(self, typ: str):
        super().__init__()
        self._type = typ


def test_save_creates_info_and_data_folder_and_writes_summary_detail(tmp_path):
    a = DummyAnalysis("laterality")
    a._data_summary = pd.DataFrame(
        {"subject": ["S1"], "laterality_index": [0.1]}
    )
    a._data_detail = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    new_info = pd.DataFrame({"froi": ["cfg0"]})
    a._save(new_info)

    root = funROI.get_analysis_output_folder() / "laterality"

    # info file created with id=0
    info_path = root / "laterality_info.csv"
    assert info_path.exists()
    info = pd.read_csv(info_path)
    assert info.shape[0] == 1
    assert info.loc[0, "id"] == 0
    assert info.loc[0, "froi"] == "cfg0"

    # data folder created
    data_folder = root / "laterality_0000"
    assert data_folder.exists() and data_folder.is_dir()

    # summary & detail written
    assert (data_folder / "laterality_summary.csv").exists()
    assert (data_folder / "laterality_detail.csv").exists()

    # content sanity
    summary = pd.read_csv(data_folder / "laterality_summary.csv")
    detail = pd.read_csv(data_folder / "laterality_detail.csv")
    assert list(summary.columns) == ["subject", "laterality_index"]
    assert detail.shape == (2, 2)


def test_save_increments_id_and_appends_info(tmp_path):
    a = DummyAnalysis("laterality")
    a._data_summary = pd.DataFrame({"subject": ["S1"], "laterality_index": [0.1]})

    root = funROI.get_analysis_output_folder() / "laterality"
    info_path = root / "laterality_info.csv"

    # first save -> id=0
    a._save(pd.DataFrame({"froi": ["cfg0"]}))
    assert info_path.exists()
    info0 = pd.read_csv(info_path)
    assert info0["id"].tolist() == [0]

    # second save -> id=1
    a._data_summary = pd.DataFrame({"subject": ["S2"], "laterality_index": [-0.2]})
    a._data_detail = None  # ensure it doesn't write detail
    a._save(pd.DataFrame({"froi": ["cfg1"]}))

    info1 = pd.read_csv(info_path)
    assert info1.shape[0] == 2
    assert info1["id"].tolist() == [0, 1]
    assert info1["froi"].tolist() == ["cfg0", "cfg1"]

    # second data folder exists and has summary, but no detail
    folder1 = root / "laterality_0001"
    assert folder1.exists()
    assert (folder1 / "laterality_summary.csv").exists()
    assert not (folder1 / "laterality_detail.csv").exists()
