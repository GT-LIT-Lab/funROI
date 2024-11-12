from .. import get_analysis_output_folder
from pathlib import Path
import pandas as pd


class AnalysisSaver:
    def __init__(self):
        self._data_summary = None
        self._data_detail = None
        self._type = None

    def _save(self, new_info):
        info_pth = (
            get_analysis_output_folder()
            / self._type
            / f"{self._type}_info.csv"
        )
        info_pth.parent.mkdir(parents=True, exist_ok=True)

        if not info_pth.exists():
            id = 0
            new_info["id"] = id
            new_info.to_csv(info_pth, index=False)
        else:
            info = pd.read_csv(info_pth)
            id = info["id"].max() + 1
            new_info["id"] = id
            info = pd.concat([info, new_info])
            info.to_csv(info_pth, index=False)

        data_folder = (
            get_analysis_output_folder()
            / self._type
            / f"{self._type}_{id:04d}"
        )
        data_folder.mkdir(parents=True, exist_ok=True)
        self._data_summary.to_csv(
            data_folder / f"{self._type}_summary.csv", index=False
        )
        self._data_detail.to_csv(
            data_folder / f"{self._type}_detail.csv", index=False
        )
