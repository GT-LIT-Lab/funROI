from .._registry import append_record, get_record_folder
from ..settings import get_analysis_output_folder
import pandas as pd


class AnalysisSaver:
    def __init__(self):
        self._data_summary = None
        self._data_detail = None
        self._type = None

    def _save(self, new_info):
        info_pth = get_analysis_output_folder() / self._type / f"{self._type}_info.csv"
        id = append_record(info_pth, new_info)
        data_folder = get_record_folder(
            get_analysis_output_folder() / self._type, self._type, id
        )
        data_folder.mkdir(parents=True, exist_ok=True)
        if self._data_summary is not None:
            self._data_summary.to_csv(
                data_folder / f"{self._type}_summary.csv", index=False
            )
        if self._data_detail is not None:
            self._data_detail.to_csv(
                data_folder / f"{self._type}_detail.csv", index=False
            )
