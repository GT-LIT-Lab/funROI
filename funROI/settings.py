import os
from pathlib import Path
from typing import Union
from functools import wraps
from .utils import ensure_paths


def ensure_attribute(attribute: str, set_method: str):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if (
                not hasattr(self, attribute)
                or getattr(self, attribute) is None
            ):
                raise RuntimeError(
                    f"{attribute} not set. Please set it using '{set_method}'."
                )
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class Settings:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
        return cls._instance

    @ensure_paths("path")
    def set_bids_data_folder(self, path: Union[str, Path]):
        self.bids_data_folder = path

    @ensure_attribute("bids_data_folder", "set_bids_data_folder")
    def get_bids_data_folder(self) -> Path:
        return self.bids_data_folder

    @ensure_paths("path")
    def set_bids_deriv_folder(self, path: Union[str, Path]):
        self.bids_deriv_folder = path

    @ensure_attribute("bids_deriv_folder", "set_bids_deriv_folder")
    def get_bids_deriv_folder(self) -> Path:
        return self.bids_deriv_folder

    @ensure_paths("path")
    def set_bids_preprocessed_folder(self, path: Union[str, Path]):
        self.bids_preprocessed_folder = path

    @ensure_attribute(
        "bids_preprocessed_folder", "set_bids_preprocessed_folder"
    )
    def get_bids_preprocessed_folder(self) -> Path:
        return self.bids_preprocessed_folder

    @ensure_attribute("bids_data_folder", "set_bids_data_folder")
    @ensure_attribute(
        "bids_preprocessed_folder", "set_bids_preprocessed_folder"
    )
    def get_bids_preprocessed_folder_relative(self) -> str:
        return os.path.relpath(
            self.bids_preprocessed_folder, start=self.bids_data_folder
        )

    @ensure_paths("path")
    def set_analysis_output_folder(self, path: Union[str, Path]):
        self.analysis_output_folder = path
        if not os.path.exists(path):
            os.makedirs(path)

    @ensure_attribute("analysis_output_folder", "set_analysis_output_folder")
    def get_analysis_output_folder(self) -> Path:
        return self.analysis_output_folder

    def reset(self):
        self.bids_data_folder = None
        self.bids_deriv_folder = None
        self.bids_preprocessed_folder = None
        self.analysis_output_folder = None
