"""
funROI: A package for functional region of interest analysis in fMRI data.
"""

from .settings import Settings

_settings = Settings()

set_bids_data_folder = _settings.set_bids_data_folder
"""Set the BIDS data folder path."""

get_bids_data_folder = _settings.get_bids_data_folder
"""Get the BIDS data folder path."""

set_bids_deriv_folder = _settings.set_bids_deriv_folder
"""Set the BIDS derivatives folder path."""

get_bids_deriv_folder = _settings.get_bids_deriv_folder
"""Get the BIDS derivatives folder path."""

set_bids_preprocessed_folder = _settings.set_bids_preprocessed_folder
"""Set the BIDS preprocessed folder path."""

get_bids_preprocessed_folder = _settings.get_bids_preprocessed_folder
"""Get the BIDS preprocessed folder path."""

get_bids_preprocessed_folder_relative = (
    _settings.get_bids_preprocessed_folder_relative
)
"""Get the BIDS preprocessed folder path relative to the data folder."""

set_analysis_output_folder = _settings.set_analysis_output_folder
"""Set the analysis output folder path."""

get_analysis_output_folder = _settings.get_analysis_output_folder
"""Get the analysis output folder path."""

reset_settings = _settings.reset
"""Reset all path settings to None."""

from .first_level import *
from .analysis import *
from .parcels import ParcelsConfig
from .froi import FROIConfig

__all__ = [
    "first_level",
    "analysis",
    "set_bids_data_folder",
    "set_bids_data_folder",
    "get_bids_data_folder",
    "set_bids_deriv_folder",
    "get_bids_deriv_folder",
    "set_bids_preprocessed_folder",
    "get_bids_preprocessed_folder",
    "get_bids_preprocessed_folder_relative",
    "set_analysis_output_folder",
    "get_analysis_output_folder",
    "reset_settings",
    "FROIConfig",
    "ParcelsConfig",
    "datasets",
]
