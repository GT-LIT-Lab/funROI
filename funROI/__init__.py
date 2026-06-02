"""
funROI: A package for functional region of interest analysis in fMRI data.
"""

from . import analysis, datasets, first_level
from .analysis import (
    EffectEstimator,
    FunctionalConnectivityEstimator,
    FROIGenerator,
    LateralityIndexAnalyzer,
    load_preprocessed_bold_for_fc,
    OverlapEstimator,
    ParcelsGenerator,
    preprocess_bold_for_fc,
    SurfaceParcelsGenerator,
    SpatialCorrelationEstimator,
)
from .froi import FROIConfig
from .parcels import ParcelsConfig, SurfaceParcelsConfig
from .settings import (
    Settings,
    get_analysis_output_folder,
    get_bids_data_folder,
    get_bids_deriv_folder,
    get_bids_preprocessed_folder,
    get_bids_preprocessed_folder_relative,
    reset_settings,
    set_analysis_output_folder,
    set_bids_data_folder,
    set_bids_deriv_folder,
    set_bids_preprocessed_folder,
)

__all__ = [
    "Settings",
    "analysis",
    "datasets",
    "first_level",
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
    "ParcelsConfig",
    "SurfaceParcelsConfig",
    "FROIConfig",
    "ParcelsGenerator",
    "SurfaceParcelsGenerator",
    "FROIGenerator",
    "EffectEstimator",
    "FunctionalConnectivityEstimator",
    "preprocess_bold_for_fc",
    "load_preprocessed_bold_for_fc",
    "SpatialCorrelationEstimator",
    "OverlapEstimator",
    "LateralityIndexAnalyzer",
]
