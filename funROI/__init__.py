from .settings import Settings

# Create a settings instance
_settings = Settings()


def set_bids_deriv_folder(path):
    _settings.set_bids_deriv_folder(path)


def get_bids_deriv_folder():
    return _settings.get_bids_deriv_folder()


def set_bids_data_folder(path):
    _settings.set_bids_data_folder(path)


def get_bids_data_folder():
    return _settings.get_bids_data_folder()


def set_bids_preprocessed_folder(path):
    _settings.set_bids_preprocessed_folder(path)


def get_bids_preprocessed_folder():
    return _settings.get_bids_preprocessed_folder()


def get_bids_preprocessed_folder_relative():
    return _settings.get_bids_preprocessed_folder_relative()


from .first_level import *
from .analysis import *

__all__ = ["first_level", "analysis"]
