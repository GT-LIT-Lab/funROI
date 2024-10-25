import os


class Settings:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance.bids_data_folder = None
        return cls._instance

    def set_bids_deriv_folder(self, path):
        self.bids_deriv_folder = path

    def get_bids_deriv_folder(self):
        if not self.bids_deriv_folder:
            raise ValueError(
                (
                    "BIDS derivatives folder not set. "
                    "Please set it using 'set_bids_deriv_folder'."
                )
            )
        return self.bids_deriv_folder

    def set_bids_data_folder(self, path):
        self.bids_data_folder = path

    def get_bids_data_folder(self):
        if not self.bids_data_folder:
            raise ValueError(
                (
                    "BIDS data folder not set. "
                    "Please set it using 'set_bids_data_folder'."
                )
            )
        return self.bids_data_folder

    def set_bids_preprocessed_folder(self, path):
        self.bids_preprocessed_folder = path

    def get_bids_preprocessed_folder(self):
        if not self.bids_preprocessed_folder:
            raise ValueError(
                (
                    "BIDS preprocessed folder not set. "
                    "Please set it using 'set_bids_preprocessed_folder'."
                )
            )
        return self.bids_preprocessed_folder

    def get_bids_preprocessed_folder_relative(self):
        return os.path.relpath(
            self.bids_preprocessed_folder, start=self.bids_data_folder
        )
