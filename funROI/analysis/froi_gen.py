from typing import List, Optional, Union, Tuple
from .. import get_analysis_output_folder
from ..froi import FROIConfig, _create_froi, _get_froi_path
from ..parcels import get_parcels, ParcelsConfig
import numpy as np
from nibabel.nifti1 import Nifti1Image
from nilearn.image import load_img
import warnings
from pathlib import Path
import pandas as pd


class FROIGenerator:
    """
    Generate fROI maps for a fROI configuration.

    .. warning:: The contrasts and parcels images are assumed to be in the same
        space and have the same dimensions. Future versions may support
        generating fROIs to native space, etc.

    :param subjects: List of subject labels.
    :type subjects: List[str]
    :param froi: fROI configuration.
    :type froi: FROIConfig
    :param run_label: Label of the run to generate the fROIs for. Default is
        "all", where a contrast map for all runs is used to generate the fROI
        for each subject. Alternatively, this can be a specific run label,
        'odd' for odd runs, 'even' for even runs, 'orth<run_label>' for
        all-but-one runs.
    :type run_label: Optional[str]
    """

    def __init__(
        self,
        subjects: List[str],
        froi: FROIConfig,
        run_label: Optional[str] = "all",
    ):
        self.subjects = subjects
        self.froi = froi
        self.run_label = run_label
        self._data = []

    def run(
        self, save: Optional[bool] = True
    ) -> Optional[List[Tuple[str, Nifti1Image]]]:
        """
        Run the fROI generation. The results are stored in the analysis output
        folder.

        :param save: Whether to save the results to the analysis output folder.
        :type save: Optional[bool]
        :return: the results are returned as a list of tuples, where each tuple
            contains the subject label and the FROI map.
        :rtype: Optional[List[Tuple[str, Nifti1Image]]]
        """
        data = []
        for subject in self.subjects:
            froi_pth = _get_froi_path(subject, self.run_label, self.froi)
            if not froi_pth.exists():
                _create_froi(subject, self.froi, self.run_label)
                if not froi_pth.exists():
                    warnings.warn(
                        f"Error generating fROI for subject {subject}"
                    )
                    continue

            froi_img = load_img(froi_pth)

            if save:
                froi_pth = self._get_analysis_froi_path(
                    subject, self.run_label, self.froi, create=True
                )
                froi_img.to_filename(froi_pth)

            data.append((subject, froi_img))

        self.subjects = [dat[0] for dat in data]
        self._data = [dat[1] for dat in data]

        return data

    def select(
        self,
        froi_label: Union[int, str],
        return_results: Optional[bool] = False,
    ) -> Optional[List[Tuple[str, Nifti1Image]]]:
        """
        Select a specific fROI label on the maps. The selected fROI label is
        kept, while all other labels are set to zero. The results are stored in
        the analysis output folder.

        :return: If return_results is True, the results are also returned as a
            list of tuples, where each tuple contains the subject label and the
            filtered FROI map.
        :rtype: Optional[List[Tuple[str, Nifti1Image]]]

        :raises ValueError: If the fROI label is not found in the fROI.
        """
        parcels_img, parcels_labels = get_parcels(self.froi.parcels)
        label_numeric = None
        label_str = None
        for k, v in parcels_labels.items():
            if isinstance(froi_label, int) and k == froi_label:
                label_numeric = k
                label_str = v
                break
            elif isinstance(froi_label, str) and v == froi_label:
                label_numeric = k
                label_str = v
                break
        if label_numeric is None:
            raise ValueError(f"Label {froi_label} not found in parcels labels")

        data = []
        for subject, img in zip(self.subjects, self._data):
            data = img.get_fdata()
            data[data != label_numeric] = 0
            img = Nifti1Image(data, img.affine)
            data.append((subject, img))

            # Save the the output directory
            froi_pth = self._get_analysis_froi_path(
                subject,
                self.run_label,
                self.froi,
                create=True,
                froi_label=label_str,
            )
            img.to_filename(froi_pth)

        if return_results:
            return data

    @staticmethod
    def _get_analysis_froi_folder(task: str) -> Path:
        return get_analysis_output_folder() / f"froi_{task}"

    @classmethod
    def _get_analysis_froi_info_path(cls, task: str) -> Path:
        return cls._get_analysis_froi_folder(task) / "froi_info.csv"

    @classmethod
    def _get_analysis_froi_path(
        cls,
        subject: str,
        run_label: str,
        config: FROIConfig,
        create: Optional[bool] = False,
        froi_label: Optional[str] = None,
    ) -> Path:
        (
            task,
            contrasts,
            conjunction_type,
            threshold_type,
            threshold_value,
            parcels,
        ) = (
            config.task,
            config.contrasts,
            config.conjunction_type,
            config.threshold_type,
            config.threshold_value,
            config.parcels,
        )

        if not isinstance(parcels, ParcelsConfig):
            parcels = ParcelsConfig(parcels)
        contrasts = str(sorted(contrasts))

        frois_new = pd.DataFrame(
            {
                "contrasts": [contrasts],
                "conjunction_type": [conjunction_type],
                "threshold_type": [threshold_type],
                "threshold_value": [threshold_value],
                "parcels": [parcels.parcels_path],
                "labels": [parcels.labels_path],
            }
        )

        froi_info_pth = cls._get_analysis_froi_info_path(config.task)
        if not froi_info_pth.exists():
            id = 0
            if create:
                frois_new["id"] = id
                froi_info_pth.parent.mkdir(parents=True, exist_ok=True)
                frois_new.to_csv(froi_info_pth, index=False)
        else:
            frois = pd.read_csv(froi_info_pth)
            frois_matched = frois[
                frois.apply(
                    lambda row: (
                        (row["contrasts"] == contrasts)
                        and (
                            (row["conjunction_type"] == conjunction_type)
                            or (
                                pd.isna(row["conjunction_type"])
                                and conjunction_type is None
                            )
                        )
                        and (row["threshold_type"] == threshold_type)
                        and (row["threshold_value"] == threshold_value)
                        and (
                            (row["parcels"] == str(parcels.parcels_path))
                            or (
                                pd.isna(row["parcels"])
                                and parcels.parcels_path is None
                            )
                        )
                        and (
                            (row["labels"] == str(parcels.labels_path))
                            or (
                                pd.isna(row["labels"])
                                and parcels.labels_path is None
                            )
                        )
                    ),
                    axis=1,
                )
            ]
            if not frois_matched.empty:
                id = frois_matched["id"].values[0]
            else:
                id = frois["id"].max() + 1
                if create:
                    frois_new["id"] = id
                    frois_new = pd.concat(
                        [frois, frois_new], ignore_index=True
                    )
                    frois_new.to_csv(froi_info_pth, index=False)

        id = f"{id:04d}"
        froi_folder = cls._get_analysis_froi_folder(task) / f"froi_{id}"
        froi_folder.mkdir(parents=True, exist_ok=True)

        if froi_label is None:
            return froi_folder / f"sub-{subject}_run-{run_label}_froi.nii.gz"
        else:
            return (
                froi_folder
                / f"sub-{subject}_run-{run_label}_label-{froi_label}.nii.gz"
            )
