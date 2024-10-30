import numpy as np
import pandas as pd
from typing import Optional, List, Union
from ..froi import _get_froi_all, _get_froi_orth, _get_froi_run
from ..parcels import get_parcels
from ..contrast import _check_orthogonal
from ..utils import (
    FROIConfig,
    ParcelsConfig,
    get_overlap_estimation_folder,
    get_next_overlap_estimation_path,
)
import warnings
import os


class OverlapEstimator:
    def __init__(
        self,
        frois: List[Union[FROIConfig, str, ParcelsConfig]],
        subjects: Optional[List[str]] = None,
        kind: Optional[str] = "overlap",
    ):
        """
        Initialize the fROI overlap analyzer.
        For now, it is assumed that no duplicate localizer names exist
        across fROIs. Localizer name are used as identifiers in the output
        DataFrame, see `compute_overlap`.

        Parameters
        ----------
        frois : List[Union[FROI, str, ParcelsConfig]]
            The fROIs to analyze. Each fROI can be a FROI object, a saved
            parcels name, or a ParcelsConfig object indicating the parcels
            and labels paths.
        subjects : Optional[List[str]]
            The list of subjects to analyze. It is required if any fROI is a
            FROI object, by default None.
        kind : str, optional
            The kind of overlap to compute. It can be 'dice', or 'overlap', by
            default 'overlap'.
            - 'dice' : Dice coefficient: 2 * |A & B| / (|A| + |B|)
            - 'overlap' : Overlap: |A & B| / min(|A|, |B|)
        """
        self.subjects = subjects
        self.frois = frois
        self.kind = kind

        any_froi = any([isinstance(froi, FROIConfig) for froi in frois])
        if any_froi:
            if subjects is None:
                raise ValueError("subjects must be provided for FROI objects")
        else:
            self.subjects = [None]

    def run(self, return_output: Optional[bool] = False) -> pd.DataFrame:
        """
        Compute the overlap between fROIs.

        Returns
        -------
        overlap : pd.DataFrame, columns=[
                'subject', 'group1', 'group2', 'froi1', 'froi2', 'overlap'
            ]
            The overlap between each pair of fROIs across networks.
        """
        overlap_all_subjects = []
        for subject in self.subjects:
            frois_avail = []
            frois_data_avail = {"all": [], "orth": [], "run": []}
            frois_labels_avail = []
            for froi in self.frois:
                try:
                    if isinstance(froi, FROIConfig):
                        froi_dat = _get_froi_all(subject, froi)
                        froi_dat_orth = _get_froi_orth(subject, froi)
                        froi_dat_run = _get_froi_run(subject, froi)
                        frois_avail.append(froi)
                        frois_data_avail["all"].append(froi_dat)
                        frois_data_avail["orth"].append(froi_dat_orth)
                        frois_data_avail["run"].append(froi_dat_run)
                        _, parcels_labels = get_parcels(
                            froi.parcels
                        )  # TODO: decide whether to backtrack labels from parcels, or save labels with fROI
                        frois_labels_avail.append(parcels_labels)
                    else:
                        parcels_img_all, parcels_labels_all = get_parcels(froi)
                        frois_avail.append(froi)
                        frois_data_avail["all"].append(
                            parcels_img_all.get_fdata().flatten()[None, :]
                        )
                        frois_data_avail["orth"].append(None)
                        frois_data_avail["run"].append(None)
                        frois_labels_avail.append(parcels_labels_all)
                except ValueError:
                    warnings.warn(
                        f"Subject {subject} does not have the following fROI, "
                        f"skipping: {froi}"
                    )
                    continue

            for i, froi1 in enumerate(frois_avail):
                for j, froi2 in enumerate(
                    frois_avail
                ):  # TODO: overlap is also not symmetric, see the notes in spcorr.py
                    if not isinstance(froi1, FROIConfig) or not isinstance(
                        froi2, FROIConfig
                    ):
                        okorth = True
                    else:
                        okorth = _check_orthogonal(
                            subject,
                            froi1.task,
                            froi1.contrasts,
                            froi2.task,
                            froi2.contrasts,
                        )
                    if okorth:
                        froi1_data = frois_data_avail["all"][i]
                        froi2_data = frois_data_avail["all"][j]
                    else:
                        froi1_data = frois_data_avail["orth"][i]
                        froi2_data = frois_data_avail["run"][j]

                    # Trim voxels not labeled in both fROIs
                    non_zero_1 = np.sum(froi1_data != 0, axis=0) > 0
                    non_zero_2 = np.sum(froi2_data != 0, axis=0) > 0
                    non_zero_either = non_zero_1 | non_zero_2
                    froi1_data = froi1_data[:, non_zero_either]
                    froi2_data = froi2_data[:, non_zero_either]

                    overlap = self._compute_overlap(
                        froi1_data, froi2_data, kind=self.kind
                    )
                    overlap_estimates = pd.DataFrame(
                        {
                            "subject": [subject] * len(overlap),
                            "group1": [i] * len(overlap),
                            "group2": [j] * len(overlap),
                            "froi1": overlap["froi_id1"].map(
                                lambda x: frois_labels_avail[i][x]
                            ),
                            "froi2": overlap["froi_id2"].map(
                                lambda x: frois_labels_avail[j][x]
                            ),
                            "overlap": overlap["overlap"],
                        }
                    )
                    overlap_all_subjects.append(overlap_estimates)

        overlap_all_subjects = pd.concat(overlap_all_subjects)

        # Form a dataframe for available fROIs
        frois_avail_df = pd.DataFrame(
            {
                "group": list(range(len(frois_avail))),
                "task": [
                    froi.task if isinstance(froi, FROIConfig) else None
                    for froi in frois_avail
                ],
                "contrasts": [
                    froi.contrasts if isinstance(froi, FROIConfig) else None
                    for froi in frois_avail
                ],
                "conjunction_type": [
                    (
                        froi.conjunction_type
                        if isinstance(froi, FROIConfig)
                        else None
                    )
                    for froi in frois_avail
                ],
                "threshold_type": [
                    (
                        froi.threshold_type
                        if isinstance(froi, FROIConfig)
                        else None
                    )
                    for froi in frois_avail
                ],
                "threshold_value": [
                    (
                        froi.threshold_value
                        if isinstance(froi, FROIConfig)
                        else None
                    )
                    for froi in frois_avail
                ],
                "parcels": [
                    froi.parcels if isinstance(froi, FROIConfig) else froi
                    for froi in frois_avail
                ],
            }
        )

        overlap_folder = get_overlap_estimation_folder()
        if not os.path.exists(overlap_folder):
            os.makedirs(overlap_folder)

        overlap_path, config_path = get_next_overlap_estimation_path()
        overlap_all_subjects.to_csv(overlap_path, index=False)
        frois_avail_df.to_csv(config_path, index=False)

        if return_output:
            return overlap_all_subjects, frois_avail_df

    @classmethod
    def _compute_overlap(
        cls,
        froi_masks1: np.ndarray,
        froi_masks2: np.ndarray,
        kind: Optional[str] = "overlap",
    ) -> pd.DataFrame:
        """
        Compute the overlap between fROIs.

        Parameters
        ----------
        froi_masks1 : np.ndarray, shape (n_runs, n_voxels)
            The fROI data, coded by numerical labels (0 for background).
        froi_masks2 : np.ndarray, shape (n_runs, n_voxels)
            The fROI data, coded by numerical labels (0 for background).
        kind : str, optional
            The kind of overlap to compute. It can be 'dice', or 'overlap', by
            default 'overlap'.
            - 'dice' : Dice coefficient: 2 * |A & B| / (|A| + |B|)
            - 'overlap' : Overlap: |A & B| / min(|A|, |B|)

        Returns
        -------
        overlap : pd.DataFrame, columns=['froi_id1', 'froi_id2', 'overlap']
            The overlap between each pair of fROIs across the masks.
        """
        assert kind in [
            "dice",
            "overlap",
        ], "kind should be 'dice' or 'overlap'"
        froi_masks1[np.isnan(froi_masks1)] = 0
        froi_masks2[np.isnan(froi_masks2)] = 0

        labels_i = np.unique(froi_masks1)
        labels_i = labels_i[labels_i != 0]
        labels_j = np.unique(froi_masks2)
        labels_j = labels_j[labels_j != 0]
        overlap_results = []  # "froi_id1", "froi_id2", "overlap"
        for label_i in labels_i:
            froi_masks2_on_label_i = froi_masks2 * (froi_masks1 == label_i)
            count_ij = []
            for label_j in labels_j:
                count_ij.append(np.sum(froi_masks2_on_label_i == label_j))
            overlap_results.append(
                {
                    "froi_id1": [label_i] * len(labels_j),
                    "froi_id2": labels_j,
                    "overlap_n": count_ij,
                }
            )
        overlap_results = pd.concat(
            [
                pd.DataFrame(overlap_result)
                for overlap_result in overlap_results
            ]
        )

        label_i_stats = {
            label_i: np.sum(froi_masks1 == label_i) for label_i in labels_i
        }
        overlap_results["n1"] = overlap_results["froi_id1"].apply(
            lambda x: label_i_stats[x]
        )
        label_j_stats = {
            label_j: np.sum(froi_masks2 == label_j) for label_j in labels_j
        }
        overlap_results["n2"] = overlap_results["froi_id2"].apply(
            lambda x: label_j_stats[x]
        )

        if kind == "dice":
            overlap_results["overlap"] = (2 * overlap_results["overlap_n"]) / (
                overlap_results["n1"] + overlap_results["n2"]
            )
        elif kind == "overlap":
            overlap_results["overlap"] = overlap_results[
                "overlap_n"
            ] / np.minimum(overlap_results["n1"], overlap_results["n2"])

        return overlap_results
