import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union
from ..froi import (
    FROI,
    _get_frois_all_multi_task,
    _get_frois_orth_single_task,
    _get_frois_runs_single_task,
)
from ..utils import get_parcels_labels, get_parcels


class OverlapEstimator:
    def __init__(
        self,
        task_frois: Union[List[Tuple[str, List[FROI]]], List[str]],
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
        task_frois : Union[List[Tuple[str, List[FROI]]], List[str]]
            List of (task, frois) tuples, or a list of parcels names.
        subjects : Optional[List[str]]
            The list of subjects to analyze.
        kind : str, optional
            The kind of overlap to compute. It can be 'dice', or 'overlap', by
            default 'overlap'.
            - 'dice' : Dice coefficient: 2 * |A & B| / (|A| + |B|)
            - 'overlap' : Overlap: |A & B| / min(|A|, |B|)
        """
        self.subjects = subjects
        self.task_frois = task_frois
        self.kind = kind

        if np.all([isinstance(task_froi, str) for task_froi in task_frois]):
            self.subjects = [None]  # dummy subject

    def compute_overlap(self) -> pd.DataFrame:
        """
        Compute the overlap between fROIs.

        Returns
        -------
        overlap : pd.DataFrame, columns=[
                'task1', 'task2', 'localizer1',
                'localizer2', 'froi1', 'froi2', 'overlap'
            ]
            The overlap between each pair of fROIs across networks.
        """
        overlap_all_subjects = []

        parcels_labels = {None: {}}

        flattened_localizer_tasks = []
        flattened_localizer_names = []
        for i, task_froi in enumerate(self.task_frois):
            if isinstance(task_froi, str):
                _, labels = get_parcels_labels(task_froi)
                parcels_labels[None][task_froi] = labels
                flattened_localizer_tasks.append(None)
                flattened_localizer_names.append(task_froi)
            else:
                task, frois = task_froi
                parcels_labels[task] = {}
                for froi in frois:
                    _, labels = get_parcels_labels(froi.parcels)
                    parcels_labels[task][froi.localizer] = labels
                    flattened_localizer_tasks.append(task)
                    flattened_localizer_names.append(froi.localizer)

        for subject in self.subjects:
            # All x All
            frois = []
            for task_froi_i, task_froi in enumerate(self.task_frois):
                if isinstance(task_froi, str):
                    parcels_img = get_parcels(task_froi)
                    parcels_flattened = parcels_img.get_fdata().flatten()
                    frois.append(parcels_flattened[np.newaxis, np.newaxis, :])
                else:
                    frois.append(
                        _get_frois_all_multi_task(subject, [task_froi])
                    )
            frois = np.concatenate(frois, axis=0)

            count_i = 0
            overlap_diff_task = []
            for task_froi_ii, task_froi_i in enumerate(self.task_frois):
                n_iter_i = (
                    1 if isinstance(task_froi_i, str) else len(task_froi_i[1])
                )
                count_j = count_i + n_iter_i
                for task_froi_jj, task_froi_j in enumerate(
                    self.task_frois[task_froi_ii + 1 :]
                ):
                    n_iter_j = (
                        1
                        if isinstance(task_froi_j, str)
                        else len(task_froi_j[1])
                    )
                    for froi_iter_i in range(count_i, count_i + n_iter_i):
                        for froi_iter_j in range(count_j, count_j + n_iter_j):
                            overlap_results = self._compute_overlap(
                                frois[froi_iter_i],
                                frois[froi_iter_j],
                                self.kind,
                            )
                            overlap_results["task1"] = (
                                flattened_localizer_tasks[froi_iter_i]
                            )
                            overlap_results["task2"] = (
                                flattened_localizer_tasks[froi_iter_j]
                            )
                            overlap_results["localizer1"] = (
                                flattened_localizer_names[froi_iter_i]
                            )
                            overlap_results["localizer2"] = (
                                flattened_localizer_names[froi_iter_j]
                            )
                            overlap_diff_task_i = pd.DataFrame(
                                {
                                    "task1": overlap_results["task1"],
                                    "task2": overlap_results["task2"],
                                    "localizer1": overlap_results[
                                        "localizer1"
                                    ],
                                    "localizer2": overlap_results[
                                        "localizer2"
                                    ],
                                    "froi1": overlap_results.apply(
                                        lambda x: parcels_labels[x["task1"]][
                                            x["localizer1"]
                                        ][int(x["froi_id1"])],
                                        axis=1,
                                    ),
                                    "froi2": overlap_results.apply(
                                        lambda x: parcels_labels[x["task2"]][
                                            x["localizer2"]
                                        ][int(x["froi_id2"])],
                                        axis=1,
                                    ),
                                    "overlap": overlap_results["overlap"],
                                }
                            )
                            overlap_diff_task.append(overlap_diff_task_i)

                    count_j += n_iter_j
                count_i += n_iter_i
            overlap_diff_task = pd.concat(overlap_diff_task)

            # Same-task condition
            overlap_same_task = []
            for task_froi in self.task_frois:
                if isinstance(task_froi, str):
                    continue
                task, frois = task_froi

                frois_runs = _get_frois_runs_single_task(subject, task, frois)
                frois_orth = _get_frois_orth_single_task(subject, task, frois)

                for i in range(len(frois_runs)):
                    for j in range(len(frois_runs)):
                        overlap_results = self._compute_overlap(
                            frois_runs[i], frois_orth[j], self.kind
                        )
                        overlap_results["task1"] = task
                        overlap_results["task2"] = task
                        overlap_results["localizer1"] = frois[i].localizer
                        overlap_results["localizer2"] = frois[j].localizer
                        overlap_same_task_i = pd.DataFrame(
                            {
                                "task1": overlap_results["task1"],
                                "task2": overlap_results["task2"],
                                "localizer1": overlap_results["localizer1"],
                                "localizer2": overlap_results["localizer2"],
                                "froi1": overlap_results.apply(
                                    lambda x: parcels_labels[x["task1"]][
                                        x["localizer1"]
                                    ][int(x["froi_id1"])],
                                    axis=1,
                                ),
                                "froi2": overlap_results.apply(
                                    lambda x: parcels_labels[x["task2"]][
                                        x["localizer2"]
                                    ][int(x["froi_id2"])],
                                    axis=1,
                                ),
                                "overlap": overlap_results["overlap"],
                            }
                        )
                        overlap_same_task.append(overlap_same_task_i)

            if len(overlap_same_task) > 0:
                overlap_same_task = pd.concat(overlap_same_task)
                overlap_all = pd.concat([overlap_diff_task, overlap_same_task])
            else:
                overlap_all = overlap_diff_task

            overlap_all.reset_index(inplace=True, drop=True)
            overlap_all["subject"] = subject
            overlap_all_subjects.append(overlap_all)

        overlap_all_subjects = pd.concat(overlap_all_subjects)
        return overlap_all_subjects

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
