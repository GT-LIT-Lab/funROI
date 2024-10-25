import numpy as np
from typing import List, Tuple, Optional
from ..froi import FROI
from ..contrast import (
    get_contrasts_all_multi_task,
    get_contrasts_runs_single_task,
    flatten_task_contrasts,
)
from ..froi import (
    _get_frois_all_multi_task,
    _get_frois_orth_single_task,
    flatten_task_frois_labels,
)
from ..utils import get_parcels_labels
import pandas as pd


class FROIEffectEstimator:
    def __init__(
        self,
        subjects: List[str],
        task_frois: List[Tuple[str, List[FROI]]],
        fill_na_with_zero: Optional[bool] = True,
    ):
        """
        Initialize the fROI effect estimator.

        Parameters
        ----------
        subjects : List[str]
            List of subject IDs.
        task_frois : List[Tuple[str, List[FROI]]]
            List of (task, frois) tuples.
        fill_na_with_zero : Optional[bool], default=False
            Whether to fill NaN values with zero. If False, NaN values will be
            ignored.
        """
        self.subjects = subjects
        self.task_frois = task_frois
        self.fill_na_with_zero = fill_na_with_zero

    def compute_effect_size(
        self, task_contrasts: List[Tuple[str, List[str]]]
    ) -> pd.DataFrame:
        """
        Compute the effect size of fROIs.

        Parameters
        ----------
        task_contrasts : List[Tuple[str, List[str]]]
            List of (task, contrasts) tuples.

        Returns
        -------
        effect_size : pd.DataFrame,
            columns=[
                'subject', 'localizer_task', 'localizer_name',
                'localizer_size', 'froi', 'effect_task',
                'effect_contrast', 'effect_size'
            ]
            The effect size of fROIs.
        """
        parcels_labels = {}
        for task, frois in self.task_frois:
            parcels_labels[task] = {}
            for froi in frois:
                _, parcels_labels[task][froi.localizer] = get_parcels_labels(
                    froi.parcels
                )

        localizer_tasks_flattened, localizer_names_flattened = (
            flatten_task_frois_labels(self.task_frois)
        )
        effect_tasks_flattened, effect_names_flattened = (
            flatten_task_contrasts(task_contrasts)
        )

        effect_size_all_subjects = []
        for subject in self.subjects:
            # All x All
            froi_mask = _get_frois_all_multi_task(
                subject,
                self.task_frois,
            )

            effect_data = get_contrasts_all_multi_task(
                subject, task_contrasts, "effect"
            )
            eff = self._compute_effect_size(
                effect_data, froi_mask, self.fill_na_with_zero
            )
            effect_size_all = pd.DataFrame(
                {
                    "localizer_task": eff["localizer_id"].map(
                        lambda x: localizer_tasks_flattened[x]
                    ),
                    "localizer_name": eff["localizer_id"].map(
                        lambda x: localizer_names_flattened[x]
                    ),
                    "localizer_size": eff["localizer_size"],
                    "effect_task": eff["contrast_id"].map(
                        lambda x: effect_tasks_flattened[x]
                    ),
                    "effect_contrast": eff["contrast_id"].map(
                        lambda x: effect_names_flattened[x]
                    ),
                    "effect_size": eff["effect_size"],
                }
            )
            effect_size_all["froi"] = eff.apply(
                lambda row: parcels_labels[
                    localizer_tasks_flattened[int(row["localizer_id"])]
                ][localizer_names_flattened[int(row["localizer_id"])]][
                    int(row["froi_id"])
                ],
                axis=1,
            )

            # Same-task condition
            effect_size_same_task = []
            for localizer_task, frois in self.task_frois:
                if localizer_task not in effect_tasks_flattened:
                    continue
                matched = np.where(
                    np.array(effect_tasks_flattened) == localizer_task
                )[0]
                effect_names_flattened_matched = [
                    effect_names_flattened[i] for i in matched
                ]
                effect_data_same_task = get_contrasts_runs_single_task(
                    subject,
                    localizer_task,
                    effect_names_flattened_matched,
                    "effect",
                )
                froi_mask_same_task = _get_frois_orth_single_task(
                    subject, localizer_task, frois
                )
                eff = self._compute_effect_size(
                    effect_data_same_task,
                    froi_mask_same_task,
                    self.fill_na_with_zero,
                )
                effect_size_same_task_i_renamed = pd.DataFrame(
                    {
                        "localizer_task": [localizer_task] * len(eff),
                        "localizer_name": eff["localizer_id"].map(
                            lambda x: frois[x].localizer
                        ),
                        "localizer_size": eff["localizer_size"],
                        "effect_task": [localizer_task] * len(eff),
                        "effect_contrast": eff["contrast_id"].map(
                            lambda x: effect_names_flattened_matched[x]
                        ),
                        "effect_size": eff["effect_size"],
                    }
                )
                effect_size_same_task_i_renamed["froi"] = eff.apply(
                    lambda row: parcels_labels[localizer_task][
                        frois[int(row["localizer_id"])].localizer
                    ][int(row["froi_id"])],
                    axis=1,
                )
                effect_size_same_task.append(effect_size_same_task_i_renamed)
            effect_size_same_task = pd.concat(effect_size_same_task)

            effect_size_all.set_index(
                [
                    "localizer_task",
                    "localizer_name",
                    "froi",
                    "effect_task",
                    "effect_contrast",
                ],
                inplace=True,
            )
            effect_size_same_task.set_index(
                [
                    "localizer_task",
                    "localizer_name",
                    "froi",
                    "effect_task",
                    "effect_contrast",
                ],
                inplace=True,
            )
            effect_size_all.update(effect_size_same_task)
            effect_size_all.reset_index(inplace=True)
            effect_size_all["subject"] = subject
            effect_size_all_subjects.append(effect_size_all)

        effect_size_all_subjects = pd.concat(effect_size_all_subjects)
        effect_size_all_subjects = effect_size_all_subjects[
            ["subject"] + effect_size_all_subjects.columns[:-1].tolist()
        ]
        return effect_size_all_subjects

    @classmethod
    def _compute_effect_size(
        cls,
        effect_data: np.ndarray,
        froi_masks: np.ndarray,
        fill_na_with_zero: bool,
    ) -> np.ndarray:
        """
        Compute the effect size of fROIs.

        Parameters
        ----------
        effect_data : np.ndarray, shape (n_contrasts, n_runs, n_voxels)
            The effect data.
        froi_masks : np.ndarray, shape (n_frois, n_runs, n_voxels)
            The fROI data.
        fill_na_with_zero : bool
            Whether to fill NaN values with zero. If False, NaN values will be
            ignored.

        Returns
        -------
        effect_size : pd.DataFrame,
            columns=[
                'localizer_id', 'froi_id', 'contrast_id', 'effect_size',
                'localizer_size'
            ]
            The effect size of fROIs.
        """
        assert (
            len(effect_data.shape) == 3
        ), "effect_data should have shape (n_contrasts, n_runs, n_voxels)"
        assert (
            len(froi_masks.shape) == 3
        ), "froi_masks should have shape (n_frois, n_runs, n_voxels)"
        assert (
            effect_data.shape[1] == froi_masks.shape[1]
        ), "effect_data and froi_masks should have the same number of runs"
        assert (
            effect_data.shape[2] == froi_masks.shape[2]
        ), "effect_data and froi_masks should have the same number of voxels"

        effect_data_expanded = effect_data[np.newaxis, :, :, :]
        froi_effect_size = []

        # Iterate over fROIs
        for froi_mask_i, froi_mask in enumerate(froi_masks):
            labels = np.unique(froi_mask)
            labels = labels[labels != 0]
            froi_mask_expanded = (froi_mask == labels[:, None, None]).astype(
                float
            )  # n_froi_labels x n_runs x n_voxels
            froi_mask_expanded[froi_mask_expanded == 0] = np.nan
            masked_effect_data = (
                effect_data_expanded * froi_mask_expanded[:, np.newaxis, :, :]
            )  # n_froi_labels x n_contrasts x n_runs x n_voxels

            if fill_na_with_zero:
                in_mask_nan = np.isnan(masked_effect_data) * (
                    ~np.isnan(froi_mask_expanded[:, np.newaxis, :, :])
                )
                masked_effect_data[in_mask_nan] = 0

            effect_size = np.nanmean(masked_effect_data, axis=(2, 3))
            n_froi_labels, n_contrasts = effect_size.shape
            localizer_sizes = np.mean(
                np.sum(~np.isnan(froi_mask_expanded), axis=-1), axis=-1
            )
            froi_effect_size_i = pd.DataFrame(
                {
                    "froi_id": np.repeat(labels, n_contrasts),
                    "contrast_id": np.tile(
                        np.arange(n_contrasts), n_froi_labels
                    ),
                    "effect_size": effect_size.flatten(),
                    "localizer_size": np.repeat(localizer_sizes, n_contrasts),
                }
            )
            froi_effect_size_i["localizer_id"] = froi_mask_i
            froi_effect_size.append(froi_effect_size_i)

        froi_effect_size = pd.concat(froi_effect_size)
        return froi_effect_size
