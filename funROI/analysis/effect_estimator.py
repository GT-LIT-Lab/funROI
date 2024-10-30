import numpy as np
from typing import List, Tuple, Optional
from ..utils import (
    FROIConfig,
    get_next_effect_estimation_paths,
    get_effect_estimation_folder,
)
from ..contrast import _get_contrast_all, _get_contrast_run, _check_orthogonal
from ..froi import _get_froi_all, _get_froi_orth
from ..parcels import get_parcels
import pandas as pd
import warnings
import json
import os


class FROIEffectEstimator:
    def __init__(
        self,
        subjects: List[str],
        froi: FROIConfig,
        fill_na_with_zero: Optional[bool] = True,
    ):
        """
        Initialize the fROI effect estimator.

        Parameters
        ----------
        subjects : List[str]
            List of subject IDs.
        froi : FROIConfig
            The configuration to define the fROIs.
        fill_na_with_zero : Optional[bool], default=True
            Whether to fill NaN values with zero. If False, NaN values will be
            ignored.
        """
        self.subjects = subjects
        self.froi = froi
        self.fill_na_with_zero = fill_na_with_zero

    def run(
        self,
        effects: List[Tuple[str, List[str]]],
        return_output: Optional[bool] = False,
    ) -> pd.DataFrame:
        """
        Compute the effect size of fROIs.

        Parameters
        ----------
        effects : List[Tuple[str, List[str]]]
            List of (task, contrasts) tuples.

        Returns
        -------
        effect_size : pd.DataFrame,
            columns=[
                'subject', 'froi_size', 'froi', 'task',
                'contrast', 'effect_size'
            ]
            The effect size of fROIs.
        """
        _, parcels_labels = get_parcels(self.froi.parcels)

        effect_size_all_subjects = []
        for subject in self.subjects:
            try:
                froi_mask_all = _get_froi_all(subject, self.froi)
                froi_mask_orth = _get_froi_orth(subject, self.froi)
            except ValueError:
                warnings.warn(
                    f"All- or orthogonal-run fROI not found for subject "
                    f"{subject} and task {self.froi.task}. Skipping."
                )
                continue

            for task, contrasts in effects:
                for contrast in contrasts:
                    if not _check_orthogonal(
                        subject,
                        self.froi.task,
                        self.froi.contrasts,
                        task,
                        [contrast],
                    ):
                        try:
                            effect_data = _get_contrast_run(
                                subject, task, contrast, "effect"
                            )
                        except ValueError:
                            warnings.warn(
                                f"Contrast {contrast} not found for subject {subject} "
                                f"and task {task}. Skipping."
                            )
                            continue

                        froi_mask = froi_mask_orth

                    else:
                        try:
                            effect_data = _get_contrast_all(
                                subject, task, contrast, "effect"
                            )
                        except ValueError:
                            warnings.warn(
                                f"Contrast {contrast} not found for subject {subject} "
                                f"and task {task}. Skipping."
                            )
                            continue

                        froi_mask = froi_mask_all

                    # Trim voxels not labeled
                    non_zero_voxels = np.sum(froi_mask != 0, axis=0) > 0
                    froi_mask = froi_mask[:, non_zero_voxels]
                    effect_data = effect_data[:, non_zero_voxels]

                    eff = self._compute_effect_size(
                        effect_data[None, :, :],
                        froi_mask[None, :, :],
                        self.fill_na_with_zero,
                    )

                    effect_size = pd.DataFrame(
                        {
                            "subject": subject,
                            "froi_size": eff["localizer_size"],
                            "froi": eff["froi_id"].map(
                                lambda x: parcels_labels[x]
                            ),
                            "task": task,
                            "contrast": contrast,
                            "effect_size": eff["effect_size"],
                        }
                    )
                    effect_size_all_subjects.append(effect_size)

        effect_size_all_subjects = pd.concat(effect_size_all_subjects)

        effect_folder = get_effect_estimation_folder()
        if not os.path.exists(effect_folder):
            os.makedirs(effect_folder)

        output_path, config_path = get_next_effect_estimation_paths()
        effect_size_all_subjects.to_csv(output_path, index=False)
        config = {
            "subjects": self.subjects,
            "froi": self.froi,
            "effects": effects,
            "fill_na_with_zero": self.fill_na_with_zero,
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        if return_output:
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
