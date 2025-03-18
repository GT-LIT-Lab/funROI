from typing import List, Optional, Tuple
from ..froi import FROIConfig, _get_orthogonalized_froi_data, _get_froi_data
from ..contrast import (
    _get_orthogonalized_contrast_data,
    _get_contrast_data,
    _check_orthogonal,
)
from ..parcels import get_parcels
import pandas as pd
from ..utils import validate_arguments
import numpy as np
import logging
import warnings
from .utils import AnalysisSaver


class EffectEstimator(AnalysisSaver):
    """
    Estimate the effect of a ROI on the data.

    :param subjects: List of subject labels.
    :type subjects: List[str]
    :param froi: fROI configuration to estimate the effect of.
    :type froi: FROIConfig
    :param fill_na_with_zero: Whether to fill NaN values with zero. If False,
        NaN values will be ignored. Default is True.
    :type fill_na_with_zero: Optional[bool]
    :param orthogonalization: The orthogonalization method. Options are
        'all-but-one' and 'odd-even'. Default is 'all-but-one'.
    :type orthogonalization: Optional[str]
    """

    @validate_arguments(orthogonalization={"all-but-one", "odd-even"})
    def __init__(
        self,
        subjects: List[str],
        froi: FROIConfig,
        fill_na_with_zero: Optional[bool] = True,
        orthogonalization: Optional[str] = "all-but-one",
    ):
        self.subjects = subjects
        self.froi = froi
        self.fill_na_with_zero = fill_na_with_zero
        self.orthogonalization = orthogonalization

        self._type = "effect"
        self._data_summary = None
        self._data_detail = None

        # Preload the parcel labels
        _, self.froi_labels = get_parcels(self.froi.parcels)

    def run(
        self, task: str, effects: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the effect estimation. The results are stored in the analysis
        output folder.

        :param task: Task label.
        :type task: str
        :param effects: List of effect labels.
        :type effects: List[str]

        :return: The results are returned as a tuple of two dataframes: the
            effect estimates averaged across runs, and the effect estimates
            detailed by run.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """
        self.task = task
        self.effects = effects

        # Load the data
        data_summary = []
        data_detail = []
        for subject in self.subjects:
            okorths = np.array(
                [
                    _check_orthogonal(
                        subject,
                        self.task,
                        [contrast],
                        self.froi.task,
                        self.froi.contrasts,
                    )
                    for contrast in self.effects
                ]
            )
            okorth = np.all(okorths)
            contrasts = np.array(self.effects)

            froi_all = _get_froi_data(subject, self.froi, "all")
            if froi_all is None:
                warnings.warn(
                    f"Data not found for subject {subject}, fROI {self.froi}, "
                    "skipping."
                )
                continue
            froi_all = froi_all[None, :]

            if not okorth:
                froi_orth, froi_orth_labels = _get_orthogonalized_froi_data(
                    subject, self.froi, 1, self.orthogonalization
                )
                if froi_orth is None:
                    warnings.warn(
                        f"Data not found for subject {subject}, fROI {self.froi} "
                        "for the speicial orthogonalization, skipping those "
                        "non-orthogonal effects."
                    )
                    continue

            for i, contrast in enumerate(contrasts):
                if okorth:
                    data_i_effect = _get_contrast_data(
                        subject, self.task, "all", contrast, "effect"
                    )
                    if data_i_effect is not None:
                        data_i_effect = data_i_effect[None, :]
                    data_i_froi = froi_all
                    effect_run_labels, froi_run_labels = ["all"], ["all"]
                else:
                    data_i_effect, run_label = (
                        _get_orthogonalized_contrast_data(
                            subject,
                            self.task,
                            contrast,
                            2,
                            "effect",
                            self.orthogonalization,
                        )
                    )
                    data_i_froi = froi_orth
                    effect_run_labels, froi_run_labels = (
                        run_label,
                        froi_orth_labels,
                    )
                if data_i_effect is None:
                    warnings.warn(
                        f"Data not found for subject {subject}, effect "
                        f"{contrast}, skipping."
                    )
                    continue

                df_summary, df_detail = self._run(
                    data_i_effect, data_i_froi, self.fill_na_with_zero
                )
                if self.froi_labels is not None:
                    df_summary["froi"] = df_summary["froi"].apply(
                        lambda x: self.froi_labels[x]
                    )
                    df_detail["froi"] = df_detail["froi"].apply(
                        lambda x: self.froi_labels[x]
                    )
                df_detail["effect_run"] = df_detail["run"].apply(
                    lambda x: effect_run_labels[x]
                )
                df_detail["froi_run"] = df_detail["run"].apply(
                    lambda x: froi_run_labels[x]
                )
                df_detail = df_detail.drop(columns=["run"])
                df_summary["subject"] = subject
                df_detail["subject"] = subject
                df_summary["effect"] = contrast
                df_detail["effect"] = contrast
                data_summary.append(df_summary)
                data_detail.append(df_detail)

        # Save and return the results
        self._data_summary = pd.concat(data_summary)
        self._data_detail = pd.concat(data_detail)
        new_effects_info = pd.DataFrame(
            {
                "task": [self.task],
                "effects": [self.effects],
                "fill_na_with_zero": [self.fill_na_with_zero],
                "orthogonalization": [self.orthogonalization],
                "froi": [self.froi],
            }
        )
        self._save(new_effects_info)

        return self._data_summary, self._data_detail

    @staticmethod
    def _run(
        effect_data: np.ndarray,
        froi_masks: np.ndarray,
        fill_na_with_zero: bool,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the effect estimation.

        :param effect_data: The effect data, with shape (n_runs, n_voxels).
        :type effect_data: np.ndarray
        :param froi_masks: The fROI masks, with shape (n_runs, n_voxels).
        :type froi_masks: np.ndarray
        :param fill_na_with_zero: Whether to fill NaN values with zero. If
            False, NaN values will be ignored.
        :type fill_na_with_zero: bool

        :return: The effect estimates averaged across runs, and the effect
            estimates detailed by run.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """
        if len(effect_data.shape) != 2:
            raise ValueError(
                "effect_data should have shape (n_runs, n_voxels)"
            )
        if len(froi_masks.shape) != 2:
            raise ValueError("froi_masks should have shape (n_runs, n_voxels)")
        if effect_data.shape != froi_masks.shape:
            raise ValueError(
                "effect_data and froi_masks should have the same shape"
            )

        # Reduce the data size
        non_nan_voxels = np.any(
            ~np.isnan(froi_masks) & (froi_masks != 0), axis=0
        )
        effect_data = effect_data[:, non_nan_voxels]
        froi_masks = froi_masks[:, non_nan_voxels]

        if fill_na_with_zero:
            effect_data[np.isnan(effect_data)] = 0

        # Compute the effect size
        froi_masks_expanded = froi_masks[None, :, :]
        froi_labels = np.unique(froi_masks)
        froi_labels = froi_labels[froi_labels != 0 & ~np.isnan(froi_labels)]
        froi_masks_expanded = (
            froi_masks_expanded == froi_labels[:, None, None]
        ).astype(float)
        froi_masks_expanded[froi_masks_expanded == 0] = np.nan

        masked_effect = effect_data[None, :, :] * froi_masks_expanded
        effect_size = np.nanmean(masked_effect, axis=(-1))
        df_detail = pd.DataFrame(
            {  # froi, run, size
                "froi": np.repeat(froi_labels, effect_data.shape[0]),
                "run": np.tile(
                    np.arange(effect_data.shape[0]), len(froi_labels)
                ),
                "size": effect_size.flatten(),
            }
        )
        df_summary = (
            df_detail.groupby("froi").agg({"size": "mean"}).reset_index()
        )
        return df_summary, df_detail
