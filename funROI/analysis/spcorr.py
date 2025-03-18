from typing import List, Optional, Union, Tuple
from ..contrast import (
    _check_orthogonal,
    _get_contrast_data,
    _get_orthogonalized_contrast_data,
)
from ..parcels import ParcelsConfig
from ..froi import FROIConfig, _get_froi_data, _get_orthogonalized_froi_data
from ..utils import validate_arguments
from ..parcels import get_parcels
import numpy as np
import pandas as pd
import warnings
import logging
from .utils import AnalysisSaver


class SpatialCorrelationEstimator(AnalysisSaver):
    """
    Estimate the spatial correlation between two contrasts.

    :param subjects: List of subject labels.
    :type subjects: List[str]
    :param froi: fROI or parcels configuration to estimate the effect of.
    :type froi: Union[str, ParcelsConfig, FROIConfig]
    :param orthogonalization: Orthogonalization method to use. Options are
        'all-but-one' and 'odd-even'. Default is 'all-but-one'.
    :type orthogonalization: Optional[str]

    :raises ValueError: If the parcels are specified, but no parcels are found
        for the given configuration.
    """

    @validate_arguments(orthogonalization={"all-but-one", "odd-even"})
    def __init__(
        self,
        subjects: List[str],
        froi: Union[str, ParcelsConfig, FROIConfig],
        orthogonalization: Optional[str] = "all-but-one",
    ):
        self.subjects = subjects
        self.froi = froi
        self.orthogonalization = orthogonalization

        self._type = "spcorr"
        self._data_summary = None
        self._data_detail = None

        # Preload the parcel labels
        if isinstance(self.froi, FROIConfig):
            self.parcels_img, self.froi_labels = get_parcels(self.froi.parcels)
        else:
            self.parcels_img, self.froi_labels = get_parcels(self.froi)
        if not isinstance(self.froi, FROIConfig) and self.parcels_img is None:
            raise ValueError(
                "Specified as parcels, but no parcels found for the given "
                "configuration."
            )

    def run(
        self,
        task1: str,
        effect1: str,
        task2: str,
        effect2: str,
        run_froi: Optional[str] = None,
        run1: Optional[str] = None,
        run2: Optional[str] = None,
    ):
        """
        Run the spatial correlation estimation. The results are stored in the
        analysis output folder.

        :param task1: Task label for the first contrast.
        :type task1: str
        :param effect1: Effect label for the first contrast.
        :type effect1: str
        :param task2: Task label for the second contrast.
        :type task2: str
        :param effect2: Effect label for the second contrast.
        :type effect2: str
        :param run_froi: Run fROI label. If not specified, the fROI is
            determined by automatic orthogonalization.
        :type run_froi: Optional[str]
        :param run1: Run label for the first contrast. If not specified, the
            run is determined by automatic orthogonalization.
        :type run1: Optional[str]
        :param run2: Run label for the second contrast. If not specified, the
            run is determined by automatic orthogonalization.
        :type run2: Optional[str]

        :return: the results are returned as a tuple of two dataframes:
            the spatial correlation estimates averaged across runs, and
            the spatial correlation estimates detailed by run.
        :rtype: Optional[Tuple[pd.DataFrame, pd.DataFrame]]

        :raises ValueError: If run labels are not specified, and fROI, effect1,
            and effect2 are all non-orthogonal to each other.
        :raises ValueError: Some but not all necessary run labels are
            specified.
        """
        orthtype = self.orthogonalization  # abbr
        is_parcels = not isinstance(self.froi, FROIConfig)
        if is_parcels:
            if (run1 is None) != (run2 is None):
                raise ValueError(
                    "Some but not all necessary run labels are specified."
                )
            run_froi = None
        else:
            num_specified = sum(
                1 for run in [run_froi, run1, run2] if run is not None
            )
            if 0 < num_specified < 3:
                raise ValueError(
                    "Some but not all necessary run labels are specified."
                )

        data_summary = []
        data_detail = []
        for subject in self.subjects:
            if not is_parcels and run1 is None:
                okorth_froi_effect1 = _check_orthogonal(
                    subject,
                    task1,
                    [effect1],
                    self.froi.task,
                    self.froi.contrasts,
                )
                okorth_froi_effect2 = _check_orthogonal(
                    subject,
                    task2,
                    [effect2],
                    self.froi.task,
                    self.froi.contrasts,
                )
            else:
                okorth_froi_effect1 = True
                okorth_froi_effect2 = True
            if run1 is None:
                okorth_effects = _check_orthogonal(
                    subject, task1, [effect1], task2, [effect2]
                )
            else:
                okorth_effects = True

            if (
                not okorth_froi_effect1
                and not okorth_froi_effect2
                and not okorth_effects
            ):
                raise ValueError(
                    "Run labels are not specified, and fROI, effect1, and "
                    "effect2 are all non-orthogonal to each other. Up to "
                    "two non-orthogonal pairs can be supported with auto "
                    "orthogonalization."
                )

            group_froi, group1, group2 = self._get_orthogonalized_group(
                okorth_froi_effect1, okorth_froi_effect2, okorth_effects
            )

            if is_parcels:
                froi_data = self.parcels_img.get_fdata().flatten()[None, :]
                froi_run_labels = ["parcels"]
            elif run_froi is not None:
                froi_data = _get_froi_data(subject, self.froi, run_froi)[
                    None, :
                ]
                froi_run_labels = [run_froi]
            elif group_froi == 0:
                froi_data = _get_froi_data(subject, self.froi, "all")[None, :]
                froi_run_labels = ["all"]
            else:
                froi_data, froi_run_labels = _get_orthogonalized_froi_data(
                    subject, self.froi, group_froi, orthtype
                )

            if run1 is not None:
                effect1_data = _get_contrast_data(
                    subject, task1, run1, effect1, "effect"
                )[None, :]
                effect1_run_labels = [run1]
            elif group1 == 0:
                effect1_data = _get_contrast_data(
                    subject, task1, "all", effect1, "effect"
                )[None, :]
                effect1_run_labels = ["all"]
            else:
                effect1_data, effect1_run_labels = (
                    _get_orthogonalized_contrast_data(
                        subject, task1, effect1, group1, "effect", orthtype
                    )
                )

            if run2 is not None:
                effect2_data = _get_contrast_data(
                    subject, task2, run2, effect2, "effect"
                )[None, :]
                effect2_run_labels = [run2]
            elif group2 == 0:
                effect2_data = _get_contrast_data(
                    subject, task2, "all", effect2, "effect"
                )[None, :]
                effect2_run_labels = ["all"]
            else:
                effect2_data, effect2_run_labels = (
                    _get_orthogonalized_contrast_data(
                        subject, task2, effect2, group2, "effect", orthtype
                    )
                )

            if (
                self.orthogonalization == "all-but-one"
                and not okorth_effects
                and okorth_froi_effect1
                and okorth_froi_effect2
            ):
                # Resolve the asymmetric orthogonalization
                effect1_data_2, effect1_run_labels_2 = (
                    _get_orthogonalized_contrast_data(
                        subject, task1, effect1, 2, "effect", orthtype
                    )
                )
                effect2_data_2, effect2_run_labels_2 = (
                    _get_orthogonalized_contrast_data(
                        subject, task2, effect2, 1, "effect", orthtype
                    )
                )
                if effect1_data is not None and effect1_data_2 is not None:
                    effect1_data = np.concat([effect1_data, effect1_data_2])
                    effect1_run_labels = (
                        effect1_run_labels + effect1_run_labels_2
                    )
                if effect2_data is not None and effect2_data_2 is not None:
                    effect2_data = np.concat([effect2_data, effect2_data_2])
                    effect2_run_labels = (
                        effect2_run_labels + effect2_run_labels_2
                    )

            if froi_data is None:
                warnings.warn(
                    f"Data not found for subject {subject}, fROI {self.froi}, "
                    "skipping."
                )
                continue
            if effect1_data is None:
                warnings.warn(
                    f"Data not found for subject {subject}, effect {effect1}, "
                    "skipping."
                )
                continue
            if effect2_data is None:
                warnings.warn(
                    f"Data not found for subject {subject}, effect {effect2}, "
                    "skipping."
                )
                continue

            df_summary, df_detail = self._run(
                effect1_data, effect2_data, froi_data
            )
            if len(froi_run_labels) == 1:
                df_detail["froi_run"] = froi_run_labels[0]
            else:
                df_detail["froi_run"] = df_detail["run"].apply(
                    lambda x: froi_run_labels[x]
                )
            if len(effect1_run_labels) == 1:
                df_detail["effect1_run"] = effect1_run_labels[0]
            else:
                df_detail["effect1_run"] = df_detail["run"].apply(
                    lambda x: effect1_run_labels[x]
                )
            if len(effect2_run_labels) == 1:
                df_detail["effect2_run"] = effect2_run_labels[0]
            else:
                df_detail["effect2_run"] = df_detail["run"].apply(
                    lambda x: effect2_run_labels[x]
                )

            df_detail = df_detail.drop(columns=["run"])

            if self.froi_labels is not None:
                df_summary["froi"] = df_summary["froi"].apply(
                    lambda x: self.froi_labels[x]
                )
                df_detail["froi"] = df_detail["froi"].apply(
                    lambda x: self.froi_labels[x]
                )
            if is_parcels:
                # rename froi to parcels
                df_summary = df_summary.rename(columns={"froi": "parcels"})
                df_detail = df_detail.rename(columns={"froi": "parcels"})

            df_summary["subject"] = subject
            df_detail["subject"] = subject
            data_summary.append(df_summary)
            data_detail.append(df_detail)

        self._data_summary = pd.concat(data_summary)
        self._data_detail = pd.concat(data_detail)
        new_effects_info = pd.DataFrame(
            {
                "task1": [task1],
                "effect1": [effect1],
                "task2": [task2],
                "effect2": [effect2],
                "orthogonalization": [self.orthogonalization],
                "froi": [self.froi],
                "customized_run_froi": [run_froi],
                "customized_run1": [run1],
                "customized_run2": [run2],
            }
        )
        self._save(new_effects_info)

        return self._data_summary, self._data_detail

    @staticmethod
    def _get_orthogonalized_group(
        okorth_froi_effect1: bool,
        okorth_froi_effect2: bool,
        okorth_effects: bool,
    ) -> Tuple[int, int, int]:
        """
        Get the orthogonalized group for the spatial correlation estimation,
        based on the orthogonalization results. No valid assignments for the
        case that all three are non-orthogonal to each other.

        :param okorth_froi_effect1: Whether the fROI and effect1 are orthogonal.
        :type okorth_froi_effect1: bool
        :param okorth_froi_effect2: Whether the fROI and effect2 are orthogonal.
        :type okorth_froi_effect2: bool
        :param okorth_effects: Whether effect1 and effect2 are orthogonal.
        :type okorth_effects: bool

        :return: The orthogonalized group for the spatial correlation estimation.
            If 0, use all-run data. If 1 or 2, use orthogonalized data with for
            corresponding group.
        :rtype: Tuple[int, int, int]
        """
        if okorth_froi_effect1 and okorth_froi_effect2:
            if okorth_effects:
                return 0, 0, 0
            else:
                return 0, 1, 2
        if okorth_froi_effect1:
            if okorth_effects:
                return 1, 0, 2
            else:
                return 1, 1, 2
        if okorth_froi_effect2:
            if okorth_effects:
                return 1, 2, 0
            else:
                return 1, 2, 1

    @staticmethod
    def _run(
        effect1_data: np.ndarray,
        effect2_data: np.ndarray,
        froi_masks: np.ndarray,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the spatial correlation estimation.

        :param effect1_data: The effect data for the first contrast, with shape
            (n_runs, n_voxels).
        :type effect1_data: np.ndarray
        :param effect2_data: The effect data for the second contrast, with shape
            (n_runs, n_voxels).
        :type effect2_data: np.ndarray
        :param froi_masks: The fROI masks, with shape (n_runs, n_voxels).
        :type froi_masks: np.ndarray

        :return: The spatial correlation estimates averaged across runs, and the
            spatial correlation estimates detailed by run.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """
        if len(effect1_data.shape) != 2:
            raise ValueError(
                "effect1_data should have shape (n_runs, n_voxels)"
            )
        if len(effect2_data.shape) != 2:
            raise ValueError(
                "effect2_data should have shape (n_runs, n_voxels)"
            )
        if len(froi_masks.shape) != 2:
            raise ValueError("froi_masks should have shape (n_runs, n_voxels)")

        # Reduce the data size
        non_nan_voxels = np.any(
            ~np.isnan(froi_masks) & (froi_masks != 0), axis=0
        )
        effect1_data = effect1_data[:, non_nan_voxels]
        effect2_data = effect2_data[:, non_nan_voxels]
        froi_masks = froi_masks[:, non_nan_voxels]

        # Compute the spatial correlation
        froi_masks_expanded = froi_masks[None, :, :]
        froi_labels = np.unique(froi_masks)
        froi_labels = froi_labels[froi_labels != 0 & ~np.isnan(froi_labels)]
        froi_masks_expanded = (
            froi_masks_expanded == froi_labels[:, None, None]
        ).astype(float)
        froi_masks_expanded[froi_masks_expanded == 0] = np.nan

        masked_effect1 = effect1_data[None, :, :] * froi_masks_expanded
        masked_effect2 = effect2_data[None, :, :] * froi_masks_expanded

        # Normalize
        masked_effect1 = masked_effect1 - np.nanmean(
            masked_effect1, axis=(-1), keepdims=True
        )
        masked_effect2 = masked_effect2 - np.nanmean(
            masked_effect2, axis=(-1), keepdims=True
        )
        masked_effect1[np.isnan(masked_effect1)] = 0
        masked_effect2[np.isnan(masked_effect2)] = 0
        norm1 = np.linalg.norm(masked_effect1, axis=(-1), keepdims=True)
        norm1[norm1 == 0] = np.finfo(float).eps
        norm2 = np.linalg.norm(masked_effect2, axis=(-1), keepdims=True)
        norm2[norm2 == 0] = np.finfo(float).eps
        masked_effect1 = masked_effect1 / norm1
        masked_effect2 = masked_effect2 / norm2

        spcorr = np.einsum("ijk,ijk->ij", masked_effect1, masked_effect2)
        spcorr = np.clip(
            spcorr, -1 + np.finfo(float).eps, 1 - np.finfo(float).eps
        )
        fisher_z = np.arctanh(spcorr)

        N = np.max(
            [effect1_data.shape[0], effect2_data.shape[0], froi_masks.shape[0]]
        )
        df_detail = pd.DataFrame(
            {  # froi, run, fisher_z
                "froi": np.repeat(froi_labels, N),
                "run": np.tile(np.arange(N), len(froi_labels)),
                "fisher_z": fisher_z.flatten(),
            }
        )
        df_summary = (
            df_detail.groupby("froi").agg({"fisher_z": "mean"}).reset_index()
        )
        return df_summary, df_detail
