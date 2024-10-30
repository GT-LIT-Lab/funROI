import numpy as np
from ..contrast import _get_contrast_run, _get_contrast_all, _get_contrast_orth
from ..froi import _get_froi_all, _get_froi_orth
from typing import List, Tuple, Union, Optional
import pandas as pd
from ..contrast import _check_orthogonal
from ..parcels import get_parcels
import warnings
from ..utils import (
    FROIConfig,
    ParcelsConfig,
    get_spatial_correlation_estimation_folder,
    get_next_spatial_correlation_estimation_path,
)
import os
import json


class SpatialCorrelationEstimator:
    def __init__(
        self,
        subjects: List[str],
        froi: Union[str, ParcelsConfig, FROIConfig],
    ):
        """
        Initialize the spatial correlation analyzer.

        Parameters
        ----------
        subjects : List[str]
            List of subject IDs.
        froi : Union[str, ParcelsConfig, FROIConfig]
            Parcels or fROI configuration.

        Raises
        ------
        ValueError
            If the parcels image is not found.
        """
        self.subjects = subjects
        self.froi = froi

        # Preload the parcels and labels
        if isinstance(froi, FROIConfig):
            self.parcels_img, self.parcels_labels = get_parcels(froi.parcels)
            self.use_parcels = False
        else:
            self.parcels_img, self.parcels_labels = get_parcels(froi)
            self.use_parcels = True

    def run(
        self,
        effects: List[Tuple[str, List[str]]],
        return_output: Optional[bool] = False,
    ) -> pd.DataFrame:
        """
        Compute the spatial correlation between fROIs.

        Parameters
        ----------
        effects : List[Tuple[str, List[str]]]
            List of (task, contrasts) tuples.

        Returns
        -------
        spatial_corr_df : pd.DataFrame
            columns=[
                "subject", "froi", "task1", "task2", "effect1", "effect2",
                "fisher_z"
            ]
            The spatial correlation DataFrame.
        """
        spatial_corr_all_subjects = []
        for subject in self.subjects:
            # Check orthogonality between fROI and each contrast first,
            # If any non-orthogonal, do orthogonalization
            # If both non-orthogonal, skip if contrast also non-orthogonal
            # to each other.
            if self.use_parcels:
                froi_data = {
                    "all": self.parcels_img.get_fdata().flatten()[None, :],
                }
            else:
                froi_data = {
                    "all": _get_froi_all(subject, self.froi),
                    "orth": _get_froi_orth(subject, self.froi),
                }

            tasks_avail = []
            contrasts_avail = []
            orth2froi = []
            effect_data = {"all": [], "orth": [], "run": []}
            for task, contrasts in effects:
                for contrast in contrasts:
                    try:
                        contrast_all = _get_contrast_all(
                            subject, task, contrast, "effect"
                        )
                    except ValueError:
                        warnings.warn(
                            f"Subject {subject} task {task} contrast {contrast} not found, skipping"
                        )
                        continue

                    tasks_avail.append(task)
                    contrasts_avail.append(contrast)
                    if self.use_parcels:
                        okorth = True
                    else:
                        okorth = _check_orthogonal(
                            subject,
                            self.froi.task,
                            self.froi.contrasts,
                            task,
                            [contrast],
                        )
                    orth2froi.append(okorth)

                    if okorth:
                        effect_data["all"].append(contrast_all)
                    else:
                        effect_data["all"].append(None)
                    effect_data["orth"].append(
                        _get_contrast_orth(subject, task, contrast, "effect")
                    )
                    effect_data["run"].append(
                        _get_contrast_run(subject, task, contrast, "effect")
                    )

            for effect1_idx in range(len(tasks_avail)):
                task1, effect1 = (
                    tasks_avail[effect1_idx],
                    contrasts_avail[effect1_idx],
                )
                for effect2_idx in range(
                    len(tasks_avail)
                ):  # For now, spcorr not symmetric because all-but-one orthogonalization does not have the same counter-part, so both spcorr(a,b) and spcorr(b,a) are computed
                    task2, effect2 = (
                        tasks_avail[effect2_idx],
                        contrasts_avail[effect2_idx],
                    )
                    okorth_between_effects = _check_orthogonal(
                        subject, task1, [effect1], task2, [effect2]
                    )
                    if not okorth_between_effects:
                        if (
                            not orth2froi[effect1_idx]
                            and not orth2froi[effect2_idx]
                        ):
                            warnings.warn(
                                f"Subject {subject}: skipping spatial "
                                f"correlation between {task1} {effect1} and "
                                f"{task2} {effect2} because they are not "
                                "orthogonal to each other, and both not "
                                "orthogonal to fROI."
                            )
                            continue
                        if not orth2froi[effect1_idx]:
                            effect1_data = effect_data["run"][effect1_idx]
                            effect2_data = effect_data["orth"][effect2_idx]
                            froi = froi_data["orth"]
                        elif not orth2froi[effect2_idx]:
                            effect1_data = effect_data["orth"][effect1_idx]
                            effect2_data = effect_data["run"][effect2_idx]
                            froi = froi_data["orth"]
                        else:
                            effect1_data = effect_data["orth"][effect1_idx]
                            effect2_data = effect_data["run"][effect2_idx]
                            froi = froi_data["all"]
                    else:
                        if (
                            not orth2froi[effect1_idx]
                            and not orth2froi[effect2_idx]
                        ):
                            effect1_data = effect_data["run"][effect1_idx]
                            effect2_data = effect_data["run"][effect2_idx]
                            froi = froi_data["orth"]
                        if not orth2froi[effect1_idx]:
                            effect1_data = effect_data["run"][effect1_idx]
                            effect2_data = effect_data["all"][effect2_idx]
                            froi = froi_data["orth"]
                        elif not orth2froi[effect2_idx]:
                            effect1_data = effect_data["all"][effect1_idx]
                            effect2_data = effect_data["run"][effect2_idx]
                            froi = froi_data["orth"]
                        else:
                            effect1_data = effect_data["all"][effect1_idx]
                            effect2_data = effect_data["all"][effect2_idx]
                            froi = froi_data["all"]

                    # Trim voxels not labeled in both fROI and both effects
                    non_zero_froi = np.sum(froi != 0, axis=0) > 0
                    non_zero_any = non_zero_froi
                    froi = froi[:, non_zero_any]
                    effect1_data = effect1_data[:, non_zero_any]
                    effect2_data = effect2_data[:, non_zero_any]

                    # Apply froi mask
                    effect1_data_mapped = self._apply_froi(
                        effect1_data[None, :], froi
                    )
                    effect2_data_mapped = self._apply_froi(
                        effect2_data[None, :], froi
                    )

                    sorted_labels = np.sort(np.unique(froi))
                    sorted_labels = sorted_labels[sorted_labels != 0]

                    spcorr = self._compute_correlation_fisher_z(
                        effect1_data_mapped, effect2_data_mapped
                    )

                    spatial_corr = pd.DataFrame(
                        {
                            "subject": [subject] * len(spcorr),
                            "froi": spcorr["froi_id"].map(
                                lambda x: self.parcels_labels[sorted_labels[x]]
                            ),
                            "task1": [task1] * len(spcorr),
                            "task2": [task2] * len(spcorr),
                            "effect1": [effect1] * len(spcorr),
                            "effect2": [effect2] * len(spcorr),
                            "fisher_z": spcorr["fisher_z"],
                        }
                    )

                    spatial_corr_all_subjects.append(spatial_corr)

        spatial_corr_all_subjects = pd.concat(spatial_corr_all_subjects)

        spcorr_folder = get_spatial_correlation_estimation_folder()
        if not os.path.exists(spcorr_folder):
            os.makedirs(spcorr_folder)

        spcorr_path, config_path = (
            get_next_spatial_correlation_estimation_path()
        )
        spatial_corr_all_subjects.to_csv(spcorr_path, index=False)
        config = {
            "subjects": self.subjects,
            "froi": self.froi,
            "effects": effects,
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        if return_output:
            return spatial_corr_all_subjects

    @classmethod
    def _compute_correlation_fisher_z(
        cls, data1: np.ndarray, data2: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute the spatial correlation between two datasets.

        Parameters
        ----------
        data1 : np.ndarray, shape (n_frois, n_contrasts_1, n_runs, n_voxels)
            The first dataset.
        data2 : np.ndarray, shape (n_frois, n_contrasts_2, n_runs, n_voxels)
            The second dataset.

        Returns
        -------
        spatial_corr : pd.DataFrame
            columns=["froi_id", "contrast1", "contrast2", "fisher_z"]
            The Fisher Z-transformed spatial correlation
        """
        assert (
            len(data1.shape) == 4
        ), "data1 should have shape (n_groups, n_contrasts_1, n_runs, n_voxels)"
        assert (
            len(data2.shape) == 4
        ), "data2 should have shape (n_groups, n_contrasts_2, n_runs, n_voxels)"
        assert (
            data1.shape[2] == data2.shape[2]
        ), "data1 and data2 should have the same number of runs"
        assert (
            data1.shape[3] == data2.shape[3]
        ), "data1 and data2 should have the same number of voxels"

        # Mask NaN values with mean on the -1 axis
        data1_for_mean = data1
        data1_for_mean[np.isinf(data1)] = np.nan
        means1 = np.nanmean(data1_for_mean, axis=-1, keepdims=True)
        np.putmask(
            data1,
            np.where(np.isinf(data1) | np.isnan(data1), 1, 0),
            np.broadcast_to(means1, data1.shape),
        )
        data2_for_mean = data2
        data2_for_mean[np.isinf(data2)] = np.nan
        means2 = np.nanmean(data2_for_mean, axis=-1, keepdims=True)
        np.putmask(
            data2,
            np.where(np.isinf(data2) | np.isnan(data2), 1, 0),
            np.broadcast_to(means2, data2.shape),
        )

        # Normalization
        data1 = data1 - means1
        data1 = data1 / np.linalg.norm(data1, axis=-1, keepdims=True)
        data2 = data2 - means2
        data2 = data2 / np.linalg.norm(data2, axis=-1, keepdims=True)

        # Pearson correlation via dot product
        spatial_corr = np.einsum("milk,mjlk->mijl", data1, data2)

        # Fisher Z-transform
        fisher_z = np.arctanh(spatial_corr)

        # Average over runs
        fisher_z = np.mean(fisher_z, axis=-1)

        n_froi_labels, n_contrasts1, n_contrasts2 = fisher_z.shape
        froi_ids, contrast_1_ids, contrast_2_ids = np.indices(
            (n_froi_labels, n_contrasts1, n_contrasts2)
        )

        # Flatten the indices and the values of the 3D array
        froi_ids_flat = froi_ids.flatten()
        contrast_1_ids_flat = contrast_1_ids.flatten()
        contrast_2_ids_flat = contrast_2_ids.flatten()
        values_flat = fisher_z.flatten()

        spatial_corr_df = pd.DataFrame(
            {
                "froi_id": froi_ids_flat,
                "contrast1": contrast_1_ids_flat,
                "contrast2": contrast_2_ids_flat,
                "fisher_z": values_flat,
            }
        )

        return spatial_corr_df

    @classmethod
    def _apply_froi(cls, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Mask data with a mask.

        Parameters
        ----------
        data : np.ndarray, shape (n_contrast, n_runs, n_voxels)
            The data to be masked.
        mask : np.ndarray, shape (n_runs, n_voxels)
            The mask labeled by non-zero integers.

        Returns
        -------
        masked_data : np.ndarray, shape (n_labels, n_contrast, n_runs, n_voxels)
            The masked data, where the data outside the mask is set to NaN.
            The NaNs within the mask is labeled as np.inf.
            The labels are ordered ascendingly.
        """
        mask_labels = np.unique(mask)
        mask_labels = mask_labels[mask_labels != 0]
        mask_expanded = (mask == (mask_labels[:, None, None, None])).astype(
            float
        )  # n_labels x 1 x n_runs x n_voxels
        mask_expanded[mask_expanded == 0] = np.nan
        masked_data = (
            data[np.newaxis, :, :, :] * mask_expanded
        )  # n_labels x n_contrast x n_runs x n_voxels

        in_mask_nan = np.isnan(masked_data) * (~np.isnan(mask_expanded))
        masked_data[in_mask_nan] = np.inf

        return masked_data
