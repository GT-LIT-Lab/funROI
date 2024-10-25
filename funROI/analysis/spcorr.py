import numpy as np
from ..contrast import (
    get_contrasts_all_multi_task,
    get_contrasts_orth_single_task,
    get_contrasts_runs_single_task,
    flatten_task_contrasts,
)
from ..froi import _get_frois_all_multi_task, FROI
from typing import List, Tuple, Union
import pandas as pd
from ..utils import get_parcels_labels, get_parcels


class SpatialCorrelationEstimator:
    def __init__(
        self,
        subjects: List[str],
        task_frois: Union[List[Tuple[str, List[FROI]]], List[str]],
    ):
        """
        Initialize the spatial correlation analyzer.

        Parameters
        ----------
        subjects : List[str]
            List of subject IDs.
        frois : Union[List[Tuple[str, List[FROI]]], List[str]]
            List of (task, frois) tuples, or a list of parcels names.
        """
        self.subjects = subjects
        self.task_frois = task_frois

    def compute_spatial_correlation(
        self, task_contrasts: List[Tuple[str, List[str]]]
    ) -> pd.DataFrame:
        """
        Compute the spatial correlation between fROIs.

        Parameters
        ----------
        task_contrasts : List[Tuple[str, List[str]]]
            List of (task, contrasts) tuples.

        Returns
        -------
        spatial_corr_df : pd.DataFrame
            columns=[
                "subject", "localizer_task", "localizer_name", "froi",
                "task1", "task2", "effect1", "effect2", "fisher_z"
            ]
            The spatial correlation DataFrame.
        """
        parcels_labels = []
        parcels_labels_sorted_keys = []
        flattened_localizer_tasks = []
        flattened_localizer_names = []
        for i, task_froi in enumerate(self.task_frois):
            if isinstance(task_froi, str):
                _, labels = get_parcels_labels(task_froi)
                parcels_labels.append(labels)
                parcels_labels_sorted_keys.append(sorted(labels.keys()))
                flattened_localizer_tasks.append(None)
                flattened_localizer_names.append(task_froi)
            else:
                task, frois = task_froi
                for froi in frois:
                    _, labels = get_parcels_labels(froi.parcels)
                    parcels_labels.append(labels)
                    parcels_labels_sorted_keys.append(sorted(labels.keys()))
                    flattened_localizer_tasks.append(task)
                    flattened_localizer_names.append(froi.localizer)

        effect_tasks_flattened, effect_names_flattened = (
            flatten_task_contrasts(task_contrasts)
        )

        spatial_corr_all_subjects = []
        for subject in self.subjects:
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

            frois = np.concatenate(
                frois, axis=0
            ).squeeze()  # get rid of run dimension

            effects = get_contrasts_all_multi_task(
                subject, task_contrasts, "effect"
            )
            spcorr_all_subject = []

            # Treat fORIs separately for efficiency since each fROI is small
            for froi_i, froi in enumerate(frois):
                spcorr_all = []
                effects_masked_i = self.mask_maps(effects, froi)
                for froi_label_i, masked_label_i in enumerate(
                    effects_masked_i
                ):
                    masked_label_i = masked_label_i[None, :]
                    nan_mask = np.isnan(masked_label_i).all(axis=(0, 1, 2))
                    masked_label_i = masked_label_i[:, :, :, ~nan_mask]

                    spatial_corr = self._compute_correlation_fisher_z(
                        masked_label_i, masked_label_i
                    )
                    spcorr_all_i = pd.DataFrame(
                        {
                            "task1": spatial_corr["contrast1"].apply(
                                lambda x: effect_tasks_flattened[x]
                            ),
                            "task2": spatial_corr["contrast2"].apply(
                                lambda x: effect_tasks_flattened[x]
                            ),
                            "effect1": spatial_corr["contrast1"].apply(
                                lambda x: effect_names_flattened[x]
                            ),
                            "effect2": spatial_corr["contrast2"].apply(
                                lambda x: effect_names_flattened[x]
                            ),
                            "fisher_z": spatial_corr["fisher_z"],
                        }
                    )
                    spcorr_all_i["localizer_task"] = flattened_localizer_tasks[
                        froi_i
                    ]
                    spcorr_all_i["localizer_name"] = flattened_localizer_names[
                        froi_i
                    ]
                    spcorr_all_i["froi"] = parcels_labels[froi_i][
                        parcels_labels_sorted_keys[froi_i][froi_label_i]
                    ]
                    spcorr_all.append(spcorr_all_i)

                spcorr_all = pd.concat(spcorr_all)

                # Same-task condition
                spcorr_same_task = []
                for task, contrasts in task_contrasts:
                    effect_run = get_contrasts_runs_single_task(
                        subject, task, contrasts, "effect"
                    )
                    effect_orth = get_contrasts_orth_single_task(
                        subject, task, contrasts, "effect"
                    )
                    effect_run_masked = self.mask_maps(effect_run, froi)
                    effect_orth_masked = self.mask_maps(effect_orth, froi)

                    for froi_label_i in range(effect_run_masked.shape[0]):
                        effect_run_masked_i = effect_run_masked[froi_label_i][
                            None, :
                        ]
                        effect_orth_masked_i = effect_orth_masked[
                            froi_label_i
                        ][None, :]

                        nan_mask_run = np.isnan(effect_run_masked_i).all(
                            axis=(0, 1, 2)
                        )
                        nan_mask_orth = np.isnan(effect_orth_masked_i).all(
                            axis=(0, 1, 2)
                        )
                        mask_to_remove = nan_mask_run & nan_mask_orth
                        valid_mask = ~mask_to_remove

                        effect_run_masked_i = effect_run_masked_i[
                            :, :, :, valid_mask
                        ]
                        effect_orth_masked_i = effect_orth_masked_i[
                            :, :, :, valid_mask
                        ]

                        spatial_corr = self._compute_correlation_fisher_z(
                            effect_run_masked_i, effect_orth_masked_i
                        )
                        spcorr_same_task_i = pd.DataFrame(
                            {
                                "task1": [task] * spatial_corr.shape[0],
                                "task2": [task] * spatial_corr.shape[0],
                                "effect1": spatial_corr["contrast1"].apply(
                                    lambda x: contrasts[x]
                                ),
                                "effect2": spatial_corr["contrast2"].apply(
                                    lambda x: contrasts[x]
                                ),
                                "fisher_z": spatial_corr["fisher_z"],
                            }
                        )
                        spcorr_same_task_i["localizer_task"] = (
                            flattened_localizer_tasks[froi_i]
                        )
                        spcorr_same_task_i["localizer_name"] = (
                            flattened_localizer_names[froi_i]
                        )
                        spcorr_same_task_i["froi"] = parcels_labels[froi_i][
                            parcels_labels_sorted_keys[froi_i][froi_label_i]
                        ]
                        spcorr_same_task.append(spcorr_same_task_i)

                spcorr_same_task = pd.concat(spcorr_same_task)
                spcorr_all.set_index(
                    [
                        "task1",
                        "task2",
                        "effect1",
                        "effect2",
                        "localizer_task",
                        "localizer_name",
                        "froi",
                    ],
                    inplace=True,
                )
                spcorr_same_task.set_index(
                    [
                        "task1",
                        "task2",
                        "effect1",
                        "effect2",
                        "localizer_task",
                        "localizer_name",
                        "froi",
                    ],
                    inplace=True,
                )

                spcorr_all.update(spcorr_same_task)
                spcorr_all.reset_index(inplace=True)
                spcorr_all_subject.append(spcorr_all)

            spcorr_all_subject = pd.concat(spcorr_all_subject)
            spcorr_all_subject["subject"] = subject
            spatial_corr_all_subjects.append(spcorr_all_subject)

        spatial_corr_df = pd.concat(spatial_corr_all_subjects)
        return spatial_corr_df

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
    def mask_maps(cls, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Mask data with a mask.

        Parameters
        ----------
        data : np.ndarray, shape (n_contrast, n_runs, n_voxels)
            The data to be masked.
        mask : np.ndarray, shape (n_voxels,)
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
        )  # n_labels x n_voxels
        mask_expanded[mask_expanded == 0] = np.nan
        masked_data = (
            data[np.newaxis, :, :, :] * mask_expanded
        )  # n_labels x n_contrast x n_runs x n_voxels

        in_mask_nan = np.isnan(masked_data) * (~np.isnan(mask_expanded))
        masked_data[in_mask_nan] = np.inf

        return masked_data
