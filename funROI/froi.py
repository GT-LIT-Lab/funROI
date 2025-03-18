from .parcels import get_parcels, ParcelsConfig
from .utils import (
    validate_arguments,
    _get_orthogonalized_run_labels,
)
from . import get_bids_deriv_folder
from .contrast import (
    _get_contrast_data,
    _get_orthogonalized_contrast_data,
    _get_contrast_runs,
    _get_contrast_runs_by_group,
    _get_contrast_path,
)
from typing import Optional, List, Union, Tuple
from nibabel.nifti1 import Nifti1Image
import numpy as np
from nilearn.image import load_img, new_img_like
from collections import namedtuple
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
import logging


class FROIConfig(dict):
    """
    Functional region of interest (fROI) configuration.

    :param task: The task label.
    :type task: str
    :param contrasts: List of contrast labels.
    :type contrasts: List[str]
    :param threshold_type: The threshold type.
        Options are 'none', 'bonferroni', 'fdr', 'n', 'percent'.
    :type threshold_type: str
    :param threshold_value: The threshold value.
    :type threshold_value: float
    :param parcels: The parcels configuration. If a string is provided, it is
        assumed to be the path to the parcels file.
    :type parcels: Union[str, ParcelsConfig]
    :param conjunction_type: The conjunction type if multiple contrasts are
        provided. Options are 'min', 'max', 'sum', 'prod', 'and', 'or', or None.
    :type conjunction_type: str, optional
    """

    @validate_arguments(
        threshold_type={"none", "bonferroni", "fdr", "n", "percent"},
        conjunction_type={"min", "max", "sum", "prod", "and", "or", None},
    )
    def __init__(
        self,
        task: str,
        contrasts: List[str],
        threshold_type: str,
        threshold_value: float,
        parcels: Union[str, ParcelsConfig],
        conjunction_type: Optional[str] = None,
    ):
        if threshold_value < 0:
            raise ValueError("Threshold value must be non-negative.")
        self.task = task
        self.contrasts = contrasts
        self.conjunction_type = conjunction_type
        self.threshold_type = threshold_type
        self.threshold_value = threshold_value

        if not isinstance(parcels, ParcelsConfig):
            parcels = ParcelsConfig(parcels)
        self.parcels = parcels

        dict.__init__(
            self,
            task=task,
            contrasts=contrasts,
            conjunction_type=conjunction_type,
            threshold_type=threshold_type,
            threshold_value=threshold_value,
            parcels=parcels,
        )

    def __repr__(self):
        return (
            f"FROIConfig(task={self.task}, "
            f"contrasts={self.contrasts}, "
            f"conjunction_type={self.conjunction_type}, "
            f"threshold_type={self.threshold_type}, "
            f"threshold_value={self.threshold_value}, "
            f"parcels={self.parcels})"
        )

    def __eq__(self, other):
        return isinstance(other, FROIConfig) and (
            self.task == other.task
            and self.contrasts == other.contrasts
            and self.conjunction_type == other.conjunction_type
            and self.threshold_type == other.threshold_type
            and self.threshold_value == other.threshold_value
            and self.parcels == other.parcels
        )


_get_subject_froi_folder = lambda subject, task: (
    get_bids_deriv_folder() / f"first_level_{task}" / f"sub-{subject}" / "froi"
)
_get_froi_info_path = lambda subject, task: (
    _get_subject_froi_folder(subject, task)
    / f"sub-{subject}_task-{task}_frois.csv"
)


def _get_froi_path(
    subject: str,
    run_label: str,
    config: FROIConfig,
    create: Optional[bool] = False,
) -> str:
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

    info_path = _get_froi_info_path(subject, task)
    if not info_path.exists():
        id = 0
        if create:
            _get_subject_froi_folder(subject, task).mkdir(
                parents=True, exist_ok=True
            )
            frois_new["id"] = 0
            frois_new.to_csv(info_path, index=False)
    else:
        frois = pd.read_csv(info_path)

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

        if len(frois_matched) == 0:
            id = frois["id"].max() + 1
            if create:
                frois_new["id"] = id
                frois = pd.concat([frois, frois_new], ignore_index=True)
                frois.to_csv(info_path, index=False)
        else:
            id = frois_matched["id"].values[0]

    id = int(id)
    id = f"{id:04d}"
    return (
        _get_subject_froi_folder(subject, task)
        / f"sub-{subject}_task-{task}_run-{run_label}_froi-{id}_mask.nii.gz"
    )


def _get_froi_runs(subject: str, config: FROIConfig):
    runs = None
    for contrast in config.contrasts:
        runs_i = _get_contrast_runs(subject, config.task, contrast)
        if runs is None:
            runs = runs_i
        else:
            runs = list(set(runs) & set(runs_i))
    return sorted(runs)


@validate_arguments(
    group={1, 2}, orthogonalization={"all-but-one", "odd-even"}
)
def _get_orthogonalized_froi_data(
    subject: str,
    config: FROIConfig,
    group: int,
    orthogonalization: Optional[str] = "all-but-one",
) -> Tuple[np.ndarray, List[str]]:
    """
    Get the orthogonalized froi data.

    :return: The froi masks, shape (n_runs, n_voxels) and the run labels.
        If any of the froi masks is not found, return None, None.
    :rtype: Tuple[np.ndarray, List[str]]
    """
    runs = _get_froi_runs(subject, config)
    if len(runs) == 0:
        return None, None
    labels = _get_orthogonalized_run_labels(runs, group, orthogonalization)

    data = []
    for label in labels:
        froi_data = _get_froi_data(subject, config, label)
        if froi_data is None:
            froi_data = _create_froi(subject, config, label)
            if froi_data is None:
                return None, None
        data.append(froi_data.flatten())
    return np.array(data), labels


def _get_froi_data(
    subject: str, config: FROIConfig, run_label: str
) -> np.ndarray:
    """
    Get the froi data by run label.

    :return: The froi mask, shape (n_voxels,). If the froi mask is not
        found, return None.
    :rtype: np.ndarray
    """
    froi_path = _get_froi_path(subject, run_label, config)
    if not froi_path.exists():
        data = _create_froi(subject, config, run_label)
        if data is None:
            return None
        return data
    return load_img(froi_path).get_fdata().flatten()


def _create_froi(
    subject: str, config: FROIConfig, run_label: str
) -> np.ndarray:
    """
    Create and save a fROI mask. The fROI labels are based on the parcels.
    Numeric labels not included in the label dictionary are not included, if
    an external label file is provided.

    :return: The froi mask, shape (n_voxels,). If any contrast data
        is not found, return None.
    :rtype: np.ndarray
    """
    parcels_img, parcel_labels = get_parcels(config.parcels)
    parcels_ref = None

    data = []
    for contrast in config.contrasts:
        data_i = _get_contrast_data(
            subject, config.task, run_label, contrast, "p"
        )
        if data_i is None:
            return None
        if parcels_img is None and parcels_ref is None:
            contrast_pth = _get_contrast_path(
                subject, config.task, run_label, contrast, "p"
            )
            parcels_ref = load_img(contrast_pth)
        data.append(data_i[None, ...])
    data = np.array(data)

    if parcels_ref is None:  # real parcels in use
        froi_mask = np.zeros_like(parcels_img.get_fdata().flatten())
        for label in parcel_labels.keys():
            froi_mask_i = (
                _create_p_map_mask(
                    data,
                    config.conjunction_type,
                    config.threshold_type,
                    config.threshold_value,
                    parcels_img.get_fdata().flatten() == label,
                )
                .squeeze()
                .astype(bool)
            )
            froi_mask[froi_mask_i] = label
    else:
        froi_mask = _create_p_map_mask(
            data,
            config.conjunction_type,
            config.threshold_type,
            config.threshold_value,
        )
        parcels_img = parcels_ref

    froi_path = _get_froi_path(subject, run_label, config, create=True)
    froi_path.parent.mkdir(parents=True, exist_ok=True)
    froi_img = new_img_like(parcels_img, froi_mask.reshape(parcels_img.shape))
    froi_img.to_filename(froi_path)

    return froi_mask.flatten()


@validate_arguments(
    threshold_type={"n", "percent", "fdr", "bonferroni", "none"},
)
def _threshold_p_map(
    data: np.ndarray, threshold_type: str, threshold_value: float
) -> np.ndarray:
    """
    Extract voxels from a p-map image based on a threshold. p-value correction
    is applied along the voxel dimension.

    :param data: The p-map data, shape (n_runs, n_voxels).
    :type data: np.ndarray
    :param threshold_type: The threshold type.
        Options are 'n', 'percent', 'fdr', 'bonferroni', or 'none'.
    :type threshold_type: str
    :param threshold_value: The threshold value.
    :type threshold_value: float

    :return: The froi mask, shape (n_runs, n_voxels).
    :rtype: np.ndarray
    """
    data = np.moveaxis(data, -1, 0)
    froi_mask = np.zeros_like(data)

    if threshold_type == "n":
        pvals_sorted = np.sort(data, axis=0)
        threshold = pvals_sorted[threshold_value - 1]
        froi_mask[data <= threshold] = 1

    # All-tie-inclusive thresholding
    elif "percent" in threshold_type:
        pvals_sorted = np.sort(data, axis=0)
        n = np.floor(
            threshold_value * np.sum(~np.isnan(data), axis=0, keepdims=True)
        ).astype(int)
        threshold = np.take_along_axis(pvals_sorted, n - 1, axis=0)
        froi_mask[data <= threshold] = 1
    elif threshold_type == "fdr":
        for i in range(data.shape[-1]):
            pvals = data[:, i]
            mask = fdrcorrection(pvals, alpha=threshold_value)[0]
            froi_mask[:, i] = mask
    elif threshold_type == "bonferroni":
        froi_mask.flat[data.flatten() < threshold_value / data.shape[-1]] = 1
    else:
        froi_mask.flat[data.flatten() < threshold_value] = 1

    froi_mask = np.moveaxis(froi_mask, 0, -1)
    return froi_mask


@validate_arguments(
    conjunction_type={"min", "max", "sum", "prod", "and", "or", None},
    threshold_type={"n", "percent", "fdr", "bonferroni", "none"},
)
def _create_p_map_mask(
    data: np.ndarray,
    conjunction_type: str,
    threshold_type: str,
    threshold_value: float,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create a mask based on a p-map data.

    :param data: The p-map data, shape (n_contrast, n_runs, n_voxels).
    :type data: np.ndarray
    :param conjunction_type: The conjunction type.
        Options are 'min', 'max', 'sum', 'prod', 'and', or 'or'.
    :type conjunction_type: str
    :param threshold_type: The threshold type.
        Options are 'n', 'percent', 'fdr', 'bonferroni', or 'none'.
    :type threshold_type: str
    :param threshold_value: The threshold value.
    :type threshold_value: float
    :param mask: The explicit mask to be applied before thresholding.
    :type mask: np.ndarray, shape (n_voxels), optional

    :return: The froi masks, shape (n_runs, n_voxels).
    :rtype: np.ndarray
    """
    assert (
        data.ndim == 3
    ), "data should have shape (n_contrast, n_runs, n_voxels)"

    if mask is not None:
        data = data.astype(float)
        data[np.isnan(data)] = np.inf
        data[:, :, mask == 0] = np.nan

    if conjunction_type in ["min", "max", "sum", "prod"]:
        combined_data = eval(f"np.{conjunction_type}(data, axis=-3)")
        froi_mask = _threshold_p_map(
            combined_data, threshold_type, threshold_value
        )
    else:
        thresholded_data = _threshold_p_map(
            data, threshold_type, threshold_value
        )
        if conjunction_type == "and":
            froi_mask = np.all(thresholded_data, axis=-3)
        else:
            froi_mask = np.any(thresholded_data, axis=-3)

    return froi_mask
