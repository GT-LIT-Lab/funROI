from typing import List, Optional, Union, Tuple
from ..utils import validate_arguments
from ..contrast import _check_orthogonal
from ..parcels import ParcelsConfig, get_parcels
from ..froi import FROIConfig, _get_froi_data, _get_orthogonalized_froi_data
import numpy as np
import pandas as pd
from .utils import AnalysisSaver


class OverlapEstimator(AnalysisSaver):
    """
    Estimate the overlap between two sets of parcels or fROIs.

    :param kind: Kind of overlap to estimate. Options are 'overlap' and
        'dice'. Default is 'overlap'.
    :type kind: Optional[str]
    :param orthogonalization: Orthogonalization method to use. Options are
        'all-but-one' and 'odd-even'. Default is 'all-but-one'.
    :type orthogonalization: Optional[str]
    """

    @validate_arguments(orthogonalization={"all-but-one", "odd-even"})
    @validate_arguments(kind={"overlap", "dice"})
    def __init__(
        self,
        kind: Optional[str] = "overlap",
        orthogonalization: Optional[str] = "all-but-one",
    ):
        self.kind = kind
        self.orthogonalization = orthogonalization

        self._type = "overlap"
        self._data_summary = None
        self._data_detail = None

    def run(
        self,
        froi1: Union[FROIConfig, str, ParcelsConfig],
        froi2: Union[FROIConfig, str, ParcelsConfig],
        subject1: Optional[str] = None,
        subject2: Optional[str] = None,
        run1: Optional[str] = None,
        run2: Optional[str] = None,
    ):
        """
        Run the overlap estimation. The results are stored in the analysis
        output folder.

        :param froi1: fROI or parcels configuration for the first set of
            parcels or fROIs.
        :type froi1: Union[FROIConfig, str, ParcelsConfig]
        :param froi2: fROI or parcels configuration for the second set of
            parcels or fROIs.
        :type froi2: Union[FROIConfig, str, ParcelsConfig]
        :param subject1: Subject label for the first set of fROIs. Required if
            fROIs are used.
        :type subject1: str
        :param subject2: Subject label for the second set of fROIs. Required if
            fROIs are used.
        :type subject2: str
        :param run1: Run label for the first set of parcels or fROIs. If not
            specified, the run is determined by automatic orthogonalization.
        :type run1: Optional[str]
        :param run2: Run label for the second set of parcels or fROIs. If not
            specified, the run is determined by automatic orthogonalization.
        :type run2: Optional[str]

        :return: the results are returned as a tuple of two dataframes: the
            overlap estimates averaged across runs, and the overlap estimates
            detailed by run.
        :rtype: Optional[Tuple[pd.DataFrame, pd.DataFrame]]

        :raises ValueError: If the parcels are used and the parcels image is
            not found.
        """
        self.froi1 = froi1
        self.froi2 = froi2
        froi1_img, froi1_labels = get_parcels(
            self.froi1.parcels
            if isinstance(self.froi1, FROIConfig)
            else self.froi1
        )
        froi2_img, froi2_labels = get_parcels(
            self.froi2.parcels
            if isinstance(self.froi2, FROIConfig)
            else self.froi2
        )
        self.run1 = run1
        self.run2 = run2

        is_parcels1 = not isinstance(self.froi1, FROIConfig)
        if is_parcels1:
            if froi1_img is None:
                raise ValueError("Parcels image 1 not found")
            froi1_data = froi1_img.get_fdata().flatten()[None, :]
        else:
            if subject1 is None:
                raise ValueError("Subject label 1 is required for fROIs")
        is_parcels2 = not isinstance(self.froi2, FROIConfig)
        if is_parcels2:
            if froi2_img is None:
                raise ValueError("Parcels image 2 not found")
            froi2_data = froi2_img.get_fdata().flatten()[None, :]
        else:
            if subject2 is None:
                raise ValueError("Subject label 2 is required for fROIs")

        # Load the data
        if is_parcels1 and is_parcels2:
            froi1_run_labels, froi2_run_labels = ["parcels"], ["parcels"]
        elif is_parcels1:
            if run2 is None:
                run2 = "all"
            froi1_run_labels, froi2_run_labels = ["parcels"], [run2]
            froi2_data = _get_froi_data(subject2, self.froi2, run2)[None, :]
        elif is_parcels2:
            if run1 is None:
                run1 = "all"
            froi1_run_labels, froi2_run_labels = ["all"], [run1]
            froi1_data = _get_froi_data(subject1, self.froi1, run1)[None, :]
        elif run1 is not None and run2 is not None:
            froi1_run_labels, froi2_run_labels = [run1], [run2]
            froi1_data = _get_froi_data(subject1, self.froi1, run1)[None, :]
            froi2_data = _get_froi_data(subject2, self.froi2, run2)[None, :]
        else:
            okorth = (subject1 != subject2) or _check_orthogonal(
                subject1,
                self.froi1.task,
                self.froi1.contrasts,
                self.froi2.task,
                self.froi2.contrasts,
            )
            if self.run1 is not None and self.run2 is not None:
                froi1_run_labels, froi2_run_labels = [self.run1], [self.run2]
                froi1_data = _get_froi_data(subject1, self.froi1, self.run1)[
                    None, :
                ]
                froi2_data = _get_froi_data(subject2, self.froi2, self.run2)[
                    None, :
                ]
            elif okorth:
                froi1_run_labels, froi2_run_labels = ["all"], ["all"]
                froi1_data = _get_froi_data(subject1, self.froi1, "all")[
                    None, :
                ]
                froi2_data = _get_froi_data(subject2, self.froi2, "all")[
                    None, :
                ]
            else:
                froi1_data, froi1_run_labels = _get_orthogonalized_froi_data(
                    subject1, self.froi1, 1, self.orthogonalization
                )
                if froi1_data is None:
                    raise ValueError(
                        f"Data not found for subject {subject1}, fROI "
                        f"{self.froi1} for the orthogonalization, skipping."
                    )
                froi2_data, froi2_run_labels = _get_orthogonalized_froi_data(
                    subject2, self.froi2, 2, self.orthogonalization
                )
                if froi2_data is None:
                    raise ValueError(
                        f"Data not found for subject {subject2}, fROI "
                        f"{self.froi2} for the orthogonalization, skipping."
                    )
                if self.orthogonalization == "all-but-one":
                    # To resolve the issue of asymmetric orthogonalization
                    froi1_data2, froi1_run_labels2 = (
                        _get_orthogonalized_froi_data(
                            subject1, self.froi1, 2, self.orthogonalization
                        )
                    )
                    if froi1_data2 is None:
                        raise ValueError(
                            f"Data not found for subject {subject1}, fROI "
                            f"{self.froi1} for the orthogonalization, "
                            "skipping."
                        )
                    froi2_data2, froi2_run_labels2 = (
                        _get_orthogonalized_froi_data(
                            subject2, self.froi2, 1, self.orthogonalization
                        )
                    )
                    if froi2_data2 is None:
                        raise ValueError(
                            f"Data not found for subject {subject2}, fROI "
                            f"{self.froi2} for the orthogonalization, "
                            "skipping."
                        )
                    froi1_data = np.concat([froi1_data, froi1_data2])
                    froi2_data = np.concat([froi2_data, froi2_data2])
                    froi1_run_labels = np.concat(
                        [froi1_run_labels, froi1_run_labels2]
                    )
                    froi2_run_labels = np.concat(
                        [froi2_run_labels, froi2_run_labels2]
                    )

        df_summary, df_detail = self._run(froi1_data, froi2_data, self.kind)
        if not is_parcels1 or not is_parcels2:
            df_detail["run1"] = df_detail["run"].apply(
                lambda x: froi1_run_labels[x]
            )
            df_detail["run2"] = df_detail["run"].apply(
                lambda x: froi2_run_labels[x]
            )
        df_detail = df_detail.drop(columns=["run"])
        if froi1_labels is not None:
            df_summary["froi1"] = df_summary["froi1"].apply(
                lambda x: froi1_labels[x]
            )
            df_detail["froi1"] = df_detail["froi1"].apply(
                lambda x: froi1_labels[x]
            )
        if froi2_labels is not None:
            df_summary["froi2"] = df_summary["froi2"].apply(
                lambda x: froi2_labels[x]
            )
            df_detail["froi2"] = df_detail["froi2"].apply(
                lambda x: froi2_labels[x]
            )
        if is_parcels1:
            df_summary = df_summary.rename(columns={"froi1": "parcel1"})
            df_detail = df_detail.rename(columns={"froi1": "parcel1"})
        if is_parcels2:
            df_summary = df_summary.rename(columns={"froi2": "parcel2"})
            df_detail = df_detail.rename(columns={"froi2": "parcel2"})
        if not is_parcels1:
            df_summary["subject1"] = subject1
            df_detail["subject1"] = subject1
        if not is_parcels2:
            df_summary["subject2"] = subject2
            df_detail["subject2"] = subject2

        # Save and return the results
        self._data_summary = df_summary
        self._data_detail = df_detail
        new_overlap_info = pd.DataFrame(
            {
                "froi1": [self.froi1],
                "froi2": [self.froi2],
                "kind": [self.kind],
                "orthogonalization": [self.orthogonalization],
                "customized_run1": [self.run1],
                "customized_run2": [self.run2],
            }
        )
        self._save(new_overlap_info)
        return self._data_summary, self._data_detail

    @staticmethod
    def _run(
        froi1_masks: np.ndarray, froi2_masks: np.ndarray, kind: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the overlap estimation.

        :param froi1_masks: The fROI masks for the first set of parcels or
            fROIs, with shape (n_runs, n_voxels).
        :type froi1_masks: np.ndarray
        :param froi2_masks: The fROI masks for the second set of parcels or
            fROIs, with shape (n_runs, n_voxels).
        :type froi2_masks: np.ndarray
        :param kind: Kind of overlap to estimate. Options are 'overlap' and
            'dice'.
        :type kind: str

        :return: The overlap estimates averaged across runs, and the overlap
            estimates detailed by run.
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """
        if len(froi1_masks.shape) != 2:
            raise ValueError("froi1_masks must have shape (n_runs, n_voxels)")
        if len(froi2_masks.shape) != 2:
            raise ValueError("froi2_masks must have shape (n_runs, n_voxels)")
        if froi1_masks.shape != froi2_masks.shape:
            raise ValueError(
                "froi1_masks and froi2_masks must have the same shape"
            )

        # Reduce the data
        valid_idx1 = np.any(
            ~np.isnan(froi1_masks) & (froi1_masks != 0), axis=0
        )
        valid_idx2 = np.any(
            ~np.isnan(froi2_masks) & (froi2_masks != 0), axis=0
        )
        any_valid_idx = valid_idx1 | valid_idx2
        froi1_masks = froi1_masks[:, any_valid_idx]
        froi2_masks = froi2_masks[:, any_valid_idx]

        froi1_masks_expanded = froi1_masks[None, ...]
        froi1_labels = np.unique(froi1_masks)
        froi1_labels = froi1_labels[
            ~np.isnan(froi1_labels) & (froi1_labels != 0)
        ]
        froi1_masks_expanded = (
            froi1_masks_expanded == froi1_labels[:, None, None]
        ).astype(int)

        froi2_masks_expanded = froi2_masks[None, ...]
        froi2_labels = np.unique(froi2_masks)
        froi2_labels = froi2_labels[
            ~np.isnan(froi2_labels) & (froi2_labels != 0)
        ]
        froi2_masks_expanded = (
            froi2_masks_expanded == froi2_labels[:, None, None]
        ).astype(int)

        # Compute intersection
        intersection = np.einsum(
            "mrv, nrv -> mnr", froi1_masks_expanded, froi2_masks_expanded
        )
        froi1_sum = np.einsum("mrv -> mr", froi1_masks_expanded)[:, None, :]
        froi2_sum = np.einsum("nrv -> nr", froi2_masks_expanded)[None, :, :]
        min_sum = np.minimum(froi1_sum, froi2_sum)

        if kind == "dice":
            overlap = 2 * intersection / (froi1_sum + froi2_sum)
        else:
            overlap = intersection / min_sum

        df_detail = pd.DataFrame(
            {  # run, froi1, froi2, overlap
                "run": np.tile(
                    np.arange(froi1_masks.shape[0]),
                    len(froi1_labels) * len(froi2_labels),
                ),
                "froi1": np.repeat(
                    froi1_labels, froi1_masks.shape[0] * len(froi2_labels)
                ),
                "froi2": np.repeat(
                    np.tile(froi2_labels, len(froi1_labels)),
                    froi2_masks.shape[0],
                ),
                "overlap": overlap.flatten(),
            }
        )
        df_detail = df_detail.sort_values(
            by=["run", "froi1", "froi2"]
        ).reset_index(drop=True)
        df_summary = (
            df_detail.groupby(["froi1", "froi2"])
            .agg({"overlap": "mean"})
            .reset_index()
        )
        return df_summary, df_detail
