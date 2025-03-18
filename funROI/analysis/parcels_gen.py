from typing import List, Optional, Union, Tuple
from nibabel.nifti1 import Nifti1Image
from ..utils import validate_arguments
from .. import get_analysis_output_folder
from ..froi import _get_froi_path, FROIConfig, _get_froi_runs, _create_froi
import warnings
import numpy as np
from scipy.ndimage import convolve1d
from scipy.special import erf
from nilearn.image import load_img, smooth_img
from pathlib import Path
import json
import pandas as pd


class ParcelsGenerator:
    """Generate parcels using with a set of subject data.

    .. warning:: In order to run parcels generation, all images are required to
        be in the same space and have the same dimensions. This is assumed
        throughout the process.

    :param parcels_name: Name of the parcels to generate.
    :type parcels_name: str
    :param smoothing_kernel_size: Size of the smoothing kernel in mm. If a
        list, the smoothing is performed with a different kernel size for each
        dimension. Default is 8.
    :type smoothing_kernel_size: Optional[Union[float, List[float]]]
    :param overlap_thr_vox: Minimum overlap proportion for a voxel to be
        included in the parcel formation. Default is 0.1.
    :type overlap_thr_vox: Optional[float]
    :param min_voxel_size: Minimum size of the parcels in voxels. Can be used
        post-hoc to filter out small parcels. Default is 0.
    :type min_voxel_size: Optional[int]
    :param overlap_thr_roi: Minimum overlap proportion for a parcel to be
        included in the final set of parcels. Can be used post-hoc to filter
        out parcels that have low overlap with subject activation data.
        Default is 0.
    :type overlap_thr_roi: Optional[float]
    :param use_spm_smooth: Whether to use SPM's smoothing function to smooth
        the data before parcel generation. If False, the smoothing is done
        using Nilearn's Gaussian smoothing. Default is True.
    :type use_spm_smooth: Optional[bool]
    """

    def __init__(
        self,
        parcels_name: str,
        smoothing_kernel_size: Optional[Union[float, List[float]]] = 8,
        overlap_thr_vox: Optional[float] = 0.1,
        min_voxel_size: Optional[int] = 0,
        overlap_thr_roi: Optional[float] = 0,
        use_spm_smooth: Optional[bool] = True,
    ):
        self.parcels_name = parcels_name
        self.smoothing_kernel_size = smoothing_kernel_size
        self.overlap_thr_vox = overlap_thr_vox
        self.min_voxel_size = min_voxel_size
        self.overlap_thr_roi = overlap_thr_roi
        self.use_spm_smooth = use_spm_smooth
        self.configs = []
        self.parcels = None
        self._data = []
        self.img_shape = None
        self.img_affine = None
        self.parcel_info = None

    @validate_arguments(
        p_threshold_type={"none", "bonferroni", "fdr"},
        conjunction_type={"min", "max", "sum", "prod", "and", "or"},
    )
    def add_subjects(
        self,
        subjects: List[str],
        task: str,
        contrasts: List[str],
        p_threshold_type: str,
        p_threshold_value: float,
        conjunction_type: Optional[str] = "and",
    ):
        """
        Add subjects to the parcels generation.

        :param subjects: List of subject labels.
        :type subjects: List[str]
        :param task: Task name.
        :type task: str
        :param contrasts: List of contrast labels.
        :type contrasts: List[str]
        :param p_threshold_type: Type of p-value thresholding. One of
            "none", "bonferroni", or "fdr". Default is "none".
        :type p_threshold_type: str
        :param p_threshold_value: P-value threshold. Default is 0.05.
        :type p_threshold_value: float
        :param conjunction_type: Type of conjunction if multiple contrasts are
            used. One of "min", "max", "sum", "prod", "and", or "or". Default
            is "and".
        :type conjunction_type: str

        :raises ValueError: If any of the subjects are already added.
        :raises ValueError: If the subjects have different image shapes or
            affines from the previous data.
        """
        existing_subjects = []
        for config in self.configs:
            existing_subjects.extend(config["subjects"])

        subjects_redundant = set(subjects).intersection(existing_subjects)
        if subjects_redundant:
            raise ValueError(
                f"Subjects {subjects_redundant} are already added."
            )

        froi = FROIConfig(
            task=task,
            contrasts=contrasts,
            threshold_type=p_threshold_type,
            threshold_value=p_threshold_value,
            conjunction_type=conjunction_type,
            parcels=None,
        )

        new_data = []
        for subject in subjects:
            froi_runs = _get_froi_runs(subject, froi)
            subject_data = []
            for run in froi_runs:
                if len(froi_runs) == 2:
                    runs_ = froi_runs.copy()
                    runs_.remove(run)
                    run = runs_[0]
                else:
                    run = f"orth{run}"
                froi_pth = _get_froi_path(subject, run, froi)
                if not froi_pth.exists():
                    _create_froi(subject, froi, run)
                    if not froi_pth.exists():
                        raise FileNotFoundError(
                            f"Could not create the FROI data for subject "
                            f"{subject} run {run}."
                        )
                img = load_img(froi_pth)
                subject_data.append(img.get_fdata().flatten())
                if self.img_shape is None and self.img_affine is None:
                    self.img_shape = img.shape
                    self.img_affine = img.affine
                elif self.img_shape != img.shape or not np.allclose(
                    self.img_affine, img.affine
                ):
                    raise ValueError(
                        "All images must have the same shape and affine. "
                        f"Subject {subject} run {run} does not match the "
                        "previous images."
                    )
            subject_data = np.array(subject_data)
            if len(subject_data) == 0:
                raise ValueError(
                    f"No data found for subject {subject} and task {task}."
                )
            new_data.append(subject_data)
        self._data.extend(new_data)
        self.configs.append({"subjects": subjects, "froi": froi})

    @classmethod
    def _run_fast(
        cls,
        parcels_name: str,
        smoothing_kernel_size: Optional[Union[float, List[float]]] = 8,
        overlap_thr_vox: Optional[float] = 0.1,
        use_spm_smooth: Optional[bool] = True,
    ) -> "ParcelsGenerator":
        parcel_gen = ParcelsGenerator(
            parcels_name,
            smoothing_kernel_size,
            overlap_thr_vox,
            use_spm_smooth=use_spm_smooth,
        )
        # Run parcel generation with a pre-calculated overlapping map.
        # Use this method only if you are sure what you are doing.

        parcel_info_pth = (
            cls._get_analysis_parcels_folder(parcels_name)
            / f"parcels-{parcels_name}_sm-{smoothing_kernel_size}_spmsmooth-{use_spm_smooth}_voxthres-{overlap_thr_vox}_info.csv"
        )
        parcels_pth = (
            cls._get_analysis_parcels_folder(parcels_name)
            / f"parcels-{parcels_name}_sm-{smoothing_kernel_size}_spmsmooth-{use_spm_smooth}_voxthres-{overlap_thr_vox}_roithres-0_sz-0.nii.gz"
        )

        if parcel_info_pth.exists() and parcels_pth.exists():
            parcel_gen.parcel_info = pd.read_csv(parcel_info_pth)
            parcel_gen.parcels = load_img(parcels_pth).get_fdata()
        else:
            overlap_map_pth = (
                cls._get_analysis_parcels_folder(parcels_name)
                / f"parcels-{parcels_name}_overlap.nii.gz"
            )
            if not overlap_map_pth.exists():
                raise FileNotFoundError(
                    "Overlapping map not found. Please double-check the "
                    "parcels name and configurations."
                )
            overlap_map = load_img(overlap_map_pth)
            parcel_gen.overlap_map = overlap_map.get_fdata()
            parcel_gen.img_shape = overlap_map.shape
            parcel_gen.img_affine = overlap_map.affine
            _, parcel_gen.parcels = cls._run(
                parcel_gen.overlap_map[None, ...],
                parcel_gen.img_shape,
                parcel_gen.img_affine,
                smoothing_kernel_size,
                overlap_thr_vox,
                use_spm_smooth,
            )

            parcel_info_data = []
            for parcel in np.unique(parcel_gen.parcels):
                if parcel == 0:
                    continue
                parcel_mask = parcel_gen.parcels == parcel
                parcel_size = np.sum(parcel_mask)
                parcel_info_data.append([parcel, parcel_size])
            parcel_gen.parcel_info = pd.DataFrame(
                parcel_info_data, columns=["id", "size"]
            )
            parcel_gen._save()

    def run(self) -> Nifti1Image:
        """
        Run the parcels generation. Both the generated parcels and the filtered
        parcels are stored in the analysis output folder.

        :return: a labelled image of the parcels is returned.
        :rtype: Nifti1Image
        """
        if self.parcel_info is None:
            binary_masks = [np.mean(dat, axis=0) > 0.5 for dat in self._data]
            self.overlap_map, self.parcels = self._run(
                binary_masks,
                self.img_shape,
                self.img_affine,
                self.smoothing_kernel_size,
                self.overlap_thr_vox,
                self.use_spm_smooth,
            )

            parcel_info_data = []
            for parcel in np.unique(self.parcels):
                if parcel == 0:
                    continue
                parcel_mask = self.parcels == parcel
                parcel_size = np.sum(parcel_mask)
                subject_coverage = np.zeros(len(self._data))
                for subjecti, data in enumerate(self._data):
                    subject_coverage[subjecti] = (
                        self._harmonic_mean(
                            np.sum(data[:, parcel_mask.flatten()], axis=1)
                        )
                        > 0
                    )
                parcel_info_data.append(
                    [parcel, parcel_size, np.mean(subject_coverage)]
                )
            self.parcel_info = pd.DataFrame(
                parcel_info_data, columns=["id", "size", "roi_overlap"]
            )
            self._save()

        if self.min_voxel_size != 0 or self.overlap_thr_roi != 0:
            self.parcels = self._filter(
                self.parcels,
                self.parcel_info,
                self.overlap_thr_roi,
                self.min_voxel_size,
            )

        self._save()

        return Nifti1Image(self.parcels, self.img_affine)

    @classmethod
    def _run(
        cls,
        binary_masks: List[np.ndarray],
        img_shape: np.ndarray,
        img_affine: np.ndarray,
        smoothing_kernel_size: Union[float, List[float]],
        overlap_thr_vox: float,
        use_spm_smooth: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the parcels generation.

        :param binary_masks: list of binary masks for each subject. Each mask
            is a flattened 1D array of (n_voxels,) shape.
        :type binary_masks: List[np.ndarray]
        :param img_shape: Shape of the image.
        :type img_shape: np.ndarray (3,)
        :param img_affine: Affine matrix of the image.
        :type img_affine: np.ndarray (4, 4)
        :param smoothing_kernel_size: Size of the smoothing kernel in mm. If a
            list, the smoothing is performed with a different kernel size for
            each dimension.
        :type smoothing_kernel_size: Union[float, List[float]]
        :param overlap_thr_vox: Minimum overlap proportion for a voxel to be
            included in the parcel formation.
        :type overlap_thr_vox: float
        :param use_spm_smooth: Whether to use SPM's smoothing function to smooth
            the data before parcel generation. If False, the smoothing is done
            using Nilearn's Gaussian smoothing.
        :type use_spm_smooth: bool

        :return: The probablistic map, and generated parcels in 3D format.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        overlap_map = np.mean(binary_masks, axis=0).reshape(img_shape)

        if use_spm_smooth:
            smoothed_map = cls._smooth_array(
                overlap_map.copy(),
                img_affine,
                smoothing_kernel_size,
            )
        else:
            smoothed_map = smooth_img(
                Nifti1Image(overlap_map.copy(), img_affine),
                fwhm=smoothing_kernel_size,
            ).get_fdata()

        smoothed_map[smoothed_map < overlap_thr_vox] = np.nan
        parcels = cls._watershed(-smoothed_map)
        return overlap_map, parcels

    def filter(
        self,
        overlap_thr_roi: Optional[float] = 0,
        min_voxel_size: Optional[int] = 0,
    ) -> Nifti1Image:
        """
        Filter the parcels with new filtering parameters. The filtered results
        are stored in the analysis output folder.

        :return: a labelled image of the filtered parcels is returned.
        :rtype: Nifti1Image

        :raises RuntimeError: If the parcels have not been generated yet.
        """

        if self.parcels is None:
            raise RuntimeError(
                "No parcels to filter. Run the parcels generation first."
            )
        if overlap_thr_roi != 0 and overlap_thr_roi <= self.overlap_thr_roi:
            warnings.warn(
                "The new overlap_thr_roi is lower than the current setup. "
                "The filtering will not be applied."
            )
            overlap_thr_roi = 0.0
        if min_voxel_size != 0 and min_voxel_size <= self.min_voxel_size:
            warnings.warn(
                "The new min_voxel_size is lower than the current setup. "
                "The filtering will not be applied."
            )
            min_voxel_size = 0

        if overlap_thr_roi != 0 or min_voxel_size != 0:
            self.parcels = self._filter(
                self.parcels,
                self._data,
                overlap_thr_roi,
                min_voxel_size,
            )
            if overlap_thr_roi != 0:
                self.overlap_thr_roi = overlap_thr_roi
            if min_voxel_size != 0:
                self.min_voxel_size = min_voxel_size

            self._save()

        return Nifti1Image(self.parcels, self.img_affine)

    @classmethod
    def _filter(
        cls,
        parcels: np.ndarray,
        parcel_info: pd.DataFrame,
        overlap_thr_roi: float,
        min_voxel_size: int,
    ) -> np.ndarray:
        """
        Filter the parcels based on overlap and size.

        :param parcels: Parcels to filter.
        :type parcels: np.ndarray
        :param parcel_info: DataFrame containing information about the parcels.
            Columns: id, size, roi_overlap.
        :type parcel_info: pd.DataFrame
        :param overlap_thr_roi: Minimum overlap proportion for a parcel to be
            included in the final set of parcels.
        :type overlap_thr_roi: float
        :param min_voxel_size: Minimum size of the parcels in voxels.
        :type min_voxel_size: int

        :return: Filtered parcels.
        :rtype: np.ndarray
        """
        filtered_parcels = parcels.copy()
        unique_parcels = np.unique(parcels)
        for parcel in unique_parcels:
            if parcel == 0:
                continue
            parcel_mask = parcels == parcel
            if (
                parcel_info.loc[
                    parcel_info["id"] == parcel, "roi_overlap"
                ].values[0]
                < overlap_thr_roi
            ):
                filtered_parcels[parcel_mask] = 0
            if (
                parcel_info.loc[parcel_info["id"] == parcel, "size"].values[0]
                < min_voxel_size
            ):
                filtered_parcels[parcel_mask] = 0
        return filtered_parcels

    @staticmethod
    def _get_analysis_parcels_folder(parcels_name: str) -> Path:
        return (
            get_analysis_output_folder()
            / f"parcels"
            / f"parcels-{parcels_name}"
        )

    def _save(self):
        parcels_info_pth = (
            self._get_analysis_parcels_folder(self.parcels_name)
            / f"parcels-{self.parcels_name}_config.json"
        )
        parcels_info_pth.parent.mkdir(parents=True, exist_ok=True)
        if not parcels_info_pth.exists():
            with open(parcels_info_pth, "w") as f:
                json.dump({"configs": self.configs}, f)

        overlap_map_pth = (
            self._get_analysis_parcels_folder(self.parcels_name)
            / f"parcels-{self.parcels_name}_overlap.nii.gz"
        )
        if not overlap_map_pth.exists():
            overlap_map_img = Nifti1Image(self.overlap_map, self.img_affine)
            overlap_map_img.to_filename(overlap_map_pth)

        parcel_info_pth = (
            self._get_analysis_parcels_folder(self.parcels_name)
            / f"parcels-{self.parcels_name}_sm-{self.smoothing_kernel_size}_spmsmooth-{self.use_spm_smooth}_voxthres-{self.overlap_thr_vox}_info.csv"
        )
        if not parcel_info_pth.exists():
            self.parcel_info.to_csv(parcel_info_pth, index=False)

        parcels_pth = (
            self._get_analysis_parcels_folder(self.parcels_name)
            / f"parcels-{self.parcels_name}_sm-{self.smoothing_kernel_size}_spmsmooth-{self.use_spm_smooth}_voxthres-{self.overlap_thr_vox}_roithres-{self.overlap_thr_roi}_sz-{self.min_voxel_size}.nii.gz"
        )
        parcels_img = Nifti1Image(self.parcels, self.img_affine)
        parcels_img.to_filename(parcels_pth)

    @staticmethod
    def _harmonic_mean(data: np.ndarray) -> float:
        data = np.array(data).flatten()
        return len(data) / np.sum(1 / data)

    @classmethod
    def _watershed(cls, A: np.ndarray) -> np.ndarray:
        """
        Watershed algorithm for 3D images. A direct reimplementation of
        the watershed algorithm in spm_ss:
        https://github.com/alfnie/spm_ss/blob/master/spm_ss_watershed.m

        Parameters
        ----------
        A : ndarray
            3D array to be segmented
        """
        assert len(A.shape) == 3, "Input array must be 3D"
        sA = A.shape

        # Zero-pad & sort
        A_flat = A.flatten(order="F")
        IDX = np.where(~np.isnan(A_flat))[0]

        a = A_flat[IDX]
        sort_idx = np.argsort(a, kind="stable")
        a = a[sort_idx]
        idx = IDX[sort_idx]

        # Convert linear indices to subscripts and adjust for zero-padding
        pidx = np.unravel_index(idx, sA, order="F")
        pidx_padded = [coord + 1 for coord in pidx]  # Add 1 for zero-padding
        sA_padded = tuple(dim + 2 for dim in sA)
        eidx = np.ravel_multi_index(pidx_padded, sA_padded, order="F")

        # Neighbors (max-connected; i.e., 26-connected for 3D)
        dd = np.meshgrid(*([np.arange(1, 4)] * len(sA_padded)), indexing="ij")
        dd_flat = [d.flatten() for d in dd]
        d = np.ravel_multi_index(dd_flat, sA_padded, order="F")
        center_idx = (len(d) - 1) // 2
        center = d[center_idx]
        d = d - center
        d = d[d != 0]

        # Initialize labels
        C = np.zeros(sA_padded, dtype=int, order="F")
        C_flat = C.flatten(order="F")

        m = 1
        for n1 in range(len(eidx)):
            current_idx = eidx[n1]
            neighbor_idxs = current_idx + d

            # Remove out-of-bounds indices
            valid_mask = (neighbor_idxs >= 0) & (neighbor_idxs < C_flat.size)
            neighbor_idxs = neighbor_idxs[valid_mask]

            c = C_flat[neighbor_idxs]
            c = c[c > 0]

            if c.size == 0:
                C_flat[current_idx] = m
                m += 1
            elif np.all(np.diff(c) == 0):
                C_flat[current_idx] = c[0]

        D_flat = np.zeros(np.prod(sA), dtype=float)
        D_flat[idx] = C_flat[eidx]
        D = D_flat.reshape(sA, order="F")
        return D

    @classmethod
    def _gaussian_convolved_with_b_spline(
        cls, x: np.ndarray, s: float
    ) -> np.ndarray:
        """
        Convolves a Gaussian with a 1st-degree B-spline (triangular)
        basis function.

        Parameters
        ----------
        x : ndarray
            Input array
        s : float
            Standard deviation of the Gaussian kernel
        """
        var = s**2
        w1 = 0.5 * np.sqrt(2 / var)
        w2 = -0.5 / var
        w3 = np.sqrt(var / (2 * np.pi))

        # 1st degree B-spline convolution (triangular kernel)
        krn = 0.5 * (
            erf(w1 * (x + 1)) * (x + 1)
            + erf(w1 * (x - 1)) * (x - 1)
            - 2 * erf(w1 * x) * x
        ) + w3 * (
            np.exp(w2 * (x + 1) ** 2)
            + np.exp(w2 * (x - 1) ** 2)
            - 2 * np.exp(w2 * x**2)
        )
        krn[krn < 0] = 0  # Remove negative values
        return krn

    @classmethod
    def _smooth_array(
        cls, arr: np.ndarray, affine: np.ndarray, fwhm: np.ndarray
    ) -> np.ndarray:
        """
        Smooths a 3D array with Gaussian + 1st-degree B-spline (triangular)
        kernel

        Parameters
        ----------
        arr : ndarray
            3D array to be smoothed
        affine : ndarray
            4x4 affine matrix of the input array
        fwhm : ndarray
            Full-width at half maximum (FWHM) of the Gaussian kernel
            along each axis.
        """
        fwhm = np.asarray([fwhm]).ravel()
        fwhm = np.asarray([0.0 if elem is None else elem for elem in fwhm])
        affine = affine[:3, :3]  # Keep only the scale part.
        vox_size = np.sqrt(np.sum(affine**2, axis=0))
        sigma = fwhm / (np.sqrt(8 * np.log(2)) * vox_size)

        for n, s in enumerate(sigma):
            if s > 0.0:
                bound = np.round(6 * s).astype(int)
                kernel = cls._gaussian_convolved_with_b_spline(
                    np.arange(-bound, bound + 1), sigma[n]
                )
                kernel = kernel / np.sum(kernel)
                arr = convolve1d(
                    arr, kernel, output=arr, axis=n, mode="constant", cval=0.0
                )

        return arr
