import numpy as np
from scipy.ndimage import convolve1d
from scipy.special import erf
from ..localizer import Localizer, get_localizers
from ..utils import (
    get_parcels_path,
    get_contrast_path,
    get_parcels_config_path,
    harmonic_mean
)
import glob
from typing import List, Union, Optional
from nibabel.nifti1 import Nifti1Image
from nilearn.image import smooth_img
import json


class ParcelsGenerator:
    def __init__(
        self,
        smoothing_kernel_size: Union[float, List[float]] = 8,
        overlap_thr_vox: float = 0.1,
        use_spm_smooth: Optional[bool] = True,
    ):
        """
        ROI parcellation generator.

        Parameters
        ----------
        smoothing_kernel_size : Union[float, List[float]], optional
            Full-width at half maximum (FWHM) of the Gaussian kernel
            along each axis, by default 8.
        overlap_thr_vox : float, optional
            Voxel-level minimal proportion of subjects overlap when
            constructing ROI parcellation, by default 0.1.
        use_spm_smooth : Optional[bool], optional
            Whether to use the same smoothing kernel as SPM, by default True.
        """
        if isinstance(smoothing_kernel_size, (int, float)):
            smoothing_kernel_size = [smoothing_kernel_size] * 3
        self.smoothing_kernel_size = smoothing_kernel_size
        self.overlap_thr_vox = overlap_thr_vox
        self.use_spm_smooth = use_spm_smooth
        self.parcels_template_img = None
        self.parcels = None
        self.info = []
        self._data = []
        self._binary_masks = None

    def add_subjects(
        self,
        subjects: List[str],
        task: str,
        localizer: Union[str, Localizer],
        threshold_type: str,
        threshold_value: float,
    ):
        """
        Add subjects and localizers for ROI parcellation.

        Parameters
        ----------
        subjects : List[str]
            List of subject IDs.
        task : str
            Task name.
        localizer : Union[str, Localizer]
            Localizer name or Localizer object.
        """
        self.info.append(
            (subjects, task, localizer, threshold_type, threshold_value)
        )

        localizer_maps = get_localizers(
            subjects, task, localizer, threshold_type, threshold_value
        )

        self._data.extend(localizer_maps)

        binary_masks = []
        for subjecti, subject in enumerate(subjects):
            binary_masks.append(
                np.mean(localizer_maps[subjecti], axis=0) > 0.5
            )
        binary_masks = np.stack(binary_masks, axis=0)

        if self._binary_masks is None:
            self._binary_masks = binary_masks
        else:
            self._binary_masks = np.stack(
                [self._binary_masks, binary_masks], axis=0
            )

        # Might switch to a better solution in the future
        if self.parcels_template_img is None:
            contrast_search_path = get_contrast_path(
                subjects[0],
                task,
                "*",
                "*",
                "p",
            )
            contrast_path = glob.glob(contrast_search_path)[0]
            self.parcels_template_img = Nifti1Image.from_filename(
                contrast_path
            )

    def generate(self) -> Nifti1Image:
        if self._data is None:
            raise ValueError("No subjects added for parcellation")
        overlap_map = np.mean(self._binary_masks, axis=0).reshape(
            self.parcels_template_img.get_fdata().shape
        )
        if self.use_spm_smooth: # SPM-style smoothing
            smoothed_map = self._smooth_array(
                overlap_map,
                self.parcels_template_img.affine,
                self.smoothing_kernel_size,
            )
        else: # Nilearn-style smoothing
            smoothed_map = smooth_img(
                Nifti1Image(overlap_map, self.parcels_template_img.affine),
                fwhm=self.smoothing_kernel_size,
            ).get_fdata()
        smoothed_map[smoothed_map < self.overlap_thr_vox] = np.nan

        self._overlap = overlap_map
        self._overlap_smooothed = smoothed_map

        self.parcels = self._watershed(-smoothed_map)

        return Nifti1Image(self.parcels, self.parcels_template_img.affine)

    def select(
        self,
        min_voxel_size: Optional[int] = 0,
        overlap_thr_roi: Optional[float] = 0,
        inplace: Optional[bool] = False,
    ) -> Nifti1Image:
        """
        Selects parcels based on the minimal voxel size and the minimal
        proportion of subjects overlap.

        Parameters
        ----------
        min_voxel_size : int
            Minimal voxel size for each parcel.
        overlap_thr_roi : float
            Minimal proportion of subjects overlap for each parcel.
        inplace : bool, optional
            Whether to modify the current parcellation in place,
            by default False.
        """
        if self.parcels is None:
            raise ValueError("Parcellation not generated yet")
        parcels = self.parcels.copy()
        unique_parcels = np.unique(parcels)
        for parcel in unique_parcels:
            if parcel == 0:
                continue
            parcel_mask = parcels == parcel
            parcel_size = np.sum(parcel_mask)
            if parcel_size < min_voxel_size:
                parcels[parcel_mask] = 0
            else:
                subject_coverage = np.zeros(len(self._data))
                for subjecti, data in enumerate(self._data):
                    subject_coverage[subjecti] = harmonic_mean(
                        np.sum(data[:, parcel_mask.flatten()], axis=1)
                    ) > 0
                if np.mean(subject_coverage) < overlap_thr_roi:
                    parcels[parcel_mask] = 0

        if inplace:
            self.parcels = parcels

        labels_remaining = np.unique(parcels)
        labels_remaining = labels_remaining[labels_remaining != 0]
        print(f"Remaining parcels: {len(labels_remaining)}")

        return Nifti1Image(parcels, self.parcels_template_img.affine)

    def save(self, parcels_name: str):
        parcels_path = get_parcels_path(parcels_name)
        if self.parcels is None:
            raise ValueError("Parcellation not generated yet")
        parcels_img = Nifti1Image(
            self.parcels, self.parcels_template_img.affine
        )
        parcels_img.to_filename(parcels_path)

        parcels_config_path = get_parcels_config_path(parcels_name)
        with open(parcels_config_path, "w") as f:
            json.dump(
                {
                    "smoothing_kernel_size": self.smoothing_kernel_size,
                    "overlap_thr_vox": self.overlap_thr_vox,
                    "info": self.info,
                },
                f,
            )

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
