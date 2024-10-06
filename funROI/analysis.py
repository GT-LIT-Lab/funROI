from . import utils
from typing import List, Optional, Tuple
from nibabel.nifti1 import Nifti1Image
from nilearn.image import new_img_like, smooth_img
import pandas as pd
import numpy as np
import json
import os


class ParcelGenerator:
    def __init__(self, threshold_type: str, threshold_value: float, smoothing_kernel_size: float = 8, min_prop_roi: float = 0.1):
        self.threshold_type = threshold_type
        self.threshold_value = threshold_value
        self.smoothing_kernel_size = smoothing_kernel_size
        self.min_prop_roi = min_prop_roi
        self.parcels = None
        self._localizers = []
        self.selected = False

    def add_localizer(self, subject: str, task: str, localizer: str):
        self._localizers.append((subject, task, localizer))

    def run(self):
        self._subject_masks = {}
        for subject, task, localizer_name in self._localizers:
            subject_run_maps = []
            runs = utils.get_runs(subject, task)
            for run_i in runs:
                run_mask = utils.get_localizer(subject, task, f"orth{run_i}", localizer_name, self.threshold_type, self.threshold_value, create_if_not_exist=True)
                subject_run_maps.append(run_mask.get_fdata())

            # individual binary mask, 1 if > 50% of the runs are 1
            self._subject_masks[subject] = np.mean(subject_run_maps, axis=0) > 0.5

        overlap_map = np.mean(list(self._subject_masks.values()), axis=0)
        overlap_map = new_img_like(run_mask, overlap_map)
        self._overlap_map = overlap_map # for debugging
        overlap_map_smoothed = smooth_img(overlap_map, self.smoothing_kernel_size).get_fdata()
        overlap_map_smoothed[overlap_map_smoothed < self.min_prop_roi] = np.nan
        self.parcels = self._watershed(-overlap_map_smoothed)
        self.parcels = new_img_like(overlap_map, self.parcels)
        self.selected = False
    

    def select_parcels(self, min_voxel_size: int = 10, min_prop_intersect: float = 0.5):
        self.selected = True
        self.min_voxel_size = min_voxel_size
        self.min_prop_intersect = min_prop_intersect

        parcel_data = self.parcels.get_fdata()
        new_parcels = np.zeros_like(parcel_data)

        for label in np.unique(parcel_data):
            if label == 0: continue
            parcel_map = parcel_data == label
            if np.sum(parcel_map) >= min_voxel_size:    # check number of voxels
                # TODO: check below
                # prop_roi by the proportion of subjects with non-zero intersection between the subject mask and the parcel
                prop_intersect = np.mean([np.sum(subject_mask * parcel_map) > 0 for subject_mask in self._subject_masks.values()])
                if prop_intersect >= min_prop_intersect:
                    new_parcels[parcel_map] = label
        self.parcels = new_img_like(self.parcels, new_parcels)
    

    def save(self, parcels_name: str):
        os.makedirs(utils.get_parcel_folder(), exist_ok=True)
        self.parcels.to_filename(utils.get_parcels_path(parcels_name))

        # Save parcels configuration to a JSON file, which includes parcel generation settings
        parcels_config = {
            'parcels_name': parcels_name,
            'smoothing_kernel_size': self.smoothing_kernel_size,
            'min_prop_roi': self.min_prop_roi,
            'localizers': self._localizers
        }
        if self.selected:
            parcels_config['min_voxel_size'] = self.min_voxel_size
            parcels_config['min_prop_intersect'] = self.min_prop_intersect

        with open(utils.get_parcels_config_path(parcels_name), 'w') as f:
            json.dump(parcels_config, f)
    

    @staticmethod
    def _watershed(A: np.ndarray) -> np.ndarray:
        sA = A.shape

        # Zero-pad & sort
        A_flat = A.flatten(order='F')
        IDX = np.where(~np.isnan(A_flat))[0]

        a = A_flat[IDX]
        sort_idx = np.argsort(a, kind='stable')
        a = a[sort_idx]
        idx = IDX[sort_idx]

        # Convert linear indices to subscripts and adjust for zero-padding
        pidx = np.unravel_index(idx, sA, order='F')
        pidx_padded = [coord + 1 for coord in pidx]  # Add 1 for zero-padding
        sA_padded = tuple(dim + 2 for dim in sA)
        eidx = np.ravel_multi_index(pidx_padded, sA_padded, order='F')

        # Neighbors (max-connected; i.e., 26-connected for 3D)
        dd = np.meshgrid(*([np.arange(1, 4)] * len(sA_padded)), indexing='ij')
        dd_flat = [d.flatten() for d in dd]
        d = np.ravel_multi_index(dd_flat, sA_padded, order='F')
        center_idx = (len(d) - 1) // 2
        center = d[center_idx]
        d = d - center
        d = d[d != 0]

        # Initialize labels
        C = np.zeros(sA_padded, dtype=int, order='F')
        C_flat = C.flatten(order='F')

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
        D = D_flat.reshape(sA, order='F')
        return D
    

class SubjectSpecificLocalizer:
    def __init__(self, subject: str, parcels: Optional[str] = None, threshold_type: Optional[str] = None, threshold_value: Optional[float] = None):
        # TODO: whether parcels are strictly required
        self.subject = subject
        self.parcels_name = parcels
        self.parcels = utils.get_parcels(parcels_name=parcels)
        self.threshold_type = threshold_type
        self.threshold_value = threshold_value


    def localize(self, localizer: str, task: str) -> Nifti1Image:
        return self._localize(localizer, task, 'all')
        

    def _localize(self, localizer: str, task: str, run_label: str) -> Nifti1Image:
        assert self.parcels is not None, "Parcels must be set for subject-specific localization."
        assert self.threshold_type is not None and self.threshold_value is not None, "Threshold must be set for subject-specific localization." 
        return utils.get_ssroi(self.subject, task, run_label, localizer, self.threshold_type, self.threshold_value, self.parcels_name, self.parcels, create_if_not_exist=True)
    

    def _orthogonalize(self, task: str) -> Tuple[List[str], List[str]]:
        runs = utils.get_runs(self.subject, task)
        orig = [str(run) for run in runs]
        orth = [f"orth{run}" for run in runs]
        return orig, orth
    

class EffectProfiler(SubjectSpecificLocalizer):
    def __init__(self, subject: str, localizer_task: str, localizer: str, effect_task: str, effect_contrast: str, 
                 parcels: Optional[str] = None, threshold_type: Optional[str] = None, threshold_value: Optional[float] = None):
        super().__init__(subject=subject, parcels=parcels, threshold_type=threshold_type, threshold_value=threshold_value)
        self.localizer_task = localizer_task
        self.localizer = localizer
        self.effect_task = effect_task
        self.effect_contrast = effect_contrast

    # TODO: no parcels case is not yet supported. Will discuss this
    def run(self) -> pd.DataFrame:
        if self.localizer_task == self.effect_task:
            effect_split, localizer_split = super()._orthogonalize(self.localizer_task)
        else:
            effect_split, localizer_split = ['all'], ['all']
        
        results_splits = pd.DataFrame(columns=['ssroi', 'n_voxels', 'split', 'mean_effect_size'])
        for split_i in range(len(effect_split)):
            localizer_ssroi_map = super()._localize(self.localizer, self.localizer_task, localizer_split[split_i]).get_fdata()
            for ssroi in np.unique(localizer_ssroi_map):
                if ssroi == 0: continue
                ssroi_mask = localizer_ssroi_map == ssroi
                effect_map = utils.get_contrast(self.subject, self.effect_task, effect_split[split_i], self.effect_contrast, 'effect').get_fdata()
                effect_map_on_ssroi = effect_map[ssroi_mask]
                mean_effect_size = np.nanmean(effect_map_on_ssroi)
                results_splits = pd.concat([results_splits, pd.DataFrame({'ssroi': ssroi, 'n_voxels': np.sum(ssroi_mask), 'split': split_i, 'mean_effect_size': mean_effect_size}, index=[0])], ignore_index=True)
 
        # Average the effect sizes across splits
        results = results_splits.groupby(['ssroi', 'n_voxels'])['mean_effect_size'].mean().reset_index()
        return results
    

class SpatialCorrelationAnalyzer(SubjectSpecificLocalizer):
    def __init__(self, subject: str, localizer_task: Optional[str] = None, localizer: Optional[str] = None,
                 parcels: Optional[str] = None, threshold_type: Optional[str] = None, threshold_value: Optional[float] = None):
        super().__init__(subject=subject, parcels=parcels, threshold_type=threshold_type, threshold_value=threshold_value)
        self.localizer_task, self.localizer = localizer_task, localizer
        self.contrasts = [] 


    def add_contrast(self, task: str, contrast: str):
        self.contrasts.append((task, contrast))


    # TODO: discuss the method
    def run(self) -> pd.DataFrame:
        correlation_table = pd.DataFrame(columns=['ssroi', 'task1', 'contrast1', 'task2', 'contrast2', 'split', 'FishersZ'])
        for ssroi_label in np.unique(self.parcels.get_fdata()):
            if ssroi_label == 0: continue
            parcel_mask = self.parcels.get_fdata() == ssroi_label
            for task1, contrast1 in self.contrasts:
                for task2, contrast2 in self.contrasts:
                    if task1 == task2:
                        [split1, split2] = super()._orthogonalize(task1)
                    else:
                        split1, split2 = ['all'], ['all']

                    for split_i in range(len(split1)):
                        effect_1 = utils.get_contrast(self.subject, task1, split1[split_i], contrast1, 'effect', create_if_not_exist=True).get_fdata()
                        effect_2 = utils.get_contrast(self.subject, task2, split2[split_i], contrast2, 'effect', create_if_not_exist=True).get_fdata()

                        if self.localizer is not None:
                            localizer_ssroi_map = super()._localize(self.localizer, self.localizer_task, 'all').get_fdata()
                            ssroi_mask = localizer_ssroi_map == ssroi_label
                            effect_1_on_ssroi = effect_1[ssroi_mask]
                            effect_2_on_ssroi = effect_2[ssroi_mask]
                        else:
                            effect_1_on_ssroi = effect_1[parcel_mask]
                            effect_2_on_ssroi = effect_2[parcel_mask]

                        non_nan_mask = ~np.isnan(effect_1_on_ssroi) & ~np.isnan(effect_2_on_ssroi)
                        effect_1_on_ssroi = effect_1_on_ssroi[non_nan_mask]
                        effect_2_on_ssroi = effect_2_on_ssroi[non_nan_mask]

                        fishers_z = np.arctanh(np.corrcoef(effect_1_on_ssroi, effect_2_on_ssroi)[0, 1])
                        correlation_table = pd.concat([correlation_table, pd.DataFrame({'ssroi': ssroi_label, 'task1': task1, 'contrast1': contrast1, 'task2': task2, 'contrast2': contrast2, 'split': split_i, 'FishersZ': fishers_z}, index=[0])], ignore_index=True)

        # Average the correlation across splits
        correlation_table = correlation_table.groupby(['ssroi', 'task1', 'contrast1', 'task2', 'contrast2'])['FishersZ'].mean().reset_index()
        return correlation_table


class OverlapAnalyzer():
    def __init__(self):
        self.imgs = []
        self.img_labels = []


    def add_ssroi(self, subject: str, task: str, localizer: str, threshold_type: str, threshold_value: float, parcels_name: str):
        parcels = utils.get_parcels(parcels_name)
        ssroi_img = utils.get_ssroi(subject, task, 'all', localizer, threshold_type, threshold_value, parcels_name, parcels, create_if_not_exist=True)
        self.imgs.append(ssroi_img)
        self.img_labels.append(f"{subject}_{task}_{localizer}")


    def add_parcels(self, parcels_name: str):
        self.imgs.append(utils.get_parcels(parcels_name))
        self.img_labels.append(parcels_name)


    def run(self) -> pd.DataFrame:
        overlap_table = pd.DataFrame(columns=['img_1', 'img_2', 'ssroi_1', 'ssroi_2', 'n_voxels_1', 'n_voxels_2', 'intersection', 'overlap_ratio'])
        for img1_i, img1 in enumerate(self.imgs):
            for img2_i in range(img1_i + 1, len(self.imgs)):
                img2 = self.imgs[img2_i]
                img1_data = img1.get_fdata()
                img2_data = img2.get_fdata()
                for img1_label in np.unique(img1_data):
                    if img1_label == 0: continue
                    img1_mask = img1_data == img1_label
                    for img2_label in np.unique(img2_data):
                        if img2_label == 0: continue
                        img2_mask = img2_data == img2_label
                        # Overlap defined as intersection divided by minimum of the two
                        intersection = np.sum(img1_mask * img2_mask)
                        overlap = intersection / min(np.sum(img1_mask), np.sum(img2_mask))
                        overlap_table = pd.concat([overlap_table, pd.DataFrame({'img_1': self.img_labels[img1_i], 'img_2': self.img_labels[img2_i], 'ssroi_1': img1_label, 'ssroi_2': img2_label, 'n_voxels_1': np.sum(img1_mask), 'n_voxels_2': np.sum(img2_mask), 'intersection': intersection, 'overlap_ratio': overlap}, index=[0])], ignore_index=True)

        return overlap_table
