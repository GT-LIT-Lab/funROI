from . import get_bids_deriv_folder
import os
from nibabel.nifti1 import Nifti1Image
from nilearn.image import load_img, math_img
from nilearn.glm import compute_fixed_effects
from typing import List, Optional
import pandas as pd
import warnings
import numpy as np
from statsmodels.stats.multitest import fdrcorrection
import ast
import re
import glob
from scipy.stats import t as t_dist


#### Path setting
# Contrast
def get_subject_contrast_folder(subject: str) -> str:
    return os.path.join(get_bids_deriv_folder(), 'funroi_contrasts', f'sub-{subject}')

def get_contrast_info_path(subject: str, task: str) -> str:
    return os.path.join(get_subject_contrast_folder(subject), f'sub-{subject}_task-{task}_contrasts.csv')

def get_contrast_path(subject: str, task: str, run_label: str, contrast: str, type: str) -> str:
    return os.path.join(get_subject_contrast_folder(subject), f'sub-{subject}_task-{task}_run-{run_label}_contrast-{contrast}_{type}.nii.gz')

# Localizer
def get_subject_localizer_folder(subject: str) -> str:
    return os.path.join(get_bids_deriv_folder(), 'funroi_localizers', f'sub-{subject}')

def get_localizer_info_path(subject: str, task: str) -> str:
    return os.path.join(get_subject_localizer_folder(subject), f'sub-{subject}_task-{task}_localizers.csv')

def get_localizer_path(subject: str, task: str, run_label: str, localizer: str, threshold_type: str, threshold_value: float) -> str:
    return os.path.join(get_subject_localizer_folder(subject), f'sub-{subject}_task-{task}_run-{run_label}_localizer-{localizer}_threshold-{threshold_type}-{threshold_value}_mask.nii.gz')

# ssROI
def get_subject_ssroi_folder(subject: str) -> str:
    return os.path.join(get_bids_deriv_folder(), 'funroi_ssroi', f'sub-{subject}')

def get_ssroi_path(subject: str, task: str, run_label: str, localizer: str, threshold_type: str, threshold_value: float, parcels: Nifti1Image) -> str:
    return os.path.join(get_subject_ssroi_folder(subject), f'sub-{subject}_task-{task}_run-{run_label}_localizer-{localizer}_threshold-{threshold_type}-{threshold_value}_parcel-{parcels}_mask.nii.gz')

# Models
def get_subject_model_folder(subject: str) -> str:
    return os.path.join(get_bids_deriv_folder(), 'funroi_models', f'sub-{subject}')

def get_dof_path(subject: str, task: str) -> str:
    return os.path.join(get_subject_model_folder(subject), f'sub-{subject}_task-{task}_dof.csv')

def get_first_level_model_paths_nilearn(subject: str, task: str, run: int):
    labels_path = os.path.join(get_subject_model_folder(subject), f'sub-{subject}_task-{task}_run-{run}_model-labels.pkl')
    estimates_path = os.path.join(get_subject_model_folder(subject), f'sub-{subject}_task-{task}_run-{run}_model-estimates.pkl')
    return labels_path, estimates_path

def get_design_matrix_path(subject: str, task: str, run: int) -> str:
    return os.path.join(get_subject_model_folder(subject), f'sub-{subject}_task-{task}_run-{run}_design-matrix.csv')

# Parcels
def get_parcel_folder() -> str:
    return os.path.join(get_bids_deriv_folder(), 'funroi_parcels')

def get_parcels_path(parcel_name: str) -> str:
    return os.path.join(get_parcel_folder(), f'{parcel_name}.nii.gz')

def get_parcels_config_path(parcel_name: str) -> str:
    return os.path.join(get_parcel_folder(), f'{parcel_name}.json')


#### Contrast
def register_contrast(subject: str, task: str, contrast_name: str, contrast_vector: List[float]):
    # The function takes a contrast vector and registers it to the contrast name
    # To get the contrast vector with formula definition and design columns, check nilearn.glm.expression_to_contrast_vector
    # https://nilearn.github.io/dev/modules/generated/nilearn.glm.expression_to_contrast_vector.html

    contrast_info_path = get_contrast_info_path(subject, task)
    if not os.path.exists(contrast_info_path):
        contrast_info = pd.DataFrame(columns=['contrast', 'vector'])
        matched = contrast_info[contrast_info['contrast'] == contrast_name]
        if not matched.empty:
            raise ValueError(f"Contrast {contrast_name} already exists for subject {subject}, task {task}.")
    else:
        contrast_info = pd.read_csv(contrast_info_path)

    contrast_info = pd.concat([contrast_info, pd.DataFrame({
        'contrast': [contrast_name],
        'vector': [contrast_vector]
    })], ignore_index=True)

    contrast_info.to_csv(contrast_info_path, index=False)


def get_contrast_info(subject: str, task: str, contrast: str) -> dict:
    contrast_info_path = get_contrast_info_path(subject, task)
    if not os.path.exists(contrast_info_path): return None

    contrast_info = pd.read_csv(contrast_info_path)
    contrast_info = contrast_info[contrast_info['contrast'] == contrast]
    contrast_info['vector'] = contrast_info['vector'].apply(ast.literal_eval)
    if contrast_info.empty: return None

    if len(contrast_info) > 1:
        warnings.warn(f"Multiple contrasts with the same name {contrast} found for subject {subject}, task {task}. Using the first one.")
    return contrast_info.to_dict(orient='records')[0]


def get_contrast(subject: str, task: str, run_label: str, contrast: str, type: str, create_if_not_exist: Optional[bool] = False) -> Nifti1Image:
    contrast_path = get_contrast_path(subject, task, run_label, contrast, type)
    if not os.path.exists(contrast_path): 
        if create_if_not_exist:
            return create_contrast(subject, task, run_label, contrast, type)
        return None
    return load_img(contrast_path)


def create_contrast(subject: str, task: str, run_label: str, contrast: str, type: str) -> Nifti1Image:
    contrast_info = get_contrast_info(subject, task, contrast)
    assert contrast_info is not None, f"Contrast {contrast} not found for subject {subject}, task {task}."

    if run_label.isdigit():
        warnings.warn(f"Creating contrast for single run {run_label}; only Nilearn first level model is supported.")
        from .first_level import _create_contrast_single_run_nilearn
        contrast_img = _create_contrast_single_run_nilearn(subject, task, int(run_label), contrast_info['vector'], type)
    elif 'orth' in run_label:
        contrast_img = _create_contrast_orthogonal(subject, task, int(run_label.replace('orth', '')), contrast, contrast_info['vector'], type)
    elif 'all' in run_label:
        contrast_img = _create_contrast_multi_run(subject, task, get_runs(subject, task), contrast, contrast_info['vector'], type)

    contrast_path = get_contrast_path(subject, task, run_label, contrast, type)
    contrast_img.to_filename(contrast_path)
    return contrast_img


def _create_contrast_orthogonal(subject: str, task: str, exclude_run_label: str, contrast_name: str, contrast_vector: List[float], type: str) -> Nifti1Image:
    runs = get_runs(subject, task)
    runs.remove(exclude_run_label)
    return _create_contrast_multi_run(subject, task, runs, contrast_name, contrast_vector, type)


def _create_contrast_multi_run(subject: str, task: str, runs: List[int], contrast_name: str, contrast_vector: List[float], type: str) -> Nifti1Image:
    effect_runs = []
    variance_runs = []
    dof_runs = []
    for run_i in runs:
        effect_map_path = get_contrast_path(subject, task, run_i, contrast_name, 'effect')
        variance_map_path = get_contrast_path(subject, task, run_i, contrast_name, 'variance')
        if not os.path.exists(effect_map_path) or not os.path.exists(variance_map_path):
            print(f"Contrast {contrast_name} not found for subject {subject}, task {task}, run {run_i}, creating it.")
            effect_map = create_contrast(subject, task, run_i, contrast_name, 'effect')
            variance_map = create_contrast(subject, task, run_i, contrast_name, 'variance')
        else:
            effect_map = load_img(effect_map_path)
            variance_map = load_img(variance_map_path)
        effect_runs.append(effect_map)
        variance_runs.append(variance_map)

        dof_path = get_dof_path(subject, task)
        dofs = pd.read_csv(dof_path)
        dof = dofs[(dofs['task'] == task) & (dofs['run'] == run_i)]['dof'].values[0]
        dof_runs.append(dof)

    contrast_map, variance_map, t_map, z_map = compute_fixed_effects(effect_runs, variance_runs, dofs=dof_runs, return_z_score=True)
    if type == 'effect':
        return contrast_map
    elif type == 'variance':
        return variance_map
    elif type == 't':
        return t_map
    elif type == 'z':
        return z_map
    if type == 'p': 
        p_values = 1 - t_dist.cdf(t_map.get_fdata(), dof_runs[0])
        return Nifti1Image(p_values, affine=t_map.affine, header=t_map.header)
    else:
        raise ValueError(f"Map type {type} not supported")


#### Localizer
def register_localizer(subject: str, task: str, contrasts: List[str], conjunction_type: str, localizer_name: str):
    os.makedirs(get_subject_localizer_folder(subject), exist_ok=True)
    localizer_info_path = get_localizer_info_path(subject, task)
    if not os.path.exists(localizer_info_path):
        localizer_info = pd.DataFrame(columns=['localizer', 'contrasts', 'conjunction_type'])
        matched = localizer_info[localizer_info['localizer'] == localizer_name]
        if not matched.empty:
            raise ValueError(f"Localizer {localizer_name} already exists for subject {subject}, task {task}.")
    else:
        localizer_info = pd.read_csv(localizer_info_path)

    localizer_info = pd.concat([localizer_info, pd.DataFrame({
        'localizer': [localizer_name],
        'contrasts': [contrasts],
        'conjunction_type': [conjunction_type]
    })], ignore_index=True)

    localizer_info.to_csv(localizer_info_path, index=False)


def get_localizer_info(subject: str, task: str, localizer: str) -> dict:
    localizer_info_path = get_localizer_info_path(subject, task)
    if not os.path.exists(localizer_info_path): return None

    localizer_info = pd.read_csv(localizer_info_path)
    localizer_info = localizer_info[localizer_info['localizer'] == localizer]
    localizer_info['contrasts'] = localizer_info['contrasts'].apply(ast.literal_eval)
    if localizer_info.empty: return None

    if len(localizer_info) > 1:
        warnings.warn(f"Multiple localizers with the same name {localizer} found for subject {subject}, task {task}. Using the first one.")
    return localizer_info.to_dict(orient='records')[0]


def get_localizer(subject: str, task: str, run_label: str, localizer: str, threshold_type: str, threshold_value: float,
                  create_if_not_exist: Optional[bool] = False) -> Nifti1Image:
    # First, check if it is a contrast-based localizer
    contrast_info = get_contrast_info(subject, task, localizer)
    if contrast_info is not None:
        p_map = get_contrast(subject, task, run_label, localizer, 'p', create_if_not_exist)
        mask = _threshold_p_map(p_map.get_fdata(), threshold_type, threshold_value)
        return Nifti1Image(mask, affine=p_map.affine, header=p_map.header)

    localizer_path = get_localizer_path(subject, task, run_label, localizer, threshold_type, threshold_value)
    if not os.path.exists(localizer_path): 
        if create_if_not_exist:
            return create_localizer(subject, task, run_label, localizer, threshold_type, threshold_value)
        return None
    return load_img(localizer_path)


def create_localizer(subject: str, task: str, run_label: str, localizer: str,
                     threshold_type: str, threshold_value: float) -> Nifti1Image:
    os.makedirs(get_subject_localizer_folder(subject), exist_ok=True)
    localizer_info = get_localizer_info(subject, task, localizer)
    assert localizer_info is not None, f"Localizer {localizer} not found for subject {subject}, task {task}."
    localizer_img = _create_localizer(subject, task, run_label, localizer_info['contrasts'], localizer_info['conjunction_type'], threshold_type, threshold_value)
    localizer_path = get_localizer_path(subject, task, run_label, localizer, threshold_type, threshold_value)
    localizer_img.to_filename(localizer_path)
    return localizer_img


def _create_localizer(subject: str, task: str, run_label: str, contrasts: List[str], conjunction_type: str,
                      threshold_type: str, threshold_value: float, mask: Optional[np.ndarray] = None) -> Nifti1Image:
    contrast_p_map_data = []
    affine = None
    header = None
    for contrast in contrasts:
        p_map = get_contrast(subject,  task, run_label, contrast, type='p')
        p_map_data = p_map.get_fdata()
        if mask is not None:
            p_map_data[mask == 0] = np.nan
        contrast_p_map_data.append(p_map_data)
        if affine is None or header is None:
            affine = p_map.affine
            header = p_map.header

    if conjunction_type in ['min', 'max', 'sum', 'prod']:
        # Combine first; threshold then
        contrast_p_map_combined = eval(f"np.{conjunction_type}(contrast_p_map_data, axis=0)")
        froi_mask = _threshold_p_map(contrast_p_map_combined, threshold_type, threshold_value)
    elif conjunction_type in ['and', 'or']:
        # Threshold first; combine then
        froi_mask = None
        for p_map in contrast_p_map_data:
            contrast_p_map_thresholded = _threshold_p_map(p_map, threshold_type, threshold_value)
            if froi_mask is None:
                froi_mask = contrast_p_map_thresholded
            else:
                froi_mask = eval(f"np.logical_{conjunction_type}(froi_mask, contrast_p_map_thresholded)")
    else:
        raise ValueError(f"Conjunction type {conjunction_type} not supported")
    
    return Nifti1Image(froi_mask, affine=affine, header=header)


#### ssROI
def get_ssroi(subject: str, task: str, run_label: str, localizer: str, 
              threshold_type: str, threshold_value: float, parcels_name: str, parcels: Nifti1Image, create_if_not_exist: Optional[bool] = False) -> Nifti1Image:
    ssroi_path = get_ssroi_path(subject, task, run_label, localizer, threshold_type, threshold_value, parcels_name)
    if not os.path.exists(ssroi_path): 
        if create_if_not_exist:
            return create_ssroi(subject, task, run_label, localizer, threshold_type, threshold_value, parcels_name, parcels)
        return None
    return load_img(ssroi_path)


def create_ssroi(subject: str, task: str, run_label: str, localizer: str, 
                 threshold_type: str, threshold_value: float, parcels_name: str, parcels: Nifti1Image) -> Nifti1Image:
    os.makedirs(get_subject_ssroi_folder(subject), exist_ok=True)
    threshold_type_orig = threshold_type
    threshold_value_orig = threshold_value

    # First, check if it is a contrast-based localizer
    contrast_info = get_contrast_info(subject, task, localizer)
    if contrast_info is not None:
        p_map = get_contrast(subject, task, run_label, localizer, 'p', create_if_not_exist=True)
        mask = np.zeros_like(p_map.get_fdata())
        for parcel_label_i in np.unique(parcels.get_fdata()):
            if parcel_label_i == 0: continue
            parcel_mask_i = parcels.get_fdata() == parcel_label_i
            p_map_i = p_map.get_fdata().copy()
            p_map_i[parcel_mask_i == 0] = np.nan

            if threshold_type_orig == 'percent':
                threshold_type = 'n'
                threshold_value = int(np.sum(parcel_mask_i) * threshold_value_orig)
            
            parcel_p_map_i = _threshold_p_map(p_map_i, threshold_type, threshold_value)
            mask[parcel_mask_i] = parcel_p_map_i[parcel_mask_i] * parcel_label_i
        ssroi_img = Nifti1Image(mask, affine=p_map.affine, header=p_map.header)
        ssroi_img.to_filename(get_ssroi_path(subject, task, run_label, localizer, threshold_type_orig, threshold_value_orig, parcels_name))
        return ssroi_img

    os.makedirs(get_subject_ssroi_folder(subject), exist_ok=True)
    localizer_info = get_localizer_info(subject, task, localizer)
    assert localizer_info is not None, f"Localizer {localizer} not found for subject {subject}, task {task}."

    parcels_data = parcels.get_fdata()
    ssroi = np.zeros_like(parcels_data)
    affine = None
    header = None
    for parcel_label_i in np.unique(parcels_data):
        if parcel_label_i == 0: continue
        parcel_mask_i = parcels_data == parcel_label_i
        if threshold_type_orig == 'percent':
            threshold_type = 'n'
            threshold_value = int(np.sum(parcel_mask_i) * threshold_value_orig)
        ssroi_i = _create_localizer(subject=subject, task=task, run_label=run_label, 
                                    contrasts=localizer_info['contrasts'], conjunction_type=localizer_info['conjunction_type'],
                                    threshold_type=threshold_type, threshold_value=threshold_value, mask=parcel_mask_i)
        if affine is None or header is None:
            affine = ssroi_i.affine
            header = ssroi_i.header
        ssroi[parcel_mask_i] = ssroi_i.get_fdata()[parcel_mask_i] * parcel_label_i
    ssroi_img = Nifti1Image(ssroi, affine=parcels.affine, header=parcels.header)
    ssroi_img.to_filename(get_ssroi_path(subject, task, run_label, localizer, threshold_type_orig, threshold_value_orig, parcels_name))
    return ssroi_img


#### Others

def get_parcels(parcels_name: str) -> Nifti1Image:
    parcels_path = get_parcels_path(parcels_name)
    if not os.path.exists(parcels_path): return None
    parcels_img = load_img(parcels_path)
    # to fix some floating point problems - labels assume to be integers
    parcels_img = math_img('np.round(img)', img=parcels_img)
    return parcels_img


def get_design_matrix(subject: str, task: str, run: int):
    design_matrix_path = get_design_matrix_path(subject, task, run)
    assert os.path.exists(design_matrix_path), f"Design matrix not found for sub-{subject}, task {task}, run {run}."
    return pd.read_csv(design_matrix_path)


def get_runs(subject: str, task: str) -> List[int]:
    template_contrast_path = get_contrast_path(subject, task, '*', '*', 'p')
    potential_paths = glob.glob(template_contrast_path)
    pattern = re.compile(
        rf".*{template_contrast_path.replace('run-*', 'run-[0-9]+').replace('contrast-*', 'contrast-.*')}"
    )    
    run_paths = [path for path in potential_paths if pattern.match(path)]
    unique_runs = list(set([int(re.search(r'run-(\d+)', run_path).group(1)) for run_path in run_paths]))
    return unique_runs


def _threshold_p_map(data: np.ndarray, threshold_type: str, threshold_value: float) -> np.ndarray:
    """
    Extract voxels from a p-map image based on a threshold
    """
    froi_mask = np.zeros_like(data)

    if threshold_type == 'n':
        topN_idx = np.argsort(data.flatten())[:threshold_value]
        froi_mask.flat[topN_idx] = 1
    elif 'pval' in threshold_type:
        pcorrection_method = threshold_type.split('_')[1]
        if pcorrection_method == 'fdr':
            pvals = data.flatten()
            _, pvals_fdr = fdrcorrection(pvals)
            froi_mask.flat[pvals_fdr < threshold_value] = 1
        elif pcorrection_method == 'bonferroni':
            froi_mask.flat[data.flatten() < threshold_value / data.size] = 1
        elif pcorrection_method == 'none':
            froi_mask.flat[data.flatten() < threshold_value] = 1
        else:
            raise ValueError(f'Unknown p-value correction method: {pcorrection_method}')
    else:
        raise ValueError(f'Unknown threshold type: {threshold_type}')
    
    return froi_mask
