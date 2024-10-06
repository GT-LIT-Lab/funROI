from . import get_bids_data_folder, get_bids_preprocessed_folder, get_bids_preprocessed_folder_relative, utils, get_bids_deriv_folder

import os
import pickle
from typing import List, Optional
from nilearn.image import new_img_like, load_img
from nilearn.glm import compute_contrast, SimpleRegressionResults
from nilearn.glm.first_level import first_level_from_bids, make_first_level_design_matrix, run_glm
from nibabel.nifti1 import Nifti1Image
import numpy as np
import pandas as pd
import re
import h5py
from scipy.stats import t as t_dist


#### Nilearn first level
def run_first_level_nilearn(subjects: List[str], tasks: List[str], space: str, confound_labels: Optional[List[str]] = [], **args):
    os.makedirs(utils.get_bids_deriv_folder(), exist_ok=True)
    dofs = {}
    for task in tasks:
        # Load data and model from preprocessed data
        (models, models_run_imgs, models_events, models_confounds) = first_level_from_bids(
            get_bids_data_folder(), task, sub_labels=subjects, derivatives_folder=get_bids_preprocessed_folder_relative(), slice_time_ref=None)
        
        for subject_i in range(len(models)):
            model, imgs, events, confounds = (models[subject_i], models_run_imgs[subject_i], models_events[subject_i], models_confounds[subject_i])
            subject_label = model.subject_label
            os.makedirs(utils.get_subject_model_folder(subject_label), exist_ok=True)
            os.makedirs(utils.get_subject_contrast_folder(subject_label), exist_ok=True)

            for run_i in range(len(events)):
                events_i = events[run_i]
                imgs_i = load_img(imgs[run_i])
                frame_times = np.arange(imgs_i.shape[-1]) * model.t_r
                design_matrix = make_first_level_design_matrix(frame_times=frame_times, events=events_i, **args)

                # Add confounds to design matrix
                confounds_i = confounds[run_i]
                for confound_label in confound_labels:
                    confound = confounds_i[confound_label]
                    # Replace NaNs with mean value
                    confound = np.where(np.isnan(confound), np.nanmean(confound), confound)
                    design_matrix[confound_label] = confound

                ses_label = re.search(r'ses-(\w+)_', imgs[run_i]).group(1)
                task_label = re.search(r'task-(\w+)_', imgs[run_i]).group(1)
                run_label = re.search(r'run-(\d+)_', imgs[run_i]).group(1)
                mask_img = load_img(os.path.join(get_bids_preprocessed_folder(), f'sub-{subject_label}', f'ses-{ses_label}', 'func', f'sub-{subject_label}_ses-{ses_label}_task-{task_label}_run-{run_label}_space-{space}_desc-brain_mask.nii.gz'))
                mask_img_data = mask_img.get_fdata()
                imgs_i_data = imgs_i.get_fdata()
                # for all images in imgs_i_data, set to NaN if mask_img_data is 0
                imgs_i_data[mask_img_data == 0] = np.nan

                labels, estimates = run_glm(imgs_i_data.reshape(-1, imgs_i_data.shape[-1]).T, design_matrix.values)
                for key, value in estimates.items():
                    if not isinstance(value, SimpleRegressionResults):
                        estimates[key] = SimpleRegressionResults(value)

                # Dump labels and estimates
                labels_path, estimates_path = utils.get_first_level_model_paths_nilearn(subject_label, task, run_i+1)
                with open(labels_path, 'wb') as f:
                    pickle.dump(labels, f)
                with open(estimates_path, 'wb') as f:
                    pickle.dump(estimates, f)

                # Save design matrix
                design_matrix.to_csv(utils.get_design_matrix_path(subject_label, task, run_i+1), index=False)

                vars = design_matrix.columns
                for var_i, var in enumerate(vars):
                    var_array = np.zeros((len(vars)))
                    var_array[var_i] = 1

                    contrast = compute_contrast(labels, estimates, var_array, stat_type='t')
                    effect_map = contrast.effect_size().reshape(imgs_i_data.shape[:-1])
                    variance_map = contrast.effect_variance().reshape(imgs_i_data.shape[:-1])
                    t_map = contrast.stat().reshape(imgs_i_data.shape[:-1])
                    z_map = contrast.z_score().reshape(imgs_i_data.shape[:-1])
                    p_map = contrast.p_value().reshape(imgs_i_data.shape[:-1])

                    # Export contrast maps
                    new_img_like(imgs_i, effect_map).to_filename(utils.get_contrast_path(subject_label, task, run_i+1, var, 'effect'))
                    new_img_like(imgs_i, variance_map).to_filename(utils.get_contrast_path(subject_label, task, run_i+1, var, 'variance'))
                    new_img_like(imgs_i, t_map).to_filename(utils.get_contrast_path(subject_label, task, run_i+1, var, 't'))
                    new_img_like(imgs_i, z_map).to_filename(utils.get_contrast_path(subject_label, task, run_i+1, var, 'z'))
                    new_img_like(imgs_i, p_map).to_filename(utils.get_contrast_path(subject_label, task, run_i+1, var, 'p'))
                    utils.register_contrast(subject_label, task, var, var_array)

                # Compute degrees of freedom
                dof = imgs_i_data.shape[-1] - len(vars)
                if subject_label not in dofs:
                    dofs[subject_label] = pd.DataFrame(columns=['task', 'run', 'dof'])
                dofs[subject_label] = pd.concat([dofs[subject_label], pd.DataFrame({'task': [task], 'run': [run_i+1], 'dof': [dof]})])

    for subject, df in dofs.items():
        df.to_csv(utils.get_dof_path(subject, task), index=False)


def get_first_level_nilearn(subject: str, task: str, run: int):
    labels_path, estimates_path = utils.get_first_level_model_paths_nilearn(subject, task, run)
    assert os.path.exists(labels_path) and os.path.exists(estimates_path), f"Model files not found for subject {subject}, task {task}, run {run}."
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)
    with open(estimates_path, 'rb') as f:
        estimates = pickle.load(f)
    return labels, estimates


def _create_contrast_single_run_nilearn(subject: str, task: str, run_label: int, contrast_vector: List[float], type: str) -> Nifti1Image:
    # This function only applies when the default nilearn first level model is used
    labels, estimates = get_first_level_nilearn(subject, task, run_label)

    run0 = utils.get_runs(subject, task)[0] 
    contrast_info_path = utils.get_contrast_info_path(subject, task)
    contrast_info = pd.read_csv(contrast_info_path)
    contrast0 = contrast_info['contrast'][0]
    ref_img = load_img(utils.get_contrast_path(subject, task, run0, contrast0, type))

    contrast = compute_contrast(labels, estimates, contrast_vector, stat_type='t')
    if type == 't':
        return new_img_like(ref_img, contrast.stat().reshape(ref_img.get_fdata().shape))
    elif type == 'p':
        return new_img_like(ref_img, contrast.p_value().reshape(ref_img.get_fdata().shape))
    elif type == 'z':
        return new_img_like(ref_img, contrast.z_score().reshape(ref_img.get_fdata().shape))
    elif type == 'effect':
        return new_img_like(ref_img, contrast.effect_size().reshape(ref_img.get_fdata().shape))
    elif type == 'variance':
        return new_img_like(ref_img, contrast.effect_variance().reshape(ref_img.get_fdata().shape))
    else:
        raise ValueError(f"Type {type} not recognized.")


#### SPM migration
def migrate_first_level_from_spm(spm_dir: str, subject: str, task: str):
    # This function will migrate the contrast files from SPM to the BIDS data folder.
    # It is assumed that all contrast files needed for the analysis are pre-calculated and stored in the SPM folder.
    os.makedirs(utils.get_subject_model_folder(subject), exist_ok=True)
    os.makedirs(utils.get_subject_contrast_folder(subject), exist_ok=True)

    spm_mat_path = os.path.join(spm_dir, 'SPM.mat')
    assert os.path.exists(spm_mat_path), f"'SPM.mat' not found in {spm_dir}"
    with h5py.File(spm_mat_path, 'r') as f:
        spm = f['SPM']

        dof_residual = int(spm['xX']['erdf'][0,0])
        # X = spm['xX']['X'][()]
        # m = X.shape[1]  # Number of model parameters
        # nf = m + dof_residual

        Bcov = spm['xX']['Bcov'][()]    # covariance matrix (Bcov)
        con_fnames = f['/SPM/xCon/name']
        con_names = []
        for fname in con_fnames:
            con_names.append(''.join([chr(c[0]) for c in f[fname[0]]]))

        con_vectors = []
        for i in range(len(con_names)):
            con = f[f['/SPM/xCon/c'][i].item()][()]
            con_vectors.append(con)
        
    resms_path = os.path.join(spm_dir, 'ResMS.nii')
    resms_img = load_img(resms_path)
    resms_data = resms_img.get_fdata()

    run_ids = []
    # Export contrasts
    for i, con_name in enumerate(con_names):
        utils.register_contrast(subject, task, con_name, con_vectors[i][0].tolist())
        effect_path = os.path.join(spm_dir, f"con_{i+1:04}.nii")
        effect_img = load_img(effect_path)
        t_path = os.path.join(spm_dir, f"spmT_{i+1:04}.nii")
        t_img = load_img(t_path)

        # Run label
        if con_name.startswith('ODD_'):
            run_label = 'odd'
            contrast_label = con_name.replace('ODD_', '')
        elif con_name.startswith('EVEN_'):
            run_label = 'even'
            contrast_label = con_name.replace('EVEN_', '')
        elif con_name.startswith('ORTH_TO_SESSION'):
            run_i = re.search(r'ORTH_TO_SESSION(\d+)', con_name).group(1)
            run_label = f"orth{int(run_i)}"
            contrast_label = con_name.replace(f'ORTH_TO_SESSION{run_i}_', '')
        elif con_name.startswith('SESSION'):
            run_i = re.search(r'SESSION(\d+)', con_name).group(1)
            run_label = f"{int(run_i)}"
            contrast_label = con_name.replace(f'SESSION{run_i}_', '')
            run_ids.append(run_i)
        else:
            run_label = 'all'
            contrast_label = con_name
        
        effect_img.to_filename(utils.get_contrast_path(subject, task, run_label, contrast_label, 'effect'))

        t_img.to_filename(utils.get_contrast_path(subject, task, run_label, contrast_label, 't'))

        p_img = new_img_like(t_img, 1 - t_dist.cdf(t_img.get_fdata(), dof_residual))
        p_img.to_filename(utils.get_contrast_path(subject, task, run_label, contrast_label, 'p'))

        var_img = new_img_like(t_img, con_vectors[i].reshape(1, -1) @ Bcov @ con_vectors[i].reshape(-1, 1) * resms_data)
        var_img.to_filename(utils.get_contrast_path(subject, task, run_label, contrast_label, 'variance'))

    dof_df = pd.DataFrame({})
    for run_i in run_ids:
        dof_df = pd.concat([dof_df, pd.DataFrame({
            'task': [task],
            'run': [run_i],
            'dof': [dof_residual]
        })], ignore_index=True)
    
    dof_df_path = utils.get_dof_path(subject, task)
    dof_df.to_csv(dof_df_path, index=False)