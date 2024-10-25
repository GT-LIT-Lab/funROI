from nilearn.image import load_img, new_img_like
from scipy.stats import t as t_dist
import h5py
import numpy as np
import os
import pandas as pd
import re
from ..utils import (
    get_subject_model_folder,
    get_subject_contrast_folder,
    get_contrast_path,
    get_dof_path,
)

from ..contrast import register_contrast


def migrate_first_level_from_spm(spm_dir: str, subject: str, task: str):
    """
    Migrate first-level results from SPM to BIDS.

    Parameters
    ----------
    spm_dir : str
        Path to the SPM directory. The directory should contain 'SPM.mat',
    subject : str
        Subject label.
    task : str
        Task label.
    """
    os.makedirs(get_subject_model_folder(subject), exist_ok=True)
    os.makedirs(get_subject_contrast_folder(subject), exist_ok=True)

    spm_mat_path = os.path.join(spm_dir, "SPM.mat")
    assert os.path.exists(spm_mat_path), f"'SPM.mat' not found in {spm_dir}"
    with h5py.File(spm_mat_path, "r") as f:
        spm = f["SPM"]

        dof_residual = int(spm["xX"]["erdf"][0, 0])
        # X = spm['xX']['X'][()]
        # m = X.shape[1]  # Number of model parameters
        # nf = m + dof_residual

        Bcov = spm["xX"]["Bcov"][()]  # covariance matrix (Bcov)
        con_fnames = f["/SPM/xCon/name"]
        con_names = []
        for fname in con_fnames:
            con_names.append("".join([chr(c[0]) for c in f[fname[0]]]))

        con_vectors = []
        for i in range(len(con_names)):
            con = f[f["/SPM/xCon/c"][i].item()][()]
            con_vectors.append(con)

    resms_path = os.path.join(spm_dir, "ResMS.nii")
    resms_img = load_img(resms_path)
    resms_data = resms_img.get_fdata()

    run_ids = set()
    # Export contrasts
    for i, con_name in enumerate(con_names):
        register_contrast(subject, task, con_name, con_vectors[i][0].tolist())
        effect_path = os.path.join(spm_dir, f"con_{i+1:04}.nii")
        effect_img = load_img(effect_path)
        t_path = os.path.join(spm_dir, f"spmT_{i+1:04}.nii")
        t_img = load_img(t_path)

        # spmT specific - convert t=0 to NaN
        t_img_data = t_img.get_fdata()
        t_img_data[t_img_data == 0] = np.nan
        t_img = new_img_like(t_img, t_img_data)

        # Run label
        if con_name.startswith("ODD_"):
            run_label = "odd"
            contrast_label = con_name.replace("ODD_", "")
        elif con_name.startswith("EVEN_"):
            run_label = "even"
            contrast_label = con_name.replace("EVEN_", "")
        elif con_name.startswith("ORTH_TO_SESSION"):
            run_i = re.search(r"ORTH_TO_SESSION(\d+)", con_name).group(1)
            run_label = f"orth{int(run_i)}"
            contrast_label = con_name.replace(f"ORTH_TO_SESSION{run_i}_", "")
        elif con_name.startswith("SESSION"):
            run_i = re.search(r"SESSION(\d+)", con_name).group(1)
            run_label = f"{int(run_i)}"
            contrast_label = con_name.replace(f"SESSION{run_i}_", "")
            run_ids.add(int(run_i))
        else:
            run_label = "all"
            contrast_label = con_name

        effect_img.to_filename(
            get_contrast_path(
                subject, task, run_label, contrast_label, "effect"
            )
        )

        t_img.to_filename(
            get_contrast_path(subject, task, run_label, contrast_label, "t")
        )

        p_img = new_img_like(
            t_img, 1 - t_dist.cdf(t_img.get_fdata(), dof_residual)
        )
        p_img.to_filename(
            get_contrast_path(subject, task, run_label, contrast_label, "p")
        )

        var_img = new_img_like(
            t_img,
            con_vectors[i].reshape(1, -1)
            @ Bcov
            @ con_vectors[i].reshape(-1, 1)
            * resms_data,
        )
        var_img.to_filename(
            get_contrast_path(
                subject, task, run_label, contrast_label, "variance"
            )
        )

    dof_df = pd.DataFrame({})
    for run_i in run_ids:
        dof_df = pd.concat(
            [
                dof_df,
                pd.DataFrame(
                    {"task": [task], "run": [run_i], "dof": [dof_residual]}
                ),
            ],
            ignore_index=True,
        )

    dof_df_path = get_dof_path(subject, task)
    dof_df.to_csv(dof_df_path, index=False)
