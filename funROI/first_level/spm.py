from pathlib import Path
from typing import Union, List
from ..contrast import (
    _get_contrast_folder,
    _get_design_matrix_path,
    _get_contrast_path,
    _get_model_folder,
    _get_contrast_info_path,
)
from ..utils import ensure_paths
from nilearn.image import load_img, new_img_like
from scipy.stats import t as t_dist
import h5py
import numpy as np
import pandas as pd
import re
from .utils import _register_contrast


@ensure_paths("spm_dir")
def migrate_first_level_from_spm(
    spm_dir: Union[str, Path], subject: str, task: str
):
    """
    Migrate first-level contrasts from SPM to BIDS, to be used with later
    stages of the pipeline.

    .. warning:: Since contrast computation is quite model-specific, it is
        assumed that all needed contrasts are already computed in SPM. E.g.,
        contrasts for each run, all-but-one, odd, and even runs, etc. These
        contrasts are assumed to be named in the following way in SPM:

        - Contrast by run: SESSION<session_number>_<contrast_name>

        - All-but-one contrast: ORTH_TO_SESSION<session_number>_<contrast_name>

        - Odd-run contrast: ODD_<contrast_name>

        - Even-run contrast: EVEN_<contrast_name>

    :param spm_dir: Path to the SPM directory. The directory should contain
        'SPM.mat'.
    :type spm_dir: Union[str, Path]
    :param subject: Subject label.
    :type subject: str
    :param task: Task label.
    :type task: str

    :raises FileNotFoundError: If 'SPM.mat' is not found in the SPM directory.
    """
    _get_model_folder(subject, task).mkdir(parents=True, exist_ok=True)
    _get_contrast_folder(subject, task).mkdir(parents=True, exist_ok=True)

    spm_mat_path = spm_dir / "SPM.mat"
    if not spm_mat_path.exists():
        raise FileNotFoundError(f"'SPM.mat' not found in {spm_dir}")

    with h5py.File(spm_mat_path, "r") as f:
        spm = f["SPM"]

        dof_residual = int(spm["xX"]["erdf"][0, 0])

        design_matrix = f["/SPM/xX/X"]
        design_matrix_names = [
            "".join([chr(c[0]) for c in f[fname[0]]])
            for fname in f["/SPM/xX/name"]
        ]
        design_matrix_df = pd.DataFrame(
            design_matrix[()].T, columns=design_matrix_names
        )
        design_matrix_path = _get_design_matrix_path(subject, task)
        design_matrix_df.to_csv(design_matrix_path, index=False)

        con_fnames = f["/SPM/xCon/name"]
        con_names = []
        for fname in con_fnames:
            con_names.append("".join([chr(c[0]) for c in f[fname[0]]]))

        con_vectors = []
        for i in range(len(con_names)):
            con = f[f["/SPM/xCon/c"][i].item()][()]
            con_vectors.append(con)

    run_ids = set()
    # Export contrasts
    for i, con_name in enumerate(con_names):
        _register_contrast(subject, task, con_name, con_vectors[i][0].tolist())
        effect_path = spm_dir / f"con_{i+1:04}.nii"
        effect_img = load_img(effect_path)
        t_path = spm_dir / f"spmT_{i+1:04}.nii"
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
            _get_contrast_path(
                subject, task, run_label, contrast_label, "effect"
            )
        )
        t_img.to_filename(
            _get_contrast_path(subject, task, run_label, contrast_label, "t")
        )

        p_img = new_img_like(
            t_img,
            (1 - t_dist.cdf(t_img.get_fdata(), dof_residual)).astype(
                np.float32
            ),
        )
        p_img.to_filename(
            _get_contrast_path(subject, task, run_label, contrast_label, "p")
        )
