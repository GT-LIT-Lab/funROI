import json
import os
import pathlib
import shutil
from typing import List, Union

import boto3
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from .._surface import write_gifti
from ..utils import ensure_paths


SURFACE_MESH_FILENAMES = {
    "L": "{subject}.L.midthickness.32k_fs_LR.surf.gii",
    "R": "{subject}.R.midthickness.32k_fs_LR.surf.gii",
}
SURFACE_DTSeries_FILENAMES = [
    "tfMRI_{run_filename}_Atlas_MSMAll.dtseries.nii",
    "tfMRI_{run_filename}_Atlas.dtseries.nii",
]
SURFACE_STRUCTURES = {
    "L": "CIFTI_STRUCTURE_CORTEX_LEFT",
    "R": "CIFTI_STRUCTURE_CORTEX_RIGHT",
}
_write_gifti = write_gifti


def _get_events(ev_folder_path, events):
    event_frames = []
    for condition in events:
        ev_file = f"{ev_folder_path}/{condition}.txt"
        ev_data = pd.read_csv(
            ev_file,
            sep="\t",
            header=None,
            names=["onset", "duration", "amplitude"],
        )
        ev_data["trial_type"] = condition
        event_frames.append(ev_data)
    events_df = pd.concat(event_frames, ignore_index=True)
    events_df = events_df.sort_values("onset").reset_index(drop=True)
    events_df = events_df[["trial_type", "onset", "duration"]]
    events_df["trial_type"] = events_df["trial_type"].str.replace(
        r"[^a-zA-Z0-9]", ""
    )
    return events_df


def _list_s3_objects(s3_client, bucket_name, prefix):
    bucket = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    return [obj["Key"] for obj in bucket.get("Contents", [])]


def _download_file(s3_client, bucket_name, s3_key, local_path):
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(bucket_name, s3_key, local_path)
    except Exception as e:
        print(e)
        print(f"Missing or failed: {s3_key}")


def _extract_surface_data_from_dtseries(dtseries_path):
    dtseries = nib.load(dtseries_path)
    data = np.asarray(dtseries.dataobj)
    brain_models = dtseries.header.get_axis(1)

    surface_data = {}
    for structure_name, data_slice, model in brain_models.iter_structures():
        for hemi, target_structure in SURFACE_STRUCTURES.items():
            if structure_name != target_structure:
                continue
            n_vertices = model.nvertices[structure_name]
            hemi_data = np.zeros((n_vertices, data.shape[0]), dtype=np.float32)
            hemi_data[model.vertex, :] = data[:, data_slice].T
            surface_data[hemi] = hemi_data
    return surface_data


def _find_surface_dtseries(run_folder, run_filename):
    for filename in SURFACE_DTSeries_FILENAMES:
        path = run_folder / filename.format(run_filename=run_filename)
        if path.exists():
            return path
    return None


def _bold_metadata(task, run_suffix):
    return {
        "RepetitionTime": 0.72,
        "EchoTime": 0.0331,
        "EffectiveEchoSpacing": 0.00058,
        "MagneticFieldStrength": 3.0,
        "Manufacturer": "Siemens",
        "ManufacturerModelName": "Skyra",
        "PhaseEncodingDirection": "i-" if run_suffix == "LR" else "i",
        "TaskName": task,
    }


def _download_selected(
    parent_dir, s3_client, subject, task, download_surface_data=True
):
    patterns = [
        f"MNINonLinear/Results/tfMRI_{task}_LR",
        f"MNINonLinear/Results/tfMRI_{task}_RL",
    ]

    for pattern in patterns:
        s3_path = f"HCP_1200/{subject}/{pattern}"
        s3_objects = _list_s3_objects(s3_client, "hcp-openaccess", s3_path)
        for s3_object in s3_objects:
            if (
                not download_surface_data
                and (
                    s3_object.endswith(".dtseries.nii")
                    or s3_object.endswith(".func.gii")
                )
            ):
                continue
            _download_file(
                s3_client,
                "hcp-openaccess",
                s3_object,
                parent_dir / s3_object,
            )

    if download_surface_data:
        for filename in SURFACE_MESH_FILENAMES.values():
            s3_key = (
                f"HCP_1200/{subject}/MNINonLinear/fsaverage_LR32k/"
                f"{filename.format(subject=subject)}"
            )
            _download_file(
                s3_client,
                "hcp-openaccess",
                s3_key,
                parent_dir / s3_key,
            )


@ensure_paths("data_dir", "bids_dir")
def _convert_to_bids(
    data_dir, bids_dir, subject, task, download_surface_data=True
):
    if download_surface_data:
        anat_dir = bids_dir / f"sub-{subject}" / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)
        for hemi, filename in SURFACE_MESH_FILENAMES.items():
            source = (
                data_dir
                / subject
                / "MNINonLinear"
                / "fsaverage_LR32k"
                / filename.format(subject=subject)
            )
            if not source.exists():
                continue
            shutil.copy(
                source,
                anat_dir
                / f"sub-{subject}_hemi-{hemi}_space-fsLR32k_midthickness.surf.gii",
            )

    runs = (data_dir / subject / "MNINonLinear" / "Results").iterdir()
    run_i = 1
    for run_folder in runs:
        if f"tfMRI_{task}" not in run_folder.name:
            continue
        run_task = run_folder.name.split("_")[1]
        run_filename = run_folder.name.split("_", 1)[1]
        run_suffix = run_folder.name.split("_")[-1]

        bids_folder = bids_dir / f"sub-{subject}" / "func"
        bids_folder.mkdir(parents=True, exist_ok=True)

        bids_prefix_no_space = (
            f"sub-{subject}_task-{run_task}_run-{run_i}_acq-{run_suffix}"
        )
        bids_prefix = bids_prefix_no_space + "_space-MNINonLinear"

        shutil.copy(
            run_folder / "brainmask_fs.2.nii.gz",
            bids_folder / f"{bids_prefix}_desc-brain_mask.nii.gz",
        )
        shutil.copy(
            run_folder / f"tfMRI_{run_filename}.nii.gz",
            bids_folder / f"{bids_prefix}_desc-preproc_bold.nii.gz",
        )

        with open(bids_folder / f"{bids_prefix}_bold.json", "w") as f:
            json.dump(_bold_metadata(task, run_suffix), f, indent=4)

        if download_surface_data:
            surface_metadata = _bold_metadata(task, run_suffix)
            dtseries_path = _find_surface_dtseries(run_folder, run_filename)
            if dtseries_path is not None:
                surface_data = _extract_surface_data_from_dtseries(
                    dtseries_path
                )
                for hemi in ["L", "R"]:
                    if hemi not in surface_data:
                        continue
                    dest_prefix = (
                        f"sub-{subject}_task-{run_task}_run-{run_i}_acq-{run_suffix}"
                        f"_hemi-{hemi}_space-fsLR32k_desc-preproc"
                    )
                    write_gifti(
                        bids_folder / f"{dest_prefix}_bold.func.gii",
                        surface_data[hemi],
                    )
                    with open(
                        bids_folder / f"{dest_prefix}_bold.json", "w"
                    ) as f:
                        json.dump(surface_metadata, f, indent=4)

        with open(run_folder / "Movement_Regressors.txt", "r") as f:
            data = [[float(x) for x in line.split()] for line in f]
        pd.DataFrame(
            data,
            columns=[
                "trans_x",
                "trans_y",
                "trans_z",
                "rot_x",
                "rot_y",
                "rot_z",
                "trans_dx",
                "trans_dy",
                "trans_dz",
                "rot_dx",
                "rot_dy",
                "rot_dz",
            ],
        ).to_csv(
            bids_folder
            / f"{bids_prefix_no_space}_desc-confounds_timeseries.tsv",
            sep="\t",
            index=False,
        )

        if task == "LANGUAGE":
            events = ["math", "story"]
        elif task == "MOTOR":
            events = ["cue", "t", "lf", "rf", "lh", "rh"]
        elif task == "WM":
            events = [
                "0bk_body",
                "0bk_faces",
                "0bk_places",
                "0bk_tools",
                "2bk_body",
                "2bk_faces",
                "2bk_places",
                "2bk_tools",
            ]
        elif task == "SOCIAL":
            events = ["mental", "rnd"]

        events_df = _get_events(run_folder / "EVs", events=events)
        events_df.to_csv(
            bids_folder / f"{bids_prefix_no_space}_events.tsv",
            sep="\t",
            index=False,
        )

        run_i += 1


@ensure_paths("data_dir")
def fetch_data(
    data_dir: Union[str, pathlib.Path],
    task: str,
    subjects: List[str],
    download_surface_data: bool = True,
) -> None:
    """
    Fetches the HCP dataset for a given task and subjects, and converts it to
    BIDS format.

    :param data_dir: Path to the directory where the data will be stored.
    :type data_dir: Union[str, pathlib.Path]
    :param task: The task to fetch data for. Options are "LANGUAGE", "MOTOR",
                 "WM", and "SOCIAL".
    :type task: str
    :param subjects: List of subject IDs to fetch data for (e.g., ["100307", "100408"]).
    :type subjects: List[str]
    :param download_surface_data: Whether to download and BIDSify native
        shared-atlas `fsLR32k` surface task data and midthickness meshes.
    :type download_surface_data: bool
    """
    task = task.upper()
    if task not in ["LANGUAGE", "MOTOR", "WM", "SOCIAL"]:
        raise ValueError(
            "Unsupported task. Choose from LANGUAGE, MOTOR, WM, SOCIAL"
        )
    data_dir = data_dir.absolute()
    bids_dir = data_dir / "bids"
    data_dir.mkdir(parents=True, exist_ok=True)
    bids_dir.mkdir(parents=True, exist_ok=True)
    s3_client = boto3.client("s3")
    for subject in tqdm(subjects):
        try:
            _download_selected(
                data_dir,
                s3_client,
                subject,
                task,
                download_surface_data=download_surface_data,
            )
            _convert_to_bids(
                data_dir / "HCP_1200",
                bids_dir,
                subject,
                task,
                download_surface_data=download_surface_data,
            )
        except Exception as e:
            print(f"Error processing {subject}: {e}")
