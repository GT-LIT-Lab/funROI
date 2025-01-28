import warnings
import os
import boto3
import pandas as pd
from ..utils import ensure_paths
import json

warnings.simplefilter(action="ignore", category=FutureWarning)


def get_events(ev_folder_path, events):
    events_df = pd.DataFrame(
        columns=["onset", "duration", "trial_type", "amplitude"]
    )
    for condition in events:
        ev_file = f"{ev_folder_path}/{condition}.txt"
        ev_data = pd.read_csv(
            ev_file,
            sep="\t",
            header=None,
            names=["onset", "duration", "amplitude"],
        )
        ev_data["trial_type"] = condition
        events_df = pd.concat([events_df, ev_data], ignore_index=True)
    events_df = events_df.sort_values("onset").reset_index(drop=True)
    events_df = events_df[["trial_type", "onset", "duration"]]
    return events_df


def list_s3_objects(s3_client, bucket_name, prefix):
    bucket = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    return [obj["Key"] for obj in bucket.get("Contents", [])]


def download_file(s3_client, bucket_name, s3_key, local_path):
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(bucket_name, s3_key, local_path)
    except Exception as e:
        print(f"Missing or failed: {s3_key}")


def download_selected(parent_dir, s3_client, subject):
    patterns = [
        "MNINonLinear/Results/tfMRI_LANGUAGE_LR",
        "MNINonLinear/Results/tfMRI_LANGUAGE_RL",
    ]

    for pattern in patterns:
        s3_path = f"HCP_1200/{subject}/{pattern}"
        s3_objects = list_s3_objects(s3_client, "hcp-openaccess", s3_path)
        for s3_object in s3_objects:
            download_file(
                s3_client,
                "hcp-openaccess",
                s3_object,
                parent_dir / s3_object,
            )


@ensure_paths("data_dir", "bids_dir")
def convert_to_bids(data_dir, bids_dir, subject):
    runs = (data_dir / subject / "MNINonLinear" / "Results").iterdir()
    for run_folder in runs:
        if "tfMRI" not in run_folder.name:
            continue
        run_task = run_folder.name.split("_")[1]
        run_filename = run_folder.name.split("_", 1)[1]
        run_suffix = run_folder.name.split("_")[-1]

        bids_folder = bids_dir / f"sub-{subject}" / "func"
        bids_folder.mkdir(parents=True, exist_ok=True)

        tab_file = list(run_folder.glob("LANGUAGE*_TAB.txt"))
        label = tab_file[0].stem.split("_TAB")[0]
        run_i = label.split("_")[-1].replace("run", "")

        bids_prefix_no_space = (
            f"sub-{subject}_task-{run_task}_run-{run_i}_acq-{run_suffix}"
        )
        bids_prefix = bids_prefix_no_space + "_space-MNINonLinear"

        # Data files
        (bids_folder / f"{bids_prefix}_desc-brain_mask.nii.gz").symlink_to(
            run_folder / "brainmask_fs.2.nii.gz"
        )
        (bids_folder / f"{bids_prefix}_desc-preproc_bold.nii.gz").symlink_to(
            run_folder / f"tfMRI_{run_filename}.nii.gz"
        )

        # BOLD configuration
        with open(bids_folder / f"{bids_prefix}_bold.json", "w") as f:
            json.dump(
                {
                    "RepetitionTime": 0.72,
                    "EchoTime": 0.0331,
                    "EffectiveEchoSpacing": 0.00058,
                    "MagneticFieldStrength": 3.0,
                    "Manufacturer": "Siemens",
                    "ManufacturerModelName": "Skyra",
                    "PhaseEncodingDirection": (
                        "i-" if run_suffix == "LR" else "i"
                    ),
                    "TaskName": "LANGUAGE",
                },
                f,
                indent=4,
            )

        # Confounds file
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

        # Events file
        events_df = get_events(run_folder / "EVs", events=["math", "story"])
        events_df.to_csv(
            bids_folder / f"{bids_prefix_no_space}_events.tsv",
            sep="\t",
            index=False,
        )


@ensure_paths("data_dir")
def fetch_language_data(data_dir, subjects):
    data_dir = data_dir.absolute()
    bids_dir = data_dir / "bids"
    data_dir.mkdir(parents=True, exist_ok=True)
    bids_dir.mkdir(parents=True, exist_ok=True)
    s3_client = boto3.client("s3")
    for subject in subjects:
        download_selected(data_dir, s3_client, subject)
        convert_to_bids(data_dir / "HCP_1200", bids_dir, subject)
