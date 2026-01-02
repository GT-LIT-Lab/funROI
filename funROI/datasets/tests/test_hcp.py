import json
from pathlib import Path
import pandas as pd
import pytest
from unittest.mock import Mock
import boto3
from botocore.stub import Stubber

import funROI.datasets.hcp as hcp


def _write_dummy_file(p: Path, content: bytes = b"dummy"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)


def test_convert_to_bids_language_lr(tmp_path):
    subject = "100307"
    task = "LANGUAGE"
    data_dir = tmp_path / "HCP_1200"
    bids_dir = tmp_path / "bids"

    run_folder = (
        data_dir / subject / "MNINonLinear" / "Results" / "tfMRI_LANGUAGE_LR"
    )
    evs_dir = run_folder / "EVs"
    evs_dir.mkdir(parents=True, exist_ok=True)
    _write_dummy_file(run_folder / "brainmask_fs.2.nii.gz")
    _write_dummy_file(run_folder / "tfMRI_LANGUAGE_LR.nii.gz")

    run_folder.mkdir(parents=True, exist_ok=True)
    (run_folder / "Movement_Regressors.txt").write_text(
        " ".join(["0"] * 12) + "\n"
    )
    (evs_dir / "math.txt").write_text("0\t1\t1\n2\t1\t1\n")
    (evs_dir / "story.txt").write_text("1\t1\t1\n3\t1\t1\n")

    hcp._convert_to_bids(data_dir=data_dir, bids_dir=bids_dir, subject=subject, task=task)

    bids_func = bids_dir / f"sub-{subject}" / "func"
    assert bids_func.exists()

    # Expected naming
    bids_prefix_no_space = f"sub-{subject}_task-LANGUAGE_run-1_acq-LR"
    bids_prefix = bids_prefix_no_space + "_space-MNINonLinear"
    assert (bids_func / f"{bids_prefix}_desc-brain_mask.nii.gz").exists()
    assert (bids_func / f"{bids_prefix}_desc-preproc_bold.nii.gz").exists()
    bold_json = bids_func / f"{bids_prefix}_bold.json"
    assert bold_json.exists()
    meta = json.loads(bold_json.read_text())
    assert meta["TaskName"] == task
    assert meta["PhaseEncodingDirection"] == "i-"  # LR -> i-
    conf_tsv = bids_func / f"{bids_prefix_no_space}_desc-confounds_timeseries.tsv"
    assert conf_tsv.exists()
    df_conf = pd.read_csv(conf_tsv, sep="\t")
    assert list(df_conf.columns) == [
        "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
        "trans_dx", "trans_dy", "trans_dz",
        "rot_dx", "rot_dy", "rot_dz",
    ]
    assert df_conf.shape[0] == 1

    events_tsv = bids_func / f"{bids_prefix_no_space}_events.tsv"
    assert events_tsv.exists()
    df_ev = pd.read_csv(events_tsv, sep="\t")
    assert list(df_ev.columns) == ["trial_type", "onset", "duration"]
    assert set(df_ev["trial_type"]) == {"math", "story"}
    assert df_ev["onset"].tolist() == sorted(df_ev["onset"].tolist())


def test_fetch_data_rejects_unknown_task(tmp_path):
    with pytest.raises(ValueError, match="Unsupported task"):
        hcp.fetch_data(data_dir=tmp_path, task="NOT_A_TASK", subjects=["100307"])

def test_list_s3_objects_returns_keys_and_handles_empty_contents():
    s3 = boto3.client("s3", region_name="us-east-1")
    stubber = Stubber(s3)

    # Case 1: has Contents
    stubber.add_response(
        "list_objects_v2",
        {
            "IsTruncated": False,
            "Name": "hcp-openaccess",
            "Prefix": "HCP_1200/100307/MNINonLinear/Results/tfMRI_LANGUAGE_LR",
            "KeyCount": 2,
            "MaxKeys": 1000,
            "Contents": [{"Key": "a"}, {"Key": "b"}],
        },
        {
            "Bucket": "hcp-openaccess",
            "Prefix": "HCP_1200/100307/MNINonLinear/Results/tfMRI_LANGUAGE_LR",
        },
    )

    # Case 2: no Contents (empty listing)
    stubber.add_response(
        "list_objects_v2",
        {
            "IsTruncated": False,
            "Name": "hcp-openaccess",
            "Prefix": "HCP_1200/100307/MNINonLinear/Results/tfMRI_LANGUAGE_RL",
            "KeyCount": 0,
            "MaxKeys": 1000,
            # no "Contents"
        },
        {
            "Bucket": "hcp-openaccess",
            "Prefix": "HCP_1200/100307/MNINonLinear/Results/tfMRI_LANGUAGE_RL",
        },
    )

    stubber.activate()
    keys = hcp._list_s3_objects(
        s3,
        "hcp-openaccess",
        "HCP_1200/100307/MNINonLinear/Results/tfMRI_LANGUAGE_LR",
    )
    assert keys == ["a", "b"]

    keys_empty = hcp._list_s3_objects(
        s3,
        "hcp-openaccess",
        "HCP_1200/100307/MNINonLinear/Results/tfMRI_LANGUAGE_RL",
    )
    assert keys_empty == []
    stubber.deactivate()


def test_download_file_creates_parent_dir_and_calls_boto_download(tmp_path: Path):
    s3 = Mock()
    s3.download_file = Mock()

    local_path = tmp_path / "nested" / "dir" / "file.nii.gz"
    assert not local_path.parent.exists()

    hcp._download_file(s3, "hcp-openaccess", "some/key", str(local_path))

    assert local_path.parent.exists()
    s3.download_file.assert_called_once_with(
        "hcp-openaccess", "some/key", str(local_path)
    )

def test_download_file_catches_exception_and_prints(capsys, tmp_path: Path):
    s3 = Mock()
    s3.download_file.side_effect = RuntimeError("boom")

    local_path = tmp_path / "x" / "y" / "z.nii.gz"
    hcp._download_file(s3, "hcp-openaccess", "missing/key", str(local_path))

    out = capsys.readouterr().out
    assert "boom" in out
    assert "Missing or failed: missing/key" in out


def test_download_selected_lists_both_patterns_and_downloads_all(monkeypatch, tmp_path: Path):
    def fake_list_s3_objects(s3_client, bucket_name, prefix):
        assert bucket_name == "hcp-openaccess"
        if prefix.endswith("tfMRI_LANGUAGE_LR"):
            return [
                "HCP_1200/100307/MNINonLinear/Results/tfMRI_LANGUAGE_LR/a.nii.gz",
                "HCP_1200/100307/MNINonLinear/Results/tfMRI_LANGUAGE_LR/b.txt",
            ]
        if prefix.endswith("tfMRI_LANGUAGE_RL"):
            return [
                "HCP_1200/100307/MNINonLinear/Results/tfMRI_LANGUAGE_RL/c.nii.gz",
            ]
        return []

    calls = []

    def fake_download_file(s3_client, bucket_name, s3_key, local_path):
        calls.append((bucket_name, s3_key, Path(local_path)))

    monkeypatch.setattr(hcp, "_list_s3_objects", fake_list_s3_objects)
    monkeypatch.setattr(hcp, "_download_file", fake_download_file)

    s3_client = Mock()
    subject = "100307"
    task = "LANGUAGE"

    hcp._download_selected(tmp_path, s3_client, subject, task)
    assert len(calls) == 3
    for bucket_name, s3_key, local_path in calls:
        assert bucket_name == "hcp-openaccess"
        assert local_path == tmp_path / s3_key


def test_fetch_data_happy_path(tmp_path: Path, monkeypatch):
    fake_s3 = Mock(name="fake_s3")

    boto_client_mock = Mock(return_value=fake_s3)
    monkeypatch.setattr(hcp.boto3, "client", boto_client_mock)

    download_selected_mock = Mock()
    convert_to_bids_mock = Mock()
    rmtree_mock = Mock()

    monkeypatch.setattr(hcp, "_download_selected", download_selected_mock)
    monkeypatch.setattr(hcp, "_convert_to_bids", convert_to_bids_mock)
    monkeypatch.setattr(hcp.shutil, "rmtree", rmtree_mock)

    data_dir = tmp_path / "my_data"
    subjects = ["100307", "100408"]
    task = "LANGUAGE"

    hcp.fetch_data(data_dir=data_dir, task=task, subjects=subjects)

    assert data_dir.exists()
    assert (data_dir / "bids").exists()

    boto_client_mock.assert_called_once_with("s3")

    assert download_selected_mock.call_count == len(subjects)
    assert convert_to_bids_mock.call_count == len(subjects)

    for i, subj in enumerate(subjects):
        args, kwargs = download_selected_mock.call_args_list[i]
        assert args[0] == data_dir
        assert args[1] is fake_s3
        assert args[2] == subj
        assert args[3] == task
        args, kwargs = convert_to_bids_mock.call_args_list[i]
        assert args[0] == data_dir / "HCP_1200"
        assert args[1] == data_dir / "bids"
        assert args[2] == subj
        assert args[3] == task


def test_fetch_data_catches_subject_error_and_continues(tmp_path, monkeypatch, capsys):
    fake_s3 = Mock()
    monkeypatch.setattr(hcp.boto3, "client", Mock(return_value=fake_s3))
    monkeypatch.setattr(hcp, "_download_selected", Mock())

    def boom(*args, **kwargs):
        raise RuntimeError("nope")

    convert_mock = Mock(side_effect=[None, RuntimeError("nope"), None])
    monkeypatch.setattr(hcp, "_convert_to_bids", convert_mock)

    rmtree_mock = Mock()
    monkeypatch.setattr(hcp.shutil, "rmtree", rmtree_mock)

    subjects = ["S1", "S2", "S3"]
    hcp.fetch_data(data_dir=tmp_path / "data", task="LANGUAGE", subjects=subjects)

    out = capsys.readouterr().out
    assert "Error processing S2: nope" in out

    assert convert_mock.call_count == 3


@pytest.mark.parametrize(
    "task,expected_events",
    [
        ("LANGUAGE", {"math", "story"}),
        ("MOTOR", {"cue", "t", "lf", "rf", "lh", "rh"}),
        ("WM", {
            "0bk_body","0bk_faces","0bk_places","0bk_tools",
            "2bk_body","2bk_faces","2bk_places","2bk_tools",
        }),
        ("SOCIAL", {"mental", "rnd"}),
    ],
)
def test_convert_to_bids_event_branches(tmp_path, task, expected_events):
    subject = "100307"
    data_dir = tmp_path / "HCP_1200"
    bids_dir = tmp_path / "bids"

    run = data_dir / subject / "MNINonLinear" / "Results" / f"tfMRI_{task}_LR"
    evs = run / "EVs"
    evs.mkdir(parents=True)

    (run / "brainmask_fs.2.nii.gz").write_bytes(b"x")
    (run / f"tfMRI_{task}_LR.nii.gz").write_bytes(b"x")
    (run / "Movement_Regressors.txt").write_text(" ".join(["0"] * 12))
    for ev in expected_events:
        (evs / f"{ev}.txt").write_text("0\t1\t1\n")

    hcp._convert_to_bids(data_dir, bids_dir, subject, task)

    events_tsv = bids_dir / f"sub-{subject}" / "func" / (
        f"sub-{subject}_task-{task}_run-1_acq-LR_events.tsv"
    )
    df = pd.read_csv(events_tsv, sep="\t")
    assert set(df["trial_type"]) == expected_events

