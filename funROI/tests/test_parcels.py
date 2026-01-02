import json

import numpy as np
import pytest
from nibabel.nifti1 import Nifti1Image

import funROI
import funROI.parcels as parcels_mod


@pytest.fixture(autouse=True)
def _reset_and_set_output(tmp_path):
    funROI.reset_settings()
    funROI.set_analysis_output_folder(tmp_path / "analysis_out")
    yield
    funROI.reset_settings()


def _make_parcels_img(data: np.ndarray) -> Nifti1Image:
    affine = np.eye(4)
    return Nifti1Image(data.astype(np.float32), affine)


def test_parcels_config_eq_repr_and_from_analysis_output(tmp_path):
    name = "demo"
    sm = 4
    vox = 0.2
    roi = 0.3
    sz = 10
    use_spm = True

    parcels_dir = (funROI.get_analysis_output_folder() / "parcels" / f"parcels-{name}")
    parcels_dir.mkdir(parents=True, exist_ok=True)

    parcels_path = parcels_dir / (
        f"parcels-{name}_sm-{sm}_spmsmooth-{use_spm}_voxthres-{vox}_roithres-{roi}_sz-{sz}.nii.gz"
    )
    _make_parcels_img(np.zeros((3, 3, 3))).to_filename(parcels_path)

    labels_path = parcels_dir / (
        f"parcels-{name}_sm-{sm}_spmsmooth-{use_spm}_voxthres-{vox}_roithres-{roi}_sz-{sz}.json"
    )
    labels = {"1": "A"}
    labels_path.write_text(json.dumps(labels))

    cfg = parcels_mod.ParcelsConfig.from_analysis_output(
        name=name,
        smoothing_kernel_size=sm,
        overlap_thr_vox=vox,
        overlap_thr_roi=roi,
        min_voxel_size=sz,
        use_spm_smooth=use_spm,
    )
    assert cfg.parcels_path == parcels_path
    assert cfg.labels_path == labels_path

    # repr + eq
    cfg2 = parcels_mod.ParcelsConfig(parcels_path=parcels_path, labels_path=labels_path)
    assert cfg == cfg2
    assert "ParcelsConfig(" in repr(cfg)


def test_from_analysis_output_raises_if_parcels_missing():
    with pytest.raises(FileNotFoundError, match="Parcels file not found"):
        parcels_mod.ParcelsConfig.from_analysis_output(
            name="missing",
            smoothing_kernel_size=1,
            overlap_thr_vox=0.1,
            overlap_thr_roi=0.1,
            min_voxel_size=5,
            use_spm_smooth=True,
        )


def test_save_parcels_and_get_parcels_default_labels():
    # save parcels without label file (save_parcels always writes json though),
    # but we test default labeling via _get_external_parcels by passing labels_path=None.
    data = np.zeros((4, 4, 4), dtype=float)
    data[0, 0, 0] = 1.49  # should round to 1
    data[1, 1, 1] = 2.51  # should round to 3 (np.round)
    img = _make_parcels_img(data)

    # write file directly (no labels)
    parcels_path = funROI.get_analysis_output_folder() / "parcels" / "ext.nii.gz"
    parcels_path.parent.mkdir(parents=True, exist_ok=True)
    img.to_filename(parcels_path)

    parcels_img, label_dict = parcels_mod.get_parcels(
        parcels_mod.ParcelsConfig(parcels_path=parcels_path, labels_path=None)
    )

    assert parcels_img is not None
    # default labels: int(label)->int(label), excluding 0
    assert label_dict == {1: 1, 3: 3}


def test_get_external_parcels_loads_labels_from_json_and_txt(tmp_path):
    # parcels image with labels {1,2}
    data = np.zeros((3, 3, 3), dtype=float)
    data[0, 0, 0] = 1
    data[0, 0, 1] = 2
    parcels_path = funROI.get_analysis_output_folder() / "parcels" / "p.nii.gz"
    parcels_path.parent.mkdir(parents=True, exist_ok=True)
    _make_parcels_img(data).to_filename(parcels_path)

    # JSON labels
    json_path = parcels_path.with_suffix(".json")
    json_path.write_text(json.dumps({"1": "Left", "2": "Right"}))

    img_json, labels_json = parcels_mod.get_parcels(
        parcels_mod.ParcelsConfig(parcels_path=parcels_path, labels_path=json_path)
    )
    assert labels_json == {1: "Left", 2: "Right"}

    # TXT labels (line i -> label i+1)
    txt_path = parcels_path.with_suffix(".txt")
    txt_path.write_text("Alpha\nBeta\n")

    img_txt, labels_txt = parcels_mod.get_parcels(
        parcels_mod.ParcelsConfig(parcels_path=parcels_path, labels_path=txt_path)
    )
    assert labels_txt == {1: "Alpha", 2: "Beta"}


def test_label_parcel_and_error():
    data = np.zeros((2, 2, 2), dtype=float)
    data[0, 0, 0] = 2
    img = _make_parcels_img(data)
    label_dict = {2: "Two"}

    mask_img, name = parcels_mod.label_parcel(img, label_dict, 2)
    assert name == "Two"
    mask = mask_img.get_fdata()
    assert mask[0, 0, 0] == 1
    assert mask.sum() == 1

    with pytest.raises(ValueError, match="Label 3 not found"):
        parcels_mod.label_parcel(img, label_dict, 3)


def test_merge_parcels_rejects_duplicate_new_label():
    data = np.zeros((3, 3, 3), dtype=float)
    data[0, 0, 0] = 1
    data[0, 0, 1] = 2
    img = _make_parcels_img(data)
    label_dict = {1: "A", 2: "B"}

    with pytest.raises(ValueError, match="already exists"):
        parcels_mod.merge_parcels(img, label_dict, 1, 2, new_label="A")


def test__merge_parcels_validation_and_basic_merge():
    # validation: must be 3D
    with pytest.raises(ValueError, match="Data must be 3D"):
        parcels_mod._merge_parcels(np.zeros((2, 2)), 1, 2)

    # if x==y returns unchanged
    d = np.zeros((3, 3, 3), dtype=float)
    d[1, 1, 1] = 5
    out = parcels_mod._merge_parcels(d.copy(), 5, 5)
    assert np.array_equal(out, d)

    # merge: all y becomes x (at minimum)
    d2 = np.zeros((3, 3, 3), dtype=float)
    d2[0, 0, 0] = 1
    d2[0, 0, 1] = 2
    out2 = parcels_mod._merge_parcels(d2.copy(), 1, 2)
    assert 2 not in np.unique(out2)
    assert 1 in np.unique(out2)


def test_save_parcels_writes_nii_and_json():
    data = np.zeros((2, 2, 2), dtype=float)
    data[0, 0, 0] = 1
    img = _make_parcels_img(data)
    labels = {1: "One"}

    parcels_mod.save_parcels(img, labels, name="saved_parcels")

    parcels_folder = funROI.get_analysis_output_folder() / "parcels"
    assert (parcels_folder / "saved_parcels.nii.gz").exists()
    assert (parcels_folder / "saved_parcels.json").exists()

    loaded = json.loads((parcels_folder / "saved_parcels.json").read_text())
    # json keys become strings
    assert loaded["1"] == "One"


def _img(data):
    return Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4))

def test_merge_parcels_merges_by_name_and_updates_label_dict(tmp_path):
    funROI.reset_settings()
    funROI.set_analysis_output_folder(tmp_path / "analysis_out")

    # labels 1="A", 2="B" adjacent
    data = np.zeros((3, 3, 3), dtype=float)
    data[1, 1, 1] = 1
    data[1, 1, 2] = 2
    img = _img(data)
    label_dict = {1: "A", 2: "B"}

    merged_img, merged_labels = parcels_mod.merge_parcels(
        img, label_dict.copy(), "A", "B", new_label="AB"
    )

    merged_data = merged_img.get_fdata()
    assert 2 not in np.unique(merged_data)
    assert 1 in np.unique(merged_data)

    # old labels removed
    assert 1 not in merged_labels
    assert 2 not in merged_labels
    assert merged_labels["AB"] == "AB"


def test_get_parcels_string_prefers_saved_and_falls_back_to_external(tmp_path):
    funROI.reset_settings()
    funROI.set_analysis_output_folder(tmp_path / "analysis_out")

    parcels_folder = funROI.get_analysis_output_folder() / "parcels"
    parcels_folder.mkdir(parents=True, exist_ok=True)

    saved_path = parcels_folder / "parcels-demo_mask.nii.gz"
    saved_img = _img(np.zeros((2, 2, 2)))
    saved_img.to_filename(saved_path)

    img1, labels1 = parcels_mod.get_parcels("demo")
    assert img1 is not None
    assert labels1 == {}  # default labeling: all zeros -> empty dict

    ext_path = tmp_path / "external.nii.gz"
    ext_data = np.zeros((2, 2, 2), dtype=float)
    ext_data[0, 0, 0] = 1
    ext_data[0, 0, 1] = 2
    _img(ext_data).to_filename(ext_path)

    img2, labels2 = parcels_mod.get_parcels(str(ext_path))
    assert img2 is not None
    assert labels2 == {1: 1, 2: 2}
