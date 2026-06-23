import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from nilearn.surface import InMemoryMesh, SurfaceImage, load_surf_data

import funROI
import funROI.analysis.surface_parcels_gen as surface_parcels_gen_mod
from funROI._surface import write_gifti


def _mesh(offset: float = 0.0) -> InMemoryMesh:
    coordinates = np.array(
        [
            [0.0 + offset, 0.0, 0.0],
            [1.0 + offset, 0.0, 0.0],
            [0.0 + offset, 1.0, 0.0],
            [1.0 + offset, 1.0, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    return InMemoryMesh(coordinates, faces)


@pytest.fixture
def tmp_settings(tmp_path):
    out = tmp_path / "analysis_out"
    funROI.set_analysis_output_folder(out)
    yield out
    funROI.reset_settings()


def test_add_subjects_collects_surface_masks(monkeypatch, tmp_settings):
    calls = []
    meshes = {"left": _mesh(), "right": _mesh(2.0)}
    mesh_paths = {"L": Path("/tmp/L.surf.gii"), "R": Path("/tmp/R.surf.gii")}

    monkeypatch.setattr(
        surface_parcels_gen_mod.SurfaceParcelsGenerator,
        "_get_subject_runs",
        lambda self, subject, task, contrasts: ["01", "02", "03"],
    )
    monkeypatch.setattr(
        surface_parcels_gen_mod.SurfaceParcelsGenerator,
        "_get_subject_meshes",
        lambda self, subject: (meshes, mesh_paths),
    )

    def fake_surface_contrast(self, subject, task, run_label, contrast, image_type):
        calls.append((subject, run_label, contrast, image_type))
        values = np.array([0.001, 0.2, 0.001, 0.2], dtype=float)
        return {"L": values.copy(), "R": values.copy()}

    monkeypatch.setattr(
        surface_parcels_gen_mod.SurfaceParcelsGenerator,
        "_get_surface_contrast_data",
        fake_surface_contrast,
    )

    gen = surface_parcels_gen_mod.SurfaceParcelsGenerator("surf")
    gen.add_subjects(
        subjects=["S1"],
        task="LANGUAGE",
        contrasts=["story", "math"],
        p_threshold_type="none",
        p_threshold_value=0.05,
        conjunction_type="and",
    )

    assert gen.hemi_sizes == {"left": 4, "right": 4}
    assert gen.hemi_slices["left"] == slice(0, 4)
    assert gen.hemi_slices["right"] == slice(4, 8)
    assert gen._data[0].shape == (3, 8)
    assert np.array_equal(
        gen._data[0][0],
        np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float),
    )
    assert [run for _, run, _, _ in calls[::2]] == [
        "orth01",
        "orth02",
        "orth03",
    ]


def test_watershed_surface_segments_separate_peaks():
    mesh = _mesh()
    values = np.array([0.9, 0.2, 0.2, 0.8], dtype=float)

    labels = surface_parcels_gen_mod.SurfaceParcelsGenerator._watershed_surface(
        values, mesh
    )

    assert labels[0] > 0
    assert labels[3] > 0
    assert labels[0] != labels[3]


def test_run_saves_surface_outputs(monkeypatch, tmp_settings):
    gen = surface_parcels_gen_mod.SurfaceParcelsGenerator(
        "surf",
        space="fsLR32k",
        smoothing_kernel_size=6,
        overlap_thr_vox=0.25,
    )
    gen.mesh = {"left": _mesh(), "right": _mesh(2.0)}
    gen.mesh_paths = {"L": Path("/tmp/L.surf.gii"), "R": Path("/tmp/R.surf.gii")}
    gen.hemi_sizes = {"left": 4, "right": 4}
    gen.hemi_slices = {"left": slice(0, 4), "right": slice(4, 8)}
    gen._data = [
        np.array([[1, 1, 0, 0, 0, 0, 1, 1]], dtype=float),
        np.array([[1, 1, 0, 0, 0, 0, 1, 1]], dtype=float),
    ]

    overlap_map = np.array([1, 1, 0, 0, 0, 0, 1, 1], dtype=float)
    parcels = np.array([1, 1, 0, 0, 0, 0, 2, 2], dtype=int)

    monkeypatch.setattr(
        surface_parcels_gen_mod.SurfaceParcelsGenerator,
        "_run",
        classmethod(lambda cls, *args, **kwargs: (overlap_map, parcels)),
    )

    out = gen.run()

    assert isinstance(out, SurfaceImage)
    assert np.array_equal(out.data.parts["left"], np.array([1, 1, 0, 0]))
    assert np.array_equal(out.data.parts["right"], np.array([0, 0, 2, 2]))

    base = tmp_settings / "parcels" / "parcels-surf"
    assert (base / "parcels-surf_config.json").exists()
    assert (base / "parcels-surf_space-fsLR32k_overlap_hemi-L.func.gii").exists()
    assert (base / "parcels-surf_space-fsLR32k_overlap_hemi-R.func.gii").exists()

    info_csv = (
        base / "parcels-surf_space-fsLR32k_sm-6_voxthres-0.25_info.csv"
    )
    assert info_csv.exists()
    df = pd.read_csv(info_csv)
    assert set(df.columns) == {"id", "size", "roi_overlap"}

    left_path = (
        base
        / "parcels-surf_space-fsLR32k_sm-6_voxthres-0.25_roithres-0_sz-0_hemi-L.func.gii"
    )
    right_path = (
        base
        / "parcels-surf_space-fsLR32k_sm-6_voxthres-0.25_roithres-0_sz-0_hemi-R.func.gii"
    )
    assert np.array_equal(load_surf_data(left_path), np.array([1, 1, 0, 0]))
    assert np.array_equal(load_surf_data(right_path), np.array([0, 0, 2, 2]))


def test_filter_removes_small_or_low_overlap_parcels(tmp_settings):
    gen = surface_parcels_gen_mod.SurfaceParcelsGenerator("surf")
    gen.mesh = {"left": _mesh(), "right": _mesh(2.0)}
    gen.hemi_sizes = {"left": 4, "right": 4}
    gen.hemi_slices = {"left": slice(0, 4), "right": slice(4, 8)}
    gen.parcels = np.array([1, 1, 2, 0, 0, 0, 3, 3], dtype=int)
    gen.parcel_info = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "size": [2, 1, 2],
            "roi_overlap": [1.0, 0.8, 0.1],
        }
    )

    out = gen.filter(overlap_thr_roi=0.5, min_voxel_size=2)

    assert isinstance(out, SurfaceImage)
    assert np.array_equal(
        np.concatenate([out.data.parts["left"], out.data.parts["right"]]),
        np.array([1, 1, 0, 0, 0, 0, 0, 0]),
    )


def test_run_fast_reuses_saved_surface_overlap_map(monkeypatch, tmp_settings):
    base = tmp_settings / "parcels" / "parcels-surf"
    base.mkdir(parents=True, exist_ok=True)

    (base / "parcels-surf_config.json").write_text(
        json.dumps(
            {
                "configs": [],
                "space": "fsLR32k",
                "mesh_paths": {
                    "L": "/tmp/L.surf.gii",
                    "R": "/tmp/R.surf.gii",
                },
            }
        )
    )
    write_gifti(
        base / "parcels-surf_space-fsLR32k_overlap_hemi-L.func.gii",
        np.array([1, 1, 0, 0], dtype=np.float32),
    )
    write_gifti(
        base / "parcels-surf_space-fsLR32k_overlap_hemi-R.func.gii",
        np.array([0, 0, 1, 1], dtype=np.float32),
    )

    monkeypatch.setattr(
        surface_parcels_gen_mod,
        "load_surf_mesh",
        lambda path: _mesh() if "L.surf.gii" in str(path) else _mesh(2.0),
    )

    expected_parcels = np.array([1, 1, 0, 0, 0, 0, 2, 2], dtype=int)
    monkeypatch.setattr(
        surface_parcels_gen_mod.SurfaceParcelsGenerator,
        "_run",
        classmethod(
            lambda cls, binary_masks, *args, **kwargs: (
                np.asarray(binary_masks[0]),
                expected_parcels,
            )
        ),
    )

    gen = surface_parcels_gen_mod.SurfaceParcelsGenerator._run_fast(
        "surf",
        space="fsLR32k",
        smoothing_kernel_size=6,
        overlap_thr_vox=0.25,
    )

    assert isinstance(gen, surface_parcels_gen_mod.SurfaceParcelsGenerator)
    assert np.array_equal(
        gen.overlap_map,
        np.array([1, 1, 0, 0, 0, 0, 1, 1], dtype=np.float32),
    )
    assert np.array_equal(gen.parcels, expected_parcels)
    assert list(gen.parcel_info.columns) == ["id", "size"]
    assert (
        base
        / "parcels-surf_space-fsLR32k_sm-6_voxthres-0.25_roithres-0_sz-0_hemi-L.func.gii"
    ).exists()
