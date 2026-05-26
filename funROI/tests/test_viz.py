from pathlib import Path

import numpy as np
from nilearn.surface import InMemoryMesh, SurfaceImage

import funROI.viz as viz_mod


def _mesh(offset: float = 0.0) -> InMemoryMesh:
    coordinates = np.array(
        [
            [0.0 + offset, 0.0, 0.0],
            [1.0 + offset, 0.0, 0.0],
            [0.0 + offset, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return InMemoryMesh(coordinates, faces)


class _DummyArtist:
    def __init__(self):
        self.clip_on = True

    def set_clip_on(self, value):
        self.clip_on = value


class _DummyAxis:
    def __init__(self):
        self.collections = [_DummyArtist()]
        self._xlim = (0.0, 1.0)
        self._ylim = (0.0, 1.0)
        self._zlim = (0.0, 1.0)
        self.axis_off = False

    def get_xlim(self):
        return self._xlim

    def get_ylim(self):
        return self._ylim

    def get_zlim(self):
        return self._zlim

    def set_xlim(self, limits):
        self._xlim = tuple(limits)

    def set_ylim(self, limits):
        self._ylim = tuple(limits)

    def set_zlim(self, limits):
        self._zlim = tuple(limits)

    def set_axis_off(self):
        self.axis_off = True


class _DummyFigure:
    def subplots_adjust(self, **kwargs):
        self.subplots_adjust_kwargs = kwargs

    def savefig(self, path, **kwargs):
        Path(path).write_bytes(b"png")
        self.savefig_kwargs = kwargs


class _DummyPyplot:
    def __init__(self):
        self.closed = []

    def subplots(self, nrows, ncols, **kwargs):
        fig = _DummyFigure()
        axes = np.array(
            [[_DummyAxis() for _ in range(ncols)] for _ in range(nrows)],
            dtype=object,
        )
        return fig, axes

    def close(self, fig):
        self.closed.append(fig)


class _DummyPlotting:
    def __init__(self):
        self.calls = []

    def plot_surf_stat_map(self, **kwargs):
        self.calls.append(("stat", kwargs))

    def plot_surf_roi(self, **kwargs):
        self.calls.append(("roi", kwargs))


class _DummySurfaceModule:
    SurfaceImage = SurfaceImage

    def __init__(self):
        self.loaded_data = []
        self.loaded_meshes = []

    def load_surf_data(self, path):
        self.loaded_data.append(str(path))
        if "left_bg" in str(path):
            return np.array([0.0, 1.0, 2.0], dtype=float)
        if "right_bg" in str(path):
            return np.array([2.0, 1.0, 0.0], dtype=float)
        if "left" in str(path):
            return np.array([1.0, 2.0, 3.0], dtype=float)
        return np.array([4.0, 5.0, 6.0], dtype=float)

    def load_surf_mesh(self, path):
        self.loaded_meshes.append(str(path))
        return f"mesh:{Path(path).name}"


class _DummyGiftiArray:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)


class _DummyGiftiImage:
    def __init__(self, data):
        self.darrays = [_DummyGiftiArray(data)]


class _DummyTransforms:
    def mni152_to_fsaverage(self, data, fsavg_density, method):
        self.last_call = {
            "data": data,
            "fsavg_density": fsavg_density,
            "method": method,
        }
        return (
            _DummyGiftiImage([1.0, 2.0, 0.0]),
            _DummyGiftiImage([0.0, 3.0, 4.0]),
        )


class _DummyFsaverage:
    pial_left = "mesh:pial_left"
    pial_right = "mesh:pial_right"
    infl_left = "mesh:infl_left"
    infl_right = "mesh:infl_right"
    sulc_left = "sulc:left"
    sulc_right = "sulc:right"


def test_plot_surface_stat_map_accepts_surface_image(monkeypatch, tmp_path):
    pyplot = _DummyPyplot()
    plotting = _DummyPlotting()
    surface = _DummySurfaceModule()
    monkeypatch.setattr(
        viz_mod,
        "_import_surface_plot_dependencies",
        lambda: (pyplot, plotting, surface),
    )

    img = SurfaceImage(
        mesh={"left": _mesh(), "right": _mesh(2.0)},
        data={
            "left": np.array([1.0, 2.0, 3.0], dtype=float),
            "right": np.array([4.0, 5.0, 6.0], dtype=float),
        },
    )

    output_path = viz_mod.plot_surface_stat_map(
        img,
        tmp_path / "surface_plot",
        views=("lateral", "medial"),
        hemispheres=("left", "right"),
    )

    assert output_path == tmp_path / "surface_plot.png"
    assert output_path.exists()
    assert len(plotting.calls) == 4
    assert all(kind == "stat" for kind, _ in plotting.calls)
    assert np.array_equal(
        plotting.calls[0][1]["stat_map"],
        np.array([1.0, 2.0, 3.0], dtype=float),
    )
    assert plotting.calls[0][1]["surf_mesh"] is img.mesh.parts["left"]


def test_plot_surface_stat_map_accepts_file_backed_mappings(
    monkeypatch, tmp_path
):
    pyplot = _DummyPyplot()
    plotting = _DummyPlotting()
    surface = _DummySurfaceModule()
    monkeypatch.setattr(
        viz_mod,
        "_import_surface_plot_dependencies",
        lambda: (pyplot, plotting, surface),
    )

    output_path = viz_mod.plot_surface_stat_map(
        data={"L": tmp_path / "left.func.gii", "R": tmp_path / "right.func.gii"},
        mesh={"L": tmp_path / "left.surf.gii", "R": tmp_path / "right.surf.gii"},
        bg_maps={
            "L": tmp_path / "left_bg.shape.gii",
            "R": tmp_path / "right_bg.shape.gii",
        },
        output_file_prefix=tmp_path / "mapped_plot",
        hemispheres=("L", "R"),
        views=("lateral",),
    )

    assert output_path == tmp_path / "mapped_plot.png"
    assert output_path.exists()
    assert len(plotting.calls) == 2
    assert all(kind == "stat" for kind, _ in plotting.calls)
    assert plotting.calls[0][1]["surf_mesh"] == "mesh:left.surf.gii"
    assert plotting.calls[1][1]["surf_mesh"] == "mesh:right.surf.gii"
    assert plotting.calls[0][1]["bg_map"] is not None
    assert plotting.calls[1][1]["bg_map"] is not None
    assert plotting.calls[0][1]["bg_on_data"] is True


def test_plot_surface_roi_map_accepts_surface_image(monkeypatch, tmp_path):
    pyplot = _DummyPyplot()
    plotting = _DummyPlotting()
    surface = _DummySurfaceModule()
    monkeypatch.setattr(
        viz_mod,
        "_import_surface_plot_dependencies",
        lambda: (pyplot, plotting, surface),
    )

    img = SurfaceImage(
        mesh={"left": _mesh(), "right": _mesh(2.0)},
        data={
            "left": np.array([0.0, 1.0, 2.0], dtype=float),
            "right": np.array([0.0, 3.0, 4.0], dtype=float),
        },
    )

    output_path = viz_mod.plot_surface_roi_map(
        img,
        tmp_path / "roi_plot",
        views=("lateral",),
        hemispheres=("left", "right"),
        colorbar=True,
    )

    assert output_path == tmp_path / "roi_plot.png"
    assert output_path.exists()
    assert len(plotting.calls) == 2
    assert all(kind == "roi" for kind, _ in plotting.calls)
    assert np.array_equal(
        plotting.calls[0][1]["roi_map"],
        np.array([0.0, 1.0, 2.0], dtype=float),
    )
    assert plotting.calls[0][1]["colorbar"] is True
    assert plotting.calls[0][1]["avg_method"] == "median"


def test_plot_surface_roi_map_accepts_file_backed_mappings(
    monkeypatch, tmp_path
):
    pyplot = _DummyPyplot()
    plotting = _DummyPlotting()
    surface = _DummySurfaceModule()
    monkeypatch.setattr(
        viz_mod,
        "_import_surface_plot_dependencies",
        lambda: (pyplot, plotting, surface),
    )

    output_path = viz_mod.plot_surface_roi_map(
        data={"L": tmp_path / "left.func.gii", "R": tmp_path / "right.func.gii"},
        mesh={"L": tmp_path / "left.surf.gii", "R": tmp_path / "right.surf.gii"},
        bg_maps={
            "L": tmp_path / "left_bg.shape.gii",
            "R": tmp_path / "right_bg.shape.gii",
        },
        output_file_prefix=tmp_path / "mapped_roi_plot",
        hemispheres=("L", "R"),
        views=("medial",),
    )

    assert output_path == tmp_path / "mapped_roi_plot.png"
    assert output_path.exists()
    assert len(plotting.calls) == 2
    assert all(kind == "roi" for kind, _ in plotting.calls)
    assert plotting.calls[0][1]["surf_mesh"] == "mesh:left.surf.gii"
    assert plotting.calls[1][1]["surf_mesh"] == "mesh:right.surf.gii"
    assert plotting.calls[0][1]["bg_map"] is not None
    assert plotting.calls[1][1]["bg_map"] is not None


def test_plot_mni152_surface_roi_map_projects_and_renders(
    monkeypatch, tmp_path
):
    pyplot = _DummyPyplot()
    plotting = _DummyPlotting()
    surface = _DummySurfaceModule()
    transforms = _DummyTransforms()
    monkeypatch.setattr(
        viz_mod,
        "_import_visualization_dependencies",
        lambda: (pyplot, None, plotting, surface, transforms),
    )
    monkeypatch.setattr(
        viz_mod,
        "_get_fsaverage",
        lambda density: _DummyFsaverage(),
    )

    output_path = viz_mod.plot_mni152_surface_roi_map(
        data="dummy_volume.nii.gz",
        output_file_prefix=tmp_path / "mni_roi_plot",
        views=("lateral",),
        hemispheres=("left", "right"),
        inflate=True,
        colorbar=True,
    )

    assert output_path == tmp_path / "mni_roi_plot_inflated.png"
    assert output_path.exists()
    assert transforms.last_call["data"] == "dummy_volume.nii.gz"
    assert transforms.last_call["fsavg_density"] == "41k"
    assert transforms.last_call["method"] == "nearest"
    assert len(plotting.calls) == 2
    assert all(kind == "roi" for kind, _ in plotting.calls)
    assert plotting.calls[0][1]["surf_mesh"] == "mesh:infl_left"
    assert plotting.calls[1][1]["surf_mesh"] == "mesh:infl_right"
    assert plotting.calls[0][1]["hemi"] == "left"
    assert plotting.calls[1][1]["hemi"] == "right"
    assert plotting.calls[0][1]["colorbar"] is True
