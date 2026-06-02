from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _import_visualization_dependencies():
    try:
        pyplot = import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise ImportError(
            "Surface visualization requires the optional 'matplotlib' "
            "dependency."
        ) from exc

    try:
        nilearn_datasets = import_module("nilearn.datasets")
        nilearn_plotting = import_module("nilearn.plotting")
        nilearn_surface = import_module("nilearn.surface")
    except ImportError as exc:
        raise ImportError(
            "Surface visualization requires the optional 'nilearn' "
            "dependency."
        ) from exc

    try:
        transforms = import_module("neuromaps.transforms")
    except ImportError as exc:
        raise ImportError(
            "Surface visualization requires the optional 'neuromaps' "
            "dependency. Install funROI with the visualization extras, or "
            "install 'neuromaps' manually."
        ) from exc

    return pyplot, nilearn_datasets, nilearn_plotting, nilearn_surface, transforms


@lru_cache(maxsize=None)
def _get_fsaverage(density: str):
    _, nilearn_datasets, _, _, _ = _import_visualization_dependencies()
    return nilearn_datasets.fetch_surf_fsaverage(density)


def _normalize_background_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.allclose(values.max(), values.min()):
        return np.full_like(values, 0.25, dtype=float)
    return (
        (values - values.min()) / (values.max() - values.min()) * 0.3
    ) + 0.1


def _coerce_output_path(output_file: str | Path, suffix: str) -> Path:
    output_path = Path(output_file)
    if output_path.suffix.lower() != ".png":
        output_path = output_path.with_suffix(".png")
    if suffix:
        output_path = output_path.with_name(
            f"{output_path.stem}{suffix}{output_path.suffix}"
        )
    return output_path


def zoom_3d(ax, scale: float = 0.8) -> None:
    """
    Zoom a 3D axis by rescaling all three bounds around their centers.

    :param ax: Matplotlib 3D axis.
    :param scale: Scale factor for the displayed extent.
        Values below 1 zoom in; values above 1 zoom out.
    :type scale: float
    """
    if scale <= 0:
        raise ValueError("scale must be positive.")

    def _rescale(limits):
        center = (limits[0] + limits[1]) / 2
        radius = (limits[1] - limits[0]) / 2 * scale
        return center - radius, center + radius

    ax.set_xlim(_rescale(ax.get_xlim()))
    ax.set_ylim(_rescale(ax.get_ylim()))
    ax.set_zlim(_rescale(ax.get_zlim()))


def plot_mni152_surface_stat_map(
    data,
    output_file_prefix: str | Path,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    threshold: float | None = None,
    cmap: str = "coolwarm",
    views: Sequence[str] = ("lateral", "medial"),
    hemispheres: Sequence[str] = ("left", "right"),
    inflate: bool = False,
    fsaverage_density: str = "fsaverage6",
    transform_density: str = "41k",
    transform_method: str = "nearest",
    figsize: tuple[float, float] = (8, 5),
    dpi: int = 100,
    zoom_scale: float = 0.61,
) -> Path:
    """
    Project a volumetric MNI152 map to fsaverage and render a 2x2 surface view.

    :param data: Volume image, path, or other MNI152-compatible niimg-like
        input supported by `neuromaps.transforms.mni152_to_fsaverage`.
    :param output_file_prefix: Prefix for the output PNG path.
    :param vmin: Lower color limit.
    :param vmax: Upper color limit.
    :param threshold: Display threshold.
    :param cmap: Matplotlib colormap name.
    :param views: Surface views to render.
    :param hemispheres: Hemispheres to render.
    :param inflate: Whether to use inflated rather than pial surfaces.
    :param fsaverage_density: fsaverage surface density to fetch from nilearn.
    :param transform_density: fsaverage density used for the MNI-to-surface
        projection in neuromaps. `41k` matches `fsaverage6`.
    :param transform_method: Interpolation method used by neuromaps.
    :param figsize: Figure size in inches.
    :param dpi: Output DPI.
    :param zoom_scale: 3D zoom factor applied to each panel.
    :return: Path to the written PNG file.
    """
    if len(views) == 0 or len(hemispheres) == 0:
        raise ValueError("views and hemispheres must be non-empty.")

    pyplot, _, nilearn_plotting, nilearn_surface, transforms = (
        _import_visualization_dependencies()
    )
    fsaverage = _get_fsaverage(fsaverage_density)
    left_gifti, right_gifti = transforms.mni152_to_fsaverage(
        data,
        fsavg_density=transform_density,
        method=transform_method,
    )
    surf_data = {
        "left": np.asarray(left_gifti.darrays[0].data),
        "right": np.asarray(right_gifti.darrays[0].data),
    }
    sulc_maps = {
        hemi: _normalize_background_map(
            nilearn_surface.load_surf_data(
                getattr(fsaverage, f"sulc_{hemi}")
            )
        )
        for hemi in hemispheres
    }

    fig, axes = pyplot.subplots(
        len(views),
        len(hemispheres),
        figsize=figsize,
        dpi=dpi,
        subplot_kw={"projection": "3d"},
        tight_layout=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1,
        wspace=0,
        hspace=0,
    )

    for column, hemisphere in enumerate(hemispheres):
        hemi_short = hemisphere[0].upper()
        for row, view in enumerate(views):
            ax = axes[row, column]
            mesh_name = (
                f"infl_{hemisphere}" if inflate else f"pial_{hemisphere}"
            )
            nilearn_plotting.plot_surf_stat_map(
                surf_mesh=getattr(fsaverage, mesh_name),
                stat_map=surf_data[hemisphere],
                hemi=hemi_short,
                view=view,
                bg_on_data=True,
                bg_map=sulc_maps[hemisphere],
                avg_method="mean",
                figure=fig,
                axes=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                threshold=threshold,
                colorbar=False,
            )
            zoom_3d(ax, scale=zoom_scale)
            ax.set_axis_off()
            for artist in ax.collections:
                artist.set_clip_on(False)

    output_path = _coerce_output_path(
        output_file_prefix,
        f"{'_inflated' if inflate else ''}",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, transparent=True)
    pyplot.close(fig)
    return output_path


def plot_mni152_surface_roi_map(
    data,
    output_file_prefix: str | Path,
    *,
    threshold: float | None = None,
    cmap: str = "gist_ncar",
    views: Sequence[str] = ("lateral", "medial"),
    hemispheres: Sequence[str] = ("left", "right"),
    inflate: bool = False,
    fsaverage_density: str = "fsaverage6",
    transform_density: str = "41k",
    transform_method: str = "nearest",
    figsize: tuple[float, float] = (8, 5),
    dpi: int = 100,
    zoom_scale: float = 0.61,
    colorbar: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """
    Project a volumetric MNI152 ROI/parcellation map to fsaverage and render it.

    This is intended for discrete label maps such as parcels or atlases.

    :param data: Volume image, path, or other MNI152-compatible niimg-like
        input supported by `neuromaps.transforms.mni152_to_fsaverage`.
    :param output_file_prefix: Prefix for the output PNG path.
    :param threshold: Display threshold.
    :param cmap: Matplotlib colormap name.
    :param views: Surface views to render.
    :param hemispheres: Hemispheres to render.
    :param inflate: Whether to use inflated rather than pial surfaces.
    :param fsaverage_density: fsaverage surface density to fetch from nilearn.
    :param transform_density: fsaverage density used for the MNI-to-surface
        projection in neuromaps. `41k` matches `fsaverage6`.
    :param transform_method: Interpolation method used by neuromaps.
    :param figsize: Figure size in inches.
    :param dpi: Output DPI.
    :param zoom_scale: 3D zoom factor applied to each panel.
    :param colorbar: Whether to draw a colorbar.
    :param vmin: Lower color limit.
    :param vmax: Upper color limit.
    :return: Path to the written PNG file.
    """
    if len(views) == 0 or len(hemispheres) == 0:
        raise ValueError("views and hemispheres must be non-empty.")

    pyplot, _, nilearn_plotting, nilearn_surface, transforms = (
        _import_visualization_dependencies()
    )
    fsaverage = _get_fsaverage(fsaverage_density)
    left_gifti, right_gifti = transforms.mni152_to_fsaverage(
        data,
        fsavg_density=transform_density,
        method=transform_method,
    )
    surf_data = {
        "left": np.asarray(left_gifti.darrays[0].data),
        "right": np.asarray(right_gifti.darrays[0].data),
    }
    sulc_maps = {
        hemi: _normalize_background_map(
            nilearn_surface.load_surf_data(
                getattr(fsaverage, f"sulc_{hemi}")
            )
        )
        for hemi in hemispheres
    }

    fig, axes = pyplot.subplots(
        len(views),
        len(hemispheres),
        figsize=figsize,
        dpi=dpi,
        subplot_kw={"projection": "3d"},
        tight_layout=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1,
        wspace=0,
        hspace=0,
    )

    for column, hemisphere in enumerate(hemispheres):
        for row, view in enumerate(views):
            ax = axes[row, column]
            mesh_name = (
                f"infl_{hemisphere}" if inflate else f"pial_{hemisphere}"
            )
            nilearn_plotting.plot_surf_roi(
                surf_mesh=getattr(fsaverage, mesh_name),
                roi_map=surf_data[hemisphere],
                hemi=hemisphere,
                view=view,
                bg_on_data=True,
                bg_map=sulc_maps[hemisphere],
                avg_method="median",
                figure=fig,
                axes=ax,
                cmap=cmap,
                threshold=threshold,
                colorbar=colorbar,
                vmin=vmin,
                vmax=vmax,
            )
            zoom_3d(ax, scale=zoom_scale)
            ax.set_axis_off()
            for artist in ax.collections:
                artist.set_clip_on(False)

    output_path = _coerce_output_path(
        output_file_prefix,
        f"{'_inflated' if inflate else ''}",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, transparent=True)
    pyplot.close(fig)
    return output_path


def _import_surface_plot_dependencies():
    try:
        pyplot = import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise ImportError(
            "Surface visualization requires the optional 'matplotlib' "
            "dependency."
        ) from exc

    try:
        nilearn_plotting = import_module("nilearn.plotting")
        nilearn_surface = import_module("nilearn.surface")
    except ImportError as exc:
        raise ImportError(
            "Surface visualization requires the optional 'nilearn' "
            "dependency."
        ) from exc

    return pyplot, nilearn_plotting, nilearn_surface


def _normalize_hemisphere_name(hemisphere: str) -> str:
    hemisphere_norm = hemisphere.lower()
    if hemisphere_norm in {"l", "left"}:
        return "left"
    if hemisphere_norm in {"r", "right"}:
        return "right"
    raise ValueError(
        "Hemisphere keys must be one of: 'L', 'R', 'left', or 'right'."
    )


def _normalize_surface_mapping(
    values: Mapping[str, object], name: str
) -> dict[str, object]:
    normalized = {}
    for hemisphere, value in values.items():
        normalized[_normalize_hemisphere_name(hemisphere)] = value

    if len(normalized) == 0:
        raise ValueError(f"{name} must contain at least one hemisphere.")
    return normalized


def _coerce_surface_vector(
    value, nilearn_surface, *, name: str, normalize: bool = False
) -> np.ndarray:
    if isinstance(value, (str, Path)):
        value = nilearn_surface.load_surf_data(value)

    array = np.asarray(value, dtype=float)
    array = np.squeeze(array)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional per hemisphere.")
    if normalize:
        array = _normalize_background_map(array)
    return array


def _coerce_surface_mesh(value, nilearn_surface):
    if isinstance(value, (str, Path)):
        return nilearn_surface.load_surf_mesh(value)
    return value


def _coerce_surface_plot_inputs(
    data,
    mesh,
    nilearn_surface,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    surface_image_type = getattr(nilearn_surface, "SurfaceImage", None)
    if surface_image_type is not None and isinstance(data, surface_image_type):
        return (
            dict(data.mesh.parts),
            {
                hemisphere: _coerce_surface_vector(
                    data.data.parts[hemisphere],
                    nilearn_surface,
                    name=f"data[{hemisphere}]",
                )
                for hemisphere in data.data.parts
            },
        )

    if not isinstance(data, Mapping):
        raise TypeError(
            "data must be a nilearn SurfaceImage or a hemisphere mapping."
        )
    if mesh is None:
        raise ValueError(
            "mesh is required when data is not provided as a SurfaceImage."
        )
    if not isinstance(mesh, Mapping):
        raise TypeError("mesh must be a hemisphere mapping.")

    data = _normalize_surface_mapping(data, "data")
    mesh = _normalize_surface_mapping(mesh, "mesh")
    missing_meshes = sorted(set(data) - set(mesh))
    if missing_meshes:
        raise ValueError(
            f"mesh is missing hemisphere entries for: {missing_meshes}"
        )

    surface_mesh = {
        hemisphere: _coerce_surface_mesh(mesh[hemisphere], nilearn_surface)
        for hemisphere in data
    }
    surface_data = {
        hemisphere: _coerce_surface_vector(
            data[hemisphere],
            nilearn_surface,
            name=f"data[{hemisphere}]",
        )
        for hemisphere in data
    }
    return surface_mesh, surface_data


def _coerce_background_maps(
    bg_maps,
    nilearn_surface,
) -> dict[str, np.ndarray]:
    if bg_maps is None:
        return {}
    if not isinstance(bg_maps, Mapping):
        raise TypeError("bg_maps must be a hemisphere mapping when provided.")

    bg_maps = _normalize_surface_mapping(bg_maps, "bg_maps")
    return {
        hemisphere: _coerce_surface_vector(
            bg_maps[hemisphere],
            nilearn_surface,
            name=f"bg_maps[{hemisphere}]",
            normalize=True,
        )
        for hemisphere in bg_maps
    }


def plot_surface_stat_map(
    data,
    output_file_prefix: str | Path,
    *,
    mesh=None,
    bg_maps: Mapping[str, object] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    threshold: float | None = None,
    cmap: str = "coolwarm",
    views: Sequence[str] = ("lateral", "medial"),
    hemispheres: Sequence[str] = ("left", "right"),
    figsize: tuple[float, float] = (8, 5),
    dpi: int = 100,
    zoom_scale: float = 0.61,
) -> Path:
    """
    Render an already-surface-based statistical map as a multi-panel PNG.

    :param data: Surface data to render. This can be either a nilearn
        `SurfaceImage`, or a hemisphere mapping like `{"L": data, "R": data}`
        where each value is a 1D array or a path accepted by
        `nilearn.surface.load_surf_data`.
    :param output_file_prefix: Prefix for the output PNG path.
    :param mesh: Surface mesh input used when `data` is not a `SurfaceImage`.
        Provide a hemisphere mapping whose values are mesh objects or paths
        accepted by `nilearn.surface.load_surf_mesh`.
    :param bg_maps: Optional hemisphere mapping of sulcal/background maps.
    :param vmin: Lower color limit.
    :param vmax: Upper color limit.
    :param threshold: Display threshold.
    :param cmap: Matplotlib colormap name.
    :param views: Surface views to render.
    :param hemispheres: Hemispheres to render.
    :param figsize: Figure size in inches.
    :param dpi: Output DPI.
    :param zoom_scale: 3D zoom factor applied to each panel.
    :return: Path to the written PNG file.
    """
    if len(views) == 0 or len(hemispheres) == 0:
        raise ValueError("views and hemispheres must be non-empty.")

    pyplot, nilearn_plotting, nilearn_surface = (
        _import_surface_plot_dependencies()
    )
    surface_mesh, surface_data = _coerce_surface_plot_inputs(
        data, mesh, nilearn_surface
    )
    background_maps = _coerce_background_maps(bg_maps, nilearn_surface)

    hemispheres_normalized = [
        _normalize_hemisphere_name(hemisphere) for hemisphere in hemispheres
    ]
    missing_hemispheres = sorted(
        set(hemispheres_normalized) - set(surface_data.keys())
    )
    if missing_hemispheres:
        raise ValueError(
            f"data does not contain requested hemispheres: {missing_hemispheres}"
        )

    fig, axes = pyplot.subplots(
        len(views),
        len(hemispheres_normalized),
        figsize=figsize,
        dpi=dpi,
        subplot_kw={"projection": "3d"},
        tight_layout=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1,
        wspace=0,
        hspace=0,
    )

    for column, hemisphere in enumerate(hemispheres_normalized):
        for row, view in enumerate(views):
            ax = axes[row, column]
            nilearn_plotting.plot_surf_stat_map(
                surf_mesh=surface_mesh[hemisphere],
                stat_map=surface_data[hemisphere],
                hemi=hemisphere,
                view=view,
                bg_on_data=hemisphere in background_maps,
                bg_map=background_maps.get(hemisphere),
                avg_method="mean",
                figure=fig,
                axes=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                threshold=threshold,
                colorbar=False,
            )
            zoom_3d(ax, scale=zoom_scale)
            ax.set_axis_off()
            for artist in ax.collections:
                artist.set_clip_on(False)

    output_path = _coerce_output_path(output_file_prefix, "")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, transparent=True)
    pyplot.close(fig)
    return output_path


def plot_surface_roi_map(
    data,
    output_file_prefix: str | Path,
    *,
    mesh=None,
    bg_maps: Mapping[str, object] | None = None,
    threshold: float | None = None,
    cmap: str = "gist_ncar",
    views: Sequence[str] = ("lateral", "medial"),
    hemispheres: Sequence[str] = ("left", "right"),
    figsize: tuple[float, float] = (8, 5),
    dpi: int = 100,
    zoom_scale: float = 0.61,
    colorbar: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """
    Render an already-surface-based ROI/parcellation map as a multi-panel PNG.

    This is intended for discrete label maps such as parcels or atlases.

    :param data: Surface ROI data to render. This can be either a nilearn
        `SurfaceImage`, or a hemisphere mapping like `{"L": data, "R": data}`
        where each value is a 1D array or a path accepted by
        `nilearn.surface.load_surf_data`.
    :param output_file_prefix: Prefix for the output PNG path.
    :param mesh: Surface mesh input used when `data` is not a `SurfaceImage`.
        Provide a hemisphere mapping whose values are mesh objects or paths
        accepted by `nilearn.surface.load_surf_mesh`.
    :param bg_maps: Optional hemisphere mapping of sulcal/background maps.
    :param threshold: Display threshold.
    :param cmap: Matplotlib colormap name.
    :param views: Surface views to render.
    :param hemispheres: Hemispheres to render.
    :param figsize: Figure size in inches.
    :param dpi: Output DPI.
    :param zoom_scale: 3D zoom factor applied to each panel.
    :param colorbar: Whether to draw a colorbar.
    :param vmin: Lower color limit.
    :param vmax: Upper color limit.
    :return: Path to the written PNG file.
    """
    if len(views) == 0 or len(hemispheres) == 0:
        raise ValueError("views and hemispheres must be non-empty.")

    pyplot, nilearn_plotting, nilearn_surface = (
        _import_surface_plot_dependencies()
    )
    surface_mesh, surface_data = _coerce_surface_plot_inputs(
        data, mesh, nilearn_surface
    )
    background_maps = _coerce_background_maps(bg_maps, nilearn_surface)

    hemispheres_normalized = [
        _normalize_hemisphere_name(hemisphere) for hemisphere in hemispheres
    ]
    missing_hemispheres = sorted(
        set(hemispheres_normalized) - set(surface_data.keys())
    )
    if missing_hemispheres:
        raise ValueError(
            f"data does not contain requested hemispheres: {missing_hemispheres}"
        )

    fig, axes = pyplot.subplots(
        len(views),
        len(hemispheres_normalized),
        figsize=figsize,
        dpi=dpi,
        subplot_kw={"projection": "3d"},
        tight_layout=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1,
        wspace=0,
        hspace=0,
    )

    for column, hemisphere in enumerate(hemispheres_normalized):
        for row, view in enumerate(views):
            ax = axes[row, column]
            nilearn_plotting.plot_surf_roi(
                surf_mesh=surface_mesh[hemisphere],
                roi_map=surface_data[hemisphere],
                hemi=hemisphere,
                view=view,
                bg_on_data=hemisphere in background_maps,
                bg_map=background_maps.get(hemisphere),
                avg_method="median",
                figure=fig,
                axes=ax,
                cmap=cmap,
                threshold=threshold,
                colorbar=colorbar,
                vmin=vmin,
                vmax=vmax,
            )
            zoom_3d(ax, scale=zoom_scale)
            ax.set_axis_off()
            for artist in ax.collections:
                artist.set_clip_on(False)

    output_path = _coerce_output_path(output_file_prefix, "")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, transparent=True)
    pyplot.close(fig)
    return output_path


__all__ = [
    "plot_mni152_surface_stat_map",
    "plot_mni152_surface_roi_map",
    "plot_surface_stat_map",
    "plot_surface_roi_map",
]
