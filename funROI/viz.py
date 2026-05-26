from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Sequence

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


__all__ = ["plot_mni152_surface_stat_map"]
