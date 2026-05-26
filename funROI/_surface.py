from pathlib import Path
from typing import Dict, Mapping, Union

import nibabel as nib
import numpy as np
from nilearn.surface import SurfaceImage, load_surf_data, load_surf_mesh


SURFACE_HEMIS = ("L", "R")
SURFACE_PARTS = {"L": "left", "R": "right"}


def _as_path_dict(
    paths: Mapping[str, Union[str, Path]]
) -> Dict[str, Path]:
    return {hemi: Path(paths[hemi]) for hemi in SURFACE_HEMIS}


def is_surface_image(img) -> bool:
    return isinstance(img, SurfaceImage)


def write_gifti(path: Union[str, Path], data: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        arrays = [data]
    else:
        arrays = [data[:, i] for i in range(data.shape[1])]
    img = nib.gifti.GiftiImage(
        darrays=[nib.gifti.GiftiDataArray(data=array) for array in arrays]
    )
    nib.save(img, path)


def load_surface_numeric_data(path: Union[str, Path]) -> np.ndarray:
    return np.asarray(load_surf_data(path), dtype=np.float32)


def load_surface_image(
    data_paths: Mapping[str, Union[str, Path]],
    mesh_paths: Mapping[str, Union[str, Path]],
) -> SurfaceImage:
    data_paths = _as_path_dict(data_paths)
    mesh_paths = _as_path_dict(mesh_paths)
    return SurfaceImage(
        mesh={
            SURFACE_PARTS[hemi]: load_surf_mesh(mesh_paths[hemi])
            for hemi in SURFACE_HEMIS
        },
        data={
            SURFACE_PARTS[hemi]: load_surface_numeric_data(data_paths[hemi])
            for hemi in SURFACE_HEMIS
        },
    )


def get_surface_data_parts(img: SurfaceImage) -> Dict[str, np.ndarray]:
    return {
        hemi: np.asarray(img.data.parts[SURFACE_PARTS[hemi]])
        for hemi in SURFACE_HEMIS
    }


def flatten_surface_parts(parts: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(parts[hemi]).reshape(-1) for hemi in SURFACE_HEMIS]
    )


def flatten_image_data(img) -> np.ndarray:
    if is_surface_image(img):
        return flatten_surface_parts(get_surface_data_parts(img))
    return np.asarray(img.get_fdata()).reshape(-1)


def surface_hemi_sizes(img: SurfaceImage) -> Dict[str, int]:
    parts = get_surface_data_parts(img)
    return {hemi: int(parts[hemi].shape[0]) for hemi in SURFACE_HEMIS}


def surface_flat_to_parts(
    flat_data: np.ndarray, reference_img: SurfaceImage
) -> Dict[str, np.ndarray]:
    flat_data = np.asarray(flat_data).reshape(-1)
    sizes = surface_hemi_sizes(reference_img)
    parts = {}
    start = 0
    for hemi in SURFACE_HEMIS:
        stop = start + sizes[hemi]
        parts[hemi] = flat_data[start:stop].astype(np.float32, copy=False)
        start = stop
    if start != flat_data.size:
        raise ValueError(
            "Surface data length does not match the reference mesh size."
        )
    return parts


def surface_image_from_flat(
    flat_data: np.ndarray, reference_img: SurfaceImage
) -> SurfaceImage:
    parts = surface_flat_to_parts(flat_data, reference_img)
    return SurfaceImage(
        mesh=reference_img.mesh,
        data={
            SURFACE_PARTS[hemi]: parts[hemi] for hemi in SURFACE_HEMIS
        },
    )


def save_surface_image(
    img: SurfaceImage, data_paths: Mapping[str, Union[str, Path]]
) -> None:
    data_paths = _as_path_dict(data_paths)
    parts = get_surface_data_parts(img)
    for hemi in SURFACE_HEMIS:
        write_gifti(data_paths[hemi], parts[hemi])
