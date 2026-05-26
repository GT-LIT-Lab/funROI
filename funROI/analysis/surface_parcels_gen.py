from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import warnings

import numpy as np
import pandas as pd
from nilearn.surface import SurfaceImage, load_surf_mesh

from .._surface import (
    SURFACE_HEMIS,
    SURFACE_PARTS,
    flatten_image_data,
    load_surface_numeric_data,
    save_surface_image,
)
from ..contrast import _get_contrast_runs, _get_surface_contrast_path
from ..first_level.nilearn import _find_surface_mesh_paths, _smooth_surface_array
from ..froi import _create_p_map_mask
from ..settings import (
    get_analysis_output_folder,
    get_bids_data_folder,
    get_bids_preprocessed_folder,
    get_bids_preprocessed_folder_relative,
)
from ..utils import validate_arguments


class SurfaceParcelsGenerator:
    """Generate group surface parcels from first-level surface contrast maps."""

    def __init__(
        self,
        parcels_name: str,
        space: str = "fsLR32k",
        smoothing_kernel_size: Optional[Union[float, List[float]]] = 8,
        overlap_thr_vox: Optional[float] = 0.1,
        min_voxel_size: Optional[int] = 0,
        overlap_thr_roi: Optional[float] = 0,
    ):
        self.parcels_name = parcels_name
        self.space = space
        self.smoothing_kernel_size = smoothing_kernel_size
        self.overlap_thr_vox = overlap_thr_vox
        self.min_voxel_size = min_voxel_size
        self.overlap_thr_roi = overlap_thr_roi

        self.configs = []
        self._data = []
        self.mesh = None
        self.mesh_paths = None
        self.hemi_sizes = None
        self.hemi_slices = None
        self.overlap_map = None
        self.parcels = None
        self.parcel_info = None

    @validate_arguments(
        p_threshold_type={"none", "bonferroni", "fdr", "n", "percent"},
        conjunction_type={"min", "max", "sum", "prod", "and", "or"},
    )
    def add_subjects(
        self,
        subjects: List[str],
        task: str,
        contrasts: List[str],
        p_threshold_type: str,
        p_threshold_value: float = 0.05,
        conjunction_type: Optional[str] = "and",
    ):
        existing_subjects = []
        for config in self.configs:
            existing_subjects.extend(config["subjects"])

        subjects_redundant = set(subjects).intersection(existing_subjects)
        if subjects_redundant:
            raise ValueError(
                f"Subjects {subjects_redundant} are already added."
            )

        config = {
            "task": task,
            "contrasts": contrasts,
            "threshold_type": p_threshold_type,
            "threshold_value": p_threshold_value,
            "conjunction_type": conjunction_type,
            "space": self.space,
        }

        new_data = []
        for subject in subjects:
            subject_runs = self._get_subject_runs(subject, task, contrasts)
            subject_mesh, subject_mesh_paths = self._get_subject_meshes(subject)
            self._set_or_validate_mesh(subject_mesh, subject_mesh_paths)

            subject_data = []
            for run_label in self._get_orthogonalized_labels(subject_runs):
                run_p_maps = []
                for contrast in contrasts:
                    hemi_data = self._get_surface_contrast_data(
                        subject, task, run_label, contrast, "p"
                    )
                    run_p_maps.append(
                        np.concatenate(
                            [hemi_data["L"], hemi_data["R"]], axis=0
                        )
                    )

                data = np.asarray(run_p_maps, dtype=float)[:, None, :]
                mask = _create_p_map_mask(
                    data,
                    conjunction_type,
                    p_threshold_type,
                    p_threshold_value,
                )
                subject_data.append(np.asarray(mask).reshape(-1).astype(float))

            if len(subject_data) == 0:
                raise ValueError(
                    f"No data found for subject {subject} and task {task}."
                )
            new_data.append(np.asarray(subject_data, dtype=float))

        self._data.extend(new_data)
        self.configs.append({"subjects": subjects, "surface": config})

    def run(self) -> SurfaceImage:
        if self.mesh is None:
            raise RuntimeError("No surface mesh loaded. Add subjects first.")

        if self.parcel_info is None:
            binary_masks = [np.mean(dat, axis=0) > 0.5 for dat in self._data]
            self.overlap_map, self.parcels = self._run(
                binary_masks,
                self.mesh,
                self.hemi_slices,
                self.smoothing_kernel_size,
                self.overlap_thr_vox,
            )

            parcel_info_data = []
            for parcel in np.unique(self.parcels):
                if parcel == 0:
                    continue
                parcel_mask = self.parcels == parcel
                parcel_size = int(np.sum(parcel_mask))
                subject_coverage = np.zeros(len(self._data))
                for subject_i, data in enumerate(self._data):
                    subject_coverage[subject_i] = (
                        self._harmonic_mean(
                            np.sum(data[:, parcel_mask], axis=1)
                        )
                        > 0
                    )
                parcel_info_data.append(
                    [int(parcel), parcel_size, float(np.mean(subject_coverage))]
                )
            self.parcel_info = pd.DataFrame(
                parcel_info_data, columns=["id", "size", "roi_overlap"]
            )
            self._save()

        if self.min_voxel_size != 0 or self.overlap_thr_roi != 0:
            self.parcels = self._filter(
                self.parcels,
                self.parcel_info,
                self.overlap_thr_roi,
                self.min_voxel_size,
            )
            self._save()

        return self._to_surface_image(self.parcels)

    def filter(
        self,
        overlap_thr_roi: Optional[float] = 0,
        min_voxel_size: Optional[int] = 0,
    ) -> SurfaceImage:
        if self.parcels is None:
            raise RuntimeError(
                "No parcels to filter. Run the parcels generation first."
            )

        if overlap_thr_roi != 0 and overlap_thr_roi <= self.overlap_thr_roi:
            warnings.warn(
                "The new overlap_thr_roi is lower than the current setup. "
                "The filtering will not be applied."
            )
            overlap_thr_roi = 0.0
        if min_voxel_size != 0 and min_voxel_size <= self.min_voxel_size:
            warnings.warn(
                "The new min_voxel_size is lower than the current setup. "
                "The filtering will not be applied."
            )
            min_voxel_size = 0

        if overlap_thr_roi != 0 or min_voxel_size != 0:
            self.parcels = self._filter(
                self.parcels,
                self.parcel_info,
                overlap_thr_roi,
                min_voxel_size,
            )
            if overlap_thr_roi != 0:
                self.overlap_thr_roi = overlap_thr_roi
            if min_voxel_size != 0:
                self.min_voxel_size = min_voxel_size
            self._save()

        return self._to_surface_image(self.parcels)

    def _get_subject_runs(
        self, subject: str, task: str, contrasts: List[str]
    ) -> List[str]:
        runs = None
        for contrast in contrasts:
            runs_i = _get_contrast_runs(subject, task, contrast)
            if runs is None:
                runs = runs_i
            else:
                runs = list(set(runs) & set(runs_i))
        runs = sorted(runs or [])
        if len(runs) == 0:
            raise ValueError(
                f"No surface contrast runs found for subject {subject} and "
                f"task {task}."
            )
        return runs

    @staticmethod
    def _get_orthogonalized_labels(runs: List[str]) -> List[str]:
        labels = []
        for run in runs:
            if len(runs) == 2:
                runs_ = runs.copy()
                runs_.remove(run)
                labels.append(runs_[0])
            else:
                labels.append(f"orth{run}")
        return labels

    @staticmethod
    def _get_derivatives_root() -> Path:
        try:
            bids_data_folder = Path(get_bids_data_folder())
            derivatives_folder = get_bids_preprocessed_folder_relative()
            if derivatives_folder == ".":
                return bids_data_folder
            return bids_data_folder / derivatives_folder
        except (ValueError, RuntimeError):
            return Path(get_bids_preprocessed_folder())

    def _get_subject_meshes(
        self, subject: str
    ) -> Tuple[Dict[str, object], Dict[str, Path]]:
        derivatives_root = self._get_derivatives_root()
        mesh_paths = _find_surface_mesh_paths(
            derivatives_root, subject, self.space
        )
        if mesh_paths is None:
            raise FileNotFoundError(
                f"Could not find surface meshes for subject {subject} in "
                f"space '{self.space}'."
            )
        return (
            {
                "left": load_surf_mesh(mesh_paths["L"]),
                "right": load_surf_mesh(mesh_paths["R"]),
            },
            mesh_paths,
        )

    def _set_or_validate_mesh(
        self, subject_mesh: Dict[str, object], mesh_paths: Dict[str, Path]
    ) -> None:
        hemi_sizes = {
            hemi: int(len(np.asarray(subject_mesh[hemi].coordinates)))
            for hemi in ["left", "right"]
        }

        if self.mesh is None:
            self.mesh = subject_mesh
            self.mesh_paths = mesh_paths
            self.hemi_sizes = hemi_sizes
            left_size = hemi_sizes["left"]
            self.hemi_slices = {
                "left": slice(0, left_size),
                "right": slice(left_size, left_size + hemi_sizes["right"]),
            }
            return

        for hemi in ["left", "right"]:
            reference_mesh = self.mesh[hemi]
            candidate_mesh = subject_mesh[hemi]
            if len(reference_mesh.coordinates) != len(candidate_mesh.coordinates):
                raise ValueError(
                    "All surface meshes must have the same number of vertices. "
                    f"Mismatch found in hemisphere {hemi}."
                )
            if not np.array_equal(reference_mesh.faces, candidate_mesh.faces):
                raise ValueError(
                    "All surface meshes must share the same topology. "
                    f"Mismatch found in hemisphere {hemi}."
                )

    def _get_surface_contrast_data(
        self,
        subject: str,
        task: str,
        run_label: str,
        contrast: str,
        image_type: str,
    ) -> Dict[str, np.ndarray]:
        hemi_data = {}
        for hemi in SURFACE_HEMIS:
            path = _get_surface_contrast_path(
                subject, task, run_label, contrast, image_type, hemi
            )
            if not path.exists():
                raise FileNotFoundError(
                    f"Surface contrast file not found: {path}"
                )
            hemi_data[hemi] = load_surface_numeric_data(path).reshape(-1)
        if image_type == "p":
            for hemi in SURFACE_HEMIS:
                hemi_data[hemi] = hemi_data[hemi].astype(float, copy=True)
                hemi_data[hemi][hemi_data[hemi] == 0] = np.nan

        expected_sizes = {
            "L": self.hemi_sizes["left"] if self.hemi_sizes else None,
            "R": self.hemi_sizes["right"] if self.hemi_sizes else None,
        }
        for hemi in SURFACE_HEMIS:
            expected_size = expected_sizes[hemi]
            if expected_size is not None and len(hemi_data[hemi]) != expected_size:
                raise ValueError(
                    "All surface contrast maps must match the shared mesh "
                    f"vertex count. Hemisphere {hemi} does not match."
                )
        return hemi_data

    @classmethod
    def _run(
        cls,
        binary_masks: List[np.ndarray],
        mesh: Dict[str, object],
        hemi_slices: Dict[str, slice],
        smoothing_kernel_size: Union[float, List[float]],
        overlap_thr_vox: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        overlap_map = np.mean(binary_masks, axis=0)
        smoothed_map = overlap_map.copy().astype(float)

        fwhm = smoothing_kernel_size
        if isinstance(fwhm, (list, tuple)):
            if len(fwhm) != 1:
                raise ValueError(
                    "Surface smoothing expects a scalar kernel size."
                )
            fwhm = fwhm[0]

        for hemi in ["left", "right"]:
            hemi_slice = hemi_slices[hemi]
            smoothed_map[hemi_slice] = _smooth_surface_array(
                smoothed_map[hemi_slice],
                mesh[hemi],
                fwhm,
            )

        smoothed_map[smoothed_map < overlap_thr_vox] = np.nan

        parcels = np.zeros_like(overlap_map, dtype=int)
        label_offset = 0
        for hemi in ["left", "right"]:
            hemi_slice = hemi_slices[hemi]
            hemi_labels = cls._watershed_surface(
                smoothed_map[hemi_slice],
                mesh[hemi],
            )
            nonzero = hemi_labels > 0
            hemi_labels = hemi_labels.astype(int)
            hemi_labels[nonzero] += label_offset
            parcels[hemi_slice] = hemi_labels
            label_offset = int(np.max(parcels))
        return overlap_map, parcels

    @classmethod
    def _watershed_surface(cls, values: np.ndarray, mesh) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1)
        adjacency = cls._mesh_adjacency(mesh)
        labels = np.zeros(values.shape[0], dtype=int)

        valid = np.flatnonzero(~np.isnan(values))
        if valid.size == 0:
            return labels

        order = valid[np.argsort(values[valid], kind="stable")[::-1]]
        next_label = 1
        for vertex in order:
            neighbor_labels = np.unique(
                labels[adjacency[vertex]][labels[adjacency[vertex]] > 0]
            )
            if neighbor_labels.size == 0:
                labels[vertex] = next_label
                next_label += 1
            elif neighbor_labels.size == 1:
                labels[vertex] = int(neighbor_labels[0])
        return labels

    @staticmethod
    def _mesh_adjacency(mesh) -> List[np.ndarray]:
        n_vertices = len(mesh.coordinates)
        neighbors = [set() for _ in range(n_vertices)]
        for face in np.asarray(mesh.faces, dtype=int):
            a, b, c = face.tolist()
            neighbors[a].update([b, c])
            neighbors[b].update([a, c])
            neighbors[c].update([a, b])
        return [
            np.asarray(sorted(vertex_neighbors), dtype=int)
            for vertex_neighbors in neighbors
        ]

    @classmethod
    def _filter(
        cls,
        parcels: np.ndarray,
        parcel_info: pd.DataFrame,
        overlap_thr_roi: float,
        min_voxel_size: int,
    ) -> np.ndarray:
        filtered_parcels = parcels.copy()
        unique_parcels = np.unique(parcels)
        for parcel in unique_parcels:
            if parcel == 0:
                continue
            parcel_mask = parcels == parcel
            if (
                parcel_info.loc[
                    parcel_info["id"] == parcel, "roi_overlap"
                ].values[0]
                < overlap_thr_roi
            ):
                filtered_parcels[parcel_mask] = 0
            if (
                parcel_info.loc[parcel_info["id"] == parcel, "size"].values[0]
                < min_voxel_size
            ):
                filtered_parcels[parcel_mask] = 0
        return filtered_parcels

    def _to_surface_image(self, data: np.ndarray) -> SurfaceImage:
        return SurfaceImage(
            mesh=self.mesh,
            data={
                "left": np.asarray(data[self.hemi_slices["left"]]),
                "right": np.asarray(data[self.hemi_slices["right"]]),
            },
        )

    @staticmethod
    def _harmonic_mean(data: np.ndarray) -> float:
        data = np.asarray(data).flatten().astype(float)
        data = data[~np.isnan(data)]
        if data.size == 0:
            return 0.0
        if np.any(data <= 0):
            return 0.0
        return data.size / np.sum(1.0 / data)

    @staticmethod
    def _get_analysis_parcels_folder(parcels_name: str) -> Path:
        return get_analysis_output_folder() / "parcels" / f"parcels-{parcels_name}"

    def _output_stem(self) -> str:
        return (
            f"parcels-{self.parcels_name}_space-{self.space}"
            f"_sm-{self.smoothing_kernel_size}"
            f"_voxthres-{self.overlap_thr_vox}"
            f"_roithres-{self.overlap_thr_roi}"
            f"_sz-{self.min_voxel_size}"
        )

    def _save(self) -> None:
        base = self._get_analysis_parcels_folder(self.parcels_name)
        base.mkdir(parents=True, exist_ok=True)

        config_path = base / f"parcels-{self.parcels_name}_config.json"
        if not config_path.exists():
            with open(config_path, "w") as f:
                json.dump(
                    {
                        "configs": self.configs,
                        "space": self.space,
                        "mesh_paths": {
                            hemi: str(path)
                            for hemi, path in (self.mesh_paths or {}).items()
                        },
                    },
                    f,
                )

        overlap_stem = (
            f"parcels-{self.parcels_name}_space-{self.space}_overlap"
        )
        if self.overlap_map is not None:
            save_surface_image(
                self._to_surface_image(self.overlap_map),
                {
                    hemi: base / f"{overlap_stem}_hemi-{hemi}.func.gii"
                    for hemi in SURFACE_HEMIS
                },
            )

        parcel_info_path = (
            base
            / (
                f"parcels-{self.parcels_name}_space-{self.space}"
                f"_sm-{self.smoothing_kernel_size}"
                f"_voxthres-{self.overlap_thr_vox}_info.csv"
            )
        )
        self.parcel_info.to_csv(parcel_info_path, index=False)

        stem = self._output_stem()
        save_surface_image(
            self._to_surface_image(self.parcels),
            {
                hemi: base / f"{stem}_hemi-{hemi}.func.gii"
                for hemi in SURFACE_HEMIS
            },
        )
