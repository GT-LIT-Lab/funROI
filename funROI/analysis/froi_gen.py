from typing import List, Optional, Union, Tuple
from .._registry import get_or_create_record_id
from .._surface import (
    SURFACE_HEMIS,
    flatten_image_data,
    is_surface_image,
    load_surface_image,
    save_surface_image,
    surface_image_from_flat,
)
from ..froi import (
    FROIConfig,
    _build_froi_registry_record,
    _create_froi,
    _get_froi_path,
    _get_surface_froi_paths,
    _get_surface_mesh_paths_for_froi,
)
from ..parcels import get_parcels
from ..settings import get_analysis_output_folder
from nibabel.nifti1 import Nifti1Image
from nilearn.image import load_img
import warnings
from pathlib import Path


class FROIGenerator:
    """
    Generate fROI maps for a fROI configuration.

    .. warning:: The contrasts and parcels images are assumed to be in the same
        space and have the same dimensions. Future versions may support
        generating fROIs to native space, etc.

    :param subjects: List of subject labels.
    :type subjects: List[str]
    :param froi: fROI configuration.
    :type froi: FROIConfig
    :param run_label: Label of the run to generate the fROIs for. Default is
        "all", where a contrast map for all runs is used to generate the fROI
        for each subject. Alternatively, this can be a specific run label,
        'odd' for odd runs, 'even' for even runs, 'orth<run_label>' for
        all-but-one runs.
    :type run_label: Optional[str]
    """

    def __init__(
        self,
        subjects: List[str],
        froi: FROIConfig,
        run_label: Optional[str] = "all",
    ):
        self.subjects = subjects
        self.froi = froi
        self.run_label = run_label
        self._data = []

    def run(
        self, save: Optional[bool] = True
    ) -> Optional[List[Tuple[str, object]]]:
        """
        Run the fROI generation. The results are stored in the analysis output
        folder.

        :param save: Whether to save the results to the analysis output folder.
        :type save: Optional[bool]
        :return: the results are returned as a list of tuples, where each tuple
            contains the subject label and the FROI map.
        :rtype: Optional[List[Tuple[str, Nifti1Image]]]
        """
        data = []
        for subject in self.subjects:
            froi_img = self._load_subject_froi(subject)
            if froi_img is None:
                warnings.warn(f"Error generating fROI for subject {subject}")
                continue

            if save:
                if is_surface_image(froi_img):
                    froi_paths = self._get_analysis_surface_froi_paths(
                        subject, self.run_label, self.froi, create=True
                    )
                    save_surface_image(froi_img, froi_paths)
                else:
                    froi_pth = self._get_analysis_froi_path(
                        subject, self.run_label, self.froi, create=True
                    )
                    froi_img.to_filename(froi_pth)

            data.append((subject, froi_img))

        self.subjects = [dat[0] for dat in data]
        self._data = [dat[1] for dat in data]

        return data

    def select(
        self,
        froi_label: Union[int, str],
        return_results: Optional[bool] = False,
    ) -> Optional[List[Tuple[str, object]]]:
        """
        Select a specific fROI label on the maps. The selected fROI label is
        kept, while all other labels are set to zero. The results are stored in
        the analysis output folder.

        :return: If return_results is True, the results are also returned as a
            list of tuples, where each tuple contains the subject label and the
            filtered FROI map.
        :rtype: Optional[List[Tuple[str, Nifti1Image]]]

        :raises ValueError: If the fROI label is not found in the fROI.
        """
        parcels_img, parcels_labels = get_parcels(self.froi.parcels)
        label_numeric = None
        label_str = None
        for k, v in parcels_labels.items():
            if isinstance(froi_label, int) and k == froi_label:
                label_numeric = k
                label_str = v
                break
            elif isinstance(froi_label, str) and v == froi_label:
                label_numeric = k
                label_str = v
                break
        if label_numeric is None:
            raise ValueError(f"Label {froi_label} not found in parcels labels")

        results = []
        for subject, img in zip(self.subjects, self._data):
            if is_surface_image(img):
                img_data = flatten_image_data(img)
                img_data[img_data != label_numeric] = 0
                img = surface_image_from_flat(img_data, img)
            else:
                img_data = img.get_fdata()
                img_data[img_data != label_numeric] = 0
                img = Nifti1Image(img_data, img.affine)
            results.append((subject, img))

            if is_surface_image(img):
                froi_paths = self._get_analysis_surface_froi_paths(
                    subject,
                    self.run_label,
                    self.froi,
                    create=True,
                    froi_label=label_str,
                )
                save_surface_image(img, froi_paths)
            else:
                froi_pth = self._get_analysis_froi_path(
                    subject,
                    self.run_label,
                    self.froi,
                    create=True,
                    froi_label=label_str,
                )
                img.to_filename(froi_pth)

        if return_results:
            return results

    def _load_subject_froi(self, subject: str):
        froi_path = _get_froi_path(subject, self.run_label, self.froi)
        if froi_path.exists():
            return load_img(froi_path)

        try:
            surface_paths = _get_surface_froi_paths(
                subject, self.run_label, self.froi
            )
        except RuntimeError:
            surface_paths = None
        if surface_paths is not None and all(
            path.exists() for path in surface_paths.values()
        ):
            mesh_paths = _get_surface_mesh_paths_for_froi(subject, self.froi)
            return load_surface_image(surface_paths, mesh_paths)

        _create_froi(subject, self.froi, self.run_label)

        if froi_path.exists():
            return load_img(froi_path)
        if surface_paths is not None and all(
            path.exists() for path in surface_paths.values()
        ):
            mesh_paths = _get_surface_mesh_paths_for_froi(subject, self.froi)
            return load_surface_image(surface_paths, mesh_paths)
        return None

    @staticmethod
    def _get_analysis_froi_folder(task: str) -> Path:
        return get_analysis_output_folder() / f"froi_{task}"

    @classmethod
    def _get_analysis_froi_info_path(cls, task: str) -> Path:
        return cls._get_analysis_froi_folder(task) / "froi_info.csv"

    @classmethod
    def _get_analysis_froi_path(
        cls,
        subject: str,
        run_label: str,
        config: FROIConfig,
        create: Optional[bool] = False,
        froi_label: Optional[str] = None,
    ) -> Path:
        task = config.task
        record_id = get_or_create_record_id(
            cls._get_analysis_froi_info_path(task),
            _build_froi_registry_record(config),
            create=create,
        )
        id = f"{int(record_id):04d}"
        froi_folder = cls._get_analysis_froi_folder(task) / f"froi_{id}"
        froi_folder.mkdir(parents=True, exist_ok=True)

        if froi_label is None:
            return froi_folder / f"sub-{subject}_run-{run_label}_froi.nii.gz"
        else:
            return (
                froi_folder
                / f"sub-{subject}_run-{run_label}_label-{froi_label}.nii.gz"
            )

    @classmethod
    def _get_analysis_surface_froi_paths(
        cls,
        subject: str,
        run_label: str,
        config: FROIConfig,
        create: Optional[bool] = False,
        froi_label: Optional[str] = None,
    ) -> dict:
        task = config.task
        record_id = get_or_create_record_id(
            cls._get_analysis_froi_info_path(task),
            _build_froi_registry_record(config),
            create=create,
        )
        record_label = f"{int(record_id):04d}"
        froi_folder = cls._get_analysis_froi_folder(task) / f"froi_{record_label}"
        froi_folder.mkdir(parents=True, exist_ok=True)

        stem = f"sub-{subject}_run-{run_label}"
        if froi_label is None:
            stem = f"{stem}_froi"
        else:
            stem = f"{stem}_label-{froi_label}"
        return {
            hemi: froi_folder / f"{stem}_hemi-{hemi}.func.gii"
            for hemi in SURFACE_HEMIS
        }
