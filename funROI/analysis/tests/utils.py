from typing import Sequence
import numpy as np
from nibabel.nifti1 import Nifti1Image

class DummyParcels:
    def __init__(self, parcels_path=None, labels_path=None):
        self.parcels_path = parcels_path
        self.labels_path = labels_path

class DummyFROI:
    """Minimal duck-typed FROIConfig for tests.

    Matches the minimal attributes accessed by the analysis tests:
    - task
    - contrasts
    - parcels

    Instances of this class can be used wherever the tests need an
    object that behaves like a real FROI configuration.
    """

    def __init__(self, task: str = "LOC", contrasts: Sequence[str] | None = None, parcels: str = "dummy_parcels"):
        self.task = task
        self.contrasts = list(contrasts) if contrasts is not None else ["c1"]
        self.parcels = parcels

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"DummyFROI(task={self.task}, contrasts={self.contrasts}, parcels={self.parcels})"



class DummyFROIConfig:
    """Minimal FROIConfig-like object: only attributes accessed by froi_gen."""
    def __init__(
        self,
        task="TASK",
        contrasts=None,
        conjunction_type=None,
        threshold_type="none",
        threshold_value=0.0,
        parcels="dummy_parcels",
    ):
        self.task = task
        self.contrasts = contrasts if contrasts is not None else ["c1"]
        self.conjunction_type = conjunction_type
        self.threshold_type = threshold_type
        self.threshold_value = threshold_value
        self.parcels = parcels

def _img_from_flat(flat: np.ndarray) -> Nifti1Image:
    """Create a tiny 3D Nifti1Image from a flat array.

    The tests use small flat arrays and only care that the returned
    object is a nibabel Nifti1Image whose data can be flattened.
    """
    data = np.asarray(flat, dtype=np.float32).reshape((1, 1, -1))
    return Nifti1Image(data, np.eye(4))

