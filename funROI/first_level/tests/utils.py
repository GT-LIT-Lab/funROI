from dataclasses import dataclass, field
from typing import List
import numpy as np
from pathlib import Path

@dataclass
class FakeImg:
    data: np.ndarray = None
    written: List[Path] = field(default_factory=list)

    @property
    def shape(self):
        return self.data.shape

    def get_fdata(self):
        return self.data

    def to_filename(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-nifti")
        self.written.append(path)

