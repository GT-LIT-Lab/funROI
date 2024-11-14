from typing import List
from ..contrast import (
    _get_contrast_info_path,
)
import pandas as pd


def _register_contrast(
    subject: str, task: str, contrast_name: str, contrast_vector: List[float]
):
    contrast_vector = [float(v) for v in contrast_vector]
    contrast_info_path = _get_contrast_info_path(subject, task)
    if not contrast_info_path.exists():
        contrast_info = pd.DataFrame(columns=["contrast", "vector"])
    else:
        contrast_info = pd.read_csv(contrast_info_path)

    contrast_info = pd.concat(
        [
            contrast_info,
            pd.DataFrame(
                {"contrast": [contrast_name], "vector": [contrast_vector]}
            ),
        ],
        ignore_index=True,
    )

    contrast_info.to_csv(contrast_info_path, index=False)
