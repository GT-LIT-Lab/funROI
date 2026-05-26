from pathlib import Path
from typing import Any, Mapping
import json

import pandas as pd


def _normalize_record_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return json.dumps(
            {
                str(key): _normalize_record_value(val)
                for key, val in value.items()
            },
            sort_keys=True,
        )
    if isinstance(value, (list, tuple)):
        return json.dumps(
            [_normalize_record_value(item) for item in value]
        )
    return value


def _row_matches(row: pd.Series, criteria: Mapping[str, Any]) -> bool:
    for column, expected in criteria.items():
        expected = _normalize_record_value(expected)
        actual = row[column]
        if expected is None:
            if not pd.isna(actual):
                return False
        elif actual != expected:
            return False
    return True


def build_record_frame(criteria: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [{key: _normalize_record_value(value) for key, value in criteria.items()}]
    )


def get_or_create_record_id(
    info_path: Path, criteria: Mapping[str, Any], create: bool = False
) -> int:
    record = build_record_frame(criteria)
    if not info_path.exists():
        record_id = 0
        if create:
            info_path.parent.mkdir(parents=True, exist_ok=True)
            record.assign(id=record_id).to_csv(info_path, index=False)
        return record_id

    info = pd.read_csv(info_path)
    matches = info.apply(lambda row: _row_matches(row, criteria), axis=1)
    if matches.any():
        return int(info.loc[matches, "id"].iloc[0])

    record_id = 0 if info.empty else int(info["id"].max()) + 1
    if create:
        info = pd.concat(
            [info, record.assign(id=record_id)], ignore_index=True
        )
        info.to_csv(info_path, index=False)
    return record_id


def append_record(info_path: Path, new_row: pd.DataFrame) -> int:
    info_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = new_row.copy()

    if not info_path.exists():
        record_id = 0
        info = new_row.assign(id=record_id)
    else:
        info = pd.read_csv(info_path)
        record_id = 0 if info.empty else int(info["id"].max()) + 1
        info = pd.concat(
            [info, new_row.assign(id=record_id)], ignore_index=True
        )

    info.to_csv(info_path, index=False)
    return record_id


def get_record_folder(root: Path, prefix: str, record_id: int) -> Path:
    return root / f"{prefix}_{record_id:04d}"
