from pathlib import Path

import pandas as pd

import funROI._registry as registry_mod


def test_record_helpers_handle_paths_and_missing_values():
    criteria = {"path": Path("/tmp/demo"), "label": None}
    frame = registry_mod.build_record_frame(criteria)

    assert frame.loc[0, "path"] == "/tmp/demo"
    assert pd.isna(frame.loc[0, "label"])

    row = pd.Series({"path": "/tmp/demo", "label": float("nan")})
    assert registry_mod._row_matches(row, criteria)
    assert not registry_mod._row_matches(
        pd.Series({"path": "/tmp/other", "label": float("nan")}), criteria
    )


def test_get_or_create_record_id_handles_empty_registry_and_append(tmp_path):
    info_path = tmp_path / "registry.csv"
    pd.DataFrame(columns=["name", "id"]).to_csv(info_path, index=False)

    assert (
        registry_mod.get_or_create_record_id(
            info_path, {"name": "cfg0"}, create=False
        )
        == 0
    )
    assert (
        registry_mod.get_or_create_record_id(
            info_path, {"name": "cfg0"}, create=True
        )
        == 0
    )
    assert (
        registry_mod.get_or_create_record_id(
            info_path, {"name": "cfg0"}, create=True
        )
        == 0
    )
    assert (
        registry_mod.get_or_create_record_id(
            info_path, {"name": "cfg1"}, create=True
        )
        == 1
    )


def test_get_or_create_record_id_matches_float_values_after_csv_roundtrip(
    tmp_path,
):
    info_path = tmp_path / "registry.csv"
    pd.DataFrame(
        {
            "threshold": [0.1999999999999999],
            "parcels": ["/tmp/demo"],
            "id": [7],
        }
    ).to_csv(info_path, index=False)

    record_id = registry_mod.get_or_create_record_id(
        info_path,
        {
            "threshold": 0.19999999999999998,
            "parcels": Path("/tmp/demo"),
        },
        create=False,
    )

    assert record_id == 7

    record_id = registry_mod.get_or_create_record_id(
        info_path,
        {
            "threshold": 0.19999999999999998,
            "parcels": Path("/tmp/demo"),
        },
        create=True,
    )

    assert record_id == 7
    written = pd.read_csv(info_path)
    assert written.shape[0] == 1


def test_append_record_and_get_record_folder(tmp_path):
    info_path = tmp_path / "rows.csv"

    assert registry_mod.append_record(
        info_path, pd.DataFrame({"name": ["first"]})
    ) == 0
    assert registry_mod.append_record(
        info_path, pd.DataFrame({"name": ["second"]})
    ) == 1

    written = pd.read_csv(info_path)
    assert written["id"].tolist() == [0, 1]
    assert written["name"].tolist() == ["first", "second"]

    assert registry_mod.get_record_folder(tmp_path, "run", 7) == (
        tmp_path / "run_0007"
    )
