from pathlib import Path

import pytest

import funROI.utils as utils_mod


def test_ensure_paths_coerces_positional_and_keyword_arguments():
    @utils_mod.ensure_paths("path1", "path2")
    def fn(path1, keep, path2=None):
        return path1, keep, path2

    path1, keep, path2 = fn("a/b", 7, path2="c/d")

    assert path1 == Path("a/b")
    assert keep == 7
    assert path2 == Path("c/d")


def test_validate_arguments_uses_defaults_and_rejects_invalid_values():
    @utils_mod.validate_arguments(mode={"a", "b"})
    def fn(mode="a"):
        return mode

    assert fn() == "a"
    assert fn("b") == "b"

    with pytest.raises(ValueError, match="Invalid mode"):
        fn("bad")


def test_get_orthogonalized_run_labels_covers_all_modes():
    assert utils_mod._get_orthogonalized_run_labels(
        ["01", "02"], group=1, orthogonalization="all-but-one"
    ) == ["02", "01"]
    assert utils_mod._get_orthogonalized_run_labels(
        ["01", "02"], group=2, orthogonalization="all-but-one"
    ) == ["01", "02"]

    assert utils_mod._get_orthogonalized_run_labels(
        ["01", "02", "03"], group=1, orthogonalization="all-but-one"
    ) == ["orth01", "orth02", "orth03"]
    assert utils_mod._get_orthogonalized_run_labels(
        ["01", "02", "03"], group=2, orthogonalization="all-but-one"
    ) == ["01", "02", "03"]

    assert utils_mod._get_orthogonalized_run_labels(
        ["01", "02"], group=1, orthogonalization="odd-even"
    ) == ["odd", "even"]
    assert utils_mod._get_orthogonalized_run_labels(
        ["01", "02"], group=2, orthogonalization="odd-even"
    ) == ["even", "odd"]


def test_get_orthogonalized_run_labels_rejects_invalid_arguments():
    with pytest.raises(ValueError, match="Invalid group"):
        utils_mod._get_orthogonalized_run_labels(
            ["01"], group=3, orthogonalization="all-but-one"
        )

    with pytest.raises(ValueError, match="Invalid orthogonalization"):
        utils_mod._get_orthogonalized_run_labels(
            ["01"], group=1, orthogonalization="bad-mode"
        )
