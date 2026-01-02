import pandas as pd
import funROI.first_level.utils as u


def test_register_contrast_creates_file_when_missing(tmp_path, monkeypatch):
    out = tmp_path / "contrast_info.csv"
    monkeypatch.setattr(u, "_get_contrast_info_path", lambda sub, task: out)

    u._register_contrast(
        subject="100307",
        task="LANGUAGE",
        contrast_name="math_gt_story",
        contrast_vector=[1, 0, -1],  # ints should be cast to float
    )

    assert out.exists()
    df = pd.read_csv(out)

    assert list(df.columns) == ["contrast", "vector"]
    assert df.shape[0] == 1
    assert df.loc[0, "contrast"] == "math_gt_story"

    vec_str = df.loc[0, "vector"]
    assert "1.0" in vec_str and "-1.0" in vec_str


def test_register_contrast_appends_when_file_exists(tmp_path, monkeypatch):
    out = tmp_path / "contrast_info.csv"
    monkeypatch.setattr(u, "_get_contrast_info_path", lambda sub, task: out)

    pd.DataFrame(
        {"contrast": ["old_con"], "vector": [[0.0, 1.0]]}
    ).to_csv(out, index=False)

    u._register_contrast(
        subject="100307",
        task="LANGUAGE",
        contrast_name="new_con",
        contrast_vector=[2, 3],  # ints -> floats
    )

    df = pd.read_csv(out)
    assert df.shape[0] == 2
    assert df["contrast"].tolist() == ["old_con", "new_con"]

    vec_str = df.loc[1, "vector"]
    assert "2.0" in vec_str and "3.0" in vec_str
