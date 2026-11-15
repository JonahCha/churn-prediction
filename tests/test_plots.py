from churn_prediction import run
from plots import save_console_image, save_figures
from tests.test_churn_prediction import make_telco_csv


def test_figures_are_written(tmp_path):
    summary = run(make_telco_csv(tmp_path / "telco.csv"))
    paths = save_figures(summary, tmp_path / "images")
    assert len(paths) == 5
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_console_image_is_written(tmp_path):
    p = save_console_image("line one\nline two", tmp_path / "console.png")
    assert p.exists() and p.stat().st_size > 0
