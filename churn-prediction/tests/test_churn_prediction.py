import numpy as np
import pandas as pd
import pytest

from churn_prediction import (
    build_features,
    load_and_clean,
    original_feature,
    run,
)


def make_telco_csv(path, n=300, seed=0):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    monthly = rng.uniform(20, 110, n).round(2)
    total = np.array([f"{t * m:.2f}" for t, m in zip(tenure, monthly)])
    total[tenure == 0] = " "  # blank strings, as in the real dataset
    p = np.clip(0.1 + 0.5 * (contract == "Month-to-month") - 0.004 * tenure, 0.02, 0.9)
    churn = np.where(rng.uniform(size=n) < p, "Yes", "No")
    df = pd.DataFrame({
        "customerID": [f"C{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "tenure": tenure,
        "Contract": contract,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": churn,
    })
    df.to_csv(path, index=False)
    return path


def test_load_and_clean_fills_blank_total_charges(tmp_path):
    path = make_telco_csv(tmp_path / "telco.csv")
    df, n_raw = load_and_clean(path)
    assert n_raw == 300
    assert "customerID" not in df.columns
    assert df["TotalCharges"].notna().all()
    assert df["TotalCharges"].dtype.kind == "f"


def test_missing_columns_raise(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"tenure": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_and_clean(path)


def test_build_features_target_is_binary(tmp_path):
    path = make_telco_csv(tmp_path / "telco.csv")
    df, _ = load_and_clean(path)
    X, y = build_features(df)
    assert set(y.unique()) == {0, 1}
    assert "Churn" not in X.columns
    assert any(c.startswith("Contract_") for c in X.columns)


def test_original_feature_maps_dummies_back():
    base = ["Contract", "Contract_extra", "tenure"]
    assert original_feature("Contract_Two year", base) == "Contract"
    assert original_feature("tenure", base) == "tenure"


def test_end_to_end_metrics_are_sane(tmp_path):
    path = make_telco_csv(tmp_path / "telco.csv")
    s = run(path)
    assert s["best"] in s["results"]
    for r in s["results"].values():
        assert 0.0 <= r["accuracy"] <= 1.0
        assert 0.0 <= r["recall"] <= 1.0
        assert 0.5 <= r["roc_auc"] <= 1.0
    assert s["n_test"] == 60  # 20% of 300
    assert "Contract" in s["top_drivers"]
