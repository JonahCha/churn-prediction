"""Customer churn prediction on the IBM Telco Customer Churn dataset.

Model selection uses 5-fold cross-validation on the training split only.
The held-out test set is scored once, for the selected model.

Usage:
    python churn_prediction.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --outdir docs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from plots import save_console_image, save_figures

DEFAULT_DATA = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
DEFAULT_OUTDIR = Path("docs")
REQUIRED_COLUMNS = {"tenure", "TotalCharges", "Churn"}
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TOP_SHARE = 0.2  # share of customers on the retention list
# Models within this CV ROC-AUC of the best are treated as tied; recall breaks the tie,
# because missing a churning customer is the costlier error.
AUC_TIE_TOLERANCE = 0.005


def load_and_clean(path: Path) -> tuple[pd.DataFrame, int]:
    """Load the raw CSV and return (cleaned_frame, raw_row_count)."""
    raw = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    n_raw = len(raw)
    df = raw.copy()

    # customerID is unique per customer, so only repeated IDs count as duplicates.
    if "customerID" in df.columns:
        df = df.drop_duplicates(subset="customerID").drop(columns="customerID")

    # TotalCharges is stored as text; blank strings become NaN.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # A blank TotalCharges on a tenure-0 customer means no charges yet, so it is 0.
    new_customer = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[new_customer, "TotalCharges"] = 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df, n_raw


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return one-hot encoded features and a 0/1 churn target."""
    y = (df["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    X = pd.get_dummies(df.drop(columns="Churn"), drop_first=True)
    return X, y


def build_models() -> dict:
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def cross_validate_models(X: pd.DataFrame, y: pd.Series) -> dict[str, dict[str, float]]:
    """Mean and std of ROC-AUC and recall across folds, computed on training data only."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for name, model in build_models().items():
        scores = cross_validate(model, X, y, cv=cv, scoring=["roc_auc", "recall"])
        results[name] = {
            "roc_auc": scores["test_roc_auc"].mean(),
            "roc_auc_std": scores["test_roc_auc"].std(),
            "recall": scores["test_recall"].mean(),
        }
    return results


def pick_best(cv_results: dict[str, dict[str, float]]) -> str:
    top_auc = max(r["roc_auc"] for r in cv_results.values())
    contenders = [n for n, r in cv_results.items() if r["roc_auc"] >= top_auc - AUC_TIE_TOLERANCE]
    return max(contenders, key=lambda n: cv_results[n]["recall"])


def original_feature(encoded: str, base_cols: list[str]) -> str:
    """Map a one-hot column such as 'Contract_Two year' back to 'Contract'."""
    for base in sorted(base_cols, key=len, reverse=True):
        if encoded == base or encoded.startswith(base + "_"):
            return base
    return encoded


def top_drivers(rf_model, X_columns, base_cols: list[str]) -> pd.Series:
    """Random Forest importances summed back to original features, highest first."""
    imp = pd.Series(rf_model.feature_importances_, index=X_columns)
    grouped = imp.groupby(lambda c: original_feature(c, base_cols)).sum()
    return grouped.sort_values(ascending=False)


def retention_list_impact(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """How many churners the top-risk list captures, scored on the test set."""
    k = int(round(TOP_SHARE * len(proba)))
    order = np.argsort(-proba)
    top = y_true[order[:k]]
    return {
        "list_size": k,
        "churners_on_list": int(top.sum()),
        "churners_total": int(y_true.sum()),
        "capture": top.sum() / y_true.sum(),
        "precision": top.mean(),
    }


def run(path: Path) -> dict:
    df, n_raw = load_and_clean(path)
    base_cols = [c for c in df.columns if c != "Churn"]
    X, y = build_features(df)

    X_trval, X_te, y_trval, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 1. choose the model using training data only
    cv_results = cross_validate_models(X_trval, y_trval)
    best = pick_best(cv_results)

    # 2. fit the winner on training data and score the test set once
    final = build_models()[best].fit(X_trval, y_trval)
    pred = final.predict(X_te)
    proba = final.predict_proba(X_te)[:, 1]
    test = {
        "accuracy": accuracy_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, proba),
    }

    # 3. feature drivers from a Random Forest fitted on the same training data
    rf = build_models()["Random Forest"].fit(X_trval, y_trval)
    drivers = top_drivers(rf, X.columns, base_cols)

    return {
        "df": df,
        "n_raw": n_raw,
        "n_clean": len(df),
        "n_train": len(y_trval),
        "n_test": len(y_te),
        "churn_rate": y.mean() * 100,
        "cv": cv_results,
        "best": best,
        "test": test,
        "best_pred": pred,
        "y_test": y_te.to_numpy(),
        "impact": retention_list_impact(y_te.to_numpy(), proba),
        "drivers": drivers,
        "top_drivers": list(drivers.index[:3]),
    }


def format_report(s: dict) -> str:
    lines = [
        f"Raw rows: {s['n_raw']:,}  after cleaning: {s['n_clean']:,}  "
        f"churn rate: {s['churn_rate']:.1f}%",
        f"Train: {s['n_train']:,}  test: {s['n_test']:,}",
        "",
        f"Model selection ({CV_FOLDS}-fold CV on training data):",
    ]
    for name, r in s["cv"].items():
        lines.append(
            f"  {name:20s} roc_auc={r['roc_auc']:.3f} (+/-{r['roc_auc_std']:.3f})  recall={r['recall']:.3f}"
        )
    t = s["test"]
    lines += [
        "",
        f"Selected: {s['best']}",
        f"Test set (scored once): acc={t['accuracy']:.1%} recall={t['recall']:.1%} roc_auc={t['roc_auc']:.2f}",
        "",
        f"Retention list (top {TOP_SHARE:.0%} by risk, test set): {s['impact']['list_size']} customers, "
        f"{s['impact']['churners_on_list']} churners "
        f"({s['impact']['capture']:.1%} of all churners, {s['impact']['precision']:.1%} precision)",
        f"Top 3 churn drivers: {', '.join(s['top_drivers'])}",
    ]
    return "\n".join(lines)


def resume_bullets(s: dict) -> list[str]:
    t = s["test"]
    return [
        f"Predicted customer churn with {t['roc_auc']:.2f} ROC-AUC, {t['accuracy']:.0%} accuracy and "
        f"{t['recall']:.0%} recall on a held-out test set of {s['n_test']:,} customers, selecting the model "
        f"with 5-fold cross-validation and handling class imbalance across {', '.join(s['cv'])}.",
        f"Identified {', '.join(s['top_drivers'])} as the top churn drivers (Random Forest feature "
        f"importance) against an overall churn rate of {s['churn_rate']:.1f}%, to inform targeted retention.",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to the Telco churn CSV")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                        help="Where to write images/ and results.txt")
    args = parser.parse_args()

    summary = run(args.data)
    report = format_report(summary)
    bullets = resume_bullets(summary)
    console = report + "\n\n===== RESUME BULLETS =====\n" + "\n".join(f"- {b}" for b in bullets)

    print(console)

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "results.txt").write_text(console + "\n", encoding="utf-8")
    save_figures(summary, args.outdir / "images")
    save_console_image(console, args.outdir / "images" / "console_output.png")


if __name__ == "__main__":
    main()
