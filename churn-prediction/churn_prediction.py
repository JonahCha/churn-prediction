"""Customer churn prediction on the IBM Telco Customer Churn dataset.

Usage:
    python churn_prediction.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --outdir docs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from plots import save_figures, save_console_image

DEFAULT_DATA = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
DEFAULT_OUTDIR = Path("docs")
REQUIRED_COLUMNS = {"tenure", "TotalCharges", "Churn"}
RANDOM_STATE = 42
TEST_SIZE = 0.2
# Models within this ROC-AUC of the best are treated as tied; the tie is broken by recall,
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


def evaluate(models: dict, X_tr, X_te, y_tr, y_te) -> dict[str, dict[str, float]]:
    """Fit each model and score it on the held-out test set."""
    results = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]
        results[name] = {
            "accuracy": accuracy_score(y_te, pred),
            "recall": recall_score(y_te, pred),
            "roc_auc": roc_auc_score(y_te, proba),
        }
    return results


def pick_best(results: dict[str, dict[str, float]]) -> str:
    top_auc = max(r["roc_auc"] for r in results.values())
    contenders = [n for n, r in results.items() if r["roc_auc"] >= top_auc - AUC_TIE_TOLERANCE]
    return max(contenders, key=lambda n: results[n]["recall"])


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


def run(path: Path) -> dict:
    df, n_raw = load_and_clean(path)
    base_cols = [c for c in df.columns if c != "Churn"]
    X, y = build_features(df)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    models = build_models()
    results = evaluate(models, X_tr, X_te, y_tr, y_te)
    best = pick_best(results)
    drivers = top_drivers(models["Random Forest"], X.columns, base_cols)

    return {
        "df": df,
        "n_raw": n_raw,
        "n_clean": len(df),
        "n_test": len(y_te),
        "churn_rate": y.mean() * 100,
        "results": results,
        "best": best,
        "best_pred": models[best].predict(X_te),
        "y_test": y_te.to_numpy(),
        "drivers": drivers,
        "top_drivers": list(drivers.index[:3]),
    }


def format_report(s: dict) -> str:
    lines = [
        f"Raw rows: {s['n_raw']:,}  after cleaning: {s['n_clean']:,}  "
        f"churn rate: {s['churn_rate']:.1f}%",
        "",
    ]
    for name, r in s["results"].items():
        lines.append(
            f"{name:20s} acc={r['accuracy']:.3f} recall={r['recall']:.3f} roc_auc={r['roc_auc']:.3f}"
        )
    b = s["results"][s["best"]]
    lines += [
        "",
        f"Best model: {s['best']}  acc={b['accuracy']:.1%} recall={b['recall']:.1%} "
        f"roc_auc={b['roc_auc']:.2f}",
        f"Top 3 churn drivers: {', '.join(s['top_drivers'])}",
    ]
    return "\n".join(lines)


def resume_bullets(s: dict) -> list[str]:
    b = s["results"][s["best"]]
    return [
        f"Predicted customer churn with {b['roc_auc']:.2f} ROC-AUC, {b['accuracy']:.0%} accuracy "
        f"and {b['recall']:.0%} recall on a held-out test set of {s['n_test']:,} customers, after "
        f"cleaning the data, handling class imbalance and comparing {', '.join(s['results'])}.",
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
