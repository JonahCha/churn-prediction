"""Figure generation for the churn project. Writes PNGs for the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402

PALETTE = ["#4C72B0", "#DD8452", "#55A868"]


def save_figures(summary: dict, figdir: Path) -> list[Path]:
    figdir.mkdir(parents=True, exist_ok=True)
    return [
        _churn_distribution(summary, figdir / "churn_distribution.png"),
        _churn_by_contract(summary, figdir / "churn_by_contract.png"),
        _model_comparison(summary, figdir / "model_comparison.png"),
        _confusion_matrix(summary, figdir / "confusion_matrix.png"),
        _feature_importance(summary, figdir / "feature_importance.png"),
    ]


def _finish(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _churn_distribution(s: dict, path: Path) -> Path:
    counts = s["df"]["Churn"].astype(str).str.strip().value_counts().reindex(["No", "Yes"]).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Stayed", "Churned"], counts.values, color=[PALETTE[0], PALETTE[1]])
    ax.bar_label(bars, labels=[f"{v:,}" for v in counts.values])
    ax.set_title(f"Customer churn distribution ({s['churn_rate']:.1f}% churned)")
    ax.set_ylabel("Customers")
    return _finish(fig, path)


def _churn_by_contract(s: dict, path: Path) -> Path:
    df = s["df"]
    rates = (
        df.assign(_c=df["Churn"].astype(str).str.strip().str.lower() == "yes")
        .groupby("Contract")["_c"].mean() * 100
    ).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(rates.index, rates.values, color=PALETTE[1])
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in rates.values])
    ax.set_title("Churn rate by contract type")
    ax.set_ylabel("Churn rate (%)")
    return _finish(fig, path)


def _model_comparison(s: dict, path: Path) -> Path:
    names = list(s["cv"])
    metrics = [("roc_auc", "ROC-AUC"), ("recall", "Recall")]
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (key, label) in enumerate(metrics):
        vals = [s["cv"][n][key] for n in names]
        xs = [x + (i - 0.5) * width for x in range(len(names))]
        bars = ax.bar(xs, vals, width, label=label, color=PALETTE[i])
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison (5-fold CV on training data)")
    ax.legend()
    return _finish(fig, path)


def _confusion_matrix(s: dict, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        s["y_test"], s["best_pred"], display_labels=["Stayed", "Churned"], ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion matrix: {s['best']}")
    return _finish(fig, path)


def _feature_importance(s: dict, path: Path) -> Path:
    top = s["drivers"].head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(top.index, top.values, color=PALETTE[0])
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in top.values], fontsize=8)
    ax.set_title("Top churn drivers (Random Forest importance)")
    ax.set_xlabel("Importance")
    return _finish(fig, path)


def save_console_image(text: str, path: Path) -> Path:
    """Render the console output as an image so it can be shown in the README."""
    lines = text.splitlines()
    width = max(len(line) for line in lines) if lines else 40
    fig_w = max(6, width * 0.085)
    fig_h = max(2, len(lines) * 0.22 + 0.4)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#1e1e1e")
    fig.text(0.02, 0.97, "\n".join(lines), family="monospace", fontsize=9,
             color="#d4d4d4", va="top", ha="left")
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
