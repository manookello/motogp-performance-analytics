"""Charts for the notebooks, README and app.

Same visual system as the companion F1 projects: a validated colour-blind-safe categorical palette
assigned in a fixed order, thin marks, hairline solid grids, one y-axis per chart, and text in ink
colours. Manufacturers always keep the same colour.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from motogp_analytics.config import FIGURES
from motogp_analytics.features import FEATURE_LABELS, MANUFACTURERS

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
OTHER = "#c3c2b7"

# Fixed manufacturer -> colour mapping (slot order), used everywhere.
MFR_COLORS = {m: PALETTE[i] for i, m in enumerate(MANUFACTURERS)} | {"Other": OTHER}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
        }
    )


def save(fig: plt.Figure, name: str, folder: Path = FIGURES) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path)
    return path


def _subtitle(ax: plt.Axes, text: str) -> None:
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")


# --------------------------------------------------------------------------------------
# EDA
# --------------------------------------------------------------------------------------


def manufacturer_share(df: pd.DataFrame, outcome: str = "win") -> plt.Figure:
    """Stacked share of wins (or podiums) per season by manufacturer."""
    order = MANUFACTURERS + ["Other"]
    table = (
        df[df[outcome] == 1]
        .groupby(["season", "manufacturer"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=order, fill_value=0)
    )
    table = table.loc[:, table.sum() > 0]
    share = table.div(table.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bottom = np.zeros(len(share))
    for m in share.columns:
        ax.bar(
            share.index,
            share[m],
            bottom=bottom,
            width=0.62,
            color=MFR_COLORS[m],
            label=m,
            edgecolor=SURFACE,
            linewidth=1.5,
        )
        bottom += share[m].to_numpy()
    ax.set_ylim(0, 100)
    ax.set_xticks(share.index)
    ax.set_ylabel(f"Share of {'race wins' if outcome == 'win' else 'podiums'} (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(
        ncol=len(share.columns),
        loc="upper left",
        bbox_to_anchor=(0, -0.08),
        fontsize=9,
        handlelength=1,
        columnspacing=1.1,
    )
    ax.set_title("Honda and Yamaha, then the Ducati era - and Aprilia's 2026", pad=22)
    noun = "Grand Prix wins" if outcome == "win" else "podium finishes"
    _subtitle(ax, f"Share of {noun} per season by manufacturer")
    return fig


def podium_rate_by_grid(df: pd.DataFrame, max_grid: int = 20) -> plt.Figure:
    rate = (
        df[df["grid"].between(1, max_grid)]
        .groupby("grid")["podium"]
        .mean()
        .mul(100)
        .reindex(range(1, max_grid + 1), fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(rate.index, rate.values, width=0.6, color=PALETTE[0])
    for g in (1, 2, 3):
        ax.annotate(
            f"{rate[g]:.0f}%",
            (g, rate[g]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=INK_2,
        )
    ax.set_xticks(range(1, max_grid + 1))
    ax.set_xlabel("Starting grid position (from qualifying)")
    ax.set_ylabel("Podium finishes (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Pole matters less on two wheels", pad=22)
    _subtitle(
        ax,
        f"Share of starts from each grid slot ending on the podium, "
        f"{df['season'].min()}-{df['season'].max()}",
    )
    return fig


def dnf_by_manufacturer(df: pd.DataFrame, since: int = 2019) -> plt.Figure:
    d = df[df["season"] >= since]
    rate = d.groupby("manufacturer")["dnf"].agg(["mean", "size"])
    rate = rate[rate["size"] >= 50].sort_values("mean")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(rate.index, rate["mean"] * 100, height=0.55, color=[MFR_COLORS[m] for m in rate.index])
    for y, v in enumerate(rate["mean"] * 100):
        ax.text(v + 0.4, y, f"{v:.0f}%", va="center", fontsize=9, color=INK_2)
    ax.set_xlabel("Starts that did not finish (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Who falls off most?", pad=22)
    _subtitle(ax, f"Race DNF rate by manufacturer, {since}-{df['season'].max()}")
    return fig


def championship_progression(df: pd.DataFrame, season: int, top: int = 4) -> plt.Figure:
    s = df[df["season"] == season].sort_values("round")
    pts = (
        s.pivot_table(index="round", columns="rider_name", values="weekend_points", aggfunc="sum")
        .fillna(0)
        .cumsum()
    )
    leaders = pts.iloc[-1].sort_values(ascending=False).index[:top]
    fig, ax = plt.subplots(figsize=(8, 4))
    for color, rider in zip(PALETTE, leaders, strict=False):
        ax.plot(pts.index, pts[rider], color=color, label=rider)
        ax.annotate(
            f"{rider.split()[-1]} {pts[rider].iloc[-1]:.0f}",
            (pts.index[-1], pts[rider].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_2,
        )
    ax.set_xlim(1, pts.index[-1] + 2.5)
    ax.set_xticks(pts.index)
    ax.set_xlabel("Round")
    ax.set_ylabel("Championship points (race + sprint)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"{season} title fight", pad=12)
    return fig


def sprint_winner_sunday(df: pd.DataFrame) -> plt.Figure:
    sw = df[df["sprint_position"] == 1]
    buckets = pd.cut(
        sw["finish_position"], [0, 1, 2, 3, 10, 100], labels=["Won", "P2", "P3", "P4-P10", "Lower"]
    )
    buckets = buckets.cat.add_categories("Did not finish")
    buckets[sw["dnf"] == 1] = "Did not finish"
    share = buckets.value_counts(normalize=True).reindex(buckets.cat.categories) * 100
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(share.index.astype(str), share.values, width=0.55, color=PALETTE[0])
    for x, v in enumerate(share.values):
        ax.text(x, v + 1, f"{v:.0f}%", ha="center", fontsize=9, color=INK_2)
    ax.set_ylabel("Share of sprint winners (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Saturday's sprint winner on Sunday", pad=22)
    _subtitle(
        ax,
        f"Grand Prix result of the sprint winner, {sw['season'].min()}-"
        f"{sw['season'].max()} ({len(sw)} weekends)",
    )
    return fig


# --------------------------------------------------------------------------------------
# Model results (shared with the F1 project)
# --------------------------------------------------------------------------------------


def model_comparison(metrics: pd.DataFrame) -> plt.Figure:
    """Small multiples of the pooled ('All') metrics; each panel highlights its best model."""
    overall = metrics[metrics["season"] == "All"].set_index("model")
    panels = [
        ("roc_auc", "Podium ROC-AUC", "{:.3f}"),
        ("top3_hit", "Podium riders found (top-3 hit rate)", "{:.0%}"),
        ("winner_acc", "Winner picked correctly", "{:.0%}"),
    ]
    order = list(overall.index)[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, (col, title, fmt) in zip(axes, panels, strict=True):
        vals = overall.loc[order, col]
        colors = [PALETTE[0] if v == vals.max() else OTHER for v in vals]
        ax.barh(order, vals, height=0.55, color=colors)
        lo = 0.5 if col == "roc_auc" else 0
        ax.set_xlim(lo, 1.0)
        for y, v in enumerate(vals):
            ax.text(v + (1 - lo) * 0.015, y, fmt.format(v), va="center", fontsize=9, color=INK_2)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        "Walk-forward test results, all test seasons pooled",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    fig.tight_layout()
    return fig


def metric_by_season(metrics: pd.DataFrame, col: str, ylabel: str) -> plt.Figure:
    per = metrics[metrics["season"] != "All"].copy()
    per["season"] = per["season"].astype(int)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for model, color in zip(per["model"].unique(), PALETTE, strict=False):
        m = per[per["model"] == model]
        ax.plot(
            m["season"],
            m[col],
            color=color,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=model,
        )
    ax.set_xticks(sorted(per["season"].unique()))
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=4, fontsize=9)
    ax.set_title(f"{ylabel} by test season", pad=12)
    return fig


def calibration(
    pred: pd.DataFrame,
    col: str = "p_lgbm",
    target: str = "podium",
    title: str = "Are the probabilities honest?",
) -> plt.Figure:
    frac_pos, mean_pred = calibration_curve(pred[target], pred[col], n_bins=10, strategy="quantile")
    hi = max(mean_pred.max(), frac_pos.max()) * 1.1
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, hi], [0, hi], color=AXIS, linewidth=1)
    ax.plot(
        mean_pred,
        frac_pos,
        color=PALETTE[0],
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
    )
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel(f"Predicted {target} probability")
    ax.set_ylabel(f"Observed {target} rate")
    ax.set_title(title, pad=22)
    _subtitle(ax, "Walk-forward test seasons, quantile bins; grey line = perfect")
    return fig


def importance(
    imp: pd.DataFrame,
    top: int = 12,
    title: str = "What drives the prediction",
    metric: str = "ROC-AUC",
) -> plt.Figure:
    imp = imp.head(top).iloc[::-1]
    labels = [FEATURE_LABELS.get(f, f) for f in imp["feature"]]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(imp) + 1.2))
    ax.barh(labels, imp["importance"], height=0.55, color=PALETTE[0])
    ax.set_xlabel(f"Drop in {metric} when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, pad=22)
    _subtitle(ax, "Permutation importance on held-out seasons")
    return fig
