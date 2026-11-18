"""Race-outcome and crash-risk models with walk-forward (season-by-season) validation.

Outcome models (same design as the companion F1 project):

* **Grid order** baseline, **logistic regression**, **LightGBM classifier** for P(podium) and
  **LightGBM LambdaRank** for the full finishing order.

Crash / DNF risk:

* **LightGBM classifier** for P(did not finish), compared with two baselines: the overall base
  rate and the rider's own rolling DNF rate. Evaluated with PR-AUC (DNFs are the minority class).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRanker
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from motogp_analytics.config import MODELS, TEST_SEASONS
from motogp_analytics.features import DNF_FEATURES, FEATURES

SCORE_COLUMNS = {
    "Grid order": "s_grid",
    "Logistic regression": "p_logreg",
    "LightGBM classifier": "p_lgbm",
    "LightGBM LambdaRank": "s_rank",
}
PROBABILISTIC = {"Logistic regression", "LightGBM classifier"}
ID_COLUMNS = [
    "race_id",
    "season",
    "round",
    "event_short",
    "event_name",
    "circuit_name",
    "date",
    "rider_id",
    "rider_name",
    "rider_number",
    "team_name",
    "manufacturer",
    "grid",
    "finish_position",
    "position",
    "status",
    "podium",
    "win",
    "dnf",
    "wet",
]
MAX_RELEVANCE = 20


def make_classifier(n_estimators: int = 150, random_state: int = 7) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=7,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state,
        verbose=-1,
    )


def make_dnf_classifier(random_state: int = 7) -> LGBMClassifier:
    """Heavily regularised: crash risk is a weak, noisy signal (chosen on 2019-2021)."""
    return LGBMClassifier(
        n_estimators=100,
        learning_rate=0.03,
        num_leaves=4,
        min_child_samples=150,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=random_state,
        verbose=-1,
    )


def make_logreg() -> Pipeline:
    return Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.5, max_iter=2000)),
        ]
    )


def make_ranker(random_state: int = 7) -> LGBMRanker:
    return LGBMRanker(
        objective="lambdarank",
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        label_gain=[i * i for i in range(MAX_RELEVANCE + 1)],
        lambdarank_truncation_level=20,
        random_state=random_state,
        verbose=-1,
    )


def relevance(finish_position: pd.Series) -> np.ndarray:
    pos = finish_position.fillna(99).to_numpy()
    return np.clip(MAX_RELEVANCE + 1 - pos, 0, MAX_RELEVANCE).astype(int)


def fit_all(train: pd.DataFrame, features: list[str] = FEATURES) -> dict:
    train = train.sort_values(["date", "race_id"], kind="stable")
    X, y = train[features], train["podium"]
    groups = train.groupby("race_id", sort=False).size().to_numpy()
    return {
        "Logistic regression": make_logreg().fit(X, y),
        "LightGBM classifier": make_classifier().fit(X, y),
        "LightGBM LambdaRank": make_ranker().fit(
            X, relevance(train["finish_position"]), group=groups
        ),
    }


def score(models: dict, df: pd.DataFrame, features: list[str] = FEATURES) -> pd.DataFrame:
    out = df[[c for c in ID_COLUMNS if c in df.columns]].copy()
    X = df[features]
    out["s_grid"] = -df["grid"].to_numpy()
    out["p_logreg"] = models["Logistic regression"].predict_proba(X)[:, 1]
    out["p_lgbm"] = models["LightGBM classifier"].predict_proba(X)[:, 1]
    out["s_rank"] = models["LightGBM LambdaRank"].predict(X)
    return out


def walk_forward(
    df: pd.DataFrame, test_seasons: list[int] = TEST_SEASONS, features: list[str] = FEATURES
) -> pd.DataFrame:
    parts = []
    for season in test_seasons:
        train, test = df[df["season"] < season], df[df["season"] == season]
        if not test.empty:
            parts.append(score(fit_all(train, features), test, features))
    return pd.concat(parts, ignore_index=True)


def race_ranking_metrics(race: pd.DataFrame, score_col: str) -> dict:
    ordered = race.sort_values(score_col, ascending=False, kind="stable")
    predicted_top3 = set(ordered["rider_id"].iloc[:3])
    actual_top3 = set(race.loc[race["finish_position"] <= 3, "rider_id"])
    winner = set(race.loc[race["finish_position"] == 1, "rider_id"])
    rho = spearmanr(-race[score_col].to_numpy(), race["finish_position"].to_numpy()).statistic
    return {
        "top3_hit": len(predicted_top3 & actual_top3) / 3,
        "winner_hit": float(ordered["rider_id"].iloc[0] in winner),
        "spearman": float(rho),
    }


def evaluate(pred: pd.DataFrame, by: str | None = "season") -> pd.DataFrame:
    groups = [("All", pred)]
    if by is not None:
        groups += [(key, g) for key, g in pred.groupby(by)]
    rows = []
    for model, col in SCORE_COLUMNS.items():
        for key, g in groups:
            per_race = pd.DataFrame(
                [race_ranking_metrics(r, col) for _, r in g.groupby("race_id")]
            ).mean()
            row = {
                "model": model,
                by or "group": key,
                "races": g["race_id"].nunique(),
                "roc_auc": roc_auc_score(g["podium"], g[col]),
                "top3_hit": per_race["top3_hit"],
                "winner_acc": per_race["winner_hit"],
                "spearman": per_race["spearman"],
                "log_loss": np.nan,
                "brier": np.nan,
            }
            if model in PROBABILISTIC:
                row["log_loss"] = log_loss(g["podium"], g[col].clip(1e-6, 1 - 1e-6))
                row["brier"] = brier_score_loss(g["podium"], g[col])
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Crash / DNF risk
# --------------------------------------------------------------------------------------


def dnf_walk_forward(
    df: pd.DataFrame, test_seasons: list[int] = TEST_SEASONS, features: list[str] = DNF_FEATURES
) -> pd.DataFrame:
    parts = []
    for season in test_seasons:
        train, test = df[df["season"] < season], df[df["season"] == season]
        if test.empty:
            continue
        clf = make_dnf_classifier().fit(train[features], train["dnf"])
        out = test[[c for c in ID_COLUMNS if c in test.columns]].copy()
        out["p_dnf"] = clf.predict_proba(test[features])[:, 1]
        out["base_rate"] = train["dnf"].mean()
        out["rider_rate"] = test["rdr_dnf_rate_30"].fillna(train["dnf"].mean()).to_numpy()
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def dnf_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, col in [
        ("Base rate", "base_rate"),
        ("Rider's DNF rate (last 30)", "rider_rate"),
        ("LightGBM", "p_dnf"),
    ]:
        y, p = pred["dnf"], pred[col]
        rows.append(
            {
                "model": name,
                "pr_auc": average_precision_score(y, p),
                "roc_auc": roc_auc_score(y, p) if p.nunique() > 1 else 0.5,
                "brier": brier_score_loss(y, p),
            }
        )
    return pd.DataFrame(rows)


def permutation_importance_table(
    model, X: pd.DataFrame, y: pd.Series, scoring: str = "roc_auc", n_repeats: int = 10
) -> pd.DataFrame:
    result = permutation_importance(
        model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=7, n_jobs=1
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train_final(df: pd.DataFrame) -> dict:
    return {
        "podium_model": make_classifier().fit(df[FEATURES], df["podium"]),
        "dnf_model": make_dnf_classifier().fit(df[DNF_FEATURES], df["dnf"]),
        "features": FEATURES,
        "dnf_features": DNF_FEATURES,
        "trained_through": str(df["date"].max().date()),
        "n_rows": len(df),
    }


def save_bundle(bundle: dict, path: Path = MODELS / "motogp_models.joblib") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: Path = MODELS / "motogp_models.joblib") -> dict:
    return joblib.load(path)
