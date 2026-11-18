"""Interactive MotoGP analytics explorer.

Run with:  streamlit run app/streamlit_app.py
Reads only the processed files committed under data/processed/ (no API calls).
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from motogp_analytics.config import DATA_PROCESSED
from motogp_analytics.features import MANUFACTURERS
from motogp_analytics.model import SCORE_COLUMNS, race_ranking_metrics
from motogp_analytics.plots import MFR_COLORS, OTHER, PALETTE

st.set_page_config(page_title="MotoGP Performance Analytics", page_icon="🏍️", layout="wide")
MODEL_COLORS = dict(zip(SCORE_COLUMNS, PALETTE, strict=False))
MFR_ORDER = MANUFACTURERS + ["Other"]


@st.cache_data
def load():
    feats = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    pred = pd.read_parquet(DATA_PROCESSED / "predictions.parquet")
    metrics = pd.read_csv(DATA_PROCESSED / "metrics.csv")
    dnf_metrics = pd.read_csv(DATA_PROCESSED / "dnf_metrics.csv")
    return feats, pred, metrics, dnf_metrics


def tidy(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
    )
    return fig


feats, pred, metrics, dnf_metrics = load()
seasons = sorted(feats["season"].unique(), reverse=True)

st.title("MotoGP Performance Analytics")
st.caption(
    f"{feats['race_id'].nunique()} Grands Prix, {feats['season'].min()}–{feats['season'].max()}, "
    "from the public MotoGP results API. Explore seasons, manufacturers and rider head-to-heads, "
    "and see out-of-sample race predictions."
)

tab_season, tab_mfr, tab_h2h, tab_pred, tab_perf = st.tabs(
    ["Season explorer", "Manufacturers", "Head-to-head", "Race predictor", "Model performance"]
)

# ---------------------------------------------------------------------------------------
with tab_season:
    season = st.selectbox("Season", seasons, key="season_explorer")
    s = feats[feats["season"] == season]
    standings = (
        s.groupby(["rider_name"])
        .agg(
            points=("weekend_points", "sum"),
            wins=("win", "sum"),
            podiums=("podium", "sum"),
            dnfs=("dnf", "sum"),
            starts=("race_id", "count"),
            bike=("manufacturer", lambda m: m.mode().iat[0]),
        )
        .sort_values("points", ascending=False)
    )
    top = standings.index[:5].tolist()
    prog = (
        s[s["rider_name"].isin(top)]
        .pivot_table(index="round", columns="rider_name", values="weekend_points", aggfunc="sum")
        .fillna(0)
        .cumsum()
        .reset_index()
        .melt(id_vars="round", var_name="rider", value_name="points")
    )
    fig = px.line(
        prog,
        x="round",
        y="points",
        color="rider",
        markers=True,
        category_orders={"rider": top},
        color_discrete_sequence=PALETTE,
        labels={"round": "Round", "points": "Points (race + sprint)", "rider": ""},
        title=f"{season} championship – top 5",
    )
    fig.update_traces(line_width=2, marker_size=7)
    st.plotly_chart(tidy(fig), use_container_width=True)
    st.dataframe(standings, use_container_width=True)

# ---------------------------------------------------------------------------------------
with tab_mfr:
    measure = st.radio("Measure", ["Wins", "Podiums", "DNF rate"], horizontal=True)
    if measure == "DNF rate":
        rate = feats.groupby(["season", "manufacturer"])["dnf"].mean().reset_index()
        fig = px.line(
            rate,
            x="season",
            y="dnf",
            color="manufacturer",
            markers=True,
            category_orders={"manufacturer": MFR_ORDER},
            color_discrete_map=MFR_COLORS,
            labels={"dnf": "DNF rate", "season": "", "manufacturer": ""},
            title="Share of starts not finished, by manufacturer",
        )
        fig.update_yaxes(tickformat=".0%")
    else:
        col = "win" if measure == "Wins" else "podium"
        counts = (
            feats[feats[col] == 1]
            .groupby(["season", "manufacturer"])
            .size()
            .rename("count")
            .reset_index()
        )
        fig = px.bar(
            counts,
            x="season",
            y="count",
            color="manufacturer",
            category_orders={"manufacturer": MFR_ORDER},
            color_discrete_map=MFR_COLORS,
            labels={"count": measure, "season": "", "manufacturer": ""},
            title=f"{measure} per season by manufacturer",
        )
        fig.update_layout(bargap=0.35)
    fig.update_xaxes(dtick=1)
    st.plotly_chart(tidy(fig, 460), use_container_width=True)

# ---------------------------------------------------------------------------------------
with tab_h2h:
    latest = feats[feats["season"] == seasons[0]]
    default = latest.groupby("rider_name")["weekend_points"].sum().nlargest(2).index.tolist()
    riders = sorted(feats["rider_name"].unique())
    c1, c2 = st.columns(2)
    a = c1.selectbox("Rider A", riders, index=riders.index(default[0]))
    b = c2.selectbox("Rider B", riders, index=riders.index(default[1]))
    ra = feats[feats["rider_name"] == a].set_index("race_id")
    rb = feats[feats["rider_name"] == b].set_index("race_id")
    common = ra.index.intersection(rb.index)
    if a == b or len(common) == 0:
        st.info("Pick two different riders who raced each other.")
    else:
        ahead_a = ra.loc[common, "finish_position"] < rb.loc[common, "finish_position"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Races together", len(common))
        k2.metric(f"{a} ahead", int(ahead_a.sum()))
        k3.metric(f"{b} ahead", int((~ahead_a).sum()))
        k4.metric(
            "Points (shared races)",
            f"{ra.loc[common, 'points'].sum():.0f} – {rb.loc[common, 'points'].sum():.0f}",
        )
        by_season = pd.DataFrame(
            {"season": ra.loc[common, "season"], a: ahead_a.astype(int), b: (~ahead_a).astype(int)}
        )
        by_season = (
            by_season.groupby("season")[[a, b]]
            .sum()
            .reset_index()
            .melt(id_vars="season", var_name="rider", value_name="races ahead")
        )
        fig = px.bar(
            by_season,
            x="season",
            y="races ahead",
            color="rider",
            barmode="group",
            color_discrete_sequence=PALETTE[:2],
            labels={"season": "", "rider": ""},
            title="Who finished ahead, by season (including DNFs)",
        )
        fig.update_xaxes(dtick=1)
        fig.update_layout(bargap=0.3, bargroupgap=0.08)
        st.plotly_chart(tidy(fig, 380), use_container_width=True)

# ---------------------------------------------------------------------------------------
with tab_pred:
    c1, c2, c3 = st.columns([1, 2, 2])
    p_season = c1.selectbox("Season", sorted(pred["season"].unique(), reverse=True))
    races = (
        pred[pred["season"] == p_season]
        .drop_duplicates("race_id")
        .sort_values("round", ascending=False)
    )
    labels = {r.race_id: f"R{r.round} · {r.event_name.title()}" for r in races.itertuples()}
    race_id = c2.selectbox("Race", list(labels), format_func=labels.get)
    model_name = c3.selectbox("Model", ["Logistic regression", "LightGBM classifier"])
    col = SCORE_COLUMNS[model_name]
    race = pred[pred["race_id"] == race_id].copy()
    race["rider"] = race["rider_name"]
    rm = race_ranking_metrics(race, col)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Predicted podium", " · ".join(n.split()[-1] for n in race.nlargest(3, col)["rider"]))
    k2.metric(
        "Actual podium",
        " · ".join(n.split()[-1] for n in race.nsmallest(3, "finish_position")["rider"]),
    )
    k3.metric("Podium riders found", f"{rm['top3_hit'] * 3:.0f} of 3")
    k4.metric("Conditions", "Wet" if race["wet"].iat[0] else "Dry")
    race = race.sort_values(col)
    race["result"] = race["finish_position"].map(
        lambda p: "Finished on the podium" if p <= 3 else "Did not finish on the podium"
    )
    fig = px.bar(
        race,
        x=col,
        y="rider",
        orientation="h",
        color="result",
        color_discrete_map={
            "Finished on the podium": PALETTE[0],
            "Did not finish on the podium": OTHER,
        },
        category_orders={"rider": race["rider"].tolist()},
        hover_data={
            "manufacturer": True,
            "grid": ":.0f",
            "finish_position": ":.0f",
            col: ":.1%",
            "rider": False,
            "result": False,
        },
        labels={col: "Predicted podium probability", "rider": "", "result": ""},
    )
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(tidy(fig, 26 * len(race) + 120), use_container_width=True)
    st.caption("Out-of-sample: this season was predicted by models trained on earlier seasons.")

# ---------------------------------------------------------------------------------------
with tab_perf:
    names = {
        "roc_auc": "Podium ROC-AUC",
        "top3_hit": "Top-3 hit rate",
        "winner_acc": "Winner accuracy",
        "spearman": "Spearman (full order)",
    }
    metric = st.radio("Metric", list(names), format_func=names.get, horizontal=True)
    per = metrics[metrics["season"] != "All"].copy()
    per["season"] = per["season"].astype(int)
    fig = px.line(
        per,
        x="season",
        y=metric,
        color="model",
        markers=True,
        color_discrete_map=MODEL_COLORS,
        labels={metric: names[metric], "season": "Test season", "model": ""},
    )
    fig.update_traces(line_width=2, marker_size=8)
    fig.update_xaxes(dtick=1)
    st.plotly_chart(tidy(fig), use_container_width=True)
    st.subheader("Crash / DNF risk model")
    st.dataframe(dnf_metrics.set_index("model").style.format("{:.3f}"), use_container_width=True)
    st.caption("PR-AUC no-skill level = DNF base rate (~0.19).")
