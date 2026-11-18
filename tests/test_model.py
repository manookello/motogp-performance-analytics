import pandas as pd
import pytest

from motogp_analytics.model import dnf_metrics, race_ranking_metrics, relevance


def _race(scores):
    return pd.DataFrame(
        {
            "race_id": "2024-01",
            "rider_id": list("abcde"),
            "finish_position": [1, 2, 3, 4, 5],
            "score": scores,
        }
    )


def test_perfect_ranking():
    m = race_ranking_metrics(_race([5, 4, 3, 2, 1]), "score")
    assert m == {"top3_hit": 1.0, "winner_hit": 1.0, "spearman": pytest.approx(1.0)}


def test_relevance_labels():
    assert relevance(pd.Series([1, 20, 21, None])).tolist() == [20, 1, 0, 0]


def test_dnf_metrics_baselines():
    pred = pd.DataFrame(
        {
            "dnf": [1, 0, 0, 1, 0, 0, 0, 0],
            "base_rate": [0.25] * 8,
            "rider_rate": [0.5, 0.1, 0.1, 0.6, 0.2, 0.1, 0.1, 0.1],
            "p_dnf": [0.9, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1],
        }
    )
    m = dnf_metrics(pred).set_index("model")
    assert m.loc["LightGBM", "pr_auc"] == pytest.approx(1.0)
    assert m.loc["Base rate", "roc_auc"] == 0.5
