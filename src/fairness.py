"""Mode décision (sujet 2) : labels, mesures d'équité et réduction de biais.

Décision : « Faut-il placer cet attaquant dans la short-list premium (valeur >= seuil) ? »
- y_hist : label historique (valeur réelle >= seuil) -> contient la prime de nationalité
- y_fair : label corrigé (valeur attendue d'après la seule performance >= seuil)

Méthodes de réduction proposées :
1. Pré-traitement  : reweighing (Kamiran & Calders) ou relabeling (y_fair)
2. Entrée du modèle : masquage de la nationalité dans le prompt
3. Post-traitement : seuils par groupe pour égaliser les taux de sélection
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bias import performance_residuals


def make_decision_labels(df: pd.DataFrame, threshold_m: float = 10) -> pd.DataFrame:
    out = performance_residuals(df)
    thr = np.log10(threshold_m * 1e6)
    out["y_hist"] = (out["log_value"] >= thr).astype(int)
    out["y_fair"] = (out["expected_log_value"] >= thr).astype(int)
    return out


def reweighing(df: pd.DataFrame, group: str, label: str) -> pd.Series:
    """Poids w(g, y) = P(g) P(y) / P(g, y) : rend le label indépendant du groupe dans le train."""
    pg = df[group].value_counts(normalize=True)
    py = df[label].value_counts(normalize=True)
    pgy = df.groupby([group, label]).size() / len(df)
    w = df.apply(lambda r: pg[r[group]] * py[r[label]] / pgy[(r[group], r[label])], axis=1)
    return w.rename("weight")


def fairness_report(df: pd.DataFrame, group: str, pred: str, ref_label: str = "y_fair",
                    min_n: int = 30) -> pd.DataFrame:
    """Taux de sélection, TPR / FPR par rapport au label « juste », par groupe."""
    d = df.dropna(subset=[pred])
    rows = []
    for g, s in d.groupby(group):
        if len(s) < min_n:
            continue
        pos, neg = s[s[ref_label] == 1], s[s[ref_label] == 0]
        rows.append({group: g, "n": len(s), "selection_rate": s[pred].mean(),
                     "TPR": pos[pred].mean() if len(pos) else np.nan,
                     "FPR": neg[pred].mean() if len(neg) else np.nan})
    t = pd.DataFrame(rows).set_index(group)
    ref = t["selection_rate"].max()
    t["disparate_impact"] = t["selection_rate"] / ref   # règle des 80 % : < 0.8 = problème
    return t.round(3)


def summary_metrics(df: pd.DataFrame, group: str, pred: str, ref_label: str = "y_fair") -> dict:
    d = df.dropna(subset=[pred])
    rep = fairness_report(d, group, pred, ref_label)
    return {"accuracy_vs_" + ref_label: round((d[pred] == d[ref_label]).mean(), 3),
            "demographic_parity_diff": round(rep["selection_rate"].max() - rep["selection_rate"].min(), 3),
            "min_disparate_impact": round(rep["disparate_impact"].min(), 3),
            "equal_opportunity_diff": round(rep["TPR"].max() - rep["TPR"].min(), 3)}


def group_thresholds(scores: pd.Series, groups: pd.Series, target_rate: float) -> pd.Series:
    """Post-traitement : un seuil par groupe pour obtenir le même taux de sélection."""
    thr = scores.groupby(groups).quantile(1 - target_rate)
    return (scores >= groups.map(thr)).astype(int)


def demo_pairs(lab: pd.DataFrame, group_a: str = "CAF", group_b: str = "CONMEBOL", group: str = "confederation",
               tol_g90: float = 0.06, tol_age: float = 2.0) -> pd.DataFrame:
    """Paires de joueurs aux stats proches (même saison, même championnat) où le label historique
    diffère alors que le label « juste » est identique : cas parfaits pour la démo finale."""
    a, b = lab[lab[group] == group_a], lab[lab[group] == group_b]
    m = a.merge(b, on=["season", "league"], suffixes=("_a", "_b"))
    m = m[((m["goals_p90_a"] - m["goals_p90_b"]).abs() <= tol_g90)
          & ((m["assists_p90_a"] - m["assists_p90_b"]).abs() <= tol_g90)
          & ((m["age_a"] - m["age_b"]).abs() <= tol_age)
          & ((m["minutes_a"] - m["minutes_b"]).abs() <= 500)
          & (m["y_hist_a"] != m["y_hist_b"]) & (m["y_fair_a"] == m["y_fair_b"])]
    cols = ["season_label_a", "league"] + [f"{c}_{s}" for s in "ab" for c in
            ["name", "nationality", "club_name", "age", "games", "minutes", "goals", "assists", "market_value",
             "y_hist", "y_fair"]]
    return m[cols].sort_values("season_label_a", ascending=False)
