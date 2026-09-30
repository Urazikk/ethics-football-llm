"""Mise en évidence des biais de valorisation (nationalité, confédération, championnat).

Idée : on explique la valeur (log10) par les seules variables sportives. Le résidu mesure
la « prime » ou la « décote » inexpliquée par la performance. Si ce résidu varie
systématiquement selon la nationalité, on tient un biais.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

SPORT_FEATURES = ["goals_p90", "assists_p90", "minutes", "games", "age", "age2"]


def _design(df: pd.DataFrame, extra_cat: list[str] | None = None) -> pd.DataFrame:
    X = df.assign(age2=df["age"] ** 2)[SPORT_FEATURES].copy()
    cats = ["league", "season"] + (extra_cat or [])
    X = X.join(pd.get_dummies(df[cats].astype(str), drop_first=True, dtype=float))
    return X


def performance_residuals(df: pd.DataFrame, control_club: bool = False) -> pd.DataFrame:
    """Ajoute 'expected_log_value' (performance seule) et 'residual' (prime inexpliquée).

    control_club=True ajoute un effet fixe club : on compare alors des joueurs d'un même club,
    ce qui retire la part de la prime due au fait de jouer dans un grand club (facteur confondant).
    """
    X = _design(df, extra_cat=["club_id"] if control_club else None)
    reg = LinearRegression().fit(X, df["log_value"])
    out = df.copy()
    out["expected_log_value"] = reg.predict(X)
    out["residual"] = out["log_value"] - out["expected_log_value"]
    # Multiplicateur de valeur : 10**résidu (1.3 = +30 % à performance égale)
    out["value_multiplier"] = 10 ** out["residual"]
    return out


def premium_by_group(df_res: pd.DataFrame, group: str, min_n: int = 30) -> pd.DataFrame:
    """Prime moyenne à performance égale, par groupe, avec intervalle de confiance à 95 %."""
    g = df_res.groupby(group)["residual"].agg(["count", "mean", "std"])
    g = g[g["count"] >= min_n].copy()
    g["ci95"] = 1.96 * g["std"] / np.sqrt(g["count"])
    g["premium_pct"] = (10 ** g["mean"] - 1) * 100
    g["premium_low_pct"] = (10 ** (g["mean"] - g["ci95"]) - 1) * 100
    g["premium_high_pct"] = (10 ** (g["mean"] + g["ci95"]) - 1) * 100
    return g.sort_values("mean", ascending=False).round(3)


def raw_vs_performance(df: pd.DataFrame, group: str, min_n: int = 30, control_club: bool = False) -> pd.DataFrame:
    """Compare valeur médiane, buts/90 et prime inexpliquée par groupe (tableau pour les slides)."""
    res = performance_residuals(df, control_club=control_club)
    t = res.groupby(group).agg(n=("player_id", "count"),
                               median_value_m=("market_value", lambda s: s.median() / 1e6),
                               goals_p90=("goals_p90", "mean"),
                               assists_p90=("assists_p90", "mean"),
                               residual=("residual", "mean"))
    t = t[t["n"] >= min_n]
    t["premium_pct"] = (10 ** t["residual"] - 1) * 100
    return t.sort_values("premium_pct", ascending=False).round(2)


def nationality_effect(df: pd.DataFrame, group: str = "nat_group", ref: str | None = None) -> pd.Series:
    """Effet marginal de la nationalité dans un modèle avec contrôles sportifs (en % de valeur)."""
    X = _design(df)
    dummies = pd.get_dummies(df[group], dtype=float)
    ref = ref or df[group].value_counts().index[0]
    dummies = dummies.drop(columns=ref)
    reg = LinearRegression().fit(X.join(dummies), df["log_value"])
    coefs = pd.Series(reg.coef_[-dummies.shape[1]:], index=dummies.columns)
    return ((10 ** coefs - 1) * 100).sort_values(ascending=False).round(1).rename(f"effet_vs_{ref}_pct")


def matched_pairs(df: pd.DataFrame, group_a: str, group_b: str, group: str = "nationality",
                  tol_goals: float = 0.05, tol_age: float = 1.5) -> pd.DataFrame:
    """Paires de joueurs aux stats quasi identiques (même saison, même championnat) mais
    de groupes différents. Utile pour la démo finale."""
    a = df[df[group] == group_a]
    b = df[df[group] == group_b]
    m = a.merge(b, on=["season", "league"], suffixes=("_a", "_b"))
    m = m[(m["goals_p90_a"] - m["goals_p90_b"]).abs() <= tol_goals]
    m = m[(m["age_a"] - m["age_b"]).abs() <= tol_age]
    m = m[(m["minutes_a"] - m["minutes_b"]).abs() <= 600]
    m["value_ratio"] = m["market_value_b"] / m["market_value_a"]
    cols = ["season_label_a", "league", "name_a", "age_a", "goals_a", "assists_a", "minutes_a", "market_value_a",
            "name_b", "age_b", "goals_b", "assists_b", "minutes_b", "market_value_b", "value_ratio"]
    return m[cols].sort_values("value_ratio", ascending=False)
