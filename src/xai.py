"""Explicabilité des décisions du LLM avec LIME et SHAP.

Le LLM prend du texte en entrée. Pour appliquer LIME / SHAP (méthodes tabulaires), on :
1. encode le profil du joueur en vecteur (numériques + catégorielles codées en entiers),
2. perturbe ce vecteur (LIME : voisinage, SHAP : coalitions de variables),
3. reconstruit un prompt pour chaque perturbation et interroge le LLM,
4. explique la sortie (log10 de la valeur, ou probabilité de décision).

Les variables non perturbées (club, etc.) restent celles du joueur expliqué.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

XAI_NUM = ["age", "games", "minutes", "goals", "assists"]
XAI_CAT = ["nationality", "league"]


class TabularEncoder:
    def __init__(self, df: pd.DataFrame, num=XAI_NUM, cat=XAI_CAT, top_cats: int = 25):
        self.num, self.cat = list(num), list(cat)
        self.features = self.num + self.cat
        self.categories = {c: list(df[c].value_counts().head(top_cats).index) for c in self.cat}

    def ensure(self, col: str, value: str):
        """Ajoute une modalité (ex. 'Morocco') si elle n'est pas dans le top."""
        if value not in self.categories[col]:
            self.categories[col].append(value)

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.num].astype(float).copy()
        for c in self.cat:
            X[c] = df[c].map({v: i for i, v in enumerate(self.categories[c])}).fillna(0)
        return X[self.features].values.astype(float)

    def decode(self, X: np.ndarray, base_row: dict) -> list[dict]:
        rows = []
        for x in np.atleast_2d(X):
            r = dict(base_row)
            for j, f in enumerate(self.features):
                if f in self.cat:
                    r[f] = self.categories[f][int(round(np.clip(x[j], 0, len(self.categories[f]) - 1)))]
                else:
                    r[f] = max(0.0, float(x[j]))
            r["games"] = max(1, round(r["games"]))
            r["minutes"] = max(r["games"], round(r["minutes"]))
            r["goals"], r["assists"] = round(r["goals"]), round(r["assists"])
            r["goals_p90"] = r["goals"] / r["minutes"] * 90
            r["assists_p90"] = r["assists"] / r["minutes"] * 90
            rows.append(r)
        return rows


def make_predict_fn(predictor, encoder: TabularEncoder, base_row: dict, task: str = "value"):
    """Fonction X (n, p) -> sortie numérique, utilisable par LIME et SHAP.

    task='value'    : renvoie log10(valeur en euros) (échelle additive, plus lisible)
    task='decision' : renvoie P(YES) si le prédicteur a predict_proba, sinon la décision 0/1
    """
    cache: dict[tuple, float] = {}

    def f(X):
        X = np.atleast_2d(X)
        rows = encoder.decode(X, base_row)
        keys = [tuple((k, str(r[k])) for k in encoder.features) for r in rows]
        todo = [i for i, k in enumerate(keys) if k not in cache]
        if todo:
            sub = [rows[i] for i in todo]
            if task == "decision" and hasattr(predictor, "predict_proba"):
                preds = predictor.predict_proba(sub)
            else:
                preds = predictor.predict(sub)
            for i, p in zip(todo, preds):
                if task == "value":
                    cache[keys[i]] = np.log10(p) if np.isfinite(p) and p > 0 else np.nan
                else:
                    cache[keys[i]] = p
        out = np.array([cache[k] for k in keys], dtype=float)
        # Réponses non parsables : on remplace par la médiane (sinon LIME/SHAP plantent)
        if np.isnan(out).any():
            out[np.isnan(out)] = np.nanmedian(out) if np.isfinite(out).any() else 0.0
        return out

    return f


def explain_lime(predictor, encoder: TabularEncoder, train: pd.DataFrame, row: dict,
                 num_samples: int = 300, task: str = "value", seed: int = 0):
    from lime.lime_tabular import LimeTabularExplainer

    for c in encoder.cat:
        encoder.ensure(c, row[c])
    Xtr = encoder.encode(train)
    cat_idx = [encoder.features.index(c) for c in encoder.cat]
    explainer = LimeTabularExplainer(
        Xtr, feature_names=encoder.features, categorical_features=cat_idx,
        categorical_names={i: encoder.categories[c] for i, c in zip(cat_idx, encoder.cat)},
        mode="regression", discretize_continuous=True, random_state=seed)
    f = make_predict_fn(predictor, encoder, row, task)
    x = encoder.encode(pd.DataFrame([row]))[0]
    exp = explainer.explain_instance(x, f, num_features=len(encoder.features), num_samples=num_samples)
    return exp


def explain_shap(predictor, encoder: TabularEncoder, train: pd.DataFrame, rows: list[dict],
                 n_background: int = 10, nsamples: int = 100, task: str = "value", seed: int = 0):
    """KernelSHAP (agnostique au modèle). Le coût est ~ n_background x nsamples appels LLM par joueur."""
    import shap

    for r in rows:
        for c in encoder.cat:
            encoder.ensure(c, r[c])
    bg = encoder.encode(train.sample(n_background, random_state=seed))
    values = []
    for r in rows:
        f = make_predict_fn(predictor, encoder, r, task)
        ex = shap.KernelExplainer(f, bg)
        x = encoder.encode(pd.DataFrame([r]))
        sv = ex.shap_values(x, nsamples=nsamples, silent=True)
        values.append(shap.Explanation(values=np.asarray(sv)[0], base_values=ex.expected_value,
                                       data=np.array([r[c] for c in encoder.features], dtype=object),
                                       feature_names=encoder.features))
    return values


def counterfactual_nationality(predictor, row: dict, nationalities: list[str], task: str = "value") -> pd.DataFrame:
    """Même joueur, seule la nationalité change. Test de biais direct et peu coûteux."""
    rows = [{**row, "nationality": n} for n in nationalities]
    preds = predictor.predict(rows)
    out = pd.DataFrame({"nationality": nationalities, "prediction": preds})
    if task == "value":
        ref = out["prediction"].iloc[0]
        out["prediction_M_EUR"] = (out["prediction"] / 1e6).round(2)
        out["vs_first_pct"] = ((out["prediction"] / ref - 1) * 100).round(1)
    return out


def surrogate_shap(X_df: pd.DataFrame, y_llm: np.ndarray, seed: int = 0):
    """Vue globale bon marché : un GBM imite le LLM sur le jeu de test, puis TreeSHAP l'explique.
    Le R² du substitut indique la fidélité de l'explication."""
    import shap
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    X = X_df.copy()
    for c in X.select_dtypes("object").columns:
        X[c] = X[c].astype("category").cat.codes
    ok = np.isfinite(y_llm)
    gbm = GradientBoostingRegressor(random_state=seed, max_depth=3, n_estimators=300)
    fidelity = cross_val_score(gbm, X[ok], y_llm[ok], cv=5, scoring="r2").mean()
    gbm.fit(X[ok], y_llm[ok])
    sv = shap.TreeExplainer(gbm)(X[ok])
    return gbm, sv, round(fidelity, 3)
