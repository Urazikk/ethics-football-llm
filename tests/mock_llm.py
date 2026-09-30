"""LLM factice (GBM) pour tester le pipeline sans GPU ni téléchargement de modèle."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


class MockPredictor:
    FEATS = ["age", "games", "minutes", "goals", "assists"]

    def __init__(self, train, task="value", label_col=None, hide=()):
        self.task, self.hide = task, hide
        self.cats = {c: sorted(train[c].unique()) for c in ["nationality", "league"] if c not in hide}
        X = self._X(train)
        if task == "value":
            self.m = GradientBoostingRegressor(random_state=0).fit(X, train["log_value"])
        else:
            self.m = GradientBoostingClassifier(random_state=0).fit(X, train[label_col])

    def _X(self, rows):
        d = pd.DataFrame(rows)
        X = d[self.FEATS].astype(float)
        for c, vals in self.cats.items():
            X[c] = d[c].map({v: i for i, v in enumerate(vals)}).fillna(-1)
        return X

    def predict(self, rows):
        X = self._X(rows.to_dict("records") if isinstance(rows, pd.DataFrame) else rows)
        if self.task == "value":
            return 10 ** self.m.predict(X)
        return self.m.predict(X).astype(float)

    def predict_proba(self, rows):
        X = self._X(rows.to_dict("records") if isinstance(rows, pd.DataFrame) else rows)
        return self.m.predict_proba(X)[:, 1]
