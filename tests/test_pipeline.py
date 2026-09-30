"""Test de bout en bout sur le vrai dataset, avec un LLM factice.
Usage : python -m tests.test_pipeline /chemin/vers/archive
"""
import sys

import numpy as np
import pandas as pd

from src.bias import matched_pairs, premium_by_group, performance_residuals
from src.data_prep import add_nationality_groups, build_player_seasons, load_raw, temporal_split
from src.fairness import fairness_report, make_decision_labels, reweighing, summary_metrics
from src.llm_methods import Retriever, pick_few_shot, regression_metrics
from src.prompts import chat_messages, decision_messages, parse_decision, parse_value
from src.xai import TabularEncoder, counterfactual_nationality, explain_lime, explain_shap, surrogate_shap
from tests.mock_llm import MockPredictor

data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
df = add_nationality_groups(build_player_seasons(load_raw(data_dir)))
train, test = temporal_split(df)
print("rows", len(df), "train", len(train), "test", len(test))

# parsing
assert abs(parse_value("Market value: 12.5 M EUR. Because...") - 12.5e6) < 1
assert abs(parse_value("around 800k") - 8e5) < 1
assert parse_decision("Decision: YES - strong") == 1 and parse_decision("Decision: NO") == 0
# prompts
print(chat_messages(test.iloc[0], pick_few_shot(train, 3))[1]["content"][:400])
r = Retriever(train, k=3)
print(r.examples_for(test.iloc[0].to_dict()))

# biais
print(premium_by_group(performance_residuals(df), "confederation"))

# mode valeur
mp = MockPredictor(train)
print(regression_metrics(test["market_value"], mp.predict(test)))
row = test.iloc[0].to_dict()
print(counterfactual_nationality(mp, row, ["France", "Brazil", "Morocco"]))
enc = TabularEncoder(train)
exp = explain_lime(mp, enc, train, row, num_samples=200)
print("LIME", exp.as_list()[:5])
sv = explain_shap(mp, enc, train, [row], n_background=5, nsamples=60)
print("SHAP", dict(zip(sv[0].feature_names, np.round(sv[0].values, 3))))
_, _, fid = surrogate_shap(test[enc.features], np.log10(mp.predict(test)))
print("surrogate fidelity", fid)

# mode décision
lab = make_decision_labels(df, 10, fit_on=df.season < df.season.max())
tr, te = temporal_split(lab)
# y_fair ajusté sur le train seulement : la saison de test ne doit pas influencer la régression
full = make_decision_labels(df, 10)
assert not np.allclose(lab.loc[te.index, "expected_log_value"], full.loc[te.index, "expected_log_value"])
assert np.isfinite(lab["expected_log_value"]).all()
tr = tr.assign(weight=reweighing(tr, "confederation", "y_fair"))
biased = MockPredictor(tr, "decision", "y_hist")
blind = MockPredictor(tr.assign(), "decision", "y_fair", hide=("nationality",))
te = te.assign(pred_biased=biased.predict(te), pred_fair=blind.predict(te))
print(fairness_report(te, "confederation", "pred_biased"))
print("biased", summary_metrics(te, "confederation", "pred_biased"))
print("fair  ", summary_metrics(te, "confederation", "pred_fair"))
print(decision_messages(row, 10, hide=("nationality",))[1]["content"])
print(matched_pairs(df, "Morocco", "Brazil").head(3))
print("OK")
