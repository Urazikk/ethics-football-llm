"""Tests unitaires de src/valuation.py sur des données synthétiques.
Usage : python -m tests.test_valuation
"""
import numpy as np
import pandas as pd

from src.valuation import (appraise, explain_messages, find_player, format_verdict, normalize, player_options,
                           trim_sentences)


def _row(pid, name, season, value, expected, conf="UEFA", nat="France"):
    return {"player_id": pid, "name": name, "season": season, "season_label": f"{season}/{(season + 1) % 100:02d}",
            "nationality": nat, "confederation": conf, "club_name": "Club", "league": "Ligue 1", "age": 24.0,
            "games": 30, "minutes": 2500, "goals": 12, "assists": 5, "market_value": value,
            "log_value": np.log10(value), "expected_log_value": np.log10(expected)}


df = pd.DataFrame([
    _row(1, "Kylian Mbappé", 2021, 160e6, 60e6),
    _row(1, "Kylian Mbappé", 2022, 180e6, 70e6),
    _row(2, "Gonzalo Higuaín", 2019, 25.5e6, 10e6, conf="CONMEBOL", nat="Argentina"),
    _row(3, "Ronaldo", 2015, 5e6, 5e6),
    _row(4, "Ronaldo", 2016, 6e6, 6e6),       # homonyme, autre joueur
    _row(5, "Cristiano Ronaldo", 2016, 100e6, 110e6),
    _row(6, "Lionel Messi", 2020, 80e6, 90e6),
    _row(7, "Junior Messias", 2021, 5e6, 4e6),
])
prem = pd.DataFrame({"premium_pct": [25.0, 0.0]}, index=["CONMEBOL", "UEFA"])

# normalisation : accents et majuscules ignorés
assert normalize("  MBAPPÉ ") == "mbappe"

# recherche sans accent, dernière saison par défaut
r = find_player(df, "mbappe")
assert r["player_id"] == 1 and r["season"] == 2022
assert find_player(df, "Mbappé", season=2021)["season"] == 2021
assert find_player(df, "higuain")["player_id"] == 2
# identifiant numérique accepté (utilisé par le widget)
assert find_player(df, 2)["name"] == "Gonzalo Higuaín"
assert find_player(df, np.int64(2))["name"] == "Gonzalo Higuaín"
assert find_player(df, player_options(df)["Gonzalo Higuaín"])["player_id"] == 2

# nom exact prioritaire sur une correspondance partielle, mais homonymes exacts -> erreur explicite
for bad, msg in [("Ronaldo", "plusieurs"), ("Zidane", "aucun"), ("mbappe", "saison")]:
    try:
        find_player(df, bad, season=1990 if bad == "mbappe" else None)
    except LookupError as e:
        assert msg in str(e).lower(), (bad, str(e))
    else:
        raise AssertionError(f"{bad} aurait dû lever LookupError")
assert find_player(df, "cristiano")["player_id"] == 5
# mot entier prioritaire sur une sous-chaîne : « messi » ne renvoie pas Messias
assert find_player(df, "messi")["player_id"] == 6
assert find_player(df, "messia")["player_id"] == 7

# options du widget : noms uniques, homonymes distingués
opts = player_options(df)
assert opts["Kylian Mbappé"] == 1
assert sum(1 for k in opts if k.startswith("Ronaldo")) == 2 and len(set(opts.values())) == 7

# verdicts et seuils (±20 %)
a = appraise(find_player(df, "higuain"), prem)
assert a["verdict"] == "surcoté" and round(a["gap_pct"]) == 155
assert round(a["group_premium_pct"]) == 25 and round(a["gap_excl_group_pct"]) == 104
assert appraise(find_player(df, "cristiano"), prem)["verdict"] == "juste prix"
low = pd.Series(_row(9, "X", 2020, 7e6, 10e6))
assert appraise(low, prem)["verdict"] == "sous-coté"
assert appraise(pd.Series(_row(9, "X", 2020, 11.9e6, 10e6)), prem)["verdict"] == "juste prix"
# groupe absent du tableau des primes : prime 0
assert appraise(pd.Series(_row(9, "X", 2020, 7e6, 10e6, conf="OFC")), prem)["group_premium_pct"] == 0

# affichage et prompt
txt = format_verdict(a)
assert "SURCOTÉ" in txt and "25,5 M€" in txt and "CONMEBOL" in txt
msgs = explain_messages(a)
assert msgs[0]["role"] == "system" and "surcoté" in msgs[1]["content"]
assert "plus cher que ce que ses statistiques justifient" in msgs[1]["content"]

# découpage du commentaire du LLM
assert trim_sentences("Un. Deux. Trois. Quatre.") == "Un. Deux. Trois."
assert trim_sentences("Phrase complète. Phrase coup") == "Phrase complète."
assert trim_sentences("Sans point final") == "Sans point final"
assert trim_sentences("Il vaut 5,5 M€.\nIl est cher !  Mais") == "Il vaut 5,5 M€. Il est cher !"
print("test_valuation OK")
