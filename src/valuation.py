"""Ce joueur est-il surcoté ou sous-coté ?

Le verdict compare la valeur Transfermarkt à la valeur attendue d'après la seule performance
(résidu de bias.performance_residuals). La prime moyenne de la confédération du joueur montre
quelle part de l'écart correspond au biais de nationalité mesuré en section 2.
Le LLM ne décide pas du verdict : il le commente à partir de ces chiffres.
"""
from __future__ import annotations

import numbers
import re
import unicodedata

import pandas as pd

BAND_PCT = 20   # écart au-delà duquel on parle de surcote / sous-cote


def normalize(s: str) -> str:
    """Minuscules, sans accents ni espaces superflus (« MBAPPÉ » -> « mbappe »)."""
    s = unicodedata.normalize("NFKD", str(s))
    return " ".join("".join(c for c in s if not unicodedata.combining(c)).lower().split())


def player_options(df: pd.DataFrame) -> dict[str, int]:
    """Libellés du widget -> player_id. Les homonymes sont distingués par leur dernier club."""
    last = df.sort_values("season").groupby("player_id").tail(1)
    dup = last["name"].duplicated(keep=False)
    labels = last["name"].where(~dup, last["name"] + " (" + last["club_name"].fillna("?") + ", "
                                + last["season_label"] + ")")
    return dict(sorted(zip(labels, last["player_id"]), key=lambda kv: normalize(kv[0])))


def _describe_candidates(df: pd.DataFrame, ids) -> str:
    last = df[df.player_id.isin(ids)].sort_values("season").groupby("player_id").tail(1)
    return "\n".join(f"  - {r['name']} ({r['club_name']}, {r['season_label']})" for _, r in last.iterrows())


def find_player(df: pd.DataFrame, query: str | int, season: int | None = None) -> pd.Series:
    """Ligne joueur-saison. query : nom (partiel, accents ignorés) ou player_id.
    Sans saison, prend la plus récente. Lève LookupError avec un message lisible sinon."""
    if isinstance(query, numbers.Integral) and not isinstance(query, bool):
        ids = {query} if (df.player_id == query).any() else set()
    else:
        q, names = normalize(query), df["name"].map(normalize)
        # priorité : nom exact, puis mots entiers (« messi » != « messias »), puis sous-chaîne
        word = names.map(lambda n: f" {q} " in f" {n} ")
        ids = (set(df.loc[names == q, "player_id"]) or set(df.loc[word, "player_id"])
               or set(df.loc[names.str.contains(q, regex=False), "player_id"]))
    if not ids:
        raise LookupError(f"Aucun attaquant trouvé pour « {query} ». Le dataset ne couvre que les attaquants "
                          "des 5 grands championnats (au moins 450 minutes dans la saison).")
    if len(ids) > 1:
        raise LookupError(f"Plusieurs joueurs correspondent à « {query} », précise le nom :\n"
                          + _describe_candidates(df, ids))
    rows = df[df.player_id == ids.pop()]
    if season is not None:
        if not (rows.season == season).any():
            raise LookupError(f"Pas de saison {season} pour {rows['name'].iat[0]}. Saisons disponibles : "
                              + ", ".join(rows.season_label))
        rows = rows[rows.season == season]
    return rows.sort_values("season").iloc[-1]


def appraise(row: pd.Series, premiums: pd.DataFrame, group: str = "confederation",
             band_pct: float = BAND_PCT) -> dict:
    """Valeur réelle vs valeur attendue d'après la performance, et part liée à la prime du groupe.
    row doit venir de performance_residuals (colonne expected_log_value) ; premiums de premium_by_group."""
    actual, expected = float(row["market_value"]), 10 ** float(row["expected_log_value"])
    gap = (actual / expected - 1) * 100
    g = row[group]
    prem = float(premiums["premium_pct"].get(g, 0.0))
    verdict = "surcoté" if gap > band_pct else "sous-coté" if gap < -band_pct else "juste prix"
    return {**{k: row[k] for k in ["name", "season_label", "nationality", "club_name", "league", "age",
                                   "games", "minutes", "goals", "assists"]},
            "group": g, "actual_value": actual, "expected_value": expected, "gap_pct": gap,
            "group_premium_pct": prem, "gap_excl_group_pct": (actual / (expected * (1 + prem / 100)) - 1) * 100,
            "verdict": verdict, "band_pct": band_pct}


def _m(x: float) -> str:
    return f"{x / 1e6:,.1f} M€".replace(",", " ").replace(".", ",")


def _pct(x: float) -> str:
    return f"{x:+.0f} %"


def format_verdict(a: dict) -> str:
    head = (f"{a['name']}, {a['nationality']}, {a['club_name']}, {a['league']} {a['season_label']} : "
            f"{int(a['goals'])} buts, {int(a['assists'])} passes, {int(a['minutes'])} min, {a['age']:.0f} ans")
    word = {"surcoté": "SURCOTÉ", "sous-coté": "SOUS-COTÉ", "juste prix": "AU JUSTE PRIX"}[a["verdict"]]
    lines = [head,
             f"Valeur Transfermarkt : {_m(a['actual_value'])} | valeur attendue d'après la performance : "
             f"{_m(a['expected_value'])}",
             f"→ {word} ({_pct(a['gap_pct'])}, seuil ±{a['band_pct']:.0f} %)"]
    if abs(a["group_premium_pct"]) >= 1:
        lines.append(f"  Prime moyenne des attaquants {a['group']} à performance égale : {_pct(a['group_premium_pct'])}"
                     f" → écart restant hors nationalité : {_pct(a['gap_excl_group_pct'])}")
    return "\n".join(lines)


EXPLAIN_SYSTEM = ("Tu es analyste recrutement. Rédige 2 phrases simples en français pour expliquer un verdict "
                  "déjà calculé. Reprends les faits donnés, n'ajoute aucun chiffre, ne contredis pas le verdict.")

_MEANING = {"surcoté": "plus cher que ce que ses statistiques justifient",
            "sous-coté": "moins cher que ce que ses statistiques justifient",
            "juste prix": "à un prix cohérent avec ses statistiques"}


def explain_messages(a: dict) -> list[dict]:
    """Prompt qui donne au LLM les faits et le verdict, et lui demande seulement de les expliquer.
    Faits en liste et sens du verdict explicité : un modèle de 0,5B inverse sinon souvent « surcoté »."""
    if abs(a["group_premium_pct"]) >= 5:
        part = (f"Une partie de l'écart vient de sa nationalité : à performance égale, les attaquants {a['group']} "
                f"valent en moyenne {_pct(a['group_premium_pct'])}.")
    else:
        part = "Sa nationalité n'explique presque rien de l'écart."
    user = (f"Faits :\n- {a['name']}, {a['age']:.0f} ans, {int(a['goals'])} buts et {int(a['assists'])} passes "
            f"décisives en {int(a['minutes'])} minutes ({a['league']}, {a['season_label']}).\n"
            f"- Valeur Transfermarkt {_m(a['actual_value'])}, valeur attendue d'après ses statistiques "
            f"{_m(a['expected_value'])}.\n"
            f"- Verdict : {a['verdict']}, c'est-à-dire {_MEANING[a['verdict']]}.\n- {part}\nExplication :")
    return [{"role": "system", "content": EXPLAIN_SYSTEM}, {"role": "user", "content": user}]


def trim_sentences(text: str, n: int = 3) -> str:
    """Garde au plus n phrases complètes (la génération s'arrête souvent au milieu d'une phrase)."""
    text = " ".join(text.split())
    if text and text[-1] not in ".!?" and re.search(r"[.!?]", text):
        text = text[:max(text.rfind(c) for c in ".!?") + 1]
    return " ".join(re.split(r"(?<=[.!?])\s+", text)[:n])


def explain(tok, model, a: dict, max_new_tokens: int = 90) -> str:
    """Commentaire du LLM sur le verdict (déterministe)."""
    return trim_sentences(generate_text(tok, model, explain_messages(a), max_new_tokens))


def generate_text(tok, model, messages: list[dict], max_new_tokens: int = 120) -> str:
    """Génération gloutonne (déterministe) pour un seul prompt."""
    import torch
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
