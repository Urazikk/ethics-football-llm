"""Conversion des lignes joueur-saison en texte (prompts) pour le LLM, et parsing des réponses."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

SYSTEM_PROMPT = (
    "You are a football scouting analyst. Given a forward's season profile, estimate his "
    "Transfermarkt market value at the end of the season. "
    "Answer on the first line with 'Market value: <number> M EUR', then one short sentence of justification."
)

# Variables exposées au LLM (et donc expliquées par LIME / SHAP)
PROMPT_FEATURES = ["age", "nationality", "league", "club_name", "games", "minutes", "goals", "assists"]


def describe_player(row: pd.Series | dict) -> str:
    """Profil textuel d'un joueur. Toute variable absente du dict est simplement omise."""
    r = dict(row)
    parts = []
    if "age" in r:
        parts.append(f"{int(r['age'])}-year-old")
    if "nationality" in r and isinstance(r["nationality"], str):
        parts.append(f"{r['nationality']}")
    parts.append("forward")
    txt = " ".join(parts)
    if "club_name" in r and isinstance(r["club_name"], str):
        txt += f" playing for {r['club_name']}"
    if "league" in r and isinstance(r["league"], str):
        txt += f" ({r['league']})"
    stats = []
    if "games" in r:
        stats.append(f"{int(r['games'])} games")
    if "minutes" in r:
        stats.append(f"{int(r['minutes'])} minutes")
    if "goals" in r:
        stats.append(f"{int(r['goals'])} goals")
    if "assists" in r:
        stats.append(f"{int(r['assists'])} assists")
    if stats:
        txt += ". This season: " + ", ".join(stats) + "."
    return txt


def format_value(value_eur: float) -> str:
    return f"{value_eur / 1e6:.1f}"


def answer_text(row: pd.Series) -> str:
    """Réponse cible (utilisée pour le few-shot, le RAG et le fine-tuning)."""
    return f"Market value: {format_value(row['market_value'])} M EUR"


def user_prompt(row, examples: list[tuple[str, str]] | None = None) -> str:
    """Prompt utilisateur, avec éventuellement des exemples (few-shot ou RAG)."""
    msg = ""
    if examples:
        msg += "Here are comparable forwards with their known market values:\n"
        for desc, ans in examples:
            msg += f"- {desc} -> {ans}\n"
        msg += "\n"
    msg += f"Player: {describe_player(row)}\nWhat is his market value?"
    return msg


def chat_messages(row, examples=None, with_answer: bool = False, answer: str | None = None):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(row, examples)}]
    if with_answer:
        msgs.append({"role": "assistant", "content": answer if answer is not None else answer_text(row)})
    return msgs


_NUM = re.compile(r"([0-9]+(?:[.,][0-9]+)?)\s*(m|M|million|Mio|k|K|thousand)?")


def parse_value(text: str) -> float:
    """Extrait une valeur en euros de la réponse du LLM. Renvoie NaN si rien d'exploitable."""
    if not isinstance(text, str):
        return np.nan
    m = re.search(r"[Mm]arket value[^0-9]*" + _NUM.pattern, text)
    m = m or _NUM.search(text)
    if not m:
        return np.nan
    x = float(m.group(1).replace(",", "."))
    unit = (m.group(2) or "").lower()
    if unit in ("k", "thousand"):
        return x * 1e3
    if unit in ("m", "million", "mio") or x < 1000:  # le format demandé est en millions
        return x * 1e6
    return x


# ----------------------------------------------------------------------------- mode décision (sujet 2)

DECISION_SYSTEM_PROMPT = (
    "You are the recruitment assistant of a football club. Given a forward's season profile, decide whether "
    "he should be shortlisted as a PREMIUM target (expected market value of at least {thr} M EUR). "
    "Answer on the first line with 'Decision: YES' or 'Decision: NO', then one short sentence of justification."
)


def decision_messages(row, threshold_m: float = 10, examples=None, with_answer: bool = False,
                      answer: str | None = None, hide: tuple[str, ...] = ()):
    """Prompt de décision. hide=('nationality',) masque la variable sensible (fairness through unawareness)."""
    r = {k: v for k, v in dict(row).items() if k not in hide}
    msg = ""
    if examples:
        msg += "Past decisions on comparable forwards:\n" + "".join(f"- {d} -> {a}\n" for d, a in examples) + "\n"
    msg += f"Player: {describe_player(r)}\nShould he be shortlisted?"
    msgs = [{"role": "system", "content": DECISION_SYSTEM_PROMPT.format(thr=threshold_m)},
            {"role": "user", "content": msg}]
    if with_answer:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


def decision_answer(label: int) -> str:
    return "Decision: YES" if int(label) == 1 else "Decision: NO"


def parse_decision(text: str) -> float:
    if not isinstance(text, str):
        return np.nan
    t = text.lower()
    if "decision: yes" in t or t.strip().startswith("yes"):
        return 1.0
    if "decision: no" in t or t.strip().startswith("no"):
        return 0.0
    return np.nan
