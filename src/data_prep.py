"""Construction du jeu de données joueur-saison à partir du dataset Transfermarkt (Kaggle).

Une ligne = un joueur sur une saison (juillet -> juin) avec :
- ses statistiques agrégées (matchs, minutes, buts, passes, cartons),
- son profil (nationalité, âge, poste, taille, pied, club, championnat),
- sa valeur marchande à la fin de la saison (cible).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Grands championnats européens (codes Transfermarkt)
BIG5 = {"GB1": "Premier League", "ES1": "LaLiga", "IT1": "Serie A", "L1": "Bundesliga", "FR1": "Ligue 1"}


def load_raw(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Charge les tables utiles du dataset."""
    d = Path(data_dir).expanduser()
    players = pd.read_csv(d / "players.csv", parse_dates=["date_of_birth"])
    apps = pd.read_csv(
        d / "appearances.csv",
        usecols=["player_id", "player_club_id", "date", "competition_id",
                 "yellow_cards", "red_cards", "goals", "assists", "minutes_played"],
        parse_dates=["date"],
    )
    vals = pd.read_csv(d / "player_valuations.csv", parse_dates=["date"])
    clubs = pd.read_csv(d / "clubs.csv", usecols=["club_id", "name", "domestic_competition_id"])
    return {"players": players, "appearances": apps, "valuations": vals, "clubs": clubs}


def season_of(dates: pd.Series) -> pd.Series:
    """Saison de début : un match d'octobre 2023 appartient à la saison 2023 (2023/24)."""
    return np.where(dates.dt.month >= 7, dates.dt.year, dates.dt.year - 1)


def build_player_seasons(raw: dict[str, pd.DataFrame], positions=("Attack",),
                         leagues: dict | None = None, min_minutes: int = 450) -> pd.DataFrame:
    """Table joueur-saison filtrée (par défaut : attaquants du Big 5, >= 450 minutes)."""
    leagues = BIG5 if leagues is None else leagues
    players, apps, vals, clubs = raw["players"], raw["appearances"], raw["valuations"], raw["clubs"]

    apps = apps.copy()
    apps["season"] = season_of(apps["date"])
    # Uniquement les matchs du championnat national (comparabilité entre joueurs)
    apps = apps[apps["competition_id"].isin(leagues)]

    agg = (apps.groupby(["player_id", "season"])
           .agg(games=("date", "count"), minutes=("minutes_played", "sum"),
                goals=("goals", "sum"), assists=("assists", "sum"),
                yellow_cards=("yellow_cards", "sum"), red_cards=("red_cards", "sum"),
                club_id=("player_club_id", lambda s: s.mode().iat[0]),
                league_id=("competition_id", lambda s: s.mode().iat[0]))
           .reset_index())
    agg = agg[agg["minutes"] >= min_minutes]

    # Valeur marchande : dernière estimation connue au 31 juillet suivant la fin de saison
    agg["ref_date"] = pd.to_datetime((agg["season"] + 1).astype(str) + "-07-31")
    agg = agg.sort_values("ref_date")
    v = vals[["player_id", "date", "market_value_in_eur"]].sort_values("date")
    agg = pd.merge_asof(agg, v, left_on="ref_date", right_on="date", by="player_id",
                        direction="backward", tolerance=pd.Timedelta(days=270))
    agg = agg.dropna(subset=["market_value_in_eur"]).drop(columns="date")

    keep = ["player_id", "name", "country_of_citizenship", "country_of_birth", "date_of_birth",
            "position", "sub_position", "foot", "height_in_cm"]
    df = agg.merge(players[keep], on="player_id", how="left")
    df = df[df["position"].isin(positions)]
    df = df.merge(clubs.rename(columns={"name": "club_name"})[["club_id", "club_name"]], on="club_id", how="left")

    df["age"] = ((df["ref_date"] - pd.DateOffset(months=1)) - df["date_of_birth"]).dt.days / 365.25
    df["age"] = df["age"].round(1)
    df["league"] = df["league_id"].map(leagues)
    df["goals_p90"] = (df["goals"] / df["minutes"] * 90).round(3)
    df["assists_p90"] = (df["assists"] / df["minutes"] * 90).round(3)
    df["log_value"] = np.log10(df["market_value_in_eur"])
    df["season_label"] = df["season"].astype(str) + "/" + ((df["season"] + 1) % 100).astype(str).str.zfill(2)
    df = df.rename(columns={"country_of_citizenship": "nationality", "market_value_in_eur": "market_value"})
    df = df.dropna(subset=["nationality", "age"])
    return df.sort_values(["season", "player_id"]).reset_index(drop=True)


def add_nationality_groups(df: pd.DataFrame, top_n: int = 15, col: str = "nationality") -> pd.DataFrame:
    """Regroupe les nationalités rares en 'Other' + ajoute une confédération simplifiée."""
    df = df.copy()
    top = df[col].value_counts().head(top_n).index
    df["nat_group"] = np.where(df[col].isin(top), df[col], "Other")
    df["confederation"] = df[col].map(CONFEDERATION).fillna("Other")
    return df


# Confédération des principales nationalités (suffisant pour l'analyse de biais)
_UEFA = ["France", "Spain", "Italy", "Germany", "England", "Portugal", "Netherlands", "Belgium", "Croatia",
         "Serbia", "Switzerland", "Austria", "Denmark", "Sweden", "Norway", "Poland", "Scotland", "Wales",
         "Ukraine", "Russia", "Turkey", "Greece", "Czech Republic", "Slovakia", "Slovenia", "Bosnia-Herzegovina",
         "Montenegro", "Albania", "North Macedonia", "Kosovo", "Georgia", "Armenia", "Hungary", "Romania",
         "Bulgaria", "Ireland", "Northern Ireland", "Iceland", "Finland", "Israel", "Luxembourg"]
_CONMEBOL = ["Brazil", "Argentina", "Uruguay", "Colombia", "Chile", "Paraguay", "Ecuador", "Peru",
             "Venezuela", "Bolivia"]
_CAF = ["Morocco", "Algeria", "Tunisia", "Egypt", "Senegal", "Nigeria", "Ghana", "Cote d'Ivoire",
        "Cameroon", "Mali", "Guinea", "Burkina Faso", "DR Congo", "Congo", "Gabon", "Gambia", "Zambia",
        "South Africa", "Angola", "Cape Verde", "Togo", "Benin", "Zimbabwe", "Kenya", "Guinea-Bissau",
        "Equatorial Guinea", "Comoros", "Madagascar", "Sierra Leone", "Liberia", "Mauritania"]
_CONCACAF = ["United States", "Mexico", "Canada", "Jamaica", "Costa Rica", "Honduras", "Panama", "Haiti",
             "Curacao", "Suriname", "Martinique", "Guadeloupe"]
_AFC = ["Japan", "Korea, South", "Australia", "Iran", "Saudi Arabia", "China", "Uzbekistan", "Iraq", "Qatar"]
CONFEDERATION = {**{c: "UEFA" for c in _UEFA}, **{c: "CONMEBOL" for c in _CONMEBOL},
                 **{c: "CAF" for c in _CAF}, **{c: "CONCACAF" for c in _CONCACAF},
                 **{c: "AFC" for c in _AFC}}


def temporal_split(df: pd.DataFrame, test_season: int | None = None):
    """Train = saisons passées, test = dernière saison complète (évite la fuite temporelle)."""
    test_season = int(df["season"].max()) if test_season is None else test_season
    return df[df["season"] < test_season].copy(), df[df["season"] == test_season].copy()
