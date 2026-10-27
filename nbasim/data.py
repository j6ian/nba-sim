"""Data layer: pulls stats from stats.nba.com via nba_api and caches them on disk.

Everything downstream works off a single `Dataset` object built here:
  - players:  blended per-player season totals (current season + down-weighted prior season)
  - teams:    blended per-team rates (pace, ratings, four factors, zone defense, ...)
  - logs:     per-game player logs for both seasons (for minutes trends and hit rates)
  - roster:   current team for every active player
"""
from __future__ import annotations

import datetime as dt
import os
import pickle
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")

# Shot zones used everywhere in the model.
ZONES = ["RA", "PAINT", "MID", "C3", "ATB3"]
ZONE_LABELS = {
    "RA": "Restricted Area",
    "PAINT": "Paint (non-RA)",
    "MID": "Mid-Range",
    "C3": "Corner 3",
    "ATB3": "Above-Break 3",
}
_NBA_ZONE = {
    "RA": "Restricted Area",
    "PAINT": "In The Paint (Non-RA)",
    "MID": "Mid-Range",
    "C3": "Corner 3",
    "ATB3": "Above the Break 3",
}

BOX_COLS = ["GP", "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB",
            "REB", "AST", "TOV", "STL", "BLK", "PF", "PFD", "PTS"]
LOG_COLS = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP",
            "WL", "MIN", "PTS", "REB", "AST", "FG3M", "FG3A", "FGA", "FGM", "FTA", "FTM",
            "STL", "BLK", "TOV", "OREB", "DREB"]


def current_season(today: dt.date | None = None) -> str:
    """NBA season string for a date, e.g. 2026-10-23 -> '2026-27'. New season from October on."""
    today = today or dt.date.today()
    start = today.year if today.month >= 10 else today.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def previous_season(season: str) -> str:
    start = int(season[:4]) - 1
    return f"{start}-{str(start + 1)[-2:]}"


# --------------------------------------------------------------------------- fetching

_PINNED: dict = {}


def _pin_dns(host: str = "stats.nba.com") -> bool:
    """Fallback when the system resolver fails for a host but DNS itself works: look the
    address up with `dig` and route Python's lookups for that host to it."""
    import socket
    import subprocess
    if host in _PINNED:
        return True
    ips = []
    # Local resolver first, then public DNS (some routers return the CDN alias but no address).
    for server in ([], ["@1.1.1.1"], ["@8.8.8.8"]):
        try:
            out = subprocess.run(["dig", "+short", *server, host], capture_output=True, text=True,
                                 timeout=10).stdout
        except Exception:
            continue
        ips = [ln for ln in out.split() if ln.count(".") == 3 and ln.replace(".", "").isdigit()]
        if ips:
            break
    if not ips:
        return False
    _PINNED[host] = ips[0]
    orig = socket.getaddrinfo

    def patched(h, *args, **kwargs):
        return orig(_PINNED.get(h, h), *args, **kwargs)

    socket.getaddrinfo = patched
    return True


def _call(endpoint_cls, retries: int = 4, **kwargs) -> list[pd.DataFrame]:
    last = None
    for attempt in range(retries):
        try:
            frames = endpoint_cls(timeout=60, **kwargs).get_data_frames()
            time.sleep(0.6)  # be polite to stats.nba.com
            return frames
        except Exception as exc:  # network hiccups / throttling are common
            last = exc
            if "resolve" in str(exc).lower():
                _pin_dns()
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"NBA stats request failed ({endpoint_cls.__name__}): {last}")


def _flatten_zone_frame(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Shot-location endpoints return two-level columns; flatten to RA_FGA, MID_FG_PCT, ..."""
    out = pd.DataFrame(index=df.index)
    for top, sub in df.columns:
        top, sub = str(top), str(sub)
        if top == "":
            out[sub] = df[(top, sub)].values
            continue
        for key, nba_name in _NBA_ZONE.items():
            if top == nba_name:
                out[f"{key}_{sub.replace(prefix, '')}"] = df[(top, sub)].values
    return out


def fetch_season(season: str) -> dict[str, pd.DataFrame]:
    from nba_api.stats.endpoints import (leaguedashplayershotlocations, leaguedashplayerstats,
                                         leaguedashteamshotlocations, leaguedashteamstats,
                                         playergamelogs)

    out: dict[str, pd.DataFrame] = {}
    out["p_base"] = _call(leaguedashplayerstats.LeagueDashPlayerStats, season=season,
                          per_mode_detailed="Totals")[0]
    out["p_adv"] = _call(leaguedashplayerstats.LeagueDashPlayerStats, season=season,
                         per_mode_detailed="Totals",
                         measure_type_detailed_defense="Advanced")[0]
    out["p_zone"] = _flatten_zone_frame(
        _call(leaguedashplayershotlocations.LeagueDashPlayerShotLocations, season=season,
              distance_range="By Zone")[0])
    out["t_base"] = _call(leaguedashteamstats.LeagueDashTeamStats, season=season,
                          per_mode_detailed="Totals")[0]
    out["t_adv"] = _call(leaguedashteamstats.LeagueDashTeamStats, season=season,
                         measure_type_detailed_defense="Advanced")[0]
    out["t_ff"] = _call(leaguedashteamstats.LeagueDashTeamStats, season=season,
                        measure_type_detailed_defense="Four Factors")[0]
    out["t_opp"] = _call(leaguedashteamstats.LeagueDashTeamStats, season=season,
                         per_mode_detailed="Totals",
                         measure_type_detailed_defense="Opponent")[0]
    out["t_zone_opp"] = _flatten_zone_frame(
        _call(leaguedashteamshotlocations.LeagueDashTeamShotLocations, season=season,
              distance_range="By Zone", measure_type_simple="Opponent")[0], prefix="OPP_")
    logs = _call(playergamelogs.PlayerGameLogs, season_nullable=season,
                 season_type_nullable="Regular Season")[0]
    out["logs"] = logs[[c for c in LOG_COLS if c in logs.columns]].copy()
    return out


def fetch_roster(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import commonallplayers, playerindex
    df = _call(commonallplayers.CommonAllPlayers, is_only_current_season=1, season=season)[0]
    df = df[df["TEAM_ABBREVIATION"].fillna("") != ""]
    df = df.rename(columns={"PERSON_ID": "PLAYER_ID", "DISPLAY_FIRST_LAST": "PLAYER_NAME"})[
        ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"]]
    pos = _call(playerindex.PlayerIndex, season=season)[0][["PERSON_ID", "POSITION"]]
    pos = pos.rename(columns={"PERSON_ID": "PLAYER_ID"}).drop_duplicates("PLAYER_ID")
    df = df.merge(pos, on="PLAYER_ID", how="left")
    df["POSITION"] = df["POSITION"].fillna("F").replace("", "F")
    return df


def _cached(name: str, max_age_hours: float | None, fn):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}.pkl")
    if os.path.exists(path):
        age_h = (time.time() - os.path.getmtime(path)) / 3600
        if max_age_hours is None or age_h < max_age_hours:
            with open(path, "rb") as fh:
                return pickle.load(fh)
    value = fn()
    with open(path, "wb") as fh:
        pickle.dump(value, fh)
    return value


# --------------------------------------------------------------------------- building

def _player_totals(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = raw["p_base"].set_index("PLAYER_ID")
    df = base[["PLAYER_NAME", "TEAM_ABBREVIATION"] + BOX_COLS].copy()
    zone = raw["p_zone"].set_index("PLAYER_ID")
    for z in ZONES:
        df[f"{z}_FGA"] = zone[f"{z}_FGA"].reindex(df.index).fillna(0).astype(float)
        df[f"{z}_FGM"] = zone[f"{z}_FGM"].reindex(df.index).fillna(0).astype(float)
    adv = raw["p_adv"].set_index("PLAYER_ID")
    df["POSS"] = adv["POSS"].reindex(df.index).fillna(0).astype(float)
    return df


def _team_rates(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = raw["t_base"].set_index("TEAM_ID")
    adv = raw["t_adv"].set_index("TEAM_ID")
    ff = raw["t_ff"].set_index("TEAM_ID")
    opp = raw["t_opp"].set_index("TEAM_ID")
    zone = raw["t_zone_opp"].set_index("TEAM_ID")

    t = pd.DataFrame(index=base.index)
    t["TEAM_NAME"] = base["TEAM_NAME"]
    t["GP"] = base["GP"].astype(float)
    t["PACE"] = adv["PACE"]
    t["OFF_RATING"] = adv["OFF_RATING"]
    t["DEF_RATING"] = adv["DEF_RATING"]
    t["NET_RATING"] = adv["NET_RATING"]
    for col in ["EFG_PCT", "FTA_RATE", "TM_TOV_PCT", "OREB_PCT",
                "OPP_EFG_PCT", "OPP_FTA_RATE", "OPP_TOV_PCT", "OPP_OREB_PCT"]:
        t[col] = ff[col]
    gp = t["GP"].clip(lower=1)
    t["PTS_PG"] = base["PTS"] / gp
    t["AST_PER_FGM"] = base["AST"] / base["FGM"].clip(lower=1)
    t["STL_PER_OPP_TOV"] = (base["STL"] / opp["OPP_TOV"].clip(lower=1)).clip(0.2, 0.8)
    t["BLK_PER_OPP_MISS"] = (base["BLK"] / (opp["OPP_FGA"] - opp["OPP_FGM"]).clip(lower=1)).clip(0.02, 0.2)
    # Opponent per-game production (what a team allows) for matchup ranks.
    for col in ["PTS", "REB", "AST", "FG3M", "FG3A", "FGA", "FTA", "STL", "BLK", "TOV"]:
        t[f"OPP_{col}_PG"] = opp[f"OPP_{col}"] / gp
    tot_fga = sum(zone[f"{z}_FGA"] for z in ZONES)
    for z in ZONES:
        t[f"OPP_{z}_FREQ"] = zone[f"{z}_FGA"] / tot_fga.clip(lower=1)
        t[f"OPP_{z}_PCT"] = zone[f"{z}_FGM"] / zone[f"{z}_FGA"].clip(lower=1)
        t[f"OPP_{z}_FGA"] = zone[f"{z}_FGA"]
        t[f"OPP_{z}_FGM"] = zone[f"{z}_FGM"]
    return t


@dataclass
class Dataset:
    season: str
    prior_season: str
    players: pd.DataFrame      # blended totals, indexed by PLAYER_ID
    teams: pd.DataFrame        # blended rates, indexed by TEAM_ID (+ TEAM_ABBREVIATION col)
    logs: pd.DataFrame         # game logs for both seasons, newest first
    league: dict               # league-average rates
    built_at: str


def _blend_players(cur: pd.DataFrame, prior: pd.DataFrame, prior_games: float) -> pd.DataFrame:
    num = [c for c in cur.columns if c not in ("PLAYER_NAME", "TEAM_ABBREVIATION")]
    ids = cur.index.union(prior.index)
    c = cur.reindex(ids)
    p = prior.reindex(ids)
    w = (prior_games / p["GP"].clip(lower=1)).clip(upper=1.0).fillna(0)
    blended = c[num].fillna(0) + p[num].fillna(0).mul(w, axis=0)
    blended["PLAYER_NAME"] = c["PLAYER_NAME"].fillna(p["PLAYER_NAME"])
    blended["CUR_GP"] = c["GP"].fillna(0)
    blended["PRIOR_GP"] = p["GP"].fillna(0)
    blended["PRIOR_W"] = w
    return blended


def _blend_teams(cur: pd.DataFrame, prior: pd.DataFrame, prior_games: float) -> pd.DataFrame:
    cur = cur.reindex(prior.index.union(cur.index))
    prior = prior.reindex(cur.index)
    wc = cur["GP"].fillna(0)
    wp = np.minimum(prior["GP"].fillna(0), prior_games)
    out = pd.DataFrame(index=cur.index)
    out["TEAM_NAME"] = cur["TEAM_NAME"].fillna(prior["TEAM_NAME"])
    zone_counts = [f"OPP_{z}_{s}" for z in ZONES for s in ("FGA", "FGM")]
    for col in cur.columns:
        if col in ("TEAM_NAME",) or col in zone_counts:
            continue
        a, b = cur[col].astype(float), prior[col].astype(float)
        tot = (wc + wp).replace(0, np.nan)
        out[col] = (a.fillna(0) * wc + b.fillna(0) * wp) / tot
    out["GP"] = wc
    out["PRIOR_GP_USED"] = wp
    return out


def _league(players: pd.DataFrame, teams: pd.DataFrame, cur_raw, prior_raw) -> dict:
    lg = {}
    tz = prior_raw["t_zone_opp"]
    for z in ZONES:
        lg[f"{z}_PCT"] = float(tz[f"{z}_FGM"].sum() / tz[f"{z}_FGA"].sum())
        lg[f"{z}_FREQ"] = float(teams[f"OPP_{z}_FREQ"].mean())
    for col in ["PACE", "OFF_RATING", "DEF_RATING", "TM_TOV_PCT", "OPP_TOV_PCT", "FTA_RATE",
                "OPP_FTA_RATE", "OREB_PCT", "OPP_OREB_PCT", "AST_PER_FGM"]:
        lg[col] = float(teams[col].mean())
    pb = prior_raw["p_base"]
    lg["FT_PCT"] = float(pb["FTM"].sum() / pb["FTA"].sum())
    return lg


def load_dataset(season: str | None = None, prior_games: float = 20.0,
                 refresh: bool = False, progress=None) -> Dataset:
    """Fetch (or load cached) data and build the blended dataset.

    prior_games: how many games' worth of weight last season gets. Early in the season this
    keeps two-game samples from dominating; as the season goes on the current season wins out.
    """
    season = season or current_season()
    prior = previous_season(season)
    say = progress or (lambda msg: None)

    say(f"Loading {prior} (cached after first run)...")
    prior_raw = _cached(f"season_{prior}", None if not refresh else 0, lambda: fetch_season(prior))
    say(f"Loading {season} (refreshes every 6 hours)...")
    try:
        cur_raw = _cached(f"season_{season}", 0 if refresh else 6, lambda: fetch_season(season))
    except RuntimeError:
        cur_raw = None  # season not started / endpoint empty
    say("Loading current rosters...")
    roster = _cached(f"roster2_{season}", 0 if refresh else 12, lambda: fetch_roster(season))

    prior_p = _player_totals(prior_raw)
    prior_t = _team_rates(prior_raw)
    has_cur = cur_raw is not None and len(cur_raw["p_base"]) > 0
    if has_cur:
        cur_p = _player_totals(cur_raw)
        cur_t = _team_rates(cur_raw)
        players = _blend_players(cur_p, prior_p, prior_games)
        teams = _blend_teams(cur_t, prior_t, prior_games)
        logs = pd.concat([cur_raw["logs"], prior_raw["logs"]], ignore_index=True)
    else:
        players = _blend_players(prior_p.iloc[0:0], prior_p, prior_games)
        teams = _blend_teams(prior_t.iloc[0:0], prior_t, prior_games)
        logs = prior_raw["logs"].copy()

    # Current team comes from the live roster (handles offseason moves).
    ros = roster.set_index("PLAYER_ID")
    players = players[players.index.isin(ros.index)].copy()
    players["TEAM_ABBREVIATION"] = ros.loc[players.index, "TEAM_ABBREVIATION"].values
    players["PLAYER_NAME"] = ros.loc[players.index, "PLAYER_NAME"].values
    players["POSITION"] = ros.loc[players.index, "POSITION"].values

    from nba_api.stats.static import teams as static_teams
    abbr = {t["id"]: t["abbreviation"] for t in static_teams.get_teams()}
    teams["TEAM_ABBREVIATION"] = [abbr.get(i, "?") for i in teams.index]

    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"])
    logs = logs.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

    league = _league(players, teams, cur_raw, prior_raw)
    return Dataset(season=season, prior_season=prior, players=players, teams=teams, logs=logs,
                   league=league, built_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
