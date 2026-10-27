"""NBA.com player-tracking, Synergy play-type, hustle, defender and matchup data.

Everything here comes from stats.nba.com (the same source as the box-score data):

  Passing           LeagueDashPtStats(Passing)       potential / secondary assists, passes, AST pts created
  Rebounding        LeagueDashPtStats(Rebounding)    rebound chances, contested boards, avg rebound distance
  Drives            LeagueDashPtStats(Drives)        drives, drive passes / assists / turnovers / FTA
  Catch & shoot     LeagueDashPtStats(CatchShoot)    catch-and-shoot FG / 3P
  Pull-ups          LeagueDashPtStats(PullUpShot)    pull-up FG / 3P
  Defender FG       LeagueDashPtDefend               FG% allowed vs the shooters' normal FG% (overall, rim)
  Hustle            LeagueHustleStatsPlayer          contests, deflections, charges, box-outs
  Shot contests     LeagueDashPlayerPtShot / LeagueDashOppPtShot, by closest-defender distance
  Play types        SynergyPlayTypes                 11 play types, offense (players, teams) + defense (teams)
  Matchups          LeagueSeasonMatchups             who guarded whom: partial possessions, FGA, shooting fouls
  Clutch            LeagueDashPlayerClutch           last 5 min within 5 pts: usage, minutes

Seasons are blended exactly like the box-score data: last season's counts are down-weighted to
at most N games' worth so early-season samples don't swing everything.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import _cached, _call, previous_season

PLAY_TYPES = ["Isolation", "Transition", "PRBallHandler", "PRRollman", "Postup", "Spotup",
              "Handoff", "Cut", "OffScreen", "OffRebound", "Misc"]
PT_LABELS = {"Isolation": "Isolation", "Transition": "Transition", "PRBallHandler": "P&R ball handler",
             "PRRollman": "P&R roll man", "Postup": "Post-up", "Spotup": "Spot-up", "Handoff": "Handoff",
             "Cut": "Cut", "OffScreen": "Off screen", "OffRebound": "Putback", "Misc": "Misc"}
CONTEST_BUCKETS = {"VT": "0-2 Feet - Very Tight", "T": "2-4 Feet - Tight",
                   "O": "4-6 Feet - Open", "WO": "6+ Feet - Wide Open"}


def fetch_tracking(season: str, progress=None) -> dict:
    from nba_api.stats import endpoints as E
    say = progress or (lambda m: None)
    out: dict = {}
    pt = {}
    for kind in ("Passing", "Rebounding", "Drives", "CatchShoot", "PullUpShot"):
        say(f"{season}: tracking {kind}")
        pt[kind] = _call(E.LeagueDashPtStats, season=season, pt_measure_type=kind,
                         player_or_team="Player", per_mode_simple="Totals")[0]
    out["pt"] = pt
    say(f"{season}: defender FG% and hustle")
    out["def_overall"] = _call(E.LeagueDashPtDefend, season=season, defense_category="Overall",
                               per_mode_simple="Totals")[0]
    out["def_rim"] = _call(E.LeagueDashPtDefend, season=season, defense_category="Less Than 6Ft",
                           per_mode_simple="Totals")[0]
    out["hustle"] = _call(E.LeagueHustleStatsPlayer, season=season, per_mode_time="Totals")[0]
    out["contest_player"], out["contest_team"] = {}, {}
    for b, label in CONTEST_BUCKETS.items():
        say(f"{season}: shots by closest defender ({label})")
        out["contest_player"][b] = _call(E.LeagueDashPlayerPtShot, season=season,
                                         close_def_dist_range_nullable=label, per_mode_simple="Totals")[0]
        out["contest_team"][b] = _call(E.LeagueDashOppPtShot, season=season,
                                       close_def_dist_range_nullable=label, per_mode_simple="Totals")[0]
    out["syn_player"], out["syn_team_off"], out["syn_team_def"] = {}, {}, {}
    for t in PLAY_TYPES:
        say(f"{season}: play type {PT_LABELS[t]}")
        out["syn_player"][t] = _call(E.SynergyPlayTypes, season=season, play_type_nullable=t,
                                     type_grouping_nullable="offensive",
                                     player_or_team_abbreviation="P", per_mode_simple="Totals")[0]
        out["syn_team_off"][t] = _call(E.SynergyPlayTypes, season=season, play_type_nullable=t,
                                       type_grouping_nullable="offensive",
                                       player_or_team_abbreviation="T", per_mode_simple="Totals")[0]
        out["syn_team_def"][t] = _call(E.SynergyPlayTypes, season=season, play_type_nullable=t,
                                       type_grouping_nullable="defensive",
                                       player_or_team_abbreviation="T", per_mode_simple="Totals")[0]
    say(f"{season}: clutch")
    out["clutch"] = _call(E.LeagueDashPlayerClutch, season=season,
                          measure_type_detailed_defense="Advanced", per_mode_detailed="Totals")[0]
    say(f"{season}: defensive matchups (largest download)")
    m = _call(E.LeagueSeasonMatchups, season=season, per_mode_simple="Totals")[0]
    out["matchups"] = m[["OFF_PLAYER_ID", "DEF_PLAYER_ID", "PARTIAL_POSS", "MATCHUP_FGA",
                         "MATCHUP_FGM", "MATCHUP_FTA", "SFL", "MATCHUP_TOV"]].copy()
    return out


# --------------------------------------------------------------------------- tidy per season

def _player_frame(raw: dict) -> pd.DataFrame:
    """One row per player: all tracking counts (season totals)."""
    pt = raw["pt"]
    p = pt["Passing"].set_index("PLAYER_ID")
    df = pd.DataFrame(index=p.index)
    df["TRK_MIN"] = p["MIN"]
    for c, src in [("PASSES", "PASSES_MADE"), ("TRK_AST", "AST"), ("SEC_AST", "SECONDARY_AST"),
                   ("POT_AST", "POTENTIAL_AST"), ("AST_PTS", "AST_PTS_CREATED")]:
        df[c] = p[src]
    r = pt["Rebounding"].set_index("PLAYER_ID")
    for c in ["OREB", "DREB", "OREB_CHANCES", "DREB_CHANCES", "OREB_CONTEST", "DREB_CONTEST"]:
        df["TRK_" + c] = r[c].reindex(df.index)
    # Store distance x rebounds so seasons can be blended as weighted averages.
    df["OREB_DIST_X"] = (r["AVG_OREB_DIST"] * r["OREB"]).reindex(df.index)
    df["DREB_DIST_X"] = (r["AVG_DREB_DIST"] * r["DREB"]).reindex(df.index)
    d = pt["Drives"].set_index("PLAYER_ID")
    for c in ["DRIVES", "DRIVE_FGA", "DRIVE_FTA", "DRIVE_PASSES", "DRIVE_AST", "DRIVE_TOV", "DRIVE_PF"]:
        df[c] = d[c].reindex(df.index)
    cs = pt["CatchShoot"].set_index("PLAYER_ID")
    for c in ["CATCH_SHOOT_FGM", "CATCH_SHOOT_FGA", "CATCH_SHOOT_FG3M", "CATCH_SHOOT_FG3A"]:
        df[c.replace("CATCH_SHOOT", "CS")] = cs[c].reindex(df.index)
    pu = pt["PullUpShot"].set_index("PLAYER_ID")
    for c in ["PULL_UP_FGM", "PULL_UP_FGA", "PULL_UP_FG3M", "PULL_UP_FG3A"]:
        df[c.replace("PULL_UP", "PU")] = pu[c].reindex(df.index)

    do = raw["def_overall"].set_index("CLOSE_DEF_PERSON_ID")
    df["D_FGA"] = do["D_FGA"].reindex(df.index)
    df["D_FGM"] = do["D_FGM"].reindex(df.index)
    df["D_EXP_FGM"] = (do["NORMAL_FG_PCT"] * do["D_FGA"]).reindex(df.index)
    dr = raw["def_rim"].set_index("CLOSE_DEF_PERSON_ID")
    df["RIM_D_FGA"] = dr["FGA_LT_06"].reindex(df.index)
    df["RIM_D_FGM"] = dr["FGM_LT_06"].reindex(df.index)
    df["RIM_D_EXP_FGM"] = (dr["NS_LT_06_PCT"] * dr["FGA_LT_06"]).reindex(df.index)
    h = raw["hustle"].set_index("PLAYER_ID")
    for c in ["CONTESTED_SHOTS", "CONTESTED_SHOTS_2PT", "CONTESTED_SHOTS_3PT", "DEFLECTIONS",
              "CHARGES_DRAWN", "SCREEN_ASSISTS", "BOX_OUTS", "DEF_BOXOUTS", "OFF_BOXOUTS"]:
        df[c] = h[c].reindex(df.index)
    df["HUSTLE_MIN"] = h["MIN"].reindex(df.index)
    for b, f in raw["contest_player"].items():
        f = f.set_index("PLAYER_ID")
        for c in ("FGA", "FGM", "FG3M", "FG3A"):
            df[f"SHOT_{b}_{c}"] = f[c].reindex(df.index)
    c = raw["clutch"].set_index("PLAYER_ID")
    df["CLUTCH_MIN"] = c["MIN"].reindex(df.index)
    df["CLUTCH_USG_X"] = (c["USG_PCT"] * c["MIN"]).reindex(df.index)
    for t, f in raw["syn_player"].items():
        f = f.groupby("PLAYER_ID").agg(POSS=("POSS", "sum"), PTS=("PTS", "sum"), FGA=("FGA", "sum"),
                                       FGM=("FGM", "sum"))
        # Rates come per possession; turn them into counts so seasons blend correctly.
        g = raw["syn_player"][t].assign(TOV=lambda x: x["TOV_POSS_PCT"] * x["POSS"],
                                        FTP=lambda x: x["FT_POSS_PCT"] * x["POSS"],
                                        P1=lambda x: x["PLUSONE_POSS_PCT"] * x["POSS"]).groupby("PLAYER_ID")
        df[f"PT_{t}_POSS"] = f["POSS"].reindex(df.index)
        df[f"PT_{t}_PTS"] = f["PTS"].reindex(df.index)
        df[f"PT_{t}_FGA"] = f["FGA"].reindex(df.index)
        df[f"PT_{t}_FGM"] = f["FGM"].reindex(df.index)
        df[f"PT_{t}_TOV"] = g["TOV"].sum().reindex(df.index)
        df[f"PT_{t}_FTP"] = g["FTP"].sum().reindex(df.index)
        df[f"PT_{t}_P1"] = g["P1"].sum().reindex(df.index)
    return df.fillna(0.0)


def _team_frame(raw: dict) -> pd.DataFrame:
    rows = {}
    for side, src in (("OFF", raw["syn_team_off"]), ("DEF", raw["syn_team_def"])):
        for t, f in src.items():
            f = f.set_index("TEAM_ID")
            for c, val in (("POSS", f["POSS"]), ("PTS", f["PTS"]), ("FGA", f["FGA"]), ("FGM", f["FGM"]),
                           ("TOV", f["TOV_POSS_PCT"] * f["POSS"]), ("FTP", f["FT_POSS_PCT"] * f["POSS"])):
                rows[f"{side}_{t}_{c}"] = val
    for b, f in raw["contest_team"].items():
        f = f.set_index("TEAM_ID")
        for c in ("FGA", "FGM", "FG3M"):
            rows[f"OPPSHOT_{b}_{c}"] = f[c]
        rows["GP"] = f["GP"]
    return pd.DataFrame(rows).fillna(0.0)


def _matchup_frame(raw: dict) -> pd.DataFrame:
    return raw["matchups"].groupby(["OFF_PLAYER_ID", "DEF_PLAYER_ID"]).sum().reset_index()


@dataclass
class Tracking:
    players: pd.DataFrame       # blended counts by PLAYER_ID
    teams: pd.DataFrame         # blended counts by TEAM_ID
    matchups: pd.DataFrame      # blended OFF/DEF pairs
    league: dict
    available: bool = True


def _blend(cur: pd.DataFrame | None, prior: pd.DataFrame, w_prior: pd.Series) -> pd.DataFrame:
    ids = prior.index if cur is None else prior.index.union(cur.index)
    p = prior.reindex(ids).fillna(0).mul(w_prior.reindex(ids).fillna(0), axis=0)
    if cur is None:
        return p
    return cur.reindex(ids).fillna(0).add(p, fill_value=0)


def _league(players: pd.DataFrame, teams: pd.DataFrame) -> dict:
    lg = {}
    tot = {t: teams[f"OFF_{t}_POSS"].sum() for t in PLAY_TYPES}
    allp = sum(tot.values())
    for t in PLAY_TYPES:
        lg[f"{t}_FREQ"] = tot[t] / allp
        lg[f"{t}_PPP"] = teams[f"OFF_{t}_PTS"].sum() / max(tot[t], 1)
        lg[f"{t}_TOV"] = teams[f"OFF_{t}_TOV"].sum() / max(tot[t], 1)
        lg[f"{t}_FTP"] = teams[f"OFF_{t}_FTP"].sum() / max(tot[t], 1)
        lg[f"{t}_EFG"] = teams[f"OFF_{t}_FGM"].sum() / max(teams[f"OFF_{t}_FGA"].sum(), 1)
    lg["PPP"] = sum(teams[f"OFF_{t}_PTS"].sum() for t in PLAY_TYPES) / allp
    for b in CONTEST_BUCKETS:
        fga, fgm, f3 = (teams[f"OPPSHOT_{b}_{c}"].sum() for c in ("FGA", "FGM", "FG3M"))
        lg[f"{b}_EFG"] = (fgm + 0.5 * f3) / max(fga, 1)
        lg[f"{b}_FREQ"] = fga
    tot_fga = sum(lg[f"{b}_FREQ"] for b in CONTEST_BUCKETS)
    for b in CONTEST_BUCKETS:
        lg[f"{b}_FREQ"] /= tot_fga
    p = players
    lg["AST_PER_POT"] = p["TRK_AST"].sum() / max(p["POT_AST"].sum(), 1)
    lg["REB_PER_CHANCE"] = (p["TRK_OREB"].sum() + p["TRK_DREB"].sum()) / max(
        p["TRK_OREB_CHANCES"].sum() + p["TRK_DREB_CHANCES"].sum(), 1)
    lg["OREB_DIST"] = p["OREB_DIST_X"].sum() / max(p["TRK_OREB"].sum(), 1)
    lg["DREB_DIST"] = p["DREB_DIST_X"].sum() / max(p["TRK_DREB"].sum(), 1)
    lg["D_PM"] = (p["D_FGM"].sum() - p["D_EXP_FGM"].sum()) / max(p["D_FGA"].sum(), 1)
    lg["CS3"] = p["CS_FG3M"].sum() / max(p["CS_FG3A"].sum(), 1)
    lg["PU3"] = p["PU_FG3M"].sum() / max(p["PU_FG3A"].sum(), 1)
    tight = ["VT", "T"]
    open_ = ["O", "WO"]
    fg = lambda bs: (sum(p[f"SHOT_{b}_FGM"].sum() + 0.5 * p[f"SHOT_{b}_FG3M"].sum() for b in bs)
                     / max(sum(p[f"SHOT_{b}_FGA"].sum() for b in bs), 1))
    lg["TIGHT_EFG"], lg["OPEN_EFG"] = fg(tight), fg(open_)
    return lg


def load_tracking(ds, prior_games: float = 20.0, refresh: bool = False, progress=None) -> Tracking:
    """Fetch (or load cached) tracking data for both seasons and blend them."""
    prior_raw = _cached(f"tracking_{ds.prior_season}", None if not refresh else 0,
                        lambda: fetch_tracking(ds.prior_season, progress))
    try:
        cur_raw = _cached(f"tracking_{ds.season}", 0 if refresh else 6,
                          lambda: fetch_tracking(ds.season, progress))
    except RuntimeError:
        cur_raw = None
    pp, pt, pm = _player_frame(prior_raw), _team_frame(prior_raw), _matchup_frame(prior_raw)
    # Same per-player prior weight as the box-score blend.
    w_player = ds.players["PRIOR_W"].reindex(pp.index.union(ds.players.index)).fillna(
        min(1.0, prior_games / 60))
    w_team = pd.Series(np.minimum(1.0, prior_games / pt["GP"].clip(lower=1)), index=pt.index)
    if cur_raw is not None and len(cur_raw["pt"]["Passing"]):
        cp, ct, cm = _player_frame(cur_raw), _team_frame(cur_raw), _matchup_frame(cur_raw)
        players = _blend(cp, pp, w_player)
        teams = _blend(ct, pt, w_team)
        key = ["OFF_PLAYER_ID", "DEF_PLAYER_ID"]
        mm = pm.set_index(key) * min(1.0, prior_games / 60)
        matchups = cm.set_index(key).add(mm, fill_value=0).reset_index()
    else:
        players, teams = _blend(None, pp, w_player), _blend(None, pt, w_team)
        matchups = pm
    league = _league(players, teams)
    return Tracking(players=players, teams=teams, matchups=matchups, league=league)
