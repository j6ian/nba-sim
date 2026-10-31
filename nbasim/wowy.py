"""With-or-without-you (WOWY) splits from game logs.

For a player i and teammate j on the same team, compare i's per-minute production in games
j played vs games j missed (while j was on the roster). This is the real-world evidence for
how a player's role changes when a teammate is out, and it's used to correct the lineup
model when you mark someone OUT.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Dataset

SHRINK_GAMES = 8      # games-without needed before trusting half of an observed split
MIN_WITHOUT = 3
MIN_WITH = 5


def _season_key(dates: pd.Series) -> pd.Series:
    # Oct-Jun seasons: shifting back 7 months maps every game to its season's start year.
    return (dates - pd.DateOffset(months=7)).dt.year


def _usage(df: pd.DataFrame) -> float:
    mins = df["MIN"].sum()
    return (df["FGA"].sum() + 0.44 * df["FTA"].sum() + df["TOV"].sum()) / mins if mins > 0 else np.nan


def game_sets(ds: Dataset, team: str, j) -> tuple[set, set]:
    """Team games teammate j played in, and team games he missed while on the roster.

    This season every team game counts (he's on the roster now); in earlier seasons only games
    between his first and last appearance for the team, so games before a trade don't count."""
    tl = ds.logs[ds.logs["TEAM_ABBREVIATION"] == team]
    if tl.empty:
        return set(), set()
    tl = tl.assign(SEASON=_season_key(tl["GAME_DATE"]))
    cur = int(ds.season[:4])
    jg = tl[(tl["PLAYER_ID"] == j) & (tl["MIN"] > 0)]
    with_ids, without_ids = set(jg["GAME_ID"]), set()
    for season, sg in tl.groupby("SEASON"):
        js = jg[jg["SEASON"] == season]
        if season == cur:
            window = sg
        elif len(js):
            window = sg[(sg["GAME_DATE"] >= js["GAME_DATE"].min()) & (sg["GAME_DATE"] <= js["GAME_DATE"].max())]
        else:
            continue
        without_ids |= set(window["GAME_ID"]) - with_ids
    return with_ids, without_ids


def split(ds: Dataset, team: str, i, j) -> dict | None:
    """Player i's per-minute production with vs without teammate j, on `team`."""
    tl = ds.logs[ds.logs["TEAM_ABBREVIATION"] == team]
    if tl.empty:
        return None
    with_ids, without_ids = game_sets(ds, team, j)
    ig = tl[(tl["PLAYER_ID"] == i) & (tl["MIN"] > 0)]
    w = ig[ig["GAME_ID"].isin(with_ids)]
    wo = ig[ig["GAME_ID"].isin(without_ids)]
    if len(w) == 0 and len(wo) == 0:
        return None

    def per36(df, col):
        m = df["MIN"].sum()
        return 36 * df[col].sum() / m if m > 0 else np.nan

    out = {"GP with": len(w), "GP without": len(wo),
           "MPG with": w["MIN"].mean() if len(w) else np.nan,
           "MPG without": wo["MIN"].mean() if len(wo) else np.nan,
           "Usage/min with": _usage(w), "Usage/min without": _usage(wo)}
    for col in ("PTS", "REB", "AST", "FG3M"):
        out[f"{col}/36 with"] = per36(w, col)
        out[f"{col}/36 without"] = per36(wo, col)
    return out


def teammate_table(ds: Dataset, team: str, pid, teammates: list[tuple]) -> pd.DataFrame:
    rows = []
    for tid, name in teammates:
        if tid == pid:
            continue
        s = split(ds, team, pid, tid)
        if s is None or s["GP without"] == 0:
            continue
        s["Teammate out"] = name
        s["Usage change"] = s["Usage/min without"] / s["Usage/min with"] - 1 if s["Usage/min with"] else np.nan
        rows.append(s)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("GP without", ascending=False)


def observed_usage_ratio(ds: Dataset, team: str, i, out_ids: list) -> tuple[float, int]:
    """Shrunk product of i's usage ratios (without / with) across tonight's absent teammates.
    Returns (ratio, total games-without evidence)."""
    ratio, evidence = 1.0, 0
    for j in out_ids:
        s = split(ds, team, i, j)
        if not s or s["GP without"] < MIN_WITHOUT or s["GP with"] < MIN_WITH:
            continue
        raw = s["Usage/min without"] / s["Usage/min with"]
        if not np.isfinite(raw):
            continue
        w = s["GP without"] / (s["GP without"] + SHRINK_GAMES)
        ratio *= 1 + w * (raw - 1)
        evidence += s["GP without"]
    return ratio, evidence


SPLIT_STATS = ["MIN", "PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "FGA", "FG3A", "FTA"]


def player_split_games(ds: Dataset, team: str, pid, out_ids: list, in_ids: list = ()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Player's game logs for `team` split into (games where every `out_ids` teammate sat and every
    `in_ids` teammate played, games where all of them played). Most recent first."""
    tl = ds.logs[(ds.logs["TEAM_ABBREVIATION"] == team) & (ds.logs["PLAYER_ID"] == pid) & (ds.logs["MIN"] > 0)]
    tl = tl.sort_values("GAME_DATE", ascending=False)
    without, together = set(tl["GAME_ID"]), set(tl["GAME_ID"])
    for j in out_ids:
        w, wo = game_sets(ds, team, j)
        without &= wo
        together &= w
    for j in in_ids:
        w, _ = game_sets(ds, team, j)
        without &= w
        together &= w
    return tl[tl["GAME_ID"].isin(without)], tl[tl["GAME_ID"].isin(together)]


def split_summary(games: pd.DataFrame) -> dict:
    """Per-game averages for a set of games."""
    if games.empty:
        return {"GP": 0}
    out = {"GP": len(games)}
    for c in SPLIT_STATS:
        out[c] = float(games[c].mean())
    out["PRA"] = float((games["PTS"] + games["REB"] + games["AST"]).mean())
    out["Usage/min"] = _usage(games)
    return out
