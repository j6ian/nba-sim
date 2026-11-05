"""Favorite picks: save props you like from the model, grade them from box scores, keep a record.

Picks are stored in data/picks.csv next to the app (the data/ folder isn't in git, so updating
the app never touches them). Each pick is one bet of 1 unit at the saved odds.

Grading uses the same NBA.com game logs as the rest of the app. A pick is graded once the
player's game against that opponent shows up in the logs:
  win / loss  the stat cleared the line on the picked side, or didn't
  push        the stat landed exactly on the line (whole-number lines)
  void        his team played that game but he didn't (books refund a DNP)
Anything can be overridden by hand in the My Picks tab.
"""
from __future__ import annotations

import datetime as dt
import os
import uuid

import numpy as np
import pandas as pd

from .props import PROP_STATS, american_payout

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PICKS_PATH = os.path.join(DATA_DIR, "picks.csv")

RESULTS = ["pending", "win", "loss", "push", "void"]
COLUMNS = ["id", "saved_at", "game_date", "player", "player_id", "team", "opp", "stat", "side",
           "line", "odds", "book", "value_score", "grade", "sim_prob", "actual", "result",
           "graded_by", "note"]


def load_picks(path: str = PICKS_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype={"id": str, "player_id": "Int64"})
    for c in COLUMNS:
        if c not in df:
            df[c] = np.nan
    df["result"] = df["result"].fillna("pending")
    df["note"] = df["note"].fillna("").astype(str)
    df["graded_by"] = df["graded_by"].fillna("").astype(str)
    return df[COLUMNS]


def save_picks(df: pd.DataFrame, path: str = PICKS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df[COLUMNS].to_csv(tmp, index=False)
    os.replace(tmp, path)


def pick_key(game_date, player: str, stat: str, side: str, line: float) -> tuple:
    return (str(pd.Timestamp(game_date).date()), player, stat, side, round(float(line), 1))


def add_picks(new: list[dict], path: str = PICKS_PATH) -> tuple[int, int]:
    """Append picks, skipping exact repeats (same game date, player, stat, side and line).
    Returns (added, skipped)."""
    df = load_picks(path)
    have = {pick_key(r.game_date, r.player, r.stat, r.side, r.line) for r in df.itertuples()}
    rows = []
    for p in new:
        k = pick_key(p["game_date"], p["player"], p["stat"], p["side"], p["line"])
        if k in have:
            continue
        have.add(k)
        rows.append({**{c: np.nan for c in COLUMNS}, **p, "id": uuid.uuid4().hex[:10],
                     "saved_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                     "game_date": k[0], "result": "pending", "graded_by": "", "note": p.get("note", "")})
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True) if len(df) else pd.DataFrame(rows)
        save_picks(df, path)
    return len(rows), len(new) - len(rows)


def _opp_of(matchup: str) -> str:
    return str(matchup).split()[-1]


def grade_one(logs: pd.DataFrame, pick) -> tuple[str, float | None]:
    """Grade one pick from game logs. Returns (result, actual stat) or ('pending', None)."""
    day = pd.Timestamp(pick["game_date"])
    window = logs[(logs["GAME_DATE"] >= day) & (logs["GAME_DATE"] <= day + pd.Timedelta(days=1))]
    if not len(window):
        return "pending", None
    team_games = window[(window["TEAM_ABBREVIATION"] == pick["team"])
                        & (window["MATCHUP"].map(_opp_of) == pick["opp"])]
    if not len(team_games):
        return "pending", None
    game_id = team_games.sort_values("GAME_DATE")["GAME_ID"].iloc[0]
    me = team_games[team_games["GAME_ID"] == game_id]
    if pd.notna(pick.get("player_id")):
        me = me[me["PLAYER_ID"] == int(pick["player_id"])]
    else:
        me = me[me["PLAYER_NAME"] == pick["player"]]
    if not len(me) or float(me["MIN"].iloc[0] or 0) <= 0:
        return "void", None
    actual = float(sum(me[c].iloc[0] for c in PROP_STATS[pick["stat"]]))
    line = float(pick["line"])
    if actual == line:
        return "push", actual
    over = actual > line
    return ("win" if over == (pick["side"] == "OVER") else "loss"), actual


def grade_pending(logs: pd.DataFrame, path: str = PICKS_PATH) -> int:
    """Grade every pending pick whose game is in the logs. Hand-set results are left alone.
    Returns how many picks were newly graded."""
    df = load_picks(path)
    todo = df.index[(df["result"] == "pending") & (df["graded_by"] != "manual")]
    n = 0
    for i in todo:
        res, actual = grade_one(logs, df.loc[i])
        if res != "pending":
            df.loc[i, ["result", "actual", "graded_by"]] = [res, actual, "auto"]
            n += 1
    if n:
        save_picks(df, path)
    return n


def profit(row) -> float:
    """Units won or lost on a 1-unit bet."""
    if row["result"] == "win":
        o = row["odds"]
        return american_payout(-110 if pd.isna(o) else o)
    if row["result"] == "loss":
        return -1.0
    return 0.0


def _summary(g: pd.DataFrame) -> dict:
    w, l, p = (int((g["result"] == r).sum()) for r in ("win", "loss", "push"))
    units = float(g["profit"].sum())
    risked = w + l
    return {"Record": f"{w}-{l}" + (f"-{p}" if p else ""), "Wins": w, "Losses": l, "Pushes": p,
            "Win %": w / risked if risked else np.nan, "Units": units,
            "ROI": units / risked if risked else np.nan}


def record(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame | dict:
    """Win/loss record of graded picks: overall (by=None), or per 'month' / 'year' / any column."""
    g = df[df["result"].isin(["win", "loss", "push"])].copy()
    g["profit"] = g.apply(profit, axis=1) if len(g) else pd.Series(dtype=float)
    if by is None:
        return _summary(g)
    d = pd.to_datetime(g["game_date"])
    if by == "month":
        g["_k"], label = d.dt.strftime("%Y-%m"), "Month"
    elif by == "year":
        g["_k"], label = d.dt.year.astype(str), "Year"
    else:
        g["_k"], label = g[by], by
    rows = [{label: k, **_summary(sub)} for k, sub in g.groupby("_k")]
    out = pd.DataFrame(rows, columns=[label, "Record", "Wins", "Losses", "Pushes", "Win %", "Units", "ROI"])
    if by == "month" and len(out):
        out[label] = pd.to_datetime(out[label] + "-01").dt.strftime("%b %Y")
        out = out.iloc[::-1].reset_index(drop=True)
    elif by == "year":
        out = out.iloc[::-1].reset_index(drop=True)
    return out
