"""Daily model report card: the model's own 3 most confident picks each day, graded and kept.

How a day's card is built:
  1. Every time sportsbook props are scored before tip-off (the Player Props tab with live
     lines, the Report Card tab's "Score every game", or scripts/run_slate.py), the best book
     and side for each player + stat is saved as a candidate in data/model_candidates.csv.
     A later scoring of the same player + stat replaces the earlier one, so the latest
     pre-game numbers count. Nothing is saved once a game has started, so the model never
     picks with hindsight.
  2. Once the day is over, its card is locked: the TOP_N candidates with the highest Value
     Score become that day's picks, copied to data/report_card.csv.
  3. Locked picks are graded from the box scores exactly like My Picks (nbasim/picks.py):
     win / loss / push at the saved odds, void when the player sat out.

Estimated lines (Book = Estimate) and hand-typed lines never count: only real sportsbook prices.
"""
from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

from . import picks as favs

CAND_PATH = os.path.join(favs.DATA_DIR, "model_candidates.csv")
CARD_PATH = os.path.join(favs.DATA_DIR, "report_card.csv")
TOP_N = 3
NOT_BOOKS = {"Estimate", "Manual", ""}
ET = "US/Eastern"

CAND_COLS = ["game_date", "scored_at", "player", "player_id", "team", "opp", "stat", "side",
             "line", "odds", "book", "value_score", "grade", "sim_prob", "source"]


def game_started(g: dict, now: pd.Timestamp | None = None) -> bool:
    """True once a slate game is live or final, or its scheduled tip-off has passed."""
    if int(g.get("status") or 1) >= 2:
        return True
    start = g.get("start")
    if start is None or pd.isna(start):
        return False
    now = now if now is not None else pd.Timestamp.now(tz=ET)
    start = pd.Timestamp(start)
    if start.tzinfo is None:
        start = start.tz_localize(ET)
    return now >= start


def load_candidates(path: str = CAND_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=CAND_COLS)
    df = pd.read_csv(path, dtype={"player_id": "Int64"})
    for c in CAND_COLS:
        if c not in df:
            df[c] = np.nan
    return df[CAND_COLS]


def record_candidates(rows: list[dict], path: str = CAND_PATH) -> int:
    """Save scored props as candidates for their day's card. Each row needs the CAND_COLS
    fields (scored_at is filled in). The newest scoring of a game date + player + stat wins.
    Rows from estimated or hand-typed lines are dropped. Returns how many rows were saved."""
    rows = [r for r in rows if str(r.get("book") or "") not in NOT_BOOKS]
    if not rows:
        return 0
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    new = pd.DataFrame([{**{c: np.nan for c in CAND_COLS}, **r, "scored_at": now} for r in rows])[CAND_COLS]
    new["game_date"] = pd.to_datetime(new["game_date"]).dt.strftime("%Y-%m-%d")
    # Best side and book per player + stat, in case several were passed.
    new = new.sort_values("value_score", ascending=False).drop_duplicates(["game_date", "player", "stat"])
    old = load_candidates(path)
    if len(old):
        keys = set(zip(new["game_date"], new["player"], new["stat"]))
        old = old[[k not in keys for k in zip(old["game_date"], old["player"], old["stat"])]]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    out.sort_values(["game_date", "value_score"], ascending=[True, False]).to_csv(tmp, index=False)
    os.replace(tmp, path)
    return len(new)


def top_picks(cands: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """The n highest Value Scores for each game date."""
    if not len(cands):
        return cands
    return (cands.sort_values(["game_date", "value_score"], ascending=[True, False])
            .groupby("game_date", sort=True).head(n).reset_index(drop=True))


def lock_days(today: dt.date, cand_path: str = CAND_PATH, card_path: str = CARD_PATH,
              n: int = TOP_N) -> int:
    """Lock the card for every finished day (before today) not locked yet. Returns picks added."""
    cands = load_candidates(cand_path)
    if not len(cands):
        return 0
    card = favs.load_picks(card_path)
    locked = set(card["game_date"].astype(str))
    todo = cands[(pd.to_datetime(cands["game_date"]).dt.date < today)
                 & ~cands["game_date"].isin(locked)]
    if not len(todo):
        return 0
    top = top_picks(todo, n)
    new = []
    for i, r in enumerate(top.itertuples()):
        new.append({"game_date": r.game_date, "player": r.player,
                    "player_id": None if pd.isna(r.player_id) else int(r.player_id),
                    "team": r.team, "opp": r.opp, "stat": r.stat, "side": r.side,
                    "line": float(r.line), "odds": float(r.odds), "book": r.book,
                    "value_score": float(r.value_score), "grade": r.grade,
                    "sim_prob": float(r.sim_prob) if pd.notna(r.sim_prob) else np.nan,
                    "note": f"scored {r.scored_at}"})
    added, _ = favs.add_picks(new, card_path)
    return added


def update(logs: pd.DataFrame, today: dt.date) -> tuple[int, int]:
    """Lock finished days, then grade every pending pick. Returns (picks locked, picks graded)."""
    return lock_days(today), favs.grade_pending(logs, CARD_PATH)


def load_card() -> pd.DataFrame:
    return favs.load_picks(CARD_PATH)


def daily_cards(card: pd.DataFrame) -> pd.DataFrame:
    """One row per day: the picks, their results, the day's record and units."""
    if not len(card):
        return pd.DataFrame(columns=["Date", "Picks", "Record", "Units"])
    c = card.copy()
    c["profit"] = c.apply(favs.profit, axis=1)
    mark = {"win": "✅", "loss": "❌", "push": "➖", "void": "🚫", "pending": "⏳"}
    rows = []
    for day, g in c.sort_values(["game_date", "value_score"], ascending=[False, False]).groupby(
            "game_date", sort=False):
        bets = [f"{mark.get(r.result, '')} {r.player} {r.side.lower()} {r.line:g} {r.stat}"
                + (f" ({r.actual:g})" if pd.notna(r.actual) else "") for r in g.itertuples()]
        rec = favs.record(g)
        rows.append({"Date": day, "Picks": " · ".join(bets),
                     "Record": rec["Record"] if rec["Wins"] + rec["Losses"] + rec["Pushes"] else "—",
                     "Units": rec["Units"] if rec["Wins"] + rec["Losses"] else np.nan})
    return pd.DataFrame(rows)
