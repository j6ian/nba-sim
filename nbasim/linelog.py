"""Line log: keep every sportsbook prop line you pull, score it, grade it, and check it against
the closing line. Builds a track record from normal use with no extra Odds API credits.

Two files in data/ (not in git, so app updates never touch them):

  line_log.csv      Every prop line from every pull (book, player, stat, line, over and under
                    prices, when it was pulled, when the game starts). A row is only added when
                    it's new or its line or price changed since the last pull, so the file stays
                    small. The last pull before tip-off is the closing line.
  scored_lines.csv  Every line the model scored before tip-off: the side it picked, its Value
                    Score and grade, its probability, the book's no-vig probability. The latest
                    pre-game scoring of a book + player + stat counts. Once the game is in the box
                    scores it's graded (win / loss / push / void, 1 unit at the scored price) and
                    compared with the closing line.

Closing line value (CLV): did the market move toward the model's side after it was scored?
  Same line at the close: CLV = the closing no-vig probability of the picked side minus the
  probability implied by the price it was scored at (vig included). Positive means the bet was
  worth more at the close than its price, which over time is the best sign of a real edge.
  Line moved: the move in points toward the pick (over: the line went up; under: it went down).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import picks as favs
from .odds import norm_name, props_frame
from .props import american_payout, american_to_prob

LOG_PATH = os.path.join(favs.DATA_DIR, "line_log.csv")
SCORED_PATH = os.path.join(favs.DATA_DIR, "scored_lines.csv")
ET = "US/Eastern"

LOG_COLS = ["pulled_at", "commence", "game_date", "event_id", "home", "away", "book", "player",
            "stat", "line", "over_odds", "under_odds", "one_sided"]
SCORED_COLS = ["game_date", "scored_at", "commence", "player", "player_id", "team", "opp", "stat",
               "book", "line", "side", "odds", "book_prob", "model_prob", "sim_prob", "value_score",
               "grade", "source", "actual", "result", "close_line", "close_odds", "close_prob",
               "clv_prob", "clv_line"]


def _write(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _read(path: str, cols: list[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    for c in cols:
        if c not in df:
            df[c] = np.nan
    return df[cols]


def _now_utc() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _et_date(commence) -> str:
    return pd.Timestamp(commence).tz_convert(ET).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- saving pulls

def log_pull(event_odds: dict, path: str = LOG_PATH, pulled_at: str | None = None) -> int:
    """Save the lines in one Odds API event-odds response. Only rows that are new or changed
    since the last saved pull of the same book + player + stat + line are added.
    Returns how many rows were added. Never raises: logging must not break a pull."""
    try:
        frame = props_frame(event_odds)
        if frame.empty or not event_odds.get("commence_time"):
            return 0
        commence = event_odds["commence_time"]
        new = pd.DataFrame({
            "pulled_at": pulled_at or _now_utc(), "commence": commence, "game_date": _et_date(commence),
            "event_id": event_odds.get("id", ""), "home": event_odds.get("home_team", ""),
            "away": event_odds.get("away_team", ""), "book": frame["Book"], "player": frame["Player"],
            "stat": frame["Stat"], "line": frame["Line"].astype(float),
            "over_odds": frame["Over odds"].astype(float), "under_odds": frame["Under odds"].astype(float),
            "one_sided": frame["One-sided"].astype(bool)})[LOG_COLS]
        old = _read(path, LOG_COLS)
        # Store a book + player + stat only when its lines or prices changed since its last
        # stored snapshot, and then store all its lines, so the latest snapshot is always the
        # full set that book had posted.
        key = ["event_id", "book", "player", "stat"]

        def sig(d):
            out = {}
            for k, x in d.sort_values(["line", "over_odds", "under_odds"]).groupby(key):
                out[k] = tuple(zip(x["line"].round(2), x["over_odds"].round(1), x["under_odds"].round(1)))
            return out
        if len(old):
            prev = old[old["event_id"] == new["event_id"].iloc[0]]
            if len(prev):
                prev = prev[prev["pulled_at"] == prev.groupby(key)["pulled_at"].transform("max")]
                ps, ns = sig(prev), sig(new)
                unchanged = {k for k, v in ns.items() if ps.get(k) == v}
                new = new[[tuple(k) not in unchanged for k in new[key].values]]
        if not len(new):
            return 0
        out = pd.concat([old, new], ignore_index=True) if len(old) else new
        _write(out, path)
        return len(new)
    except Exception:  # noqa: BLE001
        return 0


def closing_lines(log: pd.DataFrame) -> pd.DataFrame:
    """The last pull before tip-off for each game + book + player + stat (every line posted in
    that pull, in case a book had more than one). Adds a normalized player name column."""
    if not len(log):
        return log.assign(pkey=pd.Series(dtype=str))
    lg = log.copy()
    lg["_pulled"] = pd.to_datetime(lg["pulled_at"], utc=True)
    lg = lg[lg["_pulled"] <= pd.to_datetime(lg["commence"], utc=True)]
    if not len(lg):
        return lg.assign(pkey=pd.Series(dtype=str))
    key = ["game_date", "book", "player", "stat"]
    # Each stored snapshot of a book + player + stat holds all its lines, so the latest one is
    # what was posted at the last pull before tip-off.
    latest = lg[lg["_pulled"] == lg.groupby(key)["_pulled"].transform("max")]
    return latest.assign(pkey=latest["player"].map(norm_name))


# --------------------------------------------------------------------------- scored lines

def record_scored(rows: list[dict], path: str = SCORED_PATH) -> int:
    """Save lines the model scored before tip-off. Each row needs game_date, player, team, opp,
    stat, book, line, side, odds, book_prob, model_prob, value_score, grade (player_id, sim_prob,
    commence and source are optional). The newest scoring of a game date + book + player + stat
    replaces older ones. Estimates and hand-typed lines are skipped."""
    rows = [r for r in rows if str(r.get("book") or "") not in ("Estimate", "Manual", "")]
    if not rows:
        return 0
    new = pd.DataFrame([{**{c: np.nan for c in SCORED_COLS}, **r, "scored_at": _now_utc(),
                         "result": "pending"} for r in rows])[SCORED_COLS]
    new["game_date"] = pd.to_datetime(new["game_date"]).dt.strftime("%Y-%m-%d")
    key = ["game_date", "book", "player", "stat"]
    new = new.drop_duplicates(key, keep="last")
    old = _read(path, SCORED_COLS)
    if len(old):
        keys = set(map(tuple, new[key].values))
        old = old[[tuple(k) not in keys for k in old[key].values]]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    _write(out, path)
    return len(new)


def load_scored(path: str = SCORED_PATH) -> pd.DataFrame:
    df = _read(path, SCORED_COLS)
    df["result"] = df["result"].fillna("pending")
    return df


def _no_vig(over_odds, under_odds, side: str) -> float:
    io, iu = american_to_prob(over_odds), american_to_prob(under_odds)
    p_over = io / (io + iu)
    return p_over if side == "OVER" else 1 - p_over


def grade(logs: pd.DataFrame, path: str = SCORED_PATH, log_path: str = LOG_PATH) -> int:
    """Grade pending scored lines whose games are in the box scores, and fill in the closing
    line. Returns how many were graded."""
    df = load_scored(path)
    todo = df.index[df["result"] == "pending"]
    if not len(todo):
        return 0
    close = closing_lines(_read(log_path, LOG_COLS))
    n = 0
    for i in todo:
        r = df.loc[i]
        pick = {"game_date": r["game_date"], "team": r["team"], "opp": r["opp"], "stat": r["stat"],
                "side": r["side"], "line": r["line"],
                "player_id": r["player_id"] if pd.notna(r["player_id"]) else None, "player": r["player"]}
        res, actual = favs.grade_one(logs, pick)
        if res == "pending":
            continue
        df.loc[i, ["result", "actual"]] = [res, actual]
        n += 1
        if len(close):
            c = close[(close["game_date"] == r["game_date"]) & (close["book"] == r["book"])
                      & (close["stat"] == r["stat"]) & (close["pkey"] == norm_name(r["player"]))]
            if len(c):
                c = c.iloc[(c["line"] - float(r["line"])).abs().argsort()[:1]].iloc[0]
                odds = c["over_odds"] if r["side"] == "OVER" else c["under_odds"]
                df.loc[i, ["close_line", "close_odds"]] = [c["line"], odds]
                if float(c["line"]) == float(r["line"]):
                    cp = _no_vig(c["over_odds"], c["under_odds"], r["side"])
                    df.loc[i, "close_prob"] = cp
                    df.loc[i, "clv_prob"] = cp - american_to_prob(r["odds"])
                    df.loc[i, "clv_line"] = 0.0
                else:
                    move = float(c["line"]) - float(r["line"])
                    df.loc[i, "clv_line"] = move if r["side"] == "OVER" else -move
    if n:
        _write(df, path)
    return n


# --------------------------------------------------------------------------- results

def profit(row) -> float:
    if row["result"] == "win":
        return american_payout(row["odds"])
    return -1.0 if row["result"] == "loss" else 0.0


def best_per_prop(df: pd.DataFrame) -> pd.DataFrame:
    """One line per game + player + stat: the book and side with the highest Value Score, which
    is the bet you'd actually make after line shopping."""
    return (df.sort_values("value_score", ascending=False)
            .drop_duplicates(["game_date", "player", "stat"]).reset_index(drop=True))


def summary(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Record, hit rate, units, ROI, average model edge and CLV of graded lines, optionally
    grouped (by = 'grade', 'stat', 'side', 'book', 'month' or any column)."""
    g = df[df["result"].isin(["win", "loss", "push"])].copy()
    cols = ["Bets", "Record", "Win %", "Units", "ROI", "Avg edge", "Beat close %", "Avg CLV"]
    if not len(g):
        return pd.DataFrame(columns=([by] if by else []) + cols)
    g["profit"] = g.apply(profit, axis=1)
    g["edge"] = g["model_prob"] - g["book_prob"]
    g["beat"] = np.where(g["clv_prob"].notna() & (g["clv_line"] == 0), g["clv_prob"] > 0,
                         np.where(g["clv_line"].notna(), g["clv_line"] > 0, np.nan))
    if by == "month":
        g["_k"] = pd.to_datetime(g["game_date"]).dt.strftime("%Y-%m")
    elif by:
        g["_k"] = g[by]
    else:
        g["_k"] = "All"

    def one(s):
        w, l, p = (int((s["result"] == x).sum()) for x in ("win", "loss", "push"))
        risked = w + l
        beat = s["beat"].dropna()
        return {"Bets": len(s), "Record": f"{w}-{l}" + (f"-{p}" if p else ""),
                "Win %": w / risked if risked else np.nan, "Units": s["profit"].sum(),
                "ROI": s["profit"].sum() / risked if risked else np.nan,
                "Avg edge": s["edge"].mean(),
                "Beat close %": beat.astype(float).mean() if len(beat) else np.nan,
                "Avg CLV": s["clv_prob"].mean()}
    rows = [{**({by: k} if by else {}), **one(s)} for k, s in g.groupby("_k")]
    out = pd.DataFrame(rows)
    if by == "grade":
        order = {"A": 0, "B": 1, "C": 2, "Pass": 3}
        out = out.sort_values(by, key=lambda s: s.map(order).fillna(9)).reset_index(drop=True)
    elif by == "month":
        out = out.iloc[::-1].reset_index(drop=True)
    return out


def calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Model probability of the picked side, in buckets, vs how often it actually won (pushes
    and voids left out). Also the book's no-vig probability for the same bets."""
    g = df[df["result"].isin(["win", "loss"])].copy()
    if not len(g):
        return pd.DataFrame(columns=["Model prob", "Bets", "Avg model", "Avg book", "Won"])
    bins = [0, .5, .55, .6, .65, .7, .8, 1.01]
    g["b"] = pd.cut(g["model_prob"], bins, right=False)
    rows = []
    for b, s in g.groupby("b", observed=True):
        rows.append({"Model prob": f"{b.left:.0%}-{min(b.right, 1):.0%}", "Bets": len(s),
                     "Avg model": s["model_prob"].mean(), "Avg book": s["book_prob"].mean(),
                     "Won": (s["result"] == "win").mean()})
    return pd.DataFrame(rows)


def scored_row(ev_row, game_date, player_id, opp, source: str, odds=None, commence=None) -> dict:
    """A scored prop (a row of the Player Props results, or of scripts/run_slate.run_game) in
    record_scored's format. odds: the price of the picked side when the row doesn't have
    'Over odds' / 'Under odds'."""
    r = ev_row
    over = r["Pick"] == "OVER"
    if odds is None:
        odds = r["Over odds"] if over else r["Under odds"]

    def side_prob(p_over, p_under):
        p_over, p_under = float(p_over), float(p_under)
        tot = max(p_over + p_under, 1e-9)
        return (p_over if over else p_under) / tot
    book_over = float(r["Book P(over)"])
    return {"game_date": str(game_date), "commence": commence, "player": r["Player"],
            "player_id": player_id, "team": r["Team"], "opp": opp, "stat": r["Stat"],
            "book": r["Book"], "line": float(r["Line"]), "side": r["Pick"], "odds": float(odds),
            "book_prob": round(book_over if over else 1 - book_over, 4),
            "model_prob": round(side_prob(r["Model P(over)"], r["Model P(under)"]), 4),
            "sim_prob": round(side_prob(r["Sim P(over)"], r["Sim P(under)"]), 4),
            "value_score": round(float(r["Value Score"]), 1), "grade": r["Grade"], "source": source}
