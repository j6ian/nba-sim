"""Prop hit rates from game logs, grouped into 100% / 90% / 80% tiers.

Two kinds of lines:
  * Sportsbook lines: how often the player went over (and under) tonight's line in his recent games.
  * Alt-line ladder: round thresholds (10+ / 15+ / 20+ points...). For each player and stat, each
    tier shows the highest threshold he cleared at that rate, which is how "safe" alt lines and
    ladders are usually picked.
Filters: how many games back, opponent (filtered first, then the last N of those games), venue,
and whether last season's games may fill out the sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Dataset
from .props import PROP_STATS

LADDERS = {
    "PTS": [5, 10, 15, 20, 25, 30, 35, 40], "REB": [2, 4, 6, 8, 10, 12, 14, 16],
    "AST": [2, 3, 4, 6, 8, 10, 12], "3PM": [1, 2, 3, 4, 5, 6],
    "PRA": [10, 15, 20, 25, 30, 35, 40, 45, 50], "PR": [10, 15, 20, 25, 30, 35, 40],
    "PA": [10, 15, 20, 25, 30, 35, 40], "RA": [4, 6, 8, 10, 12, 15, 20],
    "STL": [1, 2, 3], "BLK": [1, 2, 3], "STL+BLK": [1, 2, 3, 4], "TOV": [1, 2, 3, 4, 5],
    "FTM": [2, 4, 6, 8, 10],
}
TIERS = [(1.0, "100%"), (0.9, "90%+"), (0.8, "80%+")]


RECENT_SHOWN = 20  # values listed in the Recent column


def player_games(ds: Dataset, pid, n: int, opp: str | None = None, venue: str = "All",
                 season_only: bool = False) -> pd.DataFrame:
    """His last `n` games (most recent first), optionally only vs `opp` and/or home/away, and
    optionally only this season's games."""
    g = ds.logs[(ds.logs["PLAYER_ID"] == pid) & (ds.logs["MIN"] > 0)]
    if season_only:
        g = g[g["GAME_DATE"] >= _season_start(ds)]
    if opp:
        g = g[g["MATCHUP"].str.split().str[-1] == opp]
    if venue == "Home":
        g = g[g["MATCHUP"].str.contains(" vs. ", regex=False)]
    elif venue == "Away":
        g = g[g["MATCHUP"].str.contains(" @ ", regex=False)]
    return g.sort_values(["GAME_DATE", "GAME_ID"], ascending=False).head(n)


def _season_start(ds: Dataset) -> pd.Timestamp:
    """Current season's games are on or after August 1 of its first year."""
    return pd.Timestamp(int(ds.season[:4]), 8, 1)


def stat_values(games: pd.DataFrame, stat: str) -> np.ndarray:
    return sum(games[p].values for p in PROP_STATS[stat]).astype(float) if len(games) else np.array([])


def tier_of(rate: float) -> str | None:
    for cut, label in TIERS:
        if rate >= cut - 1e-9:
            return label
    return None


def recent_text(vals: np.ndarray, hit: np.ndarray, push: np.ndarray | None = None) -> str:
    """Values newest first; a miss is marked ✗ and a push (exactly the line) =."""
    out = []
    for i, v in enumerate(vals[:RECENT_SHOWN]):
        mark = "" if hit[i] else "=" if push is not None and push[i] else "✗"
        out.append(f"{v:g}{mark}")
    return " ".join(out)


def _row(name, team, opp, stat, games, vals, line_label, side, hit, push=None, extra=None):
    hits = int(hit.sum())
    pushes = int(push.sum()) if push is not None else 0
    return {"Player": name, "Team": team, "Opp": opp, "Stat": stat, "Line": line_label, "Side": side,
            "Hit %": hits / len(vals), "Hits": f"{hits}/{len(vals)}" + (f" ({pushes} push)" if pushes else ""),
            "Games": len(vals), "Avg": float(vals.mean()), "Low": float(vals.min()),
            "High": float(vals.max()),
            "Last game": pd.Timestamp(games["GAME_DATE"].iloc[0]).strftime("%b %d").replace(" 0", " "),
            "Recent": recent_text(vals, hit, push), **(extra or {})}


def ladder_hits(ds: Dataset, players: list[dict], stats: list[str], n: int, opp_mode: str | None,
                venue: str, min_games: int, season_only: bool = False) -> pd.DataFrame:
    """Alt-line tiers. players: dicts with pid, name, team, opp (tonight's opponent).
    opp_mode: None (any team), "tonight" (each player's opponent tonight) or a team abbreviation."""
    rows = []
    for p in players:
        opp = p["opp"] if opp_mode == "tonight" else opp_mode
        g = player_games(ds, p["pid"], n, opp, venue, season_only)
        if len(g) < min_games:
            continue
        for stat in stats:
            vals = stat_values(g, stat)
            shown = set()
            for cut, label in TIERS:
                ok = [t for t in LADDERS.get(stat, []) if (vals >= t).mean() >= cut - 1e-9]
                if not ok or max(ok) in shown:
                    continue
                t = max(ok)
                shown.add(t)
                rows.append({"Tier": label, **_row(p["name"], p["team"], p["opp"], stat, g, vals,
                                                   f"{t}+", "Over", vals >= t)})
    return pd.DataFrame(rows)


def book_line_hits(ds: Dataset, players: list[dict], lines: pd.DataFrame, n: int,
                   opp_mode: str | None, venue: str, min_games: int,
                   season_only: bool = False) -> pd.DataFrame:
    """Hit rates at tonight's sportsbook lines (the most common line across books), both sides.
    lines: Player (sim name), Stat, Line, Over odds, Under odds, Book."""
    by_name = {p["name"]: p for p in players}
    rows = []
    for (name, stat), grp in lines.dropna(subset=["Player"]).groupby(["Player", "Stat"]):
        p = by_name.get(name)
        if p is None or stat not in PROP_STATS:
            continue
        opp = p["opp"] if opp_mode == "tonight" else opp_mode
        g = player_games(ds, p["pid"], n, opp, venue, season_only)
        if len(g) < min_games:
            continue
        vals = stat_values(g, stat)
        line = float(grp["Line"].mode().iloc[0])
        at = grp[grp["Line"] == line]
        push = vals == line
        for side, hit, col in (("Over", vals > line, "Over odds"), ("Under", vals < line, "Under odds")):
            rate = hit.sum() / len(vals)
            tier = tier_of(rate)
            if tier is None:
                continue
            best = at.loc[at[col].idxmax()]
            rows.append({"Tier": tier, **_row(name, p["team"], p["opp"], stat, g, vals, f"{line:g}", side,
                                              hit, push, {"Best odds": float(best[col]), "Book": best["Book"]})})
    return pd.DataFrame(rows)
