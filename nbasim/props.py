"""Prop evaluation: compare a line to averages, recent hit rates, and the simulated distribution.

Value Score (0-100, for the better side):
  50 = no edge. Built from four parts, each squashed so no single input dominates:
    - Sim edge (55%):     sim probability minus the book's no-vig implied probability
    - Cushion (25%):      (minutes-adjusted average - line) / game-to-game SD
    - Hit rate (20%):     share of the last 10 games that cleared the line
  then pulled toward 50 when the minutes projection is shaky (high minutes SD).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Dataset

PROP_STATS = {
    "PTS": ["PTS"], "REB": ["REB"], "AST": ["AST"], "3PM": ["FG3M"],
    "PRA": ["PTS", "REB", "AST"], "PR": ["PTS", "REB"], "PA": ["PTS", "AST"], "RA": ["REB", "AST"],
    "STL": ["STL"], "BLK": ["BLK"], "STL+BLK": ["STL", "BLK"], "TOV": ["TOV"],
    "FGA": ["FGA"], "3PA": ["FG3A"], "FTM": ["FTM"],
}
# Which opponent "allowed per game" column best describes the matchup for each stat.
OPP_ALLOWED = {"PTS": "OPP_PTS_PG", "REB": "OPP_REB_PG", "AST": "OPP_AST_PG", "3PM": "OPP_FG3M_PG",
               "PRA": "OPP_PTS_PG", "PR": "OPP_PTS_PG", "PA": "OPP_PTS_PG", "RA": "OPP_REB_PG",
               "STL": "OPP_TOV_PG", "BLK": "OPP_FGA_PG", "STL+BLK": "OPP_TOV_PG",
               "TOV": "OPP_STL_PG", "FGA": "OPP_FGA_PG", "3PA": "OPP_FG3A_PG", "FTM": "OPP_FTA_PG"}


def american_to_prob(odds: float) -> float:
    odds = float(odds)
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def american_payout(odds: float) -> float:
    """Profit per $1 staked."""
    odds = float(odds)
    return odds / 100 if odds > 0 else 100 / -odds


def prob_to_american(p: float) -> str:
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    v = -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p
    return f"{v:+.0f}"


def log_values(ds: Dataset, pid, parts: list[str]) -> np.ndarray:
    lg = ds.logs[(ds.logs["PLAYER_ID"] == pid) & (ds.logs["MIN"] > 0)]
    return sum(lg[p].values for p in parts) if len(lg) else np.array([])


def opp_rank(ds: Dataset, opp_abbr: str, stat: str) -> tuple[float, int]:
    """Opponent's allowed-per-game for the stat, and its rank (1 = allows the most = best matchup)."""
    col = OPP_ALLOWED[stat]
    t = ds.teams.set_index("TEAM_ABBREVIATION")
    ranks = t[col].rank(ascending=False, method="min")
    return float(t.loc[opp_abbr, col]), int(ranks.loc[opp_abbr])


def evaluate_prop(ds: Dataset, sim_vals: np.ndarray, pid, stat: str, line: float,
                  over_odds: float = -110, under_odds: float = -110, proj_min: float | None = None,
                  min_sd: float | None = None, opp_abbr: str | None = None) -> dict:
    parts = PROP_STATS[stat]
    hist = log_values(ds, pid, parts)
    r = ds.players.loc[pid]
    season_avg = sum(r[p] if p != "REB" else r["REB"] for p in parts) / max(r["GP"], 1)
    per_min = season_avg / max(r["MIN"] / max(r["GP"], 1), 1e-9)
    min_adj_avg = per_min * proj_min if proj_min else season_avg
    sd = float(np.std(hist)) if len(hist) >= 5 else max(np.sqrt(max(season_avg, 0.5)), 0.5)

    p_over = float((sim_vals > line).mean())
    p_under = float((sim_vals < line).mean())
    p_push = 1 - p_over - p_under
    io, iu = american_to_prob(over_odds), american_to_prob(under_odds)
    nv_over = io / (io + iu)
    # Compare on a no-push basis so integer lines are fair.
    decided = max(p_over + p_under, 1e-9)
    edge_over = p_over / decided - nv_over
    edge_under = p_under / decided - (1 - nv_over)
    ev_over = p_over * american_payout(over_odds) - p_under
    ev_under = p_under * american_payout(under_odds) - p_over

    l10 = hist[:10]
    hit10 = float((l10 > line).mean()) if len(l10) else np.nan
    hit_season = float((hist > line).mean()) if len(hist) else np.nan
    cushion = (min_adj_avg - line) / max(sd, 0.5)

    def squash(x, scale):
        return np.tanh(x / scale)

    h = 0.0 if np.isnan(hit10) else (hit10 - 0.5)
    raw_over = 0.55 * squash(edge_over, 0.10) + 0.25 * squash(cushion, 0.6) + 0.20 * squash(h, 0.25)
    raw_under = 0.55 * squash(edge_under, 0.10) + 0.25 * squash(-cushion, 0.6) + 0.20 * squash(-h, 0.25)
    cv = (min_sd / proj_min) if (min_sd and proj_min) else 0.15
    min_conf = 1 - float(np.clip(cv - 0.15, 0, 0.35))
    side = "OVER" if raw_over >= raw_under else "UNDER"
    raw = max(raw_over, raw_under)
    score = 50 + 50 * raw * min_conf
    score = float(np.clip(score, 0, 100))
    grade = "A" if score >= 75 else "B" if score >= 65 else "C" if score >= 58 else "Pass"

    out = {
        "Line": line, "Season avg": season_avg, "Min-adj avg": min_adj_avg,
        "L5 avg": float(hist[:5].mean()) if len(hist) else np.nan,
        "L10 avg": float(l10.mean()) if len(l10) else np.nan,
        "L10 hit": hit10, "Season hit": hit_season,
        "Sim mean": float(sim_vals.mean()), "Sim median": float(np.median(sim_vals)),
        "Sim P(over)": p_over, "Sim P(under)": p_under, "P(push)": p_push,
        "Book P(over)": nv_over, "Edge over": edge_over, "Edge under": edge_under,
        "EV over": ev_over, "EV under": ev_under,
        "Fair over odds": prob_to_american(p_over / decided),
        "Pick": side, "Value Score": score, "Grade": grade,
        "Cushion (SD)": cushion, "Min conf": min_conf,
    }
    if opp_abbr:
        allowed, rank = opp_rank(ds, opp_abbr, stat)
        out["Opp allows"] = allowed
        out["Opp rank"] = rank
    return out


def default_line(avg: float) -> float:
    """Book-style default: nearest .5 to the average."""
    return float(np.floor(avg) + 0.5)
