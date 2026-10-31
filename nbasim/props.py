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


# The sim's prop probabilities run too far from 50/50: across this season's games a sim 12% hit
# about 19% of the time and a sim 68% about 53%, mostly because a projection that strays far
# from a player's usual numbers is more often wrong than the sim's own spread allows.
# Probabilities are shrunk toward even in log-odds; 0.8 is the best single factor on that
# backtest (scripts/backtest_props.py), rounded toward caution.
PROB_SHRINK = 0.8


def calibrate_prob(p, shrink: float = PROB_SHRINK):
    """Shrink a sim probability toward 0.5 in log-odds space."""
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return 1 / (1 + np.exp(-shrink * np.log(p / (1 - p))))


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


POS_GROUPS = ["G", "F", "C"]
_DVP_STATS = ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV", "FGA", "FG3A", "FTM"]
_dvp_cache: dict = {}


def position_weights(pos: str) -> dict:
    """'G' -> {G: 1}; 'G-F' -> {G: .5, F: .5}; unknown -> {}."""
    groups = [g[:1] for g in str(pos or "").upper().split("-") if g[:1] in POS_GROUPS]
    return {g: groups.count(g) / len(groups) for g in set(groups)}


def allowed_by_position(ds: Dataset, prior_games: float = 20.0) -> pd.DataFrame:
    """What each defense gives up per game to guards, forwards and centers, from game logs.

    For every game, the opponent's players' box scores are split by position (a G-F counts half
    to each), summed, and averaged over the defense's games. Last season's games count, but are
    worth at most `prior_games` games in total, like the rest of the model.
    Index: (defending team, position group). Columns: per-game stats allowed.
    """
    key = (ds.season, ds.built_at, prior_games)
    if key in _dvp_cache:
        return _dvp_cache[key]
    lg = ds.logs[ds.logs["MIN"] > 0].copy()
    lg["DEF"] = lg["MATCHUP"].str.split().str[-1]
    pos = ds.players["POSITION"] if "POSITION" in ds.players else pd.Series(dtype=str)
    lg["pos"] = lg["PLAYER_ID"].map(pos)
    season_start = pd.Timestamp(int(ds.season[:4]), 10, 1)
    lg["cur"] = lg["GAME_DATE"] >= season_start
    rows = []
    for g in POS_GROUPS:
        w = lg["pos"].map(lambda p: position_weights(p).get(g, 0.0))
        part = lg[_DVP_STATS].mul(w, axis=0).assign(DEF=lg["DEF"], GAME_ID=lg["GAME_ID"], cur=lg["cur"])
        per_game = part.groupby(["DEF", "GAME_ID", "cur"])[_DVP_STATS].sum().reset_index()
        for team, tg in per_game.groupby("DEF"):
            n_cur, n_pri = int(tg["cur"].sum()), int((~tg["cur"]).sum())
            wp = min(1.0, prior_games / n_pri) if n_pri else 0.0
            wt = np.where(tg["cur"], 1.0, wp)
            avg = (tg[_DVP_STATS].mul(wt, axis=0).sum() / max(wt.sum(), 1e-9))
            rows.append({"TEAM": team, "GROUP": g, "games": n_cur + n_pri, **avg.to_dict()})
    df = pd.DataFrame(rows).set_index(["TEAM", "GROUP"])
    df = df[df.index.get_level_values("TEAM").isin(ds.teams["TEAM_ABBREVIATION"])]
    _dvp_cache[key] = df
    return df


def opp_rank(ds: Dataset, opp_abbr: str, stat: str, pid=None) -> tuple[float, int, float, str]:
    """What the opponent gives up per game in this stat to the player's position, its rank among
    the 30 teams (1 = allows the most = best matchup), how that compares with the league average
    (+0.08 = 8% more), and the position used. Falls back to the team's overall allowed numbers
    when the player's position isn't known."""
    parts = PROP_STATS[stat]
    weights = position_weights(ds.players["POSITION"].get(pid, "")) if pid is not None and \
        "POSITION" in ds.players else {}
    if weights:
        dvp = allowed_by_position(ds)
        val = sum(sum(dvp.xs(g, level="GROUP")[p] for p in parts) * w for g, w in weights.items())
        if opp_abbr in val.index:
            ranks = val.rank(ascending=False, method="min")
            label = "/".join(sorted(weights, key=POS_GROUPS.index))
            return float(val[opp_abbr]), int(ranks[opp_abbr]), float(val[opp_abbr] / val.mean() - 1), label
    col = OPP_ALLOWED[stat]
    t = ds.teams.set_index("TEAM_ABBREVIATION")
    ranks = t[col].rank(ascending=False, method="min")
    return float(t.loc[opp_abbr, col]), int(ranks.loc[opp_abbr]), float(t.loc[opp_abbr, col] / t[col].mean() - 1), "team"


def evaluate_prop(ds: Dataset, sim_vals: np.ndarray, pid, stat: str, line: float,
                  over_odds: float = -110, under_odds: float = -110, proj_min: float | None = None,
                  min_sd: float | None = None, opp_abbr: str | None = None,
                  sim_min: np.ndarray | None = None) -> dict:
    """sim_min: the player's simulated minutes per game. When given, only games he played count,
    the same way books void a prop when the player doesn't play."""
    parts = PROP_STATS[stat]
    if sim_min is not None and (sim_min > 0).any():
        sim_vals = sim_vals[sim_min > 0]
    hist = log_values(ds, pid, parts)
    r = ds.players.loc[pid]
    season_avg = sum(r[p] if p != "REB" else r["REB"] for p in parts) / max(r["GP"], 1)
    per_min = season_avg / max(r["MIN"] / max(r["GP"], 1), 1e-9)
    min_adj_avg = per_min * proj_min if proj_min else season_avg
    sd = float(np.std(hist)) if len(hist) >= 5 else max(np.sqrt(max(season_avg, 0.5)), 0.5)

    sim_over = float((sim_vals > line).mean())
    sim_under = float((sim_vals < line).mean())
    p_push = 1 - sim_over - sim_under
    dec = max(sim_over + sim_under, 1e-9)
    p_over = float(calibrate_prob(sim_over / dec)) * dec
    p_under = dec - p_over
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
        "Raw sim P(over)": sim_over,
        "Book P(over)": nv_over, "Edge over": edge_over, "Edge under": edge_under,
        "EV over": ev_over, "EV under": ev_under,
        "Fair over odds": prob_to_american(p_over / decided),
        "Pick": side, "Value Score": score, "Grade": grade,
        "Cushion (SD)": cushion, "Min conf": min_conf,
    }
    if opp_abbr:
        allowed, rank, vs_avg, pos = opp_rank(ds, opp_abbr, stat, pid)
        out["Opp allows"] = allowed
        out["Opp rank"] = rank
        out["Opp vs avg"] = vs_avg
        out["Opp vs pos"] = pos
    return out


def default_line(avg: float) -> float:
    """Book-style default: nearest .5 to the average."""
    return float(np.floor(avg) + 0.5)
