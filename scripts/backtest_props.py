"""Backtest the sim's player projections against games already played this season.

For each finished game, build both teams with the real players who played (projected minutes =
actual minutes, so minutes error doesn't hide rate bias), simulate, and compare the sim's mean
to the box score. Prints bias (sim - actual) and calibration of P(over) at the player's season line
(no historical book lines are stored, so the line is his season average rounded to .5).
P(over) is what the Player Props tab shows, after calibrate_prob.

Usage: python scripts/backtest_props.py [--games N] [--sims N]
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from nbasim.data import current_season, load_dataset  # noqa: E402
from nbasim.factors import Weights  # noqa: E402
from nbasim.model import minutes_table, set_tracking, team_profile  # noqa: E402
from nbasim.sim import simulate  # noqa: E402

try:  # probability calibration the app applies to prop odds (absent in older versions)
    from nbasim.props import calibrate_prob  # noqa: E402
except ImportError:
    def calibrate_prob(p):
        return p
from nbasim.tracking import load_tracking  # noqa: E402

STATS = ["PTS", "REB", "AST", "FG3M"]


def actual_minutes(ds, team: str, logs: pd.DataFrame) -> pd.DataFrame:
    """The team's minutes table with Proj Min set to what each player really played."""
    mt = minutes_table(ds, team)
    real = logs.set_index("PLAYER_ID")["MIN"]
    keep = mt[mt["PLAYER_ID"].isin(real.index)].copy()
    extra = [p for p in real.index if p not in set(mt["PLAYER_ID"]) and p in ds.players.index]
    for p in extra:
        keep.loc[len(keep)] = {"PLAYER_ID": p, "Player": ds.players.loc[p, "PLAYER_NAME"], "GP": 1,
                               "Season MPG": real[p], "Team L5 MPG": None, "L10 MPG": None,
                               "Min SD": 3.0, "Proj Min": real[p], "OUT": False}
    keep["Proj Min"] = keep["PLAYER_ID"].map(real).astype(float)
    keep["Min SD"] = 2.5
    keep["Play %"] = 1.0   # everyone here played
    return keep[keep["Proj Min"] > 0].reset_index(drop=True)


def run(ds, tr, weights, games, n_sims, real_minutes=True):
    season_start = pd.Timestamp(int(ds.season[:4]), 10, 1)
    logs = ds.logs[ds.logs["GAME_DATE"] >= season_start]
    rows = []
    for gid in games:
        g = logs[logs["GAME_ID"] == gid]
        teams = list(g["TEAM_ABBREVIATION"].unique())
        if len(teams) != 2:
            continue
        home = g[g["MATCHUP"].str.contains("vs.")]["TEAM_ABBREVIATION"].iloc[0]
        away = [t for t in teams if t != home][0]
        profs = {}
        for t in (home, away):
            if real_minutes:
                mt = actual_minutes(ds, t, g[g["TEAM_ABBREVIATION"] == t])
            else:
                # Project minutes only from games before this one, as the app would have.
                before = dataclasses.replace(ds, logs=ds.logs[ds.logs["GAME_DATE"] < g["GAME_DATE"].iloc[0]])
                mt = minutes_table(before, t)
            profs[t] = team_profile(ds, t, mt,
                                    tr=tr, w_pot=weights.potential_ast, w_chances=weights.rebound_chances)
        res = simulate(profs[home], profs[away], ds.league, n=n_sims, seed=7, weights=weights, tl=tr.league)
        for t in (home, away):
            box = g[g["TEAM_ABBREVIATION"] == t].set_index("PLAYER_ID")
            for i, pid in enumerate(profs[t].ids):
                if pid not in box.index:
                    continue
                r = {"GAME_ID": gid, "TEAM": t, "PLAYER_ID": pid, "MIN": box.loc[pid, "MIN"],
                     "proj_MIN": profs[t].proj_min[i],
                     "sim_MIN": res.players[t]["MIN"][:, i][res.players[t]["MIN"][:, i] > 0].mean()}
                for s in STATS:
                    sims = res.players[t][s][:, i]
                    played = res.players[t]["MIN"][:, i] > 0   # props are void on a DNP
                    sims = sims[played] if played.any() else sims
                    line = np.floor(ds.players.loc[pid, s] / max(ds.players.loc[pid, "GP"], 1)) + 0.5
                    r[f"sim_{s}"] = sims.mean()
                    r[f"sd_{s}"] = sims.std()
                    r[f"act_{s}"] = box.loc[pid, s]
                    r[f"line_{s}"] = line
                    r[f"pover_{s}"] = float(calibrate_prob((sims > line).mean()))
                rows.append(r)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, label: str) -> str:
    out = [f"### {label}  ({df['GAME_ID'].nunique()} games, {len(df)} player-games)", "",
           "| Stat | Sim avg | Actual avg | Bias | Bias % | Spread ratio | Brier (season line) | Avg P(over) | Over rate |",
           "|---|---|---|---|---|---|---|---|---|"]
    d = df[df["MIN"] > 0]   # props are void when the player does not play
    for s in STATS:
        sim, act = d[f"sim_{s}"].mean(), d[f"act_{s}"].mean()
        hit = (d[f"act_{s}"] > d[f"line_{s}"]).astype(float)
        brier = ((d[f"pover_{s}"] - hit) ** 2).mean()
        # Real misses vs the sim's own spread: 1.0 = right width, > 1 = sim too narrow (overconfident).
        spread = (d[f"act_{s}"] - d[f"sim_{s}"]).std() / np.sqrt((d[f"sd_{s}"] ** 2).mean())
        out.append(f"| {s} | {sim:.2f} | {act:.2f} | {sim - act:+.2f} | {(sim / act - 1):+.1%} | {spread:.2f} | "
                   f"{brier:.4f} | {d[f'pover_{s}'].mean():.1%} | {hit.mean():.1%} |")
    out += ["", "Calibration: P(over) bucket vs how often the over actually hit (REB + AST + PTS + 3PM)", "",
            "| P(over) | N | Avg P | Actual over rate |", "|---|---|---|---|"]
    pv = pd.concat([d[f"pover_{s}"] for s in STATS])
    hit = pd.concat([(d[f"act_{s}"] > d[f"line_{s}"]).astype(float) for s in STATS])
    for lo, hi in [(0, .2), (.2, .35), (.35, .5), (.5, .65), (.65, .8), (.8, 1.01)]:
        k = (pv >= lo) & (pv < hi)
        if k.any():
            out.append(f"| {lo:.0%}-{min(hi, 1):.0%} | {k.sum()} | {pv[k].mean():.1%} | {hit[k].mean():.1%} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--out", default=None)
    ap.add_argument("--projected", action="store_true",
                    help="use the app's projected minutes instead of the minutes actually played")
    args = ap.parse_args()
    ds = load_dataset(current_season())
    tr = load_tracking(ds)
    set_tracking(tr)
    season_start = pd.Timestamp(int(ds.season[:4]), 10, 1)
    games = list(ds.logs[ds.logs["GAME_DATE"] >= season_start].drop_duplicates("GAME_ID")["GAME_ID"])
    games = games[: args.games]
    df = run(ds, tr, Weights(), games, args.sims, real_minutes=not args.projected)
    if args.out:
        df.to_csv(args.out, index=False)
    print(report(df, "Backtest, " + ("projected" if args.projected else "actual") + " minutes"))


if __name__ == "__main__":
    main()
