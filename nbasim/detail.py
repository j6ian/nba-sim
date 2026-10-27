"""Turns tracking / play-type / defender data into per-player and per-team arrays for the sim.

Every rate is shrunk toward a league (or position-group) average by its own sample size, then
normalized so the player's overall box-score profile (shots, turnovers, free throws, FG% by
zone, assists) stays the same. The tracking data changes *how* and *when* things happen
(which play type, which zone, who passes, who rebounds, who defends), not the season totals.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .data import ZONES
from .tracking import PLAY_TYPES, Tracking

T = len(PLAY_TYPES)
TI = {t: i for i, t in enumerate(PLAY_TYPES)}
HALF_COURT = np.array([t not in ("Transition", "OffRebound") for t in PLAY_TYPES])

# Typical shot-zone mix for each play type (RA, Paint, Mid, Corner 3, Above-break 3). Synergy
# doesn't publish zones by play type, so these are starting shapes; each player's matrix is then
# fit (iterative proportional fitting) so his overall zone mix matches his real shot chart.
ZONE_BY_TYPE = {
    "Isolation":     [.25, .20, .22, .03, .30],
    "Transition":    [.50, .10, .05, .10, .25],
    "PRBallHandler": [.22, .22, .18, .02, .36],
    "PRRollman":     [.62, .18, .06, .04, .10],
    "Postup":        [.30, .45, .20, .00, .05],
    "Spotup":        [.08, .07, .10, .25, .50],
    "Handoff":       [.20, .12, .12, .06, .50],
    "Cut":           [.85, .12, .02, .00, .01],
    "OffScreen":     [.08, .10, .22, .12, .48],
    "OffRebound":    [.80, .15, .03, .00, .02],
    "Misc":          [.30, .20, .20, .10, .20],
}
# Share of made shots that are assisted, by play type (league shapes; team-calibrated in the sim).
AST_BY_TYPE = {"Isolation": .06, "Transition": .55, "PRBallHandler": .15, "PRRollman": .78,
               "Postup": .20, "Spotup": .82, "Handoff": .60, "Cut": .80, "OffScreen": .82,
               "OffRebound": .04, "Misc": .40}
CATCH_SHOOT = {"Spotup", "OffScreen", "Handoff", "Cut", "PRRollman"}
PULL_UP = {"Isolation", "PRBallHandler"}
K_TYPE = 40.0


def _ipf(start: np.ndarray, rows: np.ndarray, cols: np.ndarray, iters: int = 25) -> np.ndarray:
    J = start.copy() + 1e-9
    for _ in range(iters):
        J *= (rows / np.maximum(J.sum(1), 1e-12))[:, None]
        J *= (cols / np.maximum(J.sum(0), 1e-12))[None, :]
    return J


def _norm(x, w):
    m = np.average(x, weights=w) if np.sum(w) > 0 else np.mean(x)
    return x / m if m > 0 else np.ones_like(x)


def _group(pos: str) -> str:
    return "C" if pos.startswith("C") else (pos[:1] or "F")


def position_priors(tr: Tracking, ds) -> dict:
    """Average play-type mix by position group (for players with little Synergy data)."""
    p = tr.players
    pos = ds.players["POSITION"].reindex(p.index).fillna("F").map(_group)
    poss = np.stack([p[f"PT_{t}_POSS"].values for t in PLAY_TYPES], 1)
    out = {}
    for g in ("G", "F", "C"):
        m = (pos.values == g) & (poss.sum(1) > 200)
        v = poss[m].sum(0)
        out[g] = v / v.sum() if v.sum() > 0 else np.ones(T) / T
    return out


def offense_detail(ds, tr: Tracking, prof, priors: dict) -> dict:
    lg, tl = ds.league, tr.league
    ids = prof.ids
    P = len(ids)
    p = tr.players.reindex(ids).fillna(0.0)
    box = ds.players.loc[ids]
    mins = box["MIN"].clip(lower=1).values

    # ---- play-type mix (all plays) and per-type outcome rates
    poss = np.stack([p[f"PT_{t}_POSS"].values for t in PLAY_TYPES], 1)
    prior = np.stack([priors[_group(pos)] for pos in prof.positions])
    share = (poss + K_TYPE * prior) / (poss.sum(1, keepdims=True) + K_TYPE)
    lg_tov = np.array([tl[f"{t}_TOV"] for t in PLAY_TYPES])
    lg_ftp = np.array([tl[f"{t}_FTP"] for t in PLAY_TYPES])
    tov = (np.stack([p[f"PT_{t}_TOV"].values for t in PLAY_TYPES], 1) + K_TYPE * lg_tov) / (poss + K_TYPE)
    ftp_raw = np.stack([(p[f"PT_{t}_FTP"] - p[f"PT_{t}_P1"]).clip(lower=0).values for t in PLAY_TYPES], 1)
    trip = (ftp_raw + K_TYPE * lg_ftp * 0.85) / (poss + K_TYPE)

    # ---- zones by play type, fit to each player's real shot chart
    fga_t = np.stack([p[f"PT_{t}_FGA"].values for t in PLAY_TYPES], 1)
    fga_share_t = (fga_t + K_TYPE * share * (1 - tov - trip)) / (fga_t.sum(1, keepdims=True) + K_TYPE)
    zone_share = prof.zone_fga / np.maximum(prof.zone_fga.sum(1, keepdims=True), 1e-12)
    ZT = np.array([ZONE_BY_TYPE[t] for t in PLAY_TYPES])
    pz = np.zeros((P, T, 5))
    for i in range(P):
        J = _ipf(fga_share_t[i][:, None] * ZT, fga_share_t[i], zone_share[i])
        pz[i] = J / np.maximum(J.sum(1, keepdims=True), 1e-12)

    # ---- efficiency by play type: catch-and-shoot / pull-up 3s, Synergy FG% for 2s
    lg3 = (lg["C3_PCT"] * lg["C3_FREQ"] + lg["ATB3_PCT"] * lg["ATB3_FREQ"]) / (lg["C3_FREQ"] + lg["ATB3_FREQ"])
    p3 = (box["FG3M"].values + 60 * lg3) / (box["FG3A"].values + 60)
    cs3 = (p["CS_FG3M"].values + 60 * tl["CS3"] * p3 / lg3) / (p["CS_FG3A"].values + 60)
    pu3 = (p["PU_FG3M"].values + 60 * tl["PU3"] * p3 / lg3) / (p["PU_FG3A"].values + 60)
    mult3 = np.ones((P, T))
    for t in PLAY_TYPES:
        if t in CATCH_SHOOT:
            mult3[:, TI[t]] = cs3 / p3
        elif t in PULL_UP:
            mult3[:, TI[t]] = pu3 / p3
        elif t == "Transition":
            mult3[:, TI[t]] = 0.5 * (cs3 + pu3) / p3
    w3 = fga_share_t * pz[:, :, 3:].sum(2)
    mult3 /= np.maximum((mult3 * w3).sum(1, keepdims=True) / np.maximum(w3.sum(1, keepdims=True), 1e-12), 1e-9)

    fgm_t = np.stack([p[f"PT_{t}_FGM"].values for t in PLAY_TYPES], 1)
    exp_fg = (pz * prof.zone_pct[:, None, :]).sum(2)
    mult2 = ((fgm_t + K_TYPE * exp_fg) / (fga_t + K_TYPE)) / np.maximum(exp_fg, 1e-9)
    w2 = fga_share_t * pz[:, :, :3].sum(2)
    mult2 /= np.maximum((mult2 * w2).sum(1, keepdims=True) / np.maximum(w2.sum(1, keepdims=True), 1e-12), 1e-9)
    make_mult = np.concatenate([np.repeat(mult2[:, :, None], 3, 2), np.repeat(mult3[:, :, None], 2, 2)], 2)
    make_mult = np.clip(make_mult, 0.75, 1.3)  # tiny samples can't swing efficiency wildly

    # ---- transition and putback tendencies
    trans_share = share[:, TI["Transition"]]
    lg_pb = sum(tr.players[f"PT_OffRebound_POSS"]) / max(tr.players["TRK_OREB"].sum(), 1)
    putback = (p["PT_OffRebound_POSS"].values + 20 * lg_pb) / (p["TRK_OREB"].values + 20)

    # ---- passer affinity by play type (assist chains)
    pm = prof.proj_min
    per = lambda col: p[col].values / mins
    pot = _norm(per("POT_AST") + 1e-6, pm)
    drive = _norm(per("DRIVE_PASSES") + 1e-6, pm)
    prh = _norm(per("PT_PRBallHandler_POSS") + 1e-6, pm)
    post = _norm(per("PT_Postup_POSS") + 1e-6, pm)
    passes = _norm(per("PASSES") + 1e-6, pm)
    sig = {"Spotup": 0.5 * drive + 0.5 * pot, "PRRollman": prh, "Cut": 0.6 * pot + 0.4 * post,
           "OffScreen": passes, "Handoff": passes, "Transition": pot}
    aff = np.stack([sig.get(t, pot) for t in PLAY_TYPES], 1)

    # ---- rebounding: distance and chances
    oreb_dist = np.where(p["TRK_OREB"].values > 10, p["OREB_DIST_X"].values / np.maximum(p["TRK_OREB"].values, 1),
                         tl["OREB_DIST"])
    dreb_dist = np.where(p["TRK_DREB"].values > 10, p["DREB_DIST_X"].values / np.maximum(p["TRK_DREB"].values, 1),
                         tl["DREB_DIST"])
    oreb_ch = p["TRK_OREB_CHANCES"].values / mins
    dreb_ch = p["TRK_DREB_CHANCES"].values / mins

    # ---- as a shooter: sensitivity to tight contests
    def efg(bs):
        fgm = sum(p[f"SHOT_{b}_FGM"].values + 0.5 * p[f"SHOT_{b}_FG3M"].values for b in bs)
        fga = sum(p[f"SHOT_{b}_FGA"].values for b in bs)
        return fgm, fga
    tm, ta = efg(["VT", "T"])
    om, oa = efg(["O", "WO"])
    lg_drop = 1 - tl["TIGHT_EFG"] / tl["OPEN_EFG"]
    t_e = (tm + 100 * tl["TIGHT_EFG"]) / (ta + 100)
    o_e = (om + 100 * tl["OPEN_EFG"]) / (oa + 100)
    sens = np.clip((1 - t_e / o_e) / max(lg_drop, 1e-6), 0.5, 1.6)

    # ---- clutch usage
    plays = box["FGA"].values + 0.44 * box["FTA"].values + box["TOV"].values
    usg = plays / np.maximum(box["POSS"].values, 1)
    cmin = p["CLUTCH_MIN"].values
    cusg = np.where(cmin > 0, p["CLUTCH_USG_X"].values / np.maximum(cmin, 1e-9), usg)
    raw = np.where(usg > 0, cusg / np.maximum(usg, 1e-9), 1.0)
    clutch = 1 + (np.clip(raw, 0.5, 2.0) - 1) * cmin / (cmin + 60)

    return {"share": share, "tov": tov, "trip": trip, "pz": pz, "make_mult": make_mult,
            "trans_share": trans_share, "putback": putback, "aff": aff,
            "oreb_dist": oreb_dist, "dreb_dist": dreb_dist, "oreb_ch": oreb_ch, "dreb_ch": dreb_ch,
            "contest_sens": sens, "clutch": clutch, "cs3": cs3, "pu3": pu3, "p3": p3}


def defense_detail(ds, tr: Tracking, prof) -> dict:
    """This team's players as defenders."""
    ids = prof.ids
    p = tr.players.reindex(ids).fillna(0.0)
    box = ds.players.loc[ids]
    mins = box["MIN"].clip(lower=1).values
    tl = tr.league
    d_pm = (p["D_FGM"].values - p["D_EXP_FGM"].values) / (p["D_FGA"].values + 300) - tl["D_PM"] * \
        p["D_FGA"].values / (p["D_FGA"].values + 300)
    rim_pm = (p["RIM_D_FGM"].values - p["RIM_D_EXP_FGM"].values) / (p["RIM_D_FGA"].values + 150)
    rim_rate = p["RIM_D_FGA"].values / mins
    hmin = np.maximum(p["HUSTLE_MIN"].values, 1)
    deflect = np.where(p["HUSTLE_MIN"].values > 0, p["DEFLECTIONS"].values / hmin, np.nan)
    deflect = np.where(np.isnan(deflect), np.nanmean(deflect) if np.isfinite(np.nanmean(deflect)) else 0.05, deflect)
    m = tr.matchups
    md = m[m["DEF_PLAYER_ID"].isin(ids)].groupby("DEF_PLAYER_ID")[["PARTIAL_POSS", "SFL"]].sum()
    lg_sfl = m["SFL"].sum() / max(m["PARTIAL_POSS"].sum(), 1)
    sfl = (md["SFL"].reindex(ids).fillna(0).values + 150 * lg_sfl) / (md["PARTIAL_POSS"].reindex(ids).fillna(0).values + 150)
    return {"d_pm": d_pm, "rim_pm": rim_pm, "rim_rate": rim_rate, "deflect": deflect,
            "foul_rel": sfl / lg_sfl}


def team_detail(tr: Tracking, team_id) -> dict:
    """Team play-type tendencies on offense and defense, and contest quality allowed."""
    tl = tr.league
    t = tr.teams.loc[team_id] if team_id in tr.teams.index else None
    out = {}
    if t is None:
        out["def_freq_rel"] = np.ones(T)
        out["def_ppp_rel"] = np.ones(T)
        out["push"] = out["trans_allowed"] = out["contest_q"] = 1.0
        return out
    dposs = np.array([t[f"DEF_{x}_POSS"] for x in PLAY_TYPES])
    dpts = np.array([t[f"DEF_{x}_PTS"] for x in PLAY_TYPES])
    oposs = np.array([t[f"OFF_{x}_POSS"] for x in PLAY_TYPES])
    lgf = np.array([tl[f"{x}_FREQ"] for x in PLAY_TYPES])
    lgp = np.array([tl[f"{x}_PPP"] for x in PLAY_TYPES])
    out["def_freq_rel"] = (dposs / max(dposs.sum(), 1)) / lgf if dposs.sum() > 0 else np.ones(T)
    overall = dpts.sum() / max(dposs.sum(), 1)
    shrunk = (dpts + 150 * lgp * overall / tl["PPP"]) / (dposs + 150)
    out["def_ppp_rel"] = (shrunk / max(overall, 1e-9)) / (lgp / tl["PPP"])
    ti = TI["Transition"]
    out["push"] = (oposs[ti] / max(oposs.sum(), 1)) / lgf[ti] if oposs.sum() > 0 else 1.0
    out["trans_allowed"] = out["def_freq_rel"][ti]
    from .tracking import CONTEST_BUCKETS
    fga = np.array([t[f"OPPSHOT_{b}_FGA"] for b in CONTEST_BUCKETS])
    lg_e = np.array([tl[f"{b}_EFG"] for b in CONTEST_BUCKETS])
    lg_f = np.array([tl[f"{b}_FREQ"] for b in CONTEST_BUCKETS])
    out["contest_q"] = float((fga / max(fga.sum(), 1) * lg_e).sum() / (lg_f * lg_e).sum()) if fga.sum() > 0 else 1.0
    out["contest_mix"] = fga / max(fga.sum(), 1)
    return out


def matchup_matrix(tr: Tracking, off_prof, def_prof, prior_poss: float = 30.0) -> np.ndarray:
    """P(defender d is the primary defender | shooter s), before restricting to who's on the floor.
    Real partial possessions from LeagueSeasonMatchups plus a position-similarity prior."""
    sim = {("G", "G"): 1.0, ("G", "F"): 0.5, ("F", "G"): 0.5, ("F", "F"): 1.0, ("F", "C"): 0.6,
           ("C", "F"): 0.6, ("C", "C"): 1.0, ("G", "C"): 0.1, ("C", "G"): 0.1}
    og = [_group(p) for p in off_prof.positions]
    dg = [_group(p) for p in def_prof.positions]
    prior = np.array([[sim[(a, b)] for b in dg] for a in og]) * def_prof.proj_min[None, :]
    prior = prior / np.maximum(prior.sum(1, keepdims=True), 1e-12)
    m = tr.matchups
    sub = m[m["OFF_PLAYER_ID"].isin(off_prof.ids) & m["DEF_PLAYER_ID"].isin(def_prof.ids)]
    obs = np.zeros((len(off_prof.ids), len(def_prof.ids)))
    oi = {p: i for i, p in enumerate(off_prof.ids)}
    di = {p: i for i, p in enumerate(def_prof.ids)}
    for r in sub.itertuples(index=False):
        obs[oi[r.OFF_PLAYER_ID], di[r.DEF_PLAYER_ID]] += r.PARTIAL_POSS
    return obs + prior_poss * prior


# --------------------------------------------------------------------------- schedule effects

ARENAS = {  # lat, lon, elevation (ft)
    "ATL": (33.757, -84.396, 1050), "BOS": (42.366, -71.062, 20), "BKN": (40.683, -73.975, 30),
    "CHA": (35.225, -80.839, 750), "CHI": (41.881, -87.674, 600), "CLE": (41.496, -81.688, 650),
    "DAL": (32.790, -96.810, 430), "DEN": (39.749, -105.008, 5280), "DET": (42.341, -83.055, 600),
    "GSW": (37.768, -122.388, 10), "HOU": (29.751, -95.362, 50), "IND": (39.764, -86.156, 715),
    "LAC": (33.945, -118.343, 100), "LAL": (34.043, -118.267, 300), "MEM": (35.138, -90.051, 280),
    "MIA": (25.781, -80.188, 10), "MIL": (43.045, -87.917, 620), "MIN": (44.979, -93.276, 830),
    "NOP": (29.949, -90.082, 10), "NYK": (40.751, -73.993, 30), "OKC": (35.463, -97.515, 1200),
    "ORL": (28.539, -81.384, 100), "PHI": (39.901, -75.172, 40), "PHX": (33.446, -112.071, 1100),
    "POR": (45.532, -122.667, 50), "SAC": (38.580, -121.500, 30), "SAS": (29.427, -98.438, 650),
    "TOR": (43.643, -79.379, 250), "UTA": (40.768, -111.901, 4226), "WAS": (38.898, -77.021, 50),
}


def _miles(a: str, b: str) -> float:
    if a not in ARENAS or b not in ARENAS:
        return 0.0
    la1, lo1, _ = ARENAS[a]
    la2, lo2, _ = ARENAS[b]
    p1, p2 = math.radians(la1), math.radians(la2)
    d = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 3959 * 2 * math.asin(math.sqrt(d))


def schedule_context(ds, team: str, game_date, venue: str) -> dict:
    """Rest days, games in last 4 nights, travel since last game, altitude, from game logs."""
    gd = pd.Timestamp(game_date)
    tl = ds.logs[(ds.logs["TEAM_ABBREVIATION"] == team) & (ds.logs["GAME_DATE"] < gd)]
    games = tl.drop_duplicates("GAME_ID").sort_values("GAME_DATE", ascending=False)
    if games.empty:
        return {"rest_days": None, "b2b": False, "three_in_four": False, "travel_mi": 0.0,
                "altitude": ARENAS.get(venue, (0, 0, 0))[2], "last": None}
    last = games.iloc[0]
    rest = (gd - last["GAME_DATE"]).days - 1
    in4 = (games["GAME_DATE"] >= gd - pd.Timedelta(days=3)).sum()
    mu = last["MATCHUP"]
    last_venue = mu.split(" @ ")[1] if " @ " in mu else team
    return {"rest_days": int(rest), "b2b": rest == 0, "three_in_four": in4 >= 2,
            "travel_mi": _miles(last_venue, venue), "altitude": ARENAS.get(venue, (0, 0, 0))[2],
            "last": f"{last['GAME_DATE']:%m/%d} {mu}"}


def schedule_factor(ctx: dict, is_home: bool, strength: float) -> tuple[float, float, list]:
    """(make-probability multiplier, turnover multiplier, explanation) from research-sized priors."""
    make, tov, why = 1.0, 1.0, []
    if ctx["b2b"]:
        make *= 0.986
        tov *= 1.02
        why.append("back-to-back")
        if ctx["travel_mi"] > 1000:
            make *= 0.996
            why.append(f"traveled {ctx['travel_mi']:.0f} mi")
    if ctx["three_in_four"]:
        make *= 0.995
        why.append("3 games in 4 nights")
    if not is_home and ctx["altitude"] > 4000:
        make *= 0.994
        why.append(f"altitude {ctx['altitude']} ft")
    if ctx["rest_days"] is not None and ctx["rest_days"] >= 2:
        make *= 1.003
        why.append(f"{ctx['rest_days']} days rest")
    make = 1 + (make - 1) * strength
    tov = 1 + (tov - 1) * strength
    return make, tov, why
