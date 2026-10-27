"""Possession-by-possession game simulator, vectorized across N simulated games.

Lineups: every game gets a minute-by-minute rotation (see lineups.py). Each play:

  1. Who ends it: usage weights over the five on the floor (calibrated so a healthy rotation
     reproduces real usage), x clutch usage in close late-game minutes, x transition frequency
     on transition possessions. A rebounder may go straight back up (putback).
  2. Play type (isolation, P&R ball handler / roll man, spot-up, cut, post-up, handoff, off
     screen, transition, putback): the player's Synergy mix, shifted toward what tonight's
     defense allows.
  3. Primary defender: from real matchup data (who guarded whom), restricted to the five
     defenders on the floor.
  4. Turnover / drawn foul / shot: play-type rates aligned to the player's box-score rates,
     x on-floor deflections, x the primary defender's foul rate.
  5. Shot zone from the play type (fit to the player's real shot chart); rim attempts drop when
     strong rim protectors are on the floor.
  6. Make: zone FG% x team defense (FG% allowed blended with contest quality, scaled by the
     shooter's contest sensitivity) x play-type efficiency (catch-and-shoot vs pull-up 3s) x
     scheme x usage/efficiency x teammate passing + primary/help defender and rim protector
     impact vs the team's normal lineup.
  7. Assists by play type, passer weighted by the matching creation skill (assist chains).
  8. Misses: blocks, then rebounds by shot length (long 3s favor far-from-rim rebounders, rim
     and FT misses favor bigs), offensive rebound chance by zone.
  9. Steals, defensive rebounds and makes can trigger a transition possession the other way.

Game level: pace sampled per game, 4th-quarter rotation re-planned for blowouts, overtime.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import ZONES
from .lineups import (OT_SEGS, REG_SEGS, SEG_PER_Q, blowout_q4_template, fit_probabilities,
                      game_probabilities, role_templates, sample_lineups)
from .model import (ZONE_LEN, OffenseVsDefense, TeamProfile, expected_pace, normalize_minutes,
                    offense_vs_defense)

STATS = ["MIN", "PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "AST",
         "STL", "BLK", "TOV"]
TEAM_REB_SHARE = 0.10     # share of rebounds credited to the team, not a player
PACE_SD = 3.5             # game-to-game possessions SD
SHOOT_FORM_SD = 0.08      # per-game hot/cold shooting multiplier SD
USAGE_FORM_SD = 0.15      # per-game usage multiplier SD
MAX_PLAYS = 6             # cap on plays per possession (offensive rebound chains)
AND1 = np.array([0.07, 0.04, 0.02, 0.005, 0.005])


@dataclass
class SimResult:
    n: int
    home: TeamProfile
    away: TeamProfile
    score: dict                 # team -> (N,)
    poss: np.ndarray            # (N,) possessions per team
    team_stats: dict            # team -> stat -> (N,)
    players: dict               # team -> stat -> (N,P)
    zone_fga: dict              # team -> (P,5) mean zone attempts per game
    zone_fgm: dict
    quarters: dict              # team -> (N,5) points by quarter (5th = all OT)
    ot: np.ndarray              # (N,) number of overtimes
    exp_pace: float
    matchups: dict              # team -> OffenseVsDefense
    on: dict = None             # team -> (N,48,P) bool on-floor mask for regulation
    play_w: dict = None         # team -> (P,) usage weights used on the floor
    play_types: dict = None     # team -> (P,T) plays per game by play type
    pt_points: dict = None      # team -> (P,T) points per game by play type
    passes: dict = None         # team -> (P,T) assists per game by play type of the shot
    guarded: dict = None        # team (offense) -> (P_off, P_def) plays per game by primary defender
    log: list | None = None


def _cat(rng, w: np.ndarray) -> np.ndarray:
    """Vectorized categorical draw: one index per row of a (n,K) weight matrix."""
    c = np.cumsum(w, axis=1)
    tot = c[:, -1:]
    u = rng.random((w.shape[0], 1)) * tot
    return np.minimum((u > c).sum(1), w.shape[1] - 1)


class _Team:
    def __init__(self, mvd: OffenseVsDefense, n: int, rng):
        self.mvd = mvd
        self.x = mvd.x
        self.prof = mvd.off
        P = len(self.prof.names)
        self.P = P
        T = self.x["share"].shape[1]
        self.s = {k: np.zeros((n, P)) for k in STATS}
        self.zfga = np.zeros((P, 5))
        self.zfgm = np.zeros((P, 5))
        self.ptc = np.zeros((P, T))
        self.ptpts = np.zeros((P, T))
        self.astc = np.zeros((P, T))
        self.guard_c = np.zeros((P, len(mvd.dfn.names)))
        self.score = np.zeros(n)
        self.qpts = np.zeros((n, 5))
        self.trans_next = np.zeros(n, bool)
        # Per-game form: some nights a player is hot or more involved.
        self.shoot_form = np.exp(rng.normal(0, SHOOT_FORM_SD, (n, P)))
        self.usage_form = np.exp(rng.normal(0, USAGE_FORM_SD, (n, P)))
        self.pass_form = np.exp(rng.normal(0, USAGE_FORM_SD, (n, P)))
        # Sampled minutes per game, renormalized to 240.
        raw = rng.normal(self.prof.proj_min, self.prof.min_sd, (n, P))
        raw = np.clip(raw, 0, 46)
        raw[:, self.prof.proj_min < 0.5] = 0
        self.game_min = np.apply_along_axis(normalize_minutes, 1, raw)
        mult = self.prof.usage_mult if self.prof.usage_mult is not None else np.ones(P)
        self.play_w = mvd.outcome_rate.sum(1) * mult
        self.on = np.zeros((n, REG_SEGS + 6 * OT_SEGS, P), bool)
        self.pi = game_probabilities(self.game_min, self.prof.starter, self.prof.deep, self.prof.anchor)
        self.on[:, :36] = sample_lineups(rng, self.pi[:, :36], self.prof.big, self.prof.guard)

    def plan_q4(self, rng, margin: np.ndarray):
        """Re-plan the 4th quarter from minutes still owed; starters sit in blowouts."""
        owed = np.clip(self.game_min - self.on[:, :36].sum(1), 0, None)
        a = np.abs(margin)
        keep = np.where(a >= 22, 0.25, np.where(a >= 16, 0.5, np.where(a >= 12, 0.8, 1.0)))
        st = self.prof.starter
        freed = (owed[:, st] * (1 - keep[:, None])).sum(1)
        owed[:, st] *= keep[:, None]
        bw = owed[:, ~st] + 1.0  # +1 so deep bench gets garbage time too
        owed[:, ~st] += bw / bw.sum(1, keepdims=True) * freed[:, None]
        normal = role_templates(self.P, st, self.prof.deep, self.prof.anchor)[36:]
        tmpl = np.where((keep < 1)[:, None, None], blowout_q4_template(st)[None], normal[None])
        pi = fit_probabilities(tmpl, owed)
        self.on[:, 36:48] = sample_lineups(rng, pi, self.prof.big, self.prof.guard,
                                           prev_on=self.on[:, 35])

    def plan_ot(self, rng, rows: np.ndarray, k: int):
        """Overtime k: regulars play most of it."""
        w = self.game_min[rows] ** 1.5
        target = np.minimum(w / w.sum(1, keepdims=True) * 25, 5)
        pi = fit_probabilities(np.ones((OT_SEGS, self.P)), target)
        base = REG_SEGS + k * OT_SEGS
        prev = self.on[rows, base - 1] if k else self.on[rows, REG_SEGS - 1]
        self.on[rows, base:base + OT_SEGS] = sample_lineups(rng, pi, self.prof.big, self.prof.guard,
                                                            prev_on=prev)


def _add(o: _Team, sims, pl, pts, q):
    o.s["PTS"][sims, pl] += pts
    o.score[sims] += pts
    o.qpts[sims, q] += pts


def _resolve_possession(rng, o: _Team, d: _Team, idx: np.ndarray, q: int, seg: np.ndarray,
                        log, clock=None):
    """Run one possession for sims `idx` with `o` on offense. `seg` (n,) = current minute."""
    X, mvd = o.x, o.mvd
    W = X["w"]
    DB = X["def_base"]
    live = idx
    trans = o.trans_next[live].copy()
    o.trans_next[live] = False
    pb = np.full(live.size, -1)
    for _ in range(MAX_PLAYS):
        if live.size == 0:
            break
        n_ = live.size
        omin = o.on[live, seg[live], :].astype(float)
        dmin = d.on[live, seg[live], :].astype(float)

        # ---- 1. who ends the play
        uw = o.play_w[None, :] * omin * o.usage_form[live]
        close = (seg[live] >= 43) & (np.abs(o.score[live] - d.score[live]) <= 5)
        uw = np.where(close[:, None], uw * X["clutch"][None, :], uw)
        uw = np.where(trans[:, None], uw * X["trans_w"][None, :], uw * X["hc_mult"][None, :])
        who = _cat(rng, uw + 1e-12)
        has_pb = pb >= 0
        who = np.where(has_pb, pb, who)

        # ---- 2. play type
        ptype = _cat(rng, X["hc_share"][who])
        ptype = np.where(trans, X["TI_TRANS"], ptype)
        ptype = np.where(has_pb, X["TI_PB"], ptype)
        np.add.at(o.ptc, (who, ptype), 1)

        # ---- 3. primary defender (on the floor) and lineup defense
        dfd = _cat(rng, X["M"][who] * dmin + 1e-12 * dmin)
        np.add.at(o.guard_c, (who, dfd), 1)
        defl = (dmin * X["deflect"][None, :]).sum(1) / max(DB["deflect"], 1e-9)
        rim_on = (dmin * X["rim_rate"][None, :]).sum(1)
        rim_idx = rim_on / max(DB["rim_rate"], 1e-9)
        rim_pm_on = (dmin * (X["rim_rate"] * X["rim_pm"])[None, :]).sum(1) / np.maximum(rim_on, 1e-9)
        help_pm = (dmin * X["d_pm"][None, :]).sum(1) / 5
        foul = (X["foul_rel"][dfd] / X["foul_exp"][who]) ** W.defender_fouls

        # Lineup context for efficiency (usage load and teammates' passing).
        share = o.play_w[who] / np.maximum((o.play_w[None, :] * omin).sum(1), 1e-12)
        eff = np.clip((share / o.prof.base_share[who]) ** -W.usage_eff, 0.92, 1.05)
        tc = (o.prof.ast[None, :] * omin).sum(1) - o.prof.ast[who]
        creation = tc / o.prof.base_tc[who]
        syn = np.clip(1 + W.synergy * (creation - 1), 0.94, 1.05)

        # ---- 4. turnover / drawn foul / shot
        tovp = X["tov"][who, ptype] * np.clip(1 + W.deflections * 0.25 * (defl - 1), 0.8, 1.25)
        tripp = X["trip"][who, ptype] * foul
        u = rng.random(n_)
        kind = np.where(u < tovp, 0, np.where(u < tovp + tripp, 6, 1))

        nxt, nxt_pb = [], []
        # turnovers
        m = kind == 0
        if m.any():
            sims, pl = live[m], who[m]
            o.s["TOV"][sims, pl] += 1
            stl_p = np.clip(mvd.stl_p * np.clip(defl[m], 0.7, 1.4) ** (0.5 * W.deflections), 0, 0.9)
            st = rng.random(sims.size) < stl_p
            stealer = None
            if st.any():
                sw = (d.prof.stl[None, :] + 0.15 * W.deflections * X["deflect"][None, :]) * dmin[m][st] + 1e-12
                stealer = _cat(rng, sw)
                d.s["STL"][sims[st], stealer] += 1
            d.trans_next[sims] |= rng.random(sims.size) < np.where(st, d.x["trans"]["steal"],
                                                                   d.x["trans"]["other"])
            if log is not None:
                log(clock, o.prof.abbr, f"{o.prof.names[pl[0]]} turnover" +
                    (f" (stolen by {d.prof.names[stealer[0]]})" if st.any() else ""))

        # free-throw trips (shooting fouls without a make)
        m = kind == 6
        if m.any():
            sims, pl = live[m], who[m]
            ftp = o.prof.ft_pct[pl]
            made = (rng.random((sims.size, 2)) < ftp[:, None]).sum(1)
            o.s["FTA"][sims, pl] += 2
            o.s["FTM"][sims, pl] += made
            _add(o, sims, pl, made, q)
            np.add.at(o.ptpts, (pl, ptype[m]), made)
            if log is not None:
                log(clock, o.prof.abbr, f"{o.prof.names[pl[0]]} fouled by "
                    f"{d.prof.names[dfd[m][0]]}, {made[0]}/2 FT")
            last_miss = rng.random(sims.size) > ftp
            if last_miss.any():
                s2, p2 = _rebound(rng, o, d, sims[last_miss], seg, None, log, clock)
                nxt.append(s2)
                nxt_pb.append(p2)
            if (~last_miss).any():
                s3 = sims[~last_miss]
                d.trans_next[s3] |= rng.random(s3.size) < d.x["trans"]["other"] * 0.5

        # ---- 5-8. shots
        m = kind == 1
        if m.any():
            sims, pl, pt = live[m], who[m], ptype[m]
            pz = X["pz"][pl, pt].copy()
            pz[:, 0] *= np.clip(1 - W.rim_deterrence * 0.15 * (rim_idx[m] - 1), 0.85, 1.15)
            z = _cat(rng, pz)
            three = z >= 3
            p = (mvd.make_p[pl, z] * X["pt_mult"][pl, pt, z] * o.shoot_form[sims, pl]
                 * eff[m] * syn[m])
            # Individual defense vs the team's normal lineup (team-level defense is already in make_p).
            p = p + W.defender_individual * (0.6 * X["d_pm"][dfd[m]] + 0.4 * help_pm[m] - DB["d_pm"])
            rim = z <= 1
            p = p + rim * W.rim_deterrence * (rim_pm_on[m] - DB["rim_pm"])
            made = rng.random(sims.size) < np.clip(p, 0.02, 0.97)
            pts = np.where(three, 3, 2)
            o.s["FGA"][sims, pl] += 1
            o.s["FG3A"][sims, pl] += three
            np.add.at(o.zfga, (pl, z), 1)
            if made.any():
                ms, mp, mz, mpts, mt = sims[made], pl[made], z[made], pts[made], pt[made]
                o.s["FGM"][ms, mp] += 1
                o.s["FG3M"][ms, mp] += mz >= 3
                _add(o, ms, mp, mpts, q)
                np.add.at(o.zfgm, (mp, mz), 1)
                np.add.at(o.ptpts, (mp, mt), mpts)
                # Assisted chance depends on the play type; passing quality around the shooter
                # already shows up in his make probability (syn), so it isn't applied twice here.
                a = rng.random(ms.size) < X["ast_type"][mt]
                passer = None
                if a.any():
                    wts = (o.prof.ast[None, :] * o.on[ms[a], seg[ms[a]], :] * o.pass_form[ms[a]]
                           * X["aff"][:, mt[a]].T)
                    wts[np.arange(a.sum()), mp[a]] = 0
                    passer = _cat(rng, wts + 1e-12)
                    o.s["AST"][ms[a], passer] += 1
                    np.add.at(o.astc, (passer, mt[a]), 1)
                and1 = rng.random(ms.size) < AND1[mz] * foul[m][made]
                ft = None
                if and1.any():
                    s1, p1 = ms[and1], mp[and1]
                    ft = (rng.random(s1.size) < o.prof.ft_pct[p1]).astype(float)
                    o.s["FTA"][s1, p1] += 1
                    o.s["FTM"][s1, p1] += ft
                    _add(o, s1, p1, ft, q)
                    np.add.at(o.ptpts, (p1, mt[and1]), ft)
                d.trans_next[ms] |= rng.random(ms.size) < d.x["trans"]["other"]
                if log is not None:
                    from .tracking import PLAY_TYPES, PT_LABELS
                    nm = o.prof.names[mp[0]]
                    txt = (f"{nm} makes {'3PT' if mz[0] >= 3 else '2PT'} ({ZONES[mz[0]]}, "
                           f"{PT_LABELS[PLAY_TYPES[mt[0]]]}) over {d.prof.names[dfd[m][made][0]]}")
                    if a.any():
                        txt += f", assist {o.prof.names[passer[0]]}"
                    if and1.any():
                        txt += " + and-one " + ("good" if ft[0] else "missed")
                    log(clock, o.prof.abbr, txt)
            miss = ~made
            if miss.any():
                ms, mp, mz = sims[miss], pl[miss], z[miss]
                blk = rng.random(ms.size) < mvd.blk_p * np.where(mz >= 3, 0.25, 1.4)
                blocker = None
                if blk.any():
                    bw = ((d.prof.blk[None, :] + 0.02 * X["rim_rate"][None, :])
                          * d.on[ms[blk], seg[ms[blk]], :] + 1e-12)
                    blocker = _cat(rng, bw)
                    d.s["BLK"][ms[blk], blocker] += 1
                if log is not None:
                    from .tracking import PLAY_TYPES, PT_LABELS
                    txt = (f"{o.prof.names[mp[0]]} misses {'3PT' if mz[0] >= 3 else '2PT'} "
                           f"({ZONES[mz[0]]}, {PT_LABELS[PLAY_TYPES[pt[miss][0]]]})")
                    if blk.any():
                        txt += f", blocked by {d.prof.names[blocker[0]]}"
                    log(clock, o.prof.abbr, txt)
                s2, p2 = _rebound(rng, o, d, ms, seg, mz, log, clock)
                nxt.append(s2)
                nxt_pb.append(p2)
        if nxt:
            live = np.concatenate(nxt)
            pb = np.concatenate(nxt_pb)
        else:
            live = np.array([], dtype=int)
            pb = np.array([], dtype=int)
        trans = np.zeros(live.size, bool)


def _rebound(rng, o: _Team, d: _Team, sims: np.ndarray, seg: np.ndarray, zone, log, clock):
    """Assign a rebound by shot length. Returns (sims where the offense kept it, putback player or -1)."""
    X = o.x
    if zone is None:   # missed free throw: short, mostly bigs, offense rarely gets it
        length = np.full(sims.size, -1.5)
        p_off = o.mvd.oreb_p * 0.45
    else:
        length = ZONE_LEN[zone]
        p_off = o.mvd.oreb_p * X["zone_oreb"][zone]
    off = rng.random(sims.size) < p_off
    team_reb = rng.random(sims.size) < TEAM_REB_SHARE
    oi, di = off & ~team_reb, ~off & ~team_reb
    k = X["kappa"]
    pbk = np.full(sims.size, -1)
    if oi.any():
        s = sims[oi]
        wts = X["oreb_w"][None, :] * np.exp(k * length[oi][:, None] * X["oreb_off"][None, :])
        r = _cat(rng, (wts + 1e-4) * o.on[s, seg[s], :])
        o.s["OREB"][s, r] += 1
        pbk[oi] = np.where(rng.random(s.size) < X["putback"][r], r, -1)
        if log is not None:
            log(clock, o.prof.abbr, f"  offensive rebound {o.prof.names[r[0]]}")
    if di.any():
        s = sims[di]
        wts = X["dreb_w"][None, :] * np.exp(k * length[di][:, None] * X["dreb_off"][None, :])
        r = _cat(rng, (wts + 1e-4) * d.on[s, seg[s], :])
        d.s["DREB"][s, r] += 1
        d.trans_next[s] |= rng.random(s.size) < d.x["trans"]["dreb"]
        if log is not None:
            log(clock, d.prof.abbr, f"  defensive rebound {d.prof.names[r[0]]}")
    if (~off & team_reb).any():
        s = sims[~off & team_reb]
        d.trans_next[s] |= rng.random(s.size) < d.x["trans"]["other"]
    return sims[off], pbk[off]


def simulate(home_prof: TeamProfile, away_prof: TeamProfile, lg: dict, n: int = 2000,
             seed: int | None = None, pace_override: float | None = None,
             home_edge: float = 0.012, def_strength: float = 0.8,
             play_by_play: bool = False, weights=None, tl: dict | None = None,
             sched_home: tuple = (1.0, 1.0), sched_away: tuple = (1.0, 1.0)) -> SimResult:
    from .factors import Weights
    rng = np.random.default_rng(seed)
    if play_by_play:
        n = 1
    w = weights or Weights(def_strength=def_strength, home_edge=home_edge)
    mvd_h = offense_vs_defense(home_prof, away_prof, lg, home=True, weights=w, tl=tl, sched=sched_home)
    mvd_a = offense_vs_defense(away_prof, home_prof, lg, home=False, weights=w, tl=tl, sched=sched_away)
    if mvd_h.x is None or mvd_a.x is None:
        raise RuntimeError("Tracking data is required: build profiles with tr=load_tracking(...).")
    H, A = _Team(mvd_h, n, rng), _Team(mvd_a, n, rng)

    exp_pace = pace_override or expected_pace(home_prof, away_prof, lg)
    poss = np.maximum(np.round(rng.normal(exp_pace, PACE_SD, n)), 70).astype(int)
    qposs = np.stack([poss // 4 + (np.arange(4)[None, :] < (poss % 4)[:, None])[:, i]
                      for i in range(4)], axis=1)  # (n,4)

    events = [] if play_by_play else None

    def make_logger(q_label):
        def log(clock, team, text):
            events.append({"Q": q_label, "Clock": clock, "Team": team, "Play": text,
                           "Score": f"{int(A.score[0])}-{int(H.score[0])}"})
        return log

    def log_subs(seg_i, prev_i, clock, label):
        for t_ in (A, H):
            now, before = t_.on[0, seg_i], t_.on[0, prev_i]
            ins = [t_.prof.names[i] for i in np.nonzero(now & ~before)[0]]
            outs = [t_.prof.names[i] for i in np.nonzero(before & ~now)[0]]
            if ins:
                events.append({"Q": label, "Clock": clock, "Team": t_.prof.abbr,
                               "Play": f"SUB in: {', '.join(ins)} / out: {', '.join(outs)}",
                               "Score": f"{int(A.score[0])}-{int(H.score[0])}"})

    total_poss = poss.copy()
    for q in range(4):
        if q == 3:
            margin = H.score - A.score
            H.plan_q4(rng, margin)
            A.plan_q4(rng, margin)
        qmax = int(qposs[:, q].max())
        log = make_logger(f"Q{q + 1}") if play_by_play else None
        last_seg = -1
        for t in range(qmax):
            idx = np.nonzero(t < qposs[:, q])[0]
            seg = q * SEG_PER_Q + np.minimum(t * SEG_PER_Q // np.maximum(qposs[:, q], 1), SEG_PER_Q - 1)
            clock = None
            if play_by_play:
                secs = int(720 * (1 - t / max(qposs[0, q], 1)))
                clock = f"{secs // 60}:{secs % 60:02d}"
                if int(seg[0]) != last_seg:
                    if last_seg >= 0 or q > 0:
                        log_subs(int(seg[0]), int(seg[0]) - 1, clock, f"Q{q + 1}")
                    last_seg = int(seg[0])
            first, second = (A, H) if q % 2 == 0 else (H, A)
            _resolve_possession(rng, first, second, idx, q, seg, log, clock)
            _resolve_possession(rng, second, first, idx, q, seg, log, clock)
        # New quarter: no transition carry-over.
        H.trans_next[:] = False
        A.trans_next[:] = False

    ot = np.zeros(n, int)
    for k in range(6):
        tied = np.nonzero(H.score == A.score)[0]
        if tied.size == 0:
            break
        ot[tied] += 1
        ot_poss = max(int(round(exp_pace * 5 / 48)), 6)
        total_poss[tied] += ot_poss
        for t_ in (H, A):
            t_.plan_ot(rng, tied, k)
        log = make_logger(f"OT{k + 1}") if play_by_play else None
        base = REG_SEGS + k * OT_SEGS
        for t in range(ot_poss):
            clock = None
            if play_by_play:
                secs = int(300 * (1 - t / ot_poss))
                clock = f"{secs // 60}:{secs % 60:02d}"
            seg = np.full(n, base + min(t * OT_SEGS // ot_poss, OT_SEGS - 1))
            _resolve_possession(rng, A, H, tied, 4, seg, log, clock)
            _resolve_possession(rng, H, A, tied, 4, seg, log, clock)
    still = np.nonzero(H.score == A.score)[0]
    if still.size:
        coin = rng.random(still.size) < 0.5
        H.score[still[coin]] += 1
        A.score[still[~coin]] += 1

    for t_ in (H, A):
        t_.s["MIN"] = t_.on.sum(1).astype(float)  # one segment = one minute

    def team_totals(t_):
        out = {k: v.sum(1) for k, v in t_.s.items() if k != "MIN"}
        out["REB"] = out["OREB"] + out["DREB"]
        return out

    hk, ak = home_prof.abbr, away_prof.abbr
    res = SimResult(
        n=n, home=home_prof, away=away_prof,
        score={hk: H.score, ak: A.score}, poss=total_poss,
        team_stats={hk: team_totals(H), ak: team_totals(A)},
        players={hk: H.s, ak: A.s},
        zone_fga={hk: H.zfga / n, ak: A.zfga / n},
        zone_fgm={hk: H.zfgm / n, ak: A.zfgm / n},
        quarters={hk: H.qpts, ak: A.qpts}, ot=ot, exp_pace=exp_pace,
        matchups={hk: mvd_h, ak: mvd_a},
        on={hk: H.on[:, :REG_SEGS], ak: A.on[:, :REG_SEGS]},
        play_w={hk: H.play_w, ak: A.play_w},
        play_types={hk: H.ptc / n, ak: A.ptc / n}, pt_points={hk: H.ptpts / n, ak: A.ptpts / n},
        passes={hk: H.astc / n, ak: A.astc / n}, guarded={hk: H.guard_c / n, ak: A.guard_c / n},
        log=events)
    for team in (hk, ak):
        s = res.players[team]
        s["REB"] = s["OREB"] + s["DREB"]
    return res


def player_samples(res: SimResult, team: str, player_idx: int, stat_parts: list[str]) -> np.ndarray:
    s = res.players[team]
    return sum(s[p][:, player_idx] for p in stat_parts)
