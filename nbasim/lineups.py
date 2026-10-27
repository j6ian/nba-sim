"""Who is on the floor: minute-by-minute rotations, vectorized across sims.

Each simulated game is split into 1-minute segments (48 in regulation, 5 per overtime).

1. Every player gets a role template: how likely he is to be on the floor at each minute.
   Starters open the 1st and 3rd quarters and close the 2nd and 4th; the bench plays the
   middle stretches; deep bench is spread thin. The top-usage starter (the "anchor") is
   staggered: he comes back early to play with the bench, then rests while the other
   starters return, so all-bench units are rare.
2. The templates are scaled (iterative proportional fitting) so exactly five players are on
   the floor each minute and each player's total matches his sampled minutes for that game.
3. Actual lineups are sampled minute by minute from those probabilities, with a bonus for
   staying on the floor (real stints), a nudge for anyone behind his minutes, and position
   sanity: rarely three bigs, almost always at least one guard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEG_PER_Q = 12
REG_SEGS = 48
OT_SEGS = 5

_ST_Q13 = [1, 1, 1, 1, 1, 1, .8, .5, .3, .3, .3, .4]
_ST_Q24 = [.3, .3, .3, .5, .8, 1, 1, 1, 1, 1, 1, 1]
_BN_Q13 = [.05, .05, .05, .05, .1, .15, .3, .6, 1, 1, 1, 1]
_BN_Q24 = [1, 1, 1, 1, .8, .5, .4, .3, .25, .2, .2, .2]
_AN_Q13 = [1, 1, 1, 1, 1, 1, 1, .6, .25, .2, .4, .8]
_AN_Q24 = [1, 1, .8, .4, .3, .5, .9, 1, 1, 1, 1, 1]
_DEEP = [.3] * 12
# Garbage time: starters play the start of the 4th, then sit.
_ST_BLOWOUT = [1, 1, .8, .5, .2, .05, .02, .02, .02, .02, .02, .02]
_BN_BLOWOUT = [.3, .3, .5, .8, 1, 1, 1, 1, 1, 1, 1, 1]


def role_templates(n_players: int, starter: np.ndarray, deep: np.ndarray,
                   anchor: np.ndarray | None = None) -> np.ndarray:
    """(48, P) on-floor preference by minute for each player's role."""
    st = np.array(_ST_Q13 + _ST_Q24 + _ST_Q13 + _ST_Q24)
    an = np.array(_AN_Q13 + _AN_Q24 + _AN_Q13 + _AN_Q24)
    bn = np.array(_BN_Q13 + _BN_Q24 + _BN_Q13 + _BN_Q24)
    dp = np.array(_DEEP * 4)
    anchor = np.zeros(n_players, bool) if anchor is None else anchor
    t = np.where(starter[None, :], st[:, None], np.where(deep[None, :], dp[:, None], bn[:, None]))
    t = np.where(anchor[None, :], an[:, None], t)
    return t.astype(float)


STARTER_BIAS = 0.8   # sampling slightly favors high-probability players; pre-correct for it


def game_probabilities(targets: np.ndarray, starter: np.ndarray, deep: np.ndarray,
                       anchor: np.ndarray | None = None) -> np.ndarray:
    """Fitted (n,48,P) on-floor probabilities for regulation."""
    adj = np.clip(targets - STARTER_BIAS * starter[None, :], 0, None)
    return fit_probabilities(role_templates(len(starter), starter, deep, anchor), adj)


def blowout_q4_template(starter: np.ndarray) -> np.ndarray:
    return np.where(starter[None, :], np.array(_ST_BLOWOUT)[:, None], np.array(_BN_BLOWOUT)[:, None])


def fit_probabilities(tmpl: np.ndarray, targets: np.ndarray, iters: int = 30) -> np.ndarray:
    """Scale a (S,P) or (n,S,P) template so each segment sums to 5 and each player's column
    sums to his minutes target (n,P). Values are on-floor probabilities in [0, 1]."""
    S = tmpl.shape[-2]
    targets = np.clip(targets, 0, S)
    tot = targets.sum(1, keepdims=True)
    targets = targets * (5 * S / np.maximum(tot, 1e-9))
    pi = np.broadcast_to(tmpl, (targets.shape[0],) + tmpl.shape[-2:]).copy()
    pi *= (targets > 0)[:, None, :]
    pi += 1e-4 * (targets > 0)[:, None, :]
    for _ in range(iters):
        col = pi.sum(1)
        pi *= np.where(col > 0, targets / np.maximum(col, 1e-12), 0)[:, None, :]
        np.minimum(pi, 1, out=pi)
        row = pi.sum(2, keepdims=True)
        pi *= 5 / np.maximum(row, 1e-12)
        np.minimum(pi, 1, out=pi)
    return pi


def sample_lineups(rng, pi: np.ndarray, big: np.ndarray, guard: np.ndarray,
                   prev_on: np.ndarray | None = None) -> np.ndarray:
    """Sample on-floor masks (n,S,P) from fitted probabilities (n,S,P)."""
    n, S, P = pi.shape
    out = np.zeros((n, S, P), bool)
    on = np.zeros((n, P), bool) if prev_on is None else prev_on.copy()
    expected = np.zeros((n, P))
    played = np.zeros((n, P))
    rows = np.arange(n)
    logp = np.log(pi + 1e-6)
    for k in range(S):
        expected += pi[:, k, :]
        behind = expected - played
        pr = logp[:, k, :] + 2.5 * on + 1.2 * behind + rng.gumbel(0, 0.3, (n, P))
        new = np.zeros((n, P), bool)
        nbig = np.zeros(n)
        ngrd = np.zeros(n)
        for slot in range(min(5, P)):
            adj = pr - (nbig >= 2)[:, None] * big[None, :] * 3.0
            adj = adj - (ngrd >= 4)[:, None] * guard[None, :] * 1.0
            if slot == 4:
                adj = adj + (ngrd == 0)[:, None] * guard[None, :] * 3.0
            adj[new] = -np.inf
            pick = adj.argmax(1)
            new[rows, pick] = True
            nbig += big[pick]
            ngrd += guard[pick]
        on = new
        played += new
        out[:, k, :] = new
    return out


def lineup_codes(on: np.ndarray) -> np.ndarray:
    """Encode each (sim, segment) lineup as an integer bitmask."""
    P = on.shape[-1]
    return (on.astype(np.int64) * (1 << np.arange(P, dtype=np.int64))).sum(-1)


def top_lineups(on: np.ndarray, names: list[str], weights: np.ndarray, n_top: int = 12) -> pd.DataFrame:
    """Most-used simulated lineups with minutes per game and each player's usage share in them."""
    n = on.shape[0]
    codes = lineup_codes(on).ravel()
    uniq, counts = np.unique(codes, return_counts=True)
    order = np.argsort(-counts)[:n_top]
    rows = []
    short = [nm.split(" ")[-1] if len(nm.split(" ")) > 1 else nm for nm in names]
    for code, cnt in zip(uniq[order], counts[order]):
        members = [i for i in range(len(names)) if code >> i & 1]
        w = weights[members]
        shares = w / w.sum()
        lead = sorted(zip(members, shares), key=lambda t: -t[1])
        rows.append({
            "Lineup": " · ".join(short[i] for i in members),
            "Min / game": cnt / n,
            "Usage split": ", ".join(f"{short[i]} {s:.0%}" for i, s in lead),
        })
    return pd.DataFrame(rows)


def on_off_usage(on: np.ndarray, weights: np.ndarray, i: int, names: list[str]) -> pd.DataFrame:
    """Player i's share of team plays while on the floor, split by whether each teammate is on."""
    n = on.shape[0]
    tot = (on * weights[None, None, :]).sum(-1)              # (n, S)
    share = np.where(on[..., i], weights[i] / np.maximum(tot, 1e-9), np.nan)
    rows = []
    overall = np.nanmean(share)
    for j, nm in enumerate(names):
        if j == i:
            continue
        both = on[..., i] & on[..., j]
        alone = on[..., i] & ~on[..., j]
        rows.append({
            "Teammate": nm,
            "Min together / game": both.sum() / n,
            "Usage with": np.nanmean(np.where(both, share, np.nan)) if both.any() else np.nan,
            "Usage without": np.nanmean(np.where(alone, share, np.nan)) if alone.any() else np.nan,
        })
    df = pd.DataFrame(rows)
    df["Change"] = df["Usage without"] - df["Usage with"]
    df.attrs["overall"] = overall
    return df.sort_values("Min together / game", ascending=False)


def simulate_on(minutes: np.ndarray, starter: np.ndarray, deep: np.ndarray, big: np.ndarray,
                guard: np.ndarray, anchor: np.ndarray | None = None, n: int = 200,
                seed: int = 11) -> np.ndarray:
    """A fixed sample of regulation rotations for analysis (no game sim)."""
    rng = np.random.default_rng(seed)
    targets = np.repeat(np.asarray(minutes, float)[None, :], n, axis=0)
    return sample_lineups(rng, game_probabilities(targets, starter, deep, anchor), big, guard)


def on_floor_shares(on: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Each player's average share of the five-man total weight while he is on the floor."""
    tot = (on * w[None, None, :]).sum(-1, keepdims=True)
    share = np.where(on, w[None, None, :] / np.maximum(tot, 1e-12), 0.0)
    cnt = on.sum((0, 1))
    return share.sum((0, 1)) / np.maximum(cnt, 1)


def calibrate_weights(on: np.ndarray, rates: np.ndarray, iters: int = 8) -> np.ndarray:
    """Solve for on-floor weights so that, under these rotations, each player's share of plays
    while on the floor reproduces his observed per-minute rate.

    Observed rates were produced next to real teammates (stars mostly beside other starters),
    so plugging them straight into five-man normalization would double count that context.
    Target share_i = 48 * r_i / sum_j(r_j * min_j): every team play belongs to someone on the floor.
    """
    mins = on.sum(1).mean(0)
    target = 48 * rates / max((rates * mins).sum(), 1e-12)
    w = rates.copy() + 1e-9
    for _ in range(iters):
        s = on_floor_shares(on, w)
        w *= np.where(s > 0, target / np.maximum(s, 1e-12), 1.0)
    return w


def calibrate_assists(on: np.ndarray, ast_rates: np.ndarray, make_w: np.ndarray,
                      iters: int = 10, max_rows: int = 6000) -> np.ndarray:
    """Passer weights so simulated assist totals match observed per-minute assist rates.

    In the sim, a make by shooter k is credited to a teammate on the floor (never k himself)
    in proportion to passer weight. This accounts for who is likely to be making shots in
    each lineup and who is left to have passed it.
    """
    rows = on.reshape(-1, on.shape[-1]).astype(float)
    if rows.shape[0] > max_rows:
        rows = rows[np.linspace(0, rows.shape[0] - 1, max_rows).astype(int)]
    mins = on.sum(1).mean(0)
    target = ast_rates * mins
    target = target / max(target.sum(), 1e-12)
    shoot = rows * make_w[None, :]
    p = shoot / np.maximum(shoot.sum(1, keepdims=True), 1e-12)          # (R,P) who made it
    a = ast_rates.copy() + 1e-9
    for _ in range(iters):
        aw = rows * a[None, :]                                          # (R,P)
        W = aw.sum(1, keepdims=True)
        denom = np.maximum(W - aw, 1e-12)                               # (R,K) excluding shooter k
        # credit[r,i] = sum_k p[r,k] * aw[r,i] / (W - aw[r,k]), k != i
        credit = aw * (p / denom).sum(1, keepdims=True) - aw * p / denom
        model = credit.sum(0)
        model = model / max(model.sum(), 1e-12)
        a *= np.where(model > 0, (target / np.maximum(model, 1e-12)) ** 0.9, 1.0)
    return a


def calibrate_rebounds(on: np.ndarray, rates: np.ndarray, length_mult: np.ndarray,
                       p_len: np.ndarray, iters: int = 10, max_rows: int = 4000) -> np.ndarray:
    """Rebound weights so that, over these lineups and this mix of miss lengths, each player's
    share of rebounds matches his real rate. length_mult (P,L) tilts long vs short misses."""
    rows = on.reshape(-1, on.shape[-1]).astype(float)
    if rows.shape[0] > max_rows:
        rows = rows[np.linspace(0, rows.shape[0] - 1, max_rows).astype(int)]
    mins = on.sum(1).mean(0)
    target = rates * mins
    target = target / max(target.sum(), 1e-12)
    w = rates.copy() + 1e-9
    for _ in range(iters):
        credit = np.zeros(len(w))
        for li, pl in enumerate(p_len):
            ww = rows * (w * length_mult[:, li])[None, :]
            credit += pl * (ww / np.maximum(ww.sum(1, keepdims=True), 1e-12)).sum(0)
        credit /= max(credit.sum(), 1e-12)
        w *= np.where(credit > 0, (target / np.maximum(credit, 1e-12)) ** 0.9, 1.0)
    return w
