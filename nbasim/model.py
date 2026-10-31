"""Turns raw blended stats into per-player and per-team profiles the simulator can use.

All player rates are "per minute on the floor". Opponent adjustments are applied in
`matchup_profiles` so the same player profile can be reused against any defense.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import ZONES, Dataset

# Shrinkage: how many league-average attempts get mixed into each player's zone FG%.
ZONE_SHRINK = {"RA": 15, "PAINT": 15, "MID": 20, "C3": 20, "ATB3": 30}
FT_SHRINK = 20
# Chance a made shot in each zone is an and-one.
AND1 = {"RA": 0.07, "PAINT": 0.04, "MID": 0.02, "C3": 0.005, "ATB3": 0.005}
# Relative chance a make in each zone was assisted (3s are assisted far more than rim shots).
AST_ZONE = {"RA": 0.85, "PAINT": 0.72, "MID": 0.70, "C3": 1.45, "ATB3": 1.22}
MAX_MIN = 42.0
ROTATION_SIZE = 13


def minutes_table(ds: Dataset, team: str) -> pd.DataFrame:
    """Season, recent, and projected minutes for everyone on a team's roster with stats.

    "Proj Min" is minutes when he plays: the team's last 5 games he played in, blended with his
    season average. "Play %" is the chance he gets into the game at all, from how many of the
    team's recent games he played. The sim uses both: each night some players DNP, and the
    ones who play get their normal minutes. (Spreading every player's expected minutes over the
    whole 13-man list instead gave a few minutes a night to players who don't play and left the
    regulars 2-4 minutes short, which dragged every counting stat down.)
    """
    p = ds.players[(ds.players["TEAM_ABBREVIATION"] == team) & (ds.players["GP"] > 0)]
    season_start = pd.Timestamp(int(ds.season[:4]), 10, 1)
    tlogs = ds.logs[(ds.logs["TEAM_ABBREVIATION"] == team) & (ds.logs["GAME_DATE"] >= season_start)]
    team_games = list(tlogs.drop_duplicates("GAME_ID")["GAME_ID"])
    recent_games = team_games[:5]
    play_games = team_games[:PLAY_WINDOW]
    rows = []
    for pid, r in p.iterrows():
        lg = ds.logs[(ds.logs["PLAYER_ID"] == pid) & (ds.logs["MIN"] > 0)]
        season_mpg = r["MIN"] / max(r["GP"], 1)
        l5 = lg["MIN"].head(5).mean() if len(lg) else np.nan
        l10 = lg["MIN"].head(10).mean() if len(lg) else np.nan
        sd = lg["MIN"].head(15).std() if len(lg) >= 4 else 6.0
        played = tlogs[(tlogs["PLAYER_ID"] == pid) & (tlogs["MIN"] > 0)].set_index("GAME_ID")["MIN"]
        rec = [played[g] for g in recent_games if g in played.index]
        if rec:
            recent = float(np.mean(rec))
            w_recent = len(rec) / (len(rec) + 4)
            proj = w_recent * recent + (1 - w_recent) * season_mpg
        else:
            recent = np.nan
            proj = season_mpg if np.isnan(l5) else 0.5 * l5 + 0.5 * season_mpg
        n_played = int(played.index.isin(play_games).sum())
        rows.append({
            "PLAYER_ID": pid, "Player": r["PLAYER_NAME"], "GP": int(round(r["GP"])),
            "Season MPG": round(season_mpg, 1),
            "Team L5 MPG": round(recent, 1) if not np.isnan(recent) else None,
            "L10 MPG": round(l10, 1) if not np.isnan(l10) else None,
            "Min SD": round(float(np.clip(sd if not np.isnan(sd) else 5.0, 2.5, 6.5)), 1),
            "Proj Min": round(proj, 1),
            "Play %": float(play_probability(np.array([proj]), np.array([n_played]), len(play_games))[0]),
            "OUT": False,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    exp_min = df["Proj Min"] * df["Play %"]
    df = df.loc[exp_min.sort_values(ascending=False).index].head(ROTATION_SIZE).reset_index(drop=True)
    return df


PLAY_WINDOW = 10          # team games used to estimate how often a player gets in
PLAY_PRIOR_GAMES = 1.0    # weight of the minutes-based prior, in games
MIN_DRESSED = 7          # fewest players who see the floor in a simulated game


def play_probability(proj: np.ndarray, played: np.ndarray, k: int) -> np.ndarray:
    """Chance each player gets into tonight's game.

    Prior from his minutes when he plays (fit to this season's out-of-sample games: ~12 minutes
    plays about half the time, 30+ almost always; injuries included), then updated with how many
    of the team's last `k` games he actually played. The record dominates quickly: a player who
    sat the last few games plays only ~20% of the time next game, one who played them all ~90%+.
    """
    prior = np.clip(0.3 + 0.02 * proj, 0.3, 0.95)
    p = (played + PLAY_PRIOR_GAMES * prior) / (k + PLAY_PRIOR_GAMES)
    return np.round(np.clip(p, 0.05, 1.0), 2)


def normalize_minutes(mins: np.ndarray, total: float = 240.0, cap: float = MAX_MIN) -> np.ndarray:
    """Scale a minutes vector so it sums to `total`, respecting a per-player cap."""
    m = np.clip(np.asarray(mins, dtype=float), 0, None)
    if m.sum() <= 0:
        return m
    for _ in range(20):
        free = m < cap - 1e-9
        fixed = total - m[~free].sum()
        if m[free].sum() <= 0:
            break
        m[free] *= fixed / m[free].sum()
        if (m <= cap + 1e-9).all():
            break
        m = np.minimum(m, cap)
    return m


def normalize_minutes_rows(m: np.ndarray, total: float = 240.0, cap: float = MAX_MIN) -> np.ndarray:
    """normalize_minutes for each row of an (n,P) array."""
    m = np.clip(np.asarray(m, dtype=float), 0, None).copy()
    for _ in range(20):
        free = m < cap - 1e-9
        fixed = total - np.where(free, 0, m).sum(1, keepdims=True)
        fs = np.where(free, m, 0).sum(1, keepdims=True)
        m = np.where(free, m * fixed / np.maximum(fs, 1e-12), m)
        if (m <= cap + 1e-9).all():
            break
        m = np.minimum(m, cap)
    return m


def sample_minutes(rng, proj: np.ndarray, sd: np.ndarray, p_play: np.ndarray, n: int,
                   iters: int = 6) -> np.ndarray:
    """(n,P) minutes per simulated game that sum to 240.

    Each player gets in with probability p_play, then plays about proj / p_play +/- sd.
    Clipping at zero and fitting every game to 240 drags the regulars below their projection
    (bench players' clipped draws only push up), so the centers are solved so that each player's
    average minutes come back out at proj.
    """
    proj = np.asarray(proj, float)
    p = np.clip(np.asarray(p_play, float), 1e-3, 1.0)
    plays = rng.random((n, len(proj))) < p[None, :]
    plays[:, np.argsort(-np.where(p >= 1.0 - 1e-9, proj, -1.0))[:5]] = True   # a full five
    # With injuries thinning the bench, the random DNPs can leave too few players to cover 240
    # minutes; then the next men up by projected minutes play.
    for j in np.argsort(-proj):
        short = plays.sum(1) < MIN_DRESSED
        if not short.any() or proj[j] < 0.5:
            break
        plays[short, j] = True
    plays[:, proj < 0.5] = False
    z = rng.normal(0, 1, (n, len(proj))) * sd[None, :]
    c = proj / p
    m = None
    for _ in range(iters):
        m = normalize_minutes_rows(np.where(plays, np.clip(c[None, :] + z, 1.0, 46), 0))
        got = m.mean(0)
        c = np.clip(c * np.where(got > 1e-6, proj / np.maximum(got, 1e-6), 1.0), 0, 46)
    return m


def fit_rotation(mins: np.ndarray, total: float = 240.0) -> np.ndarray:
    """Fit projected minutes to 240 the way coaches do: when there are too many minutes,
    squeeze the end of the bench rather than the starters (keep the top player's minutes);
    when there are too few (injuries), spread the extra proportionally."""
    m = np.clip(np.asarray(mins, dtype=float), 0, None)
    if m.sum() <= total or m.max() <= 0:
        return normalize_minutes(m, total)
    top = m.max()
    x = m / top
    lo, hi = 1.0, 8.0
    for _ in range(40):
        k = (lo + hi) / 2
        if total * 1.0 / (x ** k).sum() > top:   # top player's share after the power transform
            hi = k
        else:
            lo = k
    out = total * x ** lo / (x ** lo).sum()
    return normalize_minutes(out, total)


@dataclass
class TeamProfile:
    abbr: str
    name: str
    ids: list
    names: list
    proj_min: np.ndarray            # (P,) normalized to 240
    min_sd: np.ndarray              # (P,)
    zone_fga: np.ndarray            # (P,5) attempts per minute
    zone_pct: np.ndarray            # (P,5) shrunk FG%
    trips: np.ndarray               # (P,) non-and-one free-throw trips per minute
    tov: np.ndarray                 # (P,)
    ft_pct: np.ndarray              # (P,)
    oreb: np.ndarray                # (P,) rebound rates per minute (used as shares)
    dreb: np.ndarray
    ast: np.ndarray
    stl: np.ndarray
    blk: np.ndarray
    team: dict = field(default_factory=dict)   # team-level rates
    positions: list = field(default_factory=list)
    big: np.ndarray = None          # (P,) bool: lists C
    guard: np.ndarray = None        # (P,) bool: primary position G
    starter: np.ndarray = None      # (P,) bool
    deep: np.ndarray = None         # (P,) bool: end of bench
    usage_mult: np.ndarray = None   # (P,) with/without correction applied to usage
    wowy: pd.DataFrame = None       # how that correction was derived (for display)
    team_id: int = None
    off_detail: dict = None         # tracking / play-type arrays (see detail.py)
    def_detail: dict = None         # this team's players as defenders
    team_detail: dict = None        # team play-type tendencies and contest quality
    def_base: dict = None           # normal on-floor defensive baselines
    lineup_context: np.ndarray = None  # calibration factor from typical on-floor company
    ast_target: np.ndarray = None   # (P,) real assists per minute (potential-assist blended)
    anchor: np.ndarray = None       # (P,) bool: staggered top-usage starter
    base_share: np.ndarray = None   # (P,) normal share of plays while on the floor
    base_tc: np.ndarray = None      # (P,) normal teammate passing quality around him
    p_play: np.ndarray = None       # (P,) chance he gets into the game


def _group(pos: str) -> str:
    return "C" if pos.startswith("C") else pos[:1] or "F"


def roles(proj: np.ndarray, positions: list, rates: np.ndarray | None = None) -> tuple:
    """Starters (top-5 minutes with position sanity), deep bench, bigs, guards, and the
    anchor: the starter who carries the most of the offense, whose rest gets staggered."""
    big = np.array(["C" in p for p in positions])
    guard = np.array([p.startswith("G") for p in positions])
    order = list(np.argsort(-proj))
    starter = np.zeros(len(proj), bool)
    for i in order:
        if starter.sum() == 5:
            break
        if big[i] and (starter & big).sum() >= 2:
            continue
        starter[i] = True
    if not (starter & guard).any() and guard.any():
        worst = min(np.nonzero(starter)[0], key=lambda i: proj[i])
        starter[worst] = False
        starter[next(i for i in order if guard[i])] = True
    deep = np.zeros(len(proj), bool)
    deep[[i for i in order[10:]]] = True
    deep |= proj < 6
    deep &= ~starter
    anchor = np.zeros(len(proj), bool)
    load = (rates if rates is not None else np.ones(len(proj))) * proj
    anchor[np.argmax(np.where(starter, load, -1))] = True
    return starter, deep, big, guard, anchor


SOFT_MAX_MIN = 38.0   # past this, coaches rarely add minutes for a missing teammate


def redistribute_minutes(active: pd.DataFrame, out: pd.DataFrame, positions: dict,
                         p_play: np.ndarray | None = None) -> np.ndarray:
    """Hand an absent player's minutes to teammates, favoring the same position group.

    Coaches mostly give a missing player's minutes to the bench, not to starters already near
    their limit, so each teammate's share is his minutes times his headroom below SOFT_MAX_MIN
    (minutes when he plays). Spreading in proportion to minutes alone pushed every starter to the
    42-minute cap when two or three teammates were out.
    """
    mins = active["Proj Min"].values.astype(float).copy()
    p = np.ones(len(mins)) if p_play is None else np.clip(np.asarray(p_play, float), 0.05, 1.0)
    groups = np.array([_group(positions[pid]) for pid in active["PLAYER_ID"]])
    for _, r in out.iterrows():
        freed = float(r["Proj Min"])
        if freed <= 0:
            continue
        headroom = np.clip(SOFT_MAX_MIN - mins / p, 1.0, None)
        w = mins * headroom * np.where(groups == _group(positions[r["PLAYER_ID"]]), 2.0, 1.0)
        mins += freed * w / max(w.sum(), 1e-9)
    return fit_rotation(mins)


def _play_rate(pl: pd.DataFrame) -> np.ndarray:
    return ((pl["FGA"] + 0.44 * pl["FTA"] + pl["TOV"]) / pl["MIN"].clip(lower=1)).values


def lineup_weights(ds: Dataset, abbr: str, active: pd.DataFrame, out: pd.DataFrame,
                   tonight_min: np.ndarray, positions: dict, use_wowy: bool = True,
                   tr=None, w_pot: float = 0.3) -> dict:
    """On-floor usage and assist weights for tonight's players.

    1. Calibrate weights on the healthy rotation (everyone, including tonight's OUT players at
       normal minutes) so the lineup model reproduces each player's real usage/assist rates.
    2. Tonight, OUT players are gone from the lineups, so their teammates' shares rise exactly
       in the minutes they would have shared the floor.
    3. With/without evidence: when real games without the absent player show a different usage
       change than the lineup model predicts, nudge toward the evidence (shrunk by sample size).
    """
    from .lineups import calibrate_assists, calibrate_weights, on_floor_shares, simulate_on
    from .wowy import observed_usage_ratio

    n = len(active)
    full = pd.concat([active, out])
    full_pos = [positions[p] for p in full["PLAYER_ID"]]
    full_min = full["Proj Min"].values.astype(float)
    if len(out):
        season_mpg = full["Season MPG"].values.astype(float)
        full_min[n:] = np.maximum(full_min[n:], 0.85 * season_mpg[n:])
    full_min = fit_rotation(full_min)
    fpl = ds.players.loc[full["PLAYER_ID"].values]
    on_full = simulate_on(full_min, *roles(full_min, full_pos, _play_rate(fpl)))
    w_full = calibrate_weights(on_full, _play_rate(fpl))
    fmins = fpl["MIN"].clip(lower=1).values
    make_w = w_full * (fpl["FGM"].values / fmins) / np.maximum(_play_rate(fpl), 1e-9)
    ast_rate = fpl["AST"].values / fmins
    if tr is not None:
        # Potential assists x league conversion: steadier than assists, which depend on
        # teammates making shots.
        pot = tr.players["POT_AST"].reindex(fpl.index).fillna(0).values / fmins
        has = tr.players["POT_AST"].reindex(fpl.index).fillna(0).values > 0
        ast_rate = np.where(has, (1 - w_pot) * ast_rate + w_pot * pot * tr.league["AST_PER_POT"], ast_rate)
    a_full = calibrate_assists(on_full, ast_rate, make_w)
    ast_target = ast_rate[:n]
    usage_w, ast_w = w_full[:n], a_full[:n]
    raw = _play_rate(fpl)[:n]
    context = usage_w / np.maximum(raw, 1e-12)
    context /= np.average(context, weights=tonight_min)

    mult = np.ones(n)
    wowy_df = None
    if len(out):
        on_now = simulate_on(tonight_min, *roles(tonight_min, full_pos[:n], _play_rate(fpl)[:n]))
        model_ratio = on_floor_shares(on_now, usage_w) / np.maximum(on_floor_shares(on_full, w_full)[:n], 1e-12)
        rows = []
        for k, (pid, name) in enumerate(zip(active["PLAYER_ID"], active["Player"])):
            obs, games = observed_usage_ratio(ds, abbr, pid, list(out["PLAYER_ID"])) if use_wowy else (1.0, 0)
            if games:
                mult[k] = float(np.clip(obs / model_ratio[k], 0.8, 1.25))
            rows.append({"Player": name, "Games w/o evidence": games,
                         "Lineup-model usage change": model_ratio[k] - 1,
                         "Observed usage change": obs - 1 if games else np.nan,
                         "Applied correction": mult[k] - 1})
        wowy_df = pd.DataFrame(rows)
    # Baselines for in-game lineup effects: each player's normal share of plays while on the
    # floor, and the normal passing quality of the teammates around him.
    base_share = on_floor_shares(on_full, w_full)[:n]
    rows = on_full.reshape(-1, on_full.shape[-1])
    tot_a = rows @ a_full
    base_tc = np.array([np.mean(tot_a[rows[:, i]] - a_full[i]) if rows[:, i].any() else 1.0
                        for i in range(n)])
    return {"usage_mult": context * mult, "ast": ast_w, "context": context, "wowy": wowy_df,
            "base_share": base_share, "base_tc": np.maximum(base_tc, 1e-9),
            "on_full": on_full, "full_ids": list(full["PLAYER_ID"].values), "ast_target": ast_target}


def team_profile(ds: Dataset, abbr: str, minutes: pd.DataFrame, use_wowy: bool = True,
                 tr=None, w_pot: float = 0.3, w_chances: float = 0.25) -> TeamProfile:
    lg = ds.league
    teams = ds.teams.set_index("TEAM_ABBREVIATION")
    t = teams.loc[abbr]
    allm = minutes.copy()
    allm["OUT"] = allm["OUT"].fillna(False).astype(bool)
    m = allm[~allm["OUT"] & (allm["Proj Min"] > 0)].copy()
    out = allm[allm["OUT"]]
    positions = ds.players["POSITION"].to_dict() if "POSITION" in ds.players else {}
    positions = {p: str(positions.get(p, "F") or "F") for p in allm["PLAYER_ID"]}
    pl = ds.players.loc[m["PLAYER_ID"].values]
    mins = pl["MIN"].clip(lower=1).values[:, None]

    zf = np.stack([pl[f"{z}_FGA"].values for z in ZONES], axis=1)
    zm = np.stack([pl[f"{z}_FGM"].values for z in ZONES], axis=1)
    lg_pct = np.array([lg[f"{z}_PCT"] for z in ZONES])
    k = np.array([ZONE_SHRINK[z] for z in ZONES])
    zone_pct = (zm + k * lg_pct) / (zf + k)

    # Zone data can lag box-score FGA slightly; rescale so zone attempts match total FGA.
    fga = pl["FGA"].values
    scale = np.where(zf.sum(1) > 0, fga / np.maximum(zf.sum(1), 1e-9), 1.0)
    zf = zf * scale[:, None]
    zone_fga = zf / mins

    and1_fta = (zm * scale[:, None] * np.array([AND1[z] for z in ZONES])).sum(1)
    trips = np.clip(pl["FTA"].values - and1_fta, 0, None) / 2.0 / mins[:, 0]
    ft_pct = (pl["FTM"].values + FT_SHRINK * lg["FT_PCT"]) / (pl["FTA"].values + FT_SHRINK)

    # Proj Min is minutes when he plays; the rotation is fit on expected minutes (x Play %).
    # The top five healthy players always play (someone has to start), so they keep Play % = 1;
    # a questionable or doubtful player keeps his reduced Play %.
    p_play = (m["Play %"].fillna(1.0).values.astype(float).clip(0.05, 1.0) if "Play %" in m
              else np.ones(len(m)))
    hurt = (m["Injury"].fillna("").astype(str).str.match(r"(Doubtful|Questionable|Day-To-Day)\b").values
            if "Injury" in m else np.zeros(len(m), bool))
    p_play[np.argsort(-np.where(hurt, -1.0, m["Proj Min"].values * p_play))[:5]] = 1.0
    m["Proj Min"] = m["Proj Min"].values * p_play
    if "Play %" in out:
        out = out.assign(**{"Proj Min": out["Proj Min"] * out["Play %"].fillna(1.0)})
    proj = redistribute_minutes(m, out, positions, p_play)
    pos_list = [positions[p] for p in m["PLAYER_ID"]]
    starter, deep, big, guard, anchor = roles(proj, pos_list, _play_rate(pl))
    lw = lineup_weights(ds, abbr, m, out, proj, positions, use_wowy=use_wowy, tr=tr, w_pot=w_pot)
    team = {
        "PACE": float(t["PACE"]), "OFF_RATING": float(t["OFF_RATING"]),
        "DEF_RATING": float(t["DEF_RATING"]), "OREB_PCT": float(t["OREB_PCT"]),
        "OPP_OREB_PCT": float(t["OPP_OREB_PCT"]), "OPP_TOV_PCT": float(t["OPP_TOV_PCT"]),
        "OPP_FTA_RATE": float(t["OPP_FTA_RATE"]), "AST_PER_FGM": float(t["AST_PER_FGM"]),
        "STL_PER_OPP_TOV": float(t["STL_PER_OPP_TOV"]),
        "BLK_PER_OPP_MISS": float(t["BLK_PER_OPP_MISS"]),
    }
    for z in ZONES:
        team[f"OPP_{z}_PCT"] = float(t[f"OPP_{z}_PCT"])
        team[f"OPP_{z}_FREQ"] = float(t[f"OPP_{z}_FREQ"])
    oreb = pl["OREB"].values / mins[:, 0]
    dreb = pl["DREB"].values / mins[:, 0]
    if tr is not None:
        # Rebound chances x league conversion steady the rebound weights.
        trp = tr.players.reindex(m["PLAYER_ID"].values).fillna(0)
        conv = tr.league["REB_PER_CHANCE"]
        oc = trp["TRK_OREB_CHANCES"].values / mins[:, 0] * conv
        dc = trp["TRK_DREB_CHANCES"].values / mins[:, 0] * conv
        oreb = np.where(oc > 0, (1 - w_chances) * oreb + w_chances * oc, oreb)
        dreb = np.where(dc > 0, (1 - w_chances) * dreb + w_chances * dc, dreb)
    prof = TeamProfile(
        abbr=abbr, name=str(t["TEAM_NAME"]), ids=list(m["PLAYER_ID"].values),
        names=list(pl["PLAYER_NAME"].values), proj_min=proj,
        min_sd=m["Min SD"].values.astype(float), zone_fga=zone_fga, zone_pct=zone_pct,
        trips=trips, tov=pl["TOV"].values / mins[:, 0], ft_pct=ft_pct,
        oreb=oreb, dreb=dreb,
        ast=lw["ast"], stl=pl["STL"].values / mins[:, 0],
        blk=pl["BLK"].values / mins[:, 0], team=team, positions=pos_list, big=big,
        guard=guard, starter=starter, deep=deep, usage_mult=lw["usage_mult"], wowy=lw["wowy"],
        lineup_context=lw["context"], anchor=anchor, base_share=lw["base_share"],
        base_tc=lw["base_tc"], ast_target=lw["ast_target"],
        p_play=p_play)
    prof.team_id = t.name if not isinstance(t.name, str) else ds.teams.index[ds.teams["TEAM_ABBREVIATION"] == abbr][0]
    if tr is not None:
        from types import SimpleNamespace
        from .detail import defense_detail, offense_detail, position_priors, team_detail
        prof.off_detail = offense_detail(ds, tr, prof, position_priors(tr, ds))
        prof.def_detail = defense_detail(ds, tr, prof)
        prof.team_detail = team_detail(tr, prof.team_id)
        # The team's normal on-floor defense (healthy rotation), so a lineup's defensive
        # impact is measured relative to what the team-level stats already include.
        full = defense_detail(ds, tr, SimpleNamespace(ids=lw["full_ids"]))
        rows = lw["on_full"].reshape(-1, lw["on_full"].shape[-1]).astype(float)
        rr = rows @ full["rim_rate"]
        prof.def_base = {
            "d_pm": float(np.mean(rows @ full["d_pm"]) / 5),
            "rim_rate": float(np.mean(rr)),
            "rim_pm": float(np.mean((rows @ (full["rim_rate"] * full["rim_pm"])) / np.maximum(rr, 1e-9))),
            "deflect": float(np.mean(rows @ full["deflect"])),
        }
    return prof


@dataclass
class OffenseVsDefense:
    """Everything one team needs to run its offense against a specific defense."""
    off: TeamProfile
    dfn: TeamProfile
    outcome_rate: np.ndarray        # (P,7) per-minute rates: [TOV, RA, PAINT, MID, C3, ATB3, FT trip]
    make_p: np.ndarray              # (P,5)
    oreb_p: float
    ast_p: np.ndarray               # (5,) chance a make in each zone is assisted (no-tracking fallback)
    stl_p: float
    blk_p: float
    tov_factor: float
    fta_factor: float
    make_factor: np.ndarray         # (P,5) or (5,)
    freq_factor: np.ndarray         # (5,)
    x: dict = None                  # play-type / defender / transition arrays (tracking)


# Offensive-rebound chance by missed-shot zone, relative to the team's average (rim misses
# bounce short and are rebounded by the offense more often; long 3s less).
ZONE_OREB = np.array([1.15, 0.98, 0.86, 0.95, 0.92])
# Rebound "length" by zone: -1 short (rim), 0 medium, +1 long (3s). FT misses use -1.5.
ZONE_LEN = np.array([-1.0, -0.5, 0.0, 1.0, 1.0])
# Transition triggers (scaled to the league's real transition share in offense_vs_defense).
TRANS_TRIGGER = {"steal": 0.60, "dreb": 0.20, "other": 0.06}
TRANS_CAL = 1.36   # empirical: makes the simulated league transition share match Synergy's


def offense_vs_defense(off: TeamProfile, dfn: TeamProfile, lg: dict, home: bool,
                       home_edge: float = 0.012, def_strength: float = 0.8, weights=None,
                       tl: dict | None = None, sched: tuple = (1.0, 1.0)) -> OffenseVsDefense:
    from .factors import Weights
    w = weights or Weights(def_strength=def_strength, home_edge=home_edge)
    d = dfn.team
    lg_pct = np.array([lg[f"{z}_PCT"] for z in ZONES])
    lg_freq = np.array([lg[f"{z}_FREQ"] for z in ZONES])
    opp_pct = np.array([d[f"OPP_{z}_PCT"] for z in ZONES])
    opp_freq = np.array([d[f"OPP_{z}_FREQ"] for z in ZONES])
    have = off.off_detail is not None and dfn.def_detail is not None and tl is not None

    results = opp_pct / lg_pct - 1
    if have:
        # Team shot defense = results (FG% allowed by zone) blended with process (how tightly
        # it contests shots), scaled per shooter by how much tight contests hurt him.
        q = dfn.team_detail["contest_q"] - 1
        sens = off.off_detail["contest_sens"][:, None]
        make_factor = 1 + w.def_strength * ((1 - w.contest_process) * results[None, :]
                                            + w.contest_process * q * sens)
    else:
        make_factor = 1 + w.def_strength * results
    make_factor = make_factor * ((1 + w.home_edge) if home else (1 - w.home_edge)) * sched[0]
    freq_factor = np.sqrt(opp_freq / lg_freq)
    tov_factor = (1 + w.def_strength * (d["OPP_TOV_PCT"] / lg["OPP_TOV_PCT"] - 1)) * sched[1]
    fta_factor = 1 + w.def_strength * (d["OPP_FTA_RATE"] / lg["OPP_FTA_RATE"] - 1)

    # Shift each player's shot diet toward what this defense gives up, keeping total FGA.
    zf = off.zone_fga * freq_factor
    zf *= (off.zone_fga.sum(1) / np.maximum(zf.sum(1), 1e-9))[:, None]
    outcome = np.concatenate([(off.tov * tov_factor)[:, None], zf,
                              (off.trips * fta_factor)[:, None]], axis=1)
    make_p = np.clip(off.zone_pct * make_factor, 0.05, 0.95)

    oreb_p = float(np.clip(off.team["OREB_PCT"] + d["OPP_OREB_PCT"] - lg["OREB_PCT"], 0.15, 0.45))
    zmult = np.array([AST_ZONE[z] for z in ZONES])
    makes = (off.proj_min[:, None] * off.zone_fga * off.zone_pct).sum(0)
    norm = (makes * zmult).sum() / max(makes.sum(), 1e-9)
    ast_p = np.clip(off.team["AST_PER_FGM"] * zmult / norm, 0, 0.97)
    mvd = OffenseVsDefense(off=off, dfn=dfn, outcome_rate=outcome, make_p=make_p, oreb_p=oreb_p,
                           ast_p=ast_p, stl_p=d["STL_PER_OPP_TOV"], blk_p=d["BLK_PER_OPP_MISS"],
                           tov_factor=tov_factor, fta_factor=fta_factor,
                           make_factor=make_factor, freq_factor=freq_factor)
    if have:
        mvd.x = _detail_arrays(off, dfn, outcome, freq_factor, oreb_p, w, tl)
    return mvd


def _detail_arrays(off: TeamProfile, dfn: TeamProfile, outcome: np.ndarray, freq_factor: np.ndarray,
                   oreb_p: float, w, tl: dict) -> dict:
    from .detail import AST_BY_TYPE, HALF_COURT, TI, matchup_matrix
    from .tracking import PLAY_TYPES
    od, dd, dt, ot = off.off_detail, dfn.def_detail, dfn.team_detail, off.team_detail
    P, T = od["share"].shape
    pm = off.proj_min

    # Play-type mix, shifted toward what this defense allows (its scheme).
    share = od["share"] * dt["def_freq_rel"][None, :] ** w.scheme_freq
    share /= share.sum(1, keepdims=True)
    hc = share * HALF_COURT[None, :]
    hc /= np.maximum(hc.sum(1, keepdims=True), 1e-12)

    # The play-type mix the sim will actually produce: half-court types from the player's mix,
    # transition at the league rate x his transition tendency, putbacks from his offensive
    # rebounds x putback rate. Everything below is aligned to this realized mix.
    tot = outcome.sum(1)
    ts_ = od["trans_share"]
    tau = np.clip(tl["Transition_FREQ"] * 0.92 * (ts_ / max(np.average(ts_, weights=pm), 1e-9)), 0, 0.6)
    tau *= min(w.transition, 1.5)
    pbr = np.clip(off.oreb * np.clip(od["putback"] * w.putbacks, 0, 0.85) / np.maximum(tot, 1e-9), 0, 0.3)
    mix = (1 - tau - pbr)[:, None] * hc
    mix[:, TI["Transition"]] += tau
    mix[:, TI["OffRebound"]] += pbr

    # Outcomes by play type, aligned so each player's overall turnover / foul-trip share matches
    # his (opponent-adjusted) box-score rates.
    tgt_tov, tgt_trip = outcome[:, 0] / tot, outcome[:, 6] / tot
    tov = od["tov"] * (tgt_tov / np.maximum((mix * od["tov"]).sum(1), 1e-9))[:, None]
    trip = od["trip"] * (tgt_trip / np.maximum((mix * od["trip"]).sum(1), 1e-9))[:, None]
    tov, trip = np.clip(tov, 0, 0.6), np.clip(trip, 0, 0.35)

    # Zones by play type: keep each play type's shape, but refit so the player's overall zone
    # mix matches his real shot chart, tilted toward what this defense allows.
    from .detail import _ipf
    fga_mix = mix * np.clip(1 - tov - trip, 0.2, 1)
    fga_mix /= fga_mix.sum(1, keepdims=True)
    target = off.zone_fga * freq_factor[None, :]
    target /= np.maximum(target.sum(1, keepdims=True), 1e-12)
    pz = np.zeros_like(od["pz"])
    for i in range(P):
        J = _ipf(fga_mix[i][:, None] * od["pz"][i], fga_mix[i], target[i])
        pz[i] = J / np.maximum(J.sum(1, keepdims=True), 1e-12)

    # Efficiency by play type (player's own profile) x defense's relative play-type strength.
    pt_mult = od["make_mult"] ** w.playtype_eff * (dt["def_ppp_rel"][None, :, None] ** w.scheme_eff)

    # Assisted chance by play type, calibrated to the team's real assists per make.
    prior = np.array([AST_BY_TYPE[t] for t in PLAY_TYPES])
    shot = np.clip(1 - tov - trip, 0.2, 1)
    makes_t = (pm[:, None] * outcome.sum(1)[:, None] * share * shot *
               (pz * off.zone_pct[:, None, :]).sum(2)).sum(0)
    # Solve for the scale with the 97% cap included, so capped play types don't lose assists.
    target = off.team["AST_PER_FGM"] * makes_t.sum()
    lo, hi = 0.2, 5.0
    for _ in range(40):
        c = (lo + hi) / 2
        if (makes_t * np.clip(prior * c, 0, 0.97)).sum() < target:
            lo = c
        else:
            hi = c
    ast_type = np.clip(prior * c, 0, 0.97)
    m_t = makes_t * ast_type
    m_t = m_t / max(m_t.sum(), 1e-12)
    aff = (od["aff"] + 0.2) ** (0.5 * w.assist_chain)
    aff /= np.maximum((aff * m_t[None, :]).sum(1, keepdims=True), 1e-9)
    # Passer calibration over the full chain: lineups on the floor, who makes the shot (and of
    # which play type), how often that type is assisted, and passer affinity by type. Solves a
    # per-passer scale so each player's simulated assists match his real rate.
    from .lineups import simulate_on
    on_a = simulate_on(off.proj_min, off.starter, off.deep, off.big, off.guard, off.anchor, n=60)
    rows = on_a.reshape(-1, P).astype(float)
    fg_t = (pz * off.zone_pct[:, None, :]).sum(2)                    # (P,T)
    mk = mix * np.clip(1 - tov - trip, 0.2, 1) * fg_t                 # makes by type per play
    shooter_w = outcome.sum(1) * (off.usage_mult if off.usage_mult is not None else 1) * mk.sum(1)
    q = mk / np.maximum(mk.sum(1, keepdims=True), 1e-12) * ast_type[None, :]   # P(assisted, type | k makes)
    ps = rows * shooter_w[None, :]
    ps /= np.maximum(ps.sum(1, keepdims=True), 1e-12)                 # (R,K) who made it
    a = off.ast
    tgt = off.ast_target * pm if off.ast_target is not None else a * pm
    tgt = tgt / max(tgt.sum(), 1e-12)
    scale = np.ones(P)

    def credit(sc):
        c = np.zeros(P)
        for ti in range(T):
            wt = q[:, ti]
            if wt.max() <= 0:
                continue
            aw = rows * (a * sc * aff[:, ti])[None, :]                # (R,P) passer weights
            pk = ps * wt[None, :]                                       # (R,K) assisted make by k
            r_ = pk / np.maximum(aw.sum(1, keepdims=True) - aw, 1e-12)  # shooter k can't pass to himself
            c += (aw * (r_.sum(1, keepdims=True) - r_)).sum(0)
        return c / max(c.sum(), 1e-12)
    for _ in range(6):
        c_ = credit(scale)
        scale *= np.where(c_ > 0, (tgt / np.maximum(c_, 1e-12)) ** 0.9, 1.0)
    aff = aff * scale[:, None]

    # Rebound length: offsets from the team's minutes-weighted average distance.
    oo = od["oreb_dist"] - np.average(od["oreb_dist"], weights=pm)
    do_ = dfn.off_detail["dreb_dist"] - np.average(dfn.off_detail["dreb_dist"], weights=dfn.proj_min)
    miss_mix = (pm[:, None] * off.zone_fga * (1 - off.zone_pct)).sum(0)
    zone_oreb = ZONE_OREB / max((ZONE_OREB * miss_mix).sum() / miss_mix.sum(), 1e-9)
    # Normalize the length effect per player so it changes *which* misses he gets, not how
    # many: average over this offense's miss lengths (incl. ~10% FT misses) is 1.
    kappa = 0.27 * w.rebound_length
    lens = np.append(ZONE_LEN, -1.5)
    lw_ = np.append(miss_mix / miss_mix.sum() * 0.9, 0.1)
    from .lineups import calibrate_rebounds, simulate_on
    o_mult = np.exp(kappa * lens[None, :] * oo[:, None])
    d_mult = np.exp(kappa * lens[None, :] * do_[:, None])
    on_o = simulate_on(off.proj_min, off.starter, off.deep, off.big, off.guard, off.anchor, n=80)
    on_d = simulate_on(dfn.proj_min, dfn.starter, dfn.deep, dfn.big, dfn.guard, dfn.anchor, n=80)
    oreb_w = calibrate_rebounds(on_o, off.oreb, o_mult, lw_)
    dreb_w = calibrate_rebounds(on_d, dfn.dreb, d_mult, lw_)

    # Transition: offense push rate x defense's transition allowed, scaled to league share.
    base = 0.085 * TRANS_TRIGGER["steal"] + 0.38 * TRANS_TRIGGER["dreb"] + 0.53 * TRANS_TRIGGER["other"]
    scale = tl["Transition_FREQ"] / base * TRANS_CAL
    tm = (ot["push"] * dt["trans_allowed"]) ** 0.5
    trans = {k: float(np.clip(v * scale * tm ** w.transition, 0, 0.95)) if w.transition > 0 else 0.0
             for k, v in TRANS_TRIGGER.items()}
    ts = od["trans_share"]
    trans_w = (ts / max(np.average(ts, weights=pm), 1e-9)) ** min(w.transition, 1.0)

    Mx = matchup_matrix(_TRACKING_REF["tr"], off, dfn)
    # Transition plays go to players by transition tendency and putbacks to the rebounder, so
    # without compensation half-court creators (few fast breaks, no putbacks) lose plays.
    # Scale half-court usage so each player's total share of plays matches his real usage.
    uw = outcome.sum(1) * (off.usage_mult if off.usage_mult is not None else 1)
    s_ = uw * pm / max((uw * pm).sum(), 1e-12)
    T_ = float(np.average(tau, weights=s_))
    pbw = off.oreb * X_putback(od, w) * pm
    P_ = float(np.average(pbr, weights=s_))
    pb_share = pbw / max(pbw.sum(), 1e-12)
    tw = trans_w / max(np.average(trans_w, weights=s_), 1e-9)
    hc_mult = np.clip((1 - T_ * tw - P_ * pb_share / np.maximum(s_, 1e-9)) / max(1 - T_ - P_, 1e-6), 0.4, 1.8)
    return {
        "hc_mult": hc_mult,
        "hc_share": hc, "share": share, "tov": tov, "trip": trip, "pz": pz, "pt_mult": pt_mult,
        "ast_type": ast_type, "aff": aff, "oreb_off": oo, "dreb_off": do_, "zone_oreb": zone_oreb,
        "kappa": kappa, "oreb_w": oreb_w, "dreb_w": dreb_w, "trans": trans, "trans_w": trans_w,
        "putback": np.clip(od["putback"] * w.putbacks, 0, 0.85),
        "clutch": od["clutch"] ** w.clutch,
        "M": Mx,
        # Defender foul rate relative to the defenders this shooter usually draws (team-level
        # fouling is already in fta_factor), so stars facing top defenders aren't double counted.
        "foul_exp": (Mx / Mx.sum(1, keepdims=True)) @ dd["foul_rel"],
        "d_pm": dd["d_pm"], "rim_pm": dd["rim_pm"], "rim_rate": dd["rim_rate"],
        "deflect": dd["deflect"], "foul_rel": dd["foul_rel"], "def_base": dfn.def_base,
        "w": w, "TI_TRANS": TI["Transition"], "TI_PB": TI["OffRebound"],
    }


def X_putback(od: dict, w) -> np.ndarray:
    return np.clip(od["putback"] * w.putbacks, 0, 0.85)


_TRACKING_REF: dict = {"tr": None}


def set_tracking(tr) -> None:
    """The matchup matrix needs the raw matchup table; keep a reference to the loaded tracking."""
    _TRACKING_REF["tr"] = tr


def expected_pace(home: TeamProfile, away: TeamProfile, lg: dict) -> float:
    """Classic pace estimate: product of team paces over league pace."""
    return home.team["PACE"] * away.team["PACE"] / lg["PACE"]


def player_efficiency(ds: Dataset, pid) -> dict:
    """Efficiency + shot diet summary for one player (blended sample)."""
    r = ds.players.loc[pid]
    lg = ds.league
    fga, fta = r["FGA"], r["FTA"]
    ts = r["PTS"] / max(2 * (fga + 0.44 * fta), 1e-9)
    efg = (r["FGM"] + 0.5 * r["FG3M"]) / max(fga, 1e-9)
    # Usage: share of team plays used while on the floor (approximate from possessions).
    plays = fga + 0.44 * fta + r["TOV"]
    usg = plays / max(r["POSS"], 1e-9) if r["POSS"] > 0 else np.nan
    zfga = np.array([r[f"{z}_FGA"] for z in ZONES])
    zfgm = np.array([r[f"{z}_FGM"] for z in ZONES])
    diet = pd.DataFrame({
        "Zone": ZONES,
        "FGA/G": zfga / max(r["GP"], 1),
        "Share": zfga / max(zfga.sum(), 1e-9),
        "League share": [lg[f"{z}_FREQ"] for z in ZONES],
        "FG%": np.where(zfga > 0, zfgm / np.maximum(zfga, 1e-9), np.nan),
        "League FG%": [lg[f"{z}_PCT"] for z in ZONES],
    })
    return {
        "TS%": ts, "eFG%": efg, "USG%": usg, "FTr": fta / max(fga, 1e-9),
        "3PAr": r["FG3A"] / max(fga, 1e-9), "PTS/36": 36 * r["PTS"] / max(r["MIN"], 1),
        "AST/TOV": r["AST"] / max(r["TOV"], 1e-9), "diet": diet,
    }
