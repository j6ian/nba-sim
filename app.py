"""NBA Matchup Simulator — Streamlit front end.

Run with:  .venv/bin/streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import warnings

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

from nbasim.data import ZONE_LABELS, ZONES, current_season, load_dataset
from nbasim import odds as oddsapi
from nbasim import schedule as slate_schedule
from nbasim import injuries
from nbasim.detail import schedule_context, schedule_factor
from nbasim.factors import NOT_AVAILABLE, REGISTRY, Weights
from nbasim.model import expected_pace, minutes_table, player_efficiency, set_tracking, team_profile
from nbasim.tracking import PLAY_TYPES, PT_LABELS, load_tracking
from nbasim.props import PROP_STATS, default_line, evaluate_prop, log_values, prob_to_american
from nbasim.lineups import on_off_usage, top_lineups
from nbasim.sim import simulate
from nbasim.wowy import teammate_table

st.set_page_config(page_title="NBA Matchup Simulator", page_icon="🏀", layout="wide")

BOX = ["MIN", "PTS", "REB", "AST", "FG3M", "FGA", "FG3A", "FTA", "STL", "BLK", "TOV"]


# --------------------------------------------------------------------------- data

@st.cache_resource(show_spinner=False)
def get_dataset(season: str, prior_games: float, refresh_token: int):
    return load_dataset(season, prior_games=prior_games, refresh=refresh_token > 0)


@st.cache_data(show_spinner=False)
def get_minutes(_ds, team: str, built_at: str):
    return minutes_table(_ds, team)


@st.cache_resource(show_spinner=False)
def get_tracking(_ds, built_at: str, prior_games: float, refresh_token: int):
    return load_tracking(_ds, prior_games=prior_games, refresh=refresh_token > 0)


@st.cache_data(show_spinner=False)
def get_profile(_ds, _tr, team: str, minutes: pd.DataFrame, built_at: str, w_pot: float, w_ch: float,
                model_version: float = 0.0):
    return team_profile(_ds, team, minutes, tr=_tr, w_pot=w_pot, w_chances=w_ch)


@st.cache_data(show_spinner=False, ttl=300)
def get_injuries():
    return injuries.fetch_report()


@st.cache_data(show_spinner=False, ttl=120)
def get_slate(date: dt.date, team_names: tuple):
    return slate_schedule.games_on(date, dict(team_names))


def numeric(df: pd.DataFrame, skip=("Player", "Teammate", "Teammate out", "Lineup", "Usage split")) -> pd.DataFrame:
    """Coerce mixed object columns to floats so missing values render as '—'."""
    df = df.copy()
    for c in df.columns:
        if c not in skip:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


# --------------------------------------------------------------------------- sidebar

st.sidebar.title("🏀 NBA Matchup Sim")
season = st.sidebar.text_input("Season", current_season())
game_date = st.sidebar.date_input("Game date", dt.date.today(),
                                   help="Used for rest days, back-to-backs and travel.")
with st.sidebar.expander("Model settings"):
    prior_games = st.slider("Last season counts as N games", 0, 60, 20,
                            help="Early in the season, current-season samples are tiny. Last "
                                 "season's totals are blended in, worth at most this many games.")
    pace_override = st.number_input("Pace override (0 = model)", 0.0, 130.0, 0.0, 0.5)
with st.sidebar.expander("Factor weights"):
    st.caption("How much each factor counts. 0 turns it off. Defaults and the reasoning behind "
               "them are on the 🧠 Model & Data tab.")
    defaults = Weights()
    wvals = {}
    group = None
    for key, grp, label, mx, *_ in REGISTRY:
        if grp != group:
            st.markdown(f"**{grp}**")
            group = grp
        d = getattr(defaults, key)
        step = 0.002 if mx <= 0.05 else 0.01 if mx <= 0.5 else 0.05
        wvals[key] = st.slider(label, 0.0, float(mx), float(d), step, key=f"w_{key}",
                               format="%.3f" if mx <= 0.05 else "%.2f")
    weights = Weights(**wvals)
def_strength, home_edge = weights.def_strength, weights.home_edge
use_injuries = st.sidebar.toggle(
    "Use injury report", True,
    help="From ESPN's NBA injury report. Out players are ticked OUT; questionable, doubtful and "
         "day-to-day players get a lower Play % (chance they play). Turn off to set everything "
         "by hand in ⏱ Rotations.")
n_sims = st.sidebar.select_slider("Simulations", [500, 1000, 2000, 3000, 5000, 10000], 3000)
if "refresh" not in st.session_state:
    st.session_state.refresh = 0
if st.sidebar.button("Refresh data from NBA.com", width="stretch"):
    st.session_state.refresh += 1
    st.cache_resource.clear()
    st.cache_data.clear()

with st.spinner("Loading NBA data (first run takes ~1 minute, then it's cached)..."):
    try:
        ds = get_dataset(season, float(prior_games), st.session_state.refresh)
    except Exception as exc:
        st.error(f"Couldn't load data from stats.nba.com: {exc}")
        st.stop()

with st.spinner("Loading player-tracking, play-type and matchup data (first run takes a few minutes)..."):
    try:
        tr = get_tracking(ds, ds.built_at, float(prior_games), st.session_state.refresh)
    except Exception as exc:
        st.error(f"Couldn't load tracking data from stats.nba.com: {exc}")
        st.stop()
set_tracking(tr)

teams_df = ds.teams.sort_values("TEAM_NAME")
abbrs = list(teams_df["TEAM_ABBREVIATION"])
names = dict(zip(teams_df["TEAM_ABBREVIATION"], teams_df["TEAM_NAME"]))
fmt = lambda a: f"{names[a]} ({a})"

# The day's games. The first one opens by default; clicking a game opens it below.
slate, slate_source = get_slate(game_date, tuple(sorted(names.items())))
if st.session_state.get("slate_date") != game_date:
    st.session_state.slate_date = game_date
    on_slate = {(g["away"], g["home"]) for g in slate}
    if slate and (st.session_state.get("away_sel"), st.session_state.get("home_sel")) not in on_slate:
        st.session_state.away_sel, st.session_state.home_sel = slate[0]["away"], slate[0]["home"]
for k, d in (("away_sel", "DEN"), ("home_sel", "OKC")):
    if st.session_state.get(k) not in abbrs:
        st.session_state[k] = d if d in abbrs else abbrs[0 if k == "away_sel" else 1]


def open_game(a: str, h: str):
    st.session_state.away_sel, st.session_state.home_sel = a, h


away = st.sidebar.selectbox("Away team", abbrs, key="away_sel", format_func=fmt)
home = st.sidebar.selectbox("Home team", abbrs, key="home_sel", format_func=fmt)
st.sidebar.caption(f"Data: {ds.season} blended with {ds.prior_season} · built {ds.built_at}")

# Odds settings are shared by the Props tab and the slate's "Pull props for all games".
for k, d in (("odds_key", oddsapi.api_key() or ""), ("odds_books", list(oddsapi.DEFAULT_BOOKS)),
             ("odds_markets", list(oddsapi.DEFAULT_MARKETS)), ("slate_props", {})):
    st.session_state.setdefault(k, d)


def slate_key(a: str, h: str) -> str:
    return f"{a}_{h}_{game_date}"


def load_game_lines(games: list[dict]) -> tuple[dict, dict | None, str | None]:
    """Spread, total and moneyline for every game, refreshed automatically (3 credits, at most
    once every 10 minutes while the app is in use)."""
    key, books = st.session_state.odds_key, st.session_state.odds_books
    if not (key and books and games):
        return {}, None, None
    try:
        events, q = oddsapi.fetch_game_lines(key, books)
    except oddsapi.OddsError as exc:
        return {}, None, str(exc)
    out = {}
    for g in games:
        ev = oddsapi.find_event(events, names[g["home"]], names[g["away"]], game_date)
        if ev is not None:
            out[slate_key(g["away"], g["home"])] = oddsapi.game_line_summary(ev)
    return out, q, None


def fetch_slate_props(games: list[dict]):
    """Player props for each game on the slate, for the prop types picked above."""
    key, books, mkts = (st.session_state.odds_key, st.session_state.odds_books,
                        st.session_state.odds_markets)
    used, remaining, missing, errors = 0, None, [], []
    bar = st.progress(0.0, text="Finding games...")
    try:
        events, _ = oddsapi.list_events(key)
    except oddsapi.OddsError as exc:
        st.session_state.slate_msg = ("error", str(exc))
        bar.empty()
        return
    fetched = dt.datetime.now().strftime("%H:%M")
    for i, g in enumerate(games):
        a, h = g["away"], g["home"]
        bar.progress((i + 1) / len(games), text=f"Fetching props: {a} @ {h} ({i + 1}/{len(games)})")
        ev = oddsapi.find_event(events, names[h], names[a], game_date)
        if ev is None:
            missing.append(f"{a} @ {h}")
            continue
        try:
            data, q = oddsapi.fetch_props(key, ev["id"], mkts, books)
        except oddsapi.OddsError as exc:
            errors.append(f"{a} @ {h}: {exc}")
            if "429" in str(exc):
                break
            continue
        used += 0 if q.get("cached") else int(q.get("last") or 0)
        remaining = q.get("remaining") or remaining
        raw = oddsapi.props_frame(data)
        if raw.empty:
            missing.append(f"{a} @ {h}")
            continue
        st.session_state.slate_props[slate_key(a, h)] = {"raw": raw, "quota": q, "event": ev,
                                                         "fetched": fetched}
        # Drop lines loaded earlier for this game so the Props tab picks up the new ones.
        st.session_state.pop(f"live_{slate_key(a, h)}", None)
        st.session_state.pop(f"props_{a}_{h}_live", None)
    bar.empty()
    n_ok = sum(slate_key(g["away"], g["home"]) in st.session_state.slate_props for g in games)
    msg = (f"Loaded {', '.join(mkts)} props for {n_ok} of {len(games)} games at {fetched} · "
           f"used {used} credits · credits left: {(oddsapi.latest_quota() or {}).get('remaining', remaining)}")
    if missing:
        msg += f". No props posted yet: {', '.join(missing)}"
    if errors:
        msg += ". Errors: " + "; ".join(errors)
    st.session_state.slate_msg = ("error" if errors and not n_ok else "success", msg)


inj_report, inj_error, inj_stamp = get_injuries() if use_injuries else (None, None, "")


def team_injuries(team: str) -> pd.DataFrame:
    """Tonight's injury designations for a team (empty when the report is off)."""
    if inj_report is None or inj_report.empty:
        return pd.DataFrame(columns=["Player", "Status", "note"])
    return injuries.for_team(inj_report, names[team], game_date)


def card_injuries(team: str, min_mpg: float = 15.0) -> str:
    """'WAS: Johnson Q, Poole O' for rotation players (min_mpg+ this season) who are hurt."""
    r = team_injuries(team)
    r = r[r["Status"] != "Available"]
    if r.empty:
        return ""
    pl = ds.players[ds.players["TEAM_ABBREVIATION"] == team]
    mpg = dict(zip(pl["PLAYER_NAME"], pl["MIN"] / pl["GP"].clip(lower=1)))
    mapping = oddsapi.match_players(r["Player"], list(mpg))
    hurt = [(row["Player"], row["Status"]) for _, row in r.iterrows()
            if mpg.get(mapping.get(row["Player"]), 0) >= min_mpg]
    if not hurt:
        return ""
    return f"{team}: " + ", ".join(f"{p.split()[-1]} {injuries.SHORT.get(st_, st_)}" for p, st_ in hurt)


def american(o) -> str:
    return "—" if o is None else f"{int(round(o)):+d}"


game_lines, gl_quota, gl_error = load_game_lines(slate)
st.session_state.slate_lines = game_lines


def show_credits():
    if not st.session_state.odds_key or gl_error:
        return
    q = oddsapi.latest_quota()
    left = f"credits left: {q['remaining']} (as of {q['checked']})" if q else ""
    updated = f"last updated {gl_quota.get('fetched', '')} · " if gl_quota else ""
    credits_box.caption("Spreads, totals and moneylines update automatically (3 credits, at most "
                        f"once every 10 minutes) · {updated}{left}")


day_label = "Today's games" if game_date == dt.date.today() else f"Games on {game_date:%a, %b %-d}"
with st.expander(f"📅 {day_label} ({len(slate)})", expanded=True):
    if not slate:
        st.caption(f"No games found for this date ({slate_source}). Change the game date in the "
                   "sidebar, or pick teams there.")
    if not st.session_state.odds_key:
        st.caption("Add your Odds API key in 💰 Player Props → Live sportsbook lines to get "
                   "spreads, totals and player props.")
    elif gl_error:
        st.warning(f"Couldn't update spreads and totals: {gl_error}")
    # Filled again at the end of the script, after any fetch on this run, so the credit count
    # is always the latest.
    credits_box = st.empty()
    show_credits()

    mkts = st.pills("Player props to pull", list(oddsapi.MARKETS), selection_mode="multi",
                    key="odds_markets", format_func=lambda m: oddsapi.MARKET_LABELS.get(m, m),
                    help="1 credit per prop type per game, for up to 10 books. Prop types a book "
                         "hasn't posted yet aren't charged.") or []
    if slate:
        cost = len(mkts) * len(slate)
        b1, b2 = st.columns([1, 3])
        if b1.button(f"💰 Pull props for all {len(slate)} games", type="primary", width="stretch",
                     disabled=not (st.session_state.odds_key and mkts and st.session_state.odds_books)):
            fetch_slate_props(slate)
        b2.caption(f"**Cost: up to {cost} credits** ({len(mkts)} prop type{'s' if len(mkts) != 1 else ''} "
                   f"× {len(slate)} games). Pulling again within 10 minutes is free. Pick a game "
                   "and use 💰 Player Props → Fetch live lines to pull just that game "
                   f"({len(mkts)} credits). Books are set there too.")
        if "slate_msg" in st.session_state:
            kind, msg = st.session_state.slate_msg
            (st.error if kind == "error" else st.success)(msg)
    per_row = 5
    for i in range(0, len(slate), per_row):
        cols = st.columns(per_row)
        for col, g in zip(cols, slate[i:i + per_row]):
            a, h = g["away"], g["home"]
            selected = (a, h) == (away, home)
            with col.container(border=True):
                if g["status"] >= 2 and g["away_score"] is not None:
                    st.markdown(f"**{a}** {g['away_score']}  \n**{h}** {g['home_score']}")
                else:
                    ar = f" ({g['away_record']})" if g["away_record"] else ""
                    hr = f" ({g['home_record']})" if g["home_record"] else ""
                    st.markdown(f"**{a}**{ar}  \n@ **{h}**{hr}")
                info = [slate_schedule.status_line(g)]
                gl = game_lines.get(slate_key(a, h))
                if gl and gl["home_spread"] is not None:
                    info.append(f"{h} {gl['home_spread']:+g}" if gl["home_spread"] else "PK")
                if gl and gl["total"] is not None:
                    info.append(f"O/U {gl['total']:g}")
                if gl and gl["home_ml"] is not None and gl["away_ml"] is not None:
                    info.append(f"ML {a} {american(gl['away_ml'])} / {h} {american(gl['home_ml'])}")
                sp = st.session_state.slate_props.get(slate_key(a, h))
                if sp is not None:
                    info.append(f"📡 {len(sp['raw'])} props")
                st.caption(" · ".join(info))
                inj = [x for x in (card_injuries(a), card_injuries(h)) if x]
                if inj:
                    st.caption("🚑 " + " · ".join(inj))
                st.button("✓ Open" if selected else "Open", key=f"open_{g['game_id']}_{a}{h}",
                          on_click=open_game, args=(a, h), width="stretch",
                          type="primary" if selected else "secondary", disabled=selected)
    if slate:
        st.caption(f"Schedule from {slate_source}. Open a game to simulate it in the tabs below.")

if home == away:
    st.warning("Pick two different teams.")
    st.stop()

T = ds.teams.set_index("TEAM_ABBREVIATION")
st.title(f"{names[away]} @ {names[home]}")

tab_sim, tab_props, tab_player, tab_match, tab_rot, tab_watch, tab_model = st.tabs(
    ["🎲 Game Sim", "💰 Player Props", "🔍 Player Deep Dive", "📊 Matchup", "⏱ Rotations",
     "📺 Watch a Game", "🧠 Model & Data"])

sched = {}
for team in (away, home):
    ctx = schedule_context(ds, team, game_date, home)
    mk, tv, why = schedule_factor(ctx, team == home, weights.rest_travel)
    sched[team] = {"ctx": ctx, "factor": (mk, tv), "why": why}

# --------------------------------------------------------------------------- rotations (runs first)

with tab_rot:
    st.markdown("Projected minutes drive everything. **Proj Min** blends the team's last 5 games "
                "this season (DNPs count as 0) with the season average. Edit minutes or tick "
                "**OUT** for injuries: the absent player's minutes go mostly to teammates at his "
                "position, and his shots go to whoever shares the floor in his place. "
                "Minutes are fit to 240 by trimming the end of the bench first. **Play %** is how often "
                "he gets in at all; props only count simulated games he plays, like a book voiding a DNP.")
    edited = {}
    cols = st.columns(2)
    for col, team in zip(cols, (away, home)):
        with col:
            st.subheader(names[team])
            base = get_minutes(ds, team, ds.built_at)
            tinj = team_injuries(team)
            base = injuries.apply(base, tinj)
            # A new key when the report changes for this team, so new statuses show up
            # (hand edits otherwise stay as they are).
            inj_tag = hashlib.md5(str(sorted(zip(tinj["Player"], tinj["Status"]))).encode()).hexdigest()[:8]
            edited[team] = st.data_editor(
                base, key=f"rot_{team}_{inj_tag}", hide_index=True, width="stretch",
                column_order=["Player", "OUT", "Injury", "Proj Min", "Play %", "Min SD", "Season MPG",
                              "Team L5 MPG", "L10 MPG", "GP"],
                disabled=["Player", "Injury", "Season MPG", "Team L5 MPG", "L10 MPG", "GP"],
                column_config={
                    "Injury": st.column_config.TextColumn(
                        width="medium", help="Tonight's status from ESPN's injury report and the "
                                             "latest news note. Hover a cell to read it all."),
                    "Proj Min": st.column_config.NumberColumn(min_value=0.0, max_value=48.0, step=0.5),
                    "Play %": st.column_config.NumberColumn(
                        help="Chance he plays at all (recent DNPs). Minutes when he plays = Proj Min / Play %.",
                        min_value=0.05, max_value=1.0, step=0.05, format="%.2f"),
                    "Min SD": st.column_config.NumberColumn(help="Game-to-game minutes SD", step=0.5),
                    "GP": st.column_config.NumberColumn(help="Blended games (prior season down-weighted)"),
                })
    if use_injuries:
        mult = ", ".join(f"{k.lower()} x{v:g}" for k, v in injuries.PLAY_MULT.items() if v < 1)
        st.caption(f"🚑 Injury report from ESPN (updated {inj_stamp or 'unknown'}). Out players are "
                   f"ticked OUT. Others' Play % is multiplied by their status ({mult}); a news note "
                   "about a different game counts as day-to-day. Untick OUT or edit Play % to "
                   "override, or turn the report off in the sidebar."
                   + (f" ⚠️ {inj_error}" if inj_error else ""))

with st.spinner("Building rotations and lineup weights..."):
    _mv = max(os.path.getmtime(f) for f in __import__("glob").glob(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "nbasim", "*.py")))
    home_prof = get_profile(ds, tr, home, edited[home], ds.built_at, weights.potential_ast,
                            weights.rebound_chances, _mv)
    away_prof = get_profile(ds, tr, away, edited[away], ds.built_at, weights.potential_ast,
                            weights.rebound_chances, _mv)


def role_label(prof, i):
    if prof.anchor[i]:
        return "Starter (staggered)"
    if prof.starter[i]:
        return "Starter"
    return "Deep bench" if prof.deep[i] else "Bench"


with tab_rot:
    c1, c2 = st.columns(2)
    for col, prof in ((c1, away_prof), (c2, home_prof)):
        col.caption(f"How the sim uses {prof.abbr}'s rotation")
        col.dataframe(pd.DataFrame({
            "Player": prof.names, "Pos": prof.positions,
            "Role": [role_label(prof, i) for i in range(len(prof.names))],
            "Sim Min": prof.proj_min.round(1),
            "On-floor usage": prof.base_share,
        }).style.format({"On-floor usage": "{:.0%}", "Sim Min": "{:.1f}"}), hide_index=True, width="stretch")
    st.caption("**On-floor usage** = normal share of the team's plays while he's on the court. "
               "In any given lineup, shares are re-split among the five actually on the floor.")

# --------------------------------------------------------------------------- run sims

MODEL_VERSION = max(os.path.getmtime(f) for f in __import__("glob").glob(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "nbasim", "*.py")))
sim_key = hashlib.md5(f"{MODEL_VERSION}".encode() + pd.util.hash_pandas_object(pd.concat([edited[home], edited[away]])).values.tobytes()
                      + f"{home}{away}{n_sims}{weights.as_dict()}{game_date}{pace_override}{prior_games}{ds.built_at}".encode()
                      ).hexdigest()
cache = st.session_state.setdefault("sims", {})
if sim_key not in cache:
    with st.spinner(f"Simulating {n_sims:,} games..."):
        cache[sim_key] = simulate(home_prof, away_prof, ds.league, n=n_sims, seed=7,
                                  pace_override=pace_override or None, weights=weights,
                                  tl=tr.league, sched_home=sched[home]["factor"],
                                  sched_away=sched[away]["factor"])
res = cache[sim_key]
hs, as_ = res.score[home], res.score[away]
margin = hs - as_
total = hs + as_


def box_projection(team: str, prof) -> pd.DataFrame:
    s = res.players[team]
    rows = []
    for i, nm in enumerate(prof.names):
        row = {"Player": nm}
        for k in BOX:
            row[k] = s[k][:, i].mean()
        row["PRA"] = (s["PTS"] + s["REB"] + s["AST"])[:, i].mean()
        row["PTS 25-75%"] = f"{np.percentile(s['PTS'][:, i], 25):.0f}–{np.percentile(s['PTS'][:, i], 75):.0f}"
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- lineups (after sim)

with tab_rot:
    st.subheader("Simulated lineups")
    st.markdown("Every simulated game has a minute-by-minute rotation. Starters open halves and "
                "close games, the bench covers the middle, and the top-usage starter is staggered "
                "to play with the bench. Each possession, usage, assists, rebounds, steals and "
                "blocks are split among the five on the floor. A player's shot share rises when "
                "high-usage teammates sit. Carrying more load than usual costs a little "
                "efficiency, and better passers around a shooter make his looks easier.")
    c1, c2 = st.columns(2)
    for col, prof in ((c1, away_prof), (c2, home_prof)):
        col.caption(f"{prof.abbr}: most-used lineups")
        col.dataframe(top_lineups(res.on[prof.abbr], prof.names, res.play_w[prof.abbr], 10)
                      .style.format({"Min / game": "{:.1f}"}), hide_index=True, width="stretch")
    for prof in (away_prof, home_prof):
        if prof.wowy is not None:
            st.caption(f"{prof.abbr}: usage adjustments for tonight's absences. The lineup model's "
                       "change is compared with real games played without those teammates. "
                       "The correction moves toward the evidence, scaled by sample size.")
            st.dataframe(numeric(prof.wowy).style.format({
                "Lineup-model usage change": "{:+.0%}", "Observed usage change": "{:+.0%}",
                "Applied correction": "{:+.0%}"}, na_rep="—"), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- game sim tab

with tab_sim:
    hw = (margin > 0).mean()
    c = st.columns(6)
    c[0].metric(f"{away} win %", pct(1 - hw), help="Share of simulated games won")
    c[1].metric(f"{home} win %", pct(hw))
    c[2].metric(f"{away} points", f"{as_.mean():.1f}")
    c[3].metric(f"{home} points", f"{hs.mean():.1f}")
    c[4].metric(f"Spread ({home})", f"{-margin.mean():+.1f}", help="Negative = home favored")
    c[5].metric("Total", f"{total.mean():.1f}")

    c = st.columns(5)
    c[0].metric("Possessions / team", f"{res.poss.mean():.1f}",
                help=f"Pace estimate {res.exp_pace:.1f} = {home} pace × {away} pace ÷ league pace")
    c[1].metric(f"{home} pace", f"{T.loc[home, 'PACE']:.1f}")
    c[2].metric(f"{away} pace", f"{T.loc[away, 'PACE']:.1f}")
    c[3].metric("Total FGA", f"{res.team_stats[home]['FGA'].mean() + res.team_stats[away]['FGA'].mean():.1f}")
    c[4].metric("Overtime %", pct((res.ot > 0).mean()))

    st.subheader("Compare to the market")
    m1, m2, m3 = st.columns(3)
    # Pre-filled with the books' consensus when odds were fetched, else the sim's own numbers.
    mkt = st.session_state.slate_lines.get(slate_key(away, home)) or {}
    sp0 = mkt.get("home_spread") if mkt.get("home_spread") is not None else float(np.round(-margin.mean() * 2) / 2)
    tot0 = mkt.get("total") if mkt.get("total") is not None else float(np.round(total.mean() * 2) / 2)
    spread_line = m1.number_input(f"{home} spread" + (" (market)" if mkt.get("home_spread") is not None else ""),
                                  value=float(sp0), step=0.5)
    total_line = m2.number_input("Game total" + (" (market)" if mkt.get("total") is not None else ""),
                                 value=float(tot0), step=0.5)
    cover = (margin + spread_line > 0).mean()
    over = (total > total_line).mean()
    m3.markdown(f"**{home} {spread_line:+.1f} covers:** {pct(cover)} (fair {prob_to_american(cover)})  \n"
                f"**{away} {-spread_line:+.1f} covers:** {pct(1 - cover)}  \n"
                f"**Over {total_line}:** {pct(over)} (fair {prob_to_american(over)})  \n"
                f"**Under {total_line}:** {pct(1 - over)}")

    g1, g2, g3 = st.columns(3)
    def hist(v, title, rule=None, step=2):
        chart = alt.Chart(pd.DataFrame({"v": v})).mark_bar(binSpacing=0, opacity=0.85).encode(
            x=alt.X("v:Q", bin=alt.Bin(step=step), title=title), y=alt.Y("count():Q", title="Sims"))
        if rule is not None:
            chart += alt.Chart(pd.DataFrame({"r": [rule]})).mark_rule(color="red", strokeWidth=2).encode(x="r:Q")
        return chart

    g1.altair_chart(hist(total, "Total points", total_line, 4).properties(height=230), use_container_width=True)
    g2.altair_chart(hist(margin, f"{home} margin", -spread_line, 3).properties(height=230), use_container_width=True)
    g3.altair_chart(hist(res.poss, "Possessions per team", None, 1).properties(height=230), use_container_width=True)

    st.subheader("Team shot volume & pace")
    rows = []
    for team in (away, home):
        ts = res.team_stats[team]
        rows.append({"Team": team, "PTS": res.score[team].mean(), "FGA": ts["FGA"].mean(),
                     "3PA": ts["FG3A"].mean(), "3PM": ts["FG3M"].mean(), "FTA": ts["FTA"].mean(),
                     "FG%": ts["FGM"].sum() / ts["FGA"].sum(), "3P%": ts["FG3M"].sum() / max(ts["FG3A"].sum(), 1),
                     "OREB": ts["OREB"].mean(), "REB": ts["REB"].mean(), "AST": ts["AST"].mean(),
                     "TOV": ts["TOV"].mean(), "STL": ts["STL"].mean(), "BLK": ts["BLK"].mean(),
                     "ORtg": 100 * res.score[team].mean() / res.poss.mean()})
    st.dataframe(pd.DataFrame(rows).style.format({c: "{:.1f}" for c in rows[0] if c not in ("Team", "FG%", "3P%")}
                                                 | {"FG%": "{:.1%}", "3P%": "{:.1%}"}),
                 hide_index=True, width="stretch")

    q = pd.DataFrame({team: res.quarters[team].mean(0) for team in (away, home)},
                     index=["Q1", "Q2", "Q3", "Q4", "OT (avg)"]).T
    st.caption("Average points by quarter")
    st.dataframe(q.style.format("{:.1f}"), width="stretch")

    st.subheader("Where the shots come from (simulated per game)")
    zrows = []
    for team in (away, home):
        z = res.zone_fga[team].sum(0)
        zm = res.zone_fgm[team].sum(0)
        for k, zone in enumerate(ZONES):
            zrows.append({"Team": team, "Zone": ZONE_LABELS[zone], "FGA": z[k], "FG%": zm[k] / max(z[k], 1e-9)})
    zdf = pd.DataFrame(zrows)
    st.altair_chart(alt.Chart(zdf).mark_bar().encode(
        x=alt.X("Zone:N", sort=[ZONE_LABELS[z] for z in ZONES]), xOffset="Team:N", y="FGA:Q",
        color="Team:N", tooltip=["Team", "Zone", alt.Tooltip("FGA:Q", format=".1f"),
                                 alt.Tooltip("FG%:Q", format=".1%")]).properties(height=260),
        use_container_width=True)

    st.subheader("Projected box scores (simulation means)")
    for team, prof in ((away, away_prof), (home, home_prof)):
        st.caption(names[team])
        st.dataframe(box_projection(team, prof).style.format({k: "{:.1f}" for k in BOX + ["PRA"]}),
                     hide_index=True, width="stretch")

# --------------------------------------------------------------------------- props tab

def player_lookup():
    out = {}
    for team, prof, opp in ((away, away_prof, home), (home, home_prof, away)):
        for i, (pid, nm) in enumerate(zip(prof.ids, prof.names)):
            # Minutes when he plays (props are void on a DNP).
            proj = prof.proj_min[i] / (prof.p_play[i] if prof.p_play is not None else 1.0)
            out[nm] = dict(team=team, idx=i, pid=pid, opp=opp, proj=proj, sd=prof.min_sd[i])
    return out


lookup = player_lookup()

def attach_live(raw: pd.DataFrame, quota: dict, ev: dict, fetched: str):
    """Match a book's player names to tonight's sim and make these the game's live lines."""
    mapping = oddsapi.match_players(raw["Player"], list(lookup))
    raw["Book player"] = raw["Player"]
    raw["Player"] = raw["Player"].map(mapping)
    st.session_state[live_key] = {
        "lines": raw, "quota": quota, "event": ev,
        "unmatched": sorted(set(raw.loc[raw["Player"].isna(), "Book player"])), "fetched": fetched}
    st.session_state.pop(f"props_{away}_{home}_live", None)


with tab_props:
    live_key = f"live_{away}_{home}_{game_date}"
    with st.expander("📡 Live sportsbook lines (FanDuel, DraftKings, Fanatics, ...)",
                     expanded=live_key not in st.session_state):
        st.markdown(
            "Lines and odds come from [The Odds API](https://the-odds-api.com), a licensed odds "
            "aggregator. Sportsbooks don't offer public APIs, and scraping their sites breaks their "
            "terms. You need your own free API key (500 credits/month). Set it as the "
            "`ODDS_API_KEY` environment variable, or paste it here. It's kept only for this session.")
        k1, k2, k3 = st.columns([2, 2, 2])
        api_key_in = k1.text_input("The Odds API key", type="password", key="odds_key")
        books_sel = k2.multiselect("Books (up to 10 cost the same as 1)", list(oddsapi.BOOKS),
                                   format_func=oddsapi.BOOKS.get, max_selections=10, key="odds_books")
        mk_sel = st.session_state.odds_markets or []
        k3.markdown("**Prop types**  \n" + (", ".join(oddsapi.MARKET_LABELS.get(m, m) for m in mk_sel)
                                           or "None picked")
                    + "  \n:small[Pick them under **Player props to pull** at the top of the page.]")
        st.caption(f"Fetching this game costs up to {len(mk_sel)} credits (prop types not posted yet "
                   "aren't charged). Responses are cached for 10 minutes, so reruns are free. "
                   "Listing games is free. The key and books are also used by **Pull props for "
                   "all games** at the top of the page.")
        if st.button("Fetch live lines", type="primary", disabled=not (api_key_in and mk_sel and books_sel)):
            try:
                with st.spinner("Finding the game and fetching props..."):
                    events, _ = oddsapi.list_events(api_key_in)
                    ev = oddsapi.find_event(events, names[home], names[away], game_date)
                    if ev is None:
                        raise oddsapi.OddsError(
                            f"No upcoming {names[away]} @ {names[home]} game on The Odds API. Props "
                            "are usually posted the day before or the morning of the game.")
                    data, quota = oddsapi.fetch_props(api_key_in, ev["id"], mk_sel, books_sel)
                    attach_live(oddsapi.props_frame(data), quota, ev, dt.datetime.now().strftime("%H:%M"))
            except oddsapi.OddsError as exc:
                st.error(str(exc))
        elif live_key not in st.session_state and slate_key(away, home) in st.session_state.slate_props:
            # Lines already fetched with "Fetch odds for all games": no credits needed.
            sp = st.session_state.slate_props[slate_key(away, home)]
            attach_live(sp["raw"].copy(), {**sp["quota"], "cached": True}, sp["event"], sp["fetched"])
        live = st.session_state.get(live_key)
        if live:
            q = live["quota"]
            st.success(f"Loaded {len(live['lines'])} lines for {live['event']['away_team']} @ "
                       f"{live['event']['home_team']} at {live['fetched']}"
                       + (" (from cache, no credits used)" if q.get("cached") else
                          f" · cost {q.get('last')} credits")
                       + (f" · credits left: {lq['remaining']}" if (lq := oddsapi.latest_quota()) else ""))
            if live["unmatched"]:
                st.caption("Not in tonight's simulated rotation (or OUT), so skipped: "
                           + ", ".join(live["unmatched"]))

    live = st.session_state.get(live_key)
    source = st.radio("Lines to evaluate", ["Live sportsbook lines", "My own lines"],
                      index=0 if live else 1, horizontal=True, disabled=live is None,
                      help="Fetch live lines above to enable.")
    use_live = live is not None and source == "Live sportsbook lines"
    if use_live:
        st.markdown("Live lines from each book. Edit any line or price, or add your own rows with ➕.")
    else:
        st.markdown("Enter your book's lines below (pre-filled with each player's average rounded to "
                    "the nearest .5 — **replace with real lines**, or fetch live lines above).")
    key = f"props_{away}_{home}" + ("_live" if use_live else "")
    if use_live and key not in st.session_state:
        ll = live["lines"].dropna(subset=["Player"])
        st.session_state[key] = ll[["Player", "Stat", "Line", "Over odds", "Under odds", "Book"]] \
            .sort_values(["Player", "Stat", "Book"]).reset_index(drop=True)
    if key not in st.session_state:
        rows = []
        for team, prof in ((away, away_prof), (home, home_prof)):
            top = np.argsort(-prof.proj_min)[:6]
            for i in top:
                r = ds.players.loc[prof.ids[i]]
                for stat in ("PTS", "REB", "AST", "3PM"):
                    avg = sum(r[p] for p in PROP_STATS[stat]) / max(r["GP"], 1)
                    if stat in ("3PM",) and avg < 0.8:
                        continue
                    rows.append({"Player": prof.names[i], "Stat": stat, "Line": default_line(avg),
                                 "Over odds": -110, "Under odds": -110, "Book": "Manual"})
        st.session_state[key] = pd.DataFrame(rows)
    lines = st.data_editor(
        st.session_state[key], key=f"{key}_editor", num_rows="dynamic", hide_index=True, width="stretch",
        column_config={
            "Player": st.column_config.SelectboxColumn(options=sorted(lookup), required=True),
            "Stat": st.column_config.SelectboxColumn(options=list(PROP_STATS), required=True),
            "Line": st.column_config.NumberColumn(step=0.5, required=True),
            "Over odds": st.column_config.NumberColumn(step=5, default=-110),
            "Under odds": st.column_config.NumberColumn(step=5, default=-110),
            "Book": st.column_config.TextColumn(default="Manual"),
        })

    results = []
    for _, row in lines.dropna(subset=["Player", "Stat", "Line"]).iterrows():
        if row["Player"] not in lookup:
            continue
        info = lookup[row["Player"]]
        vals = sum(res.players[info["team"]][p][:, info["idx"]] for p in PROP_STATS[row["Stat"]])
        oo_, uo_ = row.get("Over odds"), row.get("Under odds")
        oo_ = -110 if pd.isna(oo_) else oo_
        uo_ = -110 if pd.isna(uo_) else uo_
        ev = evaluate_prop(ds, vals, info["pid"], row["Stat"], float(row["Line"]), oo_, uo_,
                           proj_min=info["proj"], min_sd=info["sd"], opp_abbr=info["opp"],
                           sim_min=res.players[info["team"]]["MIN"][:, info["idx"]])
        results.append({"Player": row["Player"], "Team": info["team"], "Stat": row["Stat"],
                        "Book": row.get("Book") or "Manual", "Over odds": oo_, "Under odds": uo_,
                        "Proj Min": info["proj"], **ev})
    if results:
        rdf = pd.DataFrame(results).sort_values("Value Score", ascending=False)
        multi_book = rdf["Book"].nunique() > 1
        if multi_book:
            # Line shopping: the best book for each player + stat + side.
            st.subheader("Best line for each prop")
            st.caption("For each prop, the book and side with the highest Value Score. Different "
                       "books post different lines and prices; this picks the best one available.")
            best = rdf.sort_values("Value Score", ascending=False).drop_duplicates(["Player", "Stat"])
            best = best.assign(**{"Odds": np.where(best["Pick"] == "OVER", best["Over odds"], best["Under odds"])})
            st.dataframe(best[[c for c in ["Player", "Team", "Stat", "Pick", "Line", "Odds", "Book",
                                           "Value Score", "Grade", "Sim mean", "Sim P(over)",
                                           "Book P(over)", "L10 hit", "Opp vs pos", "Opp rank"]
                               if c in best]]
                         .style.format({"Line": "{:.1f}", "Odds": "{:+.0f}", "Value Score": "{:.0f}",
                                        "Sim mean": "{:.1f}", "Sim P(over)": "{:.0%}",
                                        "Book P(over)": "{:.0%}", "L10 hit": "{:.0%}",
                                        "Opp rank": "{:.0f}"}, na_rep="—"),
                         hide_index=True, width="stretch", height=min(42 + 35 * len(best), 600))
            with st.expander("Line shopping grid (line and over / under by book)"):
                grid = rdf.assign(Quote=lambda d: d["Line"].map("{:.1f}".format) + "  "
                                  + d["Over odds"].map("{:+.0f}".format) + " / "
                                  + d["Under odds"].map("{:+.0f}".format))
                st.dataframe(grid.pivot_table(index=["Player", "Stat"], columns="Book", values="Quote",
                                              aggfunc="first").reset_index(),
                             hide_index=True, width="stretch")
            st.subheader("Every line, every book")
        show = ["Player", "Team", "Stat", "Book", "Line", "Over odds", "Under odds", "Pick",
                "Value Score", "Grade", "Sim mean",
                "Sim P(over)", "Book P(over)", "Edge over", "EV over", "EV under", "Season avg",
                "Min-adj avg", "L5 avg", "L10 avg", "L10 hit", "Season hit", "Proj Min",
                "Opp vs pos", "Opp allows", "Opp rank", "Opp vs avg", "Fair over odds"]
        show = [c for c in show if c in rdf]

        def color_score(v):
            if v >= 75:
                return "background-color: rgba(34,139,34,0.45)"
            if v >= 65:
                return "background-color: rgba(34,139,34,0.25)"
            if v >= 58:
                return "background-color: rgba(218,165,32,0.25)"
            return ""

        st.dataframe(
            rdf[show].style.format({
                "Line": "{:.1f}", "Over odds": "{:+.0f}", "Under odds": "{:+.0f}",
                "Value Score": "{:.0f}", "Sim mean": "{:.1f}", "Sim P(over)": "{:.0%}",
                "Book P(over)": "{:.0%}", "Edge over": "{:+.1%}", "EV over": "{:+.2f}", "EV under": "{:+.2f}",
                "Season avg": "{:.1f}", "Min-adj avg": "{:.1f}", "L5 avg": "{:.1f}", "L10 avg": "{:.1f}",
                "L10 hit": "{:.0%}", "Season hit": "{:.0%}", "Proj Min": "{:.1f}", "Opp rank": "{:.0f}",
                "Opp allows": "{:.1f}", "Opp vs avg": "{:+.0%}",
            }, na_rep="—").map(color_score, subset=["Value Score"]),
            hide_index=True, width="stretch", height=min(42 + 35 * len(rdf), 700))
        with st.expander("How the Value Score works"):
            st.markdown("""
- **50 = no edge.** The score is for the side in **Pick**. ≥75 = A, ≥65 = B, ≥58 = C, below that = Pass.
- **Sim edge (55%)**: simulated probability of the pick minus the book's no-vig implied probability.
- **Cushion (25%)**: (minutes-adjusted average − line) ÷ the player's game-to-game SD.
- **Hit rate (20%)**: how often the last 10 games cleared the line.
- Pulled toward 50 when the minutes projection is shaky (high minutes SD relative to projection).
- **EV** is profit per $1 at the listed odds using the sim's probabilities. **Opp rank** is defense vs position: what tonight's opponent gives up per game in that stat to the player's position (**Opp vs pos**: G, F or C; a G/F gets the average of both), ranked among the 30 teams. 1 = gives up the most (best matchup), 30 = the least. **Opp allows** is that per-game total for all players at the position, and **Opp vs avg** compares it with the league average. Built from game logs; last season's games count, worth at most 20 games.
- **Min-adj avg** = per-minute production × tonight's projected minutes.
""")
    st.caption("Model output, not betting advice. Lines move on injury news the model can't see — "
               "always update the Rotations tab first.")

# --------------------------------------------------------------------------- player deep dive

with tab_player:
    pick = st.selectbox("Player", sorted(lookup, key=lambda n: -lookup[n]["proj"]))
    info = lookup[pick]
    pid = info["pid"]
    eff = player_efficiency(ds, pid)
    r = ds.players.loc[pid]
    c = st.columns(7)
    c[0].metric("Proj minutes", f"{info['proj']:.1f}", help=f"SD {info['sd']:.1f}")
    c[1].metric("TS%", pct(eff["TS%"]))
    c[2].metric("eFG%", pct(eff["eFG%"]))
    c[3].metric("Usage (approx)", pct(eff["USG%"]))
    c[4].metric("3PA rate", pct(eff["3PAr"]))
    c[5].metric("FT rate", pct(eff["FTr"]))
    c[6].metric("PTS / 36", f"{eff['PTS/36']:.1f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Shot diet")
        diet = eff["diet"].copy()
        diet["Zone"] = diet["Zone"].map(ZONE_LABELS)
        long = pd.concat([
            diet.assign(Who=pick.split()[-1], Value=diet["Share"])[["Zone", "Who", "Value"]],
            diet.assign(Who="League", Value=diet["League share"])[["Zone", "Who", "Value"]]])
        st.altair_chart(alt.Chart(long).mark_bar().encode(
            x=alt.X("Zone:N", sort=list(ZONE_LABELS.values())), xOffset="Who:N",
            y=alt.Y("Value:Q", axis=alt.Axis(format="%"), title="Share of FGA"), color="Who:N",
            tooltip=["Zone", "Who", alt.Tooltip("Value:Q", format=".1%")]).properties(height=260),
            use_container_width=True)
    with right:
        st.subheader("Efficiency by zone vs tonight's defense")
        opp = info["opp"]
        mvd = res.matchups[info["team"]]
        rows = []
        for k, z in enumerate(ZONES):
            rows.append({"Zone": ZONE_LABELS[z], "FGA/G": diet["FGA/G"].iloc[k],
                         "Player FG%": diet["FG%"].iloc[k], "League FG%": diet["League FG%"].iloc[k],
                         f"{opp} allows": T.loc[opp, f"OPP_{z}_PCT"],
                         "Sim FG% tonight": mvd.make_p[info["idx"], k]})
        st.dataframe(pd.DataFrame(rows).style.format(
            {"FGA/G": "{:.1f}", "Player FG%": "{:.1%}", "League FG%": "{:.1%}", f"{opp} allows": "{:.1%}",
             "Sim FG% tonight": "{:.1%}"}, na_rep="—"), hide_index=True, width="stretch")
        st.caption(f"{opp} defense: DRtg {T.loc[opp, 'DEF_RATING']:.1f} "
                   f"(rank {int(T['DEF_RATING'].rank().loc[opp])}/30) · "
                   f"forces TOV on {pct(T.loc[opp, 'OPP_TOV_PCT'])} of plays · "
                   f"opp FT rate {T.loc[opp, 'OPP_FTA_RATE']:.3f}")

    team_prof = away_prof if info["team"] == away else home_prof
    od = team_prof.off_detail
    X = res.matchups[info["team"]].x
    k = info["idx"]
    trow = tr.players.reindex([pid]).fillna(0).iloc[0]
    per36 = lambda c: 36 * trow[c] / max(trow["TRK_MIN"], 1)

    st.subheader("Play types: what he runs, and what tonight's defense allows")
    opp_det = (home_prof if info["team"] == away else away_prof).team_detail
    plays = res.play_types[info["team"]][k]
    ptpts = res.pt_points[info["team"]][k]
    rows = []
    for ti, t in enumerate(PLAY_TYPES):
        poss = trow[f"PT_{t}_POSS"]
        rows.append({"Play type": PT_LABELS[t], "His mix": od["share"][k, ti],
                     "Mix vs this D": X["share"][k, ti], "Sim plays / game": plays[ti],
                     "His pts / poss": trow[f"PT_{t}_PTS"] / poss if poss >= 10 else np.nan,
                     "Sim pts / play": ptpts[ti] / plays[ti] if plays[ti] > 0.05 else np.nan,
                     f"{info['opp']} allows (freq vs lg)": opp_det["def_freq_rel"][ti],
                     f"{info['opp']} PPP allowed vs its avg": opp_det["def_ppp_rel"][ti]})
    ptdf = pd.DataFrame(rows).sort_values("His mix", ascending=False)
    st.dataframe(numeric(ptdf, skip=("Play type",)).style.format({"His mix": "{:.0%}", "Mix vs this D": "{:.0%}", "Sim plays / game": "{:.1f}",
                                    "His pts / poss": "{:.2f}", "Sim pts / play": "{:.2f}",
                                    f"{info['opp']} allows (freq vs lg)": "{:.2f}x",
                                    f"{info['opp']} PPP allowed vs its avg": "{:.2f}x"}, na_rep="—"),
                 hide_index=True, width="stretch")
    st.caption("Synergy play types (NBA.com). 'Mix vs this D' tilts his mix toward what this defense "
               "allows (scheme). In the sim, transition and putbacks aren't drawn from the mix: they "
               "happen when game events trigger them (steals, defensive rebounds, offensive rebounds).")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Playmaking (per 36, tracking)**")
        st.dataframe(pd.DataFrame({"Stat": ["Assists", "Potential assists", "Secondary assists",
                                            "Assist points created", "Passes", "Drives", "Drive passes"],
                                   "Per 36": [per36("TRK_AST"), per36("POT_AST"), per36("SEC_AST"),
                                              per36("AST_PTS"), per36("PASSES"), per36("DRIVES"),
                                              per36("DRIVE_PASSES")]}).style.format({"Per 36": "{:.1f}"}),
                     hide_index=True, width="stretch")
        pa = res.passes[info["team"]][k]
        if pa.sum() > 0:
            st.caption("Sim: his assists by the play type of the shot: " + ", ".join(
                f"{PT_LABELS[PLAY_TYPES[i]]} {pa[i]:.1f}" for i in np.argsort(-pa)[:4] if pa[i] > 0.05))
    with c2:
        st.markdown("**Rebounding (tracking)**")
        o_ch, d_ch = per36("TRK_OREB_CHANCES"), per36("TRK_DREB_CHANCES")
        st.dataframe(pd.DataFrame({"Stat": ["Off. reb chances / 36", "Def. reb chances / 36",
                                            "Avg OREB distance (ft)", "Avg DREB distance (ft)",
                                            "Contested DREB share", "Box-outs / 36"],
                                   "Value": [o_ch, d_ch, od["oreb_dist"][k], od["dreb_dist"][k],
                                             trow["TRK_DREB_CONTEST"] / max(trow["TRK_DREB"], 1),
                                             36 * trow["BOX_OUTS"] / max(trow["HUSTLE_MIN"], 1)]})
                     .style.format({"Value": "{:.2f}"}), hide_index=True, width="stretch")
        st.caption(f"League avg rebound distance: OREB {tr.league['OREB_DIST']:.1f} ft, DREB "
                   f"{tr.league['DREB_DIST']:.1f} ft. Farther = more long rebounds off missed 3s.")
    with c3:
        st.markdown("**Shot quality**")
        st.dataframe(pd.DataFrame({"Stat": ["Catch-and-shoot 3P% (model)", "Pull-up 3P% (model)",
                                            "Overall 3P% (model)", "Tight-contest sensitivity"],
                                   "Value": [od["cs3"][k], od["pu3"][k], od["p3"][k], od["contest_sens"][k]]})
                     .style.format({"Value": "{:.3f}"}), hide_index=True, width="stretch")
        st.caption("Sensitivity 1.0 = loses as much to tight contests as the league average; "
                   "higher = hurt more by good contesting defenses.")

    g = res.guarded[info["team"]][k]
    opp_prof = home_prof if info["team"] == away else away_prof
    if g.sum() > 0:
        gd = pd.DataFrame({"Defender": opp_prof.names, "Share of his plays": g / g.sum(),
                           "FG% allowed vs expected": opp_prof.def_detail["d_pm"],
                           "Rim FG% vs expected": opp_prof.def_detail["rim_pm"],
                           "Shooting fouls vs lg": opp_prof.def_detail["foul_rel"]}).sort_values(
            "Share of his plays", ascending=False).head(5)
        st.markdown("**Who guards him tonight (simulated primary defender)**")
        st.dataframe(gd.style.format({"Share of his plays": "{:.0%}", "FG% allowed vs expected": "{:+.1%}",
                                      "Rim FG% vs expected": "{:+.1%}", "Shooting fouls vs lg": "{:.2f}x"}),
                     hide_index=True, width="stretch")
        st.caption("From NBA.com matchup data (who guarded whom), limited to defenders on the floor. "
                   "FG% vs expected is shrunk toward 0 by sample size; negative = tougher defender.")

    st.subheader("Lineup context: who he shares the floor with")
    team_prof = away_prof if info["team"] == away else home_prof
    l1, l2 = st.columns(2)
    with l1:
        oo_df = on_off_usage(res.on[info["team"]], res.play_w[info["team"]], info["idx"], team_prof.names)
        st.caption(f"Simulated tonight: {pick.split()[-1]}'s share of team plays while on the floor, "
                   f"with vs without each teammate (overall {pct(oo_df.attrs['overall'])})")
        st.dataframe(numeric(oo_df).style.format({"Min together / game": "{:.1f}", "Usage with": "{:.1%}",
                                         "Usage without": "{:.1%}", "Change": "{:+.1%}"}, na_rep="—"),
                     hide_index=True, width="stretch")
    with l2:
        mates = [(p, nm) for p, nm in zip(ds.players.index, ds.players["PLAYER_NAME"])
                 if ds.players.loc[p, "TEAM_ABBREVIATION"] == info["team"]]
        wt = teammate_table(ds, info["team"], pid, mates)
        st.caption("Real games: his per-36 production in games each teammate played vs missed")
        if wt.empty:
            st.info("No games yet where a current teammate was out while he played.")
        else:
            show = ["Teammate out", "GP with", "GP without", "Usage change", "MPG with", "MPG without",
                    "PTS/36 with", "PTS/36 without", "REB/36 with", "REB/36 without",
                    "AST/36 with", "AST/36 without"]
            st.dataframe(numeric(wt[show]).style.format({c: "{:.1f}" for c in show[4:]} | {"Usage change": "{:+.0%}"},
                                               na_rep="—"), hide_index=True, width="stretch")

    st.subheader("Prop check")
    a, b = st.columns([1, 3])
    stat = a.selectbox("Stat", list(PROP_STATS), key="dd_stat")
    hist_vals = log_values(ds, pid, PROP_STATS[stat])
    savg = sum(r[p] for p in PROP_STATS[stat]) / max(r["GP"], 1)
    line = a.number_input("Line", value=default_line(savg), step=0.5, key=f"dd_line_{pid}_{stat}")
    oo = a.number_input("Over odds", value=-110, step=5, key="dd_oo")
    uo = a.number_input("Under odds", value=-110, step=5, key="dd_uo")
    vals = sum(res.players[info["team"]][p][:, info["idx"]] for p in PROP_STATS[stat])
    ev = evaluate_prop(ds, vals, pid, stat, line, oo, uo, info["proj"], info["sd"], info["opp"],
                       sim_min=res.players[info["team"]]["MIN"][:, info["idx"]])
    with a:
        st.metric("Value Score", f"{ev['Value Score']:.0f} ({ev['Grade']})", ev["Pick"])
        st.write(f"Sim P(over): **{pct(ev['Sim P(over)'])}** · book (no-vig) {pct(ev['Book P(over)'])}")
        st.write(f"Season {ev['Season avg']:.1f} · L10 {ev['L10 avg']:.1f} · min-adj {ev['Min-adj avg']:.1f}")
        st.write(f"L10 hit rate: {pct(ev['L10 hit'], 0)}")
        if "Opp rank" in ev:
            who = "all players" if ev["Opp vs pos"] == "team" else f"{ev['Opp vs pos']} players"
            st.write(f"vs {who}: {info['opp']} gives up **{ev['Opp allows']:.1f} {stat}** a game, "
                     f"rank **{ev['Opp rank']}/30** (1 = most) · {ev['Opp vs avg']:+.0%} vs league average")
    with b:
        g = ds.logs[(ds.logs["PLAYER_ID"] == pid) & (ds.logs["MIN"] > 0)].head(20).copy()
        g["Value"] = sum(g[p] for p in PROP_STATS[stat])
        g["Game"] = g["GAME_DATE"].dt.strftime("%m/%d ") + g["MATCHUP"].str[-3:]
        g["Result"] = np.where(g["Value"] > line, "Over", "Under")
        g = g.iloc[::-1]
        bars = alt.Chart(g).mark_bar().encode(
            x=alt.X("Game:N", sort=list(g["Game"]), title="Last 20 games"), y=alt.Y("Value:Q", title=stat),
            color=alt.Color("Result:N", scale=alt.Scale(domain=["Over", "Under"], range=["#2ca02c", "#d62728"])),
            tooltip=["Game", "Value", alt.Tooltip("MIN:Q", format=".1f")])
        rule = alt.Chart(pd.DataFrame({"y": [line]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
        st.altair_chart((bars + rule).properties(height=240), use_container_width=True)
        dist = pd.DataFrame({"v": vals})
        st.altair_chart((alt.Chart(dist).mark_bar(binSpacing=0, opacity=0.85).encode(
            x=alt.X("v:Q", bin=alt.Bin(step=1 if vals.max() < 40 else 2), title=f"Simulated {stat} tonight"),
            y=alt.Y("count():Q", title="Sims"))
            + alt.Chart(pd.DataFrame({"x": [line]})).mark_rule(color="red", strokeWidth=2).encode(x="x:Q")
        ).properties(height=200), use_container_width=True)

    st.subheader("Minutes trend")
    g = ds.logs[(ds.logs["PLAYER_ID"] == pid) & (ds.logs["MIN"] > 0)].head(20).iloc[::-1].copy()
    g["Game"] = g["GAME_DATE"].dt.strftime("%m/%d ") + g["MATCHUP"].str[-3:]
    sim_min = res.players[info["team"]]["MIN"][:, info["idx"]]
    st.altair_chart((alt.Chart(g).mark_line(point=True).encode(
        x=alt.X("Game:N", sort=list(g["Game"])), y=alt.Y("MIN:Q", title="Minutes"))
        + alt.Chart(pd.DataFrame({"y": [info["proj"]]})).mark_rule(color="orange", strokeDash=[4, 4]).encode(y="y:Q")
    ).properties(height=200), use_container_width=True)
    st.caption(f"Orange = projected {info['proj']:.1f} min. Simulated minutes: mean {sim_min.mean():.1f}, "
               f"10th–90th pct {np.percentile(sim_min, 10):.0f}–{np.percentile(sim_min, 90):.0f} "
               f"(includes blowout rest and OT).")

# --------------------------------------------------------------------------- matchup tab

with tab_match:
    st.markdown(f"**Expected pace:** {expected_pace(home_prof, away_prof, ds.league):.1f} possessions "
                f"(league avg {ds.league['PACE']:.1f})")
    metrics = [  # column, label, higher_is_better, format
        ("PACE", "Pace", True, "{:.1f}"), ("OFF_RATING", "Off rating", True, "{:.1f}"),
        ("DEF_RATING", "Def rating", False, "{:.1f}"), ("NET_RATING", "Net rating", True, "{:+.1f}"),
        ("EFG_PCT", "eFG%", True, "{:.1%}"), ("TM_TOV_PCT", "TOV%", False, "{:.1%}"),
        ("OREB_PCT", "OREB%", True, "{:.1%}"), ("FTA_RATE", "FT rate", True, "{:.3f}"),
        ("OPP_EFG_PCT", "Opp eFG%", False, "{:.1%}"), ("OPP_TOV_PCT", "Opp TOV% (forced)", True, "{:.1%}"),
        ("OPP_OREB_PCT", "Opp OREB%", False, "{:.1%}"), ("OPP_FTA_RATE", "Opp FT rate", False, "{:.3f}"),
        ("OPP_PTS_PG", "Opp PTS/G", False, "{:.1f}"), ("OPP_REB_PG", "Opp REB/G", False, "{:.1f}"),
        ("OPP_AST_PG", "Opp AST/G", False, "{:.1f}"), ("OPP_FG3M_PG", "Opp 3PM/G", False, "{:.1f}"),
    ]
    rows = []
    for col, label, hib, f in metrics:
        rk = T[col].rank(ascending=not hib, method="min")
        rows.append({"Metric": label,
                     away: f"{f.format(T.loc[away, col])} (#{int(rk[away])})",
                     home: f"{f.format(T.loc[home, col])} (#{int(rk[home])})",
                     "League": f.format(T[col].mean())})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=600)
    st.caption("Ranks: #1 = best at that metric (for defensive 'Opp' rows, #1 = allows the least).")

    st.subheader("Zone defense")
    zrows = []
    for z in ZONES:
        zrows.append({"Zone": ZONE_LABELS[z], "League FG%": ds.league[f"{z}_PCT"],
                      f"{away} allows": T.loc[away, f"OPP_{z}_PCT"], f"{home} allows": T.loc[home, f"OPP_{z}_PCT"],
                      f"{away} freq allowed": T.loc[away, f"OPP_{z}_FREQ"],
                      f"{home} freq allowed": T.loc[home, f"OPP_{z}_FREQ"]})
    st.dataframe(pd.DataFrame(zrows).style.format({c: "{:.1%}" for c in zrows[0] if c != "Zone"}),
                 hide_index=True, width="stretch")

with tab_match:
    st.subheader("Scheme: play types each defense allows")
    st.caption("Coverages (drop / switch / blitz / zone) aren't published, so the scheme is read from "
               "what each defense allows by Synergy play type. Frequency >1 = allows more of that "
               "play type than average; PPP <1 = defends it better than its own average.")
    rows = []
    for ti, t in enumerate(PLAY_TYPES):
        r = {"Play type": PT_LABELS[t]}
        for prof in (away_prof, home_prof):
            r[f"{prof.abbr} freq allowed"] = prof.team_detail["def_freq_rel"][ti]
            r[f"{prof.abbr} PPP allowed"] = prof.team_detail["def_ppp_rel"][ti]
        rows.append(r)
    st.dataframe(pd.DataFrame(rows).style.format({c: "{:.2f}x" for c in rows[0] if c != "Play type"}),
                 hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    for col, prof in ((c1, away_prof), (c2, home_prof)):
        td = prof.team_detail
        col.markdown(f"**{prof.abbr}**: transition push {td['push']:.2f}x league · transition allowed "
                     f"{td['trans_allowed']:.2f}x · contest quality allowed {td['contest_q']:.3f} "
                     f"(<1 = contests tighter than average)")
        mix = td.get("contest_mix")
        if mix is not None:
            col.caption("Opponent shots by closest defender: " + ", ".join(
                f"{lbl} {v:.0%}" for lbl, v in zip(["very tight", "tight", "open", "wide open"], mix)))

# --------------------------------------------------------------------------- watch a game

with tab_watch:
    st.markdown("Simulate a single game possession by possession.")
    if st.button("▶ Simulate a game", type="primary"):
        st.session_state.pbp = simulate(home_prof, away_prof, ds.league, n=1,
                                        seed=int(np.random.default_rng().integers(1e9)),
                                        pace_override=pace_override or None, weights=weights,
                                        tl=tr.league, sched_home=sched[home]["factor"],
                                        sched_away=sched[away]["factor"], play_by_play=True)
    g = st.session_state.get("pbp")
    if g is not None and g.home.abbr == home and g.away.abbr == away:
        h, a = int(g.score[home][0]), int(g.score[away][0])
        ot = int(g.ot[0])
        st.header(f"Final{' (' + str(ot) + 'OT)' if ot else ''}: {away} {a} – {h} {home}")
        qs = pd.DataFrame({t: g.quarters[t][0] for t in (away, home)},
                          index=["Q1", "Q2", "Q3", "Q4", "OT"]).T.astype(int)
        if not ot:
            qs = qs.drop(columns="OT")
        qs["Final"] = [a, h]
        st.dataframe(qs, width="stretch")
        st.caption(f"{int(g.poss[0])} possessions per team")
        for team, prof in ((away, away_prof), (home, home_prof)):
            s = g.players[team]
            df = pd.DataFrame({"Player": prof.names, "MIN": s["MIN"][0].round(0),
                               "PTS": s["PTS"][0], "REB": s["REB"][0], "AST": s["AST"][0],
                               "FG": [f"{int(m)}-{int(x)}" for m, x in zip(s["FGM"][0], s["FGA"][0])],
                               "3P": [f"{int(m)}-{int(x)}" for m, x in zip(s["FG3M"][0], s["FG3A"][0])],
                               "FT": [f"{int(m)}-{int(x)}" for m, x in zip(s["FTM"][0], s["FTA"][0])],
                               "STL": s["STL"][0], "BLK": s["BLK"][0], "TOV": s["TOV"][0]})
            st.caption(names[team])
            st.dataframe(df.sort_values("MIN", ascending=False), hide_index=True, width="stretch")
        pbp = pd.DataFrame(g.log)
        qf = st.multiselect("Quarters", list(pbp["Q"].unique()), default=list(pbp["Q"].unique()))
        st.dataframe(pbp[pbp["Q"].isin(qf)].rename(columns={"Score": f"Score ({away}-{home})"}),
                     hide_index=True, width="stretch", height=500)

# --------------------------------------------------------------------------- model & data

with tab_model:
    st.subheader("How usage is decided for each lineup")
    st.markdown("""
1. **Base usage per player**: plays used per minute (FGA + free-throw trips + turnovers) from the
   blended box score. Calibrated so that, across a normal rotation, sharing the floor with
   the usual teammates reproduces his real usage.
2. **Every possession, only the five on the floor compete** for the play. A player's share =
   his weight ÷ the sum of the five weights. With higher-usage teammates on the floor his share
   drops; when they sit it rises.
3. **Adjustments on top**: per-game hot/cold usage (about ±15%), clutch usage in the last 5
   minutes of close games, transition frequency on fast breaks, putbacks after his own offensive
   rebounds, and a with/without correction from real games when a teammate is marked OUT.
4. **Efficiency follows the load**: above his normal on-floor share he shoots slightly worse.
   Better passers around him than usual make his looks easier.
""")
    st.subheader("Every factor: data source, weight, and why")
    reg = pd.DataFrame([{"Factor": label, "Group": grp, "Weight now": getattr(weights, key),
                         "Default": getattr(Weights(), key), "Data source": src, "How it's used": how,
                         "Shrinkage": shrink, "Why this default": why}
                        for key, grp, label, mx, src, how, shrink, why in REGISTRY])
    st.dataframe(reg, hide_index=True, width="stretch", height=740,
                 column_config={"How it's used": st.column_config.TextColumn(width="large"),
                                "Why this default": st.column_config.TextColumn(width="large")})
    st.markdown("""
**How the weights were chosen**
- **Full strength (1.0):** the data measures the same thing the sim does, with big samples:
  play-type mix, catch-and-shoot vs pull-up splits, rebound distance, transition and putback rates.
- **Partial strength (0.3–0.6):** noisy data, or data that overlaps something already counted.
  Individual defender FG%, defender fouls, scheme splits and deflections are halved or less,
  because team-level defense already includes most of their effect.
- **Priors:** effects you can't measure per player from public data (rest, travel, altitude,
  usage-efficiency) use sizes from published research and are kept small.
- **Sample-size shrinkage everywhere:** each player or team rate is blended with the league
  average in proportion to how little data it has (see the Shrinkage column). A player with 15
  isolation possessions mostly gets league-average isolation numbers.
- **Calibration checks:** with all factors on, simulated team offense against a neutral defense lands
  within about 1.5 points per 100 of real ratings, and top players' points, assists, rebounds,
  threes, turnovers and free throws land within about 1–5% of their real per-minute rates.
""")
    st.subheader("Not available from public data")
    for name, why in NOT_AVAILABLE:
        st.markdown(f"- **{name}**: {why}")
    st.subheader("Tonight's schedule context")
    st.dataframe(pd.DataFrame([{
        "Team": t, "Last game": sched[t]["ctx"]["last"], "Rest days": sched[t]["ctx"]["rest_days"],
        "Back-to-back": sched[t]["ctx"]["b2b"], "3 in 4 nights": bool(sched[t]["ctx"]["three_in_four"]),
        "Travel since last game (mi)": round(sched[t]["ctx"]["travel_mi"]),
        "Make-prob multiplier": sched[t]["factor"][0], "Adjustments": ", ".join(sched[t]["why"]) or "none"}
        for t in (away, home)]).style.format({"Make-prob multiplier": "{:.3f}"}),
        hide_index=True, width="stretch")
    st.caption(f"Tracking data: {ds.season} blended with {ds.prior_season}. Synergy play types fill in "
               "a few weeks into each season, so early on they lean on last season.")

# Refresh the credit count now that every fetch on this run has happened.
show_credits()
