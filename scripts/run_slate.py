"""Simulate a whole night's slate without the app and rank the best player props.

Steps: load stats and tracking data -> the day's games -> ESPN injury report (applied the same
way as the app: Out players OUT, questionable/doubtful lower Play %) -> sportsbook lines from
The Odds API -> simulate every game -> score every line from every book -> keep the best book
and side for each player + stat -> write a CSV and an HTML report.

Odds API credits: game lines cost 3 for every game together; props cost 1 per prop type per
game. Responses are cached for 10 minutes, so re-running right away is free.

Usage:
  python scripts/run_slate.py --dry-run              # slate, injuries, credit estimate; spends nothing
  python scripts/run_slate.py                        # PTS, REB, AST, 3PM for every game
  python scripts/run_slate.py --markets PTS,PRA --games "LAL@GSW,DEN@PHX" --sims 5000
  python scripts/run_slate.py --cached-only          # only lines already saved; spends nothing
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from nbasim import data as nbadata  # noqa: E402
from nbasim import injuries  # noqa: E402
from nbasim import odds as oddsapi  # noqa: E402
from nbasim import schedule  # noqa: E402
from nbasim.detail import schedule_context, schedule_factor  # noqa: E402
from nbasim.factors import Weights  # noqa: E402
from nbasim.model import minutes_table, set_tracking, team_profile  # noqa: E402
from nbasim.props import PROP_STATS, evaluate_prop  # noqa: E402
from nbasim.sim import simulate  # noqa: E402
from nbasim.tracking import load_tracking  # noqa: E402

REPORT_DIR = os.path.join(ROOT, "reports")
KEY_TEAMMATE_MPG = 25.0   # a teammate this big being out shifts usage a lot -> flag the prop
MINUTES_SHIFT = 4.0       # projected minutes this far from his season average -> flag


def last_name(name: str) -> str:
    """'Jimmy Butler III' -> 'Butler'."""
    parts = [x for x in name.split() if x.rstrip(".").lower() not in ("jr", "sr", "ii", "iii", "iv", "v")]
    return parts[-1] if parts else name


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def today_et() -> dt.date:
    return pd.Timestamp.now(tz="US/Eastern").date()


# --------------------------------------------------------------------------- odds

def fetch_lines(key, slate, names, game_date, markets, books, cached_only):
    """Game-line summaries and prop frames by game, plus credits used."""
    used, game_lines, props, notes = 0, {}, {}, []
    if not key:
        notes.append("No Odds API key, so no sportsbook lines.")
        return game_lines, props, used, notes
    try:
        if cached_only:
            raise oddsapi.OddsError("cached-only")
        gl, q = oddsapi.fetch_game_lines(key, books)
        used += 0 if q.get("cached") else int(q.get("last") or 0)
    except oddsapi.OddsError as exc:
        path = os.path.join(nbadata.CACHE_DIR, "odds", f"game_lines_{'_'.join(sorted(books))}.json")
        gl = json.load(open(path))["data"] if os.path.exists(path) else []
        if str(exc) != "cached-only":
            notes.append(f"Game lines: {exc}")
    events = gl
    if not cached_only:
        try:
            events, _ = oddsapi.list_events(key)   # free
        except oddsapi.OddsError as exc:
            notes.append(f"Event list: {exc}")
    for g in slate:
        a, h = g["away"], g["home"]
        ev = oddsapi.find_event(gl, names[h], names[a], game_date)
        if ev is not None:
            game_lines[(a, h)] = oddsapi.game_line_summary(ev)
        ev = oddsapi.find_event(events, names[h], names[a], game_date)
        if ev is None:
            notes.append(f"{a} @ {h}: not on The Odds API (started, or not posted yet)")
            continue
        try:
            data, q = oddsapi.fetch_props(key, ev["id"], markets, books,
                                          max_age_min=1e9 if cached_only else 10.0,
                                          cache_only=cached_only)
        except oddsapi.OddsError as exc:
            notes.append(f"{a} @ {h}: {exc}")
            if "429" in str(exc):
                break
            continue
        used += 0 if q.get("cached") else int(q.get("last") or 0)
        frame = oddsapi.props_frame(data)
        if frame.empty:
            notes.append(f"{a} @ {h}: no props posted yet")
        else:
            props[(a, h)] = frame
    return game_lines, props, used, notes


# --------------------------------------------------------------------------- sim one game

def run_game(ds, tr, weights, g, names, game_date, inj_report, lines, n_sims):
    a, h = g["away"], g["home"]
    mins, inj_rows = {}, {}
    for t in (a, h):
        rep = injuries.for_team(inj_report, names[t], game_date) if inj_report is not None else None
        m = minutes_table(ds, t)
        mins[t] = injuries.apply(m, rep) if rep is not None else m.assign(Injury="")
        inj_rows[t] = rep
    profs = {t: team_profile(ds, t, mins[t], tr=tr, w_pot=weights.potential_ast,
                             w_chances=weights.rebound_chances) for t in (a, h)}
    sched = {}
    for t in (a, h):
        ctx = schedule_context(ds, t, game_date, h)
        mk, tv, _ = schedule_factor(ctx, t == h, weights.rest_travel)
        sched[t] = (mk, tv)
    res = simulate(profs[h], profs[a], ds.league, n=n_sims, seed=7, weights=weights, tl=tr.league,
                   sched_home=sched[h], sched_away=sched[a])

    # Player lookup, as in the app: minutes when he plays (props are void on a DNP).
    lookup = {}
    for t, opp in ((a, h), (h, a)):
        p = profs[t]
        m = mins[t].set_index("Player")
        for i, (pid, nm) in enumerate(zip(p.ids, p.names)):
            proj = p.proj_min[i] / (p.p_play[i] if p.p_play is not None else 1.0)
            inj = str(m["Injury"].get(nm, "") or "") if nm in m.index else ""
            lookup[nm] = dict(team=t, opp=opp, idx=i, pid=pid, proj=proj, sd=p.min_sd[i],
                              season_mpg=float(m["Season MPG"].get(nm, np.nan)) if nm in m.index else np.nan,
                              injury=inj.split(":")[0] if inj else "")
    # Teammates out tonight who normally play big minutes.
    big_out = {}
    for t in (a, h):
        m = mins[t]
        big_out[t] = [last_name(n) for n in m.loc[m["OUT"] & (m["Season MPG"] >= KEY_TEAMMATE_MPG), "Player"]]

    hs, as_ = res.score[h], res.score[a]
    game = {"Game": f"{a} @ {h}", "away": a, "home": h, "time": schedule.status_line(g),
            "Away pts": float(as_.mean()), "Home pts": float(hs.mean()),
            "Sim spread": float(-(hs - as_).mean()), "Sim total": float((hs + as_).mean()),
            "Home win %": float((hs > as_).mean()),
            "Out": {t: list(mins[t].loc[mins[t]["OUT"], "Player"]) for t in (a, h)},
            "Questionable": {t: [f"{r['Player']} ({injuries.SHORT.get(r['Status'], r['Status'])})"
                                 for _, r in (inj_rows[t] if inj_rows[t] is not None else pd.DataFrame(
                                     columns=["Player", "Status"])).iterrows()
                                 if r["Status"] in ("Doubtful", "Questionable", "Day-To-Day")]
                             for t in (a, h)}}

    rows, unmatched = [], []
    if lines is not None and len(lines):
        mapping = oddsapi.match_players(lines["Player"], list(lookup))
        for _, r in lines.iterrows():
            nm = mapping.get(r["Player"])
            if nm is None:
                unmatched.append(r["Player"])
                continue
            info = lookup[nm]
            s = res.players[info["team"]]
            vals = sum(s[p][:, info["idx"]] for p in PROP_STATS[r["Stat"]])
            ev = evaluate_prop(ds, vals, info["pid"], r["Stat"], float(r["Line"]),
                               r["Over odds"], r["Under odds"], proj_min=info["proj"],
                               min_sd=info["sd"], opp_abbr=info["opp"], sim_min=s["MIN"][:, info["idx"]])
            flags = []
            if info["injury"] in ("Doubtful", "Questionable", "Day-To-Day", "Probable"):
                flags.append(f"he is {info['injury'].lower()}")
            if big_out[info["team"]]:
                flags.append("teammate out: " + ", ".join(big_out[info["team"]]))
            if not np.isnan(info["season_mpg"]) and abs(info["proj"] - info["season_mpg"]) >= MINUTES_SHIFT:
                flags.append(f"minutes {info['season_mpg']:.0f} -> {info['proj']:.0f}")
            if r.get("One-sided", False):
                flags.append("one-sided line")
            odds = r["Over odds"] if ev["Pick"] == "OVER" else r["Under odds"]
            rows.append({"Game": game["Game"], "Player": nm, "Team": info["team"], "Stat": r["Stat"],
                         "Pick": ev["Pick"], "Line": float(r["Line"]), "Odds": float(odds),
                         "Book": r["Book"], "Value Score": ev["Value Score"], "Grade": ev["Grade"],
                         "Sim mean": ev["Sim mean"], "Sim P(over)": ev["Sim P(over)"],
                         "Book P(over)": ev["Book P(over)"],
                         "EV": ev["EV over"] if ev["Pick"] == "OVER" else ev["EV under"],
                         "L10 hit": ev["L10 hit"], "Proj Min": info["proj"],
                         "Opp rank": ev.get("Opp rank"), "Opp vs pos": ev.get("Opp vs pos"),
                         "Flags": "; ".join(flags)})
    game["unmatched"] = sorted(set(unmatched))
    return game, rows


# --------------------------------------------------------------------------- report

CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--line:#e5e5ea;--card:#f6f6f8;--a:#0a7d32;--b:#2a6fdb;--warn:#b25000}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#f2f2f4;--mut:#9a9aa1;--line:#2c2c30;--card:#1d1d21;--a:#3ccf6e;--b:#6ea2ff;--warn:#ffa24d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px 48px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 8px}
.mut{color:var(--mut);font-size:13px}.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{background:var(--card);font-weight:600;position:sticky;top:0}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:0}.A{color:var(--a);font-weight:700}.B{color:var(--b);font-weight:700}.flag{color:var(--warn);white-space:normal;min-width:180px}
.games{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:10px}.g{background:var(--card);border-radius:10px;padding:10px 12px}
.g b{font-size:15px}.note{background:var(--card);border-radius:10px;padding:10px 12px;margin-top:10px}
"""


def fmt_odds(o):
    return f"{int(round(o)):+d}"


def picks_table(df: pd.DataFrame) -> str:
    cols = ["Player", "Game", "Stat", "Pick", "Line", "Odds", "Book", "Grade", "Value Score",
            "Sim mean", "Sim P", "Book P", "L10 hit", "Proj Min", "Flags"]
    out = ["<div class='wrap'><table><tr>" + "".join(
        f"<th class='{'num' if c in ('Line', 'Odds', 'Value Score', 'Sim mean', 'Sim P', 'Book P', 'L10 hit', 'Proj Min') else ''}'>{c}</th>"
        for c in cols) + "</tr>"]
    for _, r in df.iterrows():
        side_p = r["Sim P(over)"] if r["Pick"] == "OVER" else 1 - r["Sim P(over)"]
        book_p = r["Book P(over)"] if r["Pick"] == "OVER" else 1 - r["Book P(over)"]
        cells = [html.escape(r["Player"]), html.escape(r["Game"]), r["Stat"], r["Pick"].title(),
                 f"{r['Line']:.1f}", fmt_odds(r["Odds"]), html.escape(str(r["Book"])),
                 f"<span class='{r['Grade']}'>{r['Grade']}</span>", f"{r['Value Score']:.0f}",
                 f"{r['Sim mean']:.1f}", f"{side_p:.0%}", f"{book_p:.0%}",
                 "—" if pd.isna(r["L10 hit"]) else f"{r['L10 hit']:.0%}", f"{r['Proj Min']:.0f}",
                 html.escape(r["Flags"]) or ""]
        out.append("<tr>" + "".join(
            f"<td class='{'num' if i in (4, 5, 8, 9, 10, 11, 12, 13) else 'flag' if i == 14 else ''}'>{c}</td>"
            for i, c in enumerate(cells)) + "</tr>")
    out.append("</table></div>")
    return "\n".join(out)


def write_report(path, game_date, games, best, game_lines, meta):
    a_b = best[best["Grade"].isin(["A", "B"])]
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>NBA Slate {game_date:%b %-d}</title><style>{CSS}</style></head><body><main>",
             f"<h1>NBA slate: {game_date:%A, %B %-d}</h1>",
             f"<p class='mut'>{len(games)} games · {meta['sims']:,} sims each · generated {meta['generated']} · "
             f"prop types: {', '.join(meta['markets'])} · {meta['credits_used']} Odds API credits used"
             + (f", {meta['credits_left']} left" if meta.get('credits_left') is not None else "") + "</p>"]
    if meta["warnings"]:
        parts.append("<div class='note'><b>Check before betting</b><ul>" + "".join(
            f"<li>{html.escape(w)}</li>" for w in meta["warnings"]) + "</ul></div>")
    parts.append("<h2>Games</h2><div class='games'>")
    for g in games:
        gl = game_lines.get((g["away"], g["home"]), {})
        mkt = []
        if gl.get("home_spread") is not None:
            mkt.append(f"market {g['home']} {gl['home_spread']:+g}")
        if gl.get("total") is not None:
            mkt.append(f"O/U {gl['total']:g}")
        inj = []
        for t in (g["away"], g["home"]):
            o = [last_name(n) for n in g["Out"][t]]
            q = g["Questionable"][t]
            if o or q:
                inj.append(f"{t}: " + ", ".join([f"{n} (O)" for n in o] + q))
        parts.append(
            f"<div class='g'><b>{g['away']} {g['Away pts']:.0f} @ {g['home']} {g['Home pts']:.0f}</b> "
            f"<span class='mut'>{html.escape(g['time'])}</span><br>"
            f"<span class='mut'>Sim {g['home']} {g['Sim spread']:+.1f} · total {g['Sim total']:.1f} · "
            f"{g['home']} wins {g['Home win %']:.0%}</span><br>"
            f"<span class='mut'>{' · '.join(mkt) or 'no market lines'}</span>"
            + (f"<br><span class='mut'>🚑 {html.escape(' · '.join(inj))}</span>" if inj else "")
            + "</div>")
    parts.append("</div>")
    parts.append(f"<h2>Top props ({len(a_b)} graded A or B)</h2>"
                 "<p class='mut'>Best book and side for each player + stat, ranked by Value Score. "
                 "Flags mark props that depend on an injury call or a big minutes change: check the "
                 "news before betting those.</p>")
    parts.append(picks_table(a_b.head(meta["top"]) if len(a_b) else best.head(meta["top"])))
    for g in games:
        sub = best[best["Game"] == f"{g['away']} @ {g['home']}"]
        if len(sub):
            parts.append(f"<h2>{g['away']} @ {g['home']}: every prop</h2>")
            parts.append(picks_table(sub))
    parts.append("<p class='mut'>Model output, not betting advice. Sim probabilities are shrunk toward "
                 "50/50 because raw sim odds backtested as overconfident; still, many large edges "
                 "usually means the model is missing news, not that the books are wrong.</p>"
                 "</main></body></html>")
    with open(path, "w") as fh:
        fh.write("\n".join(parts))


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="Game date, YYYY-MM-DD (default: today, US Eastern)")
    ap.add_argument("--markets", default=",".join(oddsapi.DEFAULT_MARKETS),
                    help=f"Prop types, comma-separated. Options: {', '.join(oddsapi.MARKETS)}")
    ap.add_argument("--books", default=",".join(oddsapi.DEFAULT_BOOKS))
    ap.add_argument("--games", help='Only these games, e.g. "LAL@GSW,DEN@PHX"')
    ap.add_argument("--sims", type=int, default=3000)
    ap.add_argument("--top", type=int, default=25, help="Rows in the Top props table")
    ap.add_argument("--dry-run", action="store_true", help="Show slate, injuries and credit cost; fetch no lines")
    ap.add_argument("--cached-only", action="store_true", help="Use only lines already saved; spend no credits")
    ap.add_argument("--no-injuries", action="store_true")
    args = ap.parse_args()

    game_date = dt.date.fromisoformat(args.date) if args.date else today_et()
    markets = [m.strip().upper() for m in args.markets.split(",") if m.strip()]
    bad = [m for m in markets if m not in oddsapi.MARKETS]
    if bad:
        sys.exit(f"Unknown prop types: {', '.join(bad)}. Options: {', '.join(oddsapi.MARKETS)}")
    books = [b.strip() for b in args.books.split(",") if b.strip()]
    t0 = time.time()

    log("Loading stats (falls back to the last saved copy if NBA.com can't be reached)...")
    season = nbadata.current_season(game_date)
    ds = nbadata.load_dataset(season, prior_games=20)
    tr = load_tracking(ds, prior_games=20)
    set_tracking(tr)
    names = dict(zip(ds.teams["TEAM_ABBREVIATION"], ds.teams["TEAM_NAME"]))
    warnings_ = []
    if nbadata.STALE:
        oldest = min(nbadata.STALE.values())
        warnings_.append(f"NBA.com couldn't be reached, so stats are from {oldest:%b %-d %-I:%M %p}. "
                         "Recent games, minutes and Play % may be out of date.")

    slate, source = schedule.games_on(game_date, names)
    if args.games:
        want = {tuple(x.strip().upper().split("@")) for x in args.games.split(",")}
        slate = [g for g in slate if (g["away"], g["home"]) in want]
    if not slate:
        sys.exit(f"No games found for {game_date} ({source}).")

    inj_report = None
    if not args.no_injuries:
        inj_report, inj_err, inj_stamp = injuries.fetch_report()
        if inj_err:
            warnings_.append(inj_err)

    key = oddsapi.api_key()
    cost = 3 + len(markets) * len(slate)
    left = (oddsapi.latest_quota() or {}).get("remaining")
    log(f"{len(slate)} games on {game_date} ({source}): " + ", ".join(f"{g['away']}@{g['home']}" for g in slate))
    log(f"Credit cost: up to {cost} (3 for game lines + {len(markets)} prop types x {len(slate)} games); "
        f"credits left: {left if left is not None else 'unknown'}")
    if args.dry_run:
        for g in slate:
            for t in (g["away"], g["home"]):
                if inj_report is None:
                    continue
                r = injuries.for_team(inj_report, names[t], game_date)
                pl = ds.players[ds.players["TEAM_ABBREVIATION"] == t]
                mpg = dict(zip(pl["PLAYER_NAME"], pl["MIN"] / pl["GP"].clip(lower=1)))
                mp = oddsapi.match_players(r["Player"], list(mpg))
                hurt = [f"{p} ({injuries.SHORT.get(s, s)})" for p, s in zip(r["Player"], r["Status"])
                        if s != "Available" and mpg.get(mp.get(p), 0) >= 15]
                if hurt:
                    log(f"  {t}: {', '.join(hurt)}")
        for w in warnings_:
            log(f"Warning: {w}")
        print(json.dumps({"date": str(game_date), "games": [f"{g['away']}@{g['home']}" for g in slate],
                          "credit_cost": cost, "credits_left": left, "warnings": warnings_}))
        return

    log("Fetching sportsbook lines...")
    game_lines, props, used, notes = fetch_lines(key, slate, names, game_date, markets, books,
                                                 args.cached_only)
    warnings_ += notes
    weights = Weights()
    games, rows = [], []
    for i, g in enumerate(slate):
        log(f"Simulating {g['away']} @ {g['home']} ({i + 1}/{len(slate)})...")
        game, r = run_game(ds, tr, weights, g, names, game_date, inj_report,
                           props.get((g["away"], g["home"])), args.sims)
        games.append(game)
        rows += r
        if game["unmatched"]:
            log(f"  not in the sim's rotation (or OUT), skipped: {', '.join(game['unmatched'])}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    stem = os.path.join(REPORT_DIR, f"slate_{game_date}")
    allp = pd.DataFrame(rows)
    if allp.empty:
        best = pd.DataFrame(columns=["Game", "Player", "Stat", "Pick", "Line", "Odds", "Book",
                                     "Value Score", "Grade", "Sim mean", "Sim P(over)",
                                     "Book P(over)", "EV", "L10 hit", "Proj Min", "Flags"])
    else:
        allp.to_csv(f"{stem}_all_lines.csv", index=False)
        best = allp.sort_values("Value Score", ascending=False).drop_duplicates(["Player", "Stat"])
    best.to_csv(f"{stem}.csv", index=False)
    meta = {"sims": args.sims, "generated": dt.datetime.now().strftime("%b %-d %-I:%M %p"),
            "markets": markets, "credits_used": used,
            "credits_left": (oddsapi.latest_quota() or {}).get("remaining"), "warnings": warnings_,
            "top": args.top}
    write_report(f"{stem}.html", game_date, games, best, game_lines, meta)

    top = best[best["Grade"].isin(["A", "B"])].head(args.top)
    summary = {
        "date": str(game_date), "games": len(games), "props_scored": int(len(allp)),
        "graded_A_or_B": int(best["Grade"].isin(["A", "B"]).sum()), "credits_used": used,
        "credits_left": meta["credits_left"], "warnings": warnings_,
        "report_html": f"{stem}.html", "report_csv": f"{stem}.csv",
        "seconds": round(time.time() - t0),
        "games_detail": [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in g.items()
                          if k in ("Game", "time", "Away pts", "Home pts", "Sim spread", "Sim total")}
                         | {"market": game_lines.get((g["away"], g["home"]))} for g in games],
        "top_props": [{"player": r["Player"], "game": r["Game"], "bet": f"{r['Pick'].title()} {r['Line']:g} {r['Stat']}",
                       "odds": fmt_odds(r["Odds"]), "book": r["Book"], "grade": r["Grade"],
                       "score": round(r["Value Score"]), "sim_mean": round(r["Sim mean"], 1),
                       "flags": r["Flags"]} for _, r in top.iterrows()],
    }
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
