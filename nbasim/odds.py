"""Live player-prop lines and odds from sportsbooks via The Odds API (the-odds-api.com).

Sportsbooks don't offer public APIs, and scraping their sites breaks their terms, so lines come
from The Odds API, a licensed aggregator. You need your own API key (free plan: 500 credits a
month). Set it as the ODDS_API_KEY environment variable or paste it in the app.

Cost: listing games is free. Props are fetched one game at a time and cost 1 credit per prop
market returned for up to 10 books (FanDuel, DraftKings, Fanatics, ... all in one request).
Responses are cached on disk so reruns and page refreshes don't spend credits.
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata

import pandas as pd
import requests

from .data import CACHE_DIR

BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

# Our prop stat -> The Odds API market key.
MARKETS = {
    "PTS": "player_points", "REB": "player_rebounds", "AST": "player_assists",
    "3PM": "player_threes", "PRA": "player_points_rebounds_assists",
    "PR": "player_points_rebounds", "PA": "player_points_assists", "RA": "player_rebounds_assists",
    "STL": "player_steals", "BLK": "player_blocks", "STL+BLK": "player_blocks_steals",
    "TOV": "player_turnovers", "FTM": "player_frees_made",
}
MARKET_TO_STAT = {v: k for k, v in MARKETS.items()}
MARKET_LABELS = {
    "PTS": "Points", "REB": "Rebounds", "AST": "Assists", "3PM": "Threes made",
    "PRA": "Pts+Reb+Ast", "PR": "Pts+Reb", "PA": "Pts+Ast", "RA": "Reb+Ast", "STL": "Steals",
    "BLK": "Blocks", "STL+BLK": "Steals+Blocks", "TOV": "Turnovers", "FTM": "Free throws made",
}
DEFAULT_MARKETS = ["PTS", "REB", "AST", "3PM"]

# Book key -> display name. Up to 10 books in one request are billed as a single region.
BOOKS = {
    "fanduel": "FanDuel", "draftkings": "DraftKings", "fanatics": "Fanatics", "betmgm": "BetMGM",
    "williamhill_us": "Caesars", "espnbet": "ESPN Bet", "hardrockbet": "Hard Rock",
    "betrivers": "BetRivers", "betonlineag": "BetOnline", "bovada": "Bovada",
}
DEFAULT_BOOKS = ["fanduel", "draftkings", "fanatics", "betmgm", "williamhill_us", "espnbet"]


class OddsError(RuntimeError):
    pass


ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def api_key() -> str | None:
    """ODDS_API_KEY from the environment, else from a local .env file next to app.py."""
    if os.environ.get("ODDS_API_KEY"):
        return os.environ["ODDS_API_KEY"]
    try:
        with open(ENV_FILE) as fh:
            for line in fh:
                k, _, v = line.strip().partition("=")
                if k == "ODDS_API_KEY" and v:
                    return v.strip().strip('"').strip("'")
    except OSError:
        pass
    return None


def _get(path: str, key: str, params: dict) -> tuple[object, dict]:
    try:
        r = requests.get(f"{BASE}{path}", params={"apiKey": key, **params}, timeout=20)
    except requests.RequestException as exc:
        raise OddsError(f"Couldn't reach The Odds API: {exc}") from exc
    quota = {k: r.headers.get(f"x-requests-{k}") for k in ("remaining", "used", "last")}
    if r.status_code == 401:
        raise OddsError("The Odds API rejected the key (401). Check that it's copied correctly.")
    if r.status_code == 429:
        raise OddsError("Out of Odds API credits or rate limited (429).")
    if r.status_code == 422:
        raise OddsError(f"The Odds API couldn't fulfil the request (422): {r.text[:200]}")
    if not r.ok:
        raise OddsError(f"The Odds API returned {r.status_code}: {r.text[:200]}")
    if quota.get("remaining") is not None:
        _save_quota(quota)
    return r.json(), quota


QUOTA_FILE = os.path.join(CACHE_DIR, "odds", "quota.json")


def _save_quota(quota: dict):
    """Remember the credit count. The API's quota headers sometimes lag (two calls a few seconds
    apart can report 481 and 431 left), so within a month keep whichever reading has used more
    credits; usage only goes up until the monthly reset."""
    now = time.time()
    try:
        with open(QUOTA_FILE) as fh:
            old = json.load(fh)
        same_month = time.strftime("%Y-%m", time.localtime(old["at"])) == time.strftime("%Y-%m")
        if same_month and int(old.get("used") or 0) > int(quota.get("used") or 0):
            quota = {**quota, "used": old["used"], "remaining": old["remaining"]}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        os.makedirs(os.path.dirname(QUOTA_FILE), exist_ok=True)
        with open(QUOTA_FILE, "w") as fh:
            json.dump({**quota, "at": now}, fh)
    except OSError:
        pass


def latest_quota() -> dict | None:
    """Credits left as of the most recent request (any endpoint, including the free ones)."""
    try:
        with open(QUOTA_FILE) as fh:
            q = json.load(fh)
    except (OSError, ValueError):
        return None
    q["checked"] = time.strftime("%H:%M", time.localtime(q["at"]))
    return q


def _nick(team_name: str) -> str:
    """'Portland Trail Blazers' / 'LA Clippers' -> 'blazers' / 'clippers'."""
    return team_name.strip().split()[-1].lower()


def list_events(key: str) -> tuple[list, dict]:
    """Upcoming NBA games. Free: doesn't use credits."""
    return _get(f"/sports/{SPORT}/events", key, {"dateFormat": "iso"})


def find_event(events: list, home_name: str, away_name: str, game_date=None) -> dict | None:
    hn, an = _nick(home_name), _nick(away_name)
    matches = [e for e in events if _nick(e["home_team"]) == hn and _nick(e["away_team"]) == an]
    if not matches:
        return None
    if game_date is not None:
        target = pd.Timestamp(game_date)
        # Odds API times are UTC; US evening games fall on the next UTC day.
        matches.sort(key=lambda e: abs((pd.Timestamp(e["commence_time"]).tz_convert("US/Eastern")
                                        .tz_localize(None).normalize() - target).days))
    return matches[0]


def fetch_props(key: str, event_id: str, stats: list[str], books: list[str],
                max_age_min: float = 10.0, cache_only: bool = False) -> tuple[dict, dict]:
    """Prop odds for one game. Cached on disk for `max_age_min` minutes to save credits.
    cache_only: never spend credits; raise OddsError if these lines aren't saved."""
    markets = ",".join(MARKETS[s] for s in stats)
    tag = re.sub(r"[^a-z0-9]+", "_", f"{event_id}_{markets}_{','.join(sorted(books))}".lower())
    path = os.path.join(CACHE_DIR, "odds", f"{tag[:180]}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) / 60 < max_age_min:
        with open(path) as fh:
            saved = json.load(fh)
        return saved["data"], {**saved["quota"], "cached": True}
    if cache_only:
        raise OddsError("no saved lines for these prop types and books")
    data, quota = _get(f"/sports/{SPORT}/events/{event_id}/odds", key, {
        "bookmakers": ",".join(books), "markets": markets, "oddsFormat": "american",
        "dateFormat": "iso"})
    with open(path, "w") as fh:
        json.dump({"data": data, "quota": quota, "fetched": time.time()}, fh)
    return data, {**quota, "cached": False}


def fetch_game_lines(key: str, books: list[str], max_age_min: float = 10.0) -> tuple[list, dict]:
    """Moneyline, spread and total for every upcoming game in one request (3 credits)."""
    path = os.path.join(CACHE_DIR, "odds", f"game_lines_{'_'.join(sorted(books))}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) / 60 < max_age_min:
        with open(path) as fh:
            saved = json.load(fh)
        return saved["data"], {**saved["quota"], "cached": True,
                               "fetched": time.strftime("%H:%M", time.localtime(saved["fetched"]))}
    data, quota = _get(f"/sports/{SPORT}/odds", key, {
        "bookmakers": ",".join(books), "markets": "h2h,spreads,totals", "oddsFormat": "american",
        "dateFormat": "iso"})
    with open(path, "w") as fh:
        json.dump({"data": data, "quota": quota, "fetched": time.time()}, fh)
    return data, {**quota, "cached": False, "fetched": time.strftime("%H:%M")}


def game_line_summary(event: dict) -> dict:
    """Consensus (median across books) home spread, total and moneylines for one game."""
    spreads, totals, ml_home, ml_away = [], [], [], []
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            for oc in mk.get("outcomes", []):
                if mk["key"] == "spreads" and oc.get("name") == event["home_team"] and oc.get("point") is not None:
                    spreads.append(float(oc["point"]))
                elif mk["key"] == "totals" and str(oc.get("name")).lower() == "over" and oc.get("point") is not None:
                    totals.append(float(oc["point"]))
                elif mk["key"] == "h2h" and oc.get("price") is not None:
                    (ml_home if oc.get("name") == event["home_team"] else ml_away).append(float(oc["price"]))
    med = lambda v: float(pd.Series(v).median()) if v else None
    return {"home_spread": med(spreads), "total": med(totals), "home_ml": med(ml_home),
            "away_ml": med(ml_away), "books": len(event.get("bookmakers", []))}


def props_frame(event_odds: dict) -> pd.DataFrame:
    """Flatten an event-odds response into one row per book x player x stat x line, with the
    over and under prices side by side."""
    rows = []
    for bk in event_odds.get("bookmakers", []):
        for mk in bk.get("markets", []):
            stat = MARKET_TO_STAT.get(mk["key"])
            if stat is None:
                continue
            for oc in mk.get("outcomes", []):
                side = str(oc.get("name", "")).lower()
                if side not in ("over", "under") or oc.get("point") is None:
                    continue
                rows.append({"Book": BOOKS.get(bk["key"], bk.get("title", bk["key"])),
                             "book_key": bk["key"], "Player": oc.get("description", ""),
                             "Stat": stat, "Line": float(oc["point"]), "side": side,
                             "price": oc.get("price"), "Updated": mk.get("last_update")})
    if not rows:
        return pd.DataFrame(columns=["Book", "book_key", "Player", "Stat", "Line", "Over odds",
                                     "Under odds", "Updated"])
    df = pd.DataFrame(rows)
    df["Updated"] = df["Updated"].fillna("")
    wide = df.pivot_table(index=["Book", "book_key", "Player", "Stat", "Line", "Updated"],
                          columns="side", values="price", aggfunc="first").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"over": "Over odds", "under": "Under odds"})
    for c in ("Over odds", "Under odds"):
        if c not in wide:
            wide[c] = float("nan")
    # Some books post only one side. Infer the other from a typical 4.5% hold so the book's
    # no-vig probability stays sensible, and flag the row.
    wide["One-sided"] = wide["Over odds"].isna() | wide["Under odds"].isna()
    wide = wide[~(wide["Over odds"].isna() & wide["Under odds"].isna())].copy()
    for have, miss in (("Over odds", "Under odds"), ("Under odds", "Over odds")):
        m = wide[miss].isna()
        wide.loc[m, miss] = wide.loc[m, have].map(lambda o: _prob_to_american(1.045 - _american_to_prob(o)))
    return wide


def _american_to_prob(o: float) -> float:
    o = float(o)
    return 100 / (o + 100) if o > 0 else -o / (-o + 100)


def _prob_to_american(p: float) -> float:
    p = min(max(p, 0.01), 0.99)
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


# --------------------------------------------------------------------------- name matching

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_name(name: str) -> str:
    """'Nikola Jokić' / 'Nikola Jokic' -> 'nikola jokic'; drops Jr./III, dots and hyphens."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'’`]", "", s)
    s = re.sub(r"[-]", " ", s)
    s = _SUFFIX.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def match_players(book_names, sim_names) -> dict:
    """Map sportsbook player names to the sim's names (exact after normalizing, then
    first-initial + last name)."""
    by_norm = {norm_name(n): n for n in sim_names}
    by_short = {}
    for n in sim_names:
        parts = norm_name(n).split()
        if len(parts) >= 2:
            by_short.setdefault(f"{parts[0][0]} {parts[-1]}", []).append(n)
    out = {}
    for b in set(book_names):
        nb = norm_name(b)
        if nb in by_norm:
            out[b] = by_norm[nb]
            continue
        parts = nb.split()
        if len(parts) >= 2:
            cands = by_short.get(f"{parts[0][0]} {parts[-1]}", [])
            if len(cands) == 1:
                out[b] = cands[0]
    return out
