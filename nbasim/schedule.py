"""The day's NBA games, for the slate on the app's front page.

Sources, in order:
  1. NBA.com's season schedule (cdn.nba.com scheduleLeagueV2), cached on disk. Covers any date.
     For today, scores and game status come from NBA.com's live scoreboard.
  2. The Odds API's event list (free, no credits) when NBA.com can't be reached. Upcoming games
     only, no records or scores.
If NBA.com can't be reached, the last saved schedule is used no matter how old (it rarely changes).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time

import pandas as pd
import requests

from .data import CACHE_DIR

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"),
    "Referer": "https://www.nba.com/", "Origin": "https://www.nba.com",
    "Accept": "application/json",
}
SCHEDULE_MAX_AGE_H = 12
ET = "US/Eastern"


def _get_json(url: str) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


def _schedule() -> tuple[dict | None, str | None]:
    """Season schedule JSON (disk-cached) and an error message if NBA.com couldn't be reached."""
    path = os.path.join(CACHE_DIR, "schedule.json")
    fresh = os.path.exists(path) and (time.time() - os.path.getmtime(path)) / 3600 < SCHEDULE_MAX_AGE_H
    if not fresh:
        try:
            data = _get_json(SCHEDULE_URL)
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(path, "w") as fh:
                json.dump(data, fh)
            return data, None
        except (requests.RequestException, ValueError) as exc:
            err = f"NBA.com schedule unavailable ({exc.__class__.__name__})"
            if not os.path.exists(path):
                return None, err
    with open(path) as fh:
        return json.load(fh), None


def _et(utc: str | None) -> pd.Timestamp | None:
    if not utc:
        return None
    try:
        return pd.Timestamp(utc).tz_convert(ET) if pd.Timestamp(utc).tzinfo else \
            pd.Timestamp(utc).tz_localize("UTC").tz_convert(ET)
    except (ValueError, TypeError):
        return None


def _team(t: dict) -> dict:
    w, l = t.get("wins"), t.get("losses")
    return {"abbr": t.get("teamTricode"), "score": t.get("score"),
            "record": f"{w}-{l}" if w is not None and l is not None else ""}


def _from_nba(data: dict, date: dt.date) -> list[dict]:
    games = []
    for day in data.get("leagueSchedule", {}).get("gameDates", []):
        d = pd.to_datetime(day.get("gameDate"), errors="coerce")
        if d is pd.NaT or d.date() != date:
            continue
        for g in day.get("games", []):
            # Preseason and All-Star games are listed too; the sim only knows NBA regular teams.
            if str(g.get("gameId", "")).startswith(("001", "003")):
                continue
            a, h = _team(g.get("awayTeam", {})), _team(g.get("homeTeam", {}))
            if not a["abbr"] or not h["abbr"]:
                continue
            games.append({"game_id": g.get("gameId"), "away": a["abbr"], "home": h["abbr"],
                          "start": _et(g.get("gameDateTimeUTC")), "status": int(g.get("gameStatus") or 1),
                          "status_text": (g.get("gameStatusText") or "").strip(),
                          "away_score": a["score"], "home_score": h["score"],
                          "away_record": a["record"], "home_record": h["record"],
                          "arena": g.get("arenaName", "")})
    return games


def _live(games: list[dict]) -> list[dict]:
    """Overlay today's live status and scores."""
    try:
        sb = _get_json(SCOREBOARD_URL).get("scoreboard", {})
    except (requests.RequestException, ValueError):
        return games
    live = {g.get("gameId"): g for g in sb.get("games", [])}
    for g in games:
        lg = live.get(g["game_id"])
        if not lg:
            continue
        a, h = _team(lg.get("awayTeam", {})), _team(lg.get("homeTeam", {}))
        g.update(status=int(lg.get("gameStatus") or g["status"]),
                 status_text=(lg.get("gameStatusText") or g["status_text"]).strip(),
                 away_score=a["score"], home_score=h["score"],
                 away_record=a["record"] or g["away_record"], home_record=h["record"] or g["home_record"])
    return games


def _from_odds_api(date: dt.date, team_names: dict) -> list[dict]:
    from . import odds
    key = odds.api_key()
    if not key:
        return []
    events, _ = odds.list_events(key)
    by_nick = {odds._nick(n): a for a, n in team_names.items()}
    games = []
    for e in events:
        start = _et(e.get("commence_time"))
        if start is None or start.date() != date:
            continue
        a, h = by_nick.get(odds._nick(e["away_team"])), by_nick.get(odds._nick(e["home_team"]))
        if a and h:
            games.append({"game_id": e.get("id"), "away": a, "home": h, "start": start, "status": 1,
                          "status_text": "", "away_score": None, "home_score": None,
                          "away_record": "", "home_record": "", "arena": ""})
    return games


def games_on(date: dt.date, team_names: dict) -> tuple[list[dict], str]:
    """Games on `date` (ET), sorted by tip-off, and where they came from.

    team_names: {abbr: full team name}; games with a team the app doesn't know are dropped.
    """
    games, source = [], ""
    data, err = _schedule()
    if data is not None:
        games = _from_nba(data, date)
        source = "NBA.com"
        if games and date == pd.Timestamp.now(tz=ET).date():
            games = _live(games)
    if data is None:
        try:
            games = _from_odds_api(date, team_names)
            source = "The Odds API, upcoming games only (NBA.com unavailable)"
        except Exception:  # noqa: BLE001 - any failure here just means no slate
            games, source = [], err or "unavailable"
    games = [g for g in games if g["away"] in team_names and g["home"] in team_names]
    far = pd.Timestamp("2100-01-01", tz=ET)
    games.sort(key=lambda g: (g["start"] if g["start"] is not None else far, g["home"]))
    return games, source


def status_line(g: dict) -> str:
    """'7:30 PM ET', 'Q3 5:12' or 'Final'."""
    if g["status"] >= 2 and g.get("status_text"):
        return g["status_text"]
    if g["start"] is not None:
        return g["start"].strftime("%-I:%M %p ET")
    return g.get("status_text") or "TBD"
