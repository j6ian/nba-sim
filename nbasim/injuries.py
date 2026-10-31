"""Injury report from ESPN's public NBA injuries feed.

ESPN lists each player as "Out" or "Day-To-Day" with a dated news note. Day-to-day players'
notes usually say what they are for the next game ("downgraded to questionable for Friday's
game"), so a recent note is read for out / doubtful / questionable / probable / available.

How the sim uses it:
  * Out: ticked OUT, so his minutes and shots go to teammates.
  * Doubtful / questionable / day-to-day / probable: his Play % (chance he plays at all) is
    multiplied by PLAY_MULT. When he plays he gets his normal minutes; props only count games he
    plays, the way books void a prop when a player doesn't suit up.
The feed is cached for 20 minutes; if ESPN can't be reached, the last saved copy is used.
"""
from __future__ import annotations

import json
import os
import re
import time

import pandas as pd
import requests

from .data import CACHE_DIR
from .odds import match_players

URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
MAX_AGE_MIN = 20
NOTE_FRESH_DAYS = 2   # an older day-to-day note says nothing about tonight

# Chance he plays, relative to normal. Rough public-research rates for each designation.
PLAY_MULT = {"Doubtful": 0.25, "Questionable": 0.55, "Day-To-Day": 0.75, "Probable": 0.95,
             "Available": 1.0}
SHORT = {"Out": "O", "Doubtful": "D", "Questionable": "Q", "Day-To-Day": "GTD", "Probable": "P",
         "Available": "A"}

_NOTE_PATTERNS = [   # first match wins, checked in this order
    ("Out", r"\bruled out\b|\bwon't play\b|\bwill not play\b|\bwill miss\b|\bis out\b|\bout for\b|\bwill sit\b|\bnot be available\b|\bunavailable\b"),
    ("Doubtful", r"\bdoubtful\b"),
    ("Questionable", r"\bquestionable\b|\bgame-time decision\b"),
    ("Probable", r"\bprobable\b|\bexpected to play\b"),
    ("Available", r"\bavailable\b|\bwill play\b|\bcleared to play\b|\bupgraded to available\b|\bin the lineup\b"),
]


def _nick(team_name: str) -> str:
    return team_name.strip().split()[-1].lower()


def fetch_report() -> tuple[pd.DataFrame, str | None, str]:
    """All injuries as a frame, an error message if ESPN couldn't be reached, and the feed time."""
    path = os.path.join(CACHE_DIR, "injuries.json")
    data, err = None, None
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) / 60 < MAX_AGE_MIN:
        with open(path) as fh:
            data = json.load(fh)
    else:
        try:
            r = requests.get(URL, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            data = r.json()
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(path, "w") as fh:
                json.dump(data, fh)
        except (requests.RequestException, ValueError) as exc:
            err = f"ESPN injury report unavailable ({exc.__class__.__name__})"
            if os.path.exists(path):
                with open(path) as fh:
                    data = json.load(fh)
                err += "; using the last saved copy"
    rows = []
    for team in (data or {}).get("injuries", []):
        for x in team.get("injuries", []):
            rows.append({"team_nick": _nick(team.get("displayName", "")),
                         "Player": x.get("athlete", {}).get("displayName", ""),
                         "feed_status": x.get("status", ""), "note": x.get("shortComment", ""),
                         "date": pd.to_datetime(x.get("date"), utc=True, errors="coerce"),
                         "return": (x.get("details") or {}).get("returnDate")})
    df = pd.DataFrame(rows, columns=["team_nick", "Player", "feed_status", "note", "date", "return"])
    stamp = ""
    if data and data.get("timestamp"):
        ts = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
        if ts is not pd.NaT:
            stamp = ts.tz_convert("US/Eastern").strftime("%b %-d %-I:%M %p ET")
    return df, err, stamp


_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def classify(feed_status: str, note: str, date, game_date) -> str:
    """The player's designation for the game on `game_date`, from ESPN's status plus his latest
    note. A note about a different game ("ruled out for Thursday's game" on a Friday) only makes
    him day-to-day."""
    if feed_status.lower() == "out":
        return "Out"
    game_date = pd.Timestamp(game_date)
    fresh = date is not pd.NaT and date is not None and \
        abs((date.tz_convert("US/Eastern").tz_localize(None).normalize() - game_date).days) <= NOTE_FRESH_DAYS
    if not fresh:
        return "Day-To-Day"
    low = (note or "").lower()
    named = [d for d in _DAYS if re.search(rf"\b{d}'?s?\b", low)]
    if named and _DAYS[game_date.weekday()] not in named:
        return "Day-To-Day"
    for status, pat in _NOTE_PATTERNS:
        if re.search(pat, low):
            return status
    return "Day-To-Day"


def for_team(report: pd.DataFrame, team_name: str, game_date) -> pd.DataFrame:
    """One team's injuries, with each player's Status for the game on `game_date`."""
    r = report[report["team_nick"] == _nick(team_name)].copy()
    r["Status"] = [classify(s, n, d, game_date) for s, n, d in zip(r["feed_status"], r["note"], r["date"])]
    return r


def apply(minutes: pd.DataFrame, team_report: pd.DataFrame) -> pd.DataFrame:
    """Tick OUT and scale Play % for tonight's injured players. Adds an Injury column."""
    m = minutes.copy()
    m["Injury"] = ""
    if m.empty or team_report.empty:
        return m
    mapping = match_players(team_report["Player"], list(m["Player"]))
    for _, r in team_report.iterrows():
        name = mapping.get(r["Player"])
        if name is None:
            continue
        i = m.index[m["Player"] == name]
        status = r["Status"]
        m.loc[i, "Injury"] = f"{status}: {r['note']}" if r["note"] else status
        if status == "Out":
            m.loc[i, "OUT"] = True
        elif "Play %" in m:
            m.loc[i, "Play %"] = (m.loc[i, "Play %"] * PLAY_MULT.get(status, 1.0)).clip(0.05, 1.0).round(2)
    return m
