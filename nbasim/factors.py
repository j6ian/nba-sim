"""Every adjustable model factor in one place: strength, data source, and why it's set where it is.

How strengths are chosen (three tiers):
  * Measured, stable data used at full strength (1.0): the thing being measured is the thing
    being simulated, with large samples (play-type mix, catch-and-shoot vs pull-up splits, rebound
    distance, transition frequency).
  * Noisy or overlapping data used at partial strength (0.3-0.6): individual defender stats,
    opponent scheme splits, fouls by defender. These have small samples per pair/play type and
    overlap with team-level defense the model already uses, so they're shrunk toward "no effect".
  * Not measurable from public data -> small effects from published research (rest/travel/
    altitude), clearly labeled as priors.
On top of the strength, every per-player or per-team rate is shrunk toward the league average by
its own sample size (the "Shrinkage" column), so small samples automatically count for less.
"""
from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class Weights:
    # Team defense
    def_strength: float = 0.8
    contest_process: float = 0.5
    home_edge: float = 0.012
    # Individual defenders on the floor
    defender_individual: float = 0.5
    rim_deterrence: float = 0.5
    defender_fouls: float = 0.4
    deflections: float = 0.5
    # Play types
    playtype_eff: float = 1.0
    scheme_freq: float = 0.5
    scheme_eff: float = 0.5
    # Passing, rebounding, transition
    assist_chain: float = 1.0
    potential_ast: float = 0.3
    rebound_length: float = 1.0
    rebound_chances: float = 0.25
    transition: float = 1.0
    putbacks: float = 1.0
    # Lineup context
    usage_eff: float = 0.15
    synergy: float = 0.08
    clutch: float = 1.0
    # Schedule
    rest_travel: float = 1.0

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# key, group, label, max for slider, data source, how it's used, shrinkage, why this default
REGISTRY = [
    ("def_strength", "Team defense", "Opponent shot defense", 1.5,
     "LeagueDashTeamShotLocations (opponent FG% by zone)",
     "Scales the opponent's FG% allowed by zone, turnovers forced and fouls vs league average.",
     "Full season of team data (thousands of shots)",
     "0.8, not 1.0: part of a team's FG% allowed is opponent shooting luck that doesn't repeat."),
    ("contest_process", "Team defense", "Contest quality share", 1.0,
     "LeagueDashOppPtShot (opponent shots by closest-defender distance)",
     "Splits team shot defense between results (FG% allowed) and process (how tightly shots are "
     "contested, valued at league eFG for each contest level).",
     "Thousands of contested shots per team",
     "0.5: contest data is more stable than FG% allowed, but misses things like rim deterrence, so the two are averaged."),
    ("home_edge", "Team defense", "Home-court shooting edge", 0.03,
     "Prior (league home/away splits)",
     "Make probability +/- this fraction at home / away.",
     "-",
     "About 1.2% per team, roughly 2.5 points of spread."),
    ("defender_individual", "Defenders on the floor", "Individual defender FG% impact", 1.0,
     "LeagueDashPtDefend (FG% allowed vs shooter's normal FG%) + LeagueSeasonMatchups",
     "Each shot gets a primary defender, chosen from tonight's on-floor defenders using real "
     "matchup possessions (who guarded whom). FG% shifts by that defender's (and help defenders') "
     "allowed-vs-expected margin relative to the team's normal lineup.",
     "300 shots defended of prior toward 0 (rim: 150)",
     "0.5: defender FG% is noisy year to year and partly measured in team defense already."),
    ("rim_deterrence", "Defenders on the floor", "Rim deterrence", 1.0,
     "LeagueDashPtDefend (<6 ft) + LeagueDashPtStats(Defense) rim contests",
     "Rim attempts shift to floaters / mid-range when strong rim protectors are on the floor, "
     "and rim FG% uses the on-floor rim protectors' allowed-vs-expected.",
     "150 rim shots of prior toward 0",
     "0.5: shots avoided aren't measured directly; inferred from rim contest volume."),
    ("defender_fouls", "Defenders on the floor", "Fouls by primary defender", 1.0,
     "LeagueSeasonMatchups (shooting fouls per partial possession, by defender)",
     "Shooting-foul and and-one chances scale with the primary defender's foul rate.",
     "150 matchup possessions of prior toward league rate",
     "0.4: foul counts per defender are small; referee crews (not available) also matter."),
    ("deflections", "Defenders on the floor", "Deflections -> turnovers / steals", 1.0,
     "LeagueHustleStatsPlayer (deflections)",
     "Turnover chance and live-ball (steal) share rise with the on-floor defenders' deflection rate; "
     "steals are credited by steal rate blended with deflections.",
     "Rates are relative to the team's own normal lineup",
     "0.5: deflections predict steals well but turnovers only partly."),
    ("playtype_eff", "Play types", "Player play-type efficiency", 1.5,
     "SynergyPlayTypes (player offense) + LeagueDashPtStats(CatchShoot / PullUpShot)",
     "Each play is a play type (isolation, P&R, spot-up, cut...). The type sets the shot zones, "
     "turnover and foul rates; catch-and-shoot types use the player's catch-and-shoot 3P%, "
     "pull-up types his pull-up 3P%; 2-point efficiency by type uses his Synergy FG%.",
     "40 possessions per type (60 attempts for 3P splits) toward league type averages",
     "1.0: this is the player's own measured profile, normalized so his overall efficiency is unchanged."),
    ("scheme_freq", "Play types", "Opponent scheme: play-type mix", 1.0,
     "SynergyPlayTypes (team defense: frequency allowed by play type)",
     "Shifts each player's play-type mix toward what tonight's defense allows (e.g. a drop "
     "coverage team allows more P&R ball-handler pull-ups; a switching team more isolations).",
     "Team-season samples (hundreds of possessions per type)",
     "0.5 (square root of the ratio): offenses adapt, but players keep most of their own habits."),
    ("scheme_eff", "Play types", "Opponent scheme: play-type defense", 1.0,
     "SynergyPlayTypes (team defense: points per possession allowed by type)",
     "Efficiency by play type relative to how good the defense is overall (only the relative "
     "strength/weakness, so it doesn't double count team defense).",
     "150 possessions of prior toward the team's overall defense",
     "0.5: play-type PPP allowed is noisy, especially for rarer play types."),
    ("assist_chain", "Passing", "Assist chains by play type", 1.5,
     "SynergyPlayTypes + LeagueDashPtStats(Passing, Drives)",
     "Chance a make was assisted depends on play type (spot-up/cut/roll mostly assisted; isolation "
     "rarely). The passer is weighted by the matching creation skill: P&R ball handlers on rolls, "
     "drive-and-kick passers on spot-ups, potential-assist leaders on cuts.",
     "Team assist rate is calibrated to its real AST per FGM",
     "1.0: shares are normalized so each player's assist total is preserved; only who assists which shot type changes."),
    ("potential_ast", "Passing", "Potential assists stabilizer", 1.0,
     "LeagueDashPtStats(Passing): potential assists",
     "Passer strength blends actual assists with potential assists x league conversion rate, "
     "so a passer whose teammates have been shooting cold isn't undersold.",
     "-",
     "0.3: potential assists are steadier but actual assists are what pays out on props."),
    ("rebound_length", "Rebounding", "Rebound share by shot type", 2.0,
     "LeagueDashPtStats(Rebounding): average rebound distance",
     "Missed 3s produce long rebounds (more to players who rebound far from the rim, usually "
     "guards/wings); rim misses and FT misses go to players who rebound close (bigs). Offensive "
     "rebound chance also varies by shot type.",
     "Defaults to league distance when a player has few rebounds",
     "1.0: about a 1.3x shift per 1 ft of rebound-distance difference on long vs short misses."),
    ("rebound_chances", "Rebounding", "Rebound chances stabilizer", 1.0,
     "LeagueDashPtStats(Rebounding): rebound chances",
     "Rebound weight blends actual rebounds with rebound chances x league conversion.",
     "-",
     "0.25: chances are steadier, but elite rebounders convert more of them, and conversions are what count."),
    ("transition", "Transition", "Transition offense/defense", 2.0,
     "SynergyPlayTypes (Transition, team offense + defense), LeagueDashTeamStats(Misc) fast-break pts",
     "Possessions after steals (most often), defensive rebounds and makes can be transition, at "
     "rates set by the offense's push rate and the defense's transition allowed. Transition plays "
     "use each player's transition frequency and efficiency.",
     "Team-season samples",
     "1.0: calibrated to the league's real transition share of possessions."),
    ("putbacks", "Transition", "Putbacks", 2.0,
     "SynergyPlayTypes (OffRebound) / tracking offensive rebounds",
     "After an offensive rebound, the rebounder may go straight back up (putback play type).",
     "20 offensive rebounds of prior toward league putback rate",
     "1.0: measured directly."),
    ("usage_eff", "Lineup context", "Usage vs efficiency tradeoff", 0.4,
     "Prior (usage-efficiency research)",
     "Carrying more of the load than usual in a lineup lowers make probability.",
     "-",
     "0.15 exponent: +20% usage costs about 1.5 FG% points, in line with published estimates."),
    ("synergy", "Lineup context", "Teammate passing quality", 0.2,
     "Prior + calibrated assist weights",
     "Better passers around a shooter than usual raise make and assisted rates.",
     "-",
     "0.08: modest; real on/off data for this isn't public."),
    ("clutch", "Lineup context", "Clutch usage", 2.0,
     "LeagueDashPlayerClutch (last 5 min, score within 5)",
     "In the last 5 minutes of close games, each player's usage scales by his clutch usage vs normal.",
     "60 clutch minutes of prior toward no change",
     "1.0 with heavy shrinkage: clutch samples are small."),
    ("rest_travel", "Schedule", "Rest, travel, altitude", 2.0,
     "Game logs (schedule) + arena locations; effect sizes are priors from published research",
     "Back-to-backs, 3 games in 4 nights, long travel on no rest and visiting Denver/Utah lower "
     "make probability and raise turnovers slightly.",
     "-",
     "1.0 = research-sized effects (back-to-back about -1.5 pts per 100). Not measurable per player here."),
]

NOT_AVAILABLE = [
    ("Referee tendencies", "NBA.com stats endpoints don't publish referee crew stats or tonight's "
     "assignments in a form this app can pull. Use the defender-foul and FT settings, or adjust a "
     "player's FT projection by hand if you know the crew."),
    ("Exact coaching schemes (drop / switch / blitz / zone)", "Not published. The model infers "
     "scheme from what each defense allows by Synergy play type (frequency and points per possession)."),
    ("Usage with specific teammates on the floor (true on/off)", "Not published per player pair. "
     "Modeled with lineup simulation plus whole-game with/without splits from game logs."),
    ("Live vs dead-ball turnover types per player", "Not published by type. Live-ball share comes "
     "from team steals per opponent turnover, adjusted by on-floor deflections."),
]
