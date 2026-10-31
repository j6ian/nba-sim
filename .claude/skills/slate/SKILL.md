---
name: slate
description: Run tonight's NBA slate. Refresh stats and injuries, pull sportsbook lines for every game (after confirming the credit cost), simulate all games, and report the best player props. Use when the user asks for tonight's slate, today's best props, picks for all games, or runs /slate.
argument-hint: "[date YYYY-MM-DD] [prop types, e.g. PTS,REB,AST,3PM] [games, e.g. LAL@GSW]"
---

# Nightly slate

Everything runs through `scripts/run_slate.py` with the project's virtualenv
(`.venv/bin/python`). It writes `reports/slate_<date>.html` (the report), `reports/slate_<date>.csv`
(best book and side for each player + stat) and `reports/slate_<date>_all_lines.csv`, and prints
a JSON summary on stdout. Progress goes to stderr.

## 1. Read the arguments

From `$ARGUMENTS` or the user's words, pick out:
- **Date**: default today (US Eastern). Pass `--date YYYY-MM-DD` only for another day.
- **Prop types**: default `PTS,REB,AST,3PM`. Options: PTS, REB, AST, 3PM, PRA, PR, PA, RA, STL,
  BLK, STL+BLK, TOV, FTM. Pass as `--markets`.
- **Games**: default all. To limit, pass `--games "AWAY@HOME,..."` with team abbreviations.
- If the user says not to spend credits, add `--cached-only` and skip the cost question in step 2.

## 2. Dry run, then confirm the cost

Run the dry run first. It spends no credits:

```bash
.venv/bin/python scripts/run_slate.py --dry-run [--date ...] [--markets ...] [--games ...]
```

The last stdout line is JSON with `credit_cost`, `credits_left`, `games` and `warnings`; stderr
lists rotation players who are out or questionable.

Then **ask the user to confirm before spending credits**. Tell them the games, the cost
("up to N credits, M left") and any warnings (stale stats, injury feed errors). Offer
`--cached-only` as the free option if lines were pulled in the last few minutes. Skip the question
only if the user already approved a cost at least this high in this conversation, or this
is a scheduled run whose prompt sets a credit limit at or above the cost.

Credits are limited (500 a month on the free plan; the user pays for more). Never re-run a full
pull just to retry. Responses are cached for 10 minutes, so re-running the same pull right away
is free.

## 3. Run it

```bash
.venv/bin/python scripts/run_slate.py [same flags]
```

A 10-game night at 3,000 sims takes a few minutes; run it in the background and wait for it to
finish. Don't poll.

## 4. Report

Read the JSON summary and give the user:
1. One line: games simulated, props scored, how many graded A or B, credits used and left.
2. **Warnings first**, if any: stale stats (NBA.com unreachable), injury feed problems, games with
   no props posted.
3. The top 10 props as a table: player, game, bet (e.g. "Under 26.5 PTS"), odds and book, grade,
   sim mean and flags. Say plainly that flagged props depend on an injury call, a teammate being
   out or a big minutes change, so the user should check the news before betting them.
4. Games where the sim's spread or total is far from the market (4+ points), as a short list.
   That usually means the model is missing news, not that the market is wrong.
5. A link to the HTML report as a relative path, e.g. `[reports/slate_2026-10-30.html](reports/slate_2026-10-30.html)`.
   If a send-file tool is available, send the HTML report so the user can open it on their phone.

Keep it honest. If more than about a third of props grade A or B, say the model is probably
overconfident tonight. This is model output, not betting advice. Never place bets or open
sportsbook sites.

## Troubleshooting

- **"No games found"**: wrong date, or NBA.com and The Odds API are both unreachable. The Odds
  API only lists upcoming games, so past dates need NBA.com.
- **Stale stats warning**: NBA.com blocks some networks. The run still works on the last saved
  stats; the app's "Refresh data from NBA.com" button (or running again on another network)
  updates them.
- **429 from The Odds API**: out of credits. Stop and tell the user.
