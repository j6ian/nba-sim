# NBA Matchup Simulator

Simulates NBA games possession by possession using live stats from stats.nba.com, then scores
player props against your book's lines.

## Run

```
./run.sh
```

First launch downloads ~2 seasons of data (about a minute), cached in `cache/`. The current
season refreshes every 6 hours, or click **Refresh data from NBA.com**.

## Tabs

The top of the page lists the games on the sidebar's game date (today by default), with
tip-off times, records, and live scores once games start. The first game opens by default;
click **Open** on any game to simulate it in the tabs below. The schedule comes from NBA.com
(cached on disk); if NBA.com can't be reached, upcoming games come from The Odds API's free
event list instead.

- **Game Sim**: win %, projected score, spread, and total from thousands of simulated games.
  Also pace (possessions), each team's shot volume (FGA / 3PA / FTA / TOV), points by quarter,
  shot zones, and projected box scores. Enter a market spread or total to get cover %.
- **Player Props**: enter lines and odds. For each prop you get sim P(over), the book's no-vig
  probability, the edge, EV, season / L5 / L10 averages, hit rates, opponent rank, and a
  **Value Score** (0-100, with 50 meaning no edge) and grade for the better side.
- **Player Deep Dive**: TS%, eFG%, usage, shot diet vs league, FG% by zone vs what tonight's
  opponent allows, last-20 game log against the line, simulated distribution, and minutes trend.
  **Lineup context** shows his simulated usage with vs without each teammate on the floor, next
  to his real per-36 production in games each teammate played vs missed.
- **Matchup**: pace, offensive / defensive / net ratings, four factors, and per-game stats allowed,
  with ranks. Also zone defense.
- **Rotations**: projected minutes for every player. Edit minutes or mark injured players OUT;
  everything reruns. Below that: each player's role and normal on-floor usage, the most-used
  simulated lineups with the usage split inside each, and how tonight's absences shift usage
  (lineup model vs real games without that teammate).
- **Watch a Game**: one simulated game with a line score, box scores, and play-by-play.

## How the model works

- **Data**: current season blended with last season. Last season is worth at most N games
  (default 20), so early-season samples don't swing everything.
- **Minutes**: **Proj Min** is minutes when he plays (the team's last 5 games he played,
  blended with the season average). **Play %** is the chance he gets in at all, from how many
  of the team's last 10 games he played. Each simulated game some players DNP and the rest get
  their normal minutes; expected minutes (Proj Min x Play %) are fit to 240 by trimming the end
  of the bench first. An OUT player's minutes go mostly to teammates in his position group
  (G / F / C). Blowout rest in the 4th is balanced by heavier starter minutes in close games, so
  a player's average stays at his projection.
- **Injuries**: ESPN's NBA injury report (refreshed every 20 minutes) is applied automatically.
  Out players are ticked OUT. For day-to-day players, the latest news note is read for the game
  date (questionable, doubtful, probable...), and their Play % is multiplied by 0.25 (doubtful),
  0.55 (questionable), 0.75 (day-to-day, no detail) or 0.95 (probable). A note about a different
  game ("ruled out for Thursday's game" on a Friday) counts as plain day-to-day. Statuses show in
  the Rotations tab's **Injury** column and on the game cards (rotation players only). Override
  any of it in Rotations, or turn it off with **Use injury report** in the sidebar.
- **Prop odds**: only simulated games the player plays count (books void a DNP), and the sim's
  P(over) is shrunk toward 50/50 (`PROB_SHRINK` in props.py) because raw sim probabilities
  backtested as overconfident. Check with `python scripts/backtest_props.py --projected`.
- **Lineups**: every simulated game has a minute-by-minute rotation. Role templates have
  starters opening halves and closing games, the bench covering the middle, and the top-usage
  starter staggered to play with the bench. The templates are fit so exactly five are on the
  floor and everyone hits his sampled minutes, with real stints, position limits (rarely three
  bigs, almost always a guard), and blowout rest re-planned in the 4th quarter.
- **Usage depends on who's on the floor**: each possession, plays, assists, rebounds, steals and
  blocks are split only among the five on the court. Weights are calibrated so a healthy
  rotation reproduces each player's real usage; starters' stats came next to other starters,
  so raw rates would double count that. Shares then shift with the actual lineup: a scorer's
  share rises when other high-usage players sit, and two bigs split the rebounds.
- **Efficiency depends on who's on the floor**: carrying more of the load than usual costs a
  little efficiency, and better passers around a shooter than usual raise his make and
  assisted rates (worse ones lower them).
- **Absences, checked against real games**: when you mark someone OUT, the lineup model's
  usage change for each teammate is compared with that teammate's real games without him.
  Usage is moved toward the evidence, scaled by how many games there are.
- **Each possession**: the player who ends the play is picked from the five on the floor by
  usage. The outcome (turnover, shot from one of five zones, or drawn foul) comes from that
  player's own rates.
  Shot diet tilts toward what the defense allows. Makes use the player's zone FG%, adjusted by
  the opponent's zone FG% allowed and home court. The sim also models assists, and-ones, blocks,
  steals, and offensive rebounds (which extend the possession).
- **Pace**: home pace x away pace / league pace, with game-to-game variance.
- **Per-game variance**: hot and cold shooting and usage nights, so the prop distributions are
  realistically wide.

## Live sportsbook lines

The Props tab can pull live player-prop lines and odds from FanDuel, DraftKings, Fanatics,
BetMGM, Caesars, ESPN Bet, Hard Rock, BetRivers and others through
[The Odds API](https://the-odds-api.com). Sportsbooks don't offer public APIs, and scraping
their sites breaks their terms, so this licensed aggregator is the way in.

1. Get a free key at the-odds-api.com (500 credits a month; paid plans start at $30/month for 20,000).
2. Add it to your shell profile so the app picks it up automatically:
   `export ODDS_API_KEY=your_key`
   Or paste it into the key box on the Props tab.
3. Open the 📡 **Live sportsbook lines** panel, choose books and markets, and click **Fetch live lines**.

Cost: finding the game is free. Props cost 1 credit per market per game for up to 10 books
(4 markets = 4 credits, so about 120 games a month on the free plan). Responses are cached for
10 minutes. Every line from every book is scored, and **Best line for each prop** picks the
book and side with the highest Value Score.

**Game lines, always on:** with a key set, spreads, totals and moneylines for every game load
automatically in one request (3 credits, at most once every 10 minutes while the app is in use).
Each game card shows the consensus (median across books), and the Game Sim tab's market
comparison starts at it.

**Props for all games at once:** pick prop types under **Player props to pull** at the top of the
page (the cost updates as you pick: 1 credit per prop type per game), then click **💰 Pull props
for all games**. Opening a game then loads its props with no extra credits. The same prop types
are used by the Props tab's single-game **Fetch live lines**.

## Tracking, play-type and matchup data

All from stats.nba.com, blended across seasons like the box score. Every factor has a weight
slider in the sidebar (**Factor weights**). The **🧠 Model & Data** tab lists each one's data
source, how it's used, its sample-size shrinkage, and why its default is set where it is.

- **Play types** (Synergy): each play is an isolation, P&R ball handler or roll man, spot-up, cut,
  post-up, handoff, off-screen, transition or putback. The type sets turnover and foul rates and
  the shot zones (fit to the player's real shot chart). Catch-and-shoot types use his
  catch-and-shoot 3P%, pull-up types his pull-up 3P%.
- **Opponent scheme**: each player's play-type mix shifts toward what the defense allows. Its
  points per possession allowed by type, relative to its own average, adjust efficiency.
- **Assist chains**: the chance a make was assisted depends on play type. The passer is
  weighted by the matching skill: P&R handlers on rolls, drive-and-kick passers on spot-ups,
  potential-assist leaders on cuts. Passer weights are calibrated so totals match real assists.
- **Rebounds by shot type**: long 3-point misses go to players who rebound far from the rim,
  and rim and FT misses to bigs (tracking rebound distance). Offensive rebound chance varies by
  zone, and rebound chances steady the weights. Calibrated so totals match.
- **Shot contests**: team defense blends FG% allowed with how tightly it contests shots, scaled
  by each shooter's sensitivity to tight contests. The primary defender comes from real matchup
  data, and his FG% allowed vs expected, help defenders' and on-floor rim protectors' numbers
  move make chance relative to the team's normal lineup. Rim protectors also deter rim
  attempts. Deflections raise turnovers and steals, and defender foul rates move free throws.
- **Transition**: steals, defensive rebounds and makes can trigger a fast break, at rates set by
  the offense's push rate and the defense's transition allowed. Calibrated to the league's real
  transition share.
- **Clutch**: usage in the last 5 minutes of close games uses each player's clutch usage.
- **Rest and travel**: back-to-backs, 3 games in 4 nights, long travel and altitude, from the
  schedule (pick the game date in the sidebar). Effect sizes are research-based priors.

## Limitations

- There's no injury feed. Mark players OUT in Rotations before trusting props.
- Lineups come from role templates, not each coach's actual substitution patterns. NBA.com
  doesn't publish player usage split by which teammates are on the floor, so on/off usage is
  modeled; the with/without check uses whole games a teammate missed.
- Referee data isn't available from NBA.com's stats API.
- Coaching schemes aren't published. They're inferred from play types allowed.
- Synergy play-type data fills in a few weeks into a season, so early on it leans on last season.
- Players traded or signed since last season carry their old team's usage. The sim squeezes
  shares to fit the new roster, but watch early-season numbers for new-look teams.
- Without live lines, the prop table is pre-filled with averages. Replace them with real lines.
- This is model output, not betting advice.
