# NBA Player Stat Projections

Projects a player's **points, rebounds, and assists** for an upcoming game from their
recent form, advanced stats, head-to-head history against the opponent, the opponent's
recent defense, game context (home/away, rest, playoffs), and which players are out
for either team (injuries, rest).

**Live site: https://suhasg239.github.io/nba-player-projections/**. Pick an upcoming game to see
a projection for every rotation player on both teams, with injured players listed. Tap a player for
recent games, averages, and the context behind the number.

Data comes free from the NBA stats API via [`nba_api`](https://github.com/swar/nba_api):
every player and team game log, basic and advanced, from 2015-16 to now
(about 303,000 player-games).

## Results

Trained on 2015-16 through 2024-25 and tested on the full 2025-26 season (28,062
player-games it never saw). Mean absolute error, lower is better:

| Method | PTS | REB | AST |
|---|---|---|---|
| Season average | 4.71 | 1.93 | 1.37 |
| Last 10 games average | 4.68 | 1.94 | 1.37 |
| Model without players out | 4.54 | 1.88 | 1.33 |
| **Gradient boosting model** | **4.45** | **1.85** | **1.32** |

The model's average miss is about 5% smaller than a last-10-game average for points
and 4% for rebounds and assists. Single games are noisy (a player's points can swing 10+ from night to night),
so no model gets close to zero error.

What drives the projections (permutation importance, points model): recent form and
the player's own advanced stats do almost all of the work. Players out comes next: when
teammates who usually score 20+ more points than in the player's last 10 games are out,
he scores about 2 more points than his last-10 average. Game context (rest,
back-to-backs) follows. The opponent's defense and head-to-head history add a
small amount; past games against a specific team turn out to say little beyond the
player's overall level.

## Players out

The NBA game logs only list players who played. A player counts as out for a game when
he played for the team in at least 2 of its previous 10 games that season (and not for
another team since) but isn't in the box score. Each missing player adds his recent
minutes, points, rebounds, assists and shots to his team's "out" totals. Those totals
feed the model for both the player's team and the opponent.

For upcoming games, `project.py` reads the current
[ESPN injury report](https://www.espn.com/nba/injuries) (free, no key). Players listed
**Out** are treated as out. **Day-To-Day** players are treated as playing unless you
pass `--gtd-out`. The report reflects the latest news, so it's accurate for each team's
next game, not games further out.

## Files

- `collect_data.py`: downloads player and team game logs into `data/`. Seasons already
  saved are skipped; the current season is always refreshed.
- `features.py`: builds the model features. Every feature uses only games before the
  one being predicted.
- `player_projections.ipynb`: baselines, training, evaluation, feature importance, and
  the final models.
- `projection_models.joblib`: the saved models (one per stat).
- `build_site.py`: refreshes the game logs and writes the website's data (`docs/data/projections.json`)
  for the next few days of games.
- `docs/`: the website (plain HTML, CSS and JavaScript), served by GitHub Pages.
- `injuries.py`: downloads the current ESPN injury report (`python injuries.py NYK BOS`).
- `project.py`: projects a player for an upcoming game, using the injury report.
- `NBA data scraping.ipynb`: scoring leaders 2017-2024 (`NBA_points_leaders_2017-2024.xlsx`).
- `get_data.ipynb`: an earlier basketball-reference scraper, replaced by `collect_data.py`.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python collect_data.py        # about 10 minutes the first time
```

Project a game (team abbreviations like `BOS`, `LAL`, `NYK`; `--home` if the player's
team is at home, `--date` defaults to today):

```bash
python project.py "Nikola Jokic" MIN
```

```
Nikola Jokić  DEN @ MIN  (2026-11-05)
        Projected  Last 10  Season  vs. opp
PTS          22.4     23.8    22.0     25.3
REB          11.0     10.9     9.7     11.2
AST           8.5      9.2     9.0      7.7
Head-to-head games since 2015-16: 60
DEN out: nobody
DEN day-to-day (counted as playing): DeMar DeRozan
MIN out: Anthony Edwards (34 min)
MIN day-to-day (counted as playing): Terrence Shannon Jr
```

Override the report for any player on either team:

```bash
python project.py "Nikola Jokic" MIN --out "Jamal Murray"          # Jokić projects 24.2 PTS instead of 22.4
python project.py "Jalen Brunson" BOS --home --gtd-out --playing "Josh Hart"
python project.py "Jalen Brunson" BOS --home --no-report            # ignore the injury report
```

## Update the website

The NBA stats API blocks cloud servers, so the site's data is built on your own computer:

```bash
python build_site.py          # refreshes game logs, projects the next 3 days of games
git add docs/data
git commit -m "Update projections"
git push
```

GitHub Pages republishes within a minute or two. Run it on game days, ideally close to the first
tipoff so the injury report is current. To preview locally: `python -m http.server --directory docs`.

## Retrain

To retrain, register the kernel and run the notebook:

```bash
python -m ipykernel install --sys-prefix --name nba-projections --display-name "Python (NBA Projections)"
```

Then open `player_projections.ipynb`, pick "Python (NBA Projections)", and run all cells.

## Limitations

- Players out are inferred from box scores, so the model can't tell an injury from a
  coach's decision, and a player who's been out more than 10 games no longer counts.
- The injury report is only as current as ESPN's feed; check it close to tip-off.
  Late scratches and minutes limits aren't captured.
- A player who just changed teams is projected from their history with the old team.
- The NBA stats API is free but unofficial; if it changes, update `nba_api`.
