# NBA Player Stat Projections

Projects a player's **points, rebounds, and assists** for an upcoming game from their
recent form, advanced stats, head-to-head history against the opponent, the opponent's
recent defense, and game context (home/away, rest, playoffs).

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
| **Gradient boosting model** | **4.54** | **1.88** | **1.33** |

The model's average miss is about 3% smaller than a last-10-game average for all three
stats. Single games are noisy (a player's points can swing 10+ from night to night),
so no model gets close to zero error.

What drives the projections (permutation importance, points model): recent form and
the player's own advanced stats do almost all of the work. Game context (rest,
back-to-backs) comes next. The opponent's defense and head-to-head history add a
small amount; past games against a specific team turn out to say little beyond the
player's overall level.

## Files

- `collect_data.py`: downloads player and team game logs into `data/`. Seasons already
  saved are skipped; the current season is always refreshed.
- `features.py`: builds the model features. Every feature uses only games before the
  one being predicted.
- `player_projections.ipynb`: baselines, training, evaluation, feature importance, and
  the final models.
- `projection_models.joblib`: the saved models (one per stat).
- `project.py`: projects a player for an upcoming game.
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
python project.py "Jalen Brunson" BOS --home --date 2026-11-06
```

```
Jalen Brunson  NYK vs. BOS  (2026-11-06)
        Projected  Last 10  Season  vs. opp
PTS          24.8     28.5    24.6     22.1
REB           3.0      3.2     2.7      3.0
AST           5.9      5.2     5.3      5.6
Head-to-head games since 2015-16: 31
```

To retrain, register the kernel and run the notebook:

```bash
python -m ipykernel install --sys-prefix --name nba-projections --display-name "Python (NBA Projections)"
```

Then open `player_projections.ipynb`, pick "Python (NBA Projections)", and run all cells.

## Limitations

- Doesn't know about injuries, rest days announced in advance, or teammates being out,
  which change minutes and usage the most.
- A player who just changed teams is projected from their history with the old team.
- The NBA stats API is free but unofficial; if it changes, update `nba_api`.
