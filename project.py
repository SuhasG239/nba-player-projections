"""Project a player's points, rebounds, and assists for an upcoming game.

Uses the models saved by player_projections.ipynb and the data saved by collect_data.py.
Run collect_data.py first to pick up the latest games.

Players listed "Out" on the current ESPN injury report are treated as out for both teams.
"Day-To-Day" players are treated as playing unless you pass --gtd-out. Override any player
with --out or --playing.

    python project.py "Luka Doncic" BOS --home
    python project.py "Nikola Jokic" MIN --date 2026-11-08 --out "Jamal Murray"
    python project.py "Jalen Brunson" BOS --home --no-report --out "Josh Hart"
"""
import argparse

import joblib

from features import load_data, name_key, upcoming_features
from injuries import fetch_injury_report


def players_out(args):
    """Names of players who won't play, from the injury report and the command line."""
    out, day_to_day = set(), []
    if not args.no_report:
        try:
            report = fetch_injury_report()
        except Exception as e:
            print(f'Could not get the injury report ({e.__class__.__name__}); using only --out.')
        else:
            out.update(report.loc[report['STATUS'] == 'Out', 'PLAYER_NAME'])
            gtd = report.loc[report['STATUS'] == 'Day-To-Day', 'PLAYER_NAME']
            if args.gtd_out:
                out.update(gtd)
            else:
                day_to_day = list(gtd)
    out.update(args.out)
    playing = {name_key(n) for n in args.playing}
    out = {n for n in out if name_key(n) not in playing}
    return out, day_to_day


def main():
    parser = argparse.ArgumentParser(description='Project a player\'s stats for an upcoming game.')
    parser.add_argument('player', help='Player name, e.g. "Luka Doncic" (accents optional)')
    parser.add_argument('opponent', help='Opponent team abbreviation, e.g. BOS')
    parser.add_argument('--home', action='store_true', help='The player\'s team is at home (default: away)')
    parser.add_argument('--date', help='Game date as YYYY-MM-DD (default: today)')
    parser.add_argument('--playoffs', action='store_true', help='The game is a playoff game')
    parser.add_argument('--out', nargs='+', default=[], metavar='NAME', help='Players who won\'t play (either team)')
    parser.add_argument('--playing', nargs='+', default=[], metavar='NAME',
                        help='Players who will play even if the injury report lists them')
    parser.add_argument('--gtd-out', action='store_true', help='Treat Day-To-Day players as out')
    parser.add_argument('--no-report', action='store_true', help='Don\'t download the injury report')
    args = parser.parse_args()

    out_names, day_to_day = players_out(args)
    saved = joblib.load('projection_models.joblib')
    players, teams = load_data()
    row, rotations = upcoming_features(players, teams, args.player, args.opponent, args.home,
                                       args.date, args.playoffs, out_names)
    features = row[saved['features']]
    name = row['PLAYER_NAME'].iloc[0]

    print(f"{name}  {row['MATCHUP'].iloc[0]}  ({row['GAME_DATE'].iloc[0]:%Y-%m-%d})")
    if name_key(name) in {name_key(n) for n in out_names}:
        print(f'Note: {name} is listed as out for this game.')
    print(f"{'':6}{'Projected':>11}{'Last 10':>9}{'Season':>8}{'vs. opp':>9}")
    for stat, model in saved['models'].items():
        projected = model.predict(features)[0]
        print(f"{stat:6}{projected:11.1f}{row[f'{stat}_L10'].iloc[0]:9.1f}"
              f"{row[f'{stat}_SEASON'].iloc[0]:8.1f}{row[f'{stat}_H2H'].iloc[0]:9.1f}")
    print(f"Head-to-head games since 2015-16: {int(row['H2H_GP'].iloc[0])}")

    # Who the projection assumes is out, and who is uncertain
    gtd_keys = {name_key(n) for n in day_to_day}
    for team, group in rotations.groupby('TEAM_ABBREVIATION', sort=False):
        missing = group[group['OUT']]
        listed = ', '.join(f"{r.PLAYER_NAME} ({r.MIN:.0f} min)" for r in missing.itertuples()) or 'nobody'
        print(f'{team} out: {listed}')
        uncertain = group[~group['OUT'] & group['PLAYER_NAME'].map(name_key).isin(gtd_keys)]
        if not uncertain.empty:
            print(f"{team} day-to-day (counted as playing): {', '.join(uncertain['PLAYER_NAME'])}")


if __name__ == '__main__':
    main()
