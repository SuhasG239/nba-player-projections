"""Project a player's points, rebounds, and assists for an upcoming game.

Uses the models saved by player_projections.ipynb and the data saved by collect_data.py.
Run collect_data.py first to pick up the latest games.

    python project.py "Luka Doncic" BOS --home
    python project.py "Nikola Jokic" MIN --date 2026-11-08
"""
import argparse

import joblib

from features import load_data, upcoming_features


def main():
    parser = argparse.ArgumentParser(description='Project a player\'s stats for an upcoming game.')
    parser.add_argument('player', help='Player name, e.g. "Luka Doncic" (accents optional)')
    parser.add_argument('opponent', help='Opponent team abbreviation, e.g. BOS')
    parser.add_argument('--home', action='store_true', help='The player\'s team is at home (default: away)')
    parser.add_argument('--date', help='Game date as YYYY-MM-DD (default: today)')
    parser.add_argument('--playoffs', action='store_true', help='The game is a playoff game')
    args = parser.parse_args()

    saved = joblib.load('projection_models.joblib')
    players, teams = load_data()
    row = upcoming_features(players, teams, args.player, args.opponent, args.home, args.date, args.playoffs)
    features = row[saved['features']]

    print(f"{row['PLAYER_NAME'].iloc[0]}  {row['MATCHUP'].iloc[0]}  ({row['GAME_DATE'].iloc[0]:%Y-%m-%d})")
    print(f"{'':6}{'Projected':>11}{'Last 10':>9}{'Season':>8}{'vs. opp':>9}")
    for stat, model in saved['models'].items():
        projected = model.predict(features)[0]
        h2h = row[f'{stat}_H2H'].iloc[0]
        print(f"{stat:6}{projected:11.1f}{row[f'{stat}_L10'].iloc[0]:9.1f}"
              f"{row[f'{stat}_SEASON'].iloc[0]:8.1f}{h2h:9.1f}")
    print(f"Head-to-head games since 2015-16: {int(row['H2H_GP'].iloc[0])}")


if __name__ == '__main__':
    main()
