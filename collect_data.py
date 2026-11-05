"""Download NBA player and team game logs (basic + advanced) from the NBA stats API.

One CSV per season and season type is written to data/players/ and data/teams/.
Seasons that are already saved are skipped, except the current season, which is
always re-downloaded so new games get picked up. Run it again any time to update.

    python collect_data.py
"""
import os
import random
import time

import pandas as pd
from nba_api.stats.endpoints import playergamelogs, teamgamelogs

FIRST_SEASON = 2015          # 2015-16
CURRENT_SEASON = 2026        # 2026-27
SEASON_TYPES = ['Regular Season', 'Playoffs']

DATA_DIR = 'data'
PLAYERS_DIR = os.path.join(DATA_DIR, 'players')
TEAMS_DIR = os.path.join(DATA_DIR, 'teams')


def season_label(year):
    """2015 -> '2015-16'"""
    return f'{year}-{str(year + 1)[-2:]}'


def fetch(endpoint, season, season_type, measure_type, retries=4):
    """Call a game log endpoint, retrying with a growing pause if the API is slow or refuses."""
    for attempt in range(1, retries + 1):
        try:
            df = endpoint(season_nullable=season, season_type_nullable=season_type,
                          measure_type_player_game_logs_nullable=measure_type,
                          timeout=90).get_data_frames()[0]
            time.sleep(random.uniform(2, 5))  # be polite between requests
            return df
        except Exception as e:
            wait = 30 * attempt
            print(f'  {measure_type} request failed ({e.__class__.__name__}), retrying in {wait}s...')
            time.sleep(wait)
    raise RuntimeError(f'Giving up on {season} {season_type} {measure_type}')


def drop_rank_columns(df):
    return df.loc[:, ~df.columns.str.endswith('_RANK')]


def merge_logs(base, advanced, keys):
    """Join the basic and advanced logs for the same games, keeping shared columns once."""
    base = drop_rank_columns(base)
    advanced = drop_rank_columns(advanced)
    extra = [c for c in advanced.columns if c not in base.columns]
    return base.merge(advanced[keys + extra], on=keys, how='left')


def collect(season, season_type):
    """Download and save one season type of player and team logs. Returns the number of player rows."""
    file_name = f"{season}_{season_type.replace(' ', '_').lower()}.csv"
    player_path = os.path.join(PLAYERS_DIR, file_name)
    team_path = os.path.join(TEAMS_DIR, file_name)

    is_current = season == season_label(CURRENT_SEASON)
    if os.path.exists(player_path) and os.path.exists(team_path) and not is_current:
        print(f'{season} {season_type}: already saved, skipping')
        return None

    print(f'{season} {season_type}: downloading...')
    player_base = fetch(playergamelogs.PlayerGameLogs, season, season_type, 'Base')
    if player_base.empty:
        print(f'{season} {season_type}: no games yet')
        return 0
    player_adv = fetch(playergamelogs.PlayerGameLogs, season, season_type, 'Advanced')
    team_base = fetch(teamgamelogs.TeamGameLogs, season, season_type, 'Base')
    team_adv = fetch(teamgamelogs.TeamGameLogs, season, season_type, 'Advanced')

    players = merge_logs(player_base, player_adv, ['PLAYER_ID', 'GAME_ID'])
    teams = merge_logs(team_base, team_adv, ['TEAM_ID', 'GAME_ID'])
    players['SEASON_TYPE'] = season_type
    teams['SEASON_TYPE'] = season_type

    players.to_csv(player_path, index=False)
    teams.to_csv(team_path, index=False)
    print(f'{season} {season_type}: {len(players):,} player rows, {len(teams):,} team rows')
    return len(players)


def main():
    os.makedirs(PLAYERS_DIR, exist_ok=True)
    os.makedirs(TEAMS_DIR, exist_ok=True)

    begin = time.time()
    for year in range(FIRST_SEASON, CURRENT_SEASON + 1):
        for season_type in SEASON_TYPES:
            collect(season_label(year), season_type)
    print(f'Done in {round((time.time() - begin) / 60, 1)} minutes')


if __name__ == '__main__':
    main()
