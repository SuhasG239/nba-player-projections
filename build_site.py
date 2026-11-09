"""Build the data for the projections website (docs/data/projections.json).

Refreshes this season's game logs, gets the next few days of games from ESPN's schedule
and the current injury report, and projects every rotation player in those games.
Then commit and push, and GitHub Pages serves the updated site.

    python build_site.py              # refresh data, next 3 days of games
    python build_site.py --days 5
    python build_site.py --skip-collect   # use the game logs already saved
"""
import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import requests

import collect_data
from features import TARGETS, history_table, load_data, name_key, upcoming_batch
from injuries import TO_NBA_ABBREVIATION, fetch_injury_report

ESPN_SCOREBOARD = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'
OUTPUT = os.path.join('docs', 'data', 'projections.json')


def game_status(espn_status):
    """'scheduled', 'live' or 'final'; None for postponed or canceled games."""
    if espn_status == 'STATUS_SCHEDULED':
        return 'scheduled'
    if espn_status == 'STATUS_FINAL':
        return 'final'
    if espn_status in ('STATUS_POSTPONED', 'STATUS_CANCELED', 'STATUS_SUSPENDED', 'STATUS_DELAYED'):
        return None
    return 'live'   # in progress, halftime, end of period


def fetch_games(day):
    """All games on one date (US Eastern), with their status."""
    response = requests.get(ESPN_SCOREBOARD, params={'dates': day.strftime('%Y%m%d')}, timeout=30)
    response.raise_for_status()
    games = []
    for event in response.json().get('events', []):
        status = game_status(event['status']['type']['name'])
        if status is None:
            continue
        competition = event['competitions'][0]
        side = {}
        for team in competition['competitors']:
            abbreviation = team['team']['abbreviation']
            side[team['homeAway']] = {'abbr': TO_NBA_ABBREVIATION.get(abbreviation, abbreviation),
                                      'name': team['team']['displayName'],
                                      'short': team['team'].get('shortDisplayName') or team['team']['displayName']}
        games.append({'id': event['id'], 'date': day.isoformat(), 'tipoff_utc': event['date'],
                      'status': status, 'home': side['home'], 'away': side['away']})
    return games


def rounded(value, digits=1):
    return None if value is None or pd.isna(value) else round(float(value), digits)


def recent_games(players, player_id, n=10):
    history = players[players['PLAYER_ID'] == player_id].sort_values('GAME_DATE').tail(n)
    return [{'date': r.GAME_DATE.strftime('%Y-%m-%d'), 'opp': r.MATCHUP.split()[-1],
             'home': 'vs.' in r.MATCHUP, 'min': rounded(r.MIN, 0),
             'PTS': int(r.PTS), 'REB': int(r.REB), 'AST': int(r.AST)} for r in history.itertuples()]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--days', type=int, default=3, help='How many days of games to project (default 3)')
    parser.add_argument('--skip-collect', action='store_true', help='Don\'t download new game logs first')
    args = parser.parse_args()

    if not args.skip_collect:
        print('Refreshing game logs...')
        os.makedirs(collect_data.PLAYERS_DIR, exist_ok=True)
        os.makedirs(collect_data.TEAMS_DIR, exist_ok=True)
        for season_type in collect_data.SEASON_TYPES:
            collect_data.collect(collect_data.season_label(collect_data.CURRENT_SEASON), season_type)

    saved = joblib.load('projection_models.joblib')
    players, teams = load_data()
    team_ids = teams.drop_duplicates('TEAM_ABBREVIATION', keep='last').set_index('TEAM_ABBREVIATION')['TEAM_ID']

    report = fetch_injury_report()
    statuses = {name_key(r.PLAYER_NAME): {'status': r.STATUS, 'note': r.NOTE} for r in report.itertuples()}
    out_names = report.loc[report['STATUS'] == 'Out', 'PLAYER_NAME'].tolist()

    print('Building player history...')
    history = history_table(players, teams)

    site_games = []
    today = date.today()
    for offset in range(args.days):
        day = today + timedelta(days=offset)
        games = [g for g in fetch_games(day) if g['home']['abbr'] in team_ids and g['away']['abbr'] in team_ids]
        if offset > 0:
            games = [g for g in games if g['status'] == 'scheduled']
        if not games:
            continue

        # Project from games before this date only, so a game that has already finished
        # (and is in the logs) doesn't leak its own result into its projection
        before = pd.Timestamp(day)
        day_players, day_teams, day_history = players, teams, history
        if (players['GAME_DATE'] >= before).any():
            day_players = players[players['GAME_DATE'] < before]
            day_teams = teams[teams['GAME_DATE'] < before]
            day_history = history[history['GAME_DATE'] < before]
        actual = players[players['GAME_DATE'] == before].set_index('PLAYER_ID')
        table = pd.DataFrame({
            'GAME_ID': [g['id'] for g in games],
            'GAME_DATE': pd.Timestamp(day),
            'HOME_TEAM_ID': [team_ids[g['home']['abbr']] for g in games],
            'AWAY_TEAM_ID': [team_ids[g['away']['abbr']] for g in games],
        })
        rows, rotations = upcoming_batch(day_players, day_teams, day_history, table, out_names)
        for t in TARGETS:
            rows[f'{t}_PROJ'] = saved['models'][t].predict(rows[saved['features']])

        for game in games:
            entry = dict(game, teams={})
            for side in ('away', 'home'):
                abbr = game[side]['abbr']
                team_rows = rows[(rows['GAME_ID'] == game['id']) & (rows['TEAM_ABBREVIATION'] == abbr)]
                team_rot = rotations[(rotations['GAME_ID'] == game['id']) & (rotations['TEAM_ABBREVIATION'] == abbr)]
                listed = []
                for r in team_rows.sort_values('PTS_PROJ', ascending=False).itertuples():
                    info = statuses.get(name_key(r.PLAYER_NAME), {})
                    # Box score for finished games: the projection next to what happened
                    result = None
                    if game['status'] == 'final' and r.PLAYER_ID in actual.index:
                        box = actual.loc[[r.PLAYER_ID]].iloc[0]
                        result = {t: int(box[t]) for t in TARGETS} | {'min': rounded(box['MIN'], 0)}
                    elif game['status'] == 'final':
                        result = {'dnp': True}
                    listed.append({
                        'actual': result,
                        'id': int(r.PLAYER_ID), 'name': r.PLAYER_NAME,
                        'proj': {t: rounded(getattr(r, f'{t}_PROJ')) for t in TARGETS},
                        'last10': {t: rounded(getattr(r, f'{t}_L10')) for t in TARGETS},
                        'season': {t: rounded(getattr(r, f'{t}_SEASON')) for t in TARGETS},
                        'vs_opp': {t: rounded(getattr(r, f'{t}_H2H')) for t in TARGETS},
                        'vs_opp_games': int(r.H2H_GP),
                        'min_last10': rounded(r.MIN_L10),
                        'rest_days': rounded(r.REST_DAYS, 0),
                        'status': info.get('status'), 'note': info.get('note'),
                        'recent': recent_games(day_players, r.PLAYER_ID),
                    })
                out = [{'id': int(r.PLAYER_ID), 'name': r.PLAYER_NAME, 'min': rounded(r.MIN),
                        'note': statuses.get(name_key(r.PLAYER_NAME), {}).get('note')}
                       for r in team_rot[team_rot['OUT']].sort_values('MIN', ascending=False).itertuples()]
                entry['teams'][abbr] = {'players': listed, 'out': out}
            site_games.append(entry)
        print(f'{day}: {len(games)} games, {len(rows)} players projected')

    data = {
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='minutes'),
        'data_through': players['GAME_DATE'].max().strftime('%Y-%m-%d'),
        'season': players['SEASON_YEAR'].max(),
        'typical_miss': saved.get('test_mae', {'PTS': 4.45, 'REB': 1.85, 'AST': 1.32}),
        'games': site_games,
    }
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
    print(f'Wrote {OUTPUT}: {len(site_games)} games, {os.path.getsize(OUTPUT) / 1024:.0f} KB')


if __name__ == '__main__':
    main()
