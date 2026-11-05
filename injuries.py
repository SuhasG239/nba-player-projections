"""Current NBA injury report from ESPN's free public feed (no key needed).

It reflects the latest updates, so it's meant for the next game on each team's schedule.

    python injuries.py          # print the whole report
    python injuries.py NYK BOS  # just these teams
"""
import sys

import pandas as pd
import requests

ESPN_INJURIES = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries'

# ESPN abbreviations that differ from the NBA stats API's
TO_NBA_ABBREVIATION = {'GS': 'GSW', 'NY': 'NYK', 'NO': 'NOP', 'SA': 'SAS', 'UTAH': 'UTA', 'WSH': 'WAS'}


def fetch_injury_report(timeout=30):
    """One row per listed player: name, team, status ('Out' or 'Day-To-Day'), expected return, note."""
    response = requests.get(ESPN_INJURIES, timeout=timeout)
    response.raise_for_status()
    rows = []
    for team in response.json().get('injuries', []):
        for item in team.get('injuries', []):
            athlete = item.get('athlete', {})
            abbreviation = athlete.get('team', {}).get('abbreviation', '')
            rows.append({
                'PLAYER_NAME': athlete.get('displayName'),
                'TEAM': TO_NBA_ABBREVIATION.get(abbreviation, abbreviation),
                'STATUS': item.get('status'),
                'RETURN_DATE': item.get('details', {}).get('returnDate'),
                'UPDATED': item.get('date'),
                'NOTE': item.get('shortComment'),
            })
    return pd.DataFrame(rows, columns=['PLAYER_NAME', 'TEAM', 'STATUS', 'RETURN_DATE', 'UPDATED', 'NOTE'])


if __name__ == '__main__':
    report = fetch_injury_report()
    teams = [t.upper() for t in sys.argv[1:]]
    if teams:
        report = report[report['TEAM'].isin(teams)]
    print(report[['TEAM', 'PLAYER_NAME', 'STATUS', 'RETURN_DATE']].sort_values(['TEAM', 'STATUS']).to_string(index=False))
