"""Build model features from the game logs saved by collect_data.py.

Every feature for a game uses only games played *before* it, so the model never sees
the result it is trying to predict.
"""
import glob
import os
import unicodedata

import numpy as np
import pandas as pd

TARGETS = ['PTS', 'REB', 'AST']

# Player stats averaged over recent games
FORM_STATS = ['PTS', 'REB', 'AST', 'MIN', 'FGA', 'FG3A', 'FTA', 'OREB', 'DREB', 'TOV',
              'USG_PCT', 'AST_PCT', 'REB_PCT', 'TS_PCT', 'PIE', 'POSS']
WINDOWS = [5, 10]

# Team stats averaged over each team's last 10 games
TEAM_STATS = ['PACE', 'OFF_RATING', 'DEF_RATING', 'PTS_ALLOWED', 'REB_ALLOWED', 'AST_ALLOWED',
              'FG3A_ALLOWED']
TEAM_WINDOW = 10


def load_data(data_dir='data'):
    """Read every saved season into one player table and one team table."""
    def read(folder):
        files = sorted(glob.glob(os.path.join(data_dir, folder, '*.csv')))
        df = pd.concat([pd.read_csv(f, dtype={'GAME_ID': str}) for f in files], ignore_index=True)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        return df
    return read('players'), read('teams')


def team_game_stats(teams):
    """One row per team per game, with the opponent's id and what the team allowed."""
    cols = ['TEAM_ID', 'GAME_ID', 'GAME_DATE', 'PTS', 'REB', 'AST', 'FG3A', 'PACE', 'OFF_RATING', 'DEF_RATING']
    t = teams[cols]
    # Join each team to the other team in the same game
    opp = t[['TEAM_ID', 'GAME_ID', 'PTS', 'REB', 'AST', 'FG3A']].rename(columns={
        'TEAM_ID': 'OPP_TEAM_ID', 'PTS': 'PTS_ALLOWED', 'REB': 'REB_ALLOWED',
        'AST': 'AST_ALLOWED', 'FG3A': 'FG3A_ALLOWED'})
    t = t.merge(opp, on='GAME_ID')
    return t[t['TEAM_ID'] != t['OPP_TEAM_ID']].reset_index(drop=True)


def team_form(team_stats):
    """Each team's average over its last TEAM_WINDOW games, as of the end of each game."""
    t = team_stats.sort_values(['TEAM_ID', 'GAME_DATE'])
    form = (t.groupby('TEAM_ID')[TEAM_STATS]
             .rolling(TEAM_WINDOW, min_periods=3).mean()
             .reset_index(level=0, drop=True))
    form = pd.concat([t[['TEAM_ID', 'GAME_DATE']], form], axis=1)
    return form.sort_values('GAME_DATE').reset_index(drop=True)


def attach_team_form(df, form, team_col, prefix):
    """Add a team's form going into each game (from its games strictly before that date)."""
    renamed = form.rename(columns={'TEAM_ID': team_col, **{c: prefix + c for c in TEAM_STATS}})
    df = df.sort_values('GAME_DATE')
    df = pd.merge_asof(df, renamed, on='GAME_DATE', by=team_col, allow_exact_matches=False)
    return df


def add_player_features(df):
    """Recent form, season averages, rest, and head-to-head history for each player-game."""
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE', 'GAME_ID']).reset_index(drop=True)
    by_player = df.groupby('PLAYER_ID')

    # Stats from the player's previous games only
    prev = by_player[FORM_STATS].shift(1)
    prev_by_player = prev.groupby(df['PLAYER_ID'])
    for w in WINDOWS:
        rolled = prev_by_player.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True).sort_index()
        df[[f'{c}_L{w}' for c in FORM_STATS]] = rolled.values

    # Longer-term level that weights recent games more
    ewm = prev_by_player.ewm(span=30, min_periods=1).mean().reset_index(level=0, drop=True).sort_index()
    for c in TARGETS + ['MIN']:
        df[f'{c}_EWM30'] = ewm[c].values

    # Season-to-date averages (before this game)
    by_season = df.groupby(['PLAYER_ID', 'SEASON_YEAR'])
    games_before = by_season.cumcount()
    df['SEASON_GP'] = games_before
    for c in TARGETS + ['MIN']:
        # fillna(0) so an upcoming game (no result yet) still gets a running total
        filled = df[c].fillna(0)
        season_sum = filled.groupby([df['PLAYER_ID'], df['SEASON_YEAR']]).cumsum() - filled
        df[f'{c}_SEASON'] = season_sum / games_before.replace(0, np.nan)

    df['CAREER_GP'] = by_player.cumcount()

    # Rest
    df['REST_DAYS'] = by_player['GAME_DATE'].diff().dt.days.clip(upper=10)
    df['BACK_TO_BACK'] = (df['REST_DAYS'] == 1).astype(int)

    # Head-to-head: the player's averages in earlier games against this opponent
    by_matchup = df.groupby(['PLAYER_ID', 'OPP_TEAM_ID'])
    h2h_games = by_matchup.cumcount()
    df['H2H_GP'] = h2h_games
    for c in TARGETS + ['MIN']:
        filled = df[c].fillna(0)
        h2h_sum = filled.groupby([df['PLAYER_ID'], df['OPP_TEAM_ID']]).cumsum() - filled
        df[f'{c}_H2H'] = h2h_sum / h2h_games.replace(0, np.nan)
    # How the player does against this opponent compared to their usual level
    for c in TARGETS:
        df[f'{c}_H2H_DIFF'] = df[f'{c}_H2H'] - df[f'{c}_EWM30']

    return df


def prepare_players(players, team_stats):
    """Add home/away, playoff flag, and the opponent's team id to the player logs."""
    df = players.copy()
    df['HOME'] = df['MATCHUP'].str.contains('vs.', regex=False).astype(int)
    df['PLAYOFFS'] = (df['SEASON_TYPE'] == 'Playoffs').astype(int)
    opp = team_stats[['TEAM_ID', 'GAME_ID', 'OPP_TEAM_ID']]
    return df.merge(opp, on=['TEAM_ID', 'GAME_ID'], how='left')


def build_features(players, teams):
    """Full feature table: one row per player-game, ready for training."""
    stats = team_game_stats(teams)
    form = team_form(stats)
    df = add_player_features(prepare_players(players, stats))
    df = attach_team_form(df, form, 'OPP_TEAM_ID', 'OPP_')
    df = attach_team_form(df, form, 'TEAM_ID', 'TEAM_')
    return df.sort_values(['GAME_DATE', 'GAME_ID', 'PLAYER_ID']).reset_index(drop=True)


FEATURES = (
    [f'{c}_L{w}' for w in WINDOWS for c in FORM_STATS]
    + [f'{c}_EWM30' for c in TARGETS + ['MIN']]
    + [f'{c}_SEASON' for c in TARGETS + ['MIN']]
    + ['SEASON_GP', 'CAREER_GP', 'REST_DAYS', 'BACK_TO_BACK', 'HOME', 'PLAYOFFS', 'H2H_GP']
    + [f'{c}_H2H' for c in TARGETS + ['MIN']]
    + [f'{c}_H2H_DIFF' for c in TARGETS]
    + ['OPP_' + c for c in TEAM_STATS]
    + ['TEAM_' + c for c in TEAM_STATS]
)


def _simplify(name):
    """Lowercase and strip accents so 'Luka Doncic' matches 'Luka Dončić'."""
    return unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower().strip()


def find_player(players, name):
    """Look up a player's id by name (accents and case don't matter)."""
    names = players.drop_duplicates('PLAYER_ID', keep='last')[['PLAYER_ID', 'PLAYER_NAME']]
    matches = names[names['PLAYER_NAME'].map(_simplify) == _simplify(name)]
    if matches.empty:
        matches = names[names['PLAYER_NAME'].map(_simplify).str.contains(_simplify(name), regex=False)]
    if len(matches) != 1:
        found = ', '.join(matches['PLAYER_NAME'].head(5)) or 'none'
        raise ValueError(f'Expected one player matching "{name}", found: {found}')
    return matches.iloc[0]


def upcoming_features(players, teams, player_name, opponent, home, game_date=None, playoffs=False):
    """Features for a game that hasn't been played yet, built from all games before it."""
    player = find_player(players, player_name)
    history = players[players['PLAYER_ID'] == player['PLAYER_ID']].sort_values('GAME_DATE')
    last = history.iloc[-1]

    team_ids = teams.drop_duplicates('TEAM_ABBREVIATION', keep='last').set_index('TEAM_ABBREVIATION')['TEAM_ID']
    opponent = opponent.upper()
    if opponent not in team_ids:
        raise ValueError(f'Unknown team "{opponent}". Use an abbreviation like {", ".join(team_ids.index[:5])}')

    game_date = pd.Timestamp(game_date) if game_date else pd.Timestamp.today().normalize()
    season_year = last['SEASON_YEAR']  # assumes the game is in the player's latest season

    upcoming = {c: np.nan for c in history.columns}
    upcoming.update({
        'SEASON_YEAR': season_year, 'PLAYER_ID': player['PLAYER_ID'], 'PLAYER_NAME': player['PLAYER_NAME'],
        'TEAM_ID': last['TEAM_ID'], 'TEAM_ABBREVIATION': last['TEAM_ABBREVIATION'], 'GAME_ID': 'UPCOMING',
        'GAME_DATE': game_date, 'MATCHUP': f"{last['TEAM_ABBREVIATION']} {'vs.' if home else '@'} {opponent}",
        'SEASON_TYPE': 'Playoffs' if playoffs else 'Regular Season',
    })

    stats = team_game_stats(teams)
    form = team_form(stats)
    df = prepare_players(history, stats)
    upcoming = pd.DataFrame([upcoming])
    upcoming['HOME'] = int(home)
    upcoming['PLAYOFFS'] = int(playoffs)
    upcoming['OPP_TEAM_ID'] = team_ids[opponent]
    df = add_player_features(pd.concat([df, upcoming], ignore_index=True))
    row = df[df['GAME_ID'] == 'UPCOMING']
    row = attach_team_form(row, form, 'OPP_TEAM_ID', 'OPP_')
    row = attach_team_form(row, form, 'TEAM_ID', 'TEAM_')
    return row
