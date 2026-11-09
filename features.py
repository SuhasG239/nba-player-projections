"""Build model features from the game logs saved by collect_data.py.

Every feature for a game uses only games played *before* it, so the model never sees
the result it is trying to predict.
"""
import glob
import os
import re
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

# Players out. The game logs only list players who played, so a player counts as out
# when he was in the team's rotation (played in at least MIN_APPEARANCES of the team's
# previous ROTATION_WINDOW games this season, and not for another team since) but
# didn't play. Each missing player adds his averages from those games.
ROTATION_WINDOW = 10
MIN_APPEARANCES = 2
OUT_STATS = ['MIN', 'PTS', 'REB', 'AST', 'FGA']
OUT_COLS = ['OUT_' + c for c in OUT_STATS] + ['OUT_ROTATION']   # OUT_ROTATION: players out averaging 20+ min
OUT_RECENT = ['TEAM_OUT_MIN', 'TEAM_OUT_PTS', 'TEAM_OUT_AST', 'TEAM_OUT_FGA']


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


def team_schedule(teams):
    """Each team's games numbered in order within a season (regular season and playoffs together)."""
    s = teams[['TEAM_ID', 'TEAM_ABBREVIATION', 'SEASON_YEAR', 'GAME_ID', 'GAME_DATE']]
    s = s.sort_values(['TEAM_ID', 'GAME_DATE', 'GAME_ID']).reset_index(drop=True)
    s['TEAM_GAME_NO'] = s.groupby(['TEAM_ID', 'SEASON_YEAR']).cumcount()
    return s


def rotation_players(players, schedule, targets):
    """The rotation going into each target game, with each player's averages over his recent games.

    targets has TEAM_ID, SEASON_YEAR, TEAM_GAME_NO, GAME_ID, GAME_DATE (it can include a
    game that hasn't been played yet). Only games before each target are used.
    """
    apps = players[['PLAYER_ID', 'TEAM_ID', 'GAME_ID'] + OUT_STATS].merge(
        schedule[['TEAM_ID', 'GAME_ID', 'SEASON_YEAR', 'TEAM_GAME_NO']], on=['TEAM_ID', 'GAME_ID'])
    apps = apps[apps['TEAM_ID'].isin(targets['TEAM_ID'].unique())].drop(columns='GAME_ID')
    # Each appearance counts toward the team's next ROTATION_WINDOW games
    expanded = pd.concat([apps.assign(TARGET_NO=apps['TEAM_GAME_NO'] + d)
                          for d in range(1, ROTATION_WINDOW + 1)], ignore_index=True)
    t = targets[['TEAM_ID', 'SEASON_YEAR', 'TEAM_GAME_NO', 'GAME_ID', 'GAME_DATE']].rename(
        columns={'TEAM_GAME_NO': 'TARGET_NO'})
    expanded = expanded.merge(t, on=['TEAM_ID', 'SEASON_YEAR', 'TARGET_NO'])

    keys = ['TEAM_ID', 'GAME_ID', 'GAME_DATE', 'PLAYER_ID']
    rot = expanded.groupby(keys)[OUT_STATS].mean()
    rot['APPEARANCES'] = expanded.groupby(keys).size()
    rot = rot[rot['APPEARANCES'] >= MIN_APPEARANCES].reset_index()

    # Drop players who have since played for another team (trades, signings)
    last_team = (players[['PLAYER_ID', 'GAME_DATE', 'TEAM_ID']]
                 .rename(columns={'TEAM_ID': 'LAST_TEAM_ID'}).sort_values('GAME_DATE'))
    rot = pd.merge_asof(rot.sort_values('GAME_DATE'), last_team, on='GAME_DATE', by='PLAYER_ID',
                        allow_exact_matches=False)
    rot = rot[rot['LAST_TEAM_ID'] == rot['TEAM_ID']].drop(columns='LAST_TEAM_ID')

    names = players.drop_duplicates('PLAYER_ID', keep='last').set_index('PLAYER_ID')['PLAYER_NAME']
    rot['PLAYER_NAME'] = rot['PLAYER_ID'].map(names)
    return rot.reset_index(drop=True)


def summarize_out(out, targets):
    """Add up the players out for each target game (zero when nobody is out)."""
    out = out.assign(ROTATION=(out['MIN'] >= 20).astype(int))
    totals = out.groupby(['TEAM_ID', 'GAME_ID'])[OUT_STATS + ['ROTATION']].sum()
    totals.columns = OUT_COLS
    result = targets[['TEAM_ID', 'GAME_ID']].merge(totals.reset_index(), on=['TEAM_ID', 'GAME_ID'], how='left')
    result[OUT_COLS] = result[OUT_COLS].fillna(0)
    return result


def absences(players, schedule, targets=None):
    """For each played game: totals for the rotation players who didn't play."""
    targets = schedule if targets is None else targets
    rot = rotation_players(players, schedule, targets)
    # A player who played that day (for any team) wasn't out
    played = players[['PLAYER_ID', 'GAME_DATE']].drop_duplicates().assign(PLAYED=True)
    rot = rot.merge(played, on=['PLAYER_ID', 'GAME_DATE'], how='left')
    return summarize_out(rot[rot['PLAYED'].isna()], targets)


def attach_absences(df, out):
    """Add the player's own team's and the opponent's players-out totals."""
    team = out.rename(columns={c: 'TEAM_' + c for c in OUT_COLS})
    opp = out.rename(columns={'TEAM_ID': 'OPP_TEAM_ID', **{c: 'OPP_' + c for c in OUT_COLS}})
    df = df.merge(team, on=['TEAM_ID', 'GAME_ID'], how='left')
    return df.merge(opp, on=['OPP_TEAM_ID', 'GAME_ID'], how='left')


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

    # Teammates out tonight compared to how many were out over the player's last 10 games.
    # His recent averages already reflect those earlier absences, so the change is what matters.
    prev_out = by_player[OUT_RECENT].shift(1).groupby(df['PLAYER_ID'])
    recent_out = prev_out.rolling(10, min_periods=1).mean().reset_index(level=0, drop=True).sort_index()
    for c in OUT_RECENT:
        df[f'{c}_DIFF'] = df[c] - recent_out[c].values

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
    out = absences(players, team_schedule(teams))
    df = attach_absences(prepare_players(players, stats), out)
    df = add_player_features(df)
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
    + ['TEAM_' + c for c in OUT_COLS] + [f'{c}_DIFF' for c in OUT_RECENT]
    + ['OPP_' + c for c in OUT_COLS]
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


def history_table(players, teams):
    """Every past player-game with players-out totals attached (the input to upcoming_batch)."""
    stats = team_game_stats(teams)
    return attach_absences(prepare_players(players, stats), absences(players, team_schedule(teams)))


def upcoming_batch(players, teams, history, games, out_names=()):
    """Feature rows for every rotation player in a set of unplayed games, in one pass.

    games: GAME_ID, GAME_DATE, HOME_TEAM_ID, AWAY_TEAM_ID (all on one date, each team once).
    history: the result of history_table(). out_names: players who won't play.
    Returns (rows, rotations): one feature row per available player, and every rotation
    player with an OUT flag.
    """
    games = games.assign(GAME_DATE=pd.to_datetime(games['GAME_DATE']).astype(players['GAME_DATE'].dtype))
    form = team_form(team_game_stats(teams))
    schedule = team_schedule(teams)
    latest = schedule.groupby('TEAM_ID')['SEASON_YEAR'].max()
    current = schedule[schedule['SEASON_YEAR'] == schedule['TEAM_ID'].map(latest)]
    next_no = current.groupby('TEAM_ID')['TEAM_GAME_NO'].max() + 1
    abbreviations = schedule.drop_duplicates('TEAM_ID', keep='last').set_index('TEAM_ID')['TEAM_ABBREVIATION']

    # One row per team per game
    sides = pd.concat([
        games.assign(TEAM_ID=games['HOME_TEAM_ID'], OPP_TEAM_ID=games['AWAY_TEAM_ID'], HOME=1),
        games.assign(TEAM_ID=games['AWAY_TEAM_ID'], OPP_TEAM_ID=games['HOME_TEAM_ID'], HOME=0),
    ], ignore_index=True)
    sides['SEASON_YEAR'] = sides['TEAM_ID'].map(latest)
    sides['TEAM_GAME_NO'] = sides['TEAM_ID'].map(next_no)

    rotations = rotation_players(players, schedule, sides)
    out_keys = {name_key(n) for n in out_names}
    rotations['OUT'] = rotations['PLAYER_NAME'].map(name_key).isin(out_keys)
    rotations['TEAM_ABBREVIATION'] = rotations['TEAM_ID'].map(abbreviations)
    out_totals = summarize_out(rotations[rotations['OUT']], sides)

    upcoming = rotations.loc[~rotations['OUT'], ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'GAME_ID', 'GAME_DATE']]
    upcoming = upcoming.merge(sides[['TEAM_ID', 'GAME_ID', 'OPP_TEAM_ID', 'HOME', 'SEASON_YEAR']],
                              on=['TEAM_ID', 'GAME_ID'])
    upcoming['TEAM_ABBREVIATION'] = upcoming['TEAM_ID'].map(abbreviations)
    upcoming['MATCHUP'] = (upcoming['TEAM_ABBREVIATION'] + np.where(upcoming['HOME'] == 1, ' vs. ', ' @ ')
                           + upcoming['OPP_TEAM_ID'].map(abbreviations))
    upcoming['SEASON_TYPE'] = 'Regular Season'
    upcoming['PLAYOFFS'] = 0
    upcoming = attach_absences(upcoming, out_totals)

    # Only the history of these players is needed
    past = history[history['PLAYER_ID'].isin(upcoming['PLAYER_ID'])]
    df = add_player_features(pd.concat([past, upcoming], ignore_index=True))
    rows = df[df['GAME_ID'].isin(set(games['GAME_ID']))]
    rows = attach_team_form(rows, form, 'OPP_TEAM_ID', 'OPP_')
    rows = attach_team_form(rows, form, 'TEAM_ID', 'TEAM_')
    return rows.reset_index(drop=True), rotations


def name_key(name):
    """Looser name match for other sources: also drops punctuation and Jr./III suffixes."""
    words = re.sub(r'[^a-z ]', '', _simplify(name).replace('-', ' ')).split()
    return ' '.join(w for w in words if w not in {'jr', 'sr', 'ii', 'iii', 'iv'})


def upcoming_features(players, teams, player_name, opponent, home, game_date=None, playoffs=False,
                      out_names=()):
    """Features for a game that hasn't been played yet, built from all games before it.

    out_names: players who won't play in this game (from the injury report or typed in).
    Returns (row, rotations): the feature row, and both teams' rotation players with an OUT flag.
    """
    player = find_player(players, player_name)
    history = players[players['PLAYER_ID'] == player['PLAYER_ID']].sort_values('GAME_DATE')
    last = history.iloc[-1]

    team_ids = teams.drop_duplicates('TEAM_ABBREVIATION', keep='last').set_index('TEAM_ABBREVIATION')['TEAM_ID']
    opponent = opponent.upper()
    if opponent not in team_ids:
        raise ValueError(f'Unknown team "{opponent}". Use an abbreviation like {", ".join(team_ids.index[:5])}')
    team_id, opp_id = last['TEAM_ID'], team_ids[opponent]

    game_date = pd.Timestamp(game_date) if game_date else pd.Timestamp.today().normalize()
    season_year = last['SEASON_YEAR']  # assumes the game is in the player's latest season

    stats = team_game_stats(teams)
    form = team_form(stats)
    schedule = team_schedule(teams)

    # Players out in each of this player's past games, for his team and the opponent
    df = prepare_players(history, stats)
    past = pd.concat([df[['TEAM_ID', 'GAME_ID']],
                      df[['OPP_TEAM_ID', 'GAME_ID']].rename(columns={'OPP_TEAM_ID': 'TEAM_ID'})])
    past_targets = schedule.merge(past.drop_duplicates(), on=['TEAM_ID', 'GAME_ID'])
    df = attach_absences(df, absences(players, schedule, past_targets))

    # The upcoming game is each team's next game of its latest season
    next_games = []
    for tid in (team_id, opp_id):
        s = schedule[schedule['TEAM_ID'] == tid]
        s = s[s['SEASON_YEAR'] == s['SEASON_YEAR'].max()]
        next_games.append({'TEAM_ID': tid, 'SEASON_YEAR': s['SEASON_YEAR'].iloc[0],
                           'TEAM_GAME_NO': s['TEAM_GAME_NO'].max() + 1,
                           'GAME_ID': 'UPCOMING', 'GAME_DATE': game_date})
    next_games = pd.DataFrame(next_games)
    rotations = rotation_players(players, schedule, next_games)
    out_keys = {name_key(n) for n in out_names}
    rotations['OUT'] = (rotations['PLAYER_NAME'].map(name_key).isin(out_keys)
                        & (rotations['PLAYER_ID'] != player['PLAYER_ID']))
    abbreviations = {team_id: last['TEAM_ABBREVIATION'], opp_id: opponent}
    rotations['TEAM_ABBREVIATION'] = rotations['TEAM_ID'].map(abbreviations)

    upcoming = {c: np.nan for c in df.columns if c not in OUT_COLS and not c.endswith(tuple(OUT_COLS))}
    upcoming.update({
        'SEASON_YEAR': season_year, 'PLAYER_ID': player['PLAYER_ID'], 'PLAYER_NAME': player['PLAYER_NAME'],
        'TEAM_ID': team_id, 'TEAM_ABBREVIATION': last['TEAM_ABBREVIATION'], 'GAME_ID': 'UPCOMING',
        'GAME_DATE': game_date, 'MATCHUP': f"{last['TEAM_ABBREVIATION']} {'vs.' if home else '@'} {opponent}",
        'SEASON_TYPE': 'Playoffs' if playoffs else 'Regular Season',
        'HOME': int(home), 'PLAYOFFS': int(playoffs), 'OPP_TEAM_ID': opp_id,
    })
    upcoming = attach_absences(pd.DataFrame([upcoming]), summarize_out(rotations[rotations['OUT']], next_games))

    df = add_player_features(pd.concat([df, upcoming], ignore_index=True))
    row = df[df['GAME_ID'] == 'UPCOMING']
    row = attach_team_form(row, form, 'OPP_TEAM_ID', 'OPP_')
    row = attach_team_form(row, form, 'TEAM_ID', 'TEAM_')
    return row, rotations.sort_values(['TEAM_ID', 'MIN'], ascending=[True, False])
