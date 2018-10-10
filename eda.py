import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

matches_06_10 = pd.read_csv('./data/2006-10_data.csv')
matches14 = pd.read_csv('./data/2014_data.csv')
cups = pd.read_csv('./data/WorldCups.csv')


def create_team_stats(x, df):
    away_team_matches = df[df["Away Team Name"]==x]["Away Team Name"].value_counts()[0]
    if x in df["Home Team Name"]:
        home_team_matches = df[matches["Home Team Name"]==x]["Home Team Name"].value_counts()[0]
    else:
        home_team_matches = 0
    total_matches = away_team_matches + home_team_matches
    GF = (df[df["Home Team Name"]==x]["Home Team Goals"].sum() + df[df["Away Team Name"]==x]["Away Team Goals"].sum())
    GA = (df[df["Home Team Name"]==x]["Away Team Goals"].sum() + df[df["Away Team Name"]==x]["Home Team Goals"].sum())
    avg_score = GF/total_matches
    times_hosted = len(cups[cups["Country"]==x])
    total_wins = len(df[df["Winner"]==x])
    total_draws = len(df[(df["Home Team Name"]==x)&(df["Winner"]=='Draw')]) \
    + len(df[(df["Away Team Name"]==x)&(df["Winner"]=='Draw')])
    losses = total_matches - total_wins - total_draws
    if x in cups["Winner"]:
        champs = len(cups[cups["Winner"]==x])
    else:
        champs =0
    data = {
        'Team':x,
        'TM':total_matches,
        'GF':GF,
        'GA':GA,
        'ag':round(avg_score,2),
        'th':times_hosted,
        'tw':total_wins,
        'td':total_draws,
        'tl':losses,
        'c':champs
    }
    return data


if __name__ == '__main__':
    # team_match_ups('France','Brazil')
    # determine_match_winner('France','Brazil')
    # winners = matches.groupby('Winner')
    # print(len(winners))
    # print(winners.agg({'Winner':'count'}))
    all_teams = list(set(matches14["Away Team Name"]))
    tm = []
    GF = []
    GA = []
    ag = []
    th = []
    tw = []
    td = []
    tl = []
    c = []
    teams = []
    for team in all_teams:
        team_data = create_team_stats(team,matches14)
        teams.append(team_data['Team'])
        GF.append(team_data['GF'])
        GA.append(team_data['GA'])
        ag.append(team_data['ag'])
        th.append(team_data['th'])
        tw.append(team_data['tw'])
        td.append(team_data['td'])
        tl.append(team_data['tl'])
        tm.append(team_data['TM'])
        c.append(team_data['c'])
    teams_df = pd.DataFrame({
                            'Team':teams,
                            'Goals':GF,
                            'Goals_Against':GA,
                            'Avg_Goals_Per_Match':ag,
                            'Cups_Hosted':th,
                            'All_Wins':tw,
                            'All_Draws':td,
                            'All_Losses':tl,
                            'Total_Matches':tm
                            })
    # import pdb; pdb.set_trace()
    teams_df.to_csv('./data/test.csv')
