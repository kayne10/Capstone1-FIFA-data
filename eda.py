import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

matches = pd.read_csv('./data/CleanMatches.csv')
cups = pd.read_csv('./data/WorldCups.csv')

def create_team_stats(x):
    away_team_matches = matches[matches["Away Team Name"]==x]["Away Team Name"].value_counts()[0]
    if x in matches["Home Team Name"]:
        home_team_matches = matches[matches["Home Team Name"]==x]["Home Team Name"].value_counts()[0]
    else:
        home_team_matches = 0
    total_matches = away_team_matches + home_team_matches
    GF = (matches[matches["Home Team Name"]==x]["Home Team Goals"].sum() + matches[matches["Away Team Name"]==x]["Away Team Goals"].sum())
    GA = (matches[matches["Home Team Name"]==x]["Away Team Goals"].sum() + matches[matches["Away Team Name"]==x]["Home Team Goals"].sum())
    avg_score = GF/total_matches
    times_hosted = len(cups[cups["Country"]==x])
    total_wins = len(matches[matches["Winner"]==x])
    total_draws = len(matches[(matches["Home Team Name"]==x)&(matches["Winner"]=='Draw')]) \
    + len(matches[(matches["Away Team Name"]==x)&(matches["Winner"]=='Draw')])
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
    # print("In FiFa WorldCup to time \n")
    # print("{} played                  %d Matches".format(x) %total_matches)
    # print("{} Scored                  %d Goals".format(x) %GF)
    # print("Goals Against {}           %d".format(x) %GA)
    # print("On an Average {} Scored    %.2f Goals".format(x) %avg_score)
    # print("{} has %d total wins".format(x) %total_wins)
    # print("{} has %d total draws".format(x) %total_draws)
    # print("{} has %d total losses".format(x) %losses)
    # print("{} hosted World Cup {} times".format(x, times_hosted))
    # print("{} has won %d World Cups".format(x) %champs)


# Plot a pie chart to show Encounters of Team x vs Team y
def plot_pie(WinX, WinY, D, x, y):
    plt.figure(figsize=(9,9))
    grid = GridSpec(1,2)
    labels = 'Wins', 'Draws', 'Loss'
    fracs1 = [WinX, D, WinY]
    fracs2 = [WinY, D, WinX]
    exp = (0, 0.05, 0)
    plt.subplot(grid[0,0], aspect=1)
    plt.title('Matches Outcome for team %s'%(x))
    plt.pie(fracs1, labels=labels, autopct='%1.0f%%', shadow=True,
                            colors=['green','blue','tomato'],
                            wedgeprops={'linewidth':2,'edgecolor':'white'})
    cir = plt.Circle((0,0), .7, color="white")
    plt.gca().add_artist(cir)
    plt.show()


def team_match_ups(x,y):
    (wx1, wy1, d1) = determine_match_winner(x,y)
    (wx2, wy2, d2) = determine_match_winner(y,x)
    WinX = wx1 + wx2
    WinY = wy1 + wy2
    D = d1 + d2
    n = WinX + WinY
    print("Of %d Encounter(s) the two team had in WorldCup (till 2014)- \nTeam %s won %d Matches against %s"%(n,x,WinX,y))
    print("Team %s won %d Matches against %s"%(y,WinY,x))
    print("Matches drawn ",D)
    plot_pie(WinX, WinY, Draws, x, y)



if __name__ == '__main__':
    # team_match_ups('France','Brazil')
    # determine_match_winner('France','Brazil')
    winners = matches.groupby('Winner')
    # print(len(winners))
    # print(winners.agg({'Winner':'count'}))
    all_teams = list(set(matches["Away Team Name"]))
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
        team_data = create_team_stats(team)
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
    teams_df.to_csv('./data/teams.csv')
