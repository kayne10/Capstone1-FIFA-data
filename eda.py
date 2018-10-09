import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

matches = pd.read_csv('./data/CleanMatches.csv')
cups = pd.read_csv('./data/WorldCups.csv')

def create_team_stats(x):
    totalMatches = (matches[matches["Home Team Name"]==x]["Home Team Name"].value_counts() + matches[matches["Away Team Name"]==x]["Away Team Name"].value_counts())
    GF = (matches[matches["Home Team Name"]==x]["Home Team Goals"].sum() + matches[matches["Away Team Name"]==x]["Away Team Goals"].sum())
    GA = (matches[matches["Home Team Name"]==x]["Away Team Goals"].sum() + matches[matches["Away Team Name"]==x]["Home Team Goals"].sum())
    avg_score = GF/totalMatches
    times_hosted = len(cups[cups["Country"]==x])
    print("In FiFa WorldCup to time \n")
    print("{} played                  %d Matches".format(x) %totalMatches)
    print("{} Scored                  %d Goals".format(x) %GF)
    print("Goals Against {}           %d".format(x) %GA)
    print("On an Average {} Scored    %.2f Goals".format(x) %avg_score)
    print("{} hosted WorldCup {} times".format(x, times_hosted))


def determine_match_winner(x,y):
    WT = matches[matches["Away Team Name"]==y]
    WT = WT[WT["Home Team Name"]==x]
    Wx = Wy = d = 0

    htg = np.array(WT["Home Team Goals"].astype(int))
    atg = np.array(WT["Away Team Goals"].astype(int))

    print(htg)
    print(atg)
    # for i in matches["Home Team Name"]:
    #     if htg[i] > atg[i]:
    #         Wx += 1
    #     elif htg[i] < atg[i]:
    #         Wy += 1
    #     else:
    #         d += 1
    # return (Wx, Wy, d)

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


create_team_stats('Brazil')
# if __name__ == '__main__':
#     # team_match_ups('France','Brazil')
#     determine_match_winner('France','Brazil')
