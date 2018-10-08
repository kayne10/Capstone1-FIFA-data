import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# load data into dataframes
matches = pd.read_csv('./data/WorldCupMatches.csv')
players = pd.read_csv('./data/WorldCupPlayers.csv')
cups = pd.read_csv('./data/WorldCups.csv')

# Replace Germany FR with Germany
cups["Winner"].replace(to_replace="Germany FR", value="Germany",inplace=True)
cups["Runners-Up"].replace(to_replace="Germany FR", value="Germany",inplace=True)
cups["Third"].replace(to_replace="Germany FR", value="Germany",inplace=True)
cups["Fourth"].replace(to_replace="Germany FR", value="Germany",inplace=True)

Winnerdata = {}
listingYears = []
for i in cups["Winner"]:
    listingYears.append(list(cups[cups["Winner"]==i]["Year"]))

j = 0
for k in cups["Winner"]:
    if k not in Winnerdata.keys():
        Winnerdata[k] = listingYears[j]
    j = j + 1

print(Winnerdata)

#getting required Winner data onto list for plot
names = list(Winnerdata.keys())
values = list(Winnerdata.values())

# Plot stacked bar graph
fig_size = plt.rcParams["figure.figsize"]
# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

for j in range(8):
    plt.bar(j, len(values[j]),tick_label=names[j],hatch="-")
    j = j +1
plt.xticks(range(0,8),names)
plt.title("Countries Winning FIFA WorldCup till 2018",color='blue')
plt.xlabel("Countries", color = 'green')
plt.ylabel("Wins",color = 'green')

x=0
for i in  range(len(names)):
    y = 0.72
    for j in range(len(values[i])):
        plt.text(x, y, ""+str(values[i][j]), color='black', va='center', fontweight='bold',horizontalalignment='center')
        y = y + 0.77
    x = x+1

plt.savefig('./images/winner_countries.png')
plt.show()
