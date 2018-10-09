import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

matches = pd.read_csv('./data/WorldCupMatches.csv')
players = pd.read_csv('./data/WorldCupPlayers.csv')
cups = pd.read_csv('./data/WorldCups.csv')

# Replace Germany FR with Germany
cups["Winner"].replace(to_replace="Germany FR", value="Germany",inplace=True)
cups["Runners-Up"].replace(to_replace="Germany FR", value="Germany",inplace=True)
cups["Third"].replace(to_replace="Germany FR", value="Germany",inplace=True)
cups["Fourth"].replace(to_replace="Germany FR", value="Germany",inplace=True)

# Replace Germany FR with Germany
matches["Away Team Name"].replace(to_replace="Germany FR", value="Germany",inplace=True)
matches["Home Team Name"].replace(to_replace="Germany FR", value="Germany",inplace=True)

# Clean up Home Team Names
matches["Home Team Name"] = matches["Home Team Name"].str.replace('rn">United Arab Emirates',"United Arab Emirates")
matches["Home Team Name"] = matches["Home Team Name"].str.replace("C�te d'Ivoire","Côte d’Ivoire")
matches["Home Team Name"] = matches["Home Team Name"].str.replace('rn">Republic of Ireland',"Republic of Ireland")
matches["Home Team Name"] = matches["Home Team Name"].str.replace('rn">Bosnia and Herzegovina',"Bosnia and Herzegovina")
matches["Home Team Name"] = matches["Home Team Name"].str.replace('rn">Serbia and Montenegro',"Serbia and Montenegro")
matches["Home Team Name"] = matches["Home Team Name"].str.replace('rn">Trinidad and Tobago',"Trinidad and Tobago")
matches["Home Team Name"] = matches["Home Team Name"].str.replace("Soviet Union","Russia")
matches["Home Team Name"] = matches["Home Team Name"].str.replace("Germany FR","Germany")

# clean up Away Team Names
matches["Away Team Name"] = matches["Away Team Name"].str.replace('rn">United Arab Emirates',"United Arab Emirates")
matches["Away Team Name"] = matches["Away Team Name"].str.replace("C�te d'Ivoire","Côte d’Ivoire")
matches["Away Team Name"] = matches["Away Team Name"].str.replace('rn">Republic of Ireland',"Republic of Ireland")
matches["Away Team Name"] = matches["Away Team Name"].str.replace('rn">Bosnia and Herzegovina',"Bosnia and Herzegovina")
matches["Away Team Name"] = matches["Away Team Name"].str.replace('rn">Serbia and Montenegro',"Serbia and Montenegro")
matches["Away Team Name"] = matches["Away Team Name"].str.replace('rn">Trinidad and Tobago',"Trinidad and Tobago")
matches["Away Team Name"] = matches["Away Team Name"].str.replace("Soviet Union","Russia")
matches["Away Team Name"] = matches["Away Team Name"].str.replace("Germany FR","Germany")

# Remove all missing values
matches = matches.dropna(how='all')

# Create Winner Column
matches['Winner'] = matches['Home Team Name']
matches['Winner'][matches['Home Team Goals'] > matches['Away Team Goals']] = matches['Home Team Name']
matches['Winner'][matches['Home Team Goals'] < matches['Away Team Goals']] = matches['Away Team Name']
matches['Winner'][matches['Home Team Goals'] == matches['Away Team Goals']] = 'Draw'
print(matches[['Home Team Name','Home Team Goals','Away Team Name','Away Team Goals','Winner']].head())
all_draws = len(matches[matches['Winner']=='Draw'])
print('Total number of Draws: {}'.format(all_draws))

# Create new clean csv
matches.to_csv('./data/CleanMatches.csv')
