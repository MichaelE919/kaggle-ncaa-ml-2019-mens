"""
Two-part script.

Uses Pandas to first create a prediction dataset then Scikit-Learn to create
several machine learning models.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from xgboost.sklearn import XGBClassifier

print('Loading data...')

df_tourney = pd.read_csv('DataFiles/NCAATourneyCompactResults.csv')
df_season = pd.read_csv('DataFiles/RegularSeasonDetailedResults.csv')
df_teams = pd.read_csv('DataFiles/Teams.csv')
df_kenpom = pd.read_csv('DataFiles/KenPom.csv')
df_seeds = pd.read_csv('DataFiles/NCAATourneySeeds.csv')
df_rankings = pd.read_csv('DataFiles/MasseyOrdinals_thru_2019_day_128.csv')
df_sample_sub = pd.read_csv('SampleSubmissionStage2.csv')

print('Checking for missing teams from Kaggle\'s dataset...')

missing_teams = []
for i in df_kenpom['TeamName']:
    if not len(df_teams[df_teams['TeamName'] == i]):
        missing_teams.append(i)
if len(missing_teams):
    for i in missing_teams:
        print(i)
else:
    print('No missing teams.')

df_kenpom['TeamID'] = df_kenpom['TeamName'].apply(
    lambda x: df_teams[df_teams['TeamName'] == x].values[0][0]
)

print('Creating efficiency stats...')

# Possession
df_season['WPoss'] = df_season.apply(
    lambda row: 0.96 * (row.WFGA + row.WTO + 0.44 * row.WFTA - row.WOR), axis=1
)
df_season['LPoss'] = df_season.apply(
    lambda row: 0.96 * (row.LFGA + row.LTO + 0.44 * row.LFTA - row.LOR), axis=1
)

# Shooting Efficiency
df_season['WSEf'] = df_season.apply(
    lambda row: row.WScore / (row.WFGA + 0.44 * row.WFTA), axis=1
)
df_season['LSEf'] = df_season.apply(
    lambda row: row.LScore / (row.LFGA + 0.44 * row.LFTA), axis=1
)

# Scoring Opportunity
df_season['WSOp'] = df_season.apply(
    lambda row: (row.WFGA + 0.44 * row.WFTA) / row.WPoss, axis=1
)
df_season['LSOp'] = df_season.apply(
    lambda row: (row.LFGA + 0.44 * row.LFTA) / row.LPoss, axis=1
)

print('Create columns for true shooting percentage and player impact estimate...')

# True Shooting Percentage
df_season['WTSP'] = df_season.apply(
    lambda row: row.WScore * 100 / (2 * (row.WFGA + 0.44 * row.WFTA)), axis=1
)
df_season['LTSP'] = df_season.apply(
    lambda row: row.LScore * 100 / (2 * (row.LFGA + 0.44 * row.LFTA)), axis=1
)

wPIE = df_season.apply(
    lambda row: row.WScore
    + row.WFGM
    + row.WFTM
    - row.WFGA
    - row.WFTA
    + row.WDR
    + 0.5 * row.WOR
    + row.WAst
    + row.WStl
    + 0.5 * row.WBlk
    - row.WPF
    - row.WTO,
    axis=1,
)
lPIE = df_season.apply(
    lambda row: row.LScore
    + row.LFGM
    + row.LFTM
    - row.LFGA
    - row.LFTA
    + row.LDR
    + 0.5 * row.LOR
    + row.LAst
    + row.LStl
    + 0.5 * row.LBlk
    - row.LPF
    - row.LTO,
    axis=1,
)

df_season['WPIE'] = wPIE / (wPIE + lPIE)
df_season['LPIE'] = lPIE / (wPIE + lPIE)

# Effective Field Goal Percentage =
# (Field Goals Made + 0.5*3P Field Goals Made) / Field Goal Attempts

print('Creating columns for the Four Factors...')
print('\tEffective Field Goal Percentage')

df_season['WeFGP'] = df_season.apply(
    lambda row: (row.WFGM + 0.5 * row.WFGM3) / row.WFGA, axis=1
)
df_season['LeFGP'] = df_season.apply(
    lambda row: (row.LFGM + 0.5 * row.LFGM3) / row.LFGA, axis=1
)

# Turnover Rate =
# Turnovers/(Field Goal Attempts + 0.44*Free Throw Attempts + Turnovers)

print('\tTurnover Rate')

df_season['WToR'] = df_season.apply(
    lambda row: row.WTO / (row.WFGA + 0.44 * row.WFTA + row.WTO), axis=1
)
df_season['LToR'] = df_season.apply(
    lambda row: row.LTO / (row.LFGA + 0.44 * row.LFTA + row.LTO), axis=1
)

# Offensive Rebounding Percentage =
# Offensive Rebounds / (Offensive Rebounds + Opponent’s Defensive Rebounds)

print('\tOffensive Rebounding Percentage')

df_season['WORP'] = df_season.apply(lambda row: row.WOR / (row.WOR + row.LDR), axis=1)
df_season['LORP'] = df_season.apply(lambda row: row.LOR / (row.LOR + row.WDR), axis=1)

# Free Throw Rate =
# Free Throws Made / Field Goals Attempted or Free Throws Attempted/Field Goals Attempted

print('\tFree Throw Rate')

df_season['WFTR'] = df_season.apply(lambda row: row.WFTA / row.WFGA, axis=1)
df_season['LFTR'] = df_season.apply(lambda row: row.LFTA / row.LFGA, axis=1)


# Utility functions to isolate the efficiency stat for each team and year
def getAdjO(Year, TeamID):
    try:
        AdjO = df_kenpom[
            (df_kenpom['TeamID'] == TeamID) & (df_kenpom['Season'] == Year)
        ].values[0][2]
    except IndexError:
        AdjO = df_kenpom[df_kenpom['TeamID'] == TeamID].mean().values[1].round(2)
    return AdjO


def getAdjD(Year, TeamID):
    try:
        AdjD = df_kenpom[
            (df_kenpom['TeamID'] == TeamID) & (df_kenpom['Season'] == Year)
        ].values[0][3]
    except IndexError:
        AdjD = df_kenpom[df_kenpom['TeamID'] == TeamID].mean().values[2].round(2)
    return AdjD


def getAdjEM(Year, TeamID):
    try:
        AdjEM = df_kenpom[
            (df_kenpom['TeamID'] == TeamID) & (df_kenpom['Season'] == Year)
        ].values[0][4]
    except IndexError:
        AdjEM = df_kenpom[df_kenpom['TeamID'] == TeamID].mean().values[3].round(2)
    return AdjEM


print('Creating Adjusted Efficiency columns...')

# Adjusted Offensive Efficiency
df_season['WAdjO'] = df_season.apply(
    lambda row: getAdjO(row['Season'], row['WTeamID']), axis=1
)
df_season['LAdjO'] = df_season.apply(
    lambda row: getAdjO(row['Season'], row['LTeamID']), axis=1
)

# Adjusted Defensive Efficiency
df_season['WAdjD'] = df_season.apply(
    lambda row: getAdjD(row['Season'], row['WTeamID']), axis=1
)
df_season['LAdjD'] = df_season.apply(
    lambda row: getAdjD(row['Season'], row['LTeamID']), axis=1
)

# Adjusted Efficiency Margin
df_season['WAdjEM'] = df_season.apply(
    lambda row: getAdjEM(row['Season'], row['WTeamID']), axis=1
)
df_season['LAdjEM'] = df_season.apply(
    lambda row: getAdjEM(row['Season'], row['LTeamID']), axis=1
)

print('Create remaining columns:')

# Defensive Rebounding Percentage =
# Defensive Rebounds / (Defensive Rebounds + Opponents Offensive Rebounds)
print('\tDefensive Rebounding Percentage')

df_season['WDRP'] = df_season.apply(lambda row: row.WDR / (row.WDR + row.LOR), axis=1)
df_season['LDRP'] = df_season.apply(lambda row: row.LDR / (row.LDR + row.WOR), axis=1)

# Rebound Percentage
print('\tRebound Percentage')

df_season['WRP'] = df_season.apply(lambda row: (row.WORP + row.WDRP) / 2, axis=1)
df_season['LRP'] = df_season.apply(lambda row: (row.LORP + row.LDRP) / 2, axis=1)

# Offensive Rebound to Turnover Margin
print('\tOffensive Rebound to Turnover Margin')

df_season['WORTM'] = df_season.apply(lambda row: row.WOR - row.WTO, axis=1)
df_season['LORTM'] = df_season.apply(lambda row: row.LOR - row.LTO, axis=1)

# Assist Ratio
print('\tAssist Ratio')

df_season['WAR'] = df_season.apply(
    lambda row: row.WAst * 100 / (row.WFGA + row.WFTA * 0.44 + row.WAst + row.WTO),
    axis=1,
)
df_season['LAR'] = df_season.apply(
    lambda row: row.LAst * 100 / (row.LFGA + row.LFTA * 0.44 + row.LAst + row.LTO),
    axis=1,
)

# Block Percentage
print('\tBlock Percentage')

df_season['WBKP'] = df_season.apply(lambda row: row.WBlk * 100 / row.LFGA, axis=1)
df_season['LBKP'] = df_season.apply(lambda row: row.LBlk * 100 / row.WFGA, axis=1)

# Steal Percentage
print('\tSteal Percentage')

df_season['WSTP'] = df_season.apply(lambda row: row.WStl * 100 / row.LPoss, axis=1)
df_season['LSTP'] = df_season.apply(lambda row: row.LStl * 100 / row.WPoss, axis=1)

# Score Differential = Points scored - points allowed
print('\tScore Differential')

df_season['WPtsDf'] = df_season.apply(lambda row: row.WScore - row.LScore, axis=1)
df_season['LPtsDf'] = df_season.apply(lambda row: row.LScore - row.WScore, axis=1)

print('Dropping unused columns...')

df_season.drop(
    labels=[
        'WFGM',
        'WFGA',
        'WFGM3',
        'WFGA3',
        'WFTM',
        'WFTA',
        'WOR',
        'WDR',
        'WAst',
        'WTO',
        'WStl',
        'WBlk',
        'WPF',
    ],
    axis=1,
    inplace=True,
)
df_season.drop(
    labels=[
        'LFGM',
        'LFGA',
        'LFGM3',
        'LFGA3',
        'LFTM',
        'LFTA',
        'LOR',
        'LDR',
        'LAst',
        'LTO',
        'LStl',
        'LBlk',
        'LPF',
    ],
    axis=1,
    inplace=True,
)

print('Creating prediction dataset...')

df_season_totals = pd.DataFrame()

# Calculate wins and losses to get winning percentage
df_season_totals['Wins'] = (
    df_season['WTeamID'].groupby([df_season['Season'], df_season['WTeamID']]).count()
)
df_season_totals['Losses'] = (
    df_season['LTeamID'].groupby([df_season['Season'], df_season['LTeamID']]).count()
)
df_season_totals['WinPCT'] = df_season_totals['Wins'] / (
    df_season_totals['Wins'] + df_season_totals['Losses']
)

# Calculate averages for games team won
df_season_totals['WSEf'] = (
    df_season['WSEf'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
)
df_season_totals['WSOp'] = (
    df_season['WSOp'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
)
df_season_totals['WTSP'] = (
    df_season['WTSP'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
)
df_season_totals['WPIE'] = (
    df_season['WPIE'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
)
df_season_totals['WeFGP'] = (
    df_season['WeFGP'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
)
df_season_totals['WToR'] = (
    df_season['WToR'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
)
df_season_totals['WORP'] = (
    df_season['WORP'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
)
df_season_totals['WFTR'] = (
    df_season['WFTR'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
)
df_season_totals['WAdjO'] = (
    df_season['WAdjO'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
)
df_season_totals['WAdjD'] = (
    df_season['WAdjD'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
)
df_season_totals['WAdjEM'] = (
    df_season['WAdjEM'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
)
df_season_totals['WDRP'] = (
    df_season['WDRP'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
)
df_season_totals['WRP'] = (
    df_season['WRP'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
)
df_season_totals['WORTM'] = (
    df_season['WORTM'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
)
df_season_totals['WAR'] = (
    df_season['WAR'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
)
df_season_totals['WBKP'] = (
    df_season['WBKP'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
)
df_season_totals['WSTP'] = (
    df_season['WSTP'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
)
df_season_totals['WPtsDf'] = (
    df_season['WPtsDf'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
)

# Calculate averages for games team lost
df_season_totals['LSEf'] = (
    df_season['LSEf'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
)
df_season_totals['LSOp'] = (
    df_season['LSOp'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
)
df_season_totals['LTSP'] = (
    df_season['LTSP'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
)
df_season_totals['LPIE'] = (
    df_season['LPIE'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
)
df_season_totals['LeFGP'] = (
    df_season['LeFGP'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
)
df_season_totals['LToR'] = (
    df_season['LToR'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
)
df_season_totals['LORP'] = (
    df_season['LORP'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
)
df_season_totals['LFTR'] = (
    df_season['LFTR'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
)
df_season_totals['LAdjO'] = (
    df_season['LAdjO'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
)
df_season_totals['LAdjD'] = (
    df_season['LAdjD'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
)
df_season_totals['LAdjEM'] = (
    df_season['LAdjEM'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
)
df_season_totals['LDRP'] = (
    df_season['LDRP'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
)
df_season_totals['LRP'] = (
    df_season['LRP'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
)
df_season_totals['LORTM'] = (
    df_season['LORTM'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
)
df_season_totals['LAR'] = (
    df_season['LAR'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
)
df_season_totals['LBKP'] = (
    df_season['LBKP'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
)
df_season_totals['LSTP'] = (
    df_season['LSTP'].groupby([df_season['Season'], df_season['WTeamID']]).mean()
)
df_season_totals['LPtsDf'] = (
    df_season['LPtsDf'].groupby([df_season['Season'], df_season['LTeamID']]).mean()
)

# Calculate weighted average using winning percent to weigh the statistic
df_season_totals['SEf'] = df_season_totals['WSEf'] * df_season_totals[
    'WinPCT'
] + df_season_totals['LSEf'] * (1 - df_season_totals['WinPCT'])
df_season_totals['SOp'] = df_season_totals['WSOp'] * df_season_totals[
    'WinPCT'
] + df_season_totals['LSOp'] * (1 - df_season_totals['WinPCT'])
df_season_totals['TSP'] = df_season_totals['WTSP'] * df_season_totals[
    'WinPCT'
] + df_season_totals['LTSP'] * (1 - df_season_totals['WinPCT'])
df_season_totals['PIE'] = df_season_totals['WPIE'] * df_season_totals[
    'WinPCT'
] + df_season_totals['LPIE'] * (1 - df_season_totals['WinPCT'])
df_season_totals['eFGP'] = df_season_totals['WeFGP'] * df_season_totals[
    'WinPCT'
] + df_season_totals['LeFGP'] * (1 - df_season_totals['WinPCT'])
df_season_totals['ToR'] = df_season_totals['WToR'] * df_season_totals[
    'WinPCT'
] + df_season_totals['LToR'] * (1 - df_season_totals['WinPCT'])
df_season_totals['ORP'] = df_season_totals['WORP'] * df_season_totals[
    'WinPCT'
] + df_season_totals['LORP'] * (1 - df_season_totals['WinPCT'])
df_season_totals['FTR'] = df_season_totals['WFTR'] * df_season_totals[
    'WinPCT'
] + df_season_totals['LFTR'] * (1 - df_season_totals['WinPCT'])
df_season_totals['AdjO'] = df_season_totals['WAdjO'] * df_season_totals[
    'WinPCT'
] + df_season_totals['LAdjO'] * (1 - df_season_totals['WinPCT'])
df_season_totals['AdjD'] = df_season_totals['WAdjD'] * df_season_totals[
    'WinPCT'
] + df_season_totals['LAdjD'] * (1 - df_season_totals['WinPCT'])
df_season_totals['AdjEM'] = df_season_totals['WAdjEM'] * df_season_totals[
    'WinPCT'
] + df_season_totals['LAdjEM'] * (1 - df_season_totals['WinPCT'])
df_season_totals['DRP'] = df_season_totals['WDRP'] * df_season_totals[
    'WinPCT'
] + df_season_totals['LDRP'] * (1 - df_season_totals['WinPCT'])
df_season_totals['RP'] = df_season_totals['WRP'] * df_season_totals[
    'WinPCT'
] + df_season_totals['LRP'] * (1 - df_season_totals['WinPCT'])
df_season_totals['ORTM'] = df_season_totals['WORTM'] * df_season_totals[
    'WinPCT'
] + df_season_totals['LORTM'] * (1 - df_season_totals['WinPCT'])
df_season_totals['AR'] = df_season_totals['WAR'] * df_season_totals[
    'WinPCT'
] + df_season_totals['LAR'] * (1 - df_season_totals['WinPCT'])
df_season_totals['BKP'] = df_season_totals['WBKP'] * df_season_totals[
    'WinPCT'
] + df_season_totals['LBKP'] * (1 - df_season_totals['WinPCT'])
df_season_totals['STP'] = df_season_totals['WSTP'] * df_season_totals[
    'WinPCT'
] + df_season_totals['LSTP'] * (1 - df_season_totals['WinPCT'])
df_season_totals['PtsDf'] = df_season_totals['WPtsDf'] * df_season_totals[
    'WinPCT'
] + df_season_totals['LPtsDf'] * (1 - df_season_totals['WinPCT'])

df_season_totals.reset_index(inplace=True)

print('Replacing null values...')

df_season_totals.loc[4064, 'Losses'] = 0
df_season_totals.loc[4064, 'WinPCT'] = 1
df_season_totals.loc[4064, 'SEf'] = df_season_totals.loc[4064, 'WSEf']
df_season_totals.loc[4064, 'SOp'] = df_season_totals.loc[4064, 'WSOp']
df_season_totals.loc[4064, 'TSP'] = df_season_totals.loc[4064, 'WTSP']
df_season_totals.loc[4064, 'PIE'] = df_season_totals.loc[4064, 'WPIE']
df_season_totals.loc[4064, 'eFGP'] = df_season_totals.loc[4064, 'WeFGP']
df_season_totals.loc[4064, 'ToR'] = df_season_totals.loc[4064, 'WToR']
df_season_totals.loc[4064, 'ORP'] = df_season_totals.loc[4064, 'WORP']
df_season_totals.loc[4064, 'FTR'] = df_season_totals.loc[4064, 'WFTR']
df_season_totals.loc[4064, 'AdjO'] = df_season_totals.loc[4064, 'WAdjO']
df_season_totals.loc[4064, 'AdjD'] = df_season_totals.loc[4064, 'WAdjD']
df_season_totals.loc[4064, 'AdjEM'] = df_season_totals.loc[4064, 'WAdjEM']
df_season_totals.loc[4064, 'DRP'] = df_season_totals.loc[4064, 'WDRP']
df_season_totals.loc[4064, 'RP'] = df_season_totals.loc[4064, 'WRP']
df_season_totals.loc[4064, 'ORTM'] = df_season_totals.loc[4064, 'WORTM']
df_season_totals.loc[4064, 'AR'] = df_season_totals.loc[4064, 'WAR']
df_season_totals.loc[4064, 'BKP'] = df_season_totals.loc[4064, 'WBKP']
df_season_totals.loc[4064, 'STP'] = df_season_totals.loc[4064, 'WSTP']
df_season_totals.loc[4064, 'PtsDf'] = df_season_totals.loc[4064, 'WPtsDf']

df_season_totals.loc[4211, 'Losses'] = 0
df_season_totals.loc[4211, 'WinPCT'] = 1
df_season_totals.loc[4211, 'SEf'] = df_season_totals.loc[4211, 'WSEf']
df_season_totals.loc[4211, 'SOp'] = df_season_totals.loc[4211, 'WSOp']
df_season_totals.loc[4211, 'TSP'] = df_season_totals.loc[4211, 'WTSP']
df_season_totals.loc[4211, 'PIE'] = df_season_totals.loc[4211, 'WPIE']
df_season_totals.loc[4211, 'eFGP'] = df_season_totals.loc[4211, 'WeFGP']
df_season_totals.loc[4211, 'ToR'] = df_season_totals.loc[4211, 'WToR']
df_season_totals.loc[4211, 'ORP'] = df_season_totals.loc[4211, 'WORP']
df_season_totals.loc[4211, 'FTR'] = df_season_totals.loc[4211, 'WFTR']
df_season_totals.loc[4211, 'AdjO'] = df_season_totals.loc[4211, 'WAdjO']
df_season_totals.loc[4211, 'AdjD'] = df_season_totals.loc[4211, 'WAdjD']
df_season_totals.loc[4211, 'AdjEM'] = df_season_totals.loc[4211, 'WAdjEM']
df_season_totals.loc[4211, 'DRP'] = df_season_totals.loc[4211, 'WDRP']
df_season_totals.loc[4211, 'RP'] = df_season_totals.loc[4211, 'WRP']
df_season_totals.loc[4211, 'ORTM'] = df_season_totals.loc[4211, 'WORTM']
df_season_totals.loc[4211, 'AR'] = df_season_totals.loc[4211, 'WAR']
df_season_totals.loc[4211, 'BKP'] = df_season_totals.loc[4211, 'WBKP']
df_season_totals.loc[4211, 'STP'] = df_season_totals.loc[4211, 'WSTP']
df_season_totals.loc[4211, 'PtsDf'] = df_season_totals.loc[4211, 'WPtsDf']

df_season_totals.drop(
    labels=[
        'Wins',
        'Losses',
        'WSEf',
        'WSOp',
        'WTSP',
        'WPIE',
        'WeFGP',
        'WToR',
        'WORP',
        'WFTR',
        'WAdjO',
        'WAdjD',
        'WAdjEM',
        'WDRP',
        'WRP',
        'WORTM',
        'WAR',
        'WBKP',
        'WSTP',
        'WPtsDf',
        'LSEf',
        'LSOp',
        'LTSP',
        'LPIE',
        'LeFGP',
        'LToR',
        'LORP',
        'LFTR',
        'LAdjO',
        'LAdjD',
        'LAdjEM',
        'LDRP',
        'LRP',
        'LORTM',
        'LAR',
        'LBKP',
        'LSTP',
        'LPtsDf',
    ],
    axis=1,
    inplace=True,
)

columns = df_season_totals.columns.tolist()
columns.pop(2)
columns.append('WinPCT')
df_season_totals = df_season_totals[columns]
df_season_totals.rename(columns={'WTeamID': 'TeamID'}, inplace=True)

df_net_2019 = df_rankings[df_rankings['SystemName'] == 'NET']
df_net_2019_final = df_net_2019[df_net_2019['RankingDayNum'] == 128]


df_rpi = df_rankings[
    df_rankings['SystemName'] == 'RPI'
]  # This is necessary to get RPI rankings for 2018 and prior
df_rpi_prev_final = df_rpi[df_rpi['RankingDayNum'] == 133]

df_rnt_final = pd.concat([df_rpi_prev_final, df_net_2019_final])

df_rnt_final = df_rnt_final.drop(labels=['RankingDayNum', 'SystemName'], axis=1)

df_seeds['Seed_new'] = df_seeds['Seed'].apply(lambda x: int(x[1:3]))

df_seeds.drop(labels='Seed', axis=1, inplace=True)
df_seeds.rename(columns={'Seed_new': 'Seed'}, inplace=True)

# Use tourney seeds from 2003 on
df_seeds_final = df_seeds[df_seeds['Season'] > 2002]

df_tourney_temp = pd.merge(
    left=df_seeds_final, right=df_rnt_final, how='left', on=['Season', 'TeamID']
)
df_tourney_final = pd.merge(
    left=df_tourney_temp, right=df_season_totals, how='left', on=['Season', 'TeamID']
)

df_tourney.drop(
    labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1
)
df_tourney = df_tourney[df_tourney['Season'] > 2002]
df_tourney.reset_index(drop=True, inplace=True)

df_win_teams = pd.merge(
    left=df_tourney,
    right=df_tourney_final,
    how='left',
    left_on=['Season', 'WTeamID'],
    right_on=['Season', 'TeamID'],
)
df_win_teams.drop(labels='TeamID', inplace=True, axis=1)

df_loss_teams = pd.merge(
    left=df_tourney,
    right=df_tourney_final,
    how='left',
    left_on=['Season', 'LTeamID'],
    right_on=['Season', 'TeamID'],
)
df_loss_teams.drop(labels='TeamID', inplace=True, axis=1)

df_win_diff = df_win_teams.iloc[:, 3:] - df_loss_teams.iloc[:, 3:]
df_win_diff['result'] = 1
df_win_diff = pd.merge(
    left=df_win_diff, right=df_tourney, left_index=True, right_index=True, how='inner'
)

df_loss_diff = df_loss_teams.iloc[:, 3:] - df_win_teams.iloc[:, 3:]
df_loss_diff['result'] = 0
df_loss_diff = pd.merge(
    left=df_loss_diff, right=df_tourney, left_index=True, right_index=True, how='inner'
)

prediction_dataset = pd.concat((df_win_diff, df_loss_diff), axis=0)
prediction_dataset.sort_values('Season', inplace=True)

prediction_dataset['Seed'] = prediction_dataset['Seed'].astype('float64')
prediction_dataset['OrdinalRank'] = prediction_dataset['OrdinalRank'].astype('float64')

print('Separating dataset into features, labels, and IDs...')

y = prediction_dataset['result']
X = prediction_dataset.loc[:, :'WinPCT']
train_IDs = prediction_dataset.loc[:, 'Season':]

print('Creating training and test sets...')

# Split the data into training data and a test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42, shuffle=True
)

print('Creating classifier objects, parameter grids, and pipelines...')

# Initiate classifiers
clf1 = LogisticRegression(random_state=32)
clf2 = SVC(probability=True, random_state=32)
clf3 = XGBClassifier(objective='binary:logistic', random_state=32)
clf4 = DecisionTreeClassifier(random_state=32)
clf5 = RandomForestClassifier(random_state=32)
clf6 = GradientBoostingClassifier(random_state=32)
sclr = StandardScaler()

# Configure Parameter grids
param_grid1 = [
    {
        'clf1__C': list(np.logspace(start=-5, stop=3, num=9)),
        'clf1__solver': ['newton-cg', 'lbfgs', 'liblinear'],
        'clf1__max_iter': [10000],
        'clf1__multi_class': ['ovr'],
    }
]

param_grid2 = [
    {
        'clf2__C': np.logspace(start=-3, stop=3, num=7),
        'clf2__gamma': np.logspace(start=-4, stop=-1, num=4),
    }
]

param_grid3 = [
    {
        'clf3__learning_rate': [0.1, 0.3],
        'clf3__max_depth': [2, 4, 8, 12],
        'clf3__min_child_weight': [2, 4, 8],
        'clf3__colsample_bytree': [0.25, 0.5, 0.75],
    }
]

param_grid4 = [
    {
        'clf4__max_depth': list(range(3, 6)),
        'clf4__criterion': ['gini', 'entropy'],
        'clf4__min_samples_leaf': np.linspace(0.1, 0.5, 5),
        'clf4__min_samples_split': np.linspace(0.1, 0.5, 5),
    }
]

param_grid5 = [
    {
        'clf5__n_estimators': [16, 32, 64, 128],
        'clf5__max_depth': list(range(1, 5)),
        'clf5__criterion': ['gini', 'entropy'],
        'clf5__min_samples_leaf': [2, 4, 8, 16],
        'clf5__min_samples_split': [2, 3],
    }
]

param_grid6 = [
    {
        'clf6__learning_rate': [0.01, 0.1],
        'clf6__loss': ['deviance', 'exponential'],
        'clf6__max_depth': list(range(3, 4)),
    }
]

# Build the pipelines
pipe1 = Pipeline([('scaler', sclr), ('clf1', clf1)])

pipe2 = Pipeline([('scaler', sclr), ('clf2', clf2)])

pipe3 = Pipeline([('scaler', sclr), ('clf3', clf3)])

pipe4 = Pipeline([('scaler', sclr), ('clf4', clf4)])

pipe5 = Pipeline([('scaler', sclr), ('clf5', clf5)])

pipe6 = Pipeline([('scaler', sclr), ('clf6', clf6)])


print('Performing Grid Search Cross Validation...')

# Create empty list and initialize counter to build model comparison dataframe
mods = []
counter = 0

# Set up GridSearchCV objects, one for each algorithm
gridcvs = {}

inner_cv = StratifiedKFold(n_splits=10, shuffle=False, random_state=2)
outer_cv = StratifiedKFold(n_splits=10, shuffle=False, random_state=2)

for pgrid, est, name in zip(
    (param_grid1, param_grid2, param_grid3, param_grid4, param_grid5, param_grid6),
    (pipe1, pipe2, pipe3, pipe4, pipe5, pipe6),
    ('Logistic', 'SVM', 'XGBoost', 'DTree', 'Random Forest', 'Gradient Boosting'),
):

    # First loop runs GridSearch and does Cross validation to find the best parameters

    gcv = GridSearchCV(
        estimator=est,
        param_grid=pgrid,
        iid=False,
        scoring='neg_log_loss',
        cv=outer_cv,
        verbose=0,
        refit=True,
        return_train_score=False,
    )

    gcv.fit(X_train, y_train)

    gridcvs[name] = gcv

    print(name)
    print()
    print(gcv.best_estimator_)
    print()
    print(f'Best score on Grid Search Cross Validation is {gcv.best_score_:.2f}')
    print()

    # Inner loop runs Cross Val Score on tuned parameter model to determine accuracy of fit

    # for name, gs_est in sorted(gridcvs.items()):

    nested_score = 0
    nested_score = cross_val_score(
        gcv, X=X_test, y=y_test, cv=inner_cv, scoring='neg_log_loss'
    )

    print(
        'Name, Log Loss, Std Dev, based on Best Parameter Model using Cross Validation Scoring'
    )
    print(f'{name} | {nested_score.mean():.2f} {nested_score.std() * 100:.2f}')
    print()

    # Generate predictions and probabilities

    best_algo = gcv

    best_algo.fit(X_train, y_train)

    train_acc = accuracy_score(y_true=y_train, y_pred=best_algo.predict(X_train))
    test_acc = accuracy_score(y_true=y_test, y_pred=best_algo.predict(X_test))

    predictions = best_algo.predict(X_test)
    probability = best_algo.predict_proba(X_test)[:, 1]

    auroc = roc_auc_score(y_test, probability)
    f1 = f1_score(y_test, predictions)

    print(f'Training Accuracy: {100 * train_acc:.2f}')
    print(f'Test Accuracy: {100 * test_acc:.2f}')
    print()
    print(f'Area under the ROC curve: {auroc:.3f}')
    print()

    # Prints classification report and confusion matrix

    #     if name != 'SVM':

    print(classification_report(y_test, predictions))
    print()
    print(confusion_matrix(y_test, predictions))
    print()

    #     else:
    #         print()

    # Create a dataframe containing a single row of evaluation metrics for each algorithm

    mod = pd.DataFrame(
        {
            'model_name': name,
            'Train Accuracy': round(train_acc, 3),
            'Test Accuracy': round(test_acc, 3),
            'f1_score': round(f1, 3),
            'AUROC': round(auroc, 3),
            'Log Loss': round(nested_score.mean(), 3),
        },
        index=[counter],
    )

    mods.append(mod)
    counter += 1

# Concatenate single row dataframes into multirow dataframe for all algorithms
model_comp_plus_sclr = pd.concat(mods, ignore_index=False)

# model_comp_plus_sclr.to_pickle('model_comp_plus_sclr.pkl')

# Sort models by Test Accuracy, f1 score, area under the ROC, and Log Loss
compare_models = model_comp_plus_sclr.copy()
compare_models.set_index('model_name', inplace=True)
metrics = ['Test Accuracy', 'f1_score', 'AUROC', 'Log Loss']

for metric in metrics:
    print(compare_models.sort_values(by=metric, ascending=False))

# Initialize standard scaler and transform data
sclr = StandardScaler(copy=True, with_mean=True, with_std=True)

X = sclr.fit_transform(X)

# Initialize classifier and fit data
# clf = LogisticRegression(
#     C=10.0,
#     class_weight=None,
#     dual=False,
#     fit_intercept=True,
#     intercept_scaling=1,
#     max_iter=10000,
#     multi_class='ovr',
#     n_jobs=None,
#     penalty='l2',
#     random_state=None,
#     solver='liblinear',
#     tol=0.0001,
#     verbose=0,
#     warm_start=False,
# )

clf = SVC(
    C=1000.0,
    cache_size=200,
    class_weight=None,
    coef0=0.0,
    decision_function_shape='ovr',
    degree=3,
    gamma=0.0001,
    kernel='rbf',
    max_iter=-1,
    probability=True,
    random_state=32,
    shrinking=True,
    tol=0.001,
    verbose=False,
)

# clf = XGBClassifier(
#     base_score=0.5,
#     booster='gbtree',
#     colsample_bylevel=1,
#     colsample_bytree=0.75,
#     gamma=0,
#     learning_rate=0.1,
#     max_delta_step=0,
#     max_depth=2,
#     min_child_weight=2,
#     missing=None,
#     n_estimators=100,
#     n_jobs=1,
#     nthread=None,
#     objective='binary:logistic',
#     random_state=32,
#     reg_alpha=0,
#     reg_lambda=1,
#     scale_pos_weight=1,
#     seed=None,
#     silent=True,
#     subsample=1,
# )

# clf = DecisionTreeClassifier(
#     class_weight=None,
#     criterion='gini',
#     max_depth=3,
#     max_features=None,
#     max_leaf_nodes=None,
#     min_impurity_decrease=0.0,
#     min_impurity_split=None,
#     min_samples_leaf=50,
#     min_samples_split=2,
#     min_weight_fraction_leaf=0.0,
#     presort=False,
#     random_state=None,
#     splitter='best',
# )

# clf = RandomForestClassifier(
#     bootstrap=True,
#     class_weight=None,
#     criterion='entropy',
#     max_depth=4,
#     max_features='auto',
#     max_leaf_nodes=None,
#     min_impurity_decrease=0.0,
#     min_impurity_split=None,
#     min_samples_leaf=1,
#     min_samples_split=2,
#     min_weight_fraction_leaf=0.0,
#     n_estimators=100,
#     n_jobs=None,
#     oob_score=False,
#     random_state=None,
#     verbose=0,
#     warm_start=False,
# )

# clf = GradientBoostingClassifier(
#     criterion='friedman_mse',
#     init=None,
#     learning_rate=0.1,
#     loss='deviance',
#     max_depth=3,
#     max_features=None,
#     max_leaf_nodes=None,
#     min_impurity_decrease=0.0,
#     min_impurity_split=None,
#     min_samples_leaf=1,
#     min_samples_split=2,
#     min_weight_fraction_leaf=0.0,
#     n_estimators=100,
#     n_iter_no_change=None,
#     presort='auto',
#     random_state=None,
#     subsample=1.0,
#     tol=0.0001,
#     validation_fraction=0.1,
#     verbose=0,
#     warm_start=False,
# )

# clf = LinearSVC(dual=False, C=10.0)

print('Fitting data into classifier...')

clf.fit(X, y)

# Create data to input into the model
n_test_games = len(df_sample_sub)


def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))


print('Creating submission file...')

X_test = np.zeros(shape=(n_test_games, 1))
columns = df_tourney_final.columns.get_values()
model = []
data = []

for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)

    team1 = df_tourney_final[
        (df_tourney_final.TeamID == t1) & (df_tourney_final.Season == year)
    ].values
    team2 = df_tourney_final[
        (df_tourney_final.TeamID == t2) & (df_tourney_final.Season == year)
    ].values

    model = team1 - team2

    data.append(model)

Predictions = pd.DataFrame(
    np.array(data).reshape(n_test_games, df_tourney_final.shape[1]), columns=(columns)
)

Predictions.drop(labels=['Season', 'TeamID'], inplace=True, axis=1)

# Scale the prediction set
Predictions = sclr.fit_transform(Predictions)

# Generate the predictions

# preds = clf.predict(Predictions)
preds = clf.predict_proba(Predictions)[:, 1]

df_sample_sub['Pred'] = preds
df_sample_sub.head()


# Generate submission file
# df_sample_sub.to_csv('2019_predictions_lr.csv', index=False)
df_sample_sub.to_csv('Submissions/2019_predictions_svm_plus_sclr.csv', index=False)
# df_sample_sub.to_csv('2019_predictions_xgb.csv', index=False)
# df_sample_sub.to_csv('Submissions/2019_predictions_dtree.csv', index=False)
# df_sample_sub.to_csv('Submissions/2019_predictions_rf.csv', index=False)
# df_sample_sub.to_csv('Submissions/2019_predictions_gb.csv', index=False)
# df_sample_sub.to_csv('Predictions/2018_predictions_lsvc.csv', index=False)

# Use to fill out a bracket
def build_team_dict():
    team_ids = pd.read_csv('DataFiles/Teams.csv')
    team_id_map = {}
    for _, row in team_ids.iterrows():
        team_id_map[row['TeamID']] = row['TeamName']
    return team_id_map


print('Creating predictions file...')

team_id_map = build_team_dict()
readable = []
less_readable = []  # A version that's easy to look up.
submission_data = df_sample_sub.values.tolist()
for pred in submission_data:
    parts = pred[0].split('_')
    less_readable.append(
        [team_id_map[int(parts[1])], team_id_map[int(parts[2])], pred[1]]
    )
    # Order them properly.
    if pred[1] > 0.5:
        winning = int(parts[1])
        losing = int(parts[2])
        proba = pred[1]
    else:
        winning = int(parts[2])
        losing = int(parts[1])
        proba = 1 - pred[1]
    readable.append(
        ['%s beats %s: %f' % (team_id_map[winning], team_id_map[losing], proba)]
    )

Finalpredictions = pd.DataFrame(readable)
Finalpredictions.to_csv('Predictions/SVM_plus_sclr_predictions.csv', index=False)
