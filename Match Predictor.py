import pandas as pd
from sklearn.ensemble import RandomForestClassifier #picks up non linearities in data
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

matches = pd.read_csv('matches.csv', index_col=0) #indicates that first column is the index column

#ml algorithms cant work with strings
#check datatypes with matches.dtypes

matches['date'] = pd.to_datetime(matches['date']) #converting existing date column to a date/time column

#Predictors
matches['venue_code'] = matches['venue'].astype('category').cat.codes #converts home/away column into categorical values
matches['opp_code'] = matches['opponent'].astype('category').cat.codes #gives every club a value from 0-19
matches['hour'] = matches['time'].str.replace(':.+', '', regex=True).astype('int') #initial time is hh::mm. This removes the :mm and keeps only the hour
matches['day_code'] = matches['date'].dt.day_of_week #Assigns a number based on the day of the week the game was played on

matches['target'] = (matches['result'] == 'W').astype('int') #target is if the team won or not (returns 1/0)
print(matches)
#train model
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

train = matches[matches['date']< '2022-01-01'] #taking any data from before 2022 to put in training set
test = matches[matches['date'] > '2022-01-01'] #taking any data from after 2021 to put in testing set

predictors = ['venue_code', 'opp_code', 'hour', 'day_code']

rf.fit(train[predictors], train['target']) #train the model with the predictors to try and predicted the target (win [1] or not[0])

preds = rf.predict(test[predictors])

acc = accuracy_score(test['target'], preds)
print(acc) #score of 0.61

combined = pd.DataFrame(dict(actual=test['target'], prediction=preds)) #combines actual values with predicted values
# print(pd.crosstab(index=combined['actual'], columns=['prediction']))

print(precision_score(test['target'], preds)) #when a win was predicted, score was 0.47

grouped_matches = matches.groupby('team') #creates a dataframe for every team
group = grouped_matches.get_group('Manchester City')

def rolling_averages(group, cols, new_cols): #function which gets the given stats from the previous 3 games to help predict result in a 4th game
    group = group.sort_values('date')
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols) #drops missing data
    return group

cols = ['gf', 'ga', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt']
new_cols = [f'{c}_rolling' for c in cols]

# print(rolling_averages(group, cols, new_cols))

matches_rolling = matches.groupby('team').apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')

matches_rolling.index = range(matches_rolling.shape[0]) #assigns values from 0 to 1316 [num of rows-1] to be new indices

def make_predictions(data, predictors):
    train = data[data['date']< '2022-01-01'] #taking any data from before 2022 to put in training set
    test = data[data['date'] > '2022-01-01'] #taking any data from after 2021 to put in testing set
    rf.fit(train[predictors], train['target'])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test['target'], prediction=preds))
    precision = precision_score(test['target'], preds)
    return combined, precision

combined, precision = make_predictions(matches_rolling, predictors + new_cols) #original set of 4 cols and new predictors in new cols

print(combined) #score of 0.625

combined = combined.merge(matches_rolling[['date', 'team', 'opponent', 'result']], left_index=True, right_index=True) #merge based on the index to find which teams played

class MissingDict(dict): #to make names across data consistent (wolverhampton = wolves)
    __missing__ = lambda self, key: key

map_values = {
    'Brighton and Hove Albion': 'Brighton',
    'Manchester United': 'Man Utd',
    'Manchester City': 'Man City',  
    'Wolverhampton Wanderers': 'Wolves',
    'Tottenham Hotspur': 'Spurs',
    'Newcastle United': 'Newcastle',
    'West Ham United': 'West Ham'
}

mapping = MissingDict(**map_values)

combined['new_team'] = combined['team'].map(mapping)
merged = combined.merge(combined, left_on=['date', 'new_team'], right_on=['date', 'opponent']) #look for the new team and merge with the oppponent

# # merged[(merged['predicted_x'] == 1) & (merged['predicted_y'] == 0)]['actual_X'].value_counts()

# print(merged[(merged['prediction_x'] == 1) & (merged['prediction_y'] == 0)]['actual_x'].value_counts()) #score of 0.675
print(merged)

# # df = merged
# # df.to_csv('match_results.csv')