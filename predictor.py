import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

results = pd.read_csv('results.csv', index_col=0)

results['date'] = pd.to_datetime(results['Date'])

#predictors
results['xG Home'] = results['xG-H']
results['xG Away'] = results['xG-A']
results['stad'] = results['Venue'].astype('category').cat.codes
results['ref'] = results['Referee'].astype('category').cat.codes
results['opp'] = results['Opponent'].astype('category').cat.codes
results['day_code'] = results['date'].dt.day_of_week
results['hour'] = results['Time'].str.replace(':.+', '', regex=True).astype('int')

results['target'] = (results['Result'] == 'W').astype('int')

# print(results)

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
train = results[results['date'] < '2024-03-01'] #train on games before March 2024
test = results[results['date'] > '2024-03-01'] #test on games after March 2024

predictors = ['xG Home', 'xG Away', 'stad', 'ref', 'opp', 'day_code', 'hour']

rf.fit(train[predictors], train['target'])

preds = rf.predict(test[predictors])
# acc = accuracy_score(test['target'], preds)
# print(acc) #score of 0.66

combined = pd.DataFrame(dict(actual=test['target'], prediction=preds)) #combines actual values with predicted values
# print(pd.crosstab(index=combined['actual'], columns=['prediction']))
# print(precision_score(test['target'], preds)) #score of 0.68

grouped_matches = results.groupby('Team') #creates a dataframe for every team
group = grouped_matches.get_group('Real Madrid')

def rolling_averages(group, cols, new_cols): #function which gets the given stats from the previous 3 games to help predict result in a 4th game
    group = group.sort_values('date')
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols) 
    return group

cols = ['xG-H', 'xG-A']
new_cols = [f'{c}_rolling' for c in cols]

matches_rolling = results.groupby('Team').apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('Team')

matches_rolling.index = range(matches_rolling.shape[0]) 


def make_predictions(data, predictors):
    train = data[data['date'] < '2024-03-01']
    test = data[data['date'] > '2024-03-01'] 
    rf.fit(train[predictors], train['target'])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test['target'], prediction=preds))
    precision = precision_score(test['target'], preds)
    return combined, precision

combined, precision = make_predictions(matches_rolling, predictors + new_cols) #original set of cols and new predictors in new cols

combined = combined.merge(matches_rolling[['date', 'Team', 'Opponent', 'Result']], left_index=True, right_index=True) #merge based on the index to find which teams played

combined['new_team'] = combined['Team']
print(combined)
print(precision) #score of 0.78
# combined.to_csv('combined.csv')

#Final model accuracy score of 0.78