# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:19:26 2015
@author: Cronin
"""
import sys
reload(sys)  
sys.setdefaultencoding('utf8')
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

pd.set_option('display.float_format', lambda x: '%.3f' % x)

a = pd.read_csv('../data/OpusData/movie_summary.csv', error_bad_lines=False)
b = pd.read_csv('../data/OpusData/acting_credits.csv', error_bad_lines=False)
c = pd.read_csv('../data/OpusData/technical_credits.csv', error_bad_lines=False)
d = pd.read_csv('../data/OpusData/movie_ratings.csv', error_bad_lines=False)
e = pd.read_csv('../data/OpusData/movie_production_companies.csv', error_bad_lines=False)
f = pd.read_csv('../data/OpusData/movie_releases.csv', error_bad_lines=False)
g = pd.read_csv('../data/OpusData/movie_keywords.csv', error_bad_lines=False)

# Convert Dataframes into Strings
a['odid'] = a['odid'].astype(str)
b['odid'] = b['odid'].astype(str)
c['odid'] = c['odid'].astype(str)
d['odid'] = d['odid'].astype(str)
e['odid'] = e['odid'].astype(str)
f['odid'] = f['odid'].astype(str)
g['odid'] = g['odid'].astype(str)

# Modify Tables
a['money'] = a.international_box_office + a.inflation_adjusted_domestic_box_office
a['multiple'] = a.money / a.production_budget 
a['multiple_over_5x'] = np.where(a.multiple >= 5, 1, 0)
a['runtimesquared'] = a['running_time'].apply(lambda x: x * x)
a['profit'] = a.money - a.production_budget

def get_pbudget_bin(val):
    if val < 25000000:
        return 'bin_25'
    elif val < 50000000:
        return 'bin_50'
    elif val <  100000000:
        return 'bin_100'
    elif val <  150000000:
        return 'bin_150'
    elif val <  200000000:
        return 'bin_200'        
    elif val <  250000000:
        return 'bin_250' 
    elif val <  300000000:
        return 'bin_300'
    elif val <  350000000:
        return 'bin_350'
    return 'bin_350+'

a['pbudget_bin'] = a.production_budget.map(get_pbudget_bin)

c.rename(columns={'person':'name'}, inplace=True)

# Sort/Filter Tables
counts = pd.DataFrame(b[b['type'] == 'Leading']['person'].value_counts().reset_index())
counts.columns = ['person','count']
more_than_five = counts[counts['count'] > 5]

counts2 = pd.DataFrame(c[c['role'] == 'Director']['name'].value_counts().reset_index())
counts2.columns = ['name','count']
more_than_5 = counts2[counts2['count'] > 5]

counts3 = pd.DataFrame(e['production_company1'].value_counts().reset_index())
counts3.columns = ['production_company1','count']
more_than_10 = counts3[counts3['count'] > 10]

counts4 = pd.DataFrame(e['production_company2'].value_counts().reset_index())
counts4.columns = ['production_company2','count']
more_than_ten = counts4[counts4['count'] > 10]

counts5 = pd.DataFrame(f['distributor'].value_counts().reset_index())
counts5.columns = ['distributor','count']
more_than_fifty = counts5[counts5['count'] > 30]

counts6 = pd.DataFrame(g['keywords1'].value_counts().reset_index())
counts6.columns = ['keywords1','count']
more_than_11 = counts6[counts6['count'] > 11]

counts7 = pd.DataFrame(g['keywords2'].value_counts().reset_index())
counts7.columns = ['keywords2','count']
more_than_9 = counts7[counts7['count'] > 9]

counts8 = pd.DataFrame(g['keywords3'].value_counts().reset_index())
counts8.columns = ['keywords3','count']
more_than_7 = counts8[counts8['count'] > 7]

#make Final
final = a.copy()

# Merge Keywords 1
keywords1 = g[g['keywords1'].isin(more_than_11['keywords1'])]
keywords1 = keywords1[['odid','keywords1']].set_index('odid')
keywords1_dummies = pd.get_dummies(keywords1).reset_index()
keywords1_dummies = keywords1_dummies.groupby('odid').sum()
final = pd.merge(final,keywords1_dummies,left_on='odid',right_index=True,how='left').fillna(0)

# Merge Keywords 2
keywords2 = g[g['keywords2'].isin(more_than_9['keywords2'])]
keywords2 = keywords2[['odid','keywords2']].set_index('odid')
keywords2_dummies = pd.get_dummies(keywords2).reset_index()
keywords2_dummies = keywords2_dummies.groupby('odid').sum()
final = pd.merge(final,keywords2_dummies,left_on='odid',right_index=True,how='left').fillna(0)

# Merge Keywords 3
keywords3 = g[g['keywords3'].isin(more_than_7['keywords3'])]
keywords3 = keywords3[['odid','keywords3']].set_index('odid')
keywords3_dummies = pd.get_dummies(keywords3).reset_index()
keywords3_dummies = keywords3_dummies.groupby('odid').sum()
final = pd.merge(final,keywords3_dummies,left_on='odid',right_index=True,how='left').fillna(0)

# Merge release dates
movie_releases_date = f[f['release_date'].isin(f['release_date'])]
movie_releases_date = f[['odid','release_date']]
movie_releases_date['release_date'] = pd.to_datetime(movie_releases_date['release_date'])
movie_releases_date = movie_releases_date.sort('release_date')
movie_releases_date.drop_duplicates('odid',inplace=True)
final = pd.merge(final,movie_releases_date,left_on='odid',right_on='odid',how='left').fillna(0)
final['release_date'] = pd.to_datetime(final['release_date'])
final['day_of_year'] = final.release_date.dt.dayofyear

# merge movie releases
movie_releases = f[f['release_pattern'].isin(f['release_pattern'])]
movie_releases = f[['odid','release_pattern']].set_index('odid')
movie_releases_dummies = pd.get_dummies(movie_releases).reset_index()
movie_releases_dummies = movie_releases_dummies.groupby('odid').sum()
final = pd.merge(final,movie_releases_dummies,left_on='odid',right_index=True,how='left').fillna(0)
movie_releases_cols = list(movie_releases.release_pattern.unique())

# Merge Ratings
movie_ratings = d[d['ratings'].isin(d['ratings'])]
movie_ratings = d[['odid','ratings']]
final = pd.merge(final,movie_ratings,left_on='odid',right_on='odid',how='left').fillna(0)

rating_dummies = pd.get_dummies(final.ratings)
final = pd.merge(final,rating_dummies,left_index=True,right_index=True,how='left')
ratings_features = list(final.ratings.unique())

# Merge Distributor Companies
distributorcol = f[f['distributor'].isin(more_than_fifty['distributor'])]
distributorcol = distributorcol[['odid','distributor']].set_index('odid')
distributor_dummies = pd.get_dummies(distributorcol).reset_index()
distributor_dummies = distributor_dummies.groupby('odid').sum()
final = pd.merge(final,distributor_dummies,left_on='odid',right_index=True,how='left').fillna(0)

# Merge Production Company 1
production1 = e[e['production_company1'].isin(more_than_10['production_company1'])]
production1 = production1[['odid','production_company1']].set_index('odid')
production1_dummies = pd.get_dummies(production1).reset_index()
production1_dummies = production1_dummies.groupby('odid').sum()
final = pd.merge(final,production1_dummies,left_on='odid',right_index=True,how='left').fillna(0)
final = pd.merge(final,production1,left_on='odid',right_index=True,how='left').fillna('')

# Merge Production Company 2
production2 = e[e['production_company2'].isin(more_than_ten['production_company2'])]
production2 = production2[['odid','production_company2']].set_index('odid')
production2_dummies = pd.get_dummies(production2).reset_index()
production2_dummies = production2_dummies.groupby('odid').sum()
final = pd.merge(final,production2_dummies,left_on='odid',right_index=True,how='left').fillna(0)
final = pd.merge(final,production2,left_on='odid',right_index=True,how='left').fillna('')

#Merge in Director Dummies
directors = c[c['name'].isin(more_than_5['name'])]
directors = directors[['odid','name']].set_index('odid')
director_dummies = pd.get_dummies(directors).reset_index()
director_dummies = director_dummies.groupby('odid').sum()
final = pd.merge(final,director_dummies,left_on='odid',right_index=True,how='left').fillna(0)

#Merge in actor Dummies
actors = b[b['person'].isin(more_than_five['person'])]
actors = actors[['odid','person']].set_index('odid')
actor_dummies = pd.get_dummies(actors).reset_index()
actor_dummies = actor_dummies.groupby('odid').sum()
final = pd.merge(final,actor_dummies,left_on='odid',right_index=True,how='left').fillna(0)

#Merge in source Dummies
source_dummies = pd.get_dummies(a[['odid','source']].set_index('odid'))
final = pd.merge(final,source_dummies,left_on='odid',right_index=True,how='left')

#Merge in genre dummies
genre_dummies = pd.get_dummies(a[['odid','genre']].set_index('odid'))
final = pd.merge(final,genre_dummies,left_on='odid',right_index=True,how='left')

#Merge in creative_type dummies
creative_type_dummies = pd.get_dummies(a[['odid','creative_type']].set_index('odid'))
final = pd.merge(final,creative_type_dummies,left_on='odid',right_index=True,how='left')

#Merge in production_method dummies
production_method_dummies = pd.get_dummies(a[['odid','production_method']].set_index('odid'))
final = pd.merge(final,production_method_dummies,left_on='odid',right_index=True,how='left')

#Get rid of ZERO budget movies
final = final[final['production_budget'] != 0]

#Create date months column
def getMonth(timestamp):
    return timestamp.month
final['release_month'] = final.release_date.map(getMonth)

def is_christmas(dayofyear):
    return dayofyear == 359 or dayofyear == 360
        
final['christmas_release'] = final.release_date.dt.dayofyear.apply(is_christmas)

#View columns in python and excel
final.columns.values.tolist()
pd.Series(final.columns).to_csv('/Users/Cronin/Desktop/GA/SF_DAT_15/data/OpusData/movie_columns.csv')
final_columns = list(final.columns)

# Charts/Visualizations/Playing with data
top25 = final.sort_index(by='profit', ascending = False)[['display_name','profit']].head(25)
top25.to_csv('/Users/Cronin/Desktop/GA/SF_DAT_15/data/OpusData/top25.csv')
bottom25 = final.sort_index(by='profit', ascending = False)[['display_name','profit']].tail(25)
bottom25.to_csv('/Users/Cronin/Desktop/GA/SF_DAT_15/data/OpusData/bottom25.csv')

final.groupby('production_company1').profit.sum()
budget_sums = final.groupby(['production_company1', 'pbudget_bin']).profit.sum()
budget_percents = budget_sums.groupby(level=0).apply(lambda x: 100*x/float(x.sum()))
budget_percents.to_csv('/Users/Cronin/Desktop/GA/SF_DAT_15/data/OpusData/budget_bin.csv')

final.groupby('sequel').sequel.count() 
final.groupby('sequel').money.mean()

final.groupby('genre').genre.count()
final.groupby('ratings').ratings.count()
final.groupby('multiple_over_5x').display_name.count()
final.groupby('creative_type').creative_type.count()

production_means = final.groupby('production_year').display_name.count()
ax = production_means.plot(kind = 'bar',title='Movies Produced Per Year')
ax.set_xlabel('Year')
ax.set_ylabel('# of Movies')

production_means = final.groupby('production_year').production_budget.mean()
ax = production_means.plot(kind = 'bar',title='Avg Production Budget Per Year')
ax.set_xlabel('Year')
ax.set_ylabel('Production Budget')

money_means = final.groupby('production_year').money.mean()
ax = money_means.plot(kind = 'bar',title='Avg Money Made Per Year')
ax.set_xlabel('Year')
ax.set_ylabel('Money')

#Create function for certain features
source_features = [col for col in final_columns if 'source_' in str(col)]
genre_features = [col for col in final_columns if 'genre_' in str(col)]
actor_features = [col for col in final_columns if 'person_' in str(col)]
director_features = [col for col in final_columns if 'name_' in str(col)]
creative_type_features = [col for col in final_columns if 'creative_type_' in str(col)]
release_pattern = [col for col in final_columns if 'release_pattern_' in str(col)]
production_method_features = [col for col in final_columns if 'production_method_' in str(col)]
production1_features = [col for col in final_columns if 'production_company1_' in str(col)]
production2_features = [col for col in final_columns if 'production_company2_' in str(col)]
distributor_features = [col for col in final_columns if 'distributor_' in str(col)]
keywords1_features = [col for col in final_columns if 'keywords1_' in str(col)]
keywords2_features = [col for col in final_columns if 'keywords2_' in str(col)]
keywords3_features = [col for col in final_columns if 'keywords3_' in str(col)]

feature_cols = source_features + genre_features + actor_features[1:] \
+ director_features[1:] + creative_type_features[1:] + production_method_features[1:] + ratings_features[1:] \
+ production1_features[1:] + production2_features[1:] + distributor_features[1:] + release_pattern[1:] \
+ keywords1_features[1:] + keywords2_features[1:] + keywords3_features[1:]\
+ ['runtimesquared','sequel','production_budget','christmas_release','release_month']

feature_cols = [col for col in feature_cols if col != 0]

# create X and y
X = final[feature_cols]
y = final.money

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg_results = lin_reg.fit(X, y)

predictions = lin_reg.predict(X)
len(predictions)
final['predictions'] = predictions
final['predictions'] = final['predictions'].astype(int)

#Plot Predictions vs Fit
fig, ax = plt.subplots(1, 1)
ax.scatter(final['money'],final['predictions'])
ax.set_xlabel('Money')
ax.set_ylabel('Predictions')
ax.axis('equal')

# instantiate and fit
lin_reg.intercept_
lin_reg.coef_

final[['money','predictions']]
list(final.columns) 
zip(feature_cols, lin_reg.coef_)

# R Squared - Scikit Learn
metrics.r2_score(final['money'], final['predictions'])
# % of variance in the observed data is explained by the linear regression model. 
# MAE 
metrics.mean_absolute_error(final['money'], final['predictions'])
# MSE 
np.sqrt(metrics.mean_squared_error(final['money'], final['predictions']))

# Stats Model & P-Values
import statsmodels.formula.api as smf

lm = smf.OLS(np.array(y), np.array(X))
results = lm.fit()
results.summary()
results.params
results.rsquared
results.conf_int()
results.pvalues

#Sort by best pvalue features dataFrame
p = pd.DataFrame({'p_value':results.pvalues, 'feature':feature_cols})
p
relevant_features = list(p[p.p_value < .05]['feature'])
relevant_features

# create X and y
X_P = final[relevant_features]
y = final.money

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg_results = lin_reg.fit(X_P, y)
predictions2 = lin_reg.predict(X_P)
len(predictions2) #3525
final['predictions2'] = predictions2
final['predictions2'] = final['predictions2'].astype(int)

#Plot Predictions vs Fit
fig, ax = plt.subplots(1, 1)
ax.scatter(final['money'],final['predictions2'])
ax.axis('equal')

# instantiate and fit
lin_reg.intercept_
lin_reg.coef_

final[['money','predictions2']]
zip(relevant_features, lin_reg.coef_)

# R Squared - Scikit Learn
metrics.r2_score(final['money'], final['predictions2'])
metrics.mean_absolute_error(final['money'], final['predictions2'])

# MSE (Optional)
np.sqrt(metrics.mean_squared_error(final['money'], final['predictions2']))

from sklearn import tree
X_test = final[feature_cols]

# Add in Decision Tree model
# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1)

# Create a decision tree classifier instance (start out with a small tree for interpretability)
ctree = tree.DecisionTreeRegressor(random_state=1)

# Fit the decision tree classifier
ctree.fit(X_train, y_train)

# Create a feature vector
features = X_train.columns.tolist()

features

# Predict 
predictions_tree = ctree.predict(X_test)

# R Squared - Scikit Learn
metrics.r2_score(y_test, predictions_tree)
metrics.mean_absolute_error(y_test, predictions_tree)
np.sqrt(metrics.mean_squared_error(y_test, predictions_tree))

# Which features are the most important?
# Clean up the output. # will add up to 1. Think %
pd.DataFrame(zip(list(X.columns), ctree.feature_importances_)).sort_index(by=1, ascending=False)

# Add in random forest
from sklearn.ensemble import RandomForestRegressor
rfclf = RandomForestRegressor(n_estimators=100, max_features='auto', oob_score=True, random_state=1)
rfclf.fit(X_train, y_train)

predictions_tree = rfclf.predict(X_test)
metrics.r2_score(y_test, predictions_tree)
metrics.mean_absolute_error(y_test, predictions_tree)
np.sqrt(metrics.mean_squared_error(y_test, predictions_tree))

# compute the feature importances
rf_feature_imp = pd.DataFrame(zip(list(X.columns), rfclf.feature_importances_)).sort_index(by=1, ascending=False)


X_sub_features = final[['production_budget', 'runtimesquared', 'release_pattern_IMAX']]

X_train, X_test, y_train, y_test = train_test_split(X_sub_features,y, random_state=1)
rfclf = RandomForestRegressor(n_estimators=100, max_features='auto', oob_score=True, random_state=1)
rfclf.fit(X_train, y_train)

predictions_tree = rfclf.predict(X_test)
metrics.r2_score(y_test, predictions_tree)

# To Predict a Movie Revenue
rfclf.predict([80000000, 10404, 1])[0]
rfclf.feature_importances_
