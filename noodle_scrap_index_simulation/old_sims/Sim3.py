#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 12:24:41 2018

@author: cronin
"""
#%%
import os
import pandas as pd
import numpy as np
import csv
import warnings
import glob
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_colwidth', -1) 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import os
from dateutil.relativedelta import *
from datetime import datetime, timedelta
import datetime as dt
import itertools
#%%
os.chdir('/Users/cronin/Desktop/Simulation')
df1 = pd.read_csv('spot_prices_sample.csv')
df2 = pd.read_csv('futures_data_sample.csv')
#%%
''' Metal Margin Plot '''
df1.plot(x=df1.index, y=["amm_busheling_midwest", "hr_coil"], kind="line")   
df1['metal_margin'] = df1['hr_coil'] - df1['amm_busheling_midwest']  
#%%
df1.metal_margin.plot()
#%%
''' Change Dtypes, Index '''
# Changed integers to floats, dates to datetime, reset index
df1['hr_coil'] = df1['hr_coil'].astype(np.float)
df1['date'] = pd.to_datetime(df1['date'])
df1.set_index(['date'], inplace=True)
df1.info()
# Changed dates to datetime, reset index
df2['Date'] = pd.to_datetime(df2['Date'])
df2.set_index(['Date'], inplace=True)
df2.info()
#%%
''' Drop Unnecessary Columns '''
# Drop Aluminum alloy - not in model anymore
#df1 = df1.drop('aluminium_alloy',1)
# delete front month + 15,16,17 b/c some nulls and far out
df2 = df2.drop(['Front Month + 15', 'Front Month + 16','Front Month + 17'], axis=1)
df1.isnull().sum()
df2.isnull().sum()
#%%
''' Fill in Busheling Nans for Now '''
# fill busheling nans with mean of row
df2 = df2.fillna(df2.mean())
df2.isnull().sum()
#%%
df2 = df2.reset_index().sort_values(by='Date').set_index(['Date','Commodity'])
df2 = df2.unstack()
df2 = df2.reindex(pd.date_range(start='2010-01-04',end='2018-08-21',freq='D'))
df2 = df2.bfill()
df2 = df2.stack()
df2 = df2.reset_index(level=1)
df2.index.name = 'Date'
df2 = df2.reset_index().sort_values(by=['Commodity','Date'])
df2 = df2.set_index(['Date'])
#%%
# create date range of simulation
dates = pd.date_range(start='2017-05-04', end ='2018-04-04', freq=pd.DateOffset(months=1))
#%%
busheling = df2[df2['Commodity'] == 'Busheling'].copy()
df2 = df2[~(df2['Commodity'] == 'Busheling')]

# Doesn't include sci_nue = 0.0

def multiplier(ser):
    commodity = ser['Commodity']
    if commodity == 'SGX Iron Ore':
        return 0.517
    if commodity == 'HRC':
        return 0.247
    if commodity == 'Turkish HMS':
        return 0.436
    if commodity == 'LME_US_Aluminum':
        return 0.032
    return 1

sci_cash_constant = -51.569
df2['multiplier'] = df2.apply(multiplier,axis=1)
cols = [x for x in df2.columns if x.startswith('Front Month')]

#df2[cols].mul(df2['multiplier'],axis=0) + sci_cash_constant
new_cols = [x + 'adjusted' for x in cols]
df2[new_cols] = df2[cols].mul(df2['multiplier'],axis=0) 
sci = df2[new_cols].groupby(df2.index).sum()+ sci_cash_constant

df2 = df2.drop(new_cols+['multiplier'], axis=1)
sci.columns = cols
sci['Commodity'] = 'SCI'
df1 = df1.join(sci['Front Month'],how='left')
df1 = df1.rename(columns={'Front Month': 'sci'})
df2 = pd.concat([df2,busheling,sci], axis=0)
#%%
df1_date_range = df1[df1.index.isin(dates)]
df2_date_range = df2[df2.index.isin(dates)]
#%%  
''' Hedge Functions - Long & Short ''' # Fixed without_hedge_long
def compute_hedge_long(contract_pounds,number_contracts,tone_spot_price,tone_futures_price,tzero_spot_price,tzero_futures_price):
    basis = tone_spot_price - tone_futures_price
    without_hedge_long = - contract_pounds * number_contracts * tone_spot_price
    cost_long = - contract_pounds * number_contracts * tone_spot_price
    futures_gain_per_pound_long = tone_futures_price - tzero_futures_price
    futures_gain_position_long = futures_gain_per_pound_long * contract_pounds * number_contracts
    net_cost_long = cost_long + futures_gain_position_long
    net_cost_locked_in_if_converged_long = - contract_pounds * number_contracts * tzero_futures_price
    return futures_gain_position_long,net_cost_long,net_cost_locked_in_if_converged_long,without_hedge_long
#%%
def compute_hedge_short(contract_pounds,number_contracts,tzero_spot_price,tzero_futures_price,tone_spot_price,tone_futures_price):
    basis = tone_spot_price - tone_futures_price
    without_hedge_short = contract_pounds * number_contracts * tzero_spot_price
    price_short = contract_pounds * number_contracts * tone_spot_price
    futures_gain_per_pound_short = tzero_futures_price - tone_futures_price
    futures_gain_position_short = futures_gain_per_pound_short * contract_pounds * number_contracts
    net_price_short = price_short + futures_gain_position_short
    net_cost_locked_in_if_converged_short = contract_pounds * number_contracts * tzero_futures_price
    return futures_gain_position_short,net_price_short,net_cost_locked_in_if_converged_short, without_hedge_short   
#%%
''' Function to get the numbers to be used in Hedge Functions '''
def get_data(date_0,commodity_code,mon):
    global df1_date_range,df2_date_range
    c_list= [['hr_coil','HRC'],
             ['amm_busheling_midwest','Busheling'],
              ['sci','SCI']]
    c1 = c_list[commodity_code][0]
    #print(c1)
    c2 = c_list[commodity_code][1]
    #print(c2)
    tzero_spot_price = df1_date_range[c1].loc[date_0]
    #print(tzero_spot_price)
    date_0_datetime = pd.to_datetime(date_0)
    date_n = date_0_datetime +relativedelta(months=+mon)
    date_n = date_n.strftime('%Y-%m-%d')
    c3 = 'Front Month + ' + str(mon).zfill(2)
    c4 = 'Front Month'
    tzero_futures_price = df2_date_range[c3][df2_date_range.Commodity == c2].loc[date_0]
    #print(tzero_futures_price)
    tone_spot_price = df1_date_range[c1].loc[date_n]
    #print(tone_spot_price)
    #c4 = 'Front Month'
    tone_futures_price = df2_date_range[c4][df2_date_range.Commodity == c2].loc[date_n]
    #print(tone_futures_price)
    return tzero_spot_price,tzero_futures_price,tone_spot_price,tone_futures_price
#%%
# Example
print(get_data('2017-05-04',2,3))
#%%
''' Contract Specifics:
    1 contrat equals 10,000 tons min standard for an order. 
    Would want steel daily volume. 150000 tons a steel of month at BRS. 
'''
contract_pounds = 10000
number_contracts = 15 
weights = [.4,.4,.2]
comps = 3
#%%
busheling_df = df2[df2['Commodity']=='Busheling']
num = str(comps).zfill(2)
hedges = busheling_df['Front Month'] - busheling_df[f'Front Month + {num}']
hedges = hedges.to_frame()
# to do - refactor for more than just Bushelings
hedges.columns = ['sentiment']
hedges['c_code'] = 1
hedges.index.name = 'date'
hedges = hedges.reset_index()
#%%
''' Short '''    
output = {}
for x,(c_code,weight) in itertools.product(list(dates)[:-comps],enumerate(weights)):
    tzero_spot_price,tzero_futures_price,tone_spot_price,tone_futures_price = get_data(x,c_code,comps)
    results = compute_hedge_short(contract_pounds * weight,number_contracts,tzero_spot_price,tzero_futures_price,tone_spot_price,tone_futures_price)
    output[(x,c_code,comps)] = list(results)
   
print(output)
result = pd.DataFrame.from_dict(output,orient='index')
mi = pd.MultiIndex.from_tuples(result.index,names=['date','c_code','offset'])
result.index = mi
result.columns = ['futures_gain_position_short','net_price_short','net_cost_locked_in_if_converged_short','without_hedge_short']
result = result.reset_index()
result = pd.merge(result,hedges,on=['date','c_code'],how='left')

for col in ['futures_gain_position_short','net_price_short','net_cost_locked_in_if_converged_short']:
    result[col] = np.where(result['sentiment'] < 0, 0, result[col])
# Futures Gain Position Short - how much $ won/loss from futures transaction
g = sns.FacetGrid(result,col ='c_code', col_wrap = 2, hue='c_code')
g = g.map(plt.plot,'date','futures_gain_position_short')
# Net Price Short = 
h = sns.FacetGrid(result,col ='c_code', col_wrap = 2, hue='c_code')
h = h.map(plt.plot,'date','net_price_short')

i = sns.FacetGrid(result,col ='c_code', col_wrap = 2, hue='c_code')
i = i.map(plt.plot,'date','net_cost_locked_in_if_converged_short')

j = sns.FacetGrid(result,col ='c_code', col_wrap = 2, hue='c_code')
j = j.map(plt.plot,'date','without_hedge_short')
#%%
''' Long '''
output2 = {}
for x,(c_code,weight) in itertools.product(list(dates)[:-comps],enumerate(weights)):
    tzero_spot_price,tzero_futures_price,tone_spot_price,tone_futures_price = get_data(x,c_code,comps)
    results2 = compute_hedge_long(contract_pounds * weight,number_contracts,tzero_spot_price,tzero_futures_price,tone_spot_price,tone_futures_price)
    output2[(x,c_code,comps)] = list(results2)
   
#print(output2)
result2 = pd.DataFrame.from_dict(output2,orient='index')
mi2 = pd.MultiIndex.from_tuples(result2.index,names=['date','c_code','offset'])
result2.index = mi2
result2.columns = ['futures_gain_position_long','net_cost_long','net_cost_locked_in_if_converged_long','without_hedge_long']
result2 = result2.reset_index()
result2 = pd.merge(result2,hedges,on=['date','c_code'],how='left')

for col in ['futures_gain_position_long','net_cost_long','net_cost_locked_in_if_converged_long','without_hedge_long']:
    result2[col] = np.where(result2['sentiment'] < 0, 0, result2[col])
# Futures Gain Position Short - how much $ won/loss from futures transaction
g2 = sns.FacetGrid(result2,col ='c_code', col_wrap = 2, hue='c_code')
g2 = g2.map(plt.plot,'date','futures_gain_position_long')
# Net Price Short = 
h2 = sns.FacetGrid(result2,col ='c_code', col_wrap = 2, hue='c_code')
h2 = h2.map(plt.plot,'date','net_cost_long')

i2 = sns.FacetGrid(result2,col ='c_code', col_wrap = 2, hue='c_code')
i2 = i2.map(plt.plot,'date','net_cost_locked_in_if_converged_long')

j2 = sns.FacetGrid(result2,col ='c_code', col_wrap = 2, hue='c_code')
j2 = j2.map(plt.plot,'date','without_hedge_long')
#%%
col_list = ['futures_gain_position_long','net_cost_long','net_cost_locked_in_if_converged_long','without_hedge_long']
c_list =['hrc','busheling','sci']


for col in col_list:
    for c in range(1,3):
        y=result2[result2.c_code == c]
        plt.plot(y.date,y[col],label = c_list[c])
    plt.xticks(rotation=90)    
    plt.legend()
    plt.title(col)
    fname = col+'.png'
    plt.ticklabel_format(style='plain', axis='y')
    plt.savefig('/Users/cronin/Desktop/Simulation/'+fname,bbox_inches='tight')
    plt.show()

for col in col_list:
    for c in range(1,3):
        y=result2[result2.c_code == c]
        y_cum = y[col].cumsum()
        plt.plot(y.date,y_cum,label = c_list[c])
    plt.xticks(rotation=90)
    plt.legend()
    plt.title(col)
    fname = col+'_cum.png'
    plt.ticklabel_format(style='plain', axis='y')
    plt.savefig('/Users/cronin/Desktop/Simulation/'+fname,bbox_inches='tight')
    plt.show()  
#%%
result2.sum()
#result2.groupby('c_code')['b'].sum()[1]
result2.groupby('c_code').sum()
#%%
''' Busheling Forecast Simulation '''
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas.tools.plotting import autocorrelation_plot

# fit model
col = 'amm_busheling'
start_date = '2016-05-14'
end_date = '2017-08-14'
date_range = df1.index[(df1.index >= start_date) & (df1.index <= end_date)]
series = df1[col][df1.index.isin(date_range)]
series.plot()
autocorrelation_plot(series)
pyplot.show()

print(type(series.iloc[0]))
series = series.astype(float)
model = ARIMA(series, order=(15,0,5)) # 5 day lag, look 5 days ahead and find correlation, 10 is what you are averaging over instead of day by day
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())                        
#%%                        