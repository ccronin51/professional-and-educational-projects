#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 17:12:04 2018

@author: cronin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 17:27:31 2018

@author: chriscronin
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
os.chdir('/Users/cronin/Desktop/Simulation/')
df1 = pd.read_csv('spot_prices_sample.csv')
df2 = pd.read_csv('futures_data_sample.csv')
#%%
''' Metal Margin Plot '''
df1.plot(x=df1.index, y=["amm_busheling_midwest", "hr_coil"], kind="line")  
df1['metal_margin'] = df1['hr_coil'] - df1['amm_busheling_midwest']  
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
''' Futures Metal Margin '''
bush = df2.loc[(df2['Commodity'] == 'Busheling')].copy()
bush = bush.drop('Commodity',1)
hrc = df2.loc[(df2['Commodity'] == 'HRC')].copy()
hrc = hrc.drop('Commodity',1)
metal = hrc.subtract(bush, axis='columns', level=None, fill_value=None)
metal['Commodity'] = 'metal_margin'
metal = metal[['Commodity','Front Month', 'Front Month + 01', 'Front Month + 02','Front Month + 03', 'Front Month + 04', 'Front Month + 05','Front Month + 06', 'Front Month + 07', 'Front Month + 08','Front Month + 09', 'Front Month + 10', 'Front Month + 11','Front Month + 12', 'Front Month + 13', 'Front Month + 14']]
df2 = pd.concat([df2,metal], axis=0)
#%%
''' Scrap Index Composite (SCI) Calculation '''
busheling = df2[df2['Commodity'] == 'Busheling'].copy()
df2 = df2[~(df2['Commodity'] == 'Busheling')]

# Doesn't include sci_nue = 0.0
def multiplier(ser):
    commodity = ser['Commodity']
    if commodity == 'SGX_Iron_Ore':
        return 0.517
    if commodity == 'HRC':
        return 0.247
    if commodity == 'Turkish_HMS':
        return 0.436
    if commodity == 'LME_US_Aluminum':
        return 0.032
    return 1

sci_cash_constant = -51.569
df2['multiplier'] = df2.apply(multiplier,axis=1)
cols = [x for x in df2.columns if x.startswith('Front Month')]

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
''' Hedge Functions - Long & Short '''
def compute_hedge_long(total_bought_per_month,
                       percent_hedged,
                       contract_pounds,
                       tzero_spot_price,tzero_futures_price,tone_spot_price,
                       tone_futures_price):
    number_contracts = (total_bought_per_month * percent_hedged) / contract_pounds
    unhedge_contracts = total_hedged_per_month * (1-percent_hedged) / contract_pounds
    basis = tone_spot_price - tone_futures_price
    cost_long = - contract_pounds * number_contracts * tone_spot_price
    futures_gain_per_pound_long = tone_futures_price - tzero_futures_price
    futures_gain_position_long = futures_gain_per_pound_long * contract_pounds * number_contracts
    net_cost_long = cost_long + futures_gain_position_long
    net_net_cost_long = - contract_pounds * unhedge_contracts * tone_spot_price   
    net_cost_locked_in_if_converged_long = - contract_pounds * number_contracts * tzero_futures_price
    return futures_gain_position_long,net_cost_long,net_cost_locked_in_if_converged_long, cost_long, net_net_cost_long
#%%
def compute_hedge_short(contract_pounds,number_contracts,tzero_spot_price,tzero_futures_price,tone_spot_price,tone_futures_price):
    basis = tone_spot_price - tone_futures_price
    price_short = contract_pounds * number_contracts * tone_spot_price
    futures_gain_per_pound_short = tzero_futures_price - tone_futures_price
    futures_gain_position_short = futures_gain_per_pound_short * contract_pounds * number_contracts
    net_price_short = price_short + futures_gain_position_short
    net_cost_locked_in_if_converged_short = contract_pounds * number_contracts * tzero_futures_price
    return futures_gain_position_short,net_price_short,net_cost_locked_in_if_converged_short, price_short   
#%%
print(compute_hedge_long(25000,1,2.50,2.70,2.70,2.64))    
#%%
''' Function to get the numbers to be used in Hedge Functions '''
def get_data(date_0,commodity_code,mon):
    global df1_date_range,df2_date_range
    c_list= [['hr_coil','HRC'],
             ['amm_busheling_midwest','Busheling'],
              ['sci','SCI']]
    c1 = c_list[commodity_code][0]
    c2 = c_list[commodity_code][1]
    tzero_spot_price = df1_date_range[c1].loc[date_0]
    date_0_datetime = pd.to_datetime(date_0)
    date_n = date_0_datetime +relativedelta(months=+mon)
    date_n = date_n.strftime('%Y-%m-%d')
    c3 = 'Front Month + ' + str(mon).zfill(2)
    c4 = 'Front Month'
    tzero_futures_price = df2_date_range[c3][df2_date_range.Commodity == c2].loc[date_0]
    tone_spot_price = df1_date_range[c1].loc[date_n]
    tone_futures_price = df2_date_range[c4][df2_date_range.Commodity == c2].loc[date_n]
    return tzero_spot_price,tzero_futures_price,tone_spot_price,tone_futures_price
#%%
# Example to see if pulling correct numbers
print(get_data('2017-05-04',2,3))
#%%
''' This shows that SCI isn't currenlty matching Busheling '''
df1.plot(x=df1.index, y=["amm_busheling_midwest", "sci"], kind="line") 
#%%
''' Simulation Setup:
    Contract Specifics:
    1 contrat equals 10,000 tons min standard for an order. 
    Would want steel daily volume. 150,000 tons a steel ordered/produced a month at BRS. 
'''
#  cost_long = - contract_pounds * number_contracts * tone_spot_price
#  cost_long = - contract_pounds * (1 - number_contracts2) * tone_spot_price

total_bought_per_month = 150000

percent_hedged = .5 # [25,50,75]

contract_pounds = 10000

weights = [.00,.50,.50]

months = 2 # months out hedging


#unhedged_crap = (1 - percent_hedged) * total_bought_per_month
#%%
''' Contango or Backwardation - Sentiment '''
# Busheling
busheling_df = df2[df2['Commodity']=='Busheling']
num = str(months).zfill(2)
busheling_delta = busheling_df[f'Front Month + {num}'] - busheling_df['Front Month']
busheling_delta = busheling_delta.to_frame()
busheling_delta.columns = ['busheling_delta']
busheling_delta['c_code'] = 1
busheling_delta.index.name = 'date'
busheling_delta = busheling_delta.reset_index()
#%%
# SCI
sci_df = df2[df2['Commodity']=='SCI']
num = str(months).zfill(2)
sci_delta = sci_df[f'Front Month + {num}'] - sci_df['Front Month'] 
sci_delta = sci_delta .to_frame()
sci_delta.columns = ['sentiment_sci']
sci_delta['c_code'] = 2
sci_delta.index.name = 'date'
sci_delta = sci_delta.reset_index()
#%%
# HRC
hrc_df = df2[df2['Commodity']=='HRC']
num = str(months).zfill(2)
hrc_delta = hrc_df[f'Front Month + {num}'] - hrc_df['Front Month']
hrc_delta = hrc_delta.to_frame()
hrc_delta.columns = ['sentiment_hrc']
hrc_delta['c_code'] = 0
hrc_delta.index.name = 'date'
hrc_delta = hrc_delta.reset_index()
#%%
metal_df = df2[df2['Commodity']=='metal_margin']
num = str(months).zfill(2)
metal_delta = metal_df[f'Front Month + {num}'] - metal_df['Front Month']
metal_delta = metal_delta.to_frame()
metal_delta.columns = ['sentiment_metal']
#metal_delta['c_code'] = 0
metal_delta.index.name = 'date'
metal_delta = metal_delta.reset_index()
#%%
''' Long - Simulation '''
output2 = {}
for x,(c_code,weight) in itertools.product(list(dates)[:-months],enumerate(weights)):
    tzero_spot_price,tzero_futures_price,tone_spot_price,tone_futures_price = get_data(x,c_code,months)
    try:
        results2 = compute_hedge_long(total_bought_per_month,percent_hedged,contract_pounds * weight,tzero_spot_price,tzero_futures_price,tone_spot_price,tone_futures_price)
        output2[(x,c_code,months)] = list(results2)
    except ZeroDivisionError: 
        continue
    
#%% Metal Margin Function Attempt 

df1_date_range['spot_margin'] = np.where(df1_date_range['metal_margin'] > 450, "High", "Low") 
df1_date_range['spot_margin2'] = pd.cut(df1_date_range['metal_margin'], bins=[0,450,9999], labels = ['low','high'])

spread = df2_date_range[df2_date_range['Commodity'] == 'metal_margin']
spread['range'] = np.where(spread['Front Month + 02'] > 300, 'Lock_in', 'Pass')        

output2 = {}
for x,(c_code,weight) in itertools.product(list(dates)[:-months],enumerate(weights)):
    tzero_spot_price,tzero_futures_price,tone_spot_price,tone_futures_price = get_data(x,c_code,months)
    if spread.loc[x,'range'] == 'Lock_in':
        try:
            if (df1_date_range.loc[x,'spot_margin2'] == 'high' and c_code == 0) or (
                    df1_date_range.loc[x,'spot_margin2'] == 'low' and c_code == 1):
                metal_spread = compute_hedge_short(total_bought_per_month,percent_hedged,
                                          contract_pounds * weight,
                                          tzero_spot_price,
                                          tzero_futures_price,
                                          tone_spot_price,tone_futures_price)
            else:        
                metal_spread = compute_hedge_long(total_bought_per_month,
                                          percent_hedged,contract_pounds * weight,
                                          tzero_spot_price,tzero_futures_price,
                                          tone_spot_price,
                                          tone_futures_price) # fix so five values
        except ZeroDivisionError: 
            continue
    else: 
        metal_spread = [np.nan] * 5 # To Do
    output2[(x,c_code,months)] = list(metal_spread)

# To do for sure to fix short hedge short, and np.nan above (at least think about)
# High Metal Margin
# Short HRC @ Future Price
# Long Busheling @ Future Price
# Only if HRC FP - Busheling FP is greater than X        
        
# Low Metal Margin
# Long HRC @ Future Price
# Short Busheling @ Future Price
# Only if HRC FP - Busheling FP is greater than X     

# 2017-05-04 
# Spot Metal Margin 846 ==> High
# Future Metal Margin 272 ==> Low & will not hedge
    
#%%    
#print(output2)
result2 = pd.DataFrame.from_dict(output2,orient='index')
mi2 = pd.MultiIndex.from_tuples(result2.index,names=['date','c_code','offset'])
result2.index = mi2
''' Take out long or short for metal margin spread func '''
result2.columns = ['futures_gain_position_long','net_cost_long','net_cost_locked_in_if_converged_long','cost_long','net_net_cost_long']
result2 = result2.reset_index()
result2 = pd.merge(result2,busheling_delta,on=['date','c_code'],how='left')
result2 = pd.merge(result2,sci_delta,on=['date','c_code'],how='left')
result2 = pd.merge(result2,hrc_delta,on=['date','c_code'],how='left')
result2 = pd.merge(result2,metal_delta,on=['date'],how='left')
result2['total_cost'] = result2['net_cost_long'] + result2['net_net_cost_long']
#%%
# Rule to go long or not
for col in ['futures_gain_position_long','net_cost_long','net_cost_locked_in_if_converged_long']:
    result2[col] = np.where(result2['sentiment_bush'] < 0, 0, result2[col]) #EDIT
    
for col in ['futures_gain_position_long','net_cost_long','net_cost_locked_in_if_converged_long']:
    result2[col] = np.where(result2['sentiment_bush'] < 0, result2['cost_long'], result2[col]) #EDIT    
#%% 
''' Long Separate Plots '''      
# Futures Gain Position Long - how much $ won/loss from just futures transaction
g2 = sns.FacetGrid(result2,col ='c_code', col_wrap = 2, hue='c_code')
g2 = g2.map(plt.plot,'date','futures_gain_position_long')
# Net Cost Long = cost_long + futures_gain_position_long # Price you are locking in but can be different if tone spot/future prices don't converge
h2 = sns.FacetGrid(result2,col ='c_code', col_wrap = 2, hue='c_code')
h2 = h2.map(plt.plot,'date','net_cost_long')
# Net_cost_locked_in_if_converged_long = - contract_pounds * number_contracts * tzero_futures_price # Price you are locking in
i2 = sns.FacetGrid(result2,col ='c_code', col_wrap = 2, hue='c_code')
i2 = i2.map(plt.plot,'date','net_cost_locked_in_if_converged_long')
# Cost_Long = - contract_pounds * number_contracts * tone_spot_price # how much it would cost w/o hedging
j2 = sns.FacetGrid(result2,col ='c_code', col_wrap = 2, hue='c_code')
j2 = j2.map(plt.plot,'date','cost_long')
#%% 
col_list = ['futures_gain_position_long','net_cost_long','net_cost_locked_in_if_converged_long','cost_long','total_cost'] #EDIT
c_list =['hrc','busheling','sci']

result2.groupby(['c_code']).plot()        

c_dict = {
        0:'HRC',
        1:'Busheling',
        2:'SCI'}

for c in range(1,3):
    y=result2[result2.c_code == c]
    y.plot(x=y.date,y=['net_cost_long','cost_long'],title=c_dict[c])


plt.xticks(rotation=90)    
plt.legend()
    plt.title(col)
    fname = col+'.png'
    plt.ticklabel_format(style='plain', axis='y')
    #plt.savefig('/Users/chriscronin/Documents/OneDrive - Noodle Analytics/Code/scrap_index/'+fname,bbox_inches='tight')
    plt.show()

for col in col_list:
    for c in range(1,3):
        y=result2[result2.c_code == c]
        plt.plot(y.date,y[col],label = c_list[c])
    plt.xticks(rotation=90)    
    plt.legend()
    plt.title(col)
    fname = col+'.png'
    plt.ticklabel_format(style='plain', axis='y')
    #plt.savefig('/Users/chriscronin/Documents/OneDrive - Noodle Analytics/Code/scrap_index/'+fname,bbox_inches='tight')
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
    #plt.savefig('/Users/chriscronin/Documents/OneDrive - Noodle Analytics/Code/scrap_index/pics/'+fname,bbox_inches='tight')
    plt.show()  

#%%
result2.sum()
df1.plot(x=df1.index, y=["amm_busheling_midwest", "sci"], kind="line") 
result2.groupby('c_code').sum()
#%%
''' Short - Simulation'''    
output = {}
for x,(c_code,weight) in itertools.product(list(dates)[:-months],enumerate(weights)):
    tzero_spot_price,tzero_futures_price,tone_spot_price,tone_futures_price = get_data(x,c_code,months)
    results = compute_hedge_short(contract_pounds * weight,number_contracts,tzero_spot_price,tzero_futures_price,tone_spot_price,tone_futures_price)
    output[(x,c_code,months)] = list(results)
   
print(output)
result = pd.DataFrame.from_dict(output,orient='index')
mi = pd.MultiIndex.from_tuples(result.index,names=['date','c_code','offset'])
result.index = mi
result.columns = ['futures_gain_position_short','net_price_short','net_cost_locked_in_if_converged_short']
result = result.reset_index()
result = pd.merge(result,hedges,on=['date','c_code'],how='left')
#%%
# Rule to short or not
for col in ['futures_gain_position_short','net_price_short','net_cost_locked_in_if_converged_short']:
    result[col] = np.where(result['sentiment_bush'] < 0, 0, result[col])
#%%
''' Short Separate Plots '''    
# Futures Gain Position Short - how much $ won/loss from futures transaction
g = sns.FacetGrid(result,col ='c_code', col_wrap = 2, hue='c_code')
g = g.map(plt.plot,'date','futures_gain_position_short')
# Net Price Short = 
h = sns.FacetGrid(result,col ='c_code', col_wrap = 2, hue='c_code')
h = h.map(plt.plot,'date','net_price_short')

i = sns.FacetGrid(result,col ='c_code', col_wrap = 2, hue='c_code')
i = i.map(plt.plot,'date','net_cost_locked_in_if_converged_short')
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