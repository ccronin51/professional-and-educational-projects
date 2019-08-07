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
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.options.display.float_format = '{:.2f}'.format
pd.set_option('display.max_colwidth', -1) 
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
#from dateutil.relativedelta import *
from dateutil.relativedelta import relativedelta
#%%
os.chdir('/Users/chriscronin/Documents/OneDrive - Noodle Analytics/Code/scrap_index/data/outside_sim')
df1 = pd.read_csv('data.csv')
df2 = pd.read_csv('futures_data_updated.csv')
coefs = pd.read_csv('coef.csv',parse_dates=['train_end_date']) # parse_dates
coefs = coefs.set_index(['train_end_date']) # set_index
#%%
''' Initial Plots '''
#df1.plot(x=df1.index, y=["amm_busheling_midwest", "hr_coil"], kind="line")  
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
df1 = df1.drop('aluminium_aloy',1)
# delete front month + 15,16,17 b/c some nulls and far out
df2 = df2.drop(['Front Month + 15', 'Front Month + 16','Front Month + 17'], axis=1)
df1.isnull().sum()
df2.isnull().sum()
#%%
''' Fill in Busheling Nans for Now '''
# fill busheling nans with mean of row

# =============================================================================
# Fill Rows with avg of row test
# df = pd.DataFrame()
# df['c1'] = [1, 2, 3]
# df['c2'] = [4, 5, 6]
# df['c3'] = [7, np.nan, 9]
# df
# m = df.mean(axis=1)
# for i, col in enumerate(df):
#     df.iloc[:, i] = df.iloc[:, i].fillna(m)
# =============================================================================
m = df2.mean(axis=1)
for i, col in enumerate(df2):
    df2.iloc[:, i] = df2.iloc[:, i].fillna(m)
    
#row_test = df2.iloc[[0]]   
df2.isnull().sum()    
df2.info()

#%%
''' Create full list of dates with back fill '''
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
''' create date range of simulation '''
#dates = pd.date_range(start='2016-01-04', end ='2018-04-04', freq=pd.DateOffset(months=1))
dates = pd.date_range(start='2016-01-04', end ='2018-08-04', freq=pd.DateOffset(months=1))
#dates = pd.date_range(start='2018-10-04', end ='2020-06-04', freq=pd.DateOffset(months=1))
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

coefs_range = pd.date_range(start=coefs.index.min(), end = df1.index.max(), freq='D')
coefs = coefs.reindex(coefs_range, method='ffill')
intercept = coefs['intercept']
coefs = coefs.stack()
coefs.name = 'multiplier'

df2 = df2.set_index('Commodity',append=True)
coefs.index.names = ['Date','Commodity']
df2 = df2.join(coefs.to_frame(),how='left')

cols = [x for x in df2.columns if x.startswith('Front Month')]
new_cols = [x + 'adjusted' for x in cols]
df2[new_cols] = df2[cols].mul(df2['multiplier'],axis=0)
sci = df2[new_cols].groupby(level=0).sum()
sci = sci.add(intercept, axis=0)
#%%
df2 = df2.drop(new_cols+['multiplier'], axis=1)
sci.columns = cols
sci['Commodity'] = 'SCI'
#%%
df2 = df2.reset_index(level=['Commodity'])
df2 = pd.concat([df2,busheling,sci], axis=0)
#%%
df1_sci = df1.copy()
df1_sci = df1_sci.rename(columns={'amm_busheling_midwest':'Busheling', 
                                  'hr_coil':'HRC',  
                                  'iron_ore':'SGX_IRON_ORE',
                                  'turkish_scrap':'Turkish_HMS',
                                  'aluminium':'LME_US_Aluminum',
                                  'nue': 'Nue'})
df1_sci = df1_sci.drop('Busheling',axis=1)
df1_sci = df1_sci.drop('metal_margin',axis=1)

df1_sci = df1_sci.stack()
df1_sci.index.names = ['Date', 'Commodity']
df1_sci = df1_sci.to_frame().join(coefs.to_frame(),how='left')
df1_sci = df1_sci.mul(df1_sci['multiplier'],axis=0)
df1_sci = df1_sci.drop('multiplier',axis=1)
df1_sci = df1_sci.groupby(level=0).sum()
df1_sci = df1_sci.add(intercept, axis=0)
df1_sci.columns = ['sci']

df1 = pd.concat([df1,df1_sci], axis=1)
#%%
os.chdir('/Users/chriscronin/Documents/OneDrive - Noodle Analytics/Code/scrap_index/data')
''' Full Scenario '''
df1_date_range = df1[df1.index.isin(dates)]
df1_date_range = df1_date_range.to_csv('df1_date_range_july18.csv')
df1_date_range = pd.read_csv('df1_date_range_july18_june20.csv')
df1_date_range['date'] = pd.to_datetime(df1_date_range['date'])
df1_date_range.set_index(['date'], inplace=True)
#%%
''' SKIP '''
''' Future Scenario '''
df1_date_range = pd.read_csv('df1_date_range_final_next_two.csv')
df1_date_range['date'] = pd.to_datetime(df1_date_range['date'])
df1_date_range.set_index(['date'], inplace=True)
# df1_date_range = df1_date_range.drop(df1_date_range.index[len(df1_date_range)-2])
#%%
#df2_date_range = df2[df2.index.isin(dates)]
df2_date_range = pd.read_csv('df2_date_range_final.csv')
df2_date_range['date'] = pd.to_datetime(df2_date_range['date'])
df2_date_range.set_index(['date'], inplace=True)
#%%  
''' Hedge Functions - Long & Short '''
#%% ''' Long '''
def compute_hedge_long(total_bought_per_month,
                       percent_hedged,
                       contract_pounds,
                       tzero_spot_price,
                       tzero_futures_price,
                       tone_spot_price,
                       tone_futures_price):
    number_contracts = (total_bought_per_month * percent_hedged) / contract_pounds
    unhedge_contracts = total_bought_per_month * (1-percent_hedged) / contract_pounds
    #basis = tone_spot_price - tone_futures_price
    cost_long = - contract_pounds * number_contracts * tone_spot_price
    futures_gain_per_pound_long = tone_futures_price - tzero_futures_price
    futures_gain_position_long = futures_gain_per_pound_long * contract_pounds * number_contracts
    net_cost_long = cost_long + futures_gain_position_long
    net_net_cost_long = - contract_pounds * unhedge_contracts * tone_spot_price   
    net_cost_locked_in_if_converged_long = - contract_pounds * number_contracts * tzero_futures_price
    return futures_gain_position_long,net_cost_long,net_cost_locked_in_if_converged_long, cost_long, net_net_cost_long
#%% ''' Short '''
def compute_hedge_short(total_bought_per_month,
                                percent_hedged,
                                contract_pounds,
                                tzero_spot_price,
                                tzero_futures_price,
                                tone_spot_price,
                                tone_futures_price):
    number_contracts = (total_bought_per_month * percent_hedged) / contract_pounds
    unhedge_contracts = total_bought_per_month * (1-percent_hedged) / contract_pounds
    #basis = tone_spot_price - tone_futures_price
    price_short = contract_pounds * number_contracts * tone_spot_price
    futures_gain_per_pound_short = tzero_futures_price - tone_futures_price
    futures_gain_position_short = futures_gain_per_pound_short * contract_pounds * number_contracts
    net_price_short = price_short + futures_gain_position_short
    net_net_price_short = - contract_pounds * unhedge_contracts * tone_spot_price
    net_cost_locked_in_if_converged_short = contract_pounds * number_contracts * tzero_futures_price
    return futures_gain_position_short,net_price_short,net_cost_locked_in_if_converged_short, price_short, net_net_price_short    
#%%
''' Hedge Example to Show it Works '''
print(compute_hedge_long(25000,1,10000,2.50,2.70,2.70,2.64)) 
print(compute_hedge_long(150000,.25,10000,383,338,347,347))       
#%%
''' This shows that SCI isn't currenlty matching Busheling '''
# =============================================================================
# start = df1.index.searchsorted(dt.datetime(2012, 1, 2))
# end = df1.index.searchsorted(dt.datetime(2018, 4, 9))
# premium_discount_df = df1.ix[start:end]
# premium_discount_df.plot(x=premium_discount_df.index, y=["amm_busheling_midwest", "sci"], kind="line") 
# =============================================================================
#%%
''' Simulation Setup:
    Contract Specifics:
    1 contrat equals 10,000 tons min standard for an order. 
    Would want steel daily volume. 
    150,000 tons a steel ordered/produced a month at BRS. 
'''
# Standard
total_bought_per_month = 150000
contract_pounds = 10000
months = 3 # months out hedging
# Original
#           HRC,B,SCI
weights = [[.00,1,.00]]
samples = [.10]
# Iterating over different scenarios
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
''' Example to see if 'get_data' is pulling correct numbers '''
#print(get_data('2017-05-04',1,3))
#print(get_data('2019-05-04',1,3))  
print(get_data('2019-05-04',1,6)) 
#%%
''' Long - Simulation '''
# Generator
output2 = {}
for index,(percent_hedged_scenario,weights_scenario) in enumerate(itertools.product(samples,weights)):
    for x,(c_code,weight) in itertools.product(list(dates)[:-months],enumerate(weights_scenario)):
        tzero_spot_price,tzero_futures_price,tone_spot_price,tone_futures_price = get_data(x,c_code,months)
        try:
            results2 = compute_hedge_long(total_bought_per_month,percent_hedged_scenario,contract_pounds * weight,tzero_spot_price,tzero_futures_price,tone_spot_price,tone_futures_price)
            output2[(x,c_code,months,index)] = list(results2)
        except ZeroDivisionError:
            continue
#%%
result2 = pd.DataFrame.from_dict(output2,orient='index')
mi2 = pd.MultiIndex.from_tuples(result2.index,names=['date','c_code','offset','loop'])
result2.index = mi2
result2.columns = ['futures_gain_position_long','net_cost_long','net_cost_locked_in_if_converged_long','cost_long','net_net_cost_long']
result2 = result2.groupby(level=['date','c_code','offset']).mean()
result2 = result2.reset_index()  
result2['total_cost'] = result2['net_cost_long'] + result2['net_net_cost_long']
#%%
#result2.to_csv('long_simple_example_9_10_18.csv')
result2 = result2.drop('c_code',axis=1)
result2 = result2.drop('offset',axis=1)
result2.set_index(['date'], inplace=True)
result2 = result2.rename(columns={'futures_gain_position_long':'Scrap: Futures Gain/Loss', 
                                  'net_cost_long':'Scrap: Hedged Cost',  
                                  'net_cost_locked_in_if_converged_long':'Scrap: Hedge Cost Basis Convergence',
                                  'cost_long':'Scrap: Actual Cost',
                                  'net_net_cost_long':'Scrap: Unhedged Cost',
                                  'total_cost': 'Scrap: Total Cost'})
  
result2 = result2[['Scrap: Futures Gain/Loss',
                   'Scrap: Actual Cost',
                   'Scrap: Hedged Cost',
                   'Scrap: Unhedged Cost',
                   'Scrap: Total Cost',
                   'Scrap: Hedge Cost Basis Convergence']]
#%%
''' Short - Simulation '''
# Standard
total_bought_per_month = 150000
contract_pounds = 10000
months = 3 # months out hedging
# Original
#           HRC,B,SCI
weights = [[1,.00,.00]]
samples = [.1]
#%%
# Generator
output3 = {}
for index,(percent_hedged_scenario,weights_scenario) in enumerate(itertools.product(samples,weights)):
    for x,(c_code,weight) in itertools.product(list(dates)[:-months],enumerate(weights_scenario)):
        tzero_spot_price,tzero_futures_price,tone_spot_price,tone_futures_price = get_data(x,c_code,months)
        try:
            results3 = compute_hedge_short(total_bought_per_month,percent_hedged_scenario,contract_pounds * weight,tzero_spot_price,tzero_futures_price,tone_spot_price,tone_futures_price)
            output3[(x,c_code,months,index)] = list(results3)
        except ZeroDivisionError:
            continue
#%%
result3 = pd.DataFrame.from_dict(output3,orient='index')
mi3 = pd.MultiIndex.from_tuples(result3.index,names=['date','c_code','offset','loop']) # No loop like above
result3.index = mi3
result3.columns = ['futures_gain_position_long','net_cost_long','net_cost_locked_in_if_converged_long','cost_long','net_net_cost_long']
result3 = result3.groupby(level=['date','c_code','offset']).mean()
result3 = result3.reset_index()
result3['total_cost'] = result3['net_cost_long'] + result3['net_net_cost_long']
#%%
#result3.to_csv('short_simple_example_9_10_18.csv')
result3 = result3.drop('c_code',axis=1)
result3 = result3.drop('offset',axis=1)
result3.set_index(['date'], inplace=True)
result3 = result3.rename(columns={'futures_gain_position_long':'HRC: Futures Gain/Loss', 
                                  'net_cost_long':'HRC: Hedged Sale',  
                                  'net_cost_locked_in_if_converged_long':'HRC: Hedge Sale Basis Convergence',
                                  'cost_long':'HRC: Actual Sale',
                                  'net_net_cost_long':'HRC: Unhedged Sale',
                                  'total_cost': 'HRC: Total Sale'})
  
result3 = result3[['HRC: Futures Gain/Loss',
                   'HRC: Actual Sale',
                   'HRC: Hedged Sale',
                   'HRC: Unhedged Sale',
                   'HRC: Total Sale',
                   'HRC: Hedge Sale Basis Convergence']]
#%%
amount_tons_hedged = total_bought_per_month * samples[0]

excel_output = pd.concat([result2,result3], axis=1)
excel_output['actual metal margin'] = excel_output['HRC: Actual Sale'] + excel_output['Scrap: Actual Cost']
excel_output['hedged metal margin'] = excel_output['HRC: Hedged Sale'] + excel_output['Scrap: Hedged Cost']

excel_output['actual metal margin per ton'] = excel_output['actual metal margin'] / amount_tons_hedged
excel_output['hedged metal margin per ton'] = excel_output['hedged metal margin'] / amount_tons_hedged

excel_output['Scrap Gain/Loss Per Ton'] = excel_output['Scrap: Futures Gain/Loss'] / amount_tons_hedged
excel_output['HRC Gain/Loss Per Ton'] = excel_output['HRC: Futures Gain/Loss'] / amount_tons_hedged
excel_output['Scrap & HRC Gain/Loss Per Ton'] = excel_output['Scrap Gain/Loss Per Ton'] + excel_output['HRC Gain/Loss Per Ton']

excel_output['Cumulative Scrap Gain/Loss Per Ton'] = excel_output['Scrap Gain/Loss Per Ton'].cumsum(axis=0)
excel_output['Cumulative HRC Gain/Loss per Ton'] = excel_output['HRC Gain/Loss Per Ton'].cumsum(axis=0)
excel_output['Cumulative Scrap & HRC Gain/Loss per Ton'] = excel_output['Scrap & HRC Gain/Loss Per Ton'].cumsum(axis=0)

excel_output['Scrap Hedge Costs (Cumulative)'] = excel_output['Scrap: Hedged Cost'].cumsum(axis=0)
excel_output['Scrap Unhedged Costs (Cumulative)'] = excel_output['Scrap: Actual Cost'].cumsum(axis=0)
excel_output['Hedging Costs Benefits'] = excel_output['Scrap Hedge Costs (Cumulative)'] - excel_output['Scrap Unhedged Costs (Cumulative)']

excel_output['Scrap: Actual Cost Per Ton'] = excel_output['Scrap: Actual Cost'] / amount_tons_hedged
excel_output['Scrap: Hedged Cost Per Ton'] = excel_output['Scrap: Hedged Cost'] / amount_tons_hedged
#excel_output = excel_output[(excel_output['hedged metal margin per ton'] >= 250)]

excel_output.to_csv('excel_output_102v10vfullDECEMBER_FEB19.csv')
#%%
df1.metal_margin.mean()
#%%
''' STOP '''
''' Long Plots '''      
# Futures Gain Position Long - how much $ won/loss from just futures transaction
g2 = sns.FacetGrid(result2,col ='c_code', col_wrap = 2, hue='c_code')
g2 = g2.map(plt.plot,'date','futures_gain_position_long')

# Net Cost Long = price you are locking in = cost_long + futures_gain_position_long # Price you are locking in but can be different if tone spot/future prices don't converge
h2 = sns.FacetGrid(result2,col ='c_code', col_wrap = 2, hue='c_code')
h2 = h2.map(plt.plot,'date','net_cost_long')

# Net_cost_locked_in_if_converged_long = Price you are locking in if basis doesn't change (i.e. convergence in spot prices)
i2 = sns.FacetGrid(result2,col ='c_code', col_wrap = 2, hue='c_code')
i2 = i2.map(plt.plot,'date','net_cost_locked_in_if_converged_long')

# Cost_Long = cost in physical world = - contract_pounds * number_contracts * tone_spot_price # how much it would cost w/o hedging
j2 = sns.FacetGrid(result2,col ='c_code', col_wrap = 2, hue='c_code')
j2 = j2.map(plt.plot,'date','cost_long')
# Cost of what isn't hedged
k2 = sns.FacetGrid(result2,col ='c_code', col_wrap = 2, hue='c_code')
k2 = k2.map(plt.plot,'date','net_net_cost_long')

# Total Cost equals cost_long plus net_net_cost_long (unhedge costs)
l2 = sns.FacetGrid(result2,col ='c_code', col_wrap = 2, hue='c_code')
l2 = l2.map(plt.plot,'date','total_cost')

''' Long Other Plots '''
# Setup
col_list = ['futures_gain_position_long','net_cost_long','net_cost_locked_in_if_converged_long','cost_long','total_cost']
c_list =['hrc','busheling','sci']

result2.groupby(['c_code']).plot()

c_dict = {
        0:'HRC',
        1:'Busheling',
        2:'SCI'}

# net_cost_long & cost_long
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

# Single
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

# Cumulative
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
'''Short Plots '''      
# Futures Gain Position Long - how much $ won/loss from just futures transaction
g2 = sns.FacetGrid(result3,col ='c_code', col_wrap = 2, hue='c_code')
g2 = g2.map(plt.plot,'date','futures_gain_position_long')
#%%
# Net Cost Long = cost_long + futures_gain_position_long # Price you are locking in but can be different if tone spot/future prices don't converge
h2 = sns.FacetGrid(result3,col ='c_code', col_wrap = 2, hue='c_code')
h2 = h2.map(plt.plot,'date','net_cost_long')
#%%
# Net_cost_locked_in_if_converged_long = - contract_pounds * number_contracts * tzero_futures_price # Price you are locking in
i2 = sns.FacetGrid(result3,col ='c_code', col_wrap = 2, hue='c_code')
i2 = i2.map(plt.plot,'date','net_cost_locked_in_if_converged_long')
#%%
# Cost_Long = - contract_pounds * number_contracts * tone_spot_price # how much it would cost w/o hedging
j2 = sns.FacetGrid(result3,col ='c_code', col_wrap = 2, hue='c_code')
j2 = j2.map(plt.plot,'date','cost_long')
#%%
k2 = sns.FacetGrid(result3,col ='c_code', col_wrap = 2, hue='c_code')
k2 = k2.map(plt.plot,'date','net_net_cost_long')
# Total Cost equals cost_long plus unhedge costs
l2 = sns.FacetGrid(result3,col ='c_code', col_wrap = 2, hue='c_code')
l2 = l2.map(plt.plot,'date','total_cost')
#%%
''' Long Other Plots '''
# Setup
col_list = ['futures_gain_position_long','net_cost_long','net_cost_locked_in_if_converged_long','cost_long','total_cost']
c_list =['hrc','busheling','sci']

result3.groupby(['c_code']).plot()

c_dict = {
        0:'HRC',
        1:'Busheling',
        2:'SCI'}

# net_cost_long & cost_long
for c in range(1,3):
    y=result3[result3.c_code == c]
    y.plot(x=y.date,y=['net_cost_long','cost_long'],title=c_dict[c])   
    plt.xticks(rotation=90)
    
    plt.legend()
    plt.title(col)
    fname = col+'.png'
    plt.ticklabel_format(style='plain', axis='y')
    #plt.savefig('/Users/chriscronin/Documents/OneDrive - Noodle Analytics/Code/scrap_index/'+fname,bbox_inches='tight')
    plt.show()

# Single
for col in col_list:
    for c in range(1,3):
        y=result3[result3.c_code == c]
        plt.plot(y.date,y[col],label = c_list[c])
    
        plt.xticks(rotation=90)    
        plt.legend()
        plt.title(col)
        fname = col+'.png'
        plt.ticklabel_format(style='plain', axis='y')
        #plt.savefig('/Users/chriscronin/Documents/OneDrive - Noodle Analytics/Code/scrap_index/'+fname,bbox_inches='tight')
        plt.show()

# Cumulative
for col in col_list:
    for c in range(1,3):
        y=result3[result3.c_code == c]
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
''' Parking Lot - Old Metal Margin Function '''
#%%
output3 = {}
for x,(c_code,weight) in itertools.product(list(dates)[:-months],enumerate(weights)):
    tzero_spot_price,tzero_futures_price,tone_spot_price,tone_futures_price = get_data(x,c_code,months)
    if spread.loc[x,'range'] == 'Lock_in':
        try:
            if (df1_date_range.loc[x,'spot_margin2'] == 'high' and c_code == 0) or (
                    df1_date_range.loc[x,'spot_margin2'] == 'low' and c_code == 1):
                metal_spread = compute_hedge_short(total_bought_per_month,
                                                   percent_hedged_metal_margin ,
                                          contract_pounds * weight,
                                          tzero_spot_price,
                                          tzero_futures_price,
                                          tone_spot_price,tone_futures_price)
            else:
                metal_spread = compute_hedge_long(total_bought_per_month,
                                          percent_hedged_metal_margin ,
                                          contract_pounds * weight,
                                          tzero_spot_price,
                                          tzero_futures_price,
                                          tone_spot_price,
                                          tone_futures_price) # fix so five values
        except ZeroDivisionError:
            continue
    else:
        metal_spread = [np.nan] * 5 # To Do
    output3[(x,c_code,months)] = list(metal_spread)                                
    
#%%   
df1_date_range.to_csv('df1_date_range.csv')    
df2_date_range.to_csv('df2_date_range.csv') 


df1_date_range = pd.read_csv('df1_date_range_edit.csv')
df2_date_range = pd.read_csv('df2_date_range_edit.csv')
#%%  
''' Change Dtypes, Index '''
# Changed integers to floats, dates to datetime, reset index
df1_date_range['hr_coil'] = df1_date_range['hr_coil'].astype(np.float)
df1_date_range['date'] = pd.to_datetime(df1_date_range['date'])
df1_date_range.set_index(['date'], inplace=True)
df1_date_range.info()
# Changed dates to datetime, reset index
df2_date_range['Date'] = pd.to_datetime(df2_date_range['Date'])
df2_date_range.set_index(['Date'], inplace=True)
df2_date_range.info()
