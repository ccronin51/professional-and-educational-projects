#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 12:35:53 2018

@author: chriscronin
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
#%%
print(os.getcwd())
os.chdir('/Users/chriscronin/Dropbox/python/data')
#%%
df = pd.read_csv("mp_data.csv")
print(df)
#%%
df_2 = df
df_3 = df
#%%
def addColumn(colName, formula, df, idxs = []):
    numRows = len(df.index)
    vals = []

    for i in range(numRows): # for each line
        deps = [] # empty list of values on which formula depends
        for j in idxs: # going through indexes of columns of which formula depends
            deps.append(df.iloc[i, j]) # adding each row for the columns in idxs
        vals.append(formula(deps)) # add values from formula on columns for calculation

    kwargs = {colName : vals} # new colum
    df = df.assign(**kwargs) # assigning new col with all previous. ** will unpack dictionary. 
    return df
#%%
def delta(vals):
    return float(vals[1]) / vals[0] - 1

def share(vals):
    return float(vals[0] / vals[1])

def subtract(vals):
    return float(vals[0] - vals[1])

def addColumnVals(colName, vals, df):
    kwargs = {colName : vals} # new colum
    df = df.assign(**kwargs) # assigning new col with all previous. 
    return df

def slope(x1, y1, x2, y2):
    return (float)(y1 - y2) / (x1 - x2)

def intercept(x_val,slope_val):
    return 1 - x_val * slope_val

def normalization(vals):
    return (vals[0] * vals[1]) + vals[2]

def mean(vals):
    return sum(vals) / len(vals)
#%%
def market_position(df):
    df = addColumn('total_delta', delta, df, idxs = [1,2]) 
    df = addColumn('atd_delta', delta, df, idxs = [3,4]) 
    df = addColumn('start_share', share, df, idxs = [3,1])
    df = addColumn('end_share', share, df, idxs = [4,2])
    df = addColumn('share_delta', subtract, df, idxs = [7,8])
    df = addColumnVals("Total Slope", slope(df['end_total'].max(), 1, 0, 0), df)
    df = addColumnVals("Share Slope", slope(df['end_share'].max(), 1, 0, 0), df)
    df = addColumnVals("Total Delta Slope", slope(df['total_delta'].max(), 1, df['total_delta'].min(), 0), df)
    df = addColumnVals("Share Delta Slope", slope(df['share_delta'].max(), 1, df['share_delta'].min(), 0), df)
    df = addColumnVals("end_total_intercept", intercept(df['end_total'].max(),df['Total Slope'].max()), df)
    df = addColumnVals("end_share_intercept", intercept(df['end_share'].max(),df['Share Slope'].max()), df)
    df = addColumnVals("total_delta_intercept", intercept(df['total_delta'].max(),df['Total Delta Slope'].max()), df)
    df = addColumnVals("share_delta_intercept", intercept(df['share_delta'].max(),df['Share Delta Slope'].max()), df)
    df = addColumn("total normalized",normalization,df,idxs=[2,10,14])
    df = addColumn("share normalized",normalization,df,idxs=[8,11,15])
    df = addColumn("total growth normalized",normalization,df,idxs=[5,12,16])
    df = addColumn("share delta normalized",normalization,df,idxs=[9,13,17])
    df = addColumn("market measure",mean,df,idxs=[18,20])
    df = addColumn("share measure",mean,df,idxs=[19,21])
    return df
#%%
df = market_position(df)
# y1 - y2 / x1 - x2
# x1,y1,x2,y2
print(df)
#%%
df_list = [df_2, df_3]

for i, f in enumerate(df_list):
    processed_f = market_position(f)
    area = np.pi * processed_f['end_bing'] * processed_f['end_bing']
    plt.scatter(processed_f['market measure'],processed_f['share measure'],s=area)
    plt.savefig('output' + (str)(i) + 'png', dpi=300)
    plt.show()
#%%    
area = np.pi * df['end_bing'] * df['end_bing']
plt.scatter(df['market measure'],df['share measure'],s=area)
plt.show()    