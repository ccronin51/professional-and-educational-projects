#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 19:46:07 2018

@author: cronin
"""
import pandas as pd
from pandas.tseries.offsets import MonthEnd, DateOffset

def method_0(tnow,l,d):
    pass

def method_1():
    pass

def method_2(l,D,unit,prices,bushelings):
    tnow = pd.to_datetime('today') + MonthEnd(0)
    tnow_1 = tnow + MonthEnd(1)
    price = prices.loc[tnow_1]
    start = tnow + MonthEnd(l)
    rng = pd.date_range(start=start,freq='M',periods=D)
    units = bushelings.loc[rng].sum()
    return units * price