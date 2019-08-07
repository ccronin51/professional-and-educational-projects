#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 19:53:14 2018

@author: cronin
"""

import unittest 
import pandas as pd
from methods import method_2

class TestMethods(unittest.TestCase):
    def test_method_2(self):
        prices = pd.Series(
                [25,50,75],
                index=pd.date_range(start=pd.to_datetime('today'),
                                    freq='M',
                                    periods = 3))
        bushelings = pd.Series(
                [10,40,50,20,60,20,70,80,90,50],
                index=pd.date_range(start=pd.to_datetime('today'),
                                    freq='M',
                                    periods = 10))
        l = 3
        D = 6
        expected = 17000
        result = method_2(l,D,'M',prices,bushelings)
        self.assertEqual(result,expected)
                