# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 12:38:37 2025

@author: USER
"""
import pandas as pd

#%%
# #%% : 셸을 나눔 
# Ctrl + Enter :  현재 셸만 실행(주피터와 같이 해당 셸만 실행 가능)
print(1)
#%%
print(2)
a=[10, 20, 30]

#%%

age = pd.Series([25, 34, 19, 45, 60])
age
type(age)

data = ['spring', 'summer', 'fall', 'winter']
season = pd.Series(data)
season
type(season)
season.iloc[2]

#%%
score = pd.DataFrame([[85, 96],
                  [73, 69],
                  [78, 50]])
score
type(score)
score.index
score.columns

score.iloc[1, 1]
