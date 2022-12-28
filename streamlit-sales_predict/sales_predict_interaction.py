'''
Project: Corona sales prediction 
Author: Robbie
Date: 12/14/2022
'''


import streamlit as st
import os
import gc
import datetime

import pickle
import pandas as pd
pd.options.display.max_rows = 2000
pd.options.display.max_columns = 100

import numpy as np

from string import ascii_letters

import seaborn as sns
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import plotly
import plotly.express as px
import plotly.graph_objs as go

from sklearn.model_selection import train_test_split #Split data in testing and training
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import normalize

from functools import reduce

import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.seasonal import seasonal_decompose

import calendar as cl
from calendar import monthrange

from tqdm import tqdm

from xgboost import XGBRegressor

from catboost import CatBoostRegressor

from lightgbm import LGBMRegressor

import csv

print('Packages importing successfully.')

header = st.container()
dataset = st.container()
chart = st.container()
model = st.container()


with header: 
    st.title('Welcome to the testing website.')
    st.text('In this project we will create an interaction with tables and charts.')
    

with dataset:
    st.header('This is the dataset we will input data inside.')
    

    
    # variables by Robbie
new_variables_Robbie = pd.read_csv('US Marco data Robbie.csv')
# convert the Date to INDEX
new_variables_Robbie.index = pd.to_datetime(new_variables_Robbie['observation_date'])
new_variables_Robbie.drop("observation_date", axis = 1, inplace = True)
new_variables_Robbie = new_variables_Robbie.rename(columns = {"FEDFUNDS": "US Federal Funds Effective Rate", 
                                  "CPALTT01USM657N": "US Consumer Price Index", 
                                 "PCE": "US Personal Consumption Expenditures"})
new_variables_Robbie = new_variables_Robbie.dropna()
#new_variables_Robbie.head()

# variables by Ahmad
new_variables_AB = pd.read_csv('new_variables_Ahmad.csv')
# convert the Date to INDEX
new_variables_AB.index = pd.to_datetime(new_variables_AB['observation_date'])
new_variables_AB.drop("observation_date", axis = 1, inplace = True)
new_variables_AB = new_variables_AB.dropna()
#new_variables_AB.head()

# variables by Peter
new_variables_Peter = pd.read_csv('US Marco data Peter.csv')
# convert the Date to INDEX
new_variables_Peter.index = pd.to_datetime(new_variables_Peter['DATE'])
new_variables_Peter.drop("DATE", axis = 1, inplace = True)
#new_variables_Peter.head()

# variables by Robin
new_variables_Robin = pd.read_csv('US Marco data Robin.csv')
# convert the Date to INDEX
new_variables_Robin.index = pd.to_datetime(new_variables_Robin['date'])
new_variables_Robin.drop("date", axis = 1, inplace = True)
#new_variables_Robin.head()
    
    
combined_var = reduce(lambda left,right: pd.merge(left,right,left_index=True, right_index=True, how='left'),
                 [new_variables_Robbie, new_variables_AB, new_variables_Peter, new_variables_Robin])
combined_var = combined_var.drop(columns=['U.S Rental Vacency Rate','PrivateWages',  'US Federal Funds Effective Rate', 'Retail Sales (% Change)'])
combined_var = combined_var.rename(columns = {"US Consumer Price Index": "US CPI", 
                                  "US Personal Consumption Expenditures": "US PCE", "Unemployment Rate": "U Rate",
                                 "Real GDP Per Capita": "RGDP/Cap", "Average Price Fuel Oil": "AOP", 'Import & Export': 'IMEX', 'AvgWeeklyHourslag_3': 'AvgWkHrslag_3' })
combined_var = combined_var.dropna()
combined_var.head()

 
   # st.text('Here will be a table.')
    
with chart:
    st.header('This is the chart for pediction based on the data.')
    
    
with model:
    st.header('This is a space for modeling.')
    st.text('Here will be a demonstration of the fomula with the interpet and coefficients.')
    
    
    

