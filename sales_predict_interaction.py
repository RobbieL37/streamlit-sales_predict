'''
Project: Corona sales prediction 
Author: Robbie
Date: 12/14/2022
'''

'''
Project: Corona sales prediction 
Author: Team Acceleration 
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


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

from st_aggrid import AgGrid
from st_aggrid import GridOptionsBuilder
from st_aggrid import GridUpdateMode
from st_aggrid import DataReturnMode
from st_aggrid import AgGridTheme

import Created_Functions
from Created_Functions import Created_Functions

print('Packages importing successfully.')

header = st.container()
dataset = st.container()
chart = st.container()
model = st.container()


with header: 
    st.title('Welcome To Our Corona Sales Prediction Dashboard')
    st.text('In This Project We will Create An interaction With Tables And Charts.')
    

with dataset:
    
    new_variables_Robbie = pd.read_csv('US Marco data Robbie.csv')
# convert the Date to INDEX
new_variables_Robbie.index = pd.to_datetime(new_variables_Robbie['observation_date'])
new_variables_Robbie.drop("observation_date", axis = 1, inplace = True)
new_variables_Robbie = new_variables_Robbie.rename(columns = {"FEDFUNDS": "US Federal Funds Effective Rate", 
                                  "CPALTT01USM657N": "US Consumer Price Index", 
                                 "PCE": "US Personal Consumption Expenditures"})
new_variables_Robbie = new_variables_Robbie.dropna()
#new_variables_Robbie.head())

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
                                 "Real GDP Per Capita": "RGDP/Cap", "Average Price Fuel Oil": "AOP", 'Import & Export': 'IMEX', 'AvgWeeklyHourslag_3': 'AvgWkHrslag_3'})
combined_var = combined_var.dropna()
#st.write(combined_var.head())


# sales data import
sales = pd.read_csv('Scaled_sales_porcelana.csv', encoding='latin-1', sep=',')
# Renaming the columns - English
sales.columns = ['date', 'Booking', 'Quantity', 'seasonal', 'trend_short_6', 'trend_short_12']
# Converting date to index
sales.index = pd.to_datetime(sales["date"])
sales = sales.drop(columns='date')
#st.write(sales.head())

# 1.calendar data
monthly_calendar = pd.read_csv('Holidays_colombia_2024.csv')
monthly_calendar.index = pd.to_datetime(monthly_calendar['Fecha'])
monthly_calendar.drop("Fecha", axis = 1, inplace = True)

# 2.macro data from Corona
macro = pd.read_csv('Macro_economic_var.csv', encoding='latin-1')
macro.columns = ['date', 'Construction_lic_unt','Construction_lic_area', 'Construction_lic_unt_SI','Construction_lic_unt_NO_SI',
            'Construction_lic_area_SI','Construction_lic_area_NO_SI' ,  'Gray_cement_prod', 'Gray_cement_dispatch', 
                 'Gray_cement_dispatch_Factory','Gray_cement_dispatch_comercial','Gray_cement_dispatch_contractor','Gray_cement_dispatch_other',
                 'DTF(interest_rate)','GDP_Constr', 'GDP_Total', 'Inflation', 'USD_exchange', 'Oil_brent', 'Oil_WTI', 'ISE', 'ICC', 'IEC','ICE','Retail_Commerce', 'Unemployment%', 
              'RADAR_Constr', 'RADAR_Rev', 'RADAR_Toilets', 'RADAR_HomeAppliance', 'RADAR_Paint', 'RADAR_Furniture', 'RADAR_Tools', 'RADAR_Elect',
             'RADAR_Plumbing', 'RADAR_Wood', 'RADAR_Supply', 'RADAR_Remodel', "Construction_lic_area/unt", "Construction_lic_area/unt_SI", "Construction_lic_area/unt_NO_SI"]
macro.index = pd.to_datetime(macro['date'])
macro.drop("date", axis = 1, inplace = True)
#macro = lag_variable(macro,4)

# 3.housing data 
camacol = pd.read_csv('Camacol_Housing.csv', encoding='latin-1', sep=',')
camacol.columns = ['date', 'Housing_total_launch', 'Housing_total_launch_SI', 'Housing_total_launch_NO_SI',
                   'Housing_total_initiation','Housing_total_initiation_SI', 'Housing_total_initiation_NO_SI',
                   'Housing_total_sales',  'Housing_total_sales_SI',  'Housing_total_sales_NO_SI', 
                   'Housing_total_offer','Housing_total_Offer_SI', 'Housing_total_Offer_NO_SI']
camacol.index = pd.to_datetime(camacol["date"])
camacol = camacol.drop(columns=['date'])

# 4.ICCV data
ICCV = pd.read_csv('ICCV.csv', encoding='latin-1')
ICCV = ICCV.loc[:, ~ICCV.columns.str.contains('^Unnamed')]
ICCV["Fecha"] = pd.to_datetime(ICCV["Fecha"],  infer_datetime_format=True, exact = False)
ICCV.columns = ['date', 'ICCV_Var_Yearly', 'ICCV_Var_Monthly']
ICCV.set_index("date", inplace = True)

# 5.credit data 
credit = pd.read_csv('credit_disb.csv', encoding='latin-1', sep=',')
credit.columns = ['date', 'Consumer_credit', 'Consumer_microcredit', 'Ordinary_credit', 'Preferencial_credit', 'loan_overdrafts', 'Credit_card', 'Treasury_loan',
                 'Housing_loan']
credit.index = pd.to_datetime(credit["date"])
credit = credit.drop(columns='date')
#print("imported data successfully")




# make all the lag variables transform
monthly_calendar_lag = Created_Functions.lag_variable(monthly_calendar, 12)
macro_lag = Created_Functions.lag_variable(macro, 12)
camacol_lag = Created_Functions.lag_variable(camacol, 12)
ICCV_lag = Created_Functions.lag_variable(ICCV, 12)
credit_lag = Created_Functions.lag_variable(credit, 12)
combined_var_lag = Created_Functions.lag_variable(combined_var, 12)

# Merging all the lag DFs
total_lag = reduce(lambda left,right: pd.merge(left,right,left_index=True, right_index=True, how='left'),
                 [sales, combined_var_lag, macro_lag, camacol_lag, ICCV_lag, credit_lag, monthly_calendar_lag])
total_lag = total_lag.rename(columns = {"Quantity_x": "Quantity", "working_day": "WorkingDay", "GDP_Constr": "GDPCon", "GDP_Total": "GTotal", "USD_exchange": "USDEx", "Housing_total_sales": "HTSales", 'Housing_total_offer': 'HTOffer', 'ICCV_Var_Yearly': 'ICCV Annu',
"ICCV_Var_Monthly": "ICCV Monthly", "Consumer_credit": "ConsumerC", "Credit_card": "CC"})
#st.write(total_lag.head())

st.header('Best Six Features Related to Corona Sales of Toilets ')
st.text('Values Shown Below Are All Standardized')

# select top 6 variables (dropped 4)
corona_combined_var10 = total_lag[['Quantity', 'Gray_cement_dispatch_comercial','HTSales', 'ISE', 'RADAR_Toiletslag_4','holiday', 'Consumer_microcredit']]
# create another no null df 
corona_combined_var10 = corona_combined_var10.dropna()
std_data = Created_Functions.standardization(corona_combined_var10)

st.write(std_data)

# fig = go.Figure(data=go.Table(
#     columnwidth= [1.5,4,1.5,1.5,2.3,1.5,2.7],
#     header=dict(values=list(std_data[['Quantity', 'Gray_cement_dispatch_comercial','HTSales', 'ISE', 'RADAR_Toiletslag_4','holiday', 'Consumer_microcredit']].columns), fill_color = '#FF9912', align = "center"), 
#     cells= dict(values=[std_data.Quantity.round(3), std_data.Gray_cement_dispatch_comercial.round(3), std_data.HTSales.round(3), std_data.ISE.round(3), std_data.RADAR_Toiletslag_4.round(3), std_data.holiday.round(3), std_data.Consumer_microcredit.round(3)], fill_color = '#F7F7F7', align = "center")))

# fig.update_layout(margin=dict(l=10, r=10, b=10, t=10))
     
# st.write(fig)


#st.write(grid_return = AgGrid(std_data.head(10), editable=True))
#st.write(std_data = pd.DataFrame(grid_return['data']))
#st.write(std_data = grid_return['data'])


#**************************************************************************************************************

#with chart:
    
st.header('Features Interaction with the Sales')

import plotly.graph_objs as go

#st.write(("Feature Relationship with Sales")
          
#Standardizing the data and then create a multiple lines chart
df =  Created_Functions.standardization(corona_combined_var10)
traces = [go.Scatter(
        x = df.index,
        y = df[colname],
        mode = 'markers+lines',
        name = colname            
        ) for colname in list(df.columns)]

layout = go.Layout(title='Top 6 Fluctuation (Standardized Data)')
fig = st.write(go.Figure(data=traces, layout=layout))
#st.write(fig.show())


st.header('Regression Model.')
st.text('Here will be a demonstration of the fomula with the interpet and coefficients.')

# use the function from Created_Functions.py
#X_train, X_test, y_train, y_test = Created_Functions.data_split(corona_combined_var10, 0.3, 10)

# standardize the data
#standarded_x_train, standarded_x_test = Created_Functions.Standard(X_train, X_test)

#plt.figure(figsize=(25,10))
#st.write(sns.boxplot(x="variable", y="value",data = standarded_x_train.melt())) #Explore melt function of pandas
#st.write(plt.xlabel('')) #Erase labels
#st.write(plt.ylabel('')) #Erase label
#st.write(plt.show()


# use the function from Created_Functions.py
X = corona_combined_var10.drop(['Quantity'], axis=1)
y = corona_combined_var10['Quantity']
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=10)
#st.write(y_train)


scaler = StandardScaler()
scaler.fit(X_train)

standarded_x_train = pd.DataFrame(scaler.transform(X_train),
                        index = X_train.index, columns = X_train.columns)
standarded_x_test = pd.DataFrame(scaler.transform(X_test),
                        index = X_test.index, columns = X_test.columns)
#st.write(standarded_x_train)


Best_Model = Ridge(alpha = .1)  #Default value for alpha = 1
Best_Model.fit(standarded_x_train, y_train)

y_Train_prediction = (Best_Model.predict(standarded_x_train))
y_Test_prediction  = (Best_Model.predict(standarded_x_test))
    
#Prediction_Plots(y_train, y_Train_prediction, y_test, y_Test_prediction)
#Metrics_Printer(y_train, y_Train_prediction, y_test, y_Test_prediction)
#st.write(("Ridge regression: "))

st.write(('Intercept:',Best_Model.intercept_.round(4)))
st.write(('Coefficients:', Best_Model.coef_.round(4)))
st.write((" "))
st.write(("training metrics: ", metrics.r2_score(y_train, y_Train_prediction).round(4), "    testing metrics: ", metrics.r2_score(y_test, y_Test_prediction).round(4)))
st.write((" "))

#st.write(Best_Model.predict(standarded_x_test))




#***************************************************************************************************************

st.header('Sales Prediction')

# Standarded data + RidgeRegression

df = corona_combined_var10.copy() # deep copy a new one

X = df.drop(['Quantity'], axis=1)
y = df['Quantity']
            
model = Ridge()
model.fit(Created_Functions.standardization(X), y) #Train the model

df['Prediction'] = model.predict(Created_Functions.standardization(X))


import plotly.graph_objs as go

traces = [go.Scatter(
        x = df.index,
        y = df[colname],
        mode = 'markers+lines',
        name = colname            
        ) for colname in list(['Quantity', 'Prediction'])]

layout = go.Layout(title='Prediction vs Actual<br><sup>Standarded data + RidgeRegression</sup>')
fig = st.write(go.Figure(data=traces, layout=layout))
#fig.show()
    
    

    
#*****************************************************************************************************************

# # make a option tha users can choose using input or import the data
# import datetime
# from dateutil.relativedelta import relativedelta

# option = st.text_input("Please choose a new data insert method ('input' or 'import'): ")
# if option == 'input':
#     data_list = []
#     date_list = []
#     num_of_rows = int(input("Please enter how many rows of data you want: "))
#     for i in range(num_of_rows):
#         Gray_cement_dispatch_comercial = float(input("Row " + str(i+1) + " : Gray_cement_dispatch_comercial (450000-550000): "))
#         HTSales = float(input("Row " + str(i+1) + " : Housing Total Sales (10000-25000): "))
#         ISE = float(input("Row " + str(i+1) + " : ISE (90-120): "))
#         RADAR_Toiletslag_4 = float(input("Row " + str(i+1) + " : RADAR_Toilets lag_4 (eg. 123456789999): "))
#         holiday = float(input("Row " + str(i+1) + " : holiday days in the month (0-29): "))
#         Consumer_microcredit = float(input("Row " + str(i+1) + " : Consumer_microcredit (300000-800000): "))
#         items = [Gray_cement_dispatch_comercial, HTSales, ISE, RADAR_Toiletslag_4, holiday, Consumer_microcredit]
#         data_list.append(items)

#         start_date = "2022-01-01"
#         date = pd.to_datetime(start_date, format="%Y-%m-%d") + relativedelta(months=i)
#         date_list.append(date)

#     columns_name = ['Gray_cement_dispatch_comercial','HTSales', 'ISE', 'RADAR_Toiletslag_4','holiday', 'Consumer_microcredit']
#     data_df = pd.DataFrame(data=data_list, columns=columns_name, index=date_list)
#     data_df_copy = data_df.copy()
    
# elif option == 'import':
#     date_list = []
#     data_df_csv = pd.read_csv('new_data.csv')
#     for i in range(data_df_csv.shape[0]):
#         start_date = "2022-01-01"
#         date = pd.to_datetime(start_date, format="%Y-%m-%d") + relativedelta(months=i)
#         date_list.append(date)
#         columns_name = ['Gray_cement_dispatch_comercial','HTSales', 'ISE', 'RADAR_Toiletslag_4','holiday', 'Consumer_microcredit']
#     data_df_csv.index = date_list
#     data_df_copy = data_df_csv.copy()

# else: 
#     print("Please choose a legal option. ('input' or 'import')")
    
# #st.write(data_df_copy)


# st.header('Table with Predicted Values ')


# # standarded data + ridge regression

# corona_combined_var10['HTSales'] = corona_combined_var10['HTSales'].astype(float)
# corona_combined_var10['holiday'] = corona_combined_var10['holiday'].astype(float)

# X = corona_combined_var10.drop(['Quantity'], axis=1)
# y = corona_combined_var10['Quantity']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# St_X = Created_Functions.standardization(X_train)

# RidgeReg = Ridge(alpha=0.1)  #Default value for alpha = 1
# RidgeReg.fit(St_X, y_train)

# data_df_copy['Predict_quantity'] = RidgeReg.predict(Created_Functions.standardization(data_df_copy))
# #st.write(data_df_copy)

# predict_dataset = corona_combined_var10.append(data_df_copy)
# st.write(predict_dataset)
    

    
#********** Line Chart with prediction*********************

# import plotly.graph_objs as go

# traces = [go.Scatter(
#         x = predict_dataset.index,
#         y = predict_dataset[colname],
#         mode = 'markers+lines',
#         name = colname            
#         ) for colname in list(['Quantity', 'Predict_quantity'])]

# layout = go.Layout(title='Prediction vs Actual<br><sup>Standarded data + RidgeRegression</sup>')
# fig = st.write(go.Figure(data=traces, layout=layout))
# #fig.add_vline(x='2021-12-30', line_width=3, line_dash="dash", line_color="grey")

# #fig.show()
    
    

    

    
    
#with model:
    
    
# ----------------------------------------------------------------------------------
from metrics.get_metrics import get_data

from metrics.config import PATH_SAMPLES
filename: str = 'new_data.csv'
save_path = PATH_SAMPLES.joinpath(filename)


def generate_agrid(data: pd.DataFrame):
    gb = GridOptionsBuilder.from_dataframe(data)
    gb.configure_default_column(editable=True)  # Make columns editable
    gb.configure_pagination(paginationAutoPageSize=True)  # Add pagination
    gb.configure_side_bar()  # Add a sidebar
    gb.configure_selection('multiple', use_checkbox=True,
                           groupSelectsChildren="Group checkbox select children")  # Enable multi-row selection
    gridOptions = gb.build()

    grid_response = AgGrid(
        data,
        gridOptions=gridOptions,
        data_return_mode=DataReturnMode.AS_INPUT,
        update_on='MANUAL',  # <- Should it let me update before returning?
        fit_columns_on_grid_load=False,
        theme=AgGridTheme.STREAMLIT,  # Add theme color to the table
        enable_enterprise_modules=True,
        height=350,
        width='100%',
        # reload_data=True
    )

    data = grid_response['data']
    selected = grid_response['selected_rows']
    df = pd.DataFrame(selected)  # Pass the selected rows to a new dataframe df
    return grid_response

def update(grid_table: classmethod, filename: str = 'new_data.csv'):
    save_path = PATH_SAMPLES.joinpath(filename)
    grid_table_df = pd.DataFrame(grid_table['data'])
    grid_table_df.to_csv(save_path, index=False)

# First data gather
df = get_data() 

if __name__ == '__main__':
    # Start graphing
    grid_table = generate_agrid(df)
    
    # Update
    st.sidebar.button("Update", on_click=update, args=[grid_table])
    
    

