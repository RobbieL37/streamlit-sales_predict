# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 12:23:52 2022

@author: Team Acceleration 
"""
import os
import gc
import datetime

import pandas as pd
pd.options.display.max_rows = 2000
pd.options.display.max_columns = 100

import numpy as np

import seaborn as sns
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

import matplotlib.pyplot as plt

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
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


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

"""
-----------------------------------------------------------
"""

class Created_Functions():
    def __init__(self, df): 
        self.df = df
        self.n_lags = 12
        self.test_size = 0.3
        self.random_state = 5
        self.alpha = 0.1
    
    
    # this test function is showing first 10 rows of the df 
    def test(self):
        print(self.df.head(10))
        #print('success')
    
        
    # get the dataframe's name
    def get_df_name(data):
        name =[x for x in globals() if globals()[x] is data][0]
        return name
    
    
    # get the lag_variables
    def lag_variable(variable, n_lags):
        data=pd.DataFrame()
        variables_name=variable.columns.values
        for i in range(1,(n_lags+1)):
            for j in variables_name:
                name=str(j)+'lag_'+ str(i)
                variable[name]=variable[j].shift(i)
        #data = variable.dropna()  
        data = variable
        return data   
    
    
    # functions for data transformation
    def centering(data):
        new_data = (data-data.mean())
        return new_data

    def nomalization(data): 
        new_data = (data-data.min())/(data.max()-data.min())
        return new_data

    def standardization(data):
        new_data = (data-data.mean())/data.std()
        return new_data

    def L1(data):
        #new_data = normalize(data, norm='l1')
        new_data = pd.DataFrame(normalize(data, norm='l1'))
        new_data = new_data.rename(columns = {0: "Quantity"})
        return new_data

    def L2(data):
        #new_data = normalize(data, norm='l2')
        new_data = pd.DataFrame(normalize(data, norm='l2'))
        new_data = new_data.rename(columns = {0: "Quantity"})
        return new_data    

    def Max(data):
        new_data = pd.DataFrame(normalize(data, norm='max'))
        new_data = new_data.rename(columns = {0: "Quantity"})
        return new_data        
    
        
    # linear regression function
    def linearR(X_train, X_test, y_train, y_test):
        # X = data.drop(['Quantity'], axis=1)
        # y = data['Quantity']

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    
        LinearReg = LinearRegression() 
        LinearReg.fit(X_train, y_train) 

        y_Train_prediction = LinearReg.predict(X_train) 
        y_Test_prediction  = LinearReg.predict(X_test)  
        
        #Prediction_Plots(y_train, y_Train_prediction, y_test, y_Test_prediction)
        #Metrics_Printer(y_train, y_Train_prediction, y_test, y_Test_prediction)
        print("Linear regression: ")
        #print(" ")
        print('Intercept:',LinearReg.intercept_)
        print('Coefficients:', LinearReg.coef_)
        print(" ")
        print("training metrics: ", metrics.r2_score(y_train, y_Train_prediction), "    testing metrics: ", metrics.r2_score(y_test, y_Test_prediction))
        #print("testing metrics: ", metrics.r2_score(y_test, y_Test_prediction))
        print(" ")
        
    # ridge regression function
    def ridgeR(X_train, X_test, y_train, y_test, alpha):
        # X = data.drop(['Quantity'], axis=1)
        # y = data['Quantity']

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    
        RidgeReg = Ridge(alpha=alpha)  #Default value for alpha = 1
        RidgeReg.fit(X_train, y_train)

        y_Train_prediction = RidgeReg.predict(X_train) 
        y_Test_prediction  = RidgeReg.predict(X_test)  
    
        #Prediction_Plots(y_train, y_Train_prediction, y_test, y_Test_prediction)
        #Metrics_Printer(y_train, y_Train_prediction, y_test, y_Test_prediction)
        print("Ridge regression: ")
        #print(" ")
        print('Intercept:',RidgeReg.intercept_)
        print('Coefficients:', RidgeReg.coef_)
        print(" ")
        print("training metrics: ", metrics.r2_score(y_train, y_Train_prediction), "    testing metrics: ", metrics.r2_score(y_test, y_Test_prediction))
        print(" ")
        
    # lasso regression function
    def lassoR(X_train, X_test, y_train, y_test, alpha):
        # X = data.drop(['Quantity'], axis=1)
        # y = data['Quantity']

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    
        LassoReg = Lasso(alpha = alpha)  
        LassoReg.fit(X_train, y_train)

        y_Train_prediction = LassoReg.predict(X_train) #Predictions on training model
        y_Test_prediction  = LassoReg.predict(X_test)  #Predictions on testing model 
    
        #Prediction_Plots(y_train, y_Train_prediction, y_test, y_Test_prediction)
        #Metrics_Printer(y_train, y_Train_prediction, y_test, y_Test_prediction)
        print("Lasso regression: ")
        #print(" ")
        print('Intercept:',LassoReg.intercept_)
        print('Coefficients:', LassoReg.coef_)
        print(" ")
        print("training metrics: ", metrics.r2_score(y_train, y_Train_prediction), "    testing metrics: ", metrics.r2_score(y_test, y_Test_prediction))
        print(" ")
    
    # lasso regression function
    def elasticnetR(X_train, X_test, y_train, y_test, alpha, l1_ratio):
        # X = data.drop(['Quantity'], axis=1)
        # y = data['Quantity']

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    
        ElasticNetReg = ElasticNet(alpha = alpha, l1_ratio=l1_ratio)  #Default value for alpha = 1
        ElasticNetReg.fit(X_train, y_train)

        y_Train_prediction = ElasticNetReg.predict(X_train) #Predictions on training model
        y_Test_prediction  = ElasticNetReg.predict(X_test)  #Predictions on testing model 
    
        #Prediction_Plots(y_train, y_Train_prediction, y_test, y_Test_prediction)
        #Metrics_Printer(y_train, y_Train_prediction, y_test, y_Test_prediction)
        print("ElasticNet regression: ")
        #print(" ")
        print('Intercept:',ElasticNetReg.intercept_)
        print('Coefficients:', ElasticNetReg.coef_)
        print(" ")
        print("training metrics: ", metrics.r2_score(y_train, y_Train_prediction), "    testing metrics: ", metrics.r2_score(y_test, y_Test_prediction))
        print(" ")
    
    
    #decisionTree function
    def decisionTree(X_train, X_test, y_train, y_test, alpha):
        
        TreeReg = DecisionTreeRegressor() #Creates the function
        TreeReg.fit(X_train, y_train) #Train the model  

        y_Train_prediction = TreeReg.predict(X_train) #Predictions on training model
        y_Test_prediction = TreeReg.predict(X_test)
        
        print("Decision Tree: ")
        print(" ")
        print("training metrics: ", metrics.r2_score(y_train, y_Train_prediction), "    testing metrics: ", metrics.r2_score(y_test, y_Test_prediction))
        print(" ")    

    
    # scatter plot 
    def Prediction_Plots(y_train, y_Train_prediction, y_test, y_Test_prediction):
        fig, ax = plt.subplots(ncols=2, figsize=(10,4))
        #Training
        ax[0].scatter(y_train, y_Train_prediction)
        ax[0].set_ylim(-2,2)
        ax[0].set_xlim(-2,2)
        ax[0].grid()
        ax[0].set_xlabel('y')
        ax[0].set_ylabel('yhat')
        ax[0].set_title('Training Set')
        #Testing
        ax[1].scatter(y_test, y_Test_prediction)
        ax[1].set_ylim(-2,2)
        ax[1].set_xlim(-2,2)
        ax[1].grid()
        ax[1].set_xlabel('y')
        ax[1].set_ylabel('yhat')
        ax[1].set_title('Testing Set')
        plt.show()
    
        return()    
    
    
    # Matrics
    def Metrics_Printer(y_train, y_Train_prediction, y_test, y_Test_prediction):
        #Training 
        print('Training Metrics:')
        print('R squared:', metrics.r2_score(y_train, y_Train_prediction))
        #print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_Train_prediction))  
        #print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_Train_prediction))  
        #print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_Train_prediction)))

        print('\nTesting Metrics:')
        print('R squared:', metrics.r2_score(y_test, y_Test_prediction))
        #print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_Test_prediction))  
        #print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_Test_prediction))  
        #print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_Test_prediction)))
    
        return()


    # Box Plots function
    def Multiple_Runner(x, y):
    
        Train_MSE = [] #Empty list to Store MSEs for training data set
        Test_MSE = []  #Empty list to Store MSEs for testing data set

        Train_R2 = [] #Empty list to Store R2s for training data set
        Test_R2 = []  #Empty list to Store R2s for testing data set

        for i in range(1000):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        
            model = LinearRegression() #Creates the function
            model.fit(x_train, y_train) #Train the model
    
            y_Train_prediction  = model.predict(x_train)  #Predictions on training model
            y_Test_prediction   = model.predict(x_test)   #Predictions on testing model
    
            train_R2 = metrics.r2_score(y_train, y_Train_prediction) #Obtaining the metrics
            test_R2  = metrics.r2_score(y_test, y_Test_prediction)
    
            train_MSE = metrics.mean_squared_error(y_train, y_Train_prediction)
            test_MSE  = metrics.mean_squared_error(y_test, y_Test_prediction)
    
            Train_MSE.append(train_MSE) #Storing the metrics in the lists
            Test_MSE.append(test_MSE) 
    
            Train_R2.append(train_R2) #Storing the metrics in the lists
            Test_R2.append(test_R2)  
    
        print('Train MSE median:', np.median(Train_MSE))
        print('Test MSE median:', np.median(Test_MSE))

        print('\nTrain_R2 median:', np.median(Train_R2))
        print('Test_R2 median:', np.median(Test_R2))

        fig, ax = plt.subplots(ncols=2, figsize=(10,4))

        ax[0].boxplot([Train_MSE, Test_MSE])
        ax[0].set_xticks([1,2],minor = False)                   #setting boxplot names
        ax[0].set_xticklabels(['Train','Test'], minor = False)  #setting boxplot names
        ax[0].grid()
        ax[0].set_title('Mean Squared Error')

        ax[1].boxplot([Train_R2, Test_R2])
        ax[1].set_xticks([1,2],minor = False)
        ax[1].set_xticklabels(['Train','Test'], minor = False)
        ax[1].grid()
        ax[1].set_title('R squared')

        plt.show()

        print('Train MSE standard deviation:', np.std(Train_MSE))
        print('Test MSE standard deviation:', np.std(Test_MSE))

        print('\nTrain_R2 standard deviation:', np.std(Train_R2))
        print('Test_R2 standard deviation:', np.std(Test_R2))
    
    
    def Metrics(y_test, y_pred_Test):
        print('Test Metrics:')
        print('R squared:', metrics.r2_score(y_test, y_pred_Test))
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_Test))  
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_Test))  
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_Test)))
        
        return

    
    def Predicted_Plot(y_train, y_pred_Train, y_test, y_pred_Test):
    
        fig, ax = plt.subplots(ncols=2, figsize=(10,4))
    
        ax[0].scatter(y_train, y_pred_Train)
        ax[0].grid()
        ax[0].set_xlabel('Observed Label')
        ax[0].set_ylabel('Predicted Label')
        ax[0].set_title('Training Set')
    
        ax[1].scatter(y_test, y_pred_Test)
        ax[1].grid()
        ax[1].set_xlabel('Observed Label')
        ax[1].set_ylabel('Predicted Label')
        ax[1].set_title('Testing Set')
        plt.show()
        
        return   
    

    def Multiple_Runs(model,X, y):
        
        Train_MSE = [] #Empty list to Store MSEs for training data set
        Test_MSE = []  #Empty list to Store MSEs for testing data set
    
        Train_R2 = [] #Empty list to Store R2s for training data set
        Test_R2 = []  #Empty list to Store R2s for testing data set
    
        for i in tqdm(range(100)):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            
            mean = X_train.mean()
            stdev = X_train.std()
            
            X_train_st = (X_train - mean)/stdev 
            X_test_st = (X_test - mean)/stdev 
        
            model.fit(X_train_st, y_train) #Train the model
       
            y_pred_Train  = model.predict(X_train_st)  #Predictions on training model
            y_pred_Test   = model.predict(X_test_st)   #Predictions on testing model
        
            train_R2 = metrics.r2_score(y_train, y_pred_Train) #Obtaining the metrics
            test_R2  = metrics.r2_score(y_test, y_pred_Test)
        
            train_MSE = metrics.mean_squared_error(y_train, y_pred_Train)
            test_MSE  = metrics.mean_squared_error(y_test, y_pred_Test)
        
            Train_MSE.append(train_MSE) #Storing the metrics in the lists
            Test_MSE.append(test_MSE) 
        
            Train_R2.append(train_R2) #Storing the metrics in the lists
            Test_R2.append(test_R2)  
        
        print('Train MSE median:', np.median(Train_MSE))
        print('Test MSE median:', np.median(Test_MSE))
    
        print('\nTrain_R2 median:', np.median(Train_R2))
        print('Test_R2 median:', np.median(Test_R2))
    
        fig, ax = plt.subplots(ncols=2, figsize=(10,4))
    
        ax[0].boxplot([Train_MSE, Test_MSE])
        ax[0].set_xticks([1,2],minor = False)                   #setting boxplot names
        ax[0].set_xticklabels(['Train','Test'], minor = False)  #setting boxplot names
        ax[0].grid()
        ax[0].set_title('Mean Squared Error')
    
        ax[1].boxplot([Train_R2, Test_R2])
        ax[1].set_xticks([1,2],minor = False)
        ax[1].set_xticklabels(['Train','Test'], minor = False)
        ax[1].grid()
        ax[1].set_title('R squared')
    
        plt.show()
    
        print('Train MSE standard deviation:', np.std(Train_MSE))
        print('Test MSE standard deviation: ', np.std(Test_MSE))
    
        print('\nTrain_R2 standard deviation:', np.std(Train_R2))
        print('Test_R2 standard deviation: ', np.std(Test_R2))  
        

    def Model_Performance(model,X,y):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            
        mean = X_train.mean()
        stdev = X_train.std()
            
        X_train_st = (X_train - mean)/stdev 
        X_test_st = (X_test - mean)/stdev
        
        model.fit(X_train_st,y_train) 
    
        y_pred_Train = model.predict(X_train_st) #Predictions
        y_pred_Test = model.predict(X_test_st) #Predictions
        
        Created_Functions.Metrics(y_test, y_pred_Test)  # should be updated 
        
        Created_Functions.Predicted_Plot(y_train, y_pred_Train, y_test, y_pred_Test)  # should be updated
        
        Created_Functions.Multiple_Runs(model,X, y)  # should be updated
        
        return
    
    
    def data_split(data, test_size, random_state):
        X = data.drop(['Quantity'], axis=1)
        y = data['Quantity']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        return X_train, X_test, y_train, y_test
    
    
    def Center(X_train, X_test):
        train_mean = X_train.mean()
        
        centered_x_train = X_train - train_mean
        centered_x_test = X_test - train_mean
        
        return centered_x_train, centered_x_test
    

    # new functions for data transformation
    def MinMax_Nomal(X_train, X_test):
        scaler = MinMaxScaler()
        scaler.fit(X_train)

        normalized_x_train = pd.DataFrame(scaler.transform(X_train),
                                index = X_train.index, columns = X_train.columns)
        normalized_x_test = pd.DataFrame(scaler.transform(X_test),
                                index = X_test.index, columns = X_test.columns)
        
        return normalized_x_train, normalized_x_test
    
    
    # new functions for data transformation
    def Standard(X_train, X_test):
        scaler = StandardScaler()
        scaler.fit(X_train)

        standarded_x_train = pd.DataFrame(scaler.transform(X_train),
                                index = X_train.index, columns = X_train.columns)
        standarded_x_test = pd.DataFrame(scaler.transform(X_test),
                                index = X_test.index, columns = X_test.columns)
        
        return standarded_x_train, standarded_x_test


    def Model_Multiple(model_name, data, test_size):
    
        Train_MSE = [] #Empty list to Store MSEs for training data set
        Test_MSE = []  #Empty list to Store MSEs for testing data set
    
        Train_R2 = [] #Empty list to Store R2s for training data set
        Test_R2 = []  #Empty list to Store R2s for testing data set
    
        for i in tqdm(range(1000)):        
            X = data.drop(['Quantity'], axis=1)
            y = data['Quantity']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            
            model = model_name
            model.fit(X_train, y_train) #Train the model
       
            y_pred_Train  = model.predict(X_train)  #Predictions on training model
            y_pred_Test   = model.predict(X_test)   #Predictions on testing model
        
            train_R2 = metrics.r2_score(y_train, y_pred_Train) #Obtaining the metrics
            test_R2  = metrics.r2_score(y_test, y_pred_Test)
        
            train_MSE = metrics.mean_squared_error(y_train, y_pred_Train)
            test_MSE  = metrics.mean_squared_error(y_test, y_pred_Test)
        
            Train_MSE.append(train_MSE) #Storing the metrics in the lists
            Test_MSE.append(test_MSE) 
        
            Train_R2.append(train_R2) #Storing the metrics in the lists
            Test_R2.append(test_R2)  
        
        print('Train MSE median:', np.median(Train_MSE))
        print('Test MSE median:', np.median(Test_MSE))
    
        print('\nTrain_R2 median:', np.median(Train_R2))
        print('Test_R2 median:', np.median(Test_R2))
    
        # fig, ax = plt.subplots(ncols=2, figsize=(10,4))
    
        # ax[0].boxplot([Train_MSE, Test_MSE])
        # ax[0].set_xticks([1,2],minor = False)                   #setting boxplot names
        # ax[0].set_xticklabels(['Train','Test'], minor = False)  #setting boxplot names
        # ax[0].grid()
        # ax[0].set_title('Mean Squared Error')
    
        # ax[1].boxplot([Train_R2, Test_R2])
        # ax[1].set_xticks([1,2],minor = False)
        # ax[1].set_xticklabels(['Train','Test'], minor = False)
        # ax[1].grid()
        # ax[1].set_title('R squared')
    
        # plt.show()
    
        print('\nTrain MSE standard deviation:', np.std(Train_MSE))
        print('Test MSE standard deviation: ', np.std(Test_MSE))
    
        print('\nTrain_R2 standard deviation:', np.std(Train_R2))
        print('Test_R2 standard deviation: ', np.std(Test_R2))
     
    
    # Multiple runs for normalized data transformation
    def Model_Multiple_Normalized(model_name, data, test_size):
    
        Train_MSE = [] #Empty list to Store MSEs for training data set
        Test_MSE = []  #Empty list to Store MSEs for testing data set
    
        Train_R2 = [] #Empty list to Store R2s for training data set
        Test_R2 = []  #Empty list to Store R2s for testing data set
    
        for i in tqdm(range(1000)):        
            X = data.drop(['Quantity'], axis=1)
            y = data['Quantity']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            
            normalized_x_train, normalized_x_test = Created_Functions.MinMax_Nomal(X_train, X_test)
            
            model = model_name
            model.fit(normalized_x_train, y_train) #Train the model
       
            y_pred_Train  = model.predict(normalized_x_train)  #Predictions on training model
            y_pred_Test   = model.predict(normalized_x_test)   #Predictions on testing model
        
            train_R2 = metrics.r2_score(y_train, y_pred_Train) #Obtaining the metrics
            test_R2  = metrics.r2_score(y_test, y_pred_Test)
        
            train_MSE = metrics.mean_squared_error(y_train, y_pred_Train)
            test_MSE  = metrics.mean_squared_error(y_test, y_pred_Test)
        
            Train_MSE.append(train_MSE) #Storing the metrics in the lists
            Test_MSE.append(test_MSE) 
        
            Train_R2.append(train_R2) #Storing the metrics in the lists
            Test_R2.append(test_R2)  
        
        print('Train MSE median:', np.median(Train_MSE))
        print('Test MSE median:', np.median(Test_MSE))
    
        print('\nTrain_R2 median:', np.median(Train_R2))
        print('Test_R2 median:', np.median(Test_R2))
    
        # fig, ax = plt.subplots(ncols=2, figsize=(10,4))
    
        # ax[0].boxplot([Train_MSE, Test_MSE])
        # ax[0].set_xticks([1,2],minor = False)                   #setting boxplot names
        # ax[0].set_xticklabels(['Train','Test'], minor = False)  #setting boxplot names
        # ax[0].grid()
        # ax[0].set_title('Mean Squared Error')
    
        # ax[1].boxplot([Train_R2, Test_R2])
        # ax[1].set_xticks([1,2],minor = False)
        # ax[1].set_xticklabels(['Train','Test'], minor = False)
        # ax[1].grid()
        # ax[1].set_title('R squared')
    
        # plt.show()
    
        print('\nTrain MSE standard deviation:', np.std(Train_MSE))
        print('Test MSE standard deviation: ', np.std(Test_MSE))
    
        print('\nTrain_R2 standard deviation:', np.std(Train_R2))
        print('Test_R2 standard deviation: ', np.std(Test_R2))
        
    
    # Multiple runs for normalized data transformation
    def Model_Multiple_Standarded(model_name, data, test_size):
    
        Train_MSE = [] #Empty list to Store MSEs for training data set
        Test_MSE = []  #Empty list to Store MSEs for testing data set
    
        Train_R2 = [] #Empty list to Store R2s for training data set
        Test_R2 = []  #Empty list to Store R2s for testing data set
    
        for i in tqdm(range(1000)):        
            X = data.drop(['Quantity'], axis=1)
            y = data['Quantity']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            
            standarded_x_train, standarded_x_test = Created_Functions.Standard(X_train, X_test)
            
            model = model_name
            model.fit(standarded_x_train, y_train) #Train the model
       
            y_pred_Train  = model.predict(standarded_x_train)  #Predictions on training model
            y_pred_Test   = model.predict(standarded_x_test)   #Predictions on testing model
        
            train_R2 = metrics.r2_score(y_train, y_pred_Train) #Obtaining the metrics
            test_R2  = metrics.r2_score(y_test, y_pred_Test)
        
            train_MSE = metrics.mean_squared_error(y_train, y_pred_Train)
            test_MSE  = metrics.mean_squared_error(y_test, y_pred_Test)
        
            Train_MSE.append(train_MSE) #Storing the metrics in the lists
            Test_MSE.append(test_MSE) 
        
            Train_R2.append(train_R2) #Storing the metrics in the lists
            Test_R2.append(test_R2)  
        
        print('Train MSE median:', np.median(Train_MSE))
        print('Test MSE median:', np.median(Test_MSE))
    
        print('\nTrain_R2 median:', np.median(Train_R2))
        print('Test_R2 median:', np.median(Test_R2))
    
        # fig, ax = plt.subplots(ncols=2, figsize=(10,4))
    
        # ax[0].boxplot([Train_MSE, Test_MSE])
        # ax[0].set_xticks([1,2],minor = False)                   #setting boxplot names
        # ax[0].set_xticklabels(['Train','Test'], minor = False)  #setting boxplot names
        # ax[0].grid()
        # ax[0].set_title('Mean Squared Error')
    
        # ax[1].boxplot([Train_R2, Test_R2])
        # ax[1].set_xticks([1,2],minor = False)
        # ax[1].set_xticklabels(['Train','Test'], minor = False)
        # ax[1].grid()
        # ax[1].set_title('R squared')
    
        # plt.show()
    
        print('\nTrain MSE standard deviation:', np.std(Train_MSE))
        print('Test MSE standard deviation: ', np.std(Test_MSE))
    
        print('\nTrain_R2 standard deviation:', np.std(Train_R2))
        print('Test_R2 standard deviation: ', np.std(Test_R2))
        
        
    # Multiple runs for centered data transformation
    def Model_Multiple_Centered(model_name, data, test_size):
    
        Train_MSE = [] #Empty list to Store MSEs for training data set
        Test_MSE = []  #Empty list to Store MSEs for testing data set
    
        Train_R2 = [] #Empty list to Store R2s for training data set
        Test_R2 = []  #Empty list to Store R2s for testing data set
    
        for i in tqdm(range(1000)):        
            X = data.drop(['Quantity'], axis=1)
            y = data['Quantity']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            
            standarded_x_train, standarded_x_test = Created_Functions.Center(X_train, X_test)
            
            model = model_name
            model.fit(standarded_x_train, y_train) #Train the model
       
            y_pred_Train  = model.predict(standarded_x_train)  #Predictions on training model
            y_pred_Test   = model.predict(standarded_x_test)   #Predictions on testing model
        
            train_R2 = metrics.r2_score(y_train, y_pred_Train) #Obtaining the metrics
            test_R2  = metrics.r2_score(y_test, y_pred_Test)
        
            train_MSE = metrics.mean_squared_error(y_train, y_pred_Train)
            test_MSE  = metrics.mean_squared_error(y_test, y_pred_Test)
        
            Train_MSE.append(train_MSE) #Storing the metrics in the lists
            Test_MSE.append(test_MSE) 
        
            Train_R2.append(train_R2) #Storing the metrics in the lists
            Test_R2.append(test_R2)  
        
        print('Train MSE median:', np.median(Train_MSE))
        print('Test MSE median:', np.median(Test_MSE))
    
        print('\nTrain_R2 median:', np.median(Train_R2))
        print('Test_R2 median:', np.median(Test_R2))
    
        # fig, ax = plt.subplots(ncols=2, figsize=(10,4))
    
        # ax[0].boxplot([Train_MSE, Test_MSE])
        # ax[0].set_xticks([1,2],minor = False)                   #setting boxplot names
        # ax[0].set_xticklabels(['Train','Test'], minor = False)  #setting boxplot names
        # ax[0].grid()
        # ax[0].set_title('Mean Squared Error')
    
        # ax[1].boxplot([Train_R2, Test_R2])
        # ax[1].set_xticks([1,2],minor = False)
        # ax[1].set_xticklabels(['Train','Test'], minor = False)
        # ax[1].grid()
        # ax[1].set_title('R squared')
    
        # plt.show()
    
        print('\nTrain MSE standard deviation:', np.std(Train_MSE))
        print('Test MSE standard deviation: ', np.std(Test_MSE))
    
        print('\nTrain_R2 standard deviation:', np.std(Train_R2))
        print('Test_R2 standard deviation: ', np.std(Test_R2))
        