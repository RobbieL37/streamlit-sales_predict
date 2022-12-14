'''
Project: Corona sales prediction 
Author: Robbie
Date: 12/14/2022
'''


import streamlit as st


header = st.beta_container()
dataset = st.beta_container()
chart = st.beta_container()
model = st.beta_container()


with header: 
    st.title('Welcome to the testing website.')
    st.text('In this project I will create an interaction with tables and charts.')
    

with dataset:
    st.header('This is the dataset we will input data inside.')
    st.text('Here will be a table.')
    

with chart:
    st.header('This is the chart for pediction based on the data.')
    
    
with model:
    st.header('This is a space for modeling.')
    st.text('Here will be a demonstration of the fomula with the interpet and coefficients.')
    
    
    

