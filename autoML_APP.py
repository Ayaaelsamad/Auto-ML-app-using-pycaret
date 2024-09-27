import pandas as pd
import numpy as np
import os
import pycaret as py
import streamlit as st 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


def upload_data():
    
    st.info("Upload Your Data For Preprossesing..")
    file = st.file_uploader("Upload your dataset here")
    
    if file:
        df = pd.read_csv(file, index_col = None)
        df.to_csv("sourcedata.csv", index = None)
        st.dataframe(df)
        return df
    
def data_preprocessing(data, target):
    
    df = data.copy()
    
    #drop columns
    droped = st.multiselect('Choose columns you want to drop', df.columns)
    if droped:
        df.drop(columns=droped, axis=1, inplace=True)
        if target in df.columns:
            df.drop(columns=target, axis=1, inplace=True) 
    else:
        df.drop(columns=target, axis=1, inplace=True) 

    
        
    #handelling missing values
    categoric_imputer = None
    numeric_imputer = None
    
    numeric = st.selectbox('Choose your prefered way for filling missing values for \
                 numerical features', ['mean','median','most_frequent','constant'])            
    if numeric == 'mean':
        numeric_imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
    elif numeric == 'median':
        numeric_imputer = SimpleImputer(strategy='median', missing_values=np.nan)
    elif numeric == 'most_frequent':
        numeric_imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
    elif numeric == 'constant':
        value = st.number_input('Enter numeric value')
        if isinstance(value, int) or isinstance(value, float):
            numeric_imputer = SimpleImputer(strategy='constant', 
                                            missing_values=np.nan, fill_value= int(value))
        elif value is not None:
            st.write('Enter numeric value..')
    
    categoric = st.selectbox('Choose your prefered way for filling missing values for \
                  categorical features', ['most_frequent','constant'])    
    if categoric == 'most_frequent':
        categoric_imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
    elif categoric == 'constant':
        value = st.text_input('Enter a string')
        if isinstance(value, str):
            categoric_imputer = SimpleImputer(strategy='constant', 
                                            missing_values=np.nan, fill_value= value)
        else:
            st.write('Enter a string..')

    
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'category' or df[col].dtype == 'str':
            if categoric_imputer is not None:
                df[col] = categoric_imputer.fit_transform(df[col].values.reshape(-1,1))
        elif df[col].dtype == 'int64' or df[col].dtype == 'float64':
            if numeric_imputer is not None:
                df[col] = numeric_imputer.fit_transform(df[col].values.reshape(-1,1))
    
    #handeling dulicates
    df.drop_duplicates(inplace = True)
    
        
    #encoding categorical data
    encoded_value = st.selectbox('Choose your prefered way for encoding data', 
                                 ['Label encoder', 'one-hot encoder'])
    if encoded_value == 'Label encoder':
        label_encoder = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype == 'str':
                df[col] = label_encoder.fit_transform(df[col])
                
    elif encoded_value == 'one-hot encoder':
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype == 'str':
                df = pd.get_dummies(df, columns=[col], drop_first=True)

    #scalling data
    scale_value = st.selectbox('Enter your prefered way for data scaling', 
                               ['min-max scaler', 'standard scaler'])
    if scale_value == 'min-max scaler':
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    elif scale_value == 'standard scaler':
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    return scaled_df
    
def train_model(df):
    
    st.info('Choose your prefrences for training the model..')
    target = st.selectbox('Select your traget', df.columns)
    data_preprocessing(df, target)

    model_names = ['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 
                   'ridge', 'rf', 'qda', 'ada', 'gbc', 'lda', 'et', 'xgboost', 'lightgbm', 'dummy']
    chosen_model = st.selectbox('Select your desired ML model', model_names)
    
    
    if(st.button('Train Model')):
        if df[target].dtype in ['int64', 'float64']:
            from pycaret.classification import setup, create_model, pull, eda, compare_models
            setup(df, target=target)
            setup_df = pull()
            
            st.info(f'This is the results of the chosen ML model ({chosen_model}) using classification..')
            trained_model = create_model(chosen_model)
            trained_model_results = pull()
            st.dataframe(trained_model_results)
            
            st.info("This is the ML experiment description..")
            st.dataframe(setup_df)
            
            st.info('This is the ML experiment results..')
            best_model = compare_models()
            compared_df = pull()
            st.dataframe(compared_df)
            
            st.info("This is the best ML model for your dataset..")
            st.success(best_model)
            
        else:
            
            from pycaret.regression import setup, create_model, pull, eda, compare_models
            setup(df, target=target)
            setup_df = pull()
            
            eda(display_format = 'svg')
            
            st.info(f'This is the results of the chosen ML model ({chosen_model}) using regression..')
            trained_model = create_model(chosen_model, return_train_score = True)
            trained_model_results = pull()
            st.dataframe(trained_model_results)
            
            st.info('This is the ML experiment results..')
            st.dataframe(setup_df)
            best_model = compare_models()
            compared_df = pull()
            st.dataframe(compared_df)
            st.success(best_model)
        
        
    
def main():
    
    st.title('AUTO ML APP')
    df = upload_data()

    if df is not None:
        train_model(df)
        

main()