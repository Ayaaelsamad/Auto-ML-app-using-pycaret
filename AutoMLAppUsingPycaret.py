import pandas as pd
import numpy as np
import os
import pycaret as py
import streamlit as st 
from pycaret.classification import *
from streamlit_option_menu import option_menu
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report


# Your package should be able to load data, perform EDA, and train machine learning models.###
# It should also be able to automatically select regressors or classifiers, ###
#and it should allow the user to select which models to use. ###
#Your web app should allow the user to upload their data, select the target variable,###



def upload_data():
    
    st.title("Upload Your Data For Preprossesing..")
    file = st.file_uploader("Upload your dataset here")
    
    if file:
        df = pd.read_csv(file, index_col = None)
        df.to_csv("sourcedata.csv", index = None)
        st.dataframe(df)
        return df


def EDA(dataset):
    
    st.title("Automated Exploratory Data Analysis..")
    eda = dataset.profile_report()
    st_profile_report(eda)

def Train_model(dataset):
    
    st.title("Automated Machine Learning..")
    target = st.selectbox("Select Your Target", dataset.columns)
    model_names = ['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge', 'rf', 'qda', 'ada', 'gbc', 'lda', 'et', 'xgboost', 'lightgbm', 'dummy']
    chosen_model = st.selectbox("Select Your Desired ML Model", model_names)

    if st.button("Train Model"):
        setup(dataset, target = target)
        st.info("This is the ML experiment description..")
        setup_df = pull()
        st.dataframe(setup_df)
        
        st.info("This is the ML experiment results..")
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        
        st.info(f"This is the results of chosen Ml model ({chosen_model})..")
        trained_model = create_model(chosen_model)
        trained_model_results = pull()
        st.dataframe(trained_model_results)
        
        
        st.info("This is the best ML model for your dataset..")
        st.success(best_model)
        

def main():
    with st.sidebar: 
        st.title("AutoML App")
        st.image("https://thumbs.dreamstime.com/z/icon-internet-things-related-to-machine-learning-symbol-doodle-style-simple-design-editable-illustration-vector-icons-264749285.jpg?w=992")
        selected = option_menu("Navigation Menu", ["Uploading Data", "EDA", "ML Model"],
                               icons = ['cloud-arrow-up-fill', 'clipboard-data-fill', 'activity'], default_index=0)
    if os.path.exists("sourcedata.csv"):
        df = pd.read_csv("sourcedata.csv", index_col = None)
    if selected == "Uploading Data":
        upload_data()
    if selected == "EDA":
        EDA(df)
    if selected == "ML Model":
        Train_model(df)
    

main()