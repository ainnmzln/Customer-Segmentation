# -*- coding: utf-8 -*-
"""
Created on Wed May 18 16:02:57 2022

@author: ainnmzln
"""

import os
import pandas as pd
import pickle
from modules import ExploratoryDataAnalysis
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = os.path.join(os.getcwd(), 'statics','model.h5')
TEST_PATH=os.path.join(os.getcwd(),'new_customers.csv')
SCALER_PATH=os.path.join(os.getcwd(),'statics','ss_scaler.pkl')

#%% Load new data and dl model

new_customers= pd.read_csv(TEST_PATH)

mm=pickle.load(open(SCALER_PATH,'rb'))
model=pickle.load(open(MODEL_PATH,'rb'))

#%% Data cleaning

new_customers.drop(columns=['ID'],axis=1,inplace=True)

new_customers['Ever_Married'].fillna('Yes', inplace=True)
new_customers['Graduated'].fillna('Yes', inplace=True)
new_customers['Profession'].fillna('Artist', inplace=True)
new_customers['Family_Size'].fillna(new_customers['Family_Size'].median(), inplace=True)
new_customers['Work_Experience'].fillna(new_customers['Work_Experience'].median(), inplace=True)
new_customers.dropna(subset=['Var_1'], inplace=True)

le = LabelEncoder()

eda=ExploratoryDataAnalysis()

new_customers['Gender'] = le.fit_transform(new_customers['Gender'])  
new_customers['Ever_Married'] = le.fit_transform(new_customers['Ever_Married'])  
new_customers['Graduated'] = le.fit_transform(new_customers['Graduated'])        
new_customers['Profession'] = le.fit_transform(new_customers['Profession'])   
new_customers['Spending_Score'] = le.fit_transform(new_customers['Spending_Score']) 
new_customers['Var_1'] = le.fit_transform(new_customers['Var_1'])

X = new_customers.loc[:, 'Gender':'Var_1']

new_customers_scaled=mm.fit_transform(X)

result=model.predict(new_customers_scaled)
print(result)

