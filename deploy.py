# -*- coding: utf-8 -*-
"""
@author: ainnmzln
"""

import os
import pandas as pd
import pickle
from module import ExploratoryDataAnalysis

MODEL_PATH = os.path.join(os.getcwd(), 'statics','model.h5')
TEST_PATH=os.path.join(os.getcwd(),'new_customers.csv')
SCALER_PATH=os.path.join(os.getcwd(),'statics','scaler.pkl')

#%% Load new data and dl model

new_customers= pd.read_csv(TEST_PATH)

scaler=pickle.load(open(SCALER_PATH,'rb'))
model=pickle.load(open(MODEL_PATH,'rb'))

#%% Data cleaning
nw_drop=new_customers.drop(['ID','Work_Experience'],axis = 1)


em=nw_drop['Ever_Married']
gen=nw_drop['Gender']
grad=nw_drop['Graduated']
prof=nw_drop['Profession']
fam=nw_drop['Family_Size']
var=nw_drop['Var_1']
spen=nw_drop['Spending_Score']

eda=ExploratoryDataAnalysis()

em=eda.encoder(em)
gen=eda.encoder(gen)
grad=eda.encoder(grad)
prof=eda.encoder(prof)
fam=eda.encoder(fam)
var=eda.encoder(var)
spen=eda.encoder(spen)

nw_drop['Ever_Married']=eda.impute(em)
nw_drop['Gender']=eda.impute(grad)
nw_drop['Graduated']=eda.impute(grad)
nw_drop['Profession']=eda.impute(prof)
nw_drop['Family_Size']=eda.impute(fam)
nw_drop['Var_1']=eda.impute(var)
nw_drop['Spending_Score']=eda.impute(spen)

x= nw_drop.loc[:, 'Gender':'Var_1']

# Scaled X using MinMax Scaler 

scaler=pickle.load(open(SCALER_PATH,'rb'))
x_scaled = scaler.fit_transform(x)

result=model.predict(x_scaled)
print(result)


