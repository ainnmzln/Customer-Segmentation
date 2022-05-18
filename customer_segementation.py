# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:14:13 2022

@author: ainnmzln
"""

import os
import datetime
import pickle 
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Sequential 
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from modules import ModelEvaluation

#%%
PATH = os.path.join(os.getcwd(),'train.csv')
MODEL_SAVE_PATH=os.path.join(os.getcwd(),'statics','model.h5')
LOG_PATH=os.path.join(os.getcwd(),'log')
SCALER_PATH=os.path.join(os.getcwd(),'statics','ss_scaler.pkl')

#%% EDA

#%% Step 1 Load data

df = pd.read_csv(PATH)
df_copy=df.copy()        # duplicate the original dataset for comparision

#%% Step 2 Data inspection

df.info()
df.describe().T

# To check duplicates data 
df.duplicated().sum()

#%% Step 3 Data visualization

#%% Data cleaning 


# Drop the ID and work experience since has no correlation with other features, therefore drop ID column

df.drop(columns=['ID'],axis=1,inplace=True)

# Replace with No

df['Ever_Married'].fillna('Yes', inplace=True)
df['Graduated'].fillna('Yes', inplace=True)
df['Profession'].fillna('Artist', inplace=True)


# Fill NaN in Family_Size with mean
df['Family_Size'].fillna(df['Family_Size'].median(), inplace=True)
df['Work_Experience'].fillna(df['Work_Experience'].median(), inplace=True)

# Remove row with missing value in Var_1 and Profession

df.dropna(subset=['Var_1'], inplace=True)

df_drop=df.copy()

# Covert Gender, Ever Married, Graduated, Profession, Spending Score, Var 
# to numeric data with Label Encoder

le = LabelEncoder()

# Two class categorical 

df_drop['Gender'] = le.fit_transform(df['Gender']) 
df_drop['Ever_Married'] = le.fit_transform(df['Ever_Married'])  
df_drop['Graduated'] = le.fit_transform(df['Graduated'])        


df_drop['Spending_Score'] = le.fit_transform(df['Spending_Score']) 

# Multi class categorical

df_drop['Var_1'] = le.fit_transform(df['Var_1'])
df_drop['Profession'] = le.fit_transform(df['Profession'])   

df_drop.info()


enc=OneHotEncoder(sparse=False)
df_drop['Segmentation']=enc.fit_transform(np.expand_dims(df['Segmentation'],
                                                        axis=-1))    
                                                        
df_drop['Segmentation'] = le.fit_transform(df['Segmentation'])

# df_drop['Segmentation']=enc.fit_transform(np.expand_dims(df['Segmentation'],
#                                                         axis=-1))    

#%% Step 4 Features selection

X = df_drop.loc[:, 'Gender':'Var_1']
Y = df_drop.loc[:, 'Segmentation']

#sns.heatmap(df_drop.corr(), center=0, square=True, linewidths=1)
# Age, marital status and profession affect segmentation


print(df_drop.corr())

X.boxplot()

# Data scalling

#X=X[['Age','Spending_Score','Ever_Married']]

scaler= MinMaxScaler()
X_scaled = scaler.fit_transform(X)


pickle.dump(scaler,open(SCALER_PATH,'wb'))  # save scaler into pickle


# Step 5 Data preprocessing

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, 
                                                    test_size=0.3, 
                                                    random_state=42)
# Step 6 Deep learning model

model=Sequential()
model.add(Input(X_train.shape[1:]))
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(32,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(1,activation='softmax'))

model.summary()

plot_model(model)

model.compile(loss='CategoricalCrossentropy', 
              optimizer='adam', 
              metrics='acc')

early_stopping_callback=EarlyStopping(monitor='val_loss',patience=3)

log_files=os.path.join(LOG_PATH,datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tensorboard_callback=TensorBoard(log_dir=log_files,histogram_freq=1)


hist=model.fit(X_train, y_train, epochs=100,
               validation_data=(X_test,y_test),
               callbacks=[early_stopping_callback,
                          tensorboard_callback])

# Step 6 Machine learning model

steps_logis=[('Logis',LogisticRegression(solver='liblinear'))]
steps_knn=[('Logis',KNeighborsClassifier(n_neighbors=10))]
steps_forest=[('Forest',RandomForestClassifier(n_estimators=10))]
steps_svc=[('SVC',SVC())]
steps_tree=[('Tree',DecisionTreeClassifier())]
  
logis_pipeline=Pipeline(steps_logis)
knn_pipeline=Pipeline(steps_knn)
forest_pipeline=Pipeline(steps_forest)
svc_pipeline=Pipeline(steps_svc) 
tree_pipeline=Pipeline(steps_tree)

pipelines=[logis_pipeline,knn_pipeline,forest_pipeline,svc_pipeline,
            tree_pipeline]

# To fit the data in pipeline
for pipe in pipelines:
    pipe.fit(X_train,y_train)
    
    
#%% Step 7 Evaluation model


pipe_dict = {0:'Logistic Regression', 1:'KNN', 2: 'Random Forest', 
             3: 'SVC',4: 'Decision Tree'}

# Print the accuracy score

for index,model in enumerate(pipelines):
    y_pred = model.predict(X_test)
    print("{} Accuracy Score: {}".format(pipe_dict[index],model.score(X_test, y_test)*100 ))
    #print(classification_report(y_test, y_pred)

#%% Step 8 Model deployment

#model.save(MODEL_SAVE_PATH)
pickle.dump(svc_pipeline, open(MODEL_SAVE_PATH, 'wb'))
