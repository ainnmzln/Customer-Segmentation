# -*- coding: utf-8 -*-
"""
@author: ACER
"""

import os,pickle,datetime
import pandas as pd
import seaborn as sns
import missingno as msno
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
from module import ExploratoryDataAnalysis

#%% constant

PATH=os.path.join(os.getcwd(),'train.csv')
SCALER_PATH =os.path.join(os.getcwd(),'statics','scaler.pkl')
LOG_PATH=os.path.join(os.getcwd(),'log')
MODEL_SAVE_PATH=os.path.join(os.getcwd(),'statics','model.h5')

#%% EDA 

#%% Step 1. Data loading

df=pd.read_csv(PATH)
df.head()

#%% Step 2. Data inspection

# continuous data: age, working experience, family size
# categorical data: gender, married, graduated,score, var_1, profession
# ID not required

df.info()

## Graph visualization 

# 1. Gender vs segmentation

sns.countplot(df['Gender'],hue=df['Segmentation'])

# Most of the male are categorized under Category D. Both female and male
# do not have obvious difference in trend.

# 2. Profession vs segmentation

sns.countplot(df['Profession'],hue=df['Segmentation'])

# Most healthcare: D
# Most Artist: C
# Others, not much trend

# 3. Ever married vs segmentation

sns.countplot(df['Ever_Married'],hue=df['Segmentation'])

# can see slope
# Ever married is a strong trend 
# Yes: C, No: D

# 4. Age vs segmentation

sns.countplot(df['Age'],hue=df['Segmentation'])

# 5. Family size vs segmentation

sns.countplot(df['Family_Size'],hue=df['Segmentation'])

# 6. Groupby segmentation, ever_married, gender

df.groupby(['Segmentation','Ever_Married','Gender']).agg({'Segmentation':'count'}).plot(kind='bar')

#%% Step 3. Data cleaning 

# Clean NaN
msno.matrix(df)

# Working experience has a lot of NaN --> can be removed from dataset
# Drop ID, working experience 

df_drop=df.drop(['ID','Work_Experience'],axis = 1)

msno.matrix(df_drop)

# Method 1 

# Data imputation but first need to label encode the data 

# Encode the target with Label Encoder

em=df_drop['Ever_Married']
gen=df_drop['Gender']
grad=df_drop['Graduated']
prof=df_drop['Profession']
fam=df_drop['Family_Size']
var=df_drop['Var_1']
spen=df_drop['Spending_Score']

eda=ExploratoryDataAnalysis()

em=eda.encoder(em)
gen=eda.encoder(gen)
grad=eda.encoder(grad)
prof=eda.encoder(prof)
fam=eda.encoder(fam)
var=eda.encoder(var)
spen=eda.encoder(spen)

df_drop['Ever_Married']=eda.impute(em)
df_drop['Graduated']=eda.impute(grad)
df_drop['Profession']=eda.impute(prof)
df_drop['Family_Size']=eda.impute(fam)
df_drop['Var_1']=eda.impute(var)
df_drop['Spending_Score']=eda.impute(spen)

# Encode the target with One Hot Encoder 

df_drop['Segmentation']=eda.encoder(df['Segmentation'])

#%% Step 4. Data preprocessing
   
X = df_drop.loc[:, 'Gender':'Var_1']
Y = df_drop.loc[:, 'Segmentation']

# Scaled X using MinMax Scaler 

scaler=pickle.load(open(SCALER_PATH,'rb'))
X_scaled = scaler.fit_transform(X)

# Split the dataset to train and test set

X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, 
                                                    test_size=0.3, 
                                                    random_state=42)

#%% Step 6 Deep learning model

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

#%% Step 7 Machine Learning

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
    

pipe_dict = {0:'Logistic Regression', 1:'KNN', 2: 'Random Forest', 
             3: 'SVC',4: 'Decision Tree'}

# Print the accuracy score

for index,model in enumerate(pipelines):
    y_pred = model.predict(X_test)
    print("{} Accuracy Score: {}".format(pipe_dict[index],model.score(X_test, y_test)*100 ))
    #print(classification_report(y_test, y_pred)

#%% Step 8 Model deployment

#model.save(MODEL_SAVE_PATH)  # For deep learning

pickle.dump(svc_pipeline, open(MODEL_SAVE_PATH, 'wb'))