# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:38:50 2022

@author: ACER
"""

# constant


class ExploratoryDataAnalysis():
    
    def __init__(self):
        pass
       
    def label_encoder(self,input_data):
        '''
         This function will encode the input using label encoder approach

         Parameters
         ----------
         input_data :List, array
         input_data will undergo label encoding.

         Returns
         -------
         Label-encoded input_data in array data. Eg: 1,2 ---> [0],[1,0]


        '''
        le=LabelEncoder()
    
        return le.fit_transform(input_data)  

class ModelEvaluation():
    
    def report_metrics(self,y_true,y_pred):
        
        print(classification_report(y_true,y_pred))
        print(confusion_matrix(y_true,y_pred))
        print(accuracy_score(y_true,y_pred))
    
#%%

if __name__ == '__main__':
    import os
    import pandas as pd








