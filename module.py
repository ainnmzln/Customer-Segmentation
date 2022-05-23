# -*- coding: utf-8 -*-
"""

@author: ainnmzln
"""

from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,OneHotEncoder
import pandas as pd
import numpy as np

class ExploratoryDataAnalysis():
    
    def __init__(self):
        pass
    
    def encoder(self,data):
        
        le=LabelEncoder()
        data[data.notnull()]=le.fit_transform(data[data.notnull()])
        data=pd.to_numeric(data,errors='coerce')
        
        return data 
    
    def inversencode(self,data):
        
        ie=LabelEncoder()
        data=ie.inverse_transform(np.expand_dims(data,axis=-1))
        #data=pd.to_numeric(data,errors='coerce')
        
        return data 
    
    def impute(self,data):
        
        ii=IterativeImputer()
        data_imputed=ii.fit_transform(np.expand_dims(data,axis=-1))

        return data_imputed
    
if __name__=='__main__':
    pass