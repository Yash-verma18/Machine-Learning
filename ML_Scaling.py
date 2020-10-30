
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
  
# Sklearn library  
from sklearn import preprocessing 
  
""" PART 2 
    Importing Data """
   
data_set = pd.read_csv('C:\\Users\\dell\\Desktop\\Data_for_Feature_Scaling.csv') 
data_set.head() 
  
# here Features - Age and Salary columns  
# are taken using slicing 
# to handle values with varying magnitude 
x = data_set.iloc[:, 1:3].values 
print ("\nOriginal data values : \n",  x) 
  
  
""" PART 4 
    Handling the missing values """
  
from sklearn import preprocessing 
  
""" MIN MAX SCALER """
  
min_max_scaler = preprocessing.MinMaxScaler(feature_range =(0, 1)) 
  
# Scaled feature 
x_after_min_max_scaler = min_max_scaler.fit_transform(x) 
  
print ("\nAfter min max Scaling : \n", x_after_min_max_scaler) 
  
  
""" Standardisation """
  
Standardisation = preprocessing.StandardScaler() 
  
# Scaled feature 
x_after_Standardisation = Standardisation.fit_transform(x) 
  
print ("\nAfter Standardisation : \n", x_after_Standardisation)


Code Courtesy : GfG
