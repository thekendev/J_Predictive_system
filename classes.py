import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from matplotlib import rcParams
from matplotlib.cm import rainbow

from sklearn.linear_model import LogisticRegression


#Reading the csv file
df = pd.read_csv('heart_disease_data.csv')



#class creation
#datasets
class DataFrame(object):
        # Variables for the class
        def __init__(self, des, null, hd, vc):
            self.__describe = des
            self.__isnull_sum = null
            self.__head = hd
            self.__value_counts = vc



obj_des = df.describe()
obj_null = df.isnull().sum()
obj_hd = df.head()
obj_vc = df['target'].value_counts()
#dataset prints
print(obj_des)
print(obj_null)
print(obj_hd)

     # T_T =  Test and Train
class t_t(object):
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y

obj_X =  df.drop(columns='target', axis=1)
obj_Y = df['target']
#t & t print
print(obj_Y)
print(obj_X)




#prediction
class prediction(object):
    def __init__(self, pre):
        self.__model_predict = pre

obj_pre = ("model.predict(data_standard)")
print(obj_pre)









