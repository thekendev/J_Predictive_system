#IMPORTING THE DEPENDENCIES
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
import pytest
import matplotlib.image as pltimg
import seaborn as sns
from sklearn.linear_model import LogisticRegression




                                            #DECISION TREE ALGORITHM
#Reading the csv file
df = pd.read_csv('heart_disease_data.csv')

#shaping the data
df
df.shape

#checking the first 5 rows of the dataset
df.head()





#describing the dataset
df.describe()

#Showing if there are any null values in the dataset
df.isnull().sum()

#value count for the target column
df['target'].value_counts()

#Seperating the data and labels
X = df.drop(columns='target', axis=1)
Y = df['target']


#Decision tree
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)
data = tree.export_graphviz(dtree, out_file=None, feature_names=features) graph = pydotplus.graph_from_dot_data(data) graph.write_png('mydecisiontree.png')
img=pltimg.imread('mydecisiontree.png') imgplot = plt.imshow(img)
plt.show()


#Data standardisation

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


#Training,Testing,Splitting the dataset

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y,random_state = 2)


model = tree.DecisionTreeClassifier(random_state=0, max_depth=4)
model = model.fit(X_train, Y_train)

#Finding the accuracy score on training dataset

X_train_prediction = model.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction, Y_train)

train_data_accuracy


#Finding the accuracy score on test dataset

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

test_data_accuracy




#Predicting system
#testdata




data = (62, 0, 0, 138, 294, 1, 1, 106, 0, 1.9, 1, 3, 2)
data_array = np.asarray(data)
#reshaping the array
data_reshape = data_array.reshape(1, -1)
data_standard = scaler.transform(data_reshape)
prediction = model.predict(data_standard)


def predict_disease(data):
    # test data

    # class prediction

    if (prediction[0] == 1):
        print('This patient has a Heart Disease')
    else:
        print('This patient does not have a Heart Disease')


predict_disease(data)




















                                       #LOGISTIC REGRESSION ALGORTHM

#Loading the csv data to pandas dataframe
df = pd.read_csv('heart_disease_data.csv')

#checking for more information about the dataset
df.info()

#Describing the dataset
df.describe()

#checking for correlation
corrmat=df.corr()
top_corr_features=corrmat.index

plt.figure(figsize=(20,20))

#seaborn correlation heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True)

#print out the first 5 rows
df.head()

#Returning Unique values
df.target.value_counts()

#removing the axis 1 from Target column
X=df.drop('target',axis=1)
y=df['target']

#training,testing and splitting the data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=33,stratify=y)


#inputting the LogisticRegressionModel
log_model=LogisticRegression()
log_model.fit(X_train,y_train)

#Training accuracy
x_train_pred=log_model.predict(X_train)

training_accuracy=accuracy_score(x_train_pred,y_train)

print('Training data accuracy:',training_accuracy)


#Training accuracy
x_test_pred=log_model.predict(X_test)

testing_accuracy=accuracy_score(x_test_pred,y_test)

print('Testing data accuracy:',testing_accuracy)




#Building a predictive system

#Test data
input_data=(53,1,0,140,203,1,0,155,1,3.1,0,0,3)

#reshaping the array
np_array=np.asarray(input_data).reshape(1,-1)

prediction=log_model.predict(np_array)

if prediction[0]==0:
    print('The patient does not have a heart disease')
else:
    print('The patient has a heard disease')







                  #SUMMARY ON BEST ALGORITHM FOR THIS USE CASE
    #From the above, two algorithms were trained and tested; from the results:

    #Decision Tree Algorithm has a train data accuracy of 88% and a test data accuracy of 75%
    #Logistic Regression Algorithm has a train data accuracy of 86% and a test data accuracy of 85%

    #Therefore, Logistic Regression Algorithm is best fit for this prediction analysis