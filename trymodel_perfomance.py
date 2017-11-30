# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:43:44 2017

@author: Sudhakar
Model performance -Split automatically into test and train using train_test_split package
"""

from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


# Loading datasets
iris=datasets.load_iris()
X=iris.data
y=iris.target

# Randomly selecting the dataset from 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size =0.3,random_state =21, stratify=y)


df=pd.DataFrame(X_train,columns=iris.feature_names)

#defining the classifier and fitting the data
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)


#Predicting the dataset
prediction=knn.predict(X_test)

# Print prediction and original labels
print('prediction {}'.format(prediction))
print('Actual labels{}'.format(y_test))


# Computing the accuracy of the model
print(knn.score(X_test,y_test))