# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 13:53:11 2017

@author: Sudhakar
Simple K-Nearest neighbour classification algorithm in python.
"""
from sklearn import datasets
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


# Loading datasets
iris=datasets.load_iris()


#Dividing the original datasetinto training and testing set
x_train=iris.data[0:140]
y_train=iris.target[0:140,]

# testing set
x_test=iris.data[140:,:]
y_test=iris.target[140:,]
df=pd.DataFrame(x_train,columns=iris.feature_names)

#defining the classifier and fitting the data
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train,y_train)


#Predicting the dataset
prediction=knn.predict(x_test)

# Print prediction and original labels
print('prediction {}'.format(prediction))
print('Actual labels{}'.format(y_test))