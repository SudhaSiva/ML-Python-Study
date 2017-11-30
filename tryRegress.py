# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:33:04 2017

@author: Sudhakar

Regression learning modules
 
"""
from sklearn import datasets
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Regularization packages
from sklearn.linear_model import Ridge 
from sklearn.linear_model import Lasso

Boston=datasets.load_boston()
# Create a linear regression object: reg
reg = LinearRegression()

# Perform 3-fold CV
cvscores_3 = cross_val_score(reg,X,y,cv=3)
print(np.mean(cvscores_3))

# Perform 10-fold CV
cvscores_10 = cross_val_score(reg,X,y,cv=10)
print(np.mean(cvscores_10))


"""
Model learning packages or how good is the model

"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report # to generate confusion matrix like precsion ,recall and F1 score

from sklearn.metrics import roc_curve  #roc curve package
from sklearn.metrics import roc_auc_score # area under ths curve 


"""
Grid search and randomised search -CV ( cross validation)

"""
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import ElasticNet