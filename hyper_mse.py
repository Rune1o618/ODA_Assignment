#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin


def hyperparameter_MSE(model, X, y):
    clf = GridSearchCV(model, {'epsilon' : [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]}, cv = 5, return_train_score = False)
    best_model = clf.fit(X, y)
    
    return best_model.best_estimator_.get_params()['epsilon']

