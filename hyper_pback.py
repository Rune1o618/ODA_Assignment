#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin


def hyperparameter_pback(model, X, y):
    
    eta0 = [0.001, 0.01, 0.1, 1]
    hyperparameters = dict(eta0=eta0)
    
    rdn = RandomizedSearchCV(model, hyperparameters, cv = 5, return_train_score = False)
    best_model = rdn.fit(X, y)
    
    
    return best_model.best_estimator_.get_params()['eta0']
    


# In[ ]:




