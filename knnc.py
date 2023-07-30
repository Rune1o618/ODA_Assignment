#!/usr/bin/env python
# coding: utf-8

# ## Nearest Neighbors Classifier Function

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

def KNearestNeighborsClassifer(X_train, X_test, y_train, y_test):
    
    n_samples, n_features = X_train.shape
    n_neighbors = [1, 2, 3, 4, 5]
    if n_features == 2 and n_samples > 10000:
        n_neighbors = [1, 2, 3, 4, 5, 100, 300, 310, 320, 340, 350, 360, 370, 380, 400]
    
    hyperparameters = dict(n_neighbors=n_neighbors)
    
    model = KNeighborsClassifier(p=2)
    clf = GridSearchCV(model, hyperparameters)
    best_model = clf.fit(X_train,y_train.ravel())
    
    print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
    
    final_model = KNeighborsClassifier(n_neighbors=best_model.best_estimator_.get_params()['n_neighbors'], p=2)
    final_model.fit(X_train, y_train.ravel())
    y_pred = final_model.predict(X_test)
    
    print(f"Accuracy : {metrics.accuracy_score(y_test, y_pred)* 100} %")
    return metrics.accuracy_score(y_test, y_pred)

