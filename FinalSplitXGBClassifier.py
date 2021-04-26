import pandas as pd, numpy as np, matplotlib.pyplot as plt, os, sklearn
from operator import attrgetter
from collections import Counter

from sklearn import metrics
from sklearn.metrics import fbeta_score, make_scorer, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn import linear_model as lm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm.sklearn import LGBMClassifier
from xgboost import XGBClassifier

thisModel = "xgb"
#wd = r"C:\Code\Data"
wd = r"/users/edwardf/notebooks"

# The models and their parameter permutations to be examined
models = [
    { 
        "model" : [XGBClassifier()],
        "name" : "Extreme Gradient Boosting (XGB)",
        "params" : 
        {
            "max_depth"              : range (2, 10, 1),
            "n_estimators"           : range(60, 220, 40),
            "learning_rate"          : [0.1, 0.01, 0.05]
        }
    }]

# Load data
import pickle

cleanDataFilename = r"clean_data.pkl"
os.chdir(wd)

cleanDataFile = open(cleanDataFilename, 'rb')
cleanData = pickle.load(cleanDataFile)
cleanDataFile.close()

X, y, X_train, X_test, y_train, y_test = cleanData
print(X[:3])
print(y.head())

# Grid search for optimal model and parameters


# Minimize false positive = maximize precision
# Minimize false negative = maximize recall
# F-beta is weighted harmonic mean of preicision and recall, so weight ratio recall:precision is 5:1
custom_scorer = make_scorer(fbeta_score, beta=5)

results=[]

for model in models:

    # Classifier
    clf = model['model'][0]
    print("Grid search on", model['name'])

    # Grid search many potential models and parameters to find optimal fit and save the boss some $$$
    grid = GridSearchCV(clf, param_grid=model['params'], scoring=custom_scorer, cv=5, n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train.values.ravel())
    
    # Get metrics of interest for report
    y_pred = grid.predict(X_test)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    CM = confusion_matrix(y_test, y_pred)

    # Results for later comparison
    results.append(
        {
            'name'       : model['name'],
            'grid'       : grid,
            'classifier' : grid.best_estimator_,
            'best score' : grid.best_score_,
            'best params': grid.best_params_,
            'precision'  : pre,
            'recall'     : rec,
            'TN'         : CM[0][0],
            'FN'         : CM[1][0],
            'TP'         : CM[1][1],
            'FP'         : CM[0][1],
            'cv'         : grid.cv
        })

# Save data
modelFilename = thisModel + ".pkl"
os.chdir(wd)

modelData = results

modelDataFile = open(modelFilename, 'wb')
pickle.dump(modelData, modelDataFile)
modelDataFile.close()