import numpy as np
import pandas as pd
import scipy.sparse
import statistics
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler, Normalizer, QuantileTransformer, RobustScaler, StandardScaler, OneHotEncoder
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

from sklearn.feature_selection import SelectFromModel, SelectPercentile, GenericUnivariateSelect, chi2, f_classif, mutual_info_classif
from sklearn.decomposition import FastICA, KernelPCA, PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.kernel_approximation import RBFSampler, Nystroem

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import RandomTreesEmbedding, BaggingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

import sklearn.datasets
import sklearn.metrics

from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score
from autosklearn.metrics import balanced_accuracy
import autosklearn.classification
from autosklearn.smbo import AutoMLSMBO

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score,roc_auc_score

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from time import time 
import pynisher
import csv
import pickle
from sklearn.metrics import roc_curve, auc, log_loss
import sys

def classifier_performance_cv(X,y):
    
    rf = RandomForestClassifier(n_estimators=2000)
    results_in_test = []
    
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    
    fold_counter = 0

    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)): 
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        fold_counter = fold_counter + 1
    
        rf.fit(X_train, y_train)
        predictions_test = rf.predict_proba(X_test)
        result_test = log_loss(y_test,predictions_test)
        results_in_test.append(result_test)
        
    return results_in_test   
	
def main(dataset_name):

    dataset = dataset_name
    missing_values = ["n/a", "na", "--", "?"]
    df = pd.read_csv('{}.csv'.format(dataset), na_values = missing_values)
    x_cols = [c for c in df.columns if c != 'target']
    x = df[x_cols]
    y = df['target']
    x.fillna(x.mean(), inplace=True)
    XX = pd.get_dummies(x, prefix_sep='_', drop_first=True)
    
    
    results_rf = classifier_performance_cv(XX,y)
    
    print("#########################")
    print("Random Forest results:")
    print(results_rf)
    print(statistics.mean(results_rf))   

main(sys.argv[1])