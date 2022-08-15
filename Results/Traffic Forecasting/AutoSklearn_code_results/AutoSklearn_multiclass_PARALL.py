import numpy as np
import pandas as pd
import scipy.sparse

import statistics
from scipy.optimize import minimize

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
from autosklearn.metrics import balanced_accuracy, log_loss

import autosklearn.classification
from autosklearn.smbo import AutoMLSMBO

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score,roc_auc_score

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from time import time 
import pynisher
import csv
import pickle
from sklearn.metrics import roc_curve, auc, log_loss, accuracy_score
import sys


##########################################################################

def main(time_run,dataset_name):

    ET = int(time_run)
    dataset = dataset_name
        
    df = pd.read_csv('{}.csv'.format(dataset))
    x_cols = [c for c in df.columns if c != 'target']
    X = df[x_cols]
    y = df['target']
    list_ = list(df.target.unique())

    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, test_size = 0.2, random_state=12345)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=ET,
        per_run_time_limit=360,
        ensemble_size = 50,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 10},
    )  

    automl.fit(X_train.copy(), y_train.copy(), metric=autosklearn.metrics.log_loss)
    automl.fit_ensemble(y_train.copy(), ensemble_size = 50, metric=autosklearn.metrics.log_loss)
    automl.refit(X_train.copy(), y_train.copy())

    predictions = automl.predict_proba(X_test)
    print("Log_loss score", log_loss(y_test, predictions))

if __name__ == "__main__":
    main(*sys.argv[1:])

##########################################################################


