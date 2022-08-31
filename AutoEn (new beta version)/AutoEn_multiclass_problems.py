from configparser import MAX_INTERPOLATION_DEPTH
import numpy as np
import pandas as pd
from psutil import NoSuchProcess
import scipy.sparse

import statistics
from scipy.optimize import minimize

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler, Normalizer, QuantileTransformer, RobustScaler, StandardScaler, OneHotEncoder
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

from sklearn.feature_selection import SelectFromModel, SelectPercentile, GenericUnivariateSelect, chi2, f_classif, mutual_info_classif
from sklearn.decomposition import FastICA, KernelPCA, PCA, LatentDirichletAllocation
from sklearn.cluster import FeatureAgglomeration
from sklearn.kernel_approximation import RBFSampler, Nystroem

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import RandomTreesEmbedding, BaggingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier, SGDRegressor
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.calibration import CalibratedClassifierCV

import sklearn.datasets
import sklearn.metrics

from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score
from autosklearn.metrics import balanced_accuracy
import autosklearn.classification
from autosklearn.smbo import AutoMLSMBO

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from time import time
import pynisher
import csv
import pickle
from sklearn.metrics import roc_curve, auc, log_loss
import sys
import json

##########################################################################

classifiers_ditc = dict()
classifiers_ditc['adaboost'] = AdaBoostClassifier
classifiers_ditc['bernoulli_nb'] = BernoulliNB
classifiers_ditc['decision_tree'] = DecisionTreeClassifier
classifiers_ditc['extra_trees'] = ExtraTreesClassifier
classifiers_ditc['gaussian_nb'] = GaussianNB
classifiers_ditc['gradient_boosting'] = GradientBoostingClassifier
classifiers_ditc['k_nearest_neighbors'] = KNeighborsClassifier
classifiers_ditc['lda'] = LinearDiscriminantAnalysis
classifiers_ditc['liblinear_svc'] = CalibratedClassifierCV
classifiers_ditc['libsvm_svc'] = SVC
classifiers_ditc['multinomial_nb'] = MultinomialNB
classifiers_ditc['passive_aggressive'] = PassiveAggressiveClassifier
classifiers_ditc['qda'] = QuadraticDiscriminantAnalysis
classifiers_ditc['random_forest'] = RandomForestClassifier
classifiers_ditc['sgd'] = CalibratedClassifierCV
classifiers_ditc['xgradient_boosting'] = XGBClassifier

preprocessors_dict = dict()
preprocessors_dict['extra_trees_preproc_for_classification'] = SelectFromModel
preprocessors_dict['fast_ica'] = FastICA
preprocessors_dict['feature_agglomeration'] = FeatureAgglomeration
preprocessors_dict['kernel_pca'] = KernelPCA
preprocessors_dict['kitchen_sinks'] = RBFSampler
preprocessors_dict['liblinear_svc_preprocessor'] = SelectFromModel
preprocessors_dict['nystroem_sampler'] = Nystroem
preprocessors_dict['pca'] = PCA
preprocessors_dict['polynomial'] = PolynomialFeatures
preprocessors_dict['random_trees_embedding'] = RandomTreesEmbedding
preprocessors_dict['select_percentile_classification'] = SelectPercentile
preprocessors_dict['select_rates'] = GenericUnivariateSelect

rescaling_dict = dict()
rescaling_dict['robust_scaler'] = RobustScaler
rescaling_dict['minmax'] = MinMaxScaler
rescaling_dict['standardize'] = StandardScaler
rescaling_dict['quantile_transformer'] = QuantileTransformer
rescaling_dict['normalize'] = Normalizer

##########################################################################

def dic_for_preproc(i, preprocessing_choice):
    preproc_dict = {}
    for key, value in i.items():
        if 'preprocessor:'+preprocessing_choice+':' in str(key):
            k = key.replace('preprocessor:'+preprocessing_choice+':', '')
            preproc_dict[k] = value
            if value == 'False':
                preproc_dict[k] = False
            if value == 'True':
                preproc_dict[k] = True
            if value == 'None':
                preproc_dict[k] = None
    return preproc_dict

##########################################################################

def validate_pipelines(pipelines_list, X, y):
    xx = X[:20]
    yy = y[:20]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        xx, yy, random_state=1)

    counter = -1
    valid = True
    no_valid = False
    pipes_validated = []

    for pipe in pipelines_list:
        try:
            pipe_ = Pipeline(pipe)
            pipe_.fit(X_train, y_train)
            predictions = pipe_.predict(X_test)

            counter = counter + 1
            pipes_validated.append((counter, valid, pipe_))

        except:
            counter = counter + 1
            # pipes_validated.append((counter, no_valid, pipe_))
            print("something went wrong")
            print("Multiclasss")

    print("pipes_validated: ", len(pipes_validated))
    return pipes_validated

##########################################################################

def caruana_ensemble(list_of_pipelines, X, y, dataset):

    times_all_folds = []
    time_caruana_ensemble_function = []
    t4 = time()

    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=12345)
    fold_counter = 0

    ensemble_results_in_val = []
    ensemble_results_in_test = []

    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train_g, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_g, y_test = y[train_idx], y[test_idx]

        fold_counter = fold_counter + 1
        print("fold in progress: {}".format(fold_counter))

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_g, y_train_g, test_size=0.2, shuffle=True, stratify=y_train_g, random_state=12345)

        classifiers_predictions_in_test = []
        clfs_and_all_pred_val = []

        t1 = time()

        for i in list_of_pipelines:

            counter_, _, pipe = i

            classifier = first_training_classifier(pipe, X_train, y_train)

            if classifier == None:
                pass

                # train and test pipelines in train and validation sets
            else:
                predictions_train = classifier.predict_proba(X_train)
                result_train = log_loss(y_train, predictions_train)

                predictions_validation = classifier.predict_proba(X_val)
                result_val = log_loss(y_val, predictions_validation)

                clfs_and_all_pred_val.append(
                    (counter_, classifier, result_val, predictions_validation))

        final_time_t1 = time() - t1

        # Build Final Caruana Ensemble

        t2 = time()  # time of building the ensemble

        max_perfm_metric = 0
        initial_ensemble_in_val = []
        best_prediction_in_validation = []
        final_ensemble = []

        # extract the best pipeline in the validation set
        best_val = min(clfs_and_all_pred_val, key=lambda x: x[2])

        # extract the best pipeline in validation, its performance in validation, and its raw predictions in validation
        final_pipe_counter, best_classifier_in_validation, best_classifier_perfm_val, predictions_best_classifier_in_y_val = best_val

        # add the pipe Best_Val as first element of the ensemble
        initial_ensemble_in_val.append(
            (final_pipe_counter, best_classifier_in_validation))

        # save the raw validation predictions of Best_Val - then this is used by the ensemble
        best_prediction_in_validation.append(
            predictions_best_classifier_in_y_val)

        # calculate the performance in validation of the ensemble with only one item (this item is Best_Val)
        one_item_ensemble_prediction_y_val = 0
        for prediction in best_prediction_in_validation:
            one_item_ensemble_prediction_y_val += prediction
        max_perfm_metric = log_loss(y_val, one_item_ensemble_prediction_y_val)
        print("Initial performance:", max_perfm_metric)

        # variable to store the final ensemble (at the begininng it only has one item: BestVal)
        final_ensemble = initial_ensemble_in_val

        # variable to store the raw predictions in y_val of the ensemble in construction
        temporal_predictions = best_prediction_in_validation
        # variable to store the predictions in y_val of the final ensemble
        final_predictions_y_val = best_prediction_in_validation

        # build the ensemble with a maximum legnth of 50 classifiers
        while len(final_ensemble) < 50:

            results_temporal_ensembles = []

            for clf in clfs_and_all_pred_val:

                temporal_ensemble = initial_ensemble_in_val.copy()
                temporal_predictions = best_prediction_in_validation.copy()

                tmp_counter, classifier_, _, predictions_in_validation_ = clf

                temporal_ensemble.append((tmp_counter, classifier_))
                temporal_predictions.append(predictions_in_validation_)

                size_ensemble_temporal = len(temporal_ensemble)

                final_prediction = 0
                for prediction in temporal_predictions:
                    final_prediction += prediction/size_ensemble_temporal
                temporal_perfm_metric = log_loss(y_val, final_prediction)

                results_temporal_ensembles.append(
                    (temporal_perfm_metric, temporal_ensemble, temporal_predictions))

            maxVal_in_temporal_ensembles = min(
                results_temporal_ensembles, key=lambda x: x[0])
            candidate_ensemble_perfm, candidate_ensemble, candidate_ensemble_predictions = maxVal_in_temporal_ensembles

            # if candidate_ensemble_perfm > max_perfm_metric:
            final_ensemble = candidate_ensemble.copy()
            final_predictions_y_val = candidate_ensemble_predictions.copy()
            initial_ensemble_in_val = final_ensemble.copy()
            best_prediction_in_validation = candidate_ensemble_predictions.copy()

            max_perfm_metric = candidate_ensemble_perfm

            # else:
            #initial_ensemble_in_val = candidate_ensemble.copy() ###
            #best_prediction_in_validation = candidate_ensemble_predictions.copy()####

        print("ensemble lenght: ", len(final_ensemble))
        print("Final performance:", max_perfm_metric)
        ensemble_results_in_val.append(max_perfm_metric)

        final_time_t2 = time() - t2

        #######  ----- TEST DATA  ----- #######
        t3 = time()  # time of testing the ensemble

        ids_unique_pipes_ = []
        prev_value_ = None
        classifiers_predictions_in_test_ = []

        # ANTONIO: Para metodo FIT()Guardar todos los pipe_2 en una lista que sería una variable de clase llamada final ensemble,

        for i in final_ensemble:

            test_counter, pipeline = i

            # VALIDATION TO Re-TRAIN ONLY ONCE A PIPELINE WITH MULTIPLE REPETITIONS IN THE FINAL ENSEMBLE

            if prev_value_ is None:

                pipe_2 = second_training_classifier(
                    pipeline, X_train_g, y_train_g)

                if pipe_2 == None:
                    pass

                else:
                    prev_value_ = test_counter
                    ids_unique_pipes_.append(prev_value_)

                    predictions_test = pipe_2.predict_proba(X_test)
                    classifiers_predictions_in_test_.append(
                        (prev_value_, predictions_test))
                    classifiers_predictions_in_test.append(predictions_test)

            else:
                prev_value_ = test_counter

                if prev_value_ in ids_unique_pipes_:
                    print("YES, repeated pipeline for RE-TRAINING")
                    print("Multiclass.py")

                    for i in classifiers_predictions_in_test_:
                        tmp_id, tmp_test_preds = i

                        if tmp_id == prev_value_:
                            classifiers_predictions_in_test.append(
                                tmp_test_preds)

                else:
                    pipe_2 = second_training_classifier(
                        pipeline, X_train_g, y_train_g)

                    if pipe_2 == None:
                        pass

                    else:
                        ids_unique_pipes_.append(prev_value_)
                        predictions_test = pipe_2.predict_proba(X_test)
                        classifiers_predictions_in_test_.append(
                            (prev_value_, predictions_test))
                        classifiers_predictions_in_test.append(
                            predictions_test)

        size_ensemble_final = len(final_ensemble)

        final_prediction_in_test = 0

        for pred_t in classifiers_predictions_in_test:
            final_prediction_in_test += pred_t/size_ensemble_final
        result_in_test = log_loss(y_test, final_prediction_in_test)

        ensemble_results_in_test.append(result_in_test)

        final_time_t3 = time() - t3

        times_all_folds.append((final_time_t1, final_time_t2, final_time_t3))

    final_time_t4 = time() - t4
    time_caruana_ensemble_function.append(final_time_t4)
    return ensemble_results_in_test, ensemble_results_in_val, times_all_folds, time_caruana_ensemble_function

##########################################################################
@pynisher.enforce_limits(wall_time_in_s=60)
def zero_training_classifier(pipe_, X_train_red, y_train_red):

    zero_trained_pipe = pipe_.fit(X_train_red, y_train_red)
    return zero_trained_pipe

@pynisher.enforce_limits(wall_time_in_s=360)
def first_training_classifier(pipe, X_train, y_train):

    first_trained_pipe = pipe.fit(X_train, y_train)
    return first_trained_pipe

@pynisher.enforce_limits(wall_time_in_s=360)
def second_training_classifier(pipe, X_train_g, y_train_g):

    second_trained_pipe = pipe.fit(X_train_g, y_train_g)
    return second_trained_pipe
##########################################################################
def pipeline_steps2(choice_string, choice, pipeline_steps3):
    step = (choice_string, choice)
    pipeline_steps3.append(step)


def main(dataset_name):
    dataset = dataset_name
    list_of_times = []

    # try:
    missing_values = ["n/a", "na", "--", "?"]
    df = pd.read_csv('{}.csv'.format(dataset), na_values=missing_values)
    x_cols = [c for c in df.columns if c != 'class']
    X = df[x_cols]
    y = df['class']
    X.fillna(X.mean(), inplace=True)
    XX = pd.get_dummies(X, prefix_sep='_', drop_first=True)

    for repetition in range(1, 2):

        suggestions = []
        counter = 0

        with open('Base_of_pipelines.json') as f:
            suggestions = json.load(f)

        t0 = time()
        t = time()

        list_of_pipelines = []  # _4
        # iterates through every suggestion and implements it
        for i in suggestions:

            pipeline_steps = []

            ######################### ######################### #########################
            _data_resc = i['rescaling:__choice__']
            rescaling_obj = None
            if _data_resc == "None":
                pass

            elif _data_resc == "robust_scaler":
                q_max = i['rescaling:robust_scaler:q_max']
                q_min = i['rescaling:robust_scaler:q_min']

                rescaling_obj = rescaling_dict[_data_resc](
                    with_centering=False, quantile_range=(q_min, q_max))

            elif _data_resc == "standardize":
                rescaling_obj = rescaling_dict[_data_resc](with_mean=False)

            elif _data_resc == "quantile_transformer":
                n_quantiles = i['rescaling:quantile_transformer:n_quantiles']
                output_distribution = i['rescaling:quantile_transformer:output_distribution']
                rescaling_obj = rescaling_dict[_data_resc](
                    n_quantiles, output_distribution)

            if _data_resc == "None":

                if rescaling_obj == None:
                    pipeline_steps2(
                        _data_resc, rescaling_dict[_data_resc](), pipeline_steps)
                else:
                    pipeline_steps2(_data_resc, rescaling_obj, pipeline_steps)
            ######################### ######################### #########################
            preprocessing_choice = i['preprocessor:__choice__']
            preprocessing_obj = None
            dic_preroc = dic_for_preproc(i, preprocessing_choice)

            if preprocessing_choice == "no_preprocessing":
                pass

            elif preprocessing_choice == 'extra_trees_preproc_for_classification':
                preprocessing_obj = preprocessors_dict[preprocessing_choice](
                    estimator=ExtraTreesClassifier(**dic_preroc))
        
            elif preprocessing_choice == 'liblinear_svc_preprocessor':
                preprocessing_obj = SelectFromModel(
                    estimator=LinearSVC(**dic_preroc))

            elif preprocessing_choice == 'pca':
                del dic_preroc['keep_variance']
                
            elif preprocessing_choice == 'select_percentile_classification':
                if dic_preroc['score_func'] == "chi2":
                    dic_preroc['score_func'] = sklearn.feature_selection.chi2

                elif dic_preroc['score_func'] == "f_classif":
                    dic_preroc['score_func'] = sklearn.feature_selection.f_classif

                elif dic_preroc['score_func'] == "mutual_info":
                    dic_preroc['score_func'] = sklearn.feature_selection.mutual_info_classif
                
            elif preprocessing_choice == 'select_rates':
                del dic_preroc['alpha']
                if dic_preroc['score_func'] == "chi2":
                    dic_preroc['score_func'] = sklearn.feature_selection.chi2

                elif dic_preroc['score_func'] == "f_classif":
                    dic_preroc['score_func'] = sklearn.feature_selection.f_classif

            if _data_resc == None:

                if preprocessing_obj == None:
                    pipeline_steps2(_data_resc, rescaling_dict[_data_resc](
                        **dic_preroc), pipeline_steps)
                else:
                    pipeline_steps2(_data_resc, rescaling_obj, pipeline_steps)
            # training and testing classifiers
            counter = counter + 1
            new_dict = dict()
            for clf in classifiers_ditc:
                if i['classifier:__choice__'] == clf:
                    _clf = i['classifier:__choice__']
                    for key, value in i.items():
                        if ':'+_clf+':' in str(key):
                            k = key.replace('classifier:'+_clf+':', '')
                            new_dict[k] = value
                            if value == 'False':
                                new_dict[k] = False
                            if value == 'True':
                                new_dict[k] = True
                            if value == 'None':
                                new_dict[k] = None
                    classifiers_obj = None
                    if _clf == "None":
                        pass
                    elif _clf == "adaboost":
                        del new_dict['max_depth']
                        classifiers_obj = classifiers_ditc[_clf](**new_dict)
                    elif _clf == "decision_tree":
                        del new_dict['max_depth_factor']
                        classifiers_obj = classifiers_ditc[_clf](**new_dict)
                    elif _clf == "gradient_boosting":
                        del new_dict['scoring']
                        del new_dict['max_iter']
                        del new_dict['max_bins']
                        del new_dict['l2_regularization']
                        del new_dict['early_stop']
                        classifiers_obj = classifiers_ditc[_clf](**new_dict)
                    elif _clf == "liblinear_svc":
                        classifiers_obj = classifiers_ditc[_clf](base_estimator=LinearSVC(**new_dict))
                    elif _clf == "libsvm_svc":
                        classifiers_obj = classifiers_ditc[_clf](**new_dict, probability=True)
                    elif _clf == "sgd":
                        classifiers_obj = classifiers_ditc[_clf](base_estimator=SGDClassifier(**new_dict))
                    elif _clf == "lda":
                        classifiers_obj = classifiers_ditc[_clf]()
                    elif _clf != "None":

                        if classifiers_obj == None:
                            pipeline_steps2(_clf, classifiers_ditc[_clf](**new_dict), pipeline_steps)
                        else:
                            pipeline_steps2(_clf, classifiers_obj, pipeline_steps)

            list_of_pipelines.append(pipeline_steps)
        # HACIA ARRIBA CODIGO CONSTRUCTOR AUTOEN. List of pipelines sería variable de clase.

        missing_values = ["n/a", "na", "--", "?"]
        df = pd.read_csv('{}.csv'.format(dataset), na_values=missing_values)
        x_cols = [c for c in df.columns if c != 'class']
        x = df[x_cols]
        y = df['class']
        x.fillna(x.mean(), inplace=True)
        XX = pd.get_dummies(x, prefix_sep='_', drop_first=True)

        list_ps = validate_pipelines(list_of_pipelines, XX, y)

        time_building_validate_pipelines = time() - t0

        ensemble_results_in_test, ensemble_results_in_val, times_all_folds, time_caruana_ensemble_function = caruana_ensemble(
            list_ps, XX, y, dataset)

        final_time = time() - t

        # Hasta aquí sería método fit.

        print("#########################")
        print("ensemble_results_in_test{}".format(dataset))
        print(ensemble_results_in_test)
        print(statistics.mean(ensemble_results_in_test))
        print("#########################")
        print("ensemble_results_in_validation{}".format(dataset))
        print(ensemble_results_in_val)
        print(statistics.mean(ensemble_results_in_val))
        print("#########################")
        print("times_all_folds:")
        print(times_all_folds)
        t1 = []
        t2 = []
        t3 = []
        for i in times_all_folds:
            t_1, t_2, t_3 = i
            t1.append(t_1)
            t2.append(t_2)
            t3.append(t_3)

        print("#########################")
        print("total_time_caruana_ensemble_function: ",
              time_caruana_ensemble_function)
        print("avg x fold t1: ", statistics.mean(t1))
        print("avg x fold t2: ", statistics.mean(t2))
        print("avg x fold t3: ", statistics.mean(t3))

        list_of_times.append((final_time, "dataset{}".format(dataset)))

        # except:
    # print("something wrong loading dataset: {}".format(dataset))
    print("time_building_validate_pipelines: ",
          time_building_validate_pipelines)
    print("Total time: ", list_of_times)

##########################################################################
main(sys.argv[1])
