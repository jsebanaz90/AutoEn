import numpy as np
import pandas as pd
import scipy.sparse

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

##########################################################################

classifiers = ['adaboost', 'bernoulli_nb', 'decision_tree', 'extra_trees', 'gaussian_nb', 'gradient_boosting', 
               'k_nearest_neighbors', 'lda', 'liblinear_svc', 'libsvm_svc', 'multinomial_nb', 'passive_aggressive', 
               'qda', 'random_forest', 'sgd', 'xgradient_boosting']

preprocessors = ['extra_trees_preproc_for_classification', 'fast_ica', 'feature_agglomeration', 'kernel_pca', 
                 'kitchen_sinks', 'liblinear_svc_preprocessor', 'no_preprocessing', 'nystroem_sampler', 'pca',
                 'polynomial', 'random_trees_embedding', 'select_percentile_classification', 'select_rates']
                 
##########################################################################

def dic_for_preproc(i,preprocessing_choice):
    preproc_dict = {}
    for key, value in i.items():
        if 'preprocessor:'+preprocessing_choice+':' in str(key):
            k = key.replace('preprocessor:'+preprocessing_choice+':','')
            preproc_dict[k] = value
            if value == 'False':
                preproc_dict[k] =  False
            if value == 'True':
                preproc_dict[k] = True
            if value == 'None':
                preproc_dict[k] = None
    return preproc_dict
    
##########################################################################


def validate_pipelines(pipelines_list,X,y):
    xx = X[:20]
    yy = y[:20]
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(xx, yy, random_state=1)
    
    counter = -1
    valid =  True
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
            pipes_validated.append((counter, no_valid, pipe_))
            print("something went wrong")
    
    return pipes_validated
    
##########################################################################

def ranking_performance_train_test_pipelines(list_of_pipelines,X,y,dataset,repetition):
     
    results_of_pipelines = []
    best_results_in_test = []
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    
    fold_counter = 0

    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)): 
        X_train_g, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_g, y_test = y[train_idx], y[test_idx]
        
        fold_counter = fold_counter + 1
    
        X_train, X_val, y_train, y_val = train_test_split(X_train_g, y_train_g, test_size = 0.2, shuffle=True, stratify = y_train_g, random_state=1)

        counter = 0 
        
        all_pipes_results_in_test = []
        
        for p in list_of_pipelines:            
            
            counter, val_or_novalid, pipe = p
            
            if val_or_novalid == True:
               
                pipe_1 = first_training_classifier(pipe, X_train, y_train)
                
                if pipe_1 == None:
                    results_of_pipelines.append((counter,0,0,0))
                    
                ####### train valid pipelines in train and validation sets, then test set
                else:                      
                    predictions_train = pipe_1.predict_proba(X_train)
                    result_train = log_loss(y_train,predictions_train)
                    
                    predictions_validation = pipe_1.predict_proba(X_val)
                    result_validation = log_loss(y_val,predictions_validation)
                  
                    results_of_pipelines.append((counter,result_train,result_validation,pipe))
                    
                    #######################################
                    pipes_in_test = second_training_classifier(pipe, X_train_g, y_train_g)
                    
                    if pipes_in_test == None:
                        print("this pipe failed in test set")
                        all_pipes_results_in_test.append((counter,0,0,0))      
            
                    else:
                        predictions_in_test_ = pipes_in_test.predict_proba(X_test)
                        results_in_test = log_loss(y_test,predictions_in_test_)
                        all_pipes_results_in_test.append((counter,result_train, result_validation, results_in_test))                    
                    
                     #######################################     
            else:  
                results_of_pipelines.append((counter,0,0,0))

           
        ####### choose best pipeline in the validation set          
        best_val = min(results_of_pipelines, key = lambda x:(x[2] ==0, x[2]))
        best_in_validation = best_val
        
        ####### test the best pipeline from validation in the test set    
        counter_for_test, train_best_in_val, val_best, p =  best_in_validation           
        pipe_2 = second_training_classifier(p, X_train_g, y_train_g) 
                
        if pipe_2 == None:
            print("the best pipeline in validation fail in the test set")
            results_of_pipelines.append((counter, 0,0,0,0))
                        
        else:
            predictions_test = pipe_2.predict_proba(X_test)
            result_test = log_loss(y_test,predictions_test)
            
            best_results_in_test.append((counter_for_test,train_best_in_val, val_best,result_test))
              
                            
    return results_of_pipelines, best_results_in_test
    
##########################################################################


@pynisher.enforce_limits(wall_time_in_s=360)
def first_training_classifier(pipe, X_train, y_train):
    
    first_trained_pipe = pipe.fit(X_train, y_train)    
    return first_trained_pipe
    
@pynisher.enforce_limits(wall_time_in_s=360)
def second_training_classifier(pipe, X_train_g, y_train_g):
    
    second_trained_pipe = pipe.fit(X_train_g, y_train_g)    
    return second_trained_pipe
    
##########################################################################

def main(dataset_name):
		dataset = dataset_name
		list_of_times = []
		final_results = []
    

		missing_values = ["n/a", "na", "--", "?"]
		df = pd.read_csv('{}.csv'.format(dataset), na_values = missing_values)
		x_cols = [c for c in df.columns if c != 'target']
		X = df[x_cols]
		y = df['target']
		X.fillna(X.mean(), inplace=True)
		XX = pd.get_dummies(X, prefix_sep='_', drop_first=True)


		for repetition in range(1,2):

			X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(XX, y, random_state=1)

			automl = autosklearn.classification.AutoSklearnClassifier(
			time_left_for_this_task=120,
			per_run_time_limit=60,
			ml_memory_limit = 500000,
			delete_tmp_folder_after_terminate=True,
			initial_configurations_via_metalearning=121)

			automl.fit(X_train, y_train)
			predictions_automl = automl.predict(X_test)

			MetalearningSuggestions = automl.MetalearningSuggestions_extractor()

			suggestions = []

			for i in MetalearningSuggestions:
				pipeline = i.get_dictionary()
				suggestions.append(pipeline)

			t0 = time()
			counter = 0

			list_of_pipelines = [] ## _4

			### iterates through every suggestion and implements it
			for i in suggestions:   

				pipeline_steps = []


				######################### ######################### #########################
				_data_resc = i['rescaling:__choice__']

				if _data_resc == "none":
					pass

				elif _data_resc == "robust_scaler":
					q_max = i['rescaling:robust_scaler:q_max']
					q_min = i['rescaling:robust_scaler:q_min'] 


					#######
					robust_scaler = RobustScaler(with_centering=False,quantile_range=(q_min, q_max))
					scaling_step = (_data_resc, robust_scaler)

					pipeline_steps.append(scaling_step)
					#######

				elif _data_resc == "minmax":


					#######
					minmax_scaler = MinMaxScaler()
					scaling_step = (_data_resc, minmax_scaler)

					pipeline_steps.append(scaling_step)
					#######

				elif _data_resc == "standardize":


					#######
					standard_scaler = StandardScaler(with_mean=False)
					scaling_step = (_data_resc, standard_scaler)

					pipeline_steps.append(scaling_step)
					#######

				elif _data_resc == "quantile_transformer":
					n_quantiles = i['rescaling:quantile_transformer:n_quantiles']
					output_distribution = i['rescaling:quantile_transformer:output_distribution']

					#######
					quantile_transformer = QuantileTransformer(n_quantiles,output_distribution)
					scaling_step = (_data_resc, quantile_transformer)

					pipeline_steps.append(scaling_step)
					#######

				elif _data_resc == "normalize":


					#######
					normalizer_ = Normalizer()
					scaling_step = (_data_resc, normalizer_)

					pipeline_steps.append(scaling_step)
					#######


				######################### ######################### #########################
				preprocessing_choice = i['preprocessor:__choice__']


				if preprocessing_choice == "no_preprocessing":
					 pass

				elif preprocessing_choice == 'extra_trees_preproc_for_classification':


					#######
					extra_t_dict = dic_for_preproc(i,preprocessing_choice)
					extra_trees = SelectFromModel(estimator=ExtraTreesClassifier(**extra_t_dict))     
					feature_prep_step = (preprocessing_choice, extra_trees)

					pipeline_steps.append(feature_prep_step)
					####### 

				elif preprocessing_choice == 'fast_ica':


					#######
					fastICA_d = dic_for_preproc(i,preprocessing_choice)
					fastica = FastICA(**fastICA_d,max_iter=2000)      
					feature_prep_step = (preprocessing_choice, fastica)

					pipeline_steps.append(feature_prep_step)
					#######

				elif preprocessing_choice == 'feature_agglomeration':


					#######
					feature_agglo_d =  dic_for_preproc(i,preprocessing_choice)
					feature_aggl = FeatureAgglomeration(**feature_agglo_d) 
					feature_prep_step = (preprocessing_choice, feature_aggl)

					pipeline_steps.append(feature_prep_step)
					#######

				elif preprocessing_choice == 'kernel_pca':


					#######
					kernel_pca_d = dic_for_preproc(i,preprocessing_choice)
					kernelpca = KernelPCA(**kernel_pca_d)
					feature_prep_step = (preprocessing_choice, kernelpca)

					pipeline_steps.append(feature_prep_step)
					#######

				elif preprocessing_choice == 'kitchen_sinks':


					#######
					kitchen_sinks_d = dic_for_preproc(i,preprocessing_choice)
					kitchensinks = RBFSampler(**kitchen_sinks_d)
					feature_prep_step = (preprocessing_choice, kitchensinks)

					pipeline_steps.append(feature_prep_step)
					#######

				elif preprocessing_choice == 'liblinear_svc_preprocessor':


					#######
					liblinear_d = dic_for_preproc(i,preprocessing_choice)
					liblinearsvc_preprocessor = SelectFromModel(estimator=LinearSVC(**liblinear_d))
					feature_prep_step = (preprocessing_choice, liblinearsvc_preprocessor)

					pipeline_steps.append(feature_prep_step)
					#######

				elif preprocessing_choice == 'nystroem_sampler':


					#######
					nystroem_sampler_d = dic_for_preproc(i,preprocessing_choice)
					nystroemsampler = Nystroem(**nystroem_sampler_d)
					feature_prep_step = (preprocessing_choice, nystroemsampler)

					pipeline_steps.append(feature_prep_step)
					#######

				elif preprocessing_choice == 'pca':


					#######
					pca_d = dic_for_preproc(i,preprocessing_choice)
					del pca_d['keep_variance']
					pca_ = PCA(**pca_d) 
					feature_prep_step = (preprocessing_choice, pca_)

					pipeline_steps.append(feature_prep_step)
					#######

				elif preprocessing_choice == 'polynomial':


					#######
					polynomial_d = dic_for_preproc(i,preprocessing_choice)
					polynomial_ = PolynomialFeatures(**polynomial_d) 
					feature_prep_step = (preprocessing_choice, polynomial_)

					pipeline_steps.append(feature_prep_step)
					#######

				elif preprocessing_choice == 'random_trees_embedding':


					#######
					random_trees_d = dic_for_preproc(i,preprocessing_choice)
					del random_trees_d['bootstrap']
					del random_trees_d['min_weight_fraction_leaf']      
					randomtreesembedding = RandomTreesEmbedding(**random_trees_d)
					feature_prep_step = (preprocessing_choice, randomtreesembedding)

					pipeline_steps.append(feature_prep_step)
					#######

				elif preprocessing_choice == 'select_percentile_classification':


					######
					select_percentile_d = dic_for_preproc(i,preprocessing_choice)

					if select_percentile_d['score_func'] == "chi2":
						select_percentile_d['score_func'] = sklearn.feature_selection.chi2

					elif select_percentile_d['score_func'] == "f_classif":
						select_percentile_d['score_func'] = sklearn.feature_selection.f_classif

					elif select_percentile_d['score_func'] == "mutual_info":
						select_percentile_d['score_func'] = sklearn.feature_selection.mutual_info_classif

					select_percentile = SelectPercentile(**select_percentile_d)
					feature_prep_step = (preprocessing_choice, select_percentile)

					pipeline_steps.append(feature_prep_step)
					#######

				elif preprocessing_choice == 'select_rates':

					######
					select_rates_d= dic_for_preproc(i,preprocessing_choice)

					del select_rates_d['alpha']
					if select_rates_d['score_func'] == "chi2":
						select_rates_d['score_func'] = sklearn.feature_selection.chi2

					elif select_rates_d['score_func'] == "f_classif":
						select_rates_d['score_func'] = sklearn.feature_selection.f_classif

					selectrates = GenericUnivariateSelect(**select_rates_d)
					feature_prep_step = (preprocessing_choice, selectrates)

					pipeline_steps.append(feature_prep_step)
					#######


				#### training and testing classifiers
				counter = counter + 1
				new_dict = dict()    
				for clf in classifiers:
					if i['classifier:__choice__'] == clf:
						_clf = i['classifier:__choice__']
						for key, value in i.items():
							if ':'+_clf+':' in str(key):
								k = key.replace('classifier:'+_clf+':','')
								new_dict[k] = value
								if value == 'False':
									new_dict[k] =  False
								if value == 'True':
									new_dict[k] = True
								if value == 'None':
									new_dict[k] = None

						if _clf == "random_forest":
							classifier_ = RandomForestClassifier(**new_dict)
							classifier_step = (_clf, classifier_)

							pipeline_steps.append(classifier_step)


						elif _clf == "extra_trees":
							classifier_ = ExtraTreesClassifier(**new_dict)
							classifier_step = (_clf, classifier_)

							pipeline_steps.append(classifier_step)


						elif _clf == "adaboost":
							del new_dict['max_depth']
							classifier_ = AdaBoostClassifier(**new_dict)
							classifier_step = (_clf, classifier_)

							pipeline_steps.append(classifier_step)

						elif _clf == "bernoulli_nb":
							classifier_ = BernoulliNB(**new_dict)
							classifier_step = (_clf, classifier_)

							pipeline_steps.append(classifier_step)

						elif _clf == "decision_tree":
							del new_dict['max_depth_factor']
							classifier_ = DecisionTreeClassifier(**new_dict)
							classifier_step = (_clf, classifier_)

							pipeline_steps.append(classifier_step)


						elif _clf == "gaussian_nb":
							classifier_ = GaussianNB(**new_dict)
							classifier_step = (_clf, classifier_)

							pipeline_steps.append(classifier_step)

						elif _clf == "gradient_boosting":
							del new_dict['scoring']
							del new_dict['max_iter']
							del new_dict['max_bins']
							del new_dict['l2_regularization']
							del new_dict['early_stop'] 
							classifier_ = GradientBoostingClassifier(**new_dict)
							classifier_step = (_clf, classifier_)
                            
							pipeline_steps.append(classifier_step)


						elif _clf == "k_nearest_neighbors":
							classifier_ = KNeighborsClassifier(**new_dict)
							classifier_step = (_clf, classifier_)

							pipeline_steps.append(classifier_step)


						elif _clf == "lda":
							classifier_ = LinearDiscriminantAnalysis()
							classifier_step = (_clf, classifier_)

							pipeline_steps.append(classifier_step)



						elif _clf == "liblinear_svc":
							classifier_ = CalibratedClassifierCV(base_estimator=LinearSVC(**new_dict))
							classifier_step = (_clf, classifier_)

							pipeline_steps.append(classifier_step)


						elif _clf == "libsvm_svc":
							classifier_ = SVC(**new_dict, probability=True)
							classifier_step = (_clf, classifier_)

							pipeline_steps.append(classifier_step)


						elif _clf == "multinomial_nb":
							classifier_ = MultinomialNB(**new_dict)
							classifier_step = (_clf, classifier_)

							pipeline_steps.append(classifier_step)


						elif _clf == "passive_aggressive":#
							pass
							#classifier_ = PassiveAggressiveClassifier(**new_dict)
							#classifier_step = (_clf, classifier_)
							#pipeline_steps.append(classifier_step)

						elif _clf == "qda":
							classifier_ = QuadraticDiscriminantAnalysis(**new_dict)
							classifier_step = (_clf, classifier_)

							pipeline_steps.append(classifier_step)

						elif _clf == "sgd":
							classifier_ = CalibratedClassifierCV(base_estimator=SGDClassifier(**new_dict))
							classifier_step = (_clf, classifier_)

							pipeline_steps.append(classifier_step)


						elif _clf == "xgradient_boosting":
							classifier_ = XGBClassifier(**new_dict)
							classifier_step = (_clf, classifier_)

							pipeline_steps.append(classifier_step)

				list_of_pipelines.append(pipeline_steps)
       
			missing_values = ["n/a", "na", "--", "?"]
			df = pd.read_csv('{}.csv'.format(dataset), na_values = missing_values)
			x_cols = [c for c in df.columns if c != 'target']
			x = df[x_cols]
			y = df['target']
			x.fillna(x.mean(), inplace=True)
			XX = pd.get_dummies(x, prefix_sep='_', drop_first=True)

			list_ps = validate_pipelines(list_of_pipelines,XX,y)
            
			results_of_pipelines, best_results_in_test = ranking_performance_train_test_pipelines(list_ps,XX,y,dataset,repetition)
            
			print("#########################")
			print("Best pipelines in dataset{}".format(dataset))
			print(best_results_in_test)
            
			final_time = time() - t0
			list_of_times.append((final_time,"dataset{}".format(dataset)))
			final_results.append((best_results_in_test))

		print(list_of_times)
		print(final_results)
##########################################################################

main(sys.argv[1])
