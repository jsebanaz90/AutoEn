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
            #pipes_validated.append((counter, no_valid, pipe_))
            print("something went wrong")
    
    print("pipes_validated: ",len(pipes_validated))
    return pipes_validated
    
##########################################################################

def caruana_ensemble(list_of_pipelines,X,y,dataset):

    times_all_folds = []
    time_caruana_ensemble_function = []
    t4 = time()
    
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)    
    fold_counter = 0
    
    ensemble_results_in_val = []
    ensemble_results_in_test = []
    
    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)): 
        X_train_g, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_g, y_test = y[train_idx], y[test_idx]
        
        
        fold_counter = fold_counter + 1
        print("fold in progress: {}".format(fold_counter))
        
        X_train, X_val, y_train, y_val = train_test_split(X_train_g, y_train_g, test_size = 0.2, shuffle=True, stratify = y_train_g, random_state=1)
        
        classifiers_predictions_in_test = []    
        clfs_and_all_pred_val = []       
        
        t1 = time()
  
 
                
        for i in list_of_pipelines:
                    
            counter_, _, pipe = i             
          
            classifier = first_training_classifier(pipe, X_train, y_train)              
                
            if classifier == None:
                pass
                        
                ####### train and test pipelines in train and validation sets
            else:                      
                predictions_train = classifier.predict_proba(X_train)
                result_train = log_loss(y_train,predictions_train)

                predictions_validation = classifier.predict_proba(X_val)
                result_val = log_loss(y_val,predictions_validation)  
                       
                clfs_and_all_pred_val.append((counter_, classifier, result_val, predictions_validation))             
            
        final_time_t1 = time() - t1

        
        ############## Build Final Caruana Ensemble
        
        t2 = time() ### time of building the ensemble
        
        max_perfm_metric = 0 
        initial_ensemble_in_val = []
        best_prediction_in_validation = []
        final_ensemble = []
        
        #### extract the best pipeline in the validation set
        best_val = min(clfs_and_all_pred_val, key = lambda x:x[2])     

        #### extract the best pipeline in validation, its performance in validation, and its raw predictions in validation
        final_pipe_counter, best_classifier_in_validation, best_classifier_perfm_val, predictions_best_classifier_in_y_val = best_val

        ### add the pipe Best_Val as first element of the ensemble
        initial_ensemble_in_val.append((final_pipe_counter, best_classifier_in_validation))

        ### save the raw validation predictions of Best_Val - then this is used by the ensemble
        best_prediction_in_validation.append(predictions_best_classifier_in_y_val)    
        
        ### calculate the performance in validation of the ensemble with only one item (this item is Best_Val)
        one_item_ensemble_prediction_y_val = 0    
        for prediction in best_prediction_in_validation:
            one_item_ensemble_prediction_y_val += prediction         
        max_perfm_metric = log_loss(y_val, one_item_ensemble_prediction_y_val)    
        print("Initial performance:", max_perfm_metric)

        final_ensemble = initial_ensemble_in_val ### variable to store the final ensemble (at the begininng it only has one item: BestVal)

        temporal_predictions = best_prediction_in_validation ### variable to store the raw predictions in y_val of the ensemble in construction        
        final_predictions_y_val = best_prediction_in_validation ### variable to store the predictions in y_val of the final ensemble
        
        ### build the ensemble with a maximum legnth of 50 classifiers
        while len(final_ensemble) < 50:
        
            results_temporal_ensembles = []     

            for clf in clfs_and_all_pred_val:           

                temporal_ensemble = initial_ensemble_in_val.copy()
                temporal_predictions = best_prediction_in_validation.copy()

                tmp_counter, classifier_, _, predictions_in_validation_ =  clf

                temporal_ensemble.append((tmp_counter, classifier_))            
                temporal_predictions.append(predictions_in_validation_) 

                size_ensemble_temporal = len(temporal_ensemble)
                
                final_prediction = 0        
                for prediction in temporal_predictions:
                    final_prediction += prediction/size_ensemble_temporal
                temporal_perfm_metric = log_loss(y_val, final_prediction)

                results_temporal_ensembles.append((temporal_perfm_metric,temporal_ensemble,temporal_predictions))

            maxVal_in_temporal_ensembles = min(results_temporal_ensembles, key = lambda x:x[0]) 
            candidate_ensemble_perfm, candidate_ensemble, candidate_ensemble_predictions = maxVal_in_temporal_ensembles

            #if candidate_ensemble_perfm > max_perfm_metric: 
            final_ensemble = candidate_ensemble.copy()
            final_predictions_y_val = candidate_ensemble_predictions.copy()             
            initial_ensemble_in_val = final_ensemble.copy() ###                
            best_prediction_in_validation = candidate_ensemble_predictions.copy()####

            max_perfm_metric = candidate_ensemble_perfm                

            #else:
                #initial_ensemble_in_val = candidate_ensemble.copy() ###
                #best_prediction_in_validation = candidate_ensemble_predictions.copy()####

        print("ensemble lenght: ", len(final_ensemble))
        print("Final performance:",max_perfm_metric)
        ensemble_results_in_val.append(max_perfm_metric)
        
        final_time_t2 = time() - t2        
         
        #######  ----- TEST DATA  ----- #######
        t3 = time() ### time of testing the ensemble
        
        ids_unique_pipes_ = []
        prev_value_ = None
        classifiers_predictions_in_test_ = []
        
        for i in final_ensemble:
        
            test_counter, pipeline = i
            
            ### VALIDATION TO Re-TRAIN ONLY ONCE A PIPELINE WITH MULTIPLE REPETITIONS IN THE FINAL ENSEMBLE         
               
            if prev_value_ is None:
                 
                pipe_2 = second_training_classifier(pipeline, X_train_g, y_train_g)
                    
                if pipe_2 == None:
                    pass
                    
                else: 
                    prev_value_ = test_counter
                    ids_unique_pipes_.append(prev_value_)                   
                    
                    predictions_test = pipe_2.predict_proba(X_test)
                    classifiers_predictions_in_test_.append((prev_value_, predictions_test)) 
                    classifiers_predictions_in_test.append(predictions_test) 
                        
            else:
                prev_value_ = test_counter
                
                if prev_value_ in ids_unique_pipes_: #####
                    print("YES, repeated pipeline for RE-TRAINING")
                    
                    for i in classifiers_predictions_in_test_:
                        tmp_id, tmp_test_preds = i
                        
                        if tmp_id == prev_value_:
                            classifiers_predictions_in_test.append(tmp_test_preds)                
                  
                else:
                    pipe_2 = second_training_classifier(pipeline, X_train_g, y_train_g)
                    
                    if pipe_2 == None:
                        pass
                        
                    else: 
                        ids_unique_pipes_.append(prev_value_)
                        predictions_test = pipe_2.predict_proba(X_test)
                        classifiers_predictions_in_test_.append((prev_value_, predictions_test)) 
                        classifiers_predictions_in_test.append(predictions_test) 
      
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

def main(dataset_name):
		dataset = dataset_name
		list_of_times = []
    
	#try:
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
			time_left_for_this_task=60,
			per_run_time_limit=30,
			ml_memory_limit = 500000,
			delete_tmp_folder_after_terminate=True,
			initial_configurations_via_metalearning=121)

			automl.fit(X_train, y_train)

			MetalearningSuggestions = automl.MetalearningSuggestions_extractor()

			suggestions = []

			for i in MetalearningSuggestions:
				pipeline = i.get_dictionary()
				suggestions.append(pipeline)

			t0 = time()
			t = time()
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
            
			time_building_validate_pipelines = time() - t0
            
			ensemble_results_in_test, ensemble_results_in_val, times_all_folds, time_caruana_ensemble_function = caruana_ensemble(list_ps,XX,y,dataset)
            
			final_time = time() - t
            
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
			print("total_time_caruana_ensemble_function: ",time_caruana_ensemble_function)
			print("avg x fold t1: ",statistics.mean(t1))
			print("avg x fold t2: ",statistics.mean(t2))
			print("avg x fold t3: ",statistics.mean(t3))

			list_of_times.append((final_time,"dataset{}".format(dataset)))

	#except:
		#print("something wrong loading dataset: {}".format(dataset))
		print("time_building_validate_pipelines: ", time_building_validate_pipelines)
		print("Total time: ", list_of_times)
##########################################################################

main(sys.argv[1])
