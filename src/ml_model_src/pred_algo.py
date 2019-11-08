from __future__ import print_function
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import plot_ml_results as pr

from imblearn.over_sampling import SMOTE, SVMSMOTE
from pandas_ml import ConfusionMatrix
from scipy.stats import uniform
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error  
warnings.filterwarnings('ignore')


class PredAlgClass():
    """
    """
    def __init__(self, X,y, n_folds, flag='Train'): #poly_degree, models, training_errors, validation_errors, test_errors
        """
            Args:
                X: data set
                y: label set
                n_folds: number of folds

        """  
        self.X = X
        self.y = y
        #self.poly_degree = poly_degree
        self.n_folds = n_folds
        self.name = ""
        if (flag == 'Train'):
            self.__pred_training__()
          

    def __calculate_errors__(self, y_val, y_predictions):
        """
            Args:
                y_val: truth value
                y_predictions: predicted value
            Calculate and return confusion matrix, accuray, and root mean squared error between the predicted value wrt the truth value.
            Plot of the confusion matrix can be found in: "./MaLer/Plots/confusion_matrix/"
        """
        cnf_matrix_tra = confusion_matrix(y_val['readmitted'], y_predictions)
        cn =pd.crosstab(y_val['readmitted'], y_predictions, rownames=['True'], colnames=['Predicted']).apply(lambda r: 100.0 * r/r.sum()) # margins=True

        #print(cnf_matrix_tra)
        Confusion_Matrix = ConfusionMatrix(y_val['readmitted'], y_predictions)
        Confusion_Matrix.print_stats()
        stats = Confusion_Matrix.stats()
        accuracy = stats['overall']['Accuracy']#accuracy_score(y_val['readmitted'], y_predictions)
        rmse = np.sqrt(mean_squared_error(y_val['readmitted'], y_predictions))
        pr.plot_confusion_matrix_sn(self.type_al,self.name, cnf_matrix_tra, cn)
        pr.plot_confusion_matrix(self.type_al,self.name, cnf_matrix_tra)#Confusion_Matrix)#cnf_matrix_tra)
        
        return rmse, accuracy, cn 

    def __random_search__(self, lr, parameters, random_state, verbose, n_iter, cv, n_jobs, x_train_res, y_train_res):
        """
            Random search for optimization of hyperparameters
            Return:
                model with the best parameters
        """
        clf = RandomizedSearchCV(lr, parameters, random_state=random_state,verbose = verbose, n_iter=n_iter, cv=cv, n_jobs=n_jobs)
        clf.fit(x_train_res, y_train_res)
        print('Best clf :', clf.best_params_)
        return clf

    def __best_classifier__(self, model):
        """
            Identify optimal hyperparameter values depending on the type of algorithm
            Return:
                classifier
        """
        cl = None 
        if(self.type_al=='random_forest'):
            cl = RandomForestClassifier(n_estimators=model.best_params_['n_estimators'], max_depth=model.best_params_['max_depth'], 
            min_samples_leaf=model.best_params_['min_samples_leaf'])
            
        elif(self.type_al=='log_reg'):
            cl = LogisticRegression(C=model.best_params_['C'], penalty=model.best_estimator_.get_params()['penalty'], 
            verbose=10, multi_class='auto', solver='lbfgs',  max_iter=100) #solver='lbfgs'
            
        elif(self.type_al=='svm'):
            cl = svm.SVC(kernel='rbf', C=model.best_estimator_.get_params()['C'], gamma=model.best_estimator_.get_params()['gamma']) #rbf,
        return cl
            
    def __split_data__ (self):
        tscv = TimeSeriesSplit(n_splits=self.n_folds)#train_test_split(self.X, self.y, test_size=0.25) # stratify=self.y
        x_train, x_val, y_train, y_val = [], [], [], []  #pd.DataFrame()
        for train_index, test_index in tscv.split(self.X):
            x_train.append(self.X.iloc[train_index]) 
            x_val.append(self.X.iloc[test_index])
            y_train.append(self.y.iloc[train_index]) 
            y_val.append(self.y.iloc[test_index])
        return x_train, y_train, x_val, y_val

    def __model__(self,smt,little, classifier, parameters,n_iter, cv, n_jobs):
        """
            Args:
                smt: SMOTE algorithm
                little: string short_name of the ml_algorithm
                classifier: specific ml_algorithm
                parameters: hyperparameters to be optimized
                n_iter,cv and n_jobs are needed for RandomSearch
            Return:
                models, errors(rmse), accuracy, confusion matrix, total time of training
            Train the model depending on the ml_algorithm
        """
        print("------------------------------------------------------------------------------")
        print("---------------------------"+self.type_al+" + SMOTE------------------") 
        scores, models, rmse, accuracy, cf = [], [], [], [], []
        start = time.time()
        x_train, y_train, x_val, y_val = self.__split_data__()
        for i in range(self.n_folds):
            self.name = little+'_'+str(i+1)
            print("\rFold {}/{}.\n".format(i+1, self.n_folds), end="\r")
            sys.stdout.flush()
            #--------------------SMOTE:---------------------------------------
            print('Create oversamples with SMOTE algorithm for multiple classes ... ')
            x_train_res, y_train_res = smt.fit_sample(x_train[i], y_train[i]['readmitted']) #y_train['readmitted']
            pr.plot_total_classes_SMOTE(y_train_res, self.type_al, self.name, prefix='before_train_')
            #----------------------Training------------------------------------
            print('Training starts...')
            cl = classifier
            cl.fit(x_train_res,y_train_res)
            train_new = x_train_res
            if(self.type_al == 'random_forest'):
                print('Select from model')
                sl_model = SelectFromModel(cl, prefit=True)
                train_new = sl_model.transform(x_train_res)
                print('len train new: ', len(train_new))
            
            #------------------------Random Search-------------------------------
            print('Start random search...')
            best_model = self.__random_search__(cl, parameters, 1,5, n_iter, cv, n_jobs, train_new, y_train_res)
            cl = self.__best_classifier__(best_model)
            model = cl.fit(x_train_res,y_train_res)
            
            #Saving model
            print('Saving the model...')
            models.append(model)
            filename = './MaLer/Models/'+self.type_al+'/'+self.name+'.sav'
            joblib.dump(model, filename)
            print('Training finishes....', 'Starts validation...')
            
            #-----------------------Validation-------------------------------------
            y_predictions = model.predict(x_val[i])
            print('Validation is over..')
            scores.append(model.score(x_val[i],y_val[i]['readmitted']))
            rmses, accuracies, cfs = self.__calculate_errors__(y_val[i], y_predictions)
            rmse.append(rmses)
            accuracy.append(accuracies)
            cf.append(cf)
            print('----------------------------------------------------------------------------------------------')
        end = time.time()
        total = end - start
        return models, rmse, accuracy,cf, total
   
        
    def __dif_algo_training__(self):
        """
            Three ml_algorithms are used for training: Random Forest, Logistic Regression and Support Vector Machine (SVM)
        """
        if (self.type_al == 'random_forest'):
            smt = SMOTE(sampling_strategy = 'not majority',random_state=2, k_neighbors=3)
            rf_cl = RandomForestClassifier() #RandomForestRegressor(n_estimators=100)   
            
            # Designate distributions to sample hyperparameters from 
            n_estimators = [1, 2, 4, 8, 16, 32] #np.random.uniform(70, 80, 5).astype(int)
            max_depths = np.linspace(1, 24, 24, endpoint=True)

            parameters = {
                    'n_estimators'      :  list(n_estimators),
                    'max_depth'         : list(max_depths),
                    'min_samples_leaf'  : [2, 3, 4]
                }
            return self.__model__(smt,'rf',rf_cl,parameters,10, 5, 2)

        elif(self.type_al == 'log_reg'):
            smt = SMOTE(sampling_strategy = 'not majority', random_state=2, k_neighbors=3)
            lr = LogisticRegression( multi_class='auto', solver='lbfgs',  max_iter=100) #solver='lbfgs', 'saga'
            parameters = {
                'penalty':['l2'],
                'C': uniform(loc=0, scale=4) #np.linspace(1, 10, 10)
                }
            return self.__model__(smt,'lr',lr,parameters,10, 5, -1)

        elif(self.type_al == 'svm'):
            C_range = np.random.normal(1, 0.1, 5)
            gamma_range = np.random.uniform(1e-5,1,5) #,5
            parameters = dict(gamma=gamma_range, C=C_range)
            svm_cl = svm.SVC(kernel='rbf')

            smt_svm = SVMSMOTE(sampling_strategy = 'not majority', random_state=42)#SMOTE(kind="svm")
            return self.__model__(smt_svm,'svm',svm_cl,parameters,3, 3, -1)

        
    def __pred_training__(self):
        """
            Verify if the model has been pretrained; otherwise, starts training.
            Then predict on unseen data (test set)

        """
        type_algos = self.__check_saved_models__()
        if not (type_algos == None):
            error_models = {'models':[], 'rmse':[], 'accuracy':[], 'cf':[], 'runtime':[]}
            old_n_folds = self.n_folds
            for type_al in type_algos:
                self.type_al = type_al
                model, rmse, accuracy, cf,time= self.__dif_algo_training__()
                error_models['models'].append(model)
                error_models['rmse'].append(rmse)
                error_models['accuracy'].append(accuracy)
                error_models['cf'].append(cf)
                error_models['runtime'].append(time)
                #np.save('./MaLer/Models/metrics/error_models_'+self.type_al+'.npy', error_models)
                np.savez('./MaLer/Models/metrics/error_models_'+self.type_al+'.npz', error_models)
            print('------------------------------TRAINING IS OVER-------------------------------------------------')
    
    def __check_saved_models__(self):
        """
            Verify if the model has been pretrained
        """
        fname1 = r'./MaLer/Models/svm/svm_'+str(self.n_folds)+'.sav'
        fname2 = r'./MaLer/Models/random_forest/rf_'+str(self.n_folds)+'.sav'
        fname3 = r'./MaLer/Models/log_reg/lr_'+str(self.n_folds)+'.sav'
        # ************ Cleaning the data *****************************
        if os.path.isfile(fname1) :
            print('Model SVM has already pretrained...')
            if os.path.isfile(fname2) :
                print('Model RF has already pretrained...')
                if os.path.isfile(fname3) :
                    print('Model LR has already pretrained...')
                    return None
                else:
                    return ['log_reg']
            else:
                return ['random_forest','log_reg']
        else:
            return [ 'svm','random_forest','log_reg'] #'svm'
            
    def __compare_models__(self):
        """
            Compare the results (accuracy and rmse) of each model
            Plot the results, where they can be found in './MaLer/Plots/compare_algos/'
            Return:
                the best model name and its parameters 
        """
        type_algos = ['svm', 'random_forest','log_reg'] #'svm' predicts only one type of class 
        values = []
        rmse = []
        max_value = []
        indices = []
        for i,type_al in enumerate(type_algos):
            #i = i+1
            #read_dictionary = np.load('./MaLer/Models/metrics/error_models_'+type_al+'.npy', allow_pickle=True)#.item()
            read_dictionary = np.load('./MaLer/Models/metrics/error_models_'+type_al+'.npz', allow_pickle=True)#.item()
            #models:0, rmse:1, accuracy:2, cf:3, runtime:4
            #print(type(read_dictionary))
            a = {key:read_dictionary[key].item() for key in read_dictionary}
            print(type_al +'------------------------------------------------------')
            if (type_al=='svm'):
                continue
            #print(a['arr_0']['accuracy'])
            #print(a['arr_0']['rmse'])
            values.append(a['arr_0']['accuracy'][i])
            max_value.append(np.amax(values[i-1]))
            indices.append(np.argmax(values[i-1]))
            rmse.append(a['arr_0']['rmse'][i][indices[i-1]])
            #print(values)
            #print(rmse)
            #print(max_value)
            #print(indices)
            
            """
            for value in a['arr_0']['accuracy'][i]:
                print (value)
                print(type(value))
                print(np.amax(value))
                max_value.append(np.amax(value))
                indices.append(np.argmax(value))
            """
            pr.plot_comparison_models(values, rmse)
        #print(max_value)
        index_max =  np.argmax(max_value)
        #print(index_max)
        best_model_name = type_algos[index_max+1]
        best_param = indices[index_max+1]
        print(best_param)
        print('The best model is: ',best_model_name)
        return best_model_name, best_param

    def __rmse__(self, y, y_predicted):
        return np.sqrt(np.mean(0.5 * (y - y_predicted)**2))
    
    def __pred_testing__(self):
        """
            First verify which ml_algorithm got the lowest RMSE and highest Accuracy by comparing them.

        """
        #Get the best algorithm name and its parameters
        best_model_name, best_param = self.__compare_models__()
        names = []
        result_dict = dict()
        predictions_dict = dict()
        best_name = ''

        if (best_model_name == 'svm'):
            filename = './MaLer/Models/svm/svm_'+str(best_param+1)+'.sav'
            name = 'SVM'
            best_name = name
        elif(best_model_name == 'random_forest'):
            filename = './MaLer/Models/random_forest/rf_'+str(best_param+1)+'.sav'
            name = 'Random Forest'
            best_name = name
        elif(best_model_name == 'log_reg'):
            filename = './MaLer/Models/log_reg/lr_'+str(best_param+1)+'.sav'
            name = 'Log. Reggression'
            best_name = 'Logistic Regression'
        
        #Load the best model
        loaded_model = joblib.load(filename)
        result = loaded_model.score(self.X, self.y['readmitted'])
        predictions = loaded_model.predict(self.X)
        if name in result_dict:
            result_dict[name].append(result)
            predictions_dict[name].append(predictions)
            
        else:
            result_dict[name] = [result]
            predictions_dict[name] = [predictions]
        self.predictions = predictions
        rmse_list = []
        rmses = self.__rmse__(self.y['readmitted'], predictions)
        rmse_list.append(rmses)
        print('Plot and save the RMSE and accuracy...')
        pr.plot_predict_test(result_dict, rmse_list, best_name)


    def get_predictions(self):
        print('Get predictions on unseen data...')
        self.__pred_testing__()
        return self.predictions