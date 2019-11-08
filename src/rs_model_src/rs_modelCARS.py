from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.rs_model_src.camf as camf
import src.rs_model_src.ers as ers
import src.rs_model_src.feature_matrix_cl as fmc

import csv
import os
import scipy
import sys
import warnings
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

class CARS():
    """
    """
    def __init__(self,data, active_users):
        """
            Args:
                data: all the data set
                active_users: encounters which were predicted as readmitted from the first part of the project
        """ 
        self.data = data
        self.active_users = active_users
        self.n_folds = 3
        self.__prepare_feature_matrix__()
        self.__train__()

    def __prepare_feature_matrix__(self):
        """
        Prepare the feature matrix (user-item-ratings matrix). 
            -feature matrix which contains all unique encounters without considering the active users for the training of the recommender system
            -feature matrix pivot contains all unique encounters and the active users
        Save feature matrix, feature pivot and the list of active users. The first two can be found in '/RecSys/feature_matrix', the last can be
            located in '/RecSys/out/test/'
        """
        #fname = r'./RecSys/feature_matrix/feature_matrix.csv'
        #if os.path.isfile(fname) :
        #    self.feature_matrix = pd.read_csv(fname)
        #    fname = r'./RecSys/feature_matrix/feature_matrix_pivot.csv'
        #    self.feature_matrix_pivot = pd.read_csv(fname)
        #else:

        #object_um = fmc.FeatureMatrixClass(self.data.head(10), self.active_users.head(10))
        object_um = fmc.FeatureMatrixClass(self.data, self.active_users)
        #self.feature_matrix, self.feature_matrix_pivot = object_um.feature_matrix()
        #self.feature_matrix.to_csv("./RecSys/feature_matrix/feature_matrix.csv",index=False)
        #self.feature_matrix_pivot.to_csv("./RecSys/feature_matrix/feature_matrix_pivot.csv",index=False)
        self.feature_matrix, self.pivot = object_um.get_feature_matrix()
        self.active_users.to_csv('./RecSys/out/test/au_cars.csv',index=False)
    
    def __validation_model__(self, model, fold, algo,loop):
        """
            Method for validation using the validation set for each user and each item that this user has rated. 
            Get the real rating and the predicted rating for each user and item.
            Calculate the errors RMSE and MAE.
            Return:
                Root Mean Square Error (RMSE)
                Mean Absolute Error (MAE)
        """
        fname = r'./RecSys/out/CAMF/predictions/'+algo+'/'+str(fold+1)+'_predictions_loop'+str(loop)+'.csv'
        if not(os.path.isfile(fname) ):
            matrix_columns = ['user', 'item','prediction', 'real']
            final_pred = pd.DataFrame(columns=matrix_columns)
            i=1
            for user, items in self.validation_utility.groupby('encounter_id'):
                print("\r Validation Loop {}/{}.".format(i, len(self.validation_utility.groupby('encounter_id'))), end='\r')
                sys.stdout.flush()
                #if(i == 10):
                #    break
                i = i+1
                items = np.array([1,2])#items = ['metformin', 'insulin'
                for item in items:
                    dummy = self.validation_utility.loc[self.validation_utility['encounter_id']==user]
                    real = 0
                    #print(dummy)
                    if(item in dummy['item_id']):
                        real = dummy['ratings'].values[0]

                    pred = model.predict(user, item).ratings
                    prediction = 0
                    if not pred.empty:
                        prediction = pred.values[0]
                    record = pd.Series([user, item, prediction, real], index=matrix_columns)
                    final_pred = final_pred.append(record, ignore_index=True)
            
            self.__save_validation__(final_pred, fold, loop, algo=algo)
        final_pred = pd.read_csv(fname)
        rmse, mae = self.__evaluation__(final_pred)
        print('Validation is over..')
        return rmse, mae
 
    def __training_model__(self, algo=None, model = None):
        """
            Method for training depending on the algorithm Matrix Factorization (MF) or Item-based collaborative filter (baseline)
            Split the data in training set and validation set.
            Save the models
            Validate the models 
            Called the save method for the errors RMSE and MAE
        """
        models_all = []
        rmses_all = []
        maes_all = [ ]
        for fold in np.arange(self.n_folds):
            models = []
            rmses = []
            maes = [ ]
            print("\rFOLD Loop {}/{}.".format(fold+1, self.n_folds), end='\r')
            print('-----------------------------------------------')
            sys.stdout.flush()
            #training_data =training_utility[initial:final]
            print('Cross validation in progress...')
            training_utility, self.validation_utility = train_test_split(self.feature_matrix, test_size=0.25)
            print(training_utility.shape, self.validation_utility.shape)
            #print(training_utility.dtypes)
            name = 'data_fold_'+str(fold+1)
            learning_rate = [0.001,0.01,0.005,0.009,0.003]
            num_factors = [20,30,50,70,80]
            #crds = [20,40,60,81]

            for i in np.arange(0,1): #this can be change to the size of learning_rate array
                print('Starts training ------------------------------------------------------------')
                fname = r'./RecSys/out/CAMF/models/'+self.type_rs+'/model_'+str(fold+1)+'_loop'+str(i)+'.sav'
                if not(os.path.isfile(fname)) :
                    model = camf.CAMF_class(self.type_rs,training_utility, self.pivot, fold, learning_rate[i], num_factors[i], i,save_prefix=name)
                    models.append(model)
                    print('Training finishes....', 'Saving the model...')
                    self.__saving_models__(model,fold, i, algo=algo)
                else:
                    print('Model already saved')
                    model = joblib.load(fname)    
                print('Starts validation...')
                rmse, mae= self.__validation_model__(model, fold, algo, i)
                rmses.append(rmse)
                maes.append(mae)
            models_all.append(models)
            rmses_all.append(rmses)
            maes_all.append(maes)
        print('Saving the evaluation metrics....')
        self.__save_metrics__(rmses_all, maes_all, './RecSys/out/CAMF/predictions/'+algo+'/')

   
    def __train__(self):
        for rs_algo in ['CAMF_CU','CAMF_C','CAMF_CI' ]:#'CAMF_CU','CAMF_C','CAMF_CI' 
            self.type_rs = rs_algo
            fname = r'./RecSys/out/CAMF/models/'+rs_algo+'/model_'+str(self.n_folds)+'.sav'
            if os.path.isfile(fname) :
                print("Model was already saved for algorthim ",rs_algo)
                #model = joblib.load(fname)
                #self.__training_model__(algo=rs_algo, model=model)
            else:
                print('Algorithm ',rs_algo)
                self.__training_model__(algo=rs_algo)

    def __test__(self,algo=None):
        """
            This method is called after the models have been trained. 
            Predict the ratings for the active users (unseen data)
            Calculate the errors RMSE and MAE
            Save predictions and errors
        """
        items = np.array([1,2]) #items = ['metformin', 'insulin']
        rmses = []
        maes = []
        copy_final= pd.DataFrame()
        all_predictions = []
        for fold in np.arange(self.n_folds):
            pred_dict ={'encounter_id':[], 1:[], 2:[]}#{'encounter_id':[], 'metformin':[], 'insulin':[]}
            print("\rFOLD Loop {}/{}.".format(fold+1, self.n_folds), end='\r')
            print('-----------------------------------------------')
            sys.stdout.flush()
            mf_object = None
            fname = r'./RecSys/out/CAMF/models/'+algo+'/model_'+str(fold+1)+'_loop0.sav'
            if os.path.isfile(fname):
                mf_object = joblib.load(fname)
                print('Start prediction on testing data')
                matrix_columns = ['user', 'item','prediction', 'real']
                final_pred = pd.DataFrame(columns=matrix_columns)
                i=0
                for a_user, value in self.active_users.groupby('encounter_id'):
                    #print(a_user)
                    i = i+1
                    print("\rFOLD Loop {}/{}.".format(i, len(self.active_users)), end='\r')
                    sys.stdout.flush()
                    for item in items:
                        dummy = self.pivot.loc[self.pivot['encounter_id']==a_user]
                        real = 0
                        if(item in dummy['item_id']):
                            real = dummy['rating'].values[0]
                        prediction = 0
                        pred = mf_object.predict(a_user, item).ratings
                        if not pred.empty:
                            prediction = pred.values[0]
                        if 'encounter_id' in pred_dict:
                            if(item == 2):
                                pred_dict['encounter_id'].append(a_user)
                            pred_dict[item].append(prediction) #items
                        else:
                            if(item == 2):
                                pred_dict['encounter_id']=[a_user]
                            pred_dict[item]=[prediction] #items
                        record = pd.Series([a_user, item, prediction, real], index=matrix_columns)
                        final_pred = final_pred.append(record, ignore_index=True)
                copy_final_pred = pd.DataFrame(pred_dict)
                copy_final_pred.to_csv("./RecSys/out/CAMF/test/"+algo+'/'+str(fold+1)+"_predictions.csv",index=False)
                all_predictions.append(copy_final_pred)
                rmse, mae, = self.__evaluation__(final_pred)
                rmses.append(rmse)
                maes.append(mae)
        self.__save_metrics__(rmses, maes, './RecSys/out/CAMF/test/'+algo+'/')
        #print(len(all_predictions))
        self.__checking_predictions__(all_predictions, algo=algo)
      

    def __checking_predictions__(self, list_predictions, algo=None):
        items = np.array([1,2]) #items = ['metformin', 'insulin'
        all_good_pred = dict()
        i=0
        count = 0
        for a_user, value in self.active_users.groupby('encounter_id'):
            i = i+1
            print("\rLoop {}/{}.".format(i, len(self.active_users)), end='\r')
            sys.stdout.flush()
            all_good_pred[a_user] = {}
            #print(list_predictions)
            for pred in list_predictions:
                #print(pred)
                user_items = pred.loc[pred['encounter_id'] == a_user]
                for item in items:
                    #print(user_items[item])
                    if (float(user_items[item]) >-1):
                        if not (self.__keys_exists__(all_good_pred, (item,a_user))):
                            all_good_pred[a_user]['encounter_id'] = [a_user]
                            all_good_pred[a_user][item] = [float(user_items[item].values)]

                        else:
                            if(all_good_pred[a_user][item] !=[user_items[item]].values):
                                all_good_pred[a_user][item].append(float(user_items[item].values))
                                count = count + 1
        #print(all_good_pred)
        df = pd.DataFrame.from_dict(all_good_pred, orient='index')
        #print(df)
        df.to_csv("./RecSys/out/CAMF/test/"+algo+"/au_predictions.csv",index=False)
        print("predictions were saved at: '/RecSys/out/CAMF/test/'"+algo+'/')
        #print(count)

    def __keys_exists__(self, element, *keys):
        '''
        Check if *keys (nested) exists in `element` (dict).
        '''
        if type(element) is not dict:
            raise AttributeError('keys_exists() expects dict as first argument.')
        if len(keys) == 0:
            raise AttributeError('keys_exists() expects at least two arguments, one given.')
        _element = element
        for key in keys:
            try:
                _element = _element[key]
            except KeyError:
                return False
        return True


    def __evaluation__(self, data):
        """
            Create an evaluation object for calculating the root mean square error (RMSE) and mean absolute error (MAE)
        """
        ers_object = ers.Evaluation(data)
        rmse, mae= ers_object.calculate_errors()
        return rmse, mae
    
    def get_predictions(self):
        """
            Start testing
        """
        for rs_algo in ['CAMF_CI', 'CAMF_CU','CAMF_C']:#'CAMF_CU','CAMF_C', 
            print('Making predictions of the encounter from the first part (test data) with ',rs_algo)
            self.type_rs = rs_algo
            self.__test__(algo=rs_algo)

    def __saving_models__(self, model,fold,loop, algo=None):
        filename = './RecSys/out/CAMF/models/'+algo+'/model_'+str(fold+1)+'_loop'+str(loop)+'.sav'
        joblib.dump(model, filename)
        print("Model is saved at: '"+filename+"'")
    
    def __save_validation__(self, final_pred,fold, loop, algo=None):
        filename = './RecSys/out/CAMF/predictions/'+algo+'/'+str(fold+1)+'_predictions_loop'+str(loop)+'.csv'
        final_pred.to_csv(filename, index=False)
        print("Predictions are saved at: '"+filename+"'")

    def __save_metrics__(self, rmses, maes, save_path):
        """
            Method for saving the errors RMSE and MAE
        """
        metrics_results = pd.DataFrame(columns=['RMSE', 'MAE'])
        rmse = rmses#np.mean(rmses)
        mae = maes#np.mean(maes)
        record = pd.Series([rmse, mae], index=['RMSE', 'MAE'])
        metrics_results = metrics_results.append(record, ignore_index=True)
        save_path = save_path+'metrics_results.csv'
        metrics_results.to_csv(save_path, index=False)
        print("Metrics results can be found here: " + save_path)
    
    def isSparse(self, array,m, n) : 
        counter = 0
        # Count number of zeros 
        # in the matrix 
        for i in range(0,m) : 
            for j in range(0,n) : 
                if (array[i][j] == 0) : 
                    counter = counter + 1
        percentage = (counter/(m*n)) * 100
        return percentage  