from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import ers
import os
#import recmetrics
import rs_baseline
import scipy
import sys
import utility_matrix_cl
import warnings
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import recmetrics
import matrix_factorization
warnings.filterwarnings('ignore')

class RSmodel():
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
        self.__prepare_utility_matrix__()
        self.__train__()

    def __prepare_utility_matrix__(self):
        """
        Prepare the utility matrix (user-item-ratings matrix). 
            -utility matrix which contains all unique encounters without considering the active users for the training of the recommender system
            -utility matrix pivot contains all unique encounters and the active users
        Save utility matrix, utility pivot and the list of active users. The first two can be found in '/RecSys/utility_matrix', the last can be
            located in '/RecSys/out/test/'
        """
        fname = r'./RecSys/utility_matrix/utility_matrix.csv'
        if os.path.isfile(fname) :
            self.utility_matrix = pd.read_csv(fname)
            fname = r'./RecSys/utility_matrix/utility_matrix_pivot.csv'
            self.utility_matrix_pivot = pd.read_csv(fname)
        else:
            object_um = utility_matrix_cl.UtilityMatrixClass(self.data, self.active_users)
            self.utility_matrix, self.utility_matrix_pivot = object_um.utility_matrix()
            self.utility_matrix.to_csv("./RecSys/utility_matrix/utility_matrix.csv",index=False)
            self.utility_matrix_pivot.to_csv("./RecSys/utility_matrix/utility_matrix_pivot.csv",index=False)
        self.active_users.to_csv('./RecSys/out/test/au_list.csv',index=False)
        #long_tail =recmetrics.Recmetircs_class(self.utility_matrix, self.utility_matrix_pivot)
        #long_tail.plot()
        #m = 28499
        #n = 13
        #sparse = self.utility_matrix_pivot.drop(['encounter_id', 'readmitted'],axis=1)
        #print(isSparse(sparse.values, m, n)) 
        #exit()
    
    def __validation_model__(self, model, fold, algo):
        """
            Method for validation using the validation set for each user and each item that this user has rated. 
            Get the real rating and the predicted rating for each user and item.
            Calculate the errors RMSE and MAE.
            Return:
                Root Mean Square Error (RMSE)
                Mean Absolute Error (MAE)
        """
        fname = r'./RecSys/out/'+self.type_rs+'/predictions/'+str(fold+1)+'_predictions.csv'
        if not(os.path.isfile(fname) ):
            matrix_columns = ['user', 'item','prediction', 'real']
            final_pred = pd.DataFrame(columns=matrix_columns)
            i=1
            for user, items in self.validation_utility.groupby('encounter_id'):
                print("\r Validation Loop {}/{}.".format(i, len(self.validation_utility.groupby('encounter_id'))), end='\r')
                sys.stdout.flush()
                i = i+1
                items = items.columns.values.tolist()
                items.remove('encounter_id')
                for item in items:
                    dummy = self.validation_utility.loc[self.validation_utility['encounter_id']==user]
                    real = 0
                    if(item in dummy.keys()):
                        real = float(dummy[item].values)
                    if(self.type_rs == 'MF'):
                        prediction = float(model.predict(user, item).values) #??values needed or not
                    elif (self.type_rs == 'baseline'):
                        prediction = float(model.predict(user, item))
                    else:
                        prediction = model.predict(user, item).values
                        #print(prediction)
                    record = pd.Series([user, item, prediction, real], index=matrix_columns)
                    final_pred = final_pred.append(record, ignore_index=True)
            
            self.__save_validation__(final_pred, fold, algo=algo)
            if(self.type_rs == 'MF'):
                losses = model.get_losses()
                min_error = np.min(losses)
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
        models = []
        rmses = []
        maes = [ ]
        for fold in np.arange(self.n_folds):
            print("\rFOLD Loop {}/{}.".format(fold+1, self.n_folds), end='\r')
            print('-----------------------------------------------')
            sys.stdout.flush()
            #training_data =training_utility[initial:final]
            print('Cross validation in progress...')
            training_utility, self.validation_utility = train_test_split(self.utility_matrix, test_size=0.25)
            print(training_utility.shape, self.validation_utility.shape)
            #print(training_utility.dtypes)
            name = 'data_fold_'+str(fold+1)
            print('Starts training ------------------------------------------------------------')
            if(self.type_rs == 'MF'):
                print('-----------------------------------Matrix Factorization------------------------------------')
                model = matrix_factorization.MF_class(training_utility, self.utility_matrix_pivot, fold, save_prefix=name)

            elif(self.type_rs == 'baseline'):
                print('-----------------------------------------BASELINE------------------------------------')
                model = rs_baseline.RecS_baseline_class(training_utility, self.utility_matrix_pivot, save_prefix=name)
            
            models.append(model)
            print('Training finishes....', 'Starts validation...')
            self.__saving_models__(model,fold, algo=algo)
            rmse, mae= self.__validation_model__(model, fold, algo)
            rmses.append(rmse)
            maes.append(mae)
        self.__save_metrics__(rmses, maes, './RecSys/out/'+self.type_rs+'/predictions/')


    
    def __train__(self):
        types_algo = ['MF', 'baseline']
        for rs_algo in types_algo:
            print('Algorithm type: ', rs_algo)
            self.type_rs = rs_algo
            fname = r'./RecSys/out/'+rs_algo+'/models/model_'+str(self.n_folds)+'.sav'
            if os.path.isfile(fname) :
                print("Model was already saved")
                #model = joblib.load(fname)
                #self.__training_model__(model=model)
            else:
                self.__training_model__()
       
    def __test__(self):
        """
            This method is called after the models have been trained. It was observed that algorithm MF obtained less RMSE and MAE error
            than the baseline. Therefore, for the final predictions, the MF algorithm is used.
            Predict the ratings for the active users (unseen data)
            Calculate the errors RMSE and MAE
            Save predictions and errors
        """
        items = ['insulin', 'metformin','HBA', 'max_glu_serum', 'diag_1', 'diag_2', 'diag_3']
        rmses = []
        maes = []
        copy_final= pd.DataFrame()
        all_predictions = []
        for fold in np.arange(self.n_folds):
            pred_dict ={'encounter_id':[], 'insulin':[], 'metformin':[],'HBA':[], 'max_glu_serum':[], 'diag_1':[], 'diag_2':[], 'diag_3':[]}
            print("\rFOLD Loop {}/{}.".format(fold+1, self.n_folds), end='\r')
            sys.stdout.flush()
            mf_object = None
            fname = r'./RecSys/out/MF/models/model_'+str(fold+1)+'.sav'
            if os.path.isfile(fname) :
                mf_object = joblib.load(fname)
            print('Start prediction on testing data')
            fname = r'./RecSys/out/MF/test/'+str(fold+1)+'_predictions.csv'
            matrix_columns = ['user', 'item','prediction', 'real']
            final_pred = pd.DataFrame(columns=matrix_columns)
            if not os.path.isfile(fname):
                i=0
                for a_user, value in self.active_users.groupby('encounter_id'):
                    #print(a_user)
                    i = i+1
                    print("\rFOLD Loop {}/{}.".format(i, len(self.active_users)), end='\r')
                    sys.stdout.flush()
                    for item in items:
                        dummy = self.utility_matrix_pivot.loc[self.utility_matrix_pivot['encounter_id']==a_user]
                        real = 0
                        if(item in dummy.keys()):
                            real = float(dummy[item].values)
                        prediction = mf_object.predict(a_user, item).values
                        if not prediction:
                            prediction = 0
                        prediction = float(prediction)
                        if 'encounter_id' in pred_dict:
                            if(item == 'diag_3'):
                                pred_dict['encounter_id'].append(a_user)
                            pred_dict[item].append(prediction) #items
                        else:
                            if(item == 'diag_3'):
                                pred_dict['encounter_id']=[a_user]
                            pred_dict[item]=[prediction] #items
                        record = pd.Series([a_user, item, prediction, real], index=matrix_columns)
                        final_pred = final_pred.append(record, ignore_index=True)

                copy_final_pred = pd.DataFrame(pred_dict)
                copy_final_pred.to_csv("./RecSys/out/MF/test/"+str(fold+1)+"_predictions.csv",index=False)
                all_predictions.append(copy_final_pred)
                rmse, mae = self.__evaluation__(final_pred)
                rmses.append(rmse)
                maes.append(mae)
                #precisions.append(precision)
                #recalls.append(recall)
            else:
                all_predictions.append(pd.read_csv(fname))
        fname = r'./RecSys/out/MF/test/metrics_results.csv'
        if not os.path.isfile(fname):
            self.__save_metrics__(rmses, maes, './RecSys/out/MF/test/')
        #print(len(all_predictions))
        self.__checking_predictions__(all_predictions)
        

    def __checking_predictions__(self, list_predictions):
        items = ['insulin', 'metformin','diag_1', 'diag_2', 'diag_3']
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
                #print(a_user, user_items)
                for item in items:
                    #print(user_items[item].values)
                    if (user_items[item].values >-1):
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
        df.to_csv("./RecSys/out/MF/test/au_predictions.csv",index=False)
        print("predictions were saved at: '/RecSys/out/MF/test/au_predictions.csv'")
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
        #rmse, mae,precision, recall = ers_object.calculate_errors()
        #return rmse, mae,precision, recall
        rmse, mae = ers_object.calculate_errors()
        return rmse, mae
     
    def get_final_rs_matrix(self):
        """
            Start testing
        """
        #fname = r'./RecSys/out/MF/test/au_predictions.csv'
        #if os.path.isfile(fname) :
        #    print
        #else:
        print('Making predictions of the encounter from the first part (test data)...')
        self.__test__()

    def __saving_models__(self, model,fold, algo=None):
        filename = ''
        if(algo is not None):
            filename = './RecSys/out/'+self.type_rs+'/models/'+algo+'/model_'+str(fold+1)+'.sav'
        else:
            filename = './RecSys/out/'+self.type_rs+'/models/model_'+str(fold+1)+'.sav'
        joblib.dump(model, filename)
    def __save_validation__(self, final_pred,fold, algo=None):
        filename = ''
        if(algo is not None):
            filename = './RecSys/out/'+self.type_rs+'/predictions/'+algo+'/'+str(fold+1)+'_predictions.csv'
        else:
            filename = "./RecSys/out/"+self.type_rs+"/predictions/" + str(fold+1) + "_predictions.csv"
        final_pred.to_csv(filename, index=False)
        print("Predictions were saved at: '"+filename+"'")

    def __save_metrics__(self, rmses, maes, save_path):
        """
            Method for saving the errors RMSE and MAE
        """
        metrics_results = pd.DataFrame(columns=['RMSE', 'MAE'])
        rmse = np.mean(rmses)
        mae = np.mean(maes)
        record = pd.Series([rmse, mae], index=['RMSE', 'MAE'])
        metrics_results = metrics_results.append(record, ignore_index=True)
        save_path = save_path+'metrics_results.csv'
        metrics_results.to_csv(save_path, index=False)
        print("Metrics results can be found here: " + save_path)
    
    def isSparse(array,m, n) : 
        counter = 0
        # Count number of zeros 
        # in the matrix 
        for i in range(0,m) : 
            for j in range(0,n) : 
                if (array[i][j] == 0) : 
                    counter = counter + 1
        percentage = (counter/(m*n)) * 100
        return percentage  