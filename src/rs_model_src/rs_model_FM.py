from __future__ import print_function
import copy
import src.rs_model_src.ers as ers
import src.rs_model_src.feature_matrix_cl as feature_matrix_cl
import os
import scipy
import sys
import warnings

import numpy as np
import pandas as pd

from pyfm import pylibfm
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
warnings.filterwarnings('ignore')

class FM_RS():
    def __init__(self, data, active_users):
        self.data = data
        self.active_users = active_users
        self.n_folds=3
        self.__prepare_feature_matrix__()
        self.__preset__()
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
        object_um = feature_matrix_cl.FeatureMatrixClass(self.data, self.active_users)
        self.feature_matrix, self.pivot = object_um.get_feature_matrix()
        self.active_users.to_csv('./RecSys/out/test/au_list.csv',index=False)

    def __preset__(self):
        matrix= pd.DataFrame(columns=self.feature_matrix.columns)
        for i,a_user in enumerate(self.active_users.encounter_id):
            if not (self.feature_matrix.loc[self.feature_matrix['encounter_id']== a_user].empty):
                data = self.feature_matrix.loc[self.feature_matrix['encounter_id']== a_user]
                matrix = matrix.append(data)#.copy)
            self.feature_matrix = self.feature_matrix.drop(self.feature_matrix[self.feature_matrix.encounter_id == a_user].index)
        self.matrix_test = matrix

    def __split_data__(self):
        #items = ['metformin', 'insulin']
        item_id = pd.get_dummies(self.feature_matrix['item_id'],prefix=['item_id'],dtype=float)
        y = self.feature_matrix['ratings'].to_frame()
        dummy = self.feature_matrix.drop(['ratings', 'encounter_id'], axis=1)
        df = pd.concat([item_id, dummy],axis=1, join='inner',  sort=False)
         
        X = df #scipy.sparse.csr_matrix(df)
        return train_test_split( X, y, test_size=0.25, random_state=42)
    
    def areEqual(self,arr1, arr2, n, m): 
        # If lengths of array are not  
        # equal means array are not equal 
        if (n != m): 
            return False 
    
        # Sort both arrays 
        arr1.sort(); 
        arr2.sort(); 
    
        # Linearly compare elements 
        for i in range(0, n - 1): 
            if (arr1[i] != arr2[i]): 
                return False 
    
        # If all elements were same. 
        return True

    def __train__(self):
        rmses_all = []
        maes_all = []
        models_all = []
        params_all=[]
        for i, fold in enumerate(np.arange(self.n_folds)):
            rmses = []
            models = []
            maes = []
            print('Factorization Machine')
            num_factors=[20,30,50,70,80]#np.arange(79,84)
            #num_iter=[20, 30, 40, 50, 60]
            initial_learning_rate=[0.001, 0.01, 0.005, 0.009, 0.003]
            params=[]
            print("\rFOLD Loop {}/{}.".format(fold+1, self.n_folds), end='\r')
            print('-----------------------------------------------')
            sys.stdout.flush()
            #training_data =training_utility[initial:final]
            print('Cross validation in progress...')
            self.X_train, self.X_val, self.y_train, self.y_val = self.__split_data__()
            self.indices_X_val = np.asarray(self.X_val.index.values.tolist())
            self.indices_y_val = np.asarray(self.y_val.index.values.tolist())
            #print(self.X_train.indices.tolist())
            #print(len(self.X_train.indices.tolist()))
            n = len(self.indices_X_val)
            m=len(self.indices_y_val)
            if (self.areEqual(self.indices_X_val, self.indices_y_val, n, m)): 
                print("Yes") 
            else: 
                print("No") 
            dummy = self.X_train 
            #print(dummy)
            #print(type(dummy.values))
            self.X_train = scipy.sparse.csr_matrix((dummy) ,dtype=float)
            dummy = self.X_val
            self.X_val = scipy.sparse.csr_matrix((dummy),dtype=float)
            dummy = self.y_train
            self.y_train = dummy.values.ravel()
            dummy = self.y_val
            self.y_val = dummy.values.ravel()
            #print(self.X_train)
            for i in np.arange(0,1):
                name = 'data_fold_'+str(fold+1) +'_loop_'+str(i)
                print('Starts training ------------------------------------------------------------')
                #i = np.random.randint(0,5)
                #j = np.random.randint(0,5)
                #k = np.random.randint(0,5)
                # Train a Factorization Machine
                #param ={'num_factors':num_factors[i], 'num_iter':num_iter[j], 'initial_learning_rate':initial_learning_rate[k]}
                param ={'num_factors':num_factors[i], 'initial_learning_rate':initial_learning_rate[i]}
                params.append(param)
                fm = pylibfm.FM(num_factors=num_factors[i], num_iter=50, verbose=True, task="regression", 
                                initial_learning_rate=initial_learning_rate[i], learning_rate_schedule="constant")
                fm.fit(self.X_train,self.y_train) #[0:2000]
                self.__saving_models__(fm,fold,i)

                #Validate a FM
                df_val, rmse, mae =self.__validation__(fm)
                self.__saving_pred_training__(df_val,fold,i)
                rmses.append(rmse)
                models.append(fm)
                maes.append(mae)
            #save_path = save_path+'metrics_results.csv'
            #self.__save_metrics__(rmses, maes, './RecSys/out/FM/train/', fold)
            rmses_all.append(rmses)
            maes_all.append(maes)
            models_all.append(models)
            params_all.append(params)
            self.__save_metrics__(rmses_all, maes_all, './RecSys/out/FM/train/', fold)
        self.best_fm = self.__best_model__(rmses_all, models_all,params_all)

    def __validation__(self, fm):        
        preds = fm.predict(self.X_val)
        rmse, mae = self.__evaluation__(preds, self.y_val)
        preds_set = pd.DataFrame({'prediction': preds})
        indices_set = pd.DataFrame({'index': self.indices_X_val})
        df_val = pd.concat([indices_set, preds_set],axis=1, join='inner',  sort=False)
        return df_val, rmse, mae
        
    def __evaluation__(self, preds, y_truth):
        """
            Create an evaluation object for calculating the root mean square error (RMSE) and mean absolute error (MAE)
        """
        mse = mean_squared_error(y_truth,preds)
        mae = np.mean(np.abs(y_truth - preds))
        print("FM MSE: %.4f" % mse)
        rmse = np.sqrt(mse)
        return rmse, mae

    def __best_model__(self, mses,models, params):
        best_model_list = []
        best_param_list = []
        less_rmse_list = []
        best_model = None
        for i,rms in enumerate(mses):
            index_max =  np.argmin(rms) #np.argmax(mses)
            best_model = models[i][index_max]
            best_param = params[i][index_max]
            print('The best model is: ',best_model)
            print('The best param is: ',best_param)
            best_model_list.append(best_model)
            best_param_list.append(best_param)
            less_rmse_list.append(rms)
        return best_model

    def __saving_models__(self, model,fold,i):
        filename = './RecSys/out/FM/models/model_'+str(fold+1)+'_loop_'+str(i)+'.sav'
        joblib.dump(model, filename)

    def __saving_pred_training__(self, df_val, fold,i):
        df_val.to_csv('./RecSys/out/FM/train/pred_'+str(fold+1)+'_loop_'+str(i)+'.csv')

    def __save_metrics__(self, rmses, maes, save_path, fold):
        """
            Method for saving the errors RMSE and MAE
        """
        metrics_results = pd.DataFrame(columns=['RMSE', 'MAE'])
        #rmse = np.mean(rmses)
        #mae = np.mean(maes)
        record = pd.Series([rmses, maes], index=['RMSE', 'MAE'])
        metrics_results = metrics_results.append(record, ignore_index=True)
        save_path = save_path+'metrics_results'+str(fold+1)+'.csv'
        metrics_results.to_csv(save_path, index=False)
        print("Metrics results can be found here: " + save_path)

    def get_predictions(self):
        #items = ['metformin', 'insulin'
        item_id = pd.get_dummies(self.matrix_test['item_id'],prefix=['item_id'],dtype=float)
        y = self.matrix_test['ratings'].to_frame()
        dummy = self.matrix_test.drop(['ratings', 'encounter_id'], axis=1)
        df = pd.concat([item_id, dummy],axis=1, join='inner',  sort=False)
         
        #indices = df.index.values.tolist()
        #print(indices)
        #X = scipy.sparse.csr_matrix((df, indices))
        X = scipy.sparse.csr_matrix(df, dtype = float)
        #print(X)
        preds = self.best_fm.predict(X)
        mse = mean_squared_error(y,preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y,preds)
        print("FM MSE: %.4f" % mse)
        #print(preds)
        preds_final = pd.concat([self.matrix_test.encounter_id, pd.DataFrame(preds)],axis=1, join='inner',  sort=False)
        #pd.DataFrame(preds).to_csv('./RecSys/out/FM/results/pred_au.csv')
        preds_final.to_csv('./RecSys/out/FM/results/pred_au.csv', index=False)
        metrics_results = pd.DataFrame(columns=['RMSE', 'MAE'])
        #rmse, mae = self.__evaluation__(preds, y)
        record = pd.Series([rmse, mae], index=['RMSE', 'MAE'])
        metrics_results = metrics_results.append(record, ignore_index=True)
        metrics_results.to_csv('./RecSys/out/FM/results/metrics_au.csv', index=False)
        return preds_final
        

        