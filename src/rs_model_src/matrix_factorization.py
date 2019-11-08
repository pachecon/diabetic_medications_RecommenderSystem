from __future__ import print_function
import pandas as pd
import numpy as np
import os
import copy
import sys
import time

#import utilities_data
import matplotlib.pyplot as plt
import plot_data_rs as prs


class MF_class():
    """
        Class for Matrix Factorization for Recommender System with U-V decomposition.
    """
    def __init__(self,utility_data, original_utility, fold, save_prefix="Test"):
        """
            Args:
                utility_predictions: like a pivot which contains encounter_id (including active users) vs items
                save_prefix: used for save the results
                items_array: race, age, admission_type_id, max_glu_serum, HBA, metformin, insulin, diag_1,
                            diag_2, diag_3, time_in_hospital, number_inpatient, number_emergency
                users_array: all encounter_id's including the active users
                ratings: get the values of the rating from the utility matrix
                fold: number of iterations
        """
        self.utility_predictions = original_utility
        self.__save_prefix__ = save_prefix
        self.__items_array__ = utility_data.columns.values.tolist()
        self.__items_array__.remove('encounter_id')
        self.__users_array__ = utility_data.encounter_id
        self.__ratings__ = utility_data[self.__items_array__].values
        self.__fold__ = fold
        self.__train__()

     
    def __check_ratings__(self, prediction):
        """
            For avoiding ratings in a different range of 1 to 5; where 0 represents missing value
            Return:
                prediction in a range from 0 to 5. 
        """
        maximum = 5.0
        minimum = 0.0
        if prediction < minimum:
            prediction = minimum
        elif prediction > maximum:
            prediction = maximum
        return prediction

    def __utility_saved_training__(self, predictions_training):
        """
            For saving the predictions of the training into the pivot utility_predictions and stablish which items and user were used
            for training with -1 value.
        """
        for u, user in enumerate(self.__users_array__):
            for i, item in enumerate(self.__items_array__): 
                #print(float(self.utility_predictions.loc[self.utility_predictions['encounter_id']==user, item].values))
                if (float(self.utility_predictions.loc[self.utility_predictions['encounter_id']==user, item].values) == float(0)):
                    #print(user,item)
                    #print(predictions_training[u][i])
                    self.utility_predictions.loc[self.utility_predictions['encounter_id']==user, item] = predictions_training[u][i]
                    #if(item == 'insulin'):
                    #    exit()
                else:
                    # for training items set "-1" in order to easily ignore them 
                    self.utility_predictions.loc[self.utility_predictions['encounter_id']==user, item] = -1
        #print(self.utility_predictions['insulin'])

    def __train__(self):
        """ 
            Method for calculating whole utility matrix using U-V matrix factorization and biased SGD
                   U: user, I:items, bias_u: user's bias, bias_i: item's bias
        """
        start = time.time()
        alpha, lr, epochs, U, I, bias_u, bias_i, global_bias  = self.__prepare_data__()
        self.losses = []
        #Calculate tensor factorization and predictions
        for e in range(epochs):
            loss = 0
            print("\repoch {}/{}.".format(e, epochs), end='\r')
            sys.stdout.flush()
            for i, user in enumerate(self.__users_array__):
                for j, item in enumerate(self.__items_array__):
                    
                    if self.__ratings__[i, j] != 0:
                        prediction = global_bias + bias_u[i] +bias_i[j] + np.dot(U[i], I[j].T)#U[i].dot(I[j].T)
                        #-----------Calculate the error between real and predicted rating------------------
                        error = (self.__ratings__[i, j] - prediction)
                        squared_error = error ** 2
                        loss += squared_error

                        #------------UPDATED BASED ON SGD----------
                        bias_u[i] = alpha * (error - lr * bias_u[i])
                        bias_i[j] = alpha * (error - lr * bias_i[j])
                        I_temp = I[j]
                        U_temp = U[i]
                        temp_u = U_temp + alpha * (2 * error * I_temp - lr * U_temp)
                        U[i] = temp_u
                        temp_i = I_temp + alpha * (2 * error * U_temp - lr * I_temp)
                        I[j] = temp_i
            self.losses.append(loss)
            if loss < 0.001:
                break
            
        end = time.time()
        with open("./RecSys/out/MF/Train/"+self.__save_prefix__ +'_runtime.txt', 'w') as f:
            f.write("Time %d" % (end - start))
        predictions = U.dot(I.T)
        #dummy_pred = np.zeros((predictions.shape))
        #for r, pred_array in enumerate(predictions):
        #    for c, pred in enumerate(pred_array):
        #        dummy_pred[r][c] = self.__check_ratings__(pred)
        #predictions = dummy_pred
        #U=0
        #I=0
        # set predictions back to the pivot table
        print('Saving utility matrix.........................')
        self.__utility_saved_training__(predictions)    
    
        # save a plot with a loss function
        print('Saving loss plot on "./RecSys/out/MF/Plots/"....')
        p_losses = prs.PlotRSData()
        p_losses.plot_loss_mf(self.losses, self.__save_prefix__)

        print('Saving results of the training on "./RecSys/out/MF/Train/"....')
        # save predictions and the losses
        self.utility_predictions.to_csv("./RecSys/out/MF/Train/" + self.__save_prefix__ + "_SGD_predictions.csv", index=False)
        df = pd.DataFrame(self.losses)
        df.to_csv("./RecSys/out/MF/Train/" + self.__save_prefix__ + "_SGD_losses.csv", index=False)
        #clean the space
        self.__clean_space__()    
    
    def get_losses(self):
        """
            Return the loss between the real value and the predicted one. This will be used for the validation section.
        """
        return self.losses    

    def predict(self, user,item):
        """
            Return the predicted value based on the user and the item 
        """
        try:
            prediction = self.utility_predictions.loc[self.utility_predictions['encounter_id']==user, item]
            return prediction
        except:
            print("MF recommender system can't predict value for this pair user: " + str(user) + "; item: "
                  + str(item))
            return 0
    def __clean_space__(self):
        """
            Clean the space
        """
        self.__items_array__ = None
        self.__users_array__ = None
        self.__ratings__ = None

    def __prepare_data__(self):
        """ 
            Method for initializing required variables and setting hyperparameters for the UV decomposition and the
            bias Stochastic Gradient Descent (SGD)
        """
        # hyperparameters
        alpha = 0.001 #[0.005, 0.002, 0.001] #0.001 # 0,05  #0.03 #0.001
        learning_rate = 0.009
        rank = 16
        epoch = 200 #[700, 800, 900]
    
        # variable... srandomly initialize user/item factors from a Gaussian
        U = np.random.random((len(self.__users_array__),rank)) #U = np.random.normal(0,.1,(train.n_users,self.num_factors))
        I = np.random.random((len(self.__items_array__),rank)) #I = np.random.normal(0,.1,(train.n_items,self.num_factors))
        bias_user = np.zeros((len(self.__users_array__), 1)) #Bias of user
        bias_item = np.zeros((len(self.__items_array__), 1)) #Bias of items
        global_bias = self.__ratings__.mean()
        return alpha, learning_rate, epoch, U, I, bias_user, bias_item, global_bias 