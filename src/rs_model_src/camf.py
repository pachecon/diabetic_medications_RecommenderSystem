from __future__ import print_function
import sys
sys.path.insert(1, './src/rs_model_src/camf_src/') #files for the recommender system

import camf_ci
import camf_cu
import camf_c
import os
import pandas as pd
import plot_data_rs as prs
import numpy as np
import matplotlib.pyplot as plt



class CAMF_class():
    """Class for Context-Aware Matrix Factorization.
    """
    def __init__(self, type_camf, utility_data, original_utility,fold, learning_rate, num_factors,loop, save_prefix="Train"):
        """
            Args:
                type_camf (str): name of the camf type
                utility_data(dataframe): information from the feature matrix
                original_utility(dataframe): feature matrix pivot
                learning_rate(double): value of the learning rate
                num_factors(double): value of the number of latent factors
                loop(int): iteration number
                save_prefix (str): (optional) prefix for saving files during the training
        """
        self.type_camf = type_camf
        self.__save_prefix__ = save_prefix
        self.__items_array__ = np.array([1,2])#'metformin', 'insulin'])
        self.__users_array__ = utility_data.encounter_id.values
        self.__context_array__ = utility_data.drop(columns=['encounter_id', 'ratings'], axis=1)
        self.__ratings__ = utility_data['ratings'].values #utility_data
        self.utility_predictions = utility_data
        self.fold = fold
        self.lr = learning_rate
        self.factors = num_factors
        self.loop = loop
        self.__train__()

    def __check_ratings__(self, prediction):
        maximum = 3
        minimum = 0
        if prediction < minimum:
            prediction = minimum
        elif prediction > maximum:
            prediction = maximum
        return prediction


    def __train__(self):
        """ method for calculating whole utility matrix using U-V matrix factorization and biased SGD

        """
        if (self.type_camf == 'CAMF_CI'):
            #users, items, context, ratings
            ci = camf_ci.CI_class(self.__users_array__, self.__items_array__, self.__context_array__, self.__ratings__, self.fold, self.lr, self.factors)
            predictions, losses = ci.fit()
        elif (self.type_camf == 'CAMF_CU'):
            cu = camf_cu.CU_class(self.__users_array__, self.__items_array__, self.__context_array__, self.__ratings__, self.fold, self.lr, self.factors)
            predictions, losses = cu.fit()
        elif (self.type_camf == 'CAMF_C'):
            c = camf_c.C_class(self.__users_array__, self.__items_array__, self.__context_array__, self.__ratings__, self.fold, self.lr, self.factors)
            predictions, losses = c.fit()

        dummy_pred = np.zeros((predictions.shape))
        for r, pred_array in enumerate(predictions):
            for c, pred in enumerate(pred_array):
                dummy_pred[r][c] = self.__check_ratings__(pred)
        predictions = dummy_pred
        #save a plot with a loss function
        plots = prs.PlotRSData()
        #print(losses)
        plots.plot_loss_cars(losses, self.type_camf, self.__save_prefix__+"_loop"+str(self.loop))
        pd.DataFrame(losses).to_csv("./RecSys/out/CAMF/train/"+self.type_camf+"/" + self.__save_prefix__ +"losses_loop"+str(self.loop)+".csv")
        print('Saving the feature matrix...')
        # set predictions back to the pivot table
        self.__utility_saved_training__(predictions)    
        # save results
        self.utility_predictions.to_csv("./RecSys/out/CAMF/train/"+self.type_camf+"/" + self.__save_prefix__ + "_SGD_predictions_loop"+str(self.loop)+".csv")

    def __utility_saved_training__(self, predictions_training):
        for u, user in enumerate(self.__users_array__):
            print("\rLoop {}/{}.".format(u, len(self.__users_array__)), end='\r')
            sys.stdout.flush()
            for i, item in enumerate(self.__items_array__):
                if (len(self.utility_predictions.ratings.loc[(self.utility_predictions['encounter_id']==user) & (self.utility_predictions['item_id']==item)].values) != 0):
                    if ((self.utility_predictions.ratings.loc[(self.utility_predictions['encounter_id']==user) & (self.utility_predictions['item_id']==item)].values[0] == 0)):
                        self.utility_predictions.ratings.loc[(self.utility_predictions['encounter_id']==user) & (self.utility_predictions['item_id']==item)] = predictions_training[u][i]
                    else:
                        #for training items set "-1" in order to easily ignore them 
                        self.utility_predictions.ratings.loc[(self.utility_predictions['encounter_id']==user) & (self.utility_predictions['item_id']==item)] = -1

    def predict(self, user,item):
        try:
            return self.utility_predictions.loc[(self.utility_predictions['encounter_id']==user) & (self.utility_predictions['item_id']==item)]
        except:
            print("Advanced recommender system can't predict value for this pair user: " + str(user) + "; item: "
                  + str(item))
            return -1