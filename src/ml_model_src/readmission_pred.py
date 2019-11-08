import pandas as pd
import numpy as np
import os

import split_data
import pred_algo
import plot_data

from sklearn.model_selection import train_test_split, KFold

class MLClass():
    """
        Class for solving the machine learning problem, which consist of make prediction if the encounter or patient will be
        No readmitted, Readmitted with < 30 days of staying or Readmitted with > 30 days of staying
    """
    def __init__(self, data):
        self.data = data
        self.__split__()
        self.__train__()

    def __split__(self):
        """
            Prepare the data into data set (X) and label set(y). Then, these sets can be splitted into train set and test set.
            Train set is used for training and this set will be separated into train set and validation set.
            Test set is used for getting the predictions of the encounter who will be used as active users for the second part of the project.
        """
        prepare = split_data.PrepareData(self.data)
        X, y = prepare.separate_data()
        self.__plot_original_data__(X,y)
        self.n_folds = 5
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25) # stratify=self.y
    
    def __plot_original_data__(self, X,y):
        """
            Visualization of the data and the imbalanced classes.
        """ 
        imbalance_cl = plot_data.PlotDataClasses(X,y)
        imbalance_cl.num_lab_medications()
        imbalance_cl.medication_comparison()
        imbalance_cl.change_med_and_readmission()
        imbalance_cl.readmitted()
        #exit()

    def __train__(self):
        """
            Train the model
        """
        pred_algo.PredAlgClass(self.X_train, self.y_train,self.n_folds)
        

    def prediction_readmitted(self):
        """
            Predict
            Return:
                The encounters who will be readmitted with < 30 days of staying or with > 30 days of staying
                ">30":2,"<30":3, "NO":1 
        """
        model = pred_algo.PredAlgClass(self.X_test, self.y_test,self.n_folds, flag='predictions')
        predictions = model.get_predictions()

        dict_new = {
            'encounter_id': self.X_test['encounter_id'].values,
            'readmitted': predictions
            }
        df = pd.DataFrame(data=dict_new)
        df.to_csv("./MaLer/Predictions/out/patients_original_readmission.csv")
        encounters_readmitted = df[df['readmitted'] !=1]
        print('Saving encounter for the recommender system: "./MaLer/Predictions/out/patients_readmitted.csv"')
        encounters_readmitted.to_csv("./MaLer/Predictions/out/patients_readmitted.csv")

        return encounters_readmitted