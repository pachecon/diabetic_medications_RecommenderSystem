import numpy as np
import pandas as pd
import os
import sys
#from sklearn.metrics import recall_score, precision_score 

class Evaluation():
    def __init__(self, data): #poly_degree, models, training_errors, validation_errors, test_errors
        """
            Args:
        """  
        self.data = data

    def __rmse_mae__(self):
        """
            Calculate and return the root mean square error (RMSE) and mean absolute error (MAE) for validation splits.
        """
        #print(self.y_predicted.values)
        rmse = np.sqrt(np.mean(0.5 * (self.y - self.y_predicted)**2))
        mae = np.mean(np.abs(self.y - self.y_predicted))
        return rmse, mae
        
    def calculate_errors(self):
        """
            Calculate all metrics according to the information in the data.
        """
        self.y_predicted = self.data['prediction']
        self.y = self.data['real']
        rmse, mae = self.__rmse_mae__()
        #precision, recall = self.__precision__recall__(10)
        return rmse, mae #,precision, recall