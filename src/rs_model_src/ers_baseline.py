import numpy as np
import pandas as pd
import os
import sys
from sklearn.metrics import recall_score, precision_score 

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
        print(self.y_predicted.values)
        rmse = np.sqrt(np.mean(0.5 * (self.y - self.y_predicted)**2))
        mae = np.mean(np.abs(self.y - self.y_predicted))
        return rmse, mae
        
    def __precision__recall__(self, M):
        """
            Precision@M: is the proportion of recommended items in the top-M list that are relevant
            (1/len(u))(number of recommended items @M that are relevant to u / M)
            recall@M: is the proportion of relevant items found in the top-M recommendations
            (1/len(u))(number of recommended items @M that are relevant to u /total number of relevant items to u)
        """
        len_u = len(self.data['user'])
        precision_m_arr = []
        recall_m_arr = []
        mrr_arr = []
        relevant_items = ['insulin', 'metformin', 'diag_1']
        #user, item, prediction, real
        for user, items in self.data.groupby('user'):
            total_relevant_i_to_u = 0
            for relevant in relevant_items:
                if(relevant in items):
                    total_relevant_i_to_u = total_relevant_i_to_u + 1

            sorted_m_predictions = items.sort_values('prediction',ascending=False)[:M]

            # for Precision@M and recall@M
            recommended_items_m_relevant_to_u = 0

            for item, value in sorted_m_predictions.groupby('item'):
                # for Precision@M and recall@M
                relevance = 0
                if value.prediction.values != 0:
                    relevance = 1
                recommended_items_m_relevant_to_u += relevance

            precision_m_to_one_u = recommended_items_m_relevant_to_u / float(M)  
            precision_m_arr.append(precision_m_to_one_u)

            # recall@M
            if total_relevant_i_to_u > 0:
                recall_m_to_one_u = recommended_items_m_relevant_to_u / float(total_relevant_i_to_u)
            else:
                recall_m_to_one_u = 0
            recall_m_arr.append(recall_m_to_one_u)

        precision_m = np.mean(np.array(precision_m_arr)) 
        recall_m = np.mean(np.array(recall_m_arr)) 
        
        return precision_m, recall_m

    def calculate_errors(self):
        """
            Calculate all metrics according to the information in the data.
        """
        self.y_predicted = self.data['prediction']
        self.y = self.data['real']
        rmse, mae = self.__rmse_mae__()
        precision, recall = self.__precision__recall__(10)
        return rmse, mae,precision, recall