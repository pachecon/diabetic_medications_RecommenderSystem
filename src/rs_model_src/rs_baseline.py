from __future__ import print_function
import pandas as pd
import numpy as np
import os
import copy
import sys
import time

import warnings
warnings.filterwarnings('ignore')


class RecS_baseline_class():
    """
        Class for item-based collaborative filtering recommender system model.

    """
    def __init__(self,utility_matrix, utility_pivot,save_prefix="Test"):
        """
            Args:
            utility: encounter_id (user) vs items
            pivot: encounter_id including active users vs items
            save_prefix: used for save the results
            items_array: race, age, admission_type_id, max_glu_serum, HBA, metformin, insulin, diag_1,
                         diag_2, diag_3, time_in_hospital, number_inpatient, number_emergency
            users_array: all encounter_id's including the active users
        """
        self.__save_prefix__ = save_prefix
        self.__items_array__ = utility_matrix.columns.values.tolist()
        self.__items_array__.remove('encounter_id')
        self.__users_array__ = utility_matrix.encounter_id
        self.__utility__ = utility_matrix
        self.__pivot__ = utility_pivot
     
        self.__create_user_item_rating__() 
        self.__train__()

    def __create_user_item_rating__(self):
        """
            Create and return a matrix with columns encounter_id, item_id,rating and rating_difference. 
            It will cotain, for each user all the items that this user has rated and 
            the rating value.
            This matrix cotains also the active users.
            The rating_difference refers to the difference between the rated item and the mean of all the items that the user rated.
        """
        print('Creating for each user his/her rated items matrix ...')
        items = self.__items_array__
        matrix = dict()
        for user, value in self.__utility__.groupby('encounter_id'):
            for item in items:
                rating = value[item].values[0] #rating.append(value[item].values[0])
                if 'encounter_id' in matrix:
                    matrix['encounter_id'].append(user)
                    matrix['item_id'].append(item) #items
                    matrix['rating'].append(rating)
                else:
                    matrix['encounter_id'] = [user]
                    matrix['item_id'] = [item]
                    matrix['rating'] = [rating]

        for user, value in self.__pivot__.groupby('encounter_id'):
            rating = -1
            if (not(user in matrix['encounter_id'])):#user in matrix['encounter_id'].values)):
                for item in items:
                    if (item == 'metformin') or (item=='insulin') or (item=='change') or(item=='diabetesMed'):
                        rating = 0
                    else:
                        rating = value[item].values[0] #rating.append(value[item].values[0])
                    matrix['encounter_id'].append(user)
                    matrix['item_id'].append(item) #items
                    matrix['rating'].append(rating)
        matrix = pd.DataFrame(data=matrix)
        self.__user_item_rating__ =  matrix
        utility_copy = copy.deepcopy(self.__user_item_rating__)
        
        mean_user_rating = utility_copy.replace(0, np.nan).groupby(["encounter_id"], as_index=False,
                                                   sort=False).mean().rename(columns={'rating': 'mean_rating'})[['encounter_id','mean_rating']]
        self.__user_item_rating__= pd.merge(self.__user_item_rating__, mean_user_rating, on='encounter_id', how='left', sort=False)
        self.__user_item_rating__['rating_difference'] =  self.__user_item_rating__['rating'] - self.__user_item_rating__['mean_rating']


    def __train__(self):
        """ 
            Train the model by first computing the similarities 
            Save the similarity matrix between items and the runtime
        """
        start = time.time()
        matrix_columns = ['item1', 'item2', 'similarity']
        self.sim_matrix = pd.DataFrame(columns=matrix_columns)
        print('Start Similarity....')
        #item = 'age'
        #self.sim_matrix = self.sim_matrix.append(self.__compute_similarities__(item))
        for k, item in enumerate(self.__items_array__):
            self.sim_matrix = self.sim_matrix.append(self.__compute_similarities__(item))

        # save results
        self.sim_matrix.to_csv("./RecSys/out/baseline/sim/" + self.__save_prefix__ + "_similarities.csv")
        end = time.time()
        with open("./RecSys/out/baseline/runtime/"+self.__save_prefix__ +'_runtime.txt', 'w') as f:
            f.write("Time %d" % (end - start))


    def __compute_similarities__(self, item):
        """ 
            Method for calculating adjusted cosine similarity matrix
        """
        # prepare titles of columns and sim matrix
        matrix_columns = ['item1', 'item2', 'similarity']
        self.sim_matrix = pd.DataFrame(columns=matrix_columns)

        users_item = self.__user_item_rating__.loc[self.__user_item_rating__['item_id'] == item]
        users_who_rated_item = users_item.loc[users_item['rating']>0] #verify which users have rated the item 
        distinct_users = np.unique(users_who_rated_item['encounter_id']) #return unique users

        # save each item-item pair with its ratings values
        titles = ['encounter_id', 'item1', 'item2', 'rating1', 'rating2']
        record_pair = pd.DataFrame(columns=titles)

        # find all other items that the user has for this item
        print('-----------------------------------------------')
        print('-----------------------------------------------')
        print('Calculating similarity of item: ', item  )
        print('Finding other similar items')
        for i,user in enumerate(distinct_users):
            print("\rLoop {}/{}.".format(i+1, len(distinct_users)), end='\r')
            sys.stdout.flush()
            items_of_user = self.__user_item_rating__.loc[
                (self.__user_item_rating__["encounter_id"] == user) & (self.__user_item_rating__["item_id"] != item)]

            # how our item was rated:
            rating1 = self.__user_item_rating__.loc[
                (self.__user_item_rating__["item_id"] == item) & (self.__user_item_rating__["encounter_id"] == user)][
                "rating_difference"].values[0]

            # look at other items that this user has
            for other_item in items_of_user["item_id"]:
                # how this second item was rated:
                rating2 = self.__user_item_rating__.loc[
                    (self.__user_item_rating__["item_id"] == other_item) & (self.__user_item_rating__["encounter_id"] == user)][
                    "rating_difference"].values[0]

                # store everything
                record = pd.Series([user, item, other_item, rating1, rating2], index=titles)
                record_pair = record_pair.append(record, ignore_index=True)

        # a list of all other items
        unique_items2 = np.unique(record_pair['item2'])
        
        print('Calculate adjusted cosine with respect to ', item)
        for i,other in enumerate(unique_items2):
            print("\rLoop {}/{}.".format(i+1, len(unique_items2)), end='\r')
            sys.stdout.flush()
            # get info of the other item
            items_1_2 = record_pair.loc[record_pair['item2'] == other]

            # prepare the nominator as always
            sim_numerator = float((items_1_2['rating1'] * items_1_2['rating2']).sum())

            # for denominator we get all the ratings for items to avoid 1.0 similarities
            sim_denominator = float(
                np.sqrt(np.square(
                    self.__user_item_rating__.loc[self.__user_item_rating__["item_id"] == item]["rating_difference"].values).sum())
                *
                np.sqrt(np.square(self.__user_item_rating__.loc[self.__user_item_rating__["item_id"] == other][
                                      "rating_difference"].values).sum()))
            #sim_denominator = sim_denominator if sim_value_denominator != 0 else 1e-8
            
            # adjusted cosine similarity
            sim_value = sim_numerator / sim_denominator

            # append to sim matrix ['item1', 'item2', 'sim']
            self.sim_matrix = self.sim_matrix.append(pd.Series([item, other, sim_value], index=matrix_columns),
                                                     ignore_index=True)

        return self.sim_matrix
            

    def predict(self, user_id, item_id):
        """
        Args:

        Returns:

        """
        items_of_user = self.__pivot__.loc[(self.__pivot__["encounter_id"] == user_id)]
        #print('Items of the user: ' ,items_of_user)
        pred_nom = 0
        pred_denom = 0
        sim_matrix = self.sim_matrix.loc[(self.sim_matrix["item1"] == item_id)]
        if sim_matrix.shape[0] == 0:
            print('It was not possible to predict')
            return 0.0
        sorted_sim_matrix = sim_matrix.sort_values(by=["similarity"], ascending=False)
        # then extract the most similar ones
        new_matrix = sorted_sim_matrix.head(10)

        # compute prediction. don't use mean free ratings, but the original ones
        for l in new_matrix["item2"]:
            dummy = self.__utility__.loc[(self.__utility__["encounter_id"] == user_id)]
            rating = dummy[l].values

            if rating > 0.0:
                pred_nom += float(new_matrix.loc[new_matrix["item2"] == l]["similarity"].values[0]) * rating
                #print(pred_nom)
                pred_denom += float(np.abs(float(new_matrix.loc[new_matrix["item2"] == l]["similarity"].values[0])))
                #print(pred_denom)

        if pred_denom <= 0.0:
            pred = 0
        else :
            pred = pred_nom / pred_denom
        #print('Prediction: ', pred)
        return pred