import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import warnings
import os
import time

class FeatureMatrixClass():
    """
        This class is used for: 
            -creating the feature matrix (user-item-ratings matrix). 
        Return:
            -feature matrix which contains all unique encounters without considering the active users for the training of the recommender system
            -feature matrix pivot contains all unique encounters and the active users
    """
    def __init__(self, data, active_users, prefix="CARS"):
        """
            Args:
                data: all the data set
                active_users: encounters which were predicted as readmitted from the first part of the project
        """  
        self.original_data = data
        self.active = active_users
        self.prefix = prefix
        self.__readmitted_data_()
        self.__create_user_item_rating__()
        self.__binary_data__()
        
    
    def __create_user_item_rating__(self):
        """
            Create and return a matrix with columns encounter_id, item_id,rating and rating_difference. 
            It will cotain, for each user all the items that this user has rated and 
            the rating value.
            This matrix cotains also the active users.
            The rating_difference refers to the difference between the rated item and the mean of all the items that the user rated.
        """
        print('Creating for each user his/her rated items matrix ...')
        self.__utility__, self.__pivot__ = self.__feature_matrix__()
        self.__utility__.to_csv("./RecSys/feature_matrix/feature_matrix_"+self.prefix+".csv",index=False)
        self.__pivot__.to_csv("./RecSys/feature_matrix/feature_matrix_pivot_"+self.prefix+".csv",index=False)
        
        items = ['metformin', 'insulin']
        contexts = ['race', 'age','gender','admission_type_id', 'time_in_hospital', 'number_emergency',
        'number_inpatient', 'max_glu_serum', 'HBA', 'diag_1','diag_2'] 
        matrix = dict()
        for user, value in self.__utility__.groupby('encounter_id'):
            for i, item in enumerate(items):
                rating = value[item].values[0] #rating.append(value[item].values[0])
                #print(rating)
                if 'encounter_id' in matrix:
                    matrix['encounter_id'].append(user)
                    matrix['item_id'].append(i+1) #items
                    for context in contexts:
                        context_value = value[context].values[0]
                        matrix[context].append(context_value)    
                    matrix['rating'].append(rating)
                else:
                    matrix['encounter_id'] = [user]
                    matrix['item_id'] = [i+1]
                    for context in contexts:
                        context_value = value[context].values[0]
                        matrix[context] = [context_value]  
                    matrix['rating'] = [rating]
            #print(matrix)
        d = matrix
        length_dict = {key: len(value) for key, value in d.items()}
        #print(length_dict)
        matrix = pd.DataFrame(data=matrix)
        matrix.to_csv("./RecSys/feature_matrix/user_item_context_"+self.prefix+".csv",index=False)
        self.__user_item_rating__ =  matrix #d
        

    
    def __readmitted_data_(self):
        """
            From the original dataset, get the unique encounters which were readmitted (!=1).
        """
        self.r_data = self.original_data[self.original_data['readmitted'] != 1 ]
        self.r_data = self.r_data.sample(frac=1).reset_index(drop=True)


    def __feature_matrix__(self):
        """
            Prepare and return both feature_matrix and feature_pivot for the recommeder system
        """
        print('Preparing feature matrix pivot...')
        #start = time.process_time()
        items = ['encounter_id','metformin', 'insulin','race', 'age','gender','admission_type_id', 'time_in_hospital', 'number_emergency',
        'number_inpatient', 'max_glu_serum', 'HBA', 'diag_1','diag_2'] 
        #'gender', 'diag_3', 'discharge_disposition_id', 'admission_source_id', 'num_lab_procedures', 'num_procedures','num_medications', 'number_outpatient',
        #'change', 'diabetesMed'
        data = dict()
        for item in items:
           data[item] = self.r_data[item].values
        self.data = pd.DataFrame(data) #columns=['encounter_id', 'insulin', 'metformin', 'HBA', 'max_glu_serum', 'diag_1']
        feature_pivot = self.__feature_pivot__(items)
        #print('Second option: '+(time.process_time() - start))
        #print(len(np.unique(feature_pivot['encounter_id'])))
        print('Preparing feature matrix...')
        feature = self.__remove_active_users_labels__(feature_pivot)
        #self.__binary_data__(feature)
        return feature, feature_pivot
    
    def __feature_pivot__(self,items):
        """
            Based on the items, prepare the feature matrix pivot with the encounters information.
        """
        print('SIze of readimitted original ', len(self.r_data['encounter_id']))
        print('Before adding active users', len(self.data['encounter_id']))
        print('Len of active users ',len(self.active['encounter_id']))
        count = 0
        for au in self.active['encounter_id']:
            if not(au in self.data['encounter_id'].values):
                count = count + 1
                found_data = self.original_data.loc[self.original_data['encounter_id']==au]
                if not found_data.empty:
                    self.data = self.data.append(pd.Series([au, float(found_data.metformin.values), float(found_data.insulin.values),float(found_data.race.values),float(found_data.age.values), float(found_data.gender.values), float(found_data.admission_type_id.values), 
                    float(found_data.time_in_hospital.values), float(found_data.number_emergency.values), float(found_data.number_inpatient.values), 
                    float(found_data.max_glu_serum.values), float(found_data.HBA.values),  
                    float(found_data.diag_1.values), float(found_data.diag_2.values)], index=items ), ignore_index=True)
                    #print(self.data.loc[self.data['encounter_id']==au])
                else:
                    self.data = self.data.append(pd.Series([au,0,0,1,5,2,0,0,0,0,2,2,1,0], index=items ), ignore_index=True)
        print('times of adding new data ', count)
        print('After adding active users ', len(self.data['encounter_id']))
        print('Len of unique active users ',len(np.unique(self.data['encounter_id'])))
        return self.data
  

    def __remove_active_users_labels__(self, data):
        """
            From data (feature_pivot) remove the values of the active users. They should not be considered as part of the training process.
            Return:
                data (feature_matrix)
        """
        for au in self.active['encounter_id']:
            data.loc[data['encounter_id']==au, ['insulin','metformin']] = 0 #'change', 'diabetesMed', 'HBA', 'max_glu_serum', 'diag_1','diag_2', 'diag_3'
        return data
    
    def __binary_data__(self):
        """
            One-Hot Encoding
        """
        encounter_id = self.__user_item_rating__['encounter_id']#pd.get_dummies(self.__user_item_rating__['encounter_id'],prefix=['encounter_id'])#,dtype=float)
        #'metformin', 'insulin',
        item_id = self.__user_item_rating__['item_id'] #pd.get_dummies(self.__user_item_rating__['item_id'],prefix=['item_id'],dtype=float)
        race = pd.get_dummies(self.__user_item_rating__['race'],prefix=['race'],dtype=float)
        age = pd.get_dummies(self.__user_item_rating__['age'],prefix=['age'],dtype=float)
        gender = pd.get_dummies(self.__user_item_rating__['gender'],prefix=['gender'],dtype=float)
        admission_type_id = pd.get_dummies(self.__user_item_rating__['admission_type_id'],prefix=['admission_type_id'],dtype=float)
        time_in_hospital = pd.get_dummies(self.__user_item_rating__['time_in_hospital'],prefix=['time_in_hospital'],dtype=float)
        number_emergency = pd.get_dummies(self.__user_item_rating__['number_emergency'],prefix=['number_emergency'],dtype=float)
        number_inpatient = pd.get_dummies(self.__user_item_rating__['number_inpatient'],prefix=['number_inpatient'],dtype=float)
        max_glu_serum = pd.get_dummies(self.__user_item_rating__['max_glu_serum'],prefix=['max_glu_serum'],dtype=float)
        HBA = pd.get_dummies(self.__user_item_rating__['HBA'],prefix=['HBA'],dtype=float)
        diag_1 = pd.get_dummies(self.__user_item_rating__['diag_1'],prefix=['diag_1'],dtype=float)
        diag_2 = pd.get_dummies(self.__user_item_rating__['diag_2'],prefix=['diag_2'],dtype=float)
        rating =pd.DataFrame( self.__user_item_rating__['rating'].values, columns=['ratings'],dtype=float)

        df = pd.concat([encounter_id, item_id,race, age,gender,admission_type_id, time_in_hospital, number_emergency,
        number_inpatient, max_glu_serum, HBA, diag_1,diag_2,rating],axis=1, join='inner',  sort=False)
        save_path = "./RecSys/feature_matrix/feature_matrix_one_hot_"+self.prefix+".csv"
        df.to_csv(save_path, index=False)
        print("The feature matrix can be found here: " + save_path)
        self.final_data = df

    def get_feature_matrix(self):
        return self.final_data, self.__user_item_rating__
        