import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import warnings
import os

class UtilityMatrixClass():
    """
        This class is used for: 
            -creating the utility matrix (user-item-ratings matrix). 
            -normalizing the rating for each item column in a range between 1 to 5; otherwise ratings of values of 9 can be observed i.e. in time_in_hospital
        Return:
            -utility matrix which contains all unique encounters without considering the active users for the training of the recommender system
            -utility matrix pivot contains all unique encounters and the active users
    """
    def __init__(self, data, active_users):
        """
            Args:
                data: all the data set
                active_users: encounters which were predicted as readmitted from the first part of the project
        """  
        self.original_data = data
        self.active = active_users
        self.__readmitted_data_()
    
    def __readmitted_data_(self):
        """
            From the original dataset, get the unique encounters which were readmitted (!=1).
            ">30":2,"<30":3, "NO":1
        """
        self.r_data = self.original_data[self.original_data['readmitted'] != 1 ]
        self.r_data = self.r_data.sample(frac=1).reset_index(drop=True)

    def utility_matrix(self):
        """
            Prepare and return both utility_matrix and utility_pivot for the recommeder system
        """
        print('Preparing utility matrix pivot...')
        items = ['encounter_id','race', 'age', 'admission_type_id', 'time_in_hospital', 'number_emergency',
        'number_inpatient', 'max_glu_serum', 'HBA', 'metformin', 'insulin', 'diag_1',
        'diag_2', 'diag_3'] 
        #'gender','discharge_disposition_id', 'admission_source_id', 'num_lab_procedures', 'num_procedures','num_medications', 'number_outpatient',
        #'change', 'diabetesMed'
        data = dict()
        for item in items:
           data[item] = self.r_data[item].values
        self.data = pd.DataFrame(data) #columns=['encounter_id', 'insulin', 'metformin', 'HBA', 'max_glu_serum', 'diag_1']
        utility_pivot = self.__utility_pivot__(items)
        print(len(np.unique(utility_pivot['encounter_id'])))
        print('Preparing utility matrix...')
        utility = self.__remove_active_users_labels__(utility_pivot)
        return utility, utility_pivot
    
    def __utility_pivot__(self,items):
        """
            Based on the items, prepare the utility matrix pivot with the encounters information.
        """
        print('Size of readimitted original ', len(self.r_data['encounter_id']))
        #print('Before adding active users', len(self.data['encounter_id']))
        print('Len of active users ',len(self.active['encounter_id']))
        #print('Adding active u')
        count = 0
        for au in self.active['encounter_id']:
            if not(au in self.data['encounter_id'].values):
                count = count + 1
                found_data = self.original_data.loc[self.original_data['encounter_id']==au]
                #print(found_data)
                self.data = self.data.append(pd.Series([au, float(found_data.race.values),float(found_data.age.values), float(found_data.admission_type_id.values), 
                float(found_data.time_in_hospital.values), float(found_data.number_emergency.values), float(found_data.number_inpatient.values), 
                float(found_data.max_glu_serum.values), float(found_data.HBA.values), float(found_data.metformin.values), float(found_data.insulin.values), 
                float(found_data.diag_1.values), float(found_data.diag_2.values), float(found_data.diag_3.values)], index=items ), ignore_index=True)
                #print(self.data.loc[self.data['encounter_id']==au])
                #exit()
        #print('times of adding new data ', count)
        #print('After adding active users ', len(self.data['encounter_id']))
        print('Len of total unique users for the utility matrix: ',len(np.unique(self.data['encounter_id'])))
        for group in ['time_in_hospital', 'number_inpatient']:
            self.__modify_groups__(group)
        self.__modify_emergency__()
        return self.data
  

    def __remove_active_users_labels__(self, data):
        """
            From data (utility_pivot) remove the values of the active users. They should not be considered as part of the training process.
            Return:
                data (utility_matrix)
        """
        for au in self.active['encounter_id']:
            data.loc[data['encounter_id']==au, ['insulin','metformin', 'diag_1','diag_2', 'diag_3']] = 0 #'change', 'diabetesMed', 'HBA', 'max_glu_serum'
        return data
    
    def __modify_groups__(self, var):
        """
            Modify the data of the groups for time_in_hospital and number_inpatient. 
            Then, the rating can be in values from 1 to 5. 0 if there is no information about it.
            Where: {1-3:1, 4-6:2, 7-9:3, 10-12:4, 13-14:5}
        """
        # create a duplicate of the diagnosis column
        name = 'grouped_diag'
        self.data[name] = self.data[var]
        self.data[name] = self.data[name].replace('-1', 0)
        # iterate and recode disease codes between certain ranges to certain categories
        for index, row in self.data.iterrows():
            if (float(row[name]) >= 1 and float(row[name]) < 4):
                self.data.loc[index, name] = 1
            elif (float(row[name]) >= 4 and float(row[name]) < 7):
                self.data.loc[index, name] = 2
            elif (float(row[name]) >= 7 and float(row[name]) < 10):
                self.data.loc[index, name] = 3
            elif (float(row[name]) >= 10 and float(row[name]) < 13):
                self.data.loc[index, name] = 4
            elif (float(row[name]) >= 13 and float(row[name]) < 15):
                self.data.loc[index, name] = 5
            else:
                self.data.loc[index, name] = 0
        # convert this variable to float type to enable computations later
        self.data[name] = self.data[name].astype(int)
        self.data = self.data.drop([var], axis=1)
        self.data[var] = self.data[name]
        self.data = self.data.drop([name], axis=1)
        print('Len unique after modify group ',len(np.unique(self.data['encounter_id'])))
    
    def __modify_emergency__(self):
        """
           Modify the data for number_emergency. Then, the rating can be in values from 1 to 5. 0 if there is no information about it. 
           Where {1-9:1, 10-18:2, 19-27:3, 28-36:4, 37-45:5}
        """
        # create a duplicate of the diagnosis column
        name = 'grouped_diag'
        self.data[name] = self.data['number_emergency']
        self.data[name] = self.data[name].replace('-1', 0)
        # iterate and recode disease codes between certain ranges to certain categories
        for index, row in self.data.iterrows():
            if (float(row[name]) >= 1 and float(row[name]) < 10):
                self.data.loc[index, name] = 1
            elif (float(row[name]) >= 10 and float(row[name]) < 19):
                self.data.loc[index, name] = 2
            elif (float(row[name]) >= 19 and float(row[name]) < 28):
                self.data.loc[index, name] = 3
            elif (float(row[name]) >= 28 and float(row[name]) < 37):
                self.data.loc[index, name] = 4
            elif (float(row[name]) >= 37 and float(row[name]) < 45):
                self.data.loc[index, name] = 5
            else:
                self.data.loc[index, name] = 0
        # convert this variable to float type to enable computations later
        self.data[name] = self.data[name].astype(int)
        self.data = self.data.drop(['number_emergency'], axis=1)
        self.data['number_emergency'] = self.data[name]
        self.data = self.data.drop([name], axis=1)