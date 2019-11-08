import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Data():
    """
    Class for preparing the Data set.
    """
    def __init__(self, data):
        """
            Args:
                data(dataframe): all the data set of the patients.
        """  
        self.data = data 
    
    def get_clean_data(self):
        """This module should be called at the end.
           It will return the final cleaned data.
        """
        return self.data

    def mapping_values(self):
        """
            Map/replace the original values of the data
        """
        self.data.readmitted = self.data.readmitted.map({">30":2,"<30":3, "NO":1})
        self.data.diabetesMed = self.data.diabetesMed.map({"No":2, "Yes":1})
        self.data.change = self.data.change.map({"No":2, "Ch":1})
        self.data.insulin = self.data.insulin.map({"No":0, "Steady":1, "Down":2, "Up":3})
        self.data.metformin = self.data.metformin.map({"No":0, "Steady":1, "Down":2, "Up":3})
        self.data.HBA = self.data.HBA.map({"None":0, "Norm":1, ">7":2, ">8":3})
        self.data.max_glu_serum = self.data.max_glu_serum.map({"None":0, "Norm":1, ">200":2, ">300":3})
        self.data.age = self.data.age.map({"[0-10)":1, "[10-20)":1, "[20-30)":2, "[30-40)":2, "[40-50)":3, "[50-60)":3, "[60-70)":4, "[70-80)":4, "[80-90)":5, "[90-100)":5})
        self.data.gender = self.data.gender.map({"Male":1, "Female":2, "Unknown/Invalid":0})
        self.data.race = self.data.race.map({"-1":0, "AfricanAmerican":1, "Asian":2, "Caucasian":3, "Hispanic":4, "Other":0})
        
    def modify_diagnosis(self, diag, number):
        """
            Classification information based on the file: ./data/IDs_mapping.csv
        """
        # create a duplicate of the diagnosis column
        name = 'grouped_diag'+str(number)
        self.data[name] = self.data[diag]
        #{'Other(Injury,Musculoskeletal, genitourinary, neoplasma,others):'5, 'Circulatory':1, 'Respiratory':2, 'Digestive':3, 'Diabetes':4}
        # disease codes starting with V or E are in “other” category
        self.data.loc[self.data[diag].str.contains('V'), [name]] = 5
        self.data.loc[self.data[diag].str.contains('E'), [name]] = 5
        # also replace the unknown values with -1
        self.data[name] = self.data[name].replace('-1', 0)
        # iterate and recode disease codes between certain ranges to certain categories
        for index, row in self.data.iterrows():
            #Circulatory
            if (float(row[name]) >= 390 and float(row[name]) < 460) or (np.floor(float(row[name])) == 785):
                self.data.loc[index, name] = 1
            #Respiratory
            elif (float(row[name]) >= 460 and float(row[name]) < 520) or (np.floor(float(row[name])) == 786):
                self.data.loc[index, name] = 2
            #Digestive
            elif (float(row[name]) >= 520 and float(row[name]) < 580) or (np.floor(float(row[name])) == 787):
                self.data.loc[index, name] = 3
            #Diabetes
            elif (np.floor(float(row[name])) == 250):
                self.data.loc[index, name] = 4
            #Others(Injury,Musculoskeletal, genitourinary, neoplasma,others)
            else:
                self.data.loc[index, name] = 5
        # convert this variable to float type to enable computations later
        self.data[name] = self.data[name].astype(int)
        self.data = self.data.drop([diag], axis=1)
        self.data[diag] = self.data[name]
        self.data = self.data.drop([name], axis=1)
        
    def modify_admission_type(self):
        """
            Classification information based on the file: ./data/IDs_mapping.csv

        """
        name = 'grouped_admission'
        self.data[name] = self.data['admission_type_id']
        #{'Emergency':1, 'Urgent':2, 'Elective':3 and 'Newborn':4, Trauma:7,  'Not available othre null':0,9,5,6,8}
        # also replace the unknown values and Reserved: 6-8, with 0 
        self.data[name] = self.data[name].replace('-1', 0)
        # iterate and recode disease codes between certain ranges to certain categories
        for index, row in self.data.iterrows():
            if (float(row[name]) >= 5 and float(row[name]) < 10 and float(row[name]) != 7):
                self.data.loc[index, name] = 0
            elif(float(row[name])==7):
                self.data.loc[index,name] = 5
        # convert this variable to float type to enable computations later
        self.data[name] = self.data[name].astype(int)
        self.data = self.data.drop(['admission_type_id'], axis=1)
        self.data['admission_type_id'] = self.data[name]
        self.data = self.data.drop([name], axis=1)
        


    def remove_bias(self):
        """
        Classification information based on the file: ./data/IDs_mapping.csv

        Discharge Disposition codes of 11,13, 14, 19, 20, and 21 were removed.
        11: Expired, 13: Hospice/home, 14:Hospice/medical facility, 19:Expired at home. Medicaid only, hospice, 
        20.expiored in a medical facility. Medicaid only, hospice, and 21: expired, place unknown.Medicaid only, hospice. 
        After the end of the loop, 2423 data will be removed in total
        """
        ids_discharge = [11, 13,14, 19, 20, 21]
        
        for value in ids_discharge:
            self.data = self.data[self.data.discharge_disposition_id != value]

    def clean_medication_list(self):
        """
        Delete the drugs which were not given to more than or equal to 90% of the patients.
        """
        #Medication List
        medication_list = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
                    'pioglitazone', 'rosiglitazone', 'acarbose','miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin',
                    'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
        for medication in medication_list:
            total_zero = self.data[medication].eq('No').sum()
            total_all = len(self.data[medication])
            total = np.round((total_zero/total_all),decimals=3)*100
            #print('In medication '+ medication+ ', Total of zeros is '+str(total)+ '%')
            if (total > 80):
                self.data = self.data.drop([medication], axis=1)
    
    def drop_replace(self):
        """
        Drop the missing data which are columns: weight and payer_code.
        Replace the '?' value, for all the columns of the database, with zero value
        """
        self.data = self.data.drop(["weight", "payer_code","medical_specialty"], axis=1)
        self.data= self.data.replace('?', '-1') 

    def save_data_csv(self):
        self.data.to_csv('./data/diabetes_data_preprocessed.csv', index=False)

def prepare_data():
    """This module does different task for loading and cleaning the data.
        -Load the Clinical Database Patient Record of Strack,B. et al. (2014).
        -drop_replace()
            Drop the missing data which are columns: weight and payer_code.
            Replace the '?' value, for all the columns of the database, with zero value
        -clean_medication_list()
            Verify which medication has more than 90% of "No" values. This means that 90% 
            or more of the patients have not recieved this medication. Therefore, the 
            medicatins were deleated.
        -remove_bias()
            Avoiding bias by including only active (alive) patients and not in hospice, 
            Then, discharge disposition codes of 11, 13, 14, 19, 20, and 21 were removed.
        -mapping_values()
            To work better with the values of each column for the training and testing steps,
            they were mapped.
        -return
            The new clean final dataset which will be used for  prediction and the recommender system
    """
    print('Loading the data...')
    loc = r'data/diabetic_data.csv'
    data = pd.read_csv(loc)
    print('Considering only the first encounter for each patient....')
    data = data.groupby('patient_nbr').head(1) #delete_duplicate_encounters

    new_data = Data(data)
    print('Deleting variables with missing values higher than 52%...')
    new_data.drop_replace()
    new_data.clean_medication_list()
    print('Removing encounters which resulted in discharge to a hospice or patient death...')
    new_data.remove_bias()
    print('Categorize the diagnosis in 5 groups: Other, Circulatory, Respiratory, Digestive, Diabetes...')
    for i, diag in enumerate(['diag_1', 'diag_2', 'diag_3']):
        new_data.modify_diagnosis(diag, i+1)
        print(diag +' is done!')
    new_data.modify_admission_type()
    new_data.mapping_values()

    print('Saving the preprocessed data on the same folder "data"...')
    new_data.save_data_csv()
    return new_data.get_clean_data()