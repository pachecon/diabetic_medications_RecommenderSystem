
class PrepareData():
    """ 
       Class for split the data in X features and y labels.
       The y labels can be No readmitted, Readmitted with < 30 days of staying or Readmitted with > 30 days of staying
    """
    def __init__(self, data):
        """
            Args:
                data(dataframe): all the data set of the patients.
        """  
        self.data = data
        
    def separate_data(self):
        """
            Separate the data and its labels
            Return:
                X: all data with all features except readmission classes
                y: labels 
        """
        self.data = self.data.drop(columns=['gender', 'discharge_disposition_id', 'admission_source_id'])
        #print(self.data.keys())
        X = self.data.drop("readmitted", axis = 1)
        y = self.data.drop(columns=['race', 'age', 'admission_type_id','time_in_hospital','number_outpatient', 
        'num_lab_procedures', 'num_procedures', 'num_medications', 'diag_1', 'diag_2', 'number_diagnoses', 
        'insulin','metformin', 'change', 'diabetesMed','max_glu_serum', 'HBA','patient_nbr', 'number_emergency', 'number_inpatient',
        'diag_3'])
        return X, y