import sys
sys.path.insert(1, './src/')
sys.path.insert(1, './src/ml_model_src/') #files for the machine learning problem
sys.path.insert(1, './src/rs_model_src/') #files for the recommender system

import pandas as pd
import utilities_data
import readmission_pred
import rs_model
import rs_modelCARS
import rs_model_FM
import make_dirs
import os.path
import test
import numpy


def check_console_input(arguments):
    if len(arguments) < 2:
        print('Error on the command.')
        print('The format should be: ')
        print('Please try again.')
        exit()

def run_all():
    print('Recommender system with baseline')
    rec_sys = rs_model.RSmodel(data, encounters_readmitted)
    recommendations = rec_sys.get_final_rs_matrix()
    print('---------------------------------------------------------------')
    print('Recommender system with CAMF')
    rec_sys = rs_modelCARS.CARS(data, encounters_readmitted)
    predictions = rec_sys.get_predictions()
    print('---------------------------------------------------------------')
    print('Recommender system with FM')
    #Recommendations with Features Matrix
    rec_sys = rs_model_FM.FM_RS(data, encounters_readmitted)
    predictions = rec_sys.get_predictions()
    print('---------------------------------------------------------------')

if __name__ == "__main__":
    arguments = sys.argv
    
    #test.start_plot()
    #test.start()
    #test.compare_columns()
    #exit()
    # check the input arguments. If it is incorrect - exit
    check_console_input(arguments)

    # store checked arguments
    type_recsys = arguments[1]
    create_dirs = make_dirs.MakeDirClass()
    
    fname = r'data/diabetes_data_preprocessed.csv'
    data = 0
    print('Loading data...')
    # ************ Cleaning the data *****************************
    if os.path.isfile(fname) :
        print('File of eprocessed data has been found...')
        data = pd.read_csv(fname)
    else:
        data= utilities_data.prepare_data()
    print('---------------------------------------------------------------')
    # ************ First part: machine learning problem **************
    #Predict which encounters will be readmitted
    print('Start prediction task...')
    ml_model = readmission_pred.MLClass(data)
    encounters_readmitted = ml_model.prediction_readmitted()
    print('---------------------------------------------------------------')
    
    # ************ Second part: recommender system *******************
    #Give a recommendation if the dosage of medications should be increased, decreased, or stable
    print('Start recommender system task...')
    if(type_recsys == "baseline"):
        print('Recommender system with baseline')
        rec_sys = rs_model.RSmodel(data, encounters_readmitted)
        recommendations = rec_sys.get_final_rs_matrix()
        print('---------------------------------------------------------------')
    elif(type_recsys == "camf"):
        print('Recommender system with CAMF')
        rec_sys = rs_modelCARS.CARS(data, encounters_readmitted)
        predictions = rec_sys.get_predictions()
        print('---------------------------------------------------------------')
    elif(type_recsys == "fm"):
        print('Recommender system with FM')
        #Recommendations with Features Matrix
        rec_sys = rs_model_FM.FM_RS(data, encounters_readmitted)
        predictions = rec_sys.get_predictions()
        print('---------------------------------------------------------------')
    elif(type_recsys == "all"):
        run_all()
    else:
        print("Error")
        print("Recommender system type should be either baseline or camf or fm")
        print("Please try again")
        exit()
