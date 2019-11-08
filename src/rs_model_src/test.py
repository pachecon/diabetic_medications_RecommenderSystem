import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import src.rs_model_src.ers as ers

from sklearn.externals import joblib

"""Manually add the values from the metrics files"""

def start_plot():

    rmses = [ 0.806638436972228, 0.80803189748168, 0.80803189748168]
    num_factors=[20,30,50,70,80]#np.arange(79,84)
    #num_iter=[20, 30, 40, 50, 60]
    initial_learning_rate=[0.001, 0.003, 0.005, 0.009, 0.01]
    xi = np.arange(1,len(initial_learning_rate)+2)           
    
    # Data
    df=pd.DataFrame({'alpha': [0.001, 0.003, 0.005, 0.009, 0.01],#[0.001, 0.01, 0.005, 0.009, 0.003], 
    'CAMF_C': [0.61,0.0, 0.47, 0.611003849470343, 0.4591341851530296],  
    'CAMF_CI': [0.6262700576983395,0.90,0.4503659292762346, 0.46092627608361314,0.6105940705724321], #[, , ] 
    'CAMF_CU': [0.6003425252797135,0.0,0.4626505596551476,0.48272345774627327,0.5883506257244738], #[0, , 2,3 , ],
    'FM': [0.7731156426272349,0.7829134881472755,0.78490346648031,0.8402489182520281,0.8383711918102627]}) # 0,, 2, , ] }) #0, 1, 2, 3, 4
    
    # multiple line plot
    plt.plot(initial_learning_rate, 'CAMF_C', data=df, marker='o', color='skyblue', linewidth=4, linestyle='dashdot') #markerfacecolor='blue', markersize=12,
    plt.plot( initial_learning_rate, 'CAMF_CI', data=df, marker='o', color='pink', linewidth=2)
    plt.plot(initial_learning_rate, 'CAMF_CU', data=df, marker='o', color='olive', linewidth=2, linestyle='dashed')#, label="toto")
    plt.plot(initial_learning_rate, 'FM', data=df, marker='o', color='blue', linewidth=2) #, linestyle='dashed')#, label="toto")
    plt.legend()
    plt.xlabel(r'$\lambda$')
    plt.ylabel('RMSE')
    plt.xticks(initial_learning_rate)
    plt.title('Root Mean Square Error with different values of learning rate')
    plt.savefig('./RecSys/out/test/comparison_algos.png')
    print('Figure is saved')
    plt.close()
    #plt.show()

    df=pd.DataFrame({'alpha': [0.001, 0.003, 0.005, 0.009, 0.01],#[0.001, 0.01, 0.005, 0.009, 0.003], [20,30,50,70,80] 
    'CAMF_C': [0.61,0.4591341851530296, 0.47, 0.611003849470343,0.0],  
    'CAMF_CI': [0.6262700576983395,0.6105940705724321,0.4503659292762346, 0.46092627608361314,0.90], 
    'CAMF_CU': [0.6003425252797135,0.5883506257244738,0.4626505596551476,0.48272345774627327,0.0], #[0, , 2,3 , ], 
    'FM': [0.7731156426272349,0.7829134881472755,0.78490346648031,0.8402489182520281,0.8383711918102627]}) # 0,, 2, , ] })
    
    plt.plot(num_factors, 'CAMF_C', data=df, marker='o', color='skyblue', linewidth=4, linestyle='dashdot') #markerfacecolor='blue', markersize=12,
    plt.plot(num_factors, 'CAMF_CI', data=df, marker='o', color='pink', linewidth=2)
    plt.plot(num_factors, 'CAMF_CU', data=df, marker='o', color='olive', linewidth=2, linestyle='dashed')#, label="toto")
    plt.plot(num_factors, 'FM', data=df, marker='o', color='blue', linewidth=2) #, linestyle='dashed')#, label="toto")
    plt.legend()
    plt.xlabel('number of factors')
    plt.ylabel('RMSE')
    plt.xticks(num_factors)
    plt.title('Root Mean Square Error with different values of factors')
    plt.savefig('./RecSys/out/test/comparison_algos_1.png')
    print('Figure is saved')
    plt.close()

    item_rmse = np.round(0.714801451389264,3)
    mf_rmse = np.round(0.865545502669787,3)
    plt.bar(['Item-based CF', 'MF'],[item_rmse, mf_rmse], align='center', width=0.6)#, alpha=0.5)
    plt.xlabel('Algorithms')
    plt.ylabel('RMSE')
    plt.title('Root Mean Square Error')
    x=[item_rmse, mf_rmse]
    for i, v in enumerate(x):
        plt.text(i-0.1, v+0.018, " "+str(v), color='black', va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('./RecSys/out/test/baseline_mf.png')
    print('Figure is saved')
    plt.close()
    


def start():
    rmses = []
    maes =[]
    for i in np.arange(0,4):
        print(i)
        print('Starts training ------------------------------------------------------------')
        fname = r'./RecSys/out/CAMF/models/CAMF_C/model_1_loop'+str(i)+'.sav'
        print(fname)
        print('Model already saved')
        model = joblib.load(fname)
        print('Starts validation...')
        rmse, mae= __validation_model__(model, 0, 'CAMF_C', i)
        rmses.append(rmse)
        maes.append(mae)
    print('Saving the evaluation metrics....')
    __save_metrics__(rmses, maes, './RecSys/out/CAMF/predictions/CAMF_C/')


def compare_columns():
    algo = 'CAMF_C'
    fname = './RecSys/out/CAMF/test/'+algo+'/au_predictions.csv'
    df1 = pd.read_csv(fname)
    
    fname = './RecSys/out/FM/results/pred_au.csv'
    df2 = pd.read_csv(fname)

    fname = './RecSys/out/CAMF/test/CAMF_CI/au_predictions.csv'
    df3 = pd.read_csv(fname)

    fname = './RecSys/out/CAMF/test/CAMF_CU/au_predictions.csv'
    df4 = pd.read_csv(fname)

    df5=df2['encounter_id'].isin(df1['encounter_id'].values)
    print(df5)
    print(type(np.float(df1['encounter_id'].values)))
    print(float(df1['encounter_id'].values))

def __validation_model__(model, fold, algo, loop):
    fname = './RecSys/out/CAMF/predictions/'+algo+'/'+str(fold+1)+'_predictions_loop'+str(loop)+'.csv'
    final_pred = pd.read_csv(fname)
    rmse, mae = __evaluation__(final_pred)
    print('Validation is over..')
    return rmse, mae

def __evaluation__(data):
    ers_object = ers.Evaluation(data)
    rmse, mae= ers_object.calculate_errors()
    return rmse, mae

def __save_metrics__(rmses, maes, save_path):
    """
        Method for saving the errors RMSE and MAE
    """
    metrics_results = pd.DataFrame(columns=['RMSE', 'MAE'])
    #rmse = np.mean(rmses)
    #mae = np.mean(maes)
    record = pd.Series([rmses, maes], index=['RMSE', 'MAE'])
    metrics_results = metrics_results.append(record, ignore_index=True)
    save_path = save_path+'metrics_results.csv'
    metrics_results.to_csv(save_path, index=False)
    print("Metrics results can be found here: " + save_path)