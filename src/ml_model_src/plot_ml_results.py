import itertools
import os
import seaborn as sn
import pandas as pd

from matplotlib import pyplot as plt
#import matplotlib.pyplot as plt
import numpy as np

def plot_total_classes_SMOTE(y_train_res, type_al, name, prefix='train_'):
    #">30":2,"<30":3, "NO":1
    x = np.arange(3)
    fig, ax = plt.subplots()
    no_read = y_train_res.__eq__(1).sum()
    r_mayor30 = y_train_res.__eq__(2).sum()
    r_menor30 = y_train_res.__eq__(3).sum()
    x_classes = [no_read, r_mayor30, r_menor30]
    plt.bar(x, x_classes, color='g')
    plt.xticks(x, ('No Readmitted', '>30 days', '<30 days'))
    ax.set_ylabel('Total of encounters')
    ax.set_title('After oversampling SMOTE')
    print('Saving plots "./MaLer/Plots/imbalance_classes/SMOTE/"')
    plt.savefig("./MaLer/Plots/imbalance_classes/SMOTE/"+type_al+"/"+prefix+name+".png")
    plt.close(fig)

def plot_confusion_matrix_sn (type_al, name,cm2, cm, prefix='train_', normalize = False):
    #{">30":2,"<30":3, "NO":1}
    type_name = ''
    if (type_al == 'svm'):
        type_name = 'SVM'
    elif(type_al == 'log_reg'):
        type_name = 'Logistic Regression'
    else:
        type_name = 'Random Forest'
    classes = ['No Readmitted', '>30 days', '<30 days'] 
    fig, ax = plt.subplots()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap("coolwarm"))# annot=True, linewidths=.5)
    plt.title('Confusion matrix for '+type_name)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes, rotation= 0)
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #print(i)
        #print(j)
        #print(cm.iloc[i,j])
        plt.text(j, i, np.round(cm.iloc[i, j],3),
                     horizontalalignment="center",
                     color="black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    print('Saving the confusion matrix plot...')
    #fig = cm_plot.get_figure()
    #fig.savefig("./MaLer/Plots/confusion_matrix/"+type_al+"/"+prefix+name+".png")
    #fig.clf()
    plt.savefig("./MaLer/Plots/confusion_matrix/"+type_al+"/"+prefix+name+".png")
    plt.close(fig)

        
def plot_confusion_matrix(type_al, name, cm, normalize=True, cmap=plt.cm.get_cmap("coolwarm"), prefix='train_cm_'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    classes = ['No Readmitted', '>30 days', '<30 days']
    fig, ax = plt.subplots()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm =np.round( cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],3)
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    print('Saving the confusion matrix plot...')
    plt.savefig("./MaLer/Plots/confusion_matrix/"+type_al+"/"+prefix+name+".png")
    plt.close(fig)

def plot_comparison_models(values, rmse):
    """
        Args:
            values: array with the accuracy values
            rmse: array with the rmse values
        Plot the accuracy values of each ml_algorithm on the same figure to visualize the difference between those.
        Plot RMSE values of each ml_algorithm for the same purpose. 
    """
    fig, ax = plt.subplots()
    fig.suptitle('Algorithm Comparison Acuracy')
    ax.boxplot(values,showmeans=True)#['SVM', 'Random Forest', 'Log. Reg.'],np.array(values).astype(np.float)
    plt.xlabel('Machine Learning Algorithms')
    plt.ylabel('Accuracy')
    ax.set_xticklabels(['SVM', 'Random Forest', 'Log. Reg.'])
    plt.savefig("./MaLer/Plots/compare_algos/accuracy.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    fig.suptitle('Root Mean Square error (RMSE)')
    ax.boxplot(rmse,showmeans=True)
    plt.xlabel('Machine Learning Algorithms')
    plt.ylabel('RMSE')
    ax.set_xticklabels(['SVM', 'Random Forest', 'Log. Reg.'])
    plt.savefig("./MaLer/Plots/compare_algos/rmse.png")
    plt.close(fig)
    
def plot_predict_test(result_dict, rmse_list,name):
    """
        Args:
            result_dict: dictionary with the name of the best model with its accuracy value
            rmse_list:   list of all rmse of all the folds 
        Plot the Accuracy and the RMSE of the best model after making predictions
    """
    fig, ax = plt.subplots()
    fig.suptitle('Acuracy from unseen data')
    plt.scatter(1, result_dict.values(),s=100)
    #ax.set_xticklabels(result_dict.keys())
    plt.xlabel(name)
    plt.ylabel('Accuracy')
    ax.set_xticklabels([])
    plt.savefig("./MaLer/Plots/predict_test/accuracy.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    fig.suptitle('Root Mean Square error (RMSE) from unseen data')
    plt.scatter(1, rmse_list,s=100)
    plt.xlabel(name)
    plt.ylabel('RMSE')
    ax.set_xticklabels([])
    #ax.set_xticklabels(['Random Forest'])
    plt.savefig("./MaLer/Plots/predict_test/rmse.png")
    plt.close(fig)