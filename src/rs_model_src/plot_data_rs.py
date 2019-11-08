import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os

class PlotRSData():

    def user_behavior(self):
        data = self.utility_matrix.groupby('encounter_id')['insulin'].count().clip(upper=50)
        fig, ax = plt.subplots()
        ax = data.plot.hist(bins=10) #alpha=0.5
        plt.title('Distribution of Insulin ratings per user (patient)')
        ax.set_xlabel('Rating per user')
        ax.set_ylabel('Count number of users')
        ax.set_xlim(0,5)
        plt.savefig("./RecSys/data_behavior/user_behavior.png")
        plt.close(fig)
        

    def ratings_distribution(self):
        #"No":0, "Steady":1, "Down":2, "Up":3
        data = self.new_data['insulin'].value_counts().sort_index(ascending=True)
        fig, ax = plt.subplots()
        ax = data.plot(kind='bar',facecolor='#AA0000')
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        ax.patch.set_facecolor('#FFFFFF')
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['left'].set_linewidth(1)
        ax.set_xlabel('Ratings: "No":0, "Steady":1, "Down":2, "Up":3')
        ax.set_ylabel('Count')

        percentiles = ['{:.1f} %'.format(val) for val in (data.values / self.new_data.shape[0] * 100)]
        for i,child in enumerate(ax.get_children()[:data.index.size]):
            ax.text(i,child.get_bbox().y1+200,percentiles[i], horizontalalignment ='center')
        plt.title('Insulin ratings distribution for readmitted patients')
        plt.savefig("./RecSys/data_behavior/distribution_ratings.png")
        plt.close(fig)
    
    def isSparse(self, array,m, n) : 
        counter = 0
        # Count number of zeros 
        # in the matrix 
        for i in range(0,m) : 
            for j in range(0,n) : 
                if (array[i][j] == 0) : 
                    counter = counter + 1
        percentage = (counter/(m*n)) * 100
        return percentage

    def plot_loss_mf(self, losses, prefix):
        fig = plt.figure()
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig("./RecSys/out/MF/Plots/" + prefix + ".png")
        plt.close(fig)
    
    def plot_loss_cars(self, losses, type_camf, prefix):
        fig = plt.figure()
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
        plt.savefig("./RecSys/out/CAMF/train/Plots/"+type_camf+"/" + prefix + ".png")
        plt.close(fig)

#return (counter >  
#        ((m * n) // 2)) 
#print(len(np.unique(utility_matrix['encounter_id'])))
#print(len(np.unique(utility_matrix_pivot['encounter_id'])))
#print(len(utility_matrix['encounter_id']))
#print(len(utility_matrix_pivot['encounter_id']))
#self.__items_array__.remove('Unnamed: 0')
#sparse = utility_matrix_pivot.drop(['encounter_id', 'readmitted'],axis=1)
#items = utility_matrix_pivot.columns.values.tolist()
#long_tail =recmetrics.Recmetircs_class(utility_matrix,utility_matrix_pivot)
#long_tail.plot()
#print(sparse.keys())
#print(sparse.values)
#print(scipy.sparse.issparse(sparse.values))
#print(sparse.values.shape)
#m = 28499
#n = 13
#print(isSparse(sparse.values, m, n)) 