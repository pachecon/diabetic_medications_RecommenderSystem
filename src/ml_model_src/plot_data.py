import matplotlib.pyplot as plt
from textwrap import wrap

import numpy as np
import os

class PlotDataClasses():
    """
    Class for plotting the behavior of the data and to observe the imbalanced of the "readmitted" classes
    between No readmitted, Readmitted with < 30 days of staying and Readmitted with > 30 days of staying
    """
    def __init__(self, X, y):
        """
            Args:
                X: data set
                y: label set --> classes: No readmitted, Readmitted with < 30 days of staying and Readmitted with > 30 days of staying
            Return:
                all plots are save on the same root
                "./MaLer/Plots/imbalance_classes/"
        """  
        self.data = X
        self.labels = y

    def num_lab_medications(self):
        """ 
            Plots the comparison between the number of laboratory procedures and the number of medications
        """
        fig = plt.figure()
        ax = self.data.boxplot(column=['num_lab_procedures', 'num_medications'])#self.data.plot.box()#
        plt.title("Visualization of number of Laboratory Procedures and Medications" )
        #ax.set_xlabel("Number of laboratory procedure performed and of distinct generic names administered during the encounter")
        plt.xticks([1, 2], ['Laboratory procedure performed', 'Medications administered'])
        ax.set_ylabel("Total number of each group")
        plt.savefig("./MaLer/Plots/imbalance_classes/distribution_num.png")
        plt.close(fig)


    def medication_comparison(self):
        """
            Plots the comparison between the diabetic medication Insulin and Metformin.
            Where the values can be "No change of dose":0, "Steady dose":1, "Down dose":2, or "Up dose":3
        """
        ind = np.arange(1, 5)
        width = 0.35 
        fig = plt.figure()
        ax = fig.add_subplot(111)

        x_no = self.data['metformin'].eq(0).sum()
        x_s = self.data['metformin'].eq(1).sum()
        x_d = self.data['metformin'].eq(2).sum()
        x_up = self.data['metformin'].eq(3).sum()
        x = [x_no, x_s, x_d, x_up]
        met_bars = ax.bar(ind, x, width, color='royalblue')
        x_no = self.data['insulin'].eq(0).sum()
        x_s = self.data['insulin'].eq(1).sum()
        x_d = self.data['insulin'].eq(2).sum()
        x_up = self.data['insulin'].eq(3).sum()
        x = [x_no, x_s, x_d, x_up]
        ins_bars = ax.bar(ind+width, x,width, color='g')
        ax.set_xticks(ind)
        ax.set_xticklabels(['No change', 'Steady dose', 'Down dose', 'Up dose'])
        ax.set_xticks(ind + width / 2)
        ax.set_ylabel('Total of encounters')
        ax.set_title('Comparison between Metformin and Insulin')
        ax.legend((met_bars[0], ins_bars[0]), ('Metformin', 'Insulin'))
        plt.savefig("./MaLer/Plots/imbalance_classes/medication_comparison.png")
        plt.close(fig)


    def readmitted(self):
        """
            Plot the imbalance classes of the readmitted column.
            The labels can be No readmitted, Readmitted with < 30 days of staying and Readmitted with > 30 days of staying
            ">30":2,"<30":3, "NO":1 
        """
        total_zero = self.labels['readmitted'].eq(1).sum()
        total = len(self.labels['readmitted']) - total_zero
        total_less_30 = self.labels['readmitted'].eq(3).sum()
        total_higher_30 = self.labels['readmitted'].eq(2).sum()
        x = [total_zero, total_less_30, total_higher_30]
        ind = np.arange(1, 4)
        #print(x)
        fig, ax = plt.subplots()
        pm, pc, pn = plt.bar(ind, x)
        pm.set_facecolor('r')
        pc.set_facecolor('g')
        pn.set_facecolor('b')
        ax.set_xticks(ind)
        ax.set_xticklabels(['No readmitted', 'Readmitted <30 days', 'Readmitted >30 days'])
        ax.set_ylabel('Total of encounters')
        ax.set_title('Original data set')
        
        plt.savefig("./MaLer/Plots/imbalance_classes/imbalance_classes_original.png")
        plt.close(fig)

    def change_med_and_readmission(self):
        """
            Plot the comparison between change in any diabetic medications and readmission of the patients
            Change medication is "No": 2 or "Ch"/"change":1
            Readmitted is "No": 1 or "Yes":the difference between the len of the column with the amount of no readmitted encounters,
            where the classes of Readmitted <30 days and Readmitted >30 days are considered as one group (Yes).
        """
        width = 0.35 
        fig = plt.figure()
        ax = fig.add_subplot(111)
        total_no = self.data['change'].eq(2).sum()
        total = self.data['change'].eq(1).sum()
        ind = np.arange(1,3)
        x = [total_no, total]
        change_bars = ax.bar(ind, x, width, color='royalblue')
        
        total_no = self.labels['readmitted'].eq(1).sum()
        total = len(self.labels['readmitted']) - total_no
        x = [total_no, total]
        read_bars = ax.bar(ind+width, x,width, color='g')
        
        ax.set_xticks(ind)
        ax.set_xticklabels(['No', 'Yes'])
        ax.set_xticks(ind + width / 2)
        ax.set_ylabel('Total of encounters')
        title = ax.set_title("\n".join(wrap('Comparison between change in any diabetic medications and readmission of the patients')))
        ax.legend((change_bars[0], read_bars[0]), ('Change of medications', 'Readmitted'))
        fig.tight_layout()
        title.set_y(1.05)
        fig.subplots_adjust(top=0.8)
        plt.savefig("./MaLer/Plots/imbalance_classes/change_readmitted.png")
        plt.close(fig)