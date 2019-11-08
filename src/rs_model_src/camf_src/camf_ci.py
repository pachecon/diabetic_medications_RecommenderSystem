from __future__ import print_function
import pandas as pd
import numpy as np
import sys


class CI_class():
    """Class for Context-Aware Matrix Factorization where each contextual condition has an impact on the item.
    """
    def __init__(self, users, items, context, ratings, fold,learning_rate, num_factors):
        """
            Args:
                users (array): encounters_id
                items(array): array with all items itemId's 1 for Metfotmin and 2 for Insulin
                context(array): array of all contexts
                ratings(array): array of all ratings
                fold(int): iteration number
                learning_rate(double): value of the learning rate
                num_factors(double): value of the number of latent factors
        """
        self.__items_array__ = items
        self.__users_array__ = users
        self.__context_array__ = context
        self.__ratings__ = ratings
        self.fold = fold
        self.lr = learning_rate
        self.factors = num_factors


    def fit(self):
        #U: user, I:items, bias_u: user's bias, bias_ci: contextual rating term
        alpha, U, I, bias_u, bias_ci, global_bias, epochs = self.__prepare_data__()
        l = self.lr
        losses = []
        #Calculate tensor factorization and predictions
        for e in range(epochs):
            loss = 0
            print("\rLoop {}/{}.".format(e, epochs), end="\r")
            sys.stdout.flush()
            for i, user in enumerate(self.__users_array__):
                for j, item in enumerate(self.__items_array__):
                    if self.__ratings__[i] != 0:
                        prediction = global_bias + bias_u[i] + np.dot(U[i], I[j].T)#U[i].dot(I[j].T)

                        for c, condition in enumerate(self.__context_array__):
                            prediction += bias_ci[j, c]

                        error = (self.__ratings__[i] - prediction)
                        
                        squared_error = error ** 2
                        loss += squared_error
                        bias_u[i] = alpha * (error - l * bias_u[i])
                        
                        for c, condition in enumerate(self.__context_array__):
                            #for c in np.arange(0,crds[fold]):
                            bias_ci[j, c] = alpha * (error - l * bias_ci[j, c])
                            
                        temp_u = U[i] + alpha * (2 * error * I[j] - l * U[i])
                        U[i] = temp_u

                        temp_i = I[j] + alpha * (2 * error * U[i] - l * I[j])
                        I[j] = temp_i

            losses.append(loss)
            predictions = U.dot(I.T)#(I.T)
        return predictions, losses


    
    def __prepare_data__(self):
        """ method for initializing required variables and setting hyperparameters
        """
        # hyperparameters
        alpha = 0.01 #0.001 # otra consola: 0,05  #aqui; 0.03 #0.001
        epochs = 50#800
        #np.random.seed(6)
     
        # variable... srandomly initialize user/item factors from a Gaussian
        U = np.random.random((len(self.__users_array__),self.factors)) #U = np.random.normal(0,.1,(train.n_users,self.num_factors))
        I = np.random.random((len(self.__items_array__),self.factors)) #V = np.random.normal(0,.1,(train.n_items,self.num_factors))
        bias_one = np.zeros((len(self.__users_array__), 1)) #Bias of user
        crd = np.zeros((len(self.__items_array__),len(self.__context_array__.keys()))) #contextual rating deviation dependet on items
        #crds = [20,40,60,len(self.__context_array__.keys())]
        global_bias = self.__ratings__.mean()
        return alpha, U, I, bias_one, crd, global_bias, epochs
        #return alpha, l, ranks, bias_one, crds, global_bias, epochs