# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:16:47 2023

@author: Cotte
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import r_regression

def deconfounder_loop(Xij, C, plot = False):
    """
    Follows method in https://www.sciencedirect.com/science/article/pii/S1053811918319463
    to remove the patterns of the confounds from the data with a bias term added to the confound
    """

    #Checks the shape of the incoming confound array and if it is a single dim, adds a second dim for matrix calculations
    if len(C.shape)==1:
        C = np.reshape(C,[-1,1])
        
    #Adds in a bias term
    C_new = np.concatenate((np.ones([C.shape[0],1]),C),axis=1)
    
    #Performs half the linear regression calulation
    CT = np.transpose(C_new)
    inv_CTC = np.linalg.inv(np.matmul(CT,C_new))
    inv_CTC_CT = np.matmul(inv_CTC,CT)
    
    #Initialises the array to store corrected values
    Xij_corr = np.empty(Xij.shape)
    
    #Loops over all features
    for j in range(Xij.shape[1]):
        #Calcuating the coefficients for the confounds provided (second half of the linear regression calc)
        Bj = np.matmul(inv_CTC_CT,Xij[:,j])
        
        #Subtracting coefs*confounds from original array and assigning to corrected array
        Xij_corr[:,j] = Xij[:,j] - np.sum(C_new*Bj,axis=1)
        
        #Plots everything
        if plot:
            #If there are multiple confounds provided, only the first will be visualised
            if C.shape[1] == 1:
                plt.scatter(C,Xij[:,j])
                plt.scatter(C,Xij_corr[:,j])
                plt.title('Correlation before: '+str(r_regression(np.reshape(Xij[:,j],[-1,1]),C.flatten())[0])+"\nCorrelation after: "+str(r_regression(np.reshape(Xij_corr[:,j],[-1,1]),C.flatten())[0]))
            else:
                plt.scatter(C[:,0],Xij[:,j])
                plt.scatter(C[:,0],Xij_corr[:,j])
                plt.title('Correlation before: '+str(r_regression(np.reshape(Xij[:,j],[-1,1]),C[:,0].flatten())[0])+"\nCorrelation after: "+str(r_regression(np.reshape(Xij_corr[:,j],[-1,1]),C[:,0].flatten())[0]))
            plt.xlabel("GA (weeks)")
            plt.ylabel("Feature "+str(j+1))
            plt.show()
        
    return Xij_corr

def deconfounder_loop_test_train(X_train,X_test,C_train,C_test,plot=False):
    """
    Same method as the previous function, with added functionality to perform
    the calculation on the train-set and then perform the same operation on the
    test set to avoid data leakage
    """

    #Checks the shape of the incoming confound arrays and if it is a single dimension, adds the second for later calculations
    if len(C_train.shape)==1:
        C_train = np.reshape(C_train,[-1,1])
        C_test = np.reshape(C_test,[-1,1])
        
    #Adding in the bias terms
    C_train_new = np.concatenate((np.ones([C_train.shape[0],1]),C_train),axis=1)
    C_test_new = np.concatenate((np.ones([C_test.shape[0],1]),C_test),axis=1)
    
    #Performs half of the linear regression calculation on the train set
    CT = np.transpose(C_train_new)
    inv_CTC = np.linalg.inv(np.matmul(CT,C_train_new))
    inv_CTC_CT = np.matmul(inv_CTC,CT)
    
    #Initialises the empty arrays to store the corrected values
    X_train_corr = np.empty(X_train.shape)
    X_test_corr = np.empty(X_test.shape)
    
    for j in range(X_train.shape[1]):
        #Calcuating the coefficients for the confounds provided (using only the train set) (second half of the linear regression calc)
        Bj = np.matmul(inv_CTC_CT,X_train[:,j])
        
        #Subtracting coefs*confounds from original arrays and assigning to corrected arrays
        X_train_corr[:,j] = X_train[:,j] - np.sum(C_train_new*Bj,axis=1)
        X_test_corr[:,j] = X_test[:,j] - np.sum(C_test_new*Bj,axis=1)
        
        #Plots everything if needed
        if plot:
            #If there are multiple confounds provided, only the first will be visualised
            if C_train.shape[1] == 1:
                plt.scatter(C_train,X_train[:,j],label="Uncorrected Train")
                plt.scatter(C_train,X_train_corr[:,j],label="Corrected Train")
                plt.scatter(C_test,X_test[:,j],label="Uncorrected Test")
                plt.scatter(C_test,X_test_corr[:,j],label="Corrected Test")
                plt.title('Correlation before: '+str(r_regression(np.reshape(X_train[:,j],[-1,1]),C_train.flatten())[0])+"\nCorrelation after: "+str(r_regression(np.reshape(X_train_corr[:,j],[-1,1]),C_train.flatten())[0]))
            else:
                plt.scatter(C_train[:,0],X_train[:,j],label="Uncorrected Train")
                plt.scatter(C_train[:,0],X_train_corr[:,j],label="Corrected Train")
                plt.scatter(C_test[:,0],X_test[:,j],label="Uncorrected Test")
                plt.scatter(C_test[:,0],X_test_corr[:,j],label="Corrected Test")
                plt.title('Correlation before: '+str(r_regression(np.reshape(X_train[:,j],[-1,1]),C_train[:,0].flatten())[0])+"\nCorrelation after: "+str(r_regression(np.reshape(X_train_corr[:,j],[-1,1]),C_train[:,0].flatten())[0]))
            plt.xlabel("GA (weeks)")
            plt.ylabel("Feature "+str(j+1))
            plt.legend(loc="upper right")
            plt.show()
        
    return X_train_corr, X_test_corr