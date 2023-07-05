# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 22:13:15 2023

@author: Cotte
"""

from Deconfounding import deconfounder_loop_test_train as deconfound
from Feature_Visualisation import vis_feat_imp
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

#%% Loading in patient IDs

Patient_IDs_GA_VM = pd.read_csv(r"..\Patient IDs_VM_GA.csv",header=None).values
GAs = Patient_IDs_GA_VM[:,2]
VMs = np.asarray(Patient_IDs_GA_VM[:,1],dtype="bool")

#%% Scoring funcs

"""
Due to the fact that it is a classification, boolean multiplication can be used
for the scoring functions that do not have functions already available
"""

#Calculates the sensitivity
def sen(preds,targets):
    return np.sum(preds*targets)/np.sum(targets)

#Calculates the specificity
def spe(preds,targets):
    return np.sum((preds == 0)*(targets == 0))/np.sum(targets == 0)

#%% Train and score func

def train_predict(X_train,X_val_test,y_train,y_val_test,name):
    
    model = GradientBoostingClassifier(n_estimators=5000)
    #Fits the model to the training data
    model.fit(X_train,y_train)
    
    #Automatically generates a feature importance map for convienience
    vis_feat_imp(model, name)
    
    #Predicts results for the training data as well as the test-validation set
    train_pred = model.predict(X_train)
    val_test_pred = model.predict(X_val_test)
    
    #Seperates the validation and test sets
    val_pred = val_test_pred[:14]
    test_pred = val_test_pred[14:]
    
    #Calculates the scoring metrics and prints them
    sensitivity = sen(val_pred,y_val_test[:14])
    specificity = spe(val_pred,y_val_test[:14])
    accuracy = accuracy_score(y_val_test[:14], val_pred)
    
    print(name+" \t Accuracy: "+str(round(accuracy,4))+" Sensitivity: "+str(round(sensitivity,4))+" Specificity: "+str(round(specificity,4)))
    
    #Returns the predicted values, reshaping for easier concatenation
    return train_pred.reshape([-1,1]), val_pred.reshape([-1,1]), test_pred.reshape([-1,1])

#%% Concatenation function

def concatenate_results(file_path, name, prev_train=None, prev_val=None, prev_test=None):
    
    #Reads in the feature file
    Features = pd.read_csv(file_path+" 200 Parcels.csv",header=None).values
    
    #Does the intial split into the train set and the test/validation set
    #Val/test kept together so they can be predicted together more easily
    X_train = Features[:79,:]
    X_val_test = Features[79:,:]
    y_train = VMs[:79]
    y_val_test = VMs[79:]
    C_train = GAs[:79]
    C_val_test = GAs[79:]
    
    #Linearly Deconfound the feature array
    X_train, X_test = deconfound(X_train,X_val_test,C_train,C_val_test)
    
    #Training function called to train and score
    train_pred, val_pred, test_pred = train_predict(X_train,X_val_test,y_train,y_val_test,name)
    
    #Return the predictions by themselves if none are provided
    if prev_train is None and prev_val is None and prev_test is None:
        return train_pred, val_pred, test_pred
    #If other predictions provided, concatenate them together
    else:
        return np.concatenate((prev_train,train_pred),axis=1), np.concatenate((prev_val,val_pred),axis=1), np.concatenate((prev_test,test_pred),axis=1)

#%% Call concatenation function

train_pred, val_pred, test_pred = concatenate_results(r"C:\Users\Cotte\OneDrive - King's College London\Internship\Features\curvature","Curvature")
train_pred, val_pred, test_pred = concatenate_results(r"C:\Users\Cotte\OneDrive - King's College London\Internship\Features\sulc","Sulc",train_pred, val_pred, test_pred)
train_pred, val_pred, test_pred = concatenate_results(r"C:\Users\Cotte\OneDrive - King's College London\Internship\Features\SurfaceArea","Surface Area",train_pred, val_pred, test_pred)
train_pred, val_pred, test_pred = concatenate_results(r"C:\Users\Cotte\OneDrive - King's College London\Internship\Features\corrThickness","Thickness",train_pred, val_pred, test_pred)

#%% Results

model = LogisticRegression()
#Fit the final meta model to the val set predictions
model.fit(val_pred,VMs[79:93])

#Generate the predictions for the final meta model on the test set
preds = model.predict(test_pred)
targets = VMs[93:]

sensitivity = sen(preds,targets)
specificity = spe(preds,targets)
accuracy = accuracy_score(targets,preds)

print("Ensemble         \t Accuracy: "+str(round(accuracy,4))+" Sensitivity: "+str(round(sensitivity,4))+" Specicifity: "+str(round(specificity,4)))

print("\nCurvature: %0.4f\tSulc: %0.4f\tSurface Area: %0.4f\tThickness: %0.4f" %(model.coef_[0,0], model.coef_[0,1], model.coef_[0,2], model.coef_[0,3]))

"""
NEW:
Curvature        Accuracy: 0.7857 Sensitivity: 1.0    Specificity: 0.5714
Sulc        	 Accuracy: 0.5000 Sensitivity: 0.1429 Specificity: 0.8571
Surface Area 	 Accuracy: 0.4286 Sensitivity: 0.0000 Specificity: 0.8571
Thickness 	     Accuracy: 0.5 000Sensitivity: 1.0000 Specificity: 0.0
Ensemble         Accuracy: 0.4286 Sensitivity: 1.0000 Specicifity: 0.2

Curvature: 1.2104	Sulc: -0.0471	Surface Area: -0.4658	Thickness: 0.0000

OLD:
Curvature Deconfounded 	     Accuracy: 0.8095 Sensitivity: 0.7143 Specicifity: 0.8571
Sulc Deconfounded 	         Accuracy: 0.8571 Sensitivity: 0.8571 Specicifity: 0.8571
Surface Area Deconfounded 	 Accuracy: 0.6667 Sensitivity: 0.7143 Specicifity: 0.6429
Thickness Deconfounded   	 Accuracy: 0.8095 Sensitivity: 1.0000 Specicifity: 0.7143
Ensemble         	         Accuracy: 0.8571 Sensitivity: 1.0000 Specicifity: 0.7857

Curvature: 1.6197	Sulc: 1.6197	Surface Area: 1.6197	Thickness: 1.6197
"""