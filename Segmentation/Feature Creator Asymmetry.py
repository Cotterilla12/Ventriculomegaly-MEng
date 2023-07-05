# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%% Libraries

import pandas as pd
import numpy as np
import nibabel as nib
import os

#%% Presets
#Determines which maps are being parcellated and where the resulting csv file is saved

#Determines which of the voronoi parcellations are being used
Number_of_parcels = 200

#Feature type being calculated, needs to be the same as in the file names
feature_types = ['curvature',"corrThickness","SurfaceArea","sulc"]

#Path to the folders holding the data
data_path_string = r"../Data"

#Where the resulting csv file will be saved
Feature_CSV_string = r"C:\Users\Cotte\OneDrive - King's College London\Internship\Features\Asymmetry"

#%% Loading IDs and destational ages

#Loads in the file with Patient ID, gestational age and ventriculomegaly status and takes ID
Patient_IDs_GA_VM = pd.read_csv(r"..\Patient IDs_VM_GA.csv",header=None).values
Patient_IDs = np.asarray(Patient_IDs_GA_VM[:,0],dtype="int")
samples = len(Patient_IDs)

#%% Loading in the voronoi parcellation and taking a slice of the parcellation being used

Parcels = (nib.load(r"dhcp_all_parcels.dscalar.nii")).get_fdata()

locs = Parcels[int((Number_of_parcels/50)-1),:32492]

#%% Initialising Feature array

Features = np.zeros([samples,int(Number_of_parcels/2)])

#%% Creating the features

"""
Goes through the parcels one by one to calculate the mean of that area

Left and right are done at the same time for ease of coding. The first half of
the parcels are on the left hemisphere and the latter half on the right

All features are calculated with np.mean except for surface area which was
calculated with np.sum
"""
    
print('\nVoroni '+str(Number_of_parcels)+' parcellation:\n')

for feature_type in feature_types:

    for parcel in np.arange(1,(Number_of_parcels/2)+1):
                
        print('Parcel '+str(int(parcel))+' completed')
        
        for row, ID in enumerate(Patient_IDs):
            
            asy_path_string = os.path.join(data_path_string,str(ID)+"_left_"+feature_type+".shape.gii")
        
            asy = nib.load(asy_path_string)
            
            A_index = np.where(locs == parcel)
            
            if len(A_index[0]) > 0:
                Features[row,int(parcel-1)] = np.mean(asy.darrays[0].data[A_index])
            
    np.savetxt(os.path.join(Feature_CSV_string,feature_type+" "+str(Number_of_parcels)+" Parcels.csv"),Features,delimiter=',')
    
    print("Saved to disk")