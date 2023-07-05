# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 23:59:30 2023

@author: Cotte
"""

import nibabel as nib
import numpy as np
import os

def vis_feat_imp(model, name):
    parcels = nib.load(r"..\Segmentation\dhcp_all_parcels.dscalar.nii").get_fdata()
    L_parcels = parcels[3,:32492]
    R_parcels = parcels[3,32492:]
    
    left_ex = nib.load(r"Importances/Example Files/Left_Example.shape.gii")
    right_ex = nib.load(r"Importances/Example Files/Right_Example.shape.gii")
    
    importances = model.feature_importances_
    
    importances = importances/max(importances)
    
    L_importances = importances[:100]
    R_importances = importances[100:]
        
    left_ex.darrays[0].data[:] = -1
    right_ex.darrays[0].data[:] = -1
    
    for i in range(100):
        locs = np.where(L_parcels == i+1)
        left_ex.darrays[0].data[locs] = L_importances[i]
    
    nib.save(left_ex,"Importances/"+name+" Left Importances.shape.gii")
    
    if len(R_importances) == 100:
        for j in range(100,200):
            locs = np.where(R_parcels == j+1)
            right_ex.darrays[0].data[locs] = R_importances[j-100]
        
        nib.save(right_ex,"Importances/"+name+" Right Importances.shape.gii")