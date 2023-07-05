# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 12:12:18 2022

@author: Cotte
"""

import nibabel as nib
import numpy as np
import pandas as pd

def Asymmetry(Left,Right):
    
    for x in range(len(Left.darrays[0].data)):
        
        """
        From Weiwei's paper
        Normalised (𝐿 − 𝑅) / ((𝐿 + 𝑅) / 2))
        """
        L = Left.darrays[0].data[x]
        R = Right.darrays[0].data[x]
        
        Num = L-R
        Den = (L+R)/2
        
        Left.darrays[0].data[x] = Num/Den
    
    return Left

Patient_IDs_GA_VM = pd.read_csv(r"..\..\Patient IDs_VM_GA.csv",header=None).values
IDs = Patient_IDs_GA_VM[:,0].astype(int)

for ID in IDs:

    for feature_type in ["corrThickness", "curvature", "sulc", "SurfaceArea"]:
        
        Left = nib.load(r"../../Data/"+str(ID)+"_left_"+feature_type+".shape.gii")
        Right = nib.load(r"../../Data/"+str(ID)+"_right_"+feature_type+".shape.gii")
        
        nib.save(Asymmetry(Left,Right), "C:/Users/Cotte/OneDrive - King's College London/Internship/Data/"+str(ID)+"_asymmetry_"+feature_type+".shape.gii")
        
    print("Asymmetry maps for "+str(ID)+" comnpleted")
























