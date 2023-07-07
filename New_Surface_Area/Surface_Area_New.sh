#!/bin/bash

surface_path="/mnt/c/Users/Cotte/OneDrive - King's College London/Internship/New Data/surf/"
save_path="/mnt/c/Users/Cotte/OneDrive - King's College London/Internship/New Data/shape/"

cd /home/cotterilla12/dHCP_Images/New_Surface_Area

amount=1

for patient in $(cat<IDs) ; do

for side in 'L' 'R' ; do

wb_command -surface-vertex-areas "${surface_path}"${patient}.${side}.midthickness.native.surf.gii "${save_path}"${patient}.${side}.SurfaceArea.native.shape.gii

done

echo "Processed ${amount}/115"

amount=$((amount+1))

done
