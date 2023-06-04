# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 12:49:40 2023

@author: mg
"""

import os
import pandas as pd

#scanners
scanner_type = 'S'
resolution   = '500'

#Items Preparation
png_path    = "./dataset/sd302b/images/baseline/"+scanner_type+'/'+resolution+"/slap-segmented/png/"
#png_path    = "./dataset/sd302b/images/baseline/"+scanner_type+'/'+resolution+"/roll/png/"

csv_path    = "./dataset/participants.csv" 
fgr_types   = ['R. THUMB', 'R. INDEX', 'R. MIDDLE', 'R. RING', 'R. LITTLE',
               'L. THUMB', 'L. INDEX', 'L. MIDDLE', 'L. RING', 'L. LITTLE']

#Extracting information from .CSV file
df_participants = pd.read_csv(csv_path)

#Create output df
columns     = ['PATH', 'FRGP', 'DESCRIPTION', 'SUBJECT', 'AGE', 'YOB', 'GROUP', 'GENDER', 'RACE', 'WORK_TYPE']
df          = pd.DataFrame(columns=(columns))
idx         = 0


#Extracting information from PNG files (roll)
for filename in os.listdir(png_path):
    elements    = os.path.splitext(filename)[0].split('_')
    PATH        = png_path+filename
    FRGP        = int(elements[4])-1
    DESCRIPTION = fgr_types[FRGP]
    SUBJECT     = int(elements[0])
    try:
        AGE         = df_participants.loc[df_participants['id']==SUBJECT]['age'].item()
        YOB         = df_participants.loc[df_participants['id']==SUBJECT]['yob'].item()
        GROUP       = df_participants.loc[df_participants['id']==SUBJECT]['group'].item()
        GENDER      = df_participants.loc[df_participants['id']==SUBJECT]['gender'].item()
        RACE        = df_participants.loc[df_participants['id']==SUBJECT]['race'].item()
        WORK_TYPE   = df_participants.loc[df_participants['id']==SUBJECT]['work_type'].item()
        
        row         = [PATH, FRGP, DESCRIPTION, SUBJECT, AGE, YOB, GROUP, GENDER, RACE, WORK_TYPE] 
        df.loc[idx] = row
        idx        += 1                     
    except:
        print("Sujeto no encontrado.")
        pass
    

df.to_pickle('./dataframes/df_'+scanner_type+'_'+resolution+'.pkl')

print(df.to_string())
print("Statistics: ")
print(df.groupby(['SUBJECT']).size())


