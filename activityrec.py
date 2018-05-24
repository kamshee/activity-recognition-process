# -*- coding: utf-8 -*-
"""
Inpatient sensor stroke study

This script file combines the 'annotations' file with the accelerometer
and gyroscope data for exploratory analysis.
"""
# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

#Data extraction code modified from CIS-PD: DataPreprocessor2_wTime.ipynb
#https://github.com/adai2017/CIS_PD/blob/master/DataPreprocessor2_wTime.ipynb

#---Pandas version required to load pickle files is 0.20.1 or greater---
pd.__version__

if platform.system() == 'Windows':
    if platform.release() == '7':
        path = r'Y:\Inpatient Sensors -Stroke\Data\biostamp_data\controls'
        folder_path = r'Y:\Inpatient Sensors -Stroke\Data\biostamp_data'
#        dict_path = r'X:\CIS-PD Study\Data_dict'
#        scores_path = r'X:\CIS-PD Study\Scores'
#        features_path = r'X:\CIS-PD Study\FeatureMatrix'
#else:
#    path = '/Volumes/RTO/CIS-PD Study/Subjects/' #Mac
#    folder_path = '/Volumes/RTO/CIS-PD Study/'
#    dict_path = '../Data_dict' # Mac local path
#    scores_path = '../Scores/' # Mac local path
#    features_path = '../FeatureMatrix' # Mac local path

#For a given subject, extracts and separates accelerometer, gyroscope, and 
#EMG/ECG data into trials and sensor per activity
def  extract_data(SubID, path):
    timestamps = process_annotations(path)
    timestamps = fix_errors(SubID, timestamps)
    timestamps = add_unstruct_data(timestamps)
    
    reverse_sensors_1024 = list(['anterior_thigh_left', 'anterior_thigh_right',
                                 'distal_lateral_shank_left', 'distal_lateral_shank_right'])
    # Hard coded list of sensors needed to be reversed in X- and Y- accel/gyro Day 1 data for Subject 1024
    
    # Creates list of sensor locations from folders within subject's raw data directory
    locations = [locs for locs in os.listdir(path) if os.path.isdir(os.path.join(path, locs))]
    
    # Creates dictionary of empty dataframes to merge all accelerometer, gyroscope, and EMG/ECG data for each sensor
    accel = {locs: pd.DataFrame() for locs in locations}
    gyro = {locs: pd.DataFrame() for locs in locations}
    elec = {locs: pd.DataFrame() for locs in locations}
    
    # Finds and merges all accelerometer, gyroscope, and EMG/ECG data for each sensor, retains datetime information
    for root, dirs, files in os.walk(path, topdown=True):
        for filenames in files:
            if filenames.endswith('accel.csv'):
                p = pathlib.Path(os.path.join(root, filenames))
                location = str(p.relative_to(path)).split("\\")[0]
                temp_df = pd.read_csv(p).set_index('Timestamp (ms)')
                accel[location] = accel[location].append(temp_df)

            elif filenames.endswith('gyro.csv'):
                p = pathlib.Path(os.path.join(root, filenames))
                location = str(p.relative_to(path)).split("\\")[0]
                temp_df = pd.read_csv(p).set_index('Timestamp (ms)')
                gyro[location] = gyro[location].append(temp_df)

            elif filenames.endswith('elec.csv'):
                p = pathlib.Path(os.path.join(root, filenames))
                location = str(p.relative_to(path)).split("\\")[0]
                temp_df = pd.read_csv(p).set_index('Timestamp (ms)')
                elec[location] = elec[location].append(temp_df)



#######################


#Import Dataset
#dataset = pd.read_csv('Y:\Inpatient Sensors -Stroke\Data\biostamp_data\controls\HC01\annotations.csv')
#dataset = pd.read_csv('../Inpatient Sensors -Stroke/Data\biostamp_data/controls/HC01/annotations.csv')
dataset = pd.read_csv('annotations.csv')

#Subset Activity Recognition data
df1 = dataset[dataset['EventType'].str.match('Activity')]
df1 = df1[['Timestamp (ms)', 'Start Timestamp (ms)', 'Stop Timestamp (ms)', 'Value']]
df1.Value = df1.Value.shift(-1)
df2 = df1.dropna()
trial = ['trial 1','trial 1','trial 1','trial 1','trial 2','trial 1','trial 1','trial 3','trial 2','trial 3','trial 4','trial 2']
df2['trial'] = trial







#match timestamp to subsubfolder in sacrum sensor ##################
temp1 = pd.read_csv('../sacrum/../MATCH/accel.csv')
temp2 = pd.read_csv('../sacrum/../MATCH/accel.csv')


###############
######## SKIP datetime conversion and use UTC timestamp
###############
#change timestamp from utc to y-m-d time
# ??? keep UTC format and convert at end before graphing?
# but need it converted to check folder for sensor data
x = df2.iloc[0,0]
# x = 1510002136137
y = datetime.datetime.utcfromtimestamp(x).strftime('%Y-%m-%dT%H:%M:%SZ')
y = datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%dT%H:%M:%SZ')
y = datetime.datetime.utcfromtimestamp(1510002136137).strftime('%Y-%m-%dT%H:%M:%SZ')

###########################################
# Play
###########################################
df1.iloc[0:5,]
df1.iloc[-6:-1,]
dataset.iloc[-6:-1,]
df1.iloc[170,3]

#Basic commands to check data
dataset.shape
dataset.columns
dataset.info()
dataset.count()
#.loc for label based indexing or
#.iloc for positional indexing

X = dataset.iloc[:,:-1].values
y= dataset.iloc[:,3].values

