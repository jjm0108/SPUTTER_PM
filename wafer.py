import CONSTANTS as consts
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

ch = pd.DataFrame(pd.read_csv("./3. SPU01-FDC DATA/CH_1/SPU-01 CH_1 FDC DATA.csv"))
ch['time'] = pd.to_datetime(ch['time'])

#print(ch.columns)
lotID = ''
RECIPE = ''
STEP = 0

START = None
END = None

param_l = consts.numeric

info = {}
columns = ['start','end']
for parameter in param_l : columns.append(parameter)
DATA=pd.DataFrame(columns=columns)


for index, row in ch.iterrows():
    #step이 달라졌을 때
    if STEP != row.step:
        if index <1 : info['end'] = row.time
        else: info['end'] = ch.loc[index-1].time
        #이전 결과 출력
        for parameter in param_l : 
            if parameter in info.keys() : info[parameter] = np.mean(info[parameter])
        print(info)
        if len(info) >1 : DATA = DATA.append(info, ignore_index=True)
        print(DATA)
        
        #new
        info['start'] = row.time
        for parameter in param_l : 
            info[parameter] = [row[parameter]]
        STEP = row.step

    #step이 동일할 때    
    else:
        for parameter in param_l :
            info[parameter].append(row[parameter])
        
        
                

