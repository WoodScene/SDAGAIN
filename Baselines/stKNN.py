#baseline
#This uses the average readings of its k nearest spatial
#and temporal neighbors as a prediction (𝑘=6).

import numpy as np
import random
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from math import *
from tensorly import tucker_to_tensor
from time import *
import random
from data_loader_matrix import data_loader
disease_list = ['Coronary Heart Disease Prevalence',  'Cancer Prevalence', 'Atrial Fibrillation Prevalence']

def test_loss(ori_data_x,imputed_data_x,data_m):#计算测试误差
    n = 0
    y_rmse = 0
    y_mae = 0
    y_mape = 0

    index_list = np.where(data_m == 0)
    R_original = ori_data_x[index_list]
    R_result = imputed_data_x[index_list]
    for id in range(len(R_original)):
        result = R_result[id]
        origial = R_original[id]
        #print(id,origial,result)
        if str(origial) != "nan" and origial != 0:
            #print(origial, result)
            y_rmse = y_rmse + pow((origial - result), 2)
            y_mae = y_mae + abs(origial - result)
            y_mape = y_mape + (abs(origial - result) / origial)
            n += 1
    RMSE = sqrt(y_rmse / n)
    MAE = y_mae / n
    MAPE = y_mape / n
    print("RMSE:", RMSE)
    print("MAE:", MAE)
    print("MAPE:", MAPE)
    print()
    return RMSE, MAE, MAPE

def stkNN_method(select_diease_id,missing_rate):
    file_name = "stKNN_" + str(missing_rate) + "_" + disease_list[select_diease_id] + ".csv"
    year = 2017
    years = ["2011-" + str(i) for i in range(year, year + 1)]
    df = pd.DataFrame()
    df['year'] = [val for val in years for i in range(3)]
    df = df.set_index('year')
    ori_data_x, miss_data_x, data_m = data_loader(missing_rate, select_diease_id)
    RES = []
    k = 6
    ################################################################################
    rows, cols = miss_data_x.shape
    imputed_data_x = np.zeros((rows, cols), dtype='float64')
    for i in range(rows):
        for j in range(cols):
            if data_m[i,j] == 0:
                value_list = np.sum(miss_data_x[i-3:i,j]) + np.sum(miss_data_x[i+1:i+4,j]) + np.sum(miss_data_x[i,j-3:j]) + np.sum(miss_data_x[i,j+1:j+4])
                #print(value_list)
                ave = value_list/12
                imputed_data_x[i,j] = ave
    ################################################################################

    RMSE, MAE, MAPE = test_loss(ori_data_x, imputed_data_x, data_m)
    RES.append(RMSE)
    RES.append(MAE)
    RES.append(MAPE)
    df[disease_list[select_diease_id]] = RES
    df.to_csv("./result/" + file_name)


if __name__=='__main__':
    missing_rate_list = [99,97.5]
    for disease_id2 in range(len(disease_list)):
        for missing_rate in missing_rate_list:
            stkNN_method(disease_id2, missing_rate)
            #sys.exit(1)