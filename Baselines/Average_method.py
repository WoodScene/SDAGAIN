#baseline
#For each disease, take the temporal or spatial average
#as a complementary value.
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

def test_loss(ori_data_x,imputed_data_x,data_m):
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


def Time_average(select_diease_id,missing_rate,year):
    print(year)

    file_name = "Average_time_" + str(missing_rate) + "_" + disease_list[select_diease_id] + ".csv"

    years = ["2011-" + str(i) for i in range(year, year + 1)]
    df = pd.DataFrame()
    df['year'] = [val for val in years for i in range(3)]
    df = df.set_index('year')

    RES = []
    print("Time_average")
    ori_data_x, miss_data_x, data_m = data_loader(missing_rate, select_diease_id)
    ################################################################################
    rows, cols = miss_data_x.shape
    imputed_data_x = np.zeros((rows, cols), dtype='float64')
    for row_id in range(rows):
        line = miss_data_x[row_id]
        index_list = np.where(line != 0)
        values = line[index_list]
        if len(values) == 0:
            imputed_data_x[row_id] = [0 for i in range(cols)]
        else:
            ave_value = np.average(values)
            #print(ave_value)
            imputed_data_x[row_id] = [ave_value for i in range(cols)]
    ################################################################################
    RMSE, MAE, MAPE = test_loss(ori_data_x, imputed_data_x, data_m)
    RES.append(RMSE)
    RES.append(MAE)
    RES.append(MAPE)
    #sys.exit(1)
    df[disease_list[select_diease_id]] = RES
    df.to_csv("./result/" + file_name)


def Spatial_average(select_diease_id,missing_rate,year):
    print(year)

    file_name = "Average_spatial_" + str(missing_rate) + "_" + disease_list[select_diease_id] + ".csv"

    years = ["2011-" + str(i) for i in range(year, year + 1)]
    df = pd.DataFrame()
    df['year'] = [val for val in years for i in range(3)]
    df = df.set_index('year')
    RES = []
    print("Spatial_average")

    ori_data_x, miss_data_x, data_m = data_loader(missing_rate, select_diease_id)
    ################################################################################
    rows, cols = miss_data_x.shape
    imputed_data_x = np.zeros((rows, cols), dtype='float64')
    for col_id in range(cols):
        col = miss_data_x[:,col_id]
        index_list = np.where(col != 0)
        values = col
        if len(values) == 0:
            imputed_data_x[:,col_id] = [0 for i in range(rows)]
        else:
            ave_value = np.average(values)
            imputed_data_x[:,col_id] = [ave_value for i in range(rows)]
    ################################################################################

    RMSE, MAE, MAPE = test_loss(ori_data_x, imputed_data_x, data_m)

    RES.append(RMSE)
    RES.append(MAE)
    RES.append(MAPE)
    df[disease_list[select_diease_id]] = RES
    df.to_csv("./result/" + file_name)

if __name__=='__main__':
    begin_year = 2017
    missing_rate_list = [99,97.5]
    for disease_id2 in range(len(disease_list)):
        for missing_rate in missing_rate_list:
            Time_average(disease_id2,missing_rate,begin_year)
            Spatial_average(disease_id2,missing_rate,begin_year)
            #sys.exit(1)