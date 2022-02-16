#baseline
#NMF: For each disease, use Non-negative Matrix Factorization
#to predict the missing values.

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

def matrix_factorization(R,P,Q,K,epsilon,beta):
    Q=Q.T
    matrix_temp = np.dot(P, Q)
    position_nor_0 = np.where((R) != 0)
    original = R[position_nor_0]
    result = matrix_temp[position_nor_0]
    loss_t1 = sum(list(map(lambda x: abs(x[0] - x[1]) / x[0], zip(original, result))))
    # print(loss_t1)
    loss_t1 = loss_t1 / len(original)
    loss_t = loss_t1 + epsilon + 1

    t0 = 10000
    t=t0;
    while abs(loss_t-loss_t1)>epsilon:
        for i in range(len(R)):
            for j in range(len(R[i])):
                    eij=R[i][j]-np.dot(P[i,:],Q[:,j])
                    if R[i][j]>0:
                        alpha = 1 / sqrt(t)
                        t+=1
                        P[i,:]=P[i,:]+alpha*(2*eij*Q[:,j]-beta*P[i,:])
                        Q[:,j]=Q[:,j]+alpha*(2*eij*P[i,:]-beta*Q[:,j])
        loss_t = loss_t1
        matrix_temp=np.dot(P,Q)
        position_nor_0 = np.where((R) != 0)
        original = R[position_nor_0]
        result = matrix_temp[position_nor_0]
        loss_t1 = sum(list(map(lambda x: abs(x[0] - x[1]) / x[0], zip(original, result))))
        # print(loss_t1)
        loss_t1 = loss_t1 / len(original)
    return P,Q.T


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
        if str(origial) != "nan":
            #print(origial, result)
            y_rmse = y_rmse + pow((origial - result), 2)
            y_mae = y_mae + abs(origial - result)
            if origial == 0:
                y_mape = y_mape + 1
            else:
                y_mape = y_mape + (abs(origial - result) / origial)
                print((abs(origial - result) / origial))
            n += 1
    RMSE = sqrt(y_rmse / n)
    MAE = y_mae / n
    MAPE = y_mape / n
    print("RMSE:", RMSE)
    print("MAE:", MAE)
    print("MAPE:", MAPE)
    print()
    return RMSE, MAE, MAPE


def MF(select_diease_id,missing_rate,year):

    file_name = "NMF_" + str(missing_rate*10) + "_" + disease_list[select_diease_id] + ".csv"

    years = ["2011-" + str(i) for i in range(year, year + 1)]
    df = pd.DataFrame()
    df['year'] = [val for val in years for i in range(3)]
    df = df.set_index('year')

    RES = []

    ori_data_x, miss_data_x, data_m = data_loader(missing_rate, select_diease_id)

    N=len(miss_data_x)    #R rows
    M=len(miss_data_x[0]) #R cols
    K=10
    P=np.random.uniform(0, 1, (N,K)) #
    Q=np.random.uniform(0, 1, (M,K)) #
    epsilon = 0.1
    beta = 0.0005
    #print(P)
    nP,nQ=matrix_factorization(ori_data_x,P,Q,K,epsilon,beta)
    #print(R)
    imputed_data_x=np.dot(nP,nQ.T)

    RMSE, MAE, MAPE = test_loss(ori_data_x, imputed_data_x, data_m)

    RES.append(RMSE)
    RES.append(MAE)
    RES.append(MAPE)
    # print(RES)
    # sys.exit(1)
    df[disease_list[select_diease_id]] = RES
    df.to_csv("./result/" + file_name)

if __name__=='__main__':
    begin_year = 2017
    missing_rate_list = [99,97.5]
    for disease_id2 in range(len(disease_list)):
        for missing_rate in missing_rate_list:
            MF(disease_id2,missing_rate,begin_year)

            #sys.exit(1)