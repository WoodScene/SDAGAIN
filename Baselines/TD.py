#baseline
#TD: Construct a Tensor with three dimensions (year, grid, and
#disease), and use tensor decomposition to predict the missing values
from math import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorly import tucker_to_tensor
from time import *
import random
from data_loader_tensor import data_loader

disease_list = ['Coronary Heart Disease Prevalence',  'Cancer Prevalence', 'Atrial Fibrillation Prevalence']

def Random_gradient_descent(A,S,R,C,T,epsilon,lambda3):
    #print()
    loss_tensor = tucker_to_tensor(S, [R, C, T])
    position_nor_0 = np.where((A) != 0)
    original = A[position_nor_0]
    result = loss_tensor[position_nor_0]

    loss_t1 = sum(list(map(lambda x: abs(x[0] - x[1]) / x[0], zip(original, result))))
    loss_t1 = loss_t1 / len(original)

    #loss_t1 = sum(list(map(lambda x: pow((x[0] - x[1]), 2), zip(original, result))))
    #loss_t1 = sqrt(loss_t1)
    loss_t=loss_t1+epsilon-1
    print("loss:",loss_t1)
    #step size,1/sqrt(t++)
    t0=10
    t=t0;
    flag=0
    times = 0
    while abs(loss_t-loss_t1)>epsilon:
        if loss_t > loss_t1:
            flag = 1
        # if flag == 1 and loss_t < loss_t1:
        #     break
        for i in range(len(A)):
            for j in range(len(A[0])):
                for k in range(len(A[0][0])):
                    if A[i][j][k]==0:
                        continue
                    nita=1/sqrt(t)
                    t+=1
                    eijk = A[i][j][k] - tucker_to_tensor(S, [R[i, :], C[j, :], T[k, :]])[0]

                    RLfy=tucker_to_tensor(S,[R[i, :], C[j, :], T[k, :]],skip_factor=0)
                    CLfy=tucker_to_tensor(S,[R[i, :], C[j, :], T[k, :]],skip_factor=1)
                    TLfy=tucker_to_tensor(S,[R[i, :], C[j, :], T[k, :]],skip_factor=2)
                    SLfy=np.ones((len(S),len(S[0]),len(S[0][0])),dtype='float32')
                    temp = np.array([C[j, :]]).T * [T[k, :]]
                    for tt in range(len(S)):
                        SLfy[tt] = R[i, tt] * temp
                    #print(RLfy,CLfy,TLfy,SLfy)
                    #print("--")
                    #print(nita,lambda3,eijk)
                    R[i, :]=(1-nita*lambda3)*R[i, :]+nita*eijk*RLfy
                    C[j, :]=(1-nita*lambda3)*C[j, :]+nita*eijk*CLfy
                    T[k, :]=(1-nita*lambda3)*T[k, :]+nita*eijk*TLfy
                    S=(1-nita*lambda3)*S+nita*eijk*SLfy

        #compute function loss
        loss_tensor=tucker_to_tensor(S, [R,C,T])
        position_nor_0=np.where((A)!=0)
        original=A[position_nor_0]
        result=loss_tensor[position_nor_0]

        loss_t=loss_t1
        loss_t1=sum(list(map(lambda x: abs(x[0]-x[1])/x[0], zip(original, result))))
        loss_t1=loss_t1/len(original)

        times += 1
    print("test loss:", loss_t1)
    return S,R,C,T

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

def TD(select_diease_id, missing_rate):
    file_name = "TD_" + str(missing_rate) + "_" + disease_list[select_diease_id] + ".csv"
    year = 2017
    years = ["2011-" + str(i) for i in range(year, year + 1)]
    df = pd.DataFrame()
    df['year'] = [val for val in years for i in range(3)]
    df = df.set_index('year')
    data_x, miss_data_x, data_m, data_age = data_loader(missing_rate, select_diease_id)
    rows, cols = miss_data_x.shape
    A = np.ones((2, rows, cols), dtype='float64')
    A[0,:,:] =  miss_data_x
    A[1,:,:] =  data_age
    dim1 = 2
    dim2 = rows
    dim3 = cols

    # size of core Tensor
    dimX = 10
    dimY = 10
    dimZ = 10
    S = np.random.uniform(0, 0.1, (dimX, dimY, dimZ))
    R = np.random.uniform(0, 0.1, (dim1, dimX))
    C = np.random.uniform(0, 0.1, (dim2, dimY))
    T = np.random.uniform(0, 0.1, (dim3, dimZ))
    #print(R,C,T)
    nS, nR, nC, nT = Random_gradient_descent(A, S, R, C, T,0.0003,0.0005)
    A_result = tucker_to_tensor(nS, [nR, nC, nT])
    RES = []
    RMSE, MAE, MAPE = test_loss(data_x, A_result[0], data_m)
    RES.append(RMSE)
    RES.append(MAE)
    RES.append(MAPE)
    #sys.exit(1)
    df[disease_list[select_diease_id]] = RES
    df.to_csv("./result/" + file_name)

if __name__=='__main__':
    missing_rate_list = [99]
    for disease_id2 in range(len(disease_list)):
        for missing_rate in missing_rate_list:
            TD(disease_id2, missing_rate)
            #sys.exit(1)





