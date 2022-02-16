
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from math import *
import random
import os
import sys
from geomloss import SamplesLoss

from imputers import OTimputer, RRimputer

from utils import *
from data_loaders import dataset_loader
from softimpute import softimpute, cv_softimpute

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.debug("test")

torch.set_default_tensor_type('torch.DoubleTensor')

disease_list = ['Coronary Heart Disease Prevalence',  'Cancer Prevalence', 'Atrial Fibrillation Prevalence']


def data_loader(miss_rate, disease_id):
  seed = 25 + disease_id
  random.seed(seed)

  year = 2011
  diease_select_list = disease_id
  #year = 2017
  N1 = 624
  N3 = 2017 - year + 1
  data_x = np.ones((N1, N3), dtype='float64')
  data_m = np.ones((N1, N3), dtype='float64')


  for y in range(year,2017+1):
    df = pd.read_csv("./DATA/age_diease_population_rate_60_90_norm.csv")
    ward_code_list=list(df['Ward Code'])
    # print(list(df))
    df = df[disease_list[diease_select_list]+"_"+str(y)]

    data_x[:,y - year] = df.values

  miss_data_x = data_x.copy()


  ward_number = int(N1 * (100 - miss_rate) / 100)

  for y in range(0, N3):
    #print(y)
    data_year = data_x[:,y]

    ward_list = []  #
    ward_nor_list = []
    num = 0

    while num < ward_number:
      id = random.randint(0, N1 - 1)
      if id in ward_list:
          continue
      diease_rate = data_year[id]
      if diease_rate != 0:
        ward_list.append(id)
        num = num + 1


    for i in range(N1):
      if i in ward_list:
        continue
      ward_nor_list.append(i)
      data_m[i,y] = 0
  miss_data_x[data_m == 0] = np.nan

  return data_x, miss_data_x, data_m

def test_loss(ori_data_x,imputed_data_x,data_m):#
    n = 0
    y_rmse = 0
    y_mae = 0
    y_mape = 0

    index_list = np.where(data_m == 0)
    #print(index_list)

    R_original = ori_data_x[index_list]
    R_result = imputed_data_x[index_list]
    #print(len(R_original))
    #sys.exit(1)
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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    missing_rate_list = [99,97.5]
    for disease_id in range(len(disease_list)):

        for missing_rate in missing_rate_list:
            print("missing rate:",missing_rate)
            year = 2011
            file_name = "Optimal_transport_" + str(missing_rate*10) + "_" + disease_list[disease_id] + ".csv"
            years=[str(i)+"-2017" for i in range(2011,2012)]
            df = pd.DataFrame()
            df['year'] = [val for val in years for i in range(3)]
            df = df.set_index('year')
            RES = []
            mmin = 100
            MAPE2 = 100
            RMSE_list = []
            MAE_list = []
            MAPE_list = []
            for count in range(5):
                setup_seed(25 + count)
                ground_truth, miss_data_x,data_m = data_loader(missing_rate, disease_id)
                X_true = torch.from_numpy(ground_truth)

                X_miss = torch.from_numpy(miss_data_x)

                n, d = X_miss.shape
                batchsize = 128 # If the batch size is larger than half the dataset's size,
                                # it will be redefined in the imputation methods.
                #0.01
                lr = 0.01

                #print(lr)
                epsilon = pick_epsilon(X_miss) # Set the regularization parameter as a multiple of the median distance, as per the paper.
                #print(epsilon)

                sk_imputer = OTimputer(eps=epsilon, batchsize=batchsize, lr=lr, niter=500)

                sk_imp, sk_maes, sk_rmses = sk_imputer.fit_transform(X_miss, verbose=True, report_interval=500, X_true=X_true)

                X_true = X_true.numpy()
                sk_imp = sk_imp.detach().numpy()

                RMSE, MAE, MAPE = test_loss(ground_truth, sk_imp, data_m)
                RMSE_list.append(RMSE)
                MAE_list.append(MAE)
                MAPE_list.append(MAPE)
                if RMSE + MAE < mmin and MAPE2 > MAPE:
                    RMSE2 = RMSE
                    MAE2 = MAE
                    MAPE2 = MAPE
                    mmin = RMSE + MAE

            RES.append(RMSE2)
            RES.append(MAE2)
            RES.append(MAPE2)

            RMSE_mean = np.mean(RMSE_list)
            MAE_mean = np.mean(MAE_list)
            MAPE_mean = np.mean(MAPE_list)

            RMSE_var = np.var(RMSE_list)
            MAE_var = np.var(MAE_list)
            MAPE_var = np.var(MAPE_list)

            df[disease_list[disease_id] + "_min"] = RES
            df[disease_list[disease_id] + "_mean"] = [RMSE_mean, MAE_mean, MAPE_mean]
            df[disease_list[disease_id] + "_var"] = [RMSE_var, MAE_var, MAPE_var]
            df['RMSE_list'] = [str(RMSE_list),"",""]
            df['MAPE_list'] = [str(MAPE_list),"",""]
            df.to_csv("./result/" + file_name)
            #break