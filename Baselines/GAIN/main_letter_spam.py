'''Main function
'''
# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
from data_loader import data_loader2
from gain import gain
from utils import rmse_loss
from math import *
import sys
import random
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

def main (args,yy,times,disease_id):

  data_name = args.data_name
  miss_rate = args.miss_rate

  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}

  ori_data_x, miss_data_x, data_m = data_loader2(data_name, miss_rate, yy, times,disease_id)
  
  # Impute missing data
  imputed_data_x = gain(miss_data_x, gain_parameters,times,ori_data_x, data_m, disease_id)
  print(disease_list[disease_id] + " results:")
  RMSE, MAE, MAPE = test_loss(ori_data_x,imputed_data_x,data_m)
  #sys.exit(1)
  return RMSE, MAE, MAPE

if __name__ == '__main__':
    missing_rate_list = [99,97.5]
    for disease_id in range(len(disease_list)):
        for missing_rate in missing_rate_list:
            file_name = "GAIN_" + str(missing_rate) + "_"+str(disease_list[disease_id])+".csv"
            years=[str(i)+"-2017" for i in range(2011,2012)]
            df=pd.DataFrame()
            df['year']=[val for val in years for i in range(3)]
            df=df.set_index('year')
            res_perl = []
            for yy in range(2011,2012):
                mmin = 100
                MAPE2 = 100
                RMSE_list = []
                MAE_list = []
                MAPE_list = []
                for i in range(5):
                    print("year:" + str(yy) + "-2017")
                    print("times:", str(i))
                    # Inputs for the main function
                    parser = argparse.ArgumentParser()
                    parser.add_argument(
                      '--data_name',
                      choices=['letter','spam'],
                      default='spam',
                      type=str)
                    parser.add_argument(
                      '--miss_rate',
                      help='missing data probability',
                      default=missing_rate/100,
                      type=float)
                    parser.add_argument(
                      '--batch_size',
                      help='the number of samples in mini-batch',
                      default=624,
                      type=int)
                    parser.add_argument(
                      '--hint_rate',
                      help='hint probability',
                      default=0.9,
                      type=float)
                    parser.add_argument(
                      '--alpha',
                      help='hyperparameter',
                      default=100,
                      type=float)
                    parser.add_argument(
                      '--iterations',
                      help='number of training interations',
                      default=5000,
                      type=int)

                    args = parser.parse_args()
                    print(args)
                    #sys.exit(1)
                    # Calls main function
                    RMSE, MAE, MAPE = main(args,yy,i,disease_id)
                    RMSE_list.append(RMSE)
                    MAE_list.append(MAE)
                    MAPE_list.append(MAPE)
                    if RMSE + MAE < mmin and MAPE2 > MAPE:
                        RMSE2= RMSE
                        MAE2 = MAE
                        MAPE2 = MAPE
                        mmin = RMSE+MAE
                res_perl.append(RMSE2)
                res_perl.append(MAE2)
                res_perl.append(MAPE2)
                RMSE_mean = np.mean(RMSE_list)
                MAE_mean = np.mean(MAE_list)
                MAPE_mean = np.mean(MAPE_list)
                print("mean RMSE:", RMSE_mean)
                print("mean MAE:", MAE_mean)
                print("mean PERL:", MAPE_mean)
                RMSE_var = np.var(RMSE_list)
                MAE_var = np.var(MAE_list)
                MAPE_var = np.var(MAPE_list)
                df[disease_list[disease_id] + "_min"] = res_perl
                df[disease_list[disease_id] + "_mean"] = [RMSE_mean, MAE_mean, MAPE_mean]
                df[disease_list[disease_id] + "_var"] = [RMSE_var, MAE_var, MAPE_var]
                df[disease_list[disease_id] + "_RMSE"] = str(RMSE_list)
                df[disease_list[disease_id] + "_MAE"] = str(MAE_list)
                df[disease_list[disease_id] + "_MAPE"] = str(MAPE_list)
                df.to_csv("./result/" + file_name)