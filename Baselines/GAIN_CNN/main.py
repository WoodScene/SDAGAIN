'''Main function
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import argparse
import numpy as np
import pandas as pd
from data_loader import data_loader2
#from gain import gain
from GAIN_CNN import Model
from math import *
import random
disease_list = ['Coronary Heart Disease Prevalence',  'Cancer Prevalence', 'Atrial Fibrillation Prevalence']

def test_loss(ori_data_x,imputed_data_x,data_m):#计算测试误差
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

def main (args,yy,times,disease_id):
  
  data_name = args.data_name
  miss_rate = args.miss_rate

  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
  ori_data_x, miss_data_x, data_m, data_age = data_loader2(data_name, miss_rate, yy, times,disease_id)

  # Impute missing data
  imputed_data_x = Model(miss_data_x, data_age, gain_parameters, times)
  res = []
  for dd in range(len(disease_id)):
    print(disease_list[disease_id[dd]]+" results:")
    RMSE, MAE, MAPE = test_loss(ori_data_x[dd,:,:],imputed_data_x,data_m[dd,:,:])
    res.append(RMSE)
    res.append(MAE)
    res.append(MAPE)
  #sys.exit(1)
  #return RMSE, MAE, MAPE
  return res

if __name__ == '__main__':
    missing_rate_list = [99,97.5]
    for disease_id2 in range(len(disease_list)):
        for missing_rate in missing_rate_list:
            disease_id = [disease_id2]
            external_data = ["Age"]

            file_name = "GAIN_CNN_" + str(missing_rate) + "_"+disease_list[disease_id[0]]+".csv"
            years=[str(i)+"-2017" for i in range(2011,2012)]
            df=pd.DataFrame()
            df['year']=[val for val in years for i in range(3)]
            df=df.set_index('year')

            df_res = pd.DataFrame()
            name_list = []
            for i in disease_id:
                name_list.append(disease_list[i] + "_RMSE")
                name_list.append(disease_list[i] + "_MAE")
                name_list.append(disease_list[i] + "_MAPE")
            df_res['index_name'] = name_list

            res_perl = []
            for yy in range(2011,2012):
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
                      default=1,
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
                    res = main(args,yy,i,disease_id)
                    df_res[i+1] = res
                print(df_res)
                df_res = df_res.set_index('index_name')

                mean_list = []
                var_list = []
                for row in range(len(df_res)):
                    #print()
                    mm = np.mean(df_res.iloc[row,:])
                    vv = np.var(df_res.iloc[row, :])
                    mean_list.append(mm)
                    var_list.append(vv)

                df_res['mean'] = mean_list
                df_res['var'] = var_list

                df_res.to_csv("./result/" + file_name, encoding='utf_8_sig')