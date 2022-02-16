
# Necessary packages
import os
import numpy as np
import sys
import pandas as pd
import random
from utils import sample_index
disease_list = ['Coronary Heart Disease Prevalence',  'Cancer Prevalence', 'Atrial Fibrillation Prevalence']

def data_loader(miss_rate, yy, times, disease_id):
  year = yy
  N1 = 624
  N2 = len(disease_id)
  N3 = 2017 - year + 1
  data_tensor = np.ones((N2, N1, N3), dtype='float64')
  miss_tensor = np.ones((N2, N1, N3), dtype='float64')
  mask_tensor = np.ones((N2, N1, N3), dtype='float64')

  age_tensor = np.ones((N2, N1, N3), dtype='float64')

  for d_id in range(len(disease_id)):
    seed = 25 + disease_id[d_id]
    random.seed(seed)
    data_x = np.ones((N1, N3), dtype='float64')
    data_m = np.ones((N1, N3), dtype='float64')
    data_age = np.ones((N1, N3), dtype='float64')
    disease_select_list = disease_id[d_id]
    #print("disease id: ", diease_select_list)
    for y in range(year,2017+1):
      df = pd.read_csv("../Data/age_diease_population_rate_60_90_norm.csv")
      ward_code_list=list(df['Ward Code'])
      # print(list(df))
      df1 = df[disease_list[disease_select_list]+"_"+str(y)]
      data_x[:,y - year] = df1.values

      #df2 = df["population_"+str(y)+"_60_90_rate"]
      df2 = df["population_"+str(y)+"_60_90_rate_norm"]
      data_age[:,y - year] = df2.values

    miss_data_x = data_x.copy()
    ward_number = int(N1 * (100 - miss_rate*100) / 100)

    for y in range(0, N3):
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
    data_tensor[d_id,:,:] = data_x
    miss_tensor[d_id,:,:] = miss_data_x
    mask_tensor[d_id,:,:] = data_m
    age_tensor[d_id,:,:] = data_age

  return data_tensor, miss_tensor, mask_tensor, age_tensor