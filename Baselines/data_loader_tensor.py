
# Necessary packages
import os
import numpy as np
import sys
import pandas as pd
import random
disease_list = ['Coronary Heart Disease Prevalence',  'Cancer Prevalence', 'Atrial Fibrillation Prevalence']

def data_loader(miss_rate, disease_id):
  year = 2011
  #year = 2017
  N1 = 624
  N3 = 2017 - year + 1

  seed = 25 + disease_id
  random.seed(seed)
  data_x = np.ones((N1, N3), dtype='float64')
  data_m = np.ones((N1, N3), dtype='float64')
  data_age = np.ones((N1, N3), dtype='float64')
  disease_select_list = disease_id
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
    # print(df2.values)
    # sys.exit(1)

  miss_data_x = data_x.copy()

  ward_number = int(N1 * (100 - miss_rate*100) / 100)

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

  miss_data_x[data_m == 0] = 0
  return data_x, miss_data_x, data_m, data_age