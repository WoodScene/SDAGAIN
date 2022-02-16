'''Main function
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
from data_loader_matrix import data_loader
from utils import rmse_loss
from math import *
import random
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import random
from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

disease_list = ['Coronary Heart Disease Prevalence',  'Cancer Prevalence', 'Atrial Fibrillation Prevalence']

def test_loss(ori_data_x,imputed_data_x,data_m):
    n = 0
    y_rmse = 0
    y_mae = 0
    y_mape = 0

    index_list = np.where(data_m == 0)
    print(index_list)

    R_original = ori_data_x[index_list]
    R_result = imputed_data_x[index_list]
    print(len(R_original))
    #sys.exit(1)
    for id in range(len(R_original)):
        result = R_result[id]
        origial = R_original[id]
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

def Model(data_x,data_m, gain_parameters, times):
    # print(data_x)
    # sys.exit(1)
    seed = 25 + times
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)


    # System parameters
    batch_size = gain_parameters['batch_size']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']

    # Other parameters
    no, dim = data_x.shape
    h_dim = int(dim)


    # Normalization
    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)


    # Input placeholders
    # Data vector
    X = tf.placeholder(tf.float32, shape=[None, dim])
    # Mask vector
    M = tf.placeholder(tf.float32, shape=[None, dim])


    G_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W3 = tf.Variable(xavier_init([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    # Generator
    def generator(x, m):
        # Concatenate Mask and Data
        inputs = tf.concat(values=[x, m], axis=1)
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
        # MinMax normalized output
        G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
        return G_prob

    G_sample = generator(X, M)

    MSE_loss = \
        tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

    G_loss = alpha * MSE_loss
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    ## Iterations
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Start Iterations
    for it in tqdm(range(iterations)):
        # Sample batch
        batch_idx = sample_batch_index(no, batch_size)
        X_mb = norm_data_x[batch_idx, :]
        M_mb = data_m[batch_idx, :]
        Z_mb = uniform_sampler(0, 0.01, batch_size, dim)

        # Sample random vectors

        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        _, MSE_loss_curr = \
            sess.run([G_solver, MSE_loss],
                     feed_dict={X: X_mb, M: M_mb})
        # if it % 1000 == 0:
        #     print(MSE_loss_curr)

    ## Return imputed data
    M_mb = data_m
    X_mb = norm_data_x
    Z_mb = uniform_sampler(0, 0.01, no, dim)
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
    imputed_data = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]

    imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data

    # Renormalization
    imputed_data = renormalization(imputed_data, norm_parameters)

    # Rounding
    imputed_data = rounding(imputed_data, data_x)

    return imputed_data

def main (args,yy, times,disease_id):

  miss_rate = args.miss_rate
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}

  ori_data_x, miss_data_x, data_m = data_loader(miss_rate*100,disease_id)

  # Impute missing data
  imputed_data_x = Model(miss_data_x, data_m, gain_parameters, times)

  RMSE, MAE, MAPE = test_loss(ori_data_x,imputed_data_x,data_m)

  return RMSE, MAE, MAPE

if __name__ == '__main__':
    missing_rate_list = [99,97.5]
    for disease_id in range(len(disease_list)):
        for missing_rate in missing_rate_list:
            file_name = "Auto_encoder_" + str(missing_rate) + "_" + str(disease_list[disease_id]) + ".csv"
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
                    print("times:",str(i))
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
                      default=128,
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
                    if str(RMSE) == "nan":
                        #print("ex")
                        continue
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