# Necessary packages
import sys
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import random
import math
import matplotlib.pyplot as plt
from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def Model (data_x, data_age, gain_parameters, times):
  seed = 25 + times
  random.seed(seed)
  np.random.seed(seed)
  tf.set_random_seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  tf.reset_default_graph()
  tf.set_random_seed(seed)
  '''Impute missing values in data_x
  
  Args:
    - data_x: original data with missing values
    - gain_parameters: network parameters:
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations
      
  Returns:
    - imputed_data: imputed data
  '''
  # Define mask matrix
  data_m = 1-np.isnan(data_x) #1*624*7
  data_x[np.isnan(data_x)] = 0

  age_m = 1-np.isnan(data_age)
  # print(data_m.shape)
  # sys.exit(1)
  # System parameters
  batch_size = gain_parameters['batch_size']
  hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  iterations = gain_parameters['iterations']
  z_range = 0.01
  cnn_filter_height = 2
  # Other parameters
  data_number = data_x.shape[0]
  seq_length = data_x.shape[1]
  seq_dim = data_x.shape[2]
  num_of_attention_heads = 1
  hidden_size = int(seq_dim*0.8)*num_of_attention_heads
  attention_head_size = int(hidden_size / num_of_attention_heads)

  # Hidden state dimensions
  h_dim = int(seq_dim)
  dim = h_dim
  #print(h_dim)
  
  # Normalization
  # norm_data, norm_parameters = normalization(data_x)
  # norm_data_x = np.nan_to_num(norm_data, 0)


  data_image = np.ones((data_number, seq_length*2, seq_dim, 1), dtype='float64') #1*624*7*2
  for item in range(seq_length):
    data_image[0, item * 2, :, 0] = data_x[0, item, :]
    data_image[0, item * 2 + 1, :, 0] = data_age[0, item, :]

  ## GAIN architecture   
  # Input placeholders
  # Image vector
  X_IMAGE  = tf.placeholder(tf.float32, shape=[None, seq_length*2, seq_dim, 1])
  # Data vector
  X = tf.placeholder(tf.float32, shape=[None, seq_length, seq_dim])
  AGE = tf.placeholder(tf.float32, shape=[None, seq_length, seq_dim])
  # Mask vector 
  M = tf.placeholder(tf.float32, shape=[None, seq_length, seq_dim])
  M_age = tf.placeholder(tf.float32, shape=[None, seq_length, seq_dim])
  M_disease = tf.placeholder(tf.float32, shape=[None, dim])
  # Hint vector
  H = tf.placeholder(tf.float32, shape = [None, dim])
  
  # Discriminator variables
  D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
  D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W3 = tf.Variable(xavier_init([h_dim, dim]))
  D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
  
  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
  
  #Generator variables
  # Data + Mask as inputs (Random noise is in missing components)

  
  ## GAIN functions
  # Generator
  def generator(x,m):
    inputs = x
    #print(inputs)
    with tf.variable_scope("g_enerator", reuse=False):

      conv_filter_w1 = tf.Variable(tf.random_normal([cnn_filter_height, 2, 1, 1])) # [ filter_height, filter_width, in_channel, out_channels ]
      conv_filter_b1 = tf.Variable(tf.random_normal([1]))

      conv_filter_w2 = tf.Variable(tf.random_normal([cnn_filter_height, 2, 1, 1]))
      conv_filter_b2 = tf.Variable(tf.random_normal([1]))


      #transformer
      wq = tf.Variable(xavier_init([seq_dim-cnn_filter_height+1, hidden_size]))
      wk = tf.Variable(xavier_init([seq_dim-cnn_filter_height+1, hidden_size]))
      wv = tf.Variable(xavier_init([seq_dim-cnn_filter_height+1, hidden_size]))

      wo = tf.Variable(xavier_init([hidden_size, seq_dim-cnn_filter_height+1]))
      bo = tf.Variable(tf.zeros(shape=[seq_dim-cnn_filter_height+1]))
      ########################################################
      relu_feature_maps1 = tf.nn.relu( \
        tf.nn.conv2d(inputs, conv_filter_w1, strides=[1, 2, 1, 1], padding='VALID') + conv_filter_b1)

      max_pool1 = tf.nn.max_pool(relu_feature_maps1, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')
      max_pool1 = tf.reshape(max_pool1,(-1, seq_length, seq_dim-cnn_filter_height+1))

      # Concatenate Mask and Data
      mix_q = tf.matmul(max_pool1, wq)  # [Batch_size x Seq_length x Hidden_size]

      mix_k = tf.matmul(max_pool1, wk)  # [Batch_size x Seq_length x Hidden_size]
      mix_v = tf.matmul(max_pool1, wv)  # [Batch_size x Seq_length x Hidden_size]

      mix_q = tf.reshape(mix_q, (
      -1, seq_length, num_of_attention_heads, attention_head_size))  # [Batch_size x Seq_length x ]
      query_layer = tf.transpose(mix_q, [0, 2, 1, 3])  # [Batch_size x Num_of_heads x Seq_length x Head_size]

      mix_k = tf.reshape(mix_k, (
      -1, seq_length, num_of_attention_heads, attention_head_size))  # [Batch_size x Seq_length x ]
      key_layer = tf.transpose(mix_k, [0, 2, 1, 3])  # [Batch_size x Num_of_heads x Seq_length x Head_size]

      mix_v = tf.reshape(mix_v, (
      -1, seq_length, num_of_attention_heads, attention_head_size))  # [Batch_size x Seq_length x ]
      value_layer = tf.transpose(mix_v, [0, 2, 1, 3])  # [Batch_size x Num_of_heads x Seq_length x Head_size]

      k_t = tf.transpose(key_layer, [0, 1, 3, 2])
      attention_scores = tf.matmul(query_layer, k_t)
      attention_scores = attention_scores / math.sqrt(attention_head_size)
      attention_probs = tf.nn.softmax(attention_scores)

      context_layer = tf.matmul(attention_probs, value_layer)

      output = tf.layers.dense(inputs=context_layer, units=attention_head_size,
                               activation=tf.nn.relu)  # [Batch_size x Seq_length x Hidden_size]
      output = tf.nn.dropout(output, 0.1)
      output2 = tf.reshape(output, (-1, seq_length, hidden_size))
      output3 = tf.nn.relu(tf.matmul(output2, wo) + bo) #[Batch_size x Seq_length x seq_dim] 1 624 7
      output3 = tf.reshape(output3,(-1, seq_length, seq_dim-cnn_filter_height+1, 1))
      #print(output3) #(?, 624, 6, 1)

      relu_feature_maps2 = tf.nn.sigmoid( \
        tf.nn.conv2d_transpose(output3, conv_filter_w2, output_shape=[data_number, seq_length*2, seq_dim, 1], strides=[1, 2, 1, 1], padding='VALID') + conv_filter_b2)
      #print(relu_feature_maps2) #shape=(1, 1248, 7, 1)

    return relu_feature_maps2
      
  # Discriminator
  def discriminator(x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h], axis = 1)
    print(inputs)
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob
  
  ## GAIN structure
  # Generator
  G_output = generator(X_IMAGE, M)
  G_sample = G_output[:,0:1248:2,:,0]
  Age_sample = G_output[:,1:1248:2,:,0]

  # Combine with observed data
  Hat_X = X * M + G_sample * (1-M)
  #print(Hat_X.shape) #1 624 7
  Hat_X2 = Hat_X[0,:,:]
  #print(Hat_X2.shape) # 624 7
  #sys.exit(1)
  # Discriminator
  D_prob = discriminator(Hat_X2, H)
  
  ## GAIN loss
  D_loss_temp = -tf.reduce_mean(M_disease * tf.log(D_prob + 1e-8) \
                                + (1-M_disease) * tf.log(1. - D_prob + 1e-8))
  
  G_loss_temp = -tf.reduce_mean((1-M_disease) * tf.log(D_prob + 1e-8))
  
  MSE_loss = \
  tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M) + tf.reduce_mean((M_age * AGE - M_age * Age_sample)**2) / tf.reduce_mean(M_age)

  D_loss = D_loss_temp
  G_loss = G_loss_temp + alpha * MSE_loss

  t_vars = tf.trainable_variables()
  # print(t_vars)
  # print("len Var:",len(t_vars))
  theta_G = [var for var in t_vars if 'g_' in var.name]
  #print(theta_G)
  #sys.exit(1)

  ## GAIN solver
  D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
  
  ## Iterations
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # Start Iterations
  for it in tqdm(range(iterations)):    
      
    # Sample batch
    batch_idx = sample_batch_index(data_number, batch_size)
    X_mb = data_x[batch_idx, :, :]
    Age_mb = data_age[batch_idx, :, :]
    Image_mb = data_image[batch_idx, :, :, :]

    M_mb = data_m[batch_idx, :, :]
    M_disease_mb = data_m[batch_idx, :, :]
    M_disease_mb = M_disease_mb.reshape(seq_length, dim)
    M_mb_age = age_m[batch_idx, :, :]
    # Sample random vectors  
    Z_mb = uniform_sampler(0, z_range, batch_size, seq_length, seq_dim)
    # Sample hint vectors
    H_mb_temp = binary_sampler(hint_rate, seq_length, dim)
    #print(M_disease_mb.shape)
    #sys.exit(1)
    #print(M_mb)
    H_mb = M_disease_mb * H_mb_temp

    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
    Image_mb[:,0:1248:2,:,0] = X_mb
    #print(Image_mb[:,:,:,0])
      
    _, D_loss_curr = sess.run([D_solver, D_loss_temp],
                              feed_dict = {M: M_mb, X: X_mb, H: H_mb, M_disease: M_disease_mb, M_age: M_mb_age, AGE: Age_mb, X_IMAGE:Image_mb})
    _, G_loss_curr, MSE_loss_curr = \
    sess.run([G_solver, G_loss_temp, MSE_loss],
             feed_dict = {X: X_mb, M: M_mb, H: H_mb, M_disease: M_disease_mb, M_age: M_mb_age, AGE: Age_mb, X_IMAGE:Image_mb})
    if it % 1000 == 0:
      print(MSE_loss_curr)
  ## Return imputed data
  Z_mb = uniform_sampler(0, z_range, data_number, seq_length, seq_dim)
  M_mb = data_m
  M_disease_mb = data_m[:, :, 0]
  X_mb = data_x
  X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
  Image_mb = data_image
  Image_mb[:, 0:1248:2, :, 0] = X_mb

  imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb, X_IMAGE:Image_mb})[0]
  
  imputed_data = data_m * data_x + (1-data_m) * imputed_data
  
  # Renormalization
  #imputed_data = renormalization(imputed_data, norm_parameters)

  # Rounding
  #imputed_data = rounding(imputed_data, data_x)

  return imputed_data