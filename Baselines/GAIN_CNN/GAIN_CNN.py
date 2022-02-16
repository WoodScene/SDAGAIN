'''
CNN-GAIN function.
'''

# Necessary packages
import tensorflow as tf
import sys
import numpy as np
from tqdm import tqdm
import os
import random
from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index, uniform_sampler_2d
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

  data_x = data_x.reshape(624,7)
  # Define mask matrix
  data_m = 1-np.isnan(data_x)
  
  # System parameters
  batch_size = gain_parameters['batch_size']
  hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  iterations = gain_parameters['iterations']
  
  # Other parameters
  no, dim = data_x.shape
  
  # Hidden state dimensions
  h_dim = int(dim)
  #print(h_dim)
  
  # Normalization
  norm_data, norm_parameters = normalization(data_x)
  #norm_data_x = np.nan_to_num(norm_data, 0)
  norm_data_x = np.nan_to_num(data_x, 0)

  data_image = np.ones((1, no, dim, 2), dtype='float64') #1*624*7*2
  data_image[:,:,:,0] = norm_data_x
  data_image[:,:,:,1] = data_age

  # Input placeholders
  X_pre = tf.placeholder(tf.float32, shape=[1, 624, dim, 2])
  # Data vector
  #X = tf.placeholder(tf.float32, shape = [None, dim])
  # Mask vector 
  M = tf.placeholder(tf.float32, shape = [None, dim])
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
  conv_filter_w1 = tf.Variable(tf.random_normal([1, 2, 2, 3]))
  conv_filter_b1 = tf.Variable(tf.random_normal([3]))

  conv_filter_w2 = tf.Variable(tf.random_normal([1, 2, 3, 1]))
  conv_filter_b2 = tf.Variable(tf.random_normal([1]))
  # Data + Mask as inputs (Random noise is in missing components)
  G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))  
  G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W3 = tf.Variable(xavier_init([h_dim, dim]))
  G_b3 = tf.Variable(tf.zeros(shape = [dim]))
  
  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3, conv_filter_w1, conv_filter_b1, conv_filter_w2, conv_filter_b2]

  # CNN + Generator
  def generator(x,m):
    relu_feature_maps1 = tf.nn.relu( \
      tf.nn.conv2d(x, conv_filter_w1, strides=[1, 1, 1, 1], padding='SAME') + conv_filter_b1)
    max_pool1 = tf.nn.max_pool(relu_feature_maps1, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

    relu_feature_maps2 = tf.nn.relu( \
      tf.nn.conv2d(max_pool1, conv_filter_w2, strides=[1, 1, 1, 1], padding='SAME') + conv_filter_b2)
    max_pool2 = tf.nn.max_pool(relu_feature_maps2, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

    x2 = tf.reshape(max_pool2, [no, dim])

    # Concatenate Mask and Data
    inputs = tf.concat(values = [x2, m], axis = 1)
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
    return G_prob
      
  # Discriminator
  def discriminator(x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h], axis = 1) 
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob
  
  ## CPH structure
  # Generator
  G_sample = generator(X_pre, M)
  X2 = X_pre[0,:,:,0]
  # Combine with observed data
  Hat_X = X2 * M + G_sample * (1-M)
  
  # Discriminator
  D_prob = discriminator(Hat_X, H)
  
  ## CPH loss
  D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                + (1-M) * tf.log(1. - D_prob + 1e-8)) 
  
  G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))
  
  MSE_loss = \
  tf.reduce_mean((M * X2 - M * G_sample)**2) / tf.reduce_mean(M)
  
  D_loss = D_loss_temp
  G_loss = G_loss_temp + alpha * MSE_loss 
  
  ## CPH solver
  D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
  
  ## Iterations
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
   
  # Start Iterations
  for it in tqdm(range(iterations)):    
      
    # Sample batch
    batch_idx = sample_batch_index(1, batch_size)
    #print(len(batch_idx))
    image_mb = data_image[:,:, :, :]
    X_mb = norm_data_x[:, :]
    M_mb = data_m[:, :]
    # Sample random vectors  
    Z_mb = uniform_sampler_2d(0, 0.01, no, dim)
    # Sample hint vectors
    H_mb_temp = binary_sampler(hint_rate, no, dim)

    H_mb = M_mb * H_mb_temp
    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
    image_mb[0, :, :, 0] = X_mb

    _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                              feed_dict = {M: M_mb, X_pre: image_mb, H: H_mb})
    _, G_loss_curr, MSE_loss_curr = \
    sess.run([G_solver, G_loss_temp, MSE_loss],
             feed_dict = {X_pre: image_mb, M: M_mb, H: H_mb})
            
  ## Return imputed data      
  Z_mb = uniform_sampler_2d(0, 0.01, no, dim)
  M_mb = data_m
  X_mb = norm_data_x          
  X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
  image_mb = data_image
  image_mb[0, :, :, 0] = X_mb

  imputed_data = sess.run([G_sample], feed_dict = {X_pre: image_mb, M: M_mb})[0]
  
  imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data
  
  # Renormalization
  #imputed_data = renormalization(imputed_data, norm_parameters)
  
  # Rounding
  imputed_data = rounding(imputed_data, data_x)  
          
  return imputed_data