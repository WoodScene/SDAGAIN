
# Necessary packages
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



def Model(data_x, data_age, gain_parameters, times):
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
    - gain_parameters: GAIN network parameters:
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations

  Returns:
    - imputed_data: imputed data
  '''
  # Define mask matrix
  data_x = data_x.reshape(624,7)
  data_age = data_age.reshape(624,7)
  data_m = 1 - np.isnan(data_x)

  # System parameters
  batch_size = gain_parameters['batch_size']
  hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  iterations = gain_parameters['iterations']

  # Normalization
  norm_data, norm_parameters = normalization(data_x)
  norm_data_x = np.nan_to_num(norm_data, 0)
  # Other parameters
  no, dim = norm_data_x.shape
  print(no, dim)

  data_image = np.ones((no, dim, 2), dtype='float64') #624*7*2
  data_image[:,:,0] = norm_data_x
  data_image[:,:,1] = data_age


  # sys.exit(1)
  # Hidden state dimensions
  h_dim = int(dim)
  # print(h_dim)
  ## GAIN architecture
  # Input placeholders
  # Data vector
  X = tf.placeholder(tf.float32, shape=[None, dim, 2])
  # Mask vector
  M = tf.placeholder(tf.float32, shape=[None, dim])
  # Hint vector
  H = tf.placeholder(tf.float32, shape=[None, dim])

  # Discriminator variables
  D_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))  # Data + Hint as inputs
  D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

  D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

  D_W3 = tf.Variable(xavier_init([h_dim, dim]))
  D_b3 = tf.Variable(tf.zeros(shape=[dim]))  # Multi-variate outputs

  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

  # Generator variables
  # Data + Mask as inputs (Random noise is in missing components)

  ## GAIN functions
  # Generator
  def generator(x):
    with tf.variable_scope("g_enerator", reuse=False):

      G_W3 = tf.Variable(xavier_init([h_dim, dim]), name="g_w3")
      G_b3 = tf.Variable(tf.zeros(shape=[dim]), name="g_b3")


      #print(x.shape)
      #sys.exit(1)
      image = tf.reshape(x, [-1, 7, 2])
      #print(image.shape)

      #rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=int(h_dim))
      lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_units=int(h_dim))
                    for layer in range(2)]
      multi_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

      outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
        multi_cell,  # cell you have chosen
        image,  # input
        initial_state=None,  # the initial hidden state
        dtype=tf.float32,  # must given if set initial_state = None
        time_major=False,  # False: (batch, time step, input); True: (time step, batch, input)
      )

      # MinMax normalized output
      output = tf.nn.sigmoid(tf.matmul(outputs[:, -1, :], G_W3) + G_b3)
      #sys.exit(1)
      return output


  def generator2(x):
    with tf.variable_scope("g_enerator", reuse=False):
      G_W3 = tf.Variable(xavier_init([int(h_dim*2), dim]), name="g_w3")
      G_b3 = tf.Variable(tf.zeros(shape=[dim]), name="g_b3")
      #print(x.shape)
      #sys.exit(1)
      image = tf.reshape(x, [-1, 7, 2])
      #print(image.shape)
      n_neurons = int(h_dim)

      lstm_fw1 = tf.nn.rnn_cell.LSTMCell(num_units=n_neurons)
      lstm_fw2 = tf.nn.rnn_cell.LSTMCell(num_units=n_neurons)
      lstm_forward = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_fw1, lstm_fw2])

      lstm_bc1 = tf.nn.rnn_cell.LSTMCell(num_units=n_neurons)
      lstm_bc2 = tf.nn.rnn_cell.LSTMCell(num_units=n_neurons)
      lstm_backward = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_bc1, lstm_bc2])

      outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_forward, cell_bw=lstm_backward, inputs=image,
                                                        dtype=tf.float32)

      state_forward = states[0][-1][-1]
      state_backward = states[1][-1][-1]

      output_forward = outputs[0]
      output_backward = outputs[1]
      #print(output_forward)
      #output_final = output_forward[:, -1, :] + output_backward[:, -1, :]
      #output_final = tf.concat((output_forward[:, -1, :], output_backward[:, -1, :]), 1)
      output_final = tf.concat((state_forward, state_backward), 1)

      #print(output_final)
      #sys.exit(1)
      # MinMax normalized output
      output = tf.nn.sigmoid(tf.matmul(output_final, G_W3) + G_b3)
      #sys.exit(1)
      return output

  # Discriminator
  def discriminator(x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values=[x, h], axis=1)
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob

  ## GAIN structure
  # Generator
  G_sample = generator2(X)
  #print(G_sample.shape)
  #sys.exit(1)
  X2 = X[:,:,0]
  # Combine with observed data
  Hat_X = X2 * M + G_sample * (1 - M)

  # Discriminator
  D_prob = discriminator(Hat_X, H)

  ## GAIN loss
  D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                + (1 - M) * tf.log(1. - D_prob + 1e-8))

  G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))

  MSE_loss = \
    tf.reduce_mean((M * X2 - M * G_sample) ** 2) / tf.reduce_mean(M)

  D_loss = D_loss_temp
  G_loss = G_loss_temp + alpha * MSE_loss

  t_vars = tf.trainable_variables()
  #print(t_vars)
  g_vars = [var for var in t_vars if 'g_' in var.name]
  # for var in t_vars:
  #   if 'g_' in var.name:
  #     print(var.name)
  # sys.exit(1)
  #print(g_vars)
  #print(len(g_vars))
  #sys.exit(1)
  ## GAIN solver
  D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=g_vars)

  ## Iterations
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # Start Iterations
  for it in tqdm(range(iterations)):

    # Sample batch
    batch_idx = sample_batch_index(no, batch_size)
    image_mb = data_image[batch_idx, :, :]
    X_mb = norm_data_x[batch_idx, :]
    M_mb = data_m[batch_idx, :]
    # Sample random vectors
    Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
    # Sample hint vectors
    H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
    # print(H_mb_temp)
    # print(M_mb)
    H_mb = M_mb * H_mb_temp

    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
    image_mb[:, :, 0] = X_mb
    _, D_loss_curr = sess.run([D_solver, D_loss_temp],
                              feed_dict={M: M_mb, X: image_mb, H: H_mb})

    _, G_loss_curr, MSE_loss_curr = \
      sess.run([G_solver, G_loss_temp, MSE_loss],
               feed_dict={X: image_mb, M: M_mb, H: H_mb})
    if it % 1000 == 0:
      print(D_loss_curr, G_loss_curr, MSE_loss_curr)
  # sys.exit(1)
  ## Return imputed data
  Z_mb = uniform_sampler(0, 0.01, no, dim)
  M_mb = data_m
  X_mb = norm_data_x

  X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
  image_mb = data_image
  image_mb[:, :, 0] = X_mb

  imputed_data = sess.run([G_sample], feed_dict={X: image_mb, M: M_mb})[0]
  imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data

  # Renormalization
  imputed_data = renormalization(imputed_data, norm_parameters)

  # Rounding
  imputed_data = rounding(imputed_data, data_x)

  return imputed_data