import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import time
import json

from utils import *
from f_wb import f_wb_log, sigmoid
from public_tests import *
from cost_function import compute_cost
from gradient_function import compute_gradient, gradient_descent
from normalise import zscore_normalise_features

def predict(X, w, b):
  m, n = X.shape
  p = np.zeros(m)

  for i in range(m):
    p[i] = 1 if f_wb_log(X[i,:], w, b) >= 0.2 else 0
  
  return p

X_train, y_train = load_data()
m, n = X_train.shape
X_mapped = map_feature(X_train[:, 0], X_train[:, 1])

def something():
  np.random.seed(1)
  initial_w = 0.01 * (np.random.rand(2) - 0.5)
  initial_b = -8

  # Some gradient descent settings
  iterations = 1000
  alpha = 0.001

  w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations, 0)

  plot_db(X_train, y_train, w, b)

# something()
# plot_data(X_train, y_train)

def something_2():
  start_time = time.time()

  # Initialize fitting parameters
  np.random.seed(1)
  initial_w = np.random.rand(X_mapped.shape[1])-0.5
  initial_b = 1.

  # Set regularization parameter lambda_ to 1 (you can try varying this)
  lambda_ = 0.01;                                          
  # Some gradient descent settings
  iterations = 100
  alpha = 0.01

  w,b, J_history,_ = gradient_descent(X_mapped, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations, lambda_)

  print(time.time() - start_time)

  p = predict(X_mapped, w, b)
  print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))

  plot_db(X_mapped, y_train, w, b)

# something_2()

def something_file(): 
  f = open('logings/logs_10.json')
  data = json.load(f)
  b = data['b'][-1]
  w = data['w'][-1]

  p = predict(X_mapped, w, b)
  print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))

  plot_db(X_mapped, y_train, w, b)
  f.close()

something_file()

# UNIT TESTS
def test():
  sigmoid_test(sigmoid)
  compute_cost_test(compute_cost)
  compute_gradient_test(compute_gradient)
  predict_test(predict)