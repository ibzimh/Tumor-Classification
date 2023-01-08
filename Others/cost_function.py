import numpy as np
from f_wb import f_wb_lin, f_wb_log

# returns a number which is the cost given the w's and b
def sq_err_cost_function(X, y, w, b):
  m = X.shape[0]
  cost = 0.0
  for i in range(m):
    cost += ( f_wb_lin(X[i], w, b) - y[i] ) ** 2
  return cost / (2 * m)

def log_cost_function(X, y, w, b):
  m = X.shape[0] # no. of training examples
  cost = 0

  for i in range(m): # for each training example
    cost += log_loss(f_wb_log(X[i,:], w, b), y[i])

  return cost / m

def log_loss(f_x, y):
  return ( -y * np.log(f_x) ) - ( (1 - y) * np.log(1 - f_x) )