import numpy as np
from f_wb import f_wb_log

def loss(f_wb, y):
  return ( -y * np.log(f_wb) ) - ( (1 - y) * np.log(1 - f_wb) )

def compute_cost(X, y, w, b, lambda_=1):
  m, n = X.shape # no. of training examples, no. of features
  cost = 0

  for i in range(m):
    cost += loss(f_wb_log(X[i,:], w, b), y[i])
  cost /= m

  # cost += sum(np.square(w)) * (lambda_ / (2 * m)) # regularising 

  return cost