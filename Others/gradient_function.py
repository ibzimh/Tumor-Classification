import numpy as np
from f_wb import f_wb_lin, f_wb_log

def compute_gradient(X, y, w, b, f_wb):
  m,n = X.shape # number of examples, number of features
  dj_dw = np.zeros((n,))
  dj_db = 0
  for i in range(m):
    f_wb_i = f_wb(X[i], w, b)
    
    for j in range(n):
      dj_dw[j] += ( f_wb_i - y[i] ) * X[i, j]

    dj_db += ( f_wb_i - y[i] )
  return (dj_db/m), (dj_dw/m)

def lr_gradient_function(X, y, w, b):
  return compute_gradient(X, y, w, b, f_wb_lin)

def log_gradient_function(X, y, w, b):
  return compute_gradient(X, y, w, b, f_wb_log)