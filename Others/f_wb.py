import numpy as np

# returns an np array of predicted values for each training example (or returns number of X is 1d)
def f_wb_lin(X, w, b):
  return np.dot(X, w) + b

def sigmoid(z):
  g = 1 / (1 + np.exp(-z))
  return g

# returns an np array of predicted values for each training example (or returns number of X is 1d)
def f_wb_log(X, w, b):
  return sigmoid(np.dot(X, w) + b)

def polynomialify(X, narr):
  return np.power(X, narr)