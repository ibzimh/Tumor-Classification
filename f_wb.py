import numpy as np

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def f_wb_log(x, w, b): # x is the ith training example
  return sigmoid(np.dot(x, w) + b)

  