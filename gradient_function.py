import numpy as np
from f_wb import f_wb_log
import math

def compute_gradient(X, y, w, b, lambda_=1):
  m, n = X.shape # no. of training examples, no. of features

  dj_dw = np.zeros(n)
  dj_db = 0.

  for i in range(m):
    dj_db += f_wb_log(X[i,:], w, b) - y[i]
    for j in range(n):
      dj_dw[j] += (f_wb_log(X[i,:], w, b) - y[i]) * (X[i,j])

  dj_dw /= m
  dj_db /= m

  for j in range(n):
    dj_dw[j] += (lambda_/m) * w[j]

  return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
  # number of training examples
  m = len(X)
  
  # An array to store cost J and w's at each iteration primarily for graphing later
  J_history = []
  w_history = []
  b_history = []
  
  for i in range(num_iters):

    # Calculate the gradient and update the parameters
    dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

    # Update Parameters using w, b, alpha and gradient
    w_in = w_in - alpha * dj_dw               
    b_in = b_in - alpha * dj_db              
    
    # Save cost J at each iteration
    if i<1000000:      # prevent resource exhaustion 
      cost =  cost_function(X, y, w_in, b_in, lambda_)
      J_history.append(cost)

    # Print cost every at intervals 10 times or as many iterations if < 10
    if i % math.ceil(num_iters/100) == 0 or i == (num_iters-1):
      w_history.append(w_in.tolist())
      b_history.append(b_in)
      print(f"Iteration {i:4}: Cost {J_history[-1]}")
    # # Print cost every at intervals 10 times or as many iterations if < 10
    # if i % math.ceil(num_iters/10) == 0 or i == (num_iters-1):
    #   file = open("logs", "a") 
    #   content = "{" + "\n" + "\"w\" : " + str(w_history) + ", " + "\n" + "\n" 
    #   file.write(content)
    #   content = "\"b\" : " + str(b_history) + ", " + "\n" + "\n"
    #   file.write(content)
    #   content = "\"J\" : " + str(J_history) + "\n" + "}" + ", " + "\n" + "\n" 
    #   file.write(content)

    #   file.close()
      
  return w_in, b_in, J_history, w_history #return w and J,w history for graphing