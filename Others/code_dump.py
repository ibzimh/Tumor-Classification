# return array of predicted values
def compute_f_wb(x, w, b):
  m = x.shape[0]
  f_wb = np.zeros(m)
  for i in range(m):
    f_wb[i] = w * x[i] + b
  return f_wb

# computes the cost of a set of w's and b's
def compute_cost(x, y, w, b):
  cost_sum = 0
  m = x.shape[0]
  f_wb = compute_f_wb(x, w, b)
  for i in range(m):
    cost_sum += (f_wb[i] - y[i]) ** 2
  return (1 / (2 * m)) * cost_sum

# returns the partial derivatives of the cost function for graident descent 
def compute_gradient(x, y, w, b):
  f_wb = compute_f_wb(x, w, b)
  dj_dw = 0
  dj_db = 0
  m = x.shape[0]

  for i in range (m):
    dj_dw += (f_wb[i] - y[i]) * (x[i])
    dj_db += (f_wb[i] - y[i])
  dj_dw *= (1/m)
  dj_db *= (1/m)

  return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
  J_history = []
  p_history = []
  w = w_in
  b = b_in

  for i in range(num_iters):
    dj_dw, dj_db = gradient_function(x, y, w, b)
    w = w - alpha * dj_dw
    b = b - alpha * dj_db
    if i < 100000:
      J_history.append(cost_function(x, y, w, b))
      p_history.append([w, b])
  return w, b, J_history, p_history

# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

# plt.plot(J_hist, list(map(lambda arr: arr[0], p_hist)), c='b', label='gradient descent')
# plt.show()

plt.scatter(x_train, y_train)
plt.show()

# fig = plt.figure()
# ax = plt.axes(projection ='3d')
# x = list(map(lambda arr: arr[0], p_hist)) # w
# y = list(map(lambda arr: arr[1], p_hist)) # b
# z = J_hist
# ax.plot3D(x, y, z, 'magenta')
# ax.set_title("3d gradient descent")
# plt.show()
# plt.plot(x_train, tmp_f_wb, c='b', label="Prediction")

# plt.scatter(x_train, y_train, marker='x', c='r', label="Actual Values")
# plt.title("Size in sq feet")
# plt.legend()
# plt.show()

# fig = plt.figure()
# ax = plt.axes(projection ='3d')
# x = w_hist
# y = b_hist
# z = J_hist
# ax.plot3D(x, y, z, 'blue', label='gradient descent')
# ax.set_xlabel('w[0]')
# ax.set_ylabel('w[1]')
# ax.set_zlabel('csot')
# plt.legend()
# plt.show()
# # plt.plot(x_train, tmp_f_wb, c='b', label="Prediction")









import math, copy
import matplotlib.pyplot as plt
import numpy as np

def f_wb(x, w, b):
  return np.dot(x, w) + b

def compute_cost(X, y, w, b):
  m = X.shape[0]
  cost = 0.0
  for i in range(m):
    cost += ( f_wb(X[i], w, b) - y[i] ) ** 2
  return cost / (2 * m)

def compute_gradient(X, y, w, b):
  m,n = X.shape # number of examples, number of features
  dj_dw = np.zeros((n,))
  dj_db = 0
  for i in range(m):
    f_wb_i = f_wb(X[i], w, b)
    
    for j in range(n):
      dj_dw[j] += ( f_wb_i - y[i] ) * X[i, j]

    dj_db += ( f_wb_i - y[i] )
  return (dj_db/m), (dj_dw/m)

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
  w = w_in
  b = b_in
  dj_db = 0
  dj_dw = 0
  J_history = []

  for i in range(num_iters):
    dj_db, dj_dw = gradient_function(X, y, w, b)
    w = w - alpha * dj_dw
    b = b - alpha * dj_db
    if i < 100000:      
      J_history.append(cost_function(X, y, w, b))
    # if i% math.ceil(num_iters / 10) == 0:
    #   print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

  return w, b, J_history

# X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
# y_train = np.array([460, 232, 178])

b_init = 0
w_init = np.zeros(X_train.shape[1])
alpha = 5.0e-7
iterations = 1000

w_final, b_final, J_hist = gradient_descent(X_train, y_train, w_init, b_init, compute_cost, compute_gradient, alpha, iterations)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

# plt.scatter(X_train[:,0], y_train, c='r', marker='x', label='data')
# plt.plot(X_train[:,0], np.dot(X_train, w_final) + b_final, label='predicted')
# plt.legend()
# plt.show()
