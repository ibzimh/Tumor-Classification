import math

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
    if i% math.ceil(num_iters / 10) == 0:
      print(f"Iteration {i:4d}: Cost {J_history[-1]}")

  return w, b, J_history