import numpy as np
import matplotlib.pyplot as plt

def load_data():
  data = np.loadtxt('ex2data2.txt', delimiter=',')
  X = data[:,:2]
  y = data[:,2]
  return X, y

def plot_data(X, y):
  plot_data_helper(X, y)
  plt.show()

def plot_cost(J):
  l = list(np.linspace(0, len(J), len(J)))
  plt.plot(J, l)
  plt.show()

def plot_data_helper(X, y):
  neg = ( 0 == y )
  pos = ( 1 == y )
  plt.scatter(X[pos, 0], X[pos, 1], label="Admitted")
  plt.scatter(X[neg, 0], X[neg, 1], label="Not Admitted")
  plt.legend()

def plot_data_helper_2(X, y):
  neg = ( 0 == y )
  pos = ( 1 == y )
  fig, (ax1, ax2, ax3) = plt.subplots(3, 9)
  axs = [ax1, ax2, ax3]
  b=-1
  for a in axs:
    b += 1
    for i in range(9):
      a[i].scatter(X[pos, b], X[pos, i+3*b], label="Admitted")
      a[i].scatter(X[neg, b], X[neg, i+3*b], label="Not Admitted")
  plt.legend()
  return axs

def plot_db(X, y, w, b):
  # axs = plot_data_helper_2(X, y)
  plot_data_helper(X, y)
  
  if X.shape[1] <= 2:
    plot_x = np.array([min(X[:, 0]), max(X[:, 0])])
    plot_y = (-1. / w[1]) * (w[0] * plot_x + b)
    
    plt.plot(plot_x, plot_y, c="b")
    
  else:
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    
    z = np.zeros((len(u), len(v)))

    # Evaluate z = theta*x over the grid
    for i in range(len(u)):
      for j in range(len(v)):
        z[i,j] = sig(np.dot(map_feature(u[i], v[j]), w) + b)
    
    # important to transpose z before calling contour       
    z = z.T
    
    # Plot z = 0
    # for a in axs:
    #   for i in range(9):
    #     a[i].contour(u,v,z, levels = [0.5], colors="g")
    plt.contour(u,v,z, levels = [0.5], colors="g")

  plt.show()

def sig(z):
 
    return 1/(1+np.exp(-z))

def map_feature(X1, X2):
    """
    Feature mapping function to polynomial features    
    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((X1**(i-j) * (X2**j)))
    return np.stack(out, axis=1)