import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from f_wb import f_wb_log
from cost_function import log_cost_function
from gradient_function import log_gradient_function
from gradient_descent import gradient_descent
from normalise_features import zscore_normalise_features

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# X_norm, mean, std = zscore_normalise_features(X_train)

w_in = np.zeros((2))
b_in = 0.
alpha = 0.1
iterations = 10000

w_final, b_final, J_hist = gradient_descent(X_train, y_train, w_in, b_in, log_cost_function, log_gradient_function, alpha, iterations)
my_pred = f_wb_log(X_train, w_final, b_final)

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
skl_pred = lr_model.predict(X_train)
print(skl_pred.tolist())
print(my_pred.tolist())

# pos = y_train == 1
# neg = y_train == 0

# fig,ax = plt.subplots(1, 1, figsize=(5,3))
# ax.scatter(X[:,0], y, c='b', marker='o', label='y=0')
# ax.scatter(X[:,0], X[:,1], c='r', marker='x', label='y=1')
# ax.legend()
# plt.show()