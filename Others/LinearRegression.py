import math, copy
import matplotlib.pyplot as plt
import numpy as np

from f_wb import f_wb_lin, polynomialify
from cost_function import sq_err_cost_function
from gradient_function import lr_gradient_function
from gradient_descent import gradient_descent
from normalise_features import zscore_normalise_features
import data

X_train = data.houses_lr_input
y_train = data.houses_lr_output
X_features = data.houses_lr_features

X_norm, mean, std = zscore_normalise_features(X_train)

b_init = 0
w_init = np.zeros(X_train.shape[1])
alpha = 9e-7
alpha_norm = 1.0e-1
iterations = 1000

w_final, b_final, J_hist = gradient_descent(X_train, y_train, w_init, b_init, sq_err_cost_function, lr_gradient_function, alpha, iterations)
print("")
w_norm, b_norm, J_norm_hist = gradient_descent(X_norm, y_train, w_init, b_init, sq_err_cost_function, lr_gradient_function, alpha_norm, iterations)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")

# m,_ = X_train.shape
# for i in range(m):
#     print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

# fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
# for i in range(len(ax)):
#     ax[i].scatter(X_train[:,i],y_train)
#     ax[i].set_xlabel(X_features[i])
# ax[0].set_ylabel("Price (1000's)")
# plt.show()

# mu     = np.mean(X_train,axis=0)
# sigma  = np.std(X_train,axis=0)
# X_mean = (X_train - mu)
# X_norm = (X_train - mu)/sigma

fig,ax=plt.subplots(1, 2, figsize=(12, 3))
ax[0].scatter(X_train[:,0], X_train[:,3])
ax[0].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3])
ax[0].set_title("unnormalized")
ax[0].axis('equal')

ax[1].scatter(X_norm[:,0], X_norm[:,3])
ax[1].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3])
ax[1].set_title(r"Z-score normalized")
ax[1].axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle("distribution of features before, during, after normalization")
plt.show()