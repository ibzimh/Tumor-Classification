import numpy as np
import matplotlib as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from data import houses_lr_input, houses_lr_output, houses_lr_features

X_train = houses_lr_input
y_train = houses_lr_output

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_train, y_train)

y_pred_sgd = sgdr.predict(X_norm)