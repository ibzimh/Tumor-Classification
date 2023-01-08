import numpy as np

def zscore_normalise_features(X):
  std = np.std(X, axis=0) # std for each feature (shape(n,))
  mean = np.mean(X, axis=0) # mean for each feature (shape(n,))
  X_norm = (X - mean) / std
  return X_norm, mean, std