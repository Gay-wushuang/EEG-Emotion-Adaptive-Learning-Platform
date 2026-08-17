from sklearn.preprocessing import StandardScaler
import numpy as np


def standardize_3d_features(X_train, X_test):
    n_samples_train, T, F = X_train.shape
    n_samples_test = X_test.shape[0]
    X_train_flat = X_train.reshape(-1, F)
    X_test_flat = X_test.reshape(-1, F)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    X_train_scaled = X_train_scaled.reshape(n_samples_train, T, F)
    X_test_scaled = X_test_scaled.reshape(n_samples_test, T, F)
    return X_train_scaled, X_test_scaled, scaler
