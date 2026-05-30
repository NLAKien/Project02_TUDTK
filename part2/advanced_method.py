import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import solve

class KernelRidgeRegression:
    """
    Kernel Ridge Regression dùng RBF kernel
    """
    def __init__(self, lambda_=1.0, gamma=0.1):
        self.lambda_ = lambda_
        self.gamma = gamma
        self.alpha = None
        self.X_train = None

    def fit(self, X, y):
        self.X_train = X
        n = X.shape[0]
        K = rbf_kernel(X, X, gamma=self.gamma)
        self.alpha = solve(K + self.lambda_ * n * np.eye(n), y)
        return self

    def predict(self, X):
        K_test = rbf_kernel(X, self.X_train, gamma=self.gamma)
        return K_test.dot(self.alpha)

class BayesianLinearRegression:
    """
    Bayesian Linear Regression đơn giản (conjugate prior)
    """
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha  # Prior precision
        self.beta = beta    # Noise precision
        self.mean_w = None
        self.cov_w = None

    def fit(self, X, y):
        # Posterior: N(mean, cov)
        XTX = X.T @ X
        self.cov_w = np.linalg.inv(self.alpha * np.eye(X.shape[1]) + self.beta * XTX)
        self.mean_w = self.beta * (self.cov_w @ (X.T @ y))
        return self

    def predict(self, X):
        return X @ self.mean_w

    def predict_with_uncertainty(self, X):
        y_mean = self.predict(X)
        y_var = 1/self.beta + np.diag(X @ self.cov_w @ X.T)
        return y_mean, y_var