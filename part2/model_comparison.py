import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

class RidgeRegression:
    """Ridge Regression từ scratch"""
    def __init__(self, lambda_=1.0, learning_rate=0.01, n_iterations=1000):
        self.lambda_ = lambda_
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            y_pred = self.predict(X)
            dw = (1/n_samples) * (X.T.dot(y_pred - y)) + (self.lambda_/n_samples) * self.weights
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict(self, X):
        return X.dot(self.weights) + self.bias

class LassoRegression:
    """Lasso Regression từ scratch (dùng subgradient)"""
    def __init__(self, lambda_=1.0, learning_rate=0.01, n_iterations=1000):
        self.lambda_ = lambda_
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            y_pred = self.predict(X)
            dw = (1/n_samples) * (X.T.dot(y_pred - y)) + self.lambda_ * np.sign(self.weights)
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict(self, X):
        return X.dot(self.weights) + self.bias

def cross_validate_model(model_class, X, y, lambda_values, k=5):
    """
    K-fold cross-validation để tìm lambda tối ưu
    Trả về: dict {lambda: avg_val_score}
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results = {}

    for lam in lambda_values:
        val_rmse_scores = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = model_class(lambda_=lam)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            val_rmse_scores.append(rmse)

        results[lam] = np.mean(val_rmse_scores)

    # Tìm lambda tối ưu (RMSE nhỏ nhất)
    optimal_lambda = min(results, key=results.get)
    return results, optimal_lambda

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Đánh giá mô hình trên tập test"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'y_pred': y_pred
    }

def compare_models(models_dict, X_train, y_train, X_test, y_test):
    """So sánh nhiều mô hình"""
    results = {}
    for name, model in models_dict.items():
        eval_result = evaluate_model(model, X_train, y_train, X_test, y_test)
        results[name] = {k: v for k, v in eval_result.items() if k != 'y_pred'}
    return pd.DataFrame(results).T