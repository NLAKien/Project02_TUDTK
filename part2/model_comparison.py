import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


class RidgeRegression:
    """Ridge Regression từ scratch với learning rate decay"""
    def __init__(self, lambda_=1.0, learning_rate=0.1, n_iterations=3000,
                 decay=1e-4, tol=1e-6):
        self.lambda_ = lambda_
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.decay = decay          # lr_t = lr / (1 + decay * t)
        self.tol = tol              # early stopping khi ||grad|| < tol
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for t in range(self.n_iter):
            y_pred = self.predict(X)
            residual = y_pred - y

            # Gradient MSE + L2 penalty (đều scale 1/n_samples)
            dw = (1/n_samples) * X.T.dot(residual) + (self.lambda_/n_samples) * self.weights
            db = (1/n_samples) * np.sum(residual)

            # Early stopping
            if np.linalg.norm(dw) < self.tol and abs(db) < self.tol:
                break

            # Learning rate decay
            lr_t = self.lr / (1 + self.decay * t)
            self.weights -= lr_t * dw
            self.bias    -= lr_t * db

        return self

    def predict(self, X):
        return X.dot(self.weights) + self.bias


class LassoRegression:
    """Lasso Regression từ scratch (subgradient + learning rate decay)"""
    def __init__(self, lambda_=1.0, learning_rate=0.1, n_iterations=3000,
                 decay=1e-4, tol=1e-6):
        self.lambda_ = lambda_
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.decay = decay
        self.tol = tol
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for t in range(self.n_iter):
            y_pred = self.predict(X)
            residual = y_pred - y

            # Gradient MSE + subgradient L1 (đều scale 1/n_samples)
            dw = (1/n_samples) * X.T.dot(residual) + (self.lambda_/n_samples) * np.sign(self.weights)
            db = (1/n_samples) * np.sum(residual)

            if np.linalg.norm(dw) < self.tol and abs(db) < self.tol:
                break

            lr_t = self.lr / (1 + self.decay * t)
            self.weights -= lr_t * dw
            self.bias    -= lr_t * db

        return self

    def predict(self, X):
        return X.dot(self.weights) + self.bias


class OLSRegression:
    """OLS Regression từ scratch sử dụng Normal Equation (lstsq)"""
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # lstsq ổn định hơn inv khi ma trận gần suy biến
        theta_best, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
        self.bias    = theta_best[0]
        self.weights = theta_best[1:]
        return self

    def predict(self, X):
        return X.dot(self.weights) + self.bias


class OLSFeatureSelection:
    """
    OLS kết hợp Forward Stepwise Selection để chọn k biến tốt nhất.

    Thuật toán Forward Stepwise:
        1. Bắt đầu với tập rỗng.
        2. ửe lượt: thêm feature nào làm giảm RSS nhiều nhất (hay tăng R² nhiều nhất)
           trên tập huấn luyện.
        3. Dừng khi đạt đủ top_k biến.

    Ưu điểm so với correlation filter:
        - Không chọn 2 biến gần giống nhau (tránh multicollinearity).
        - Mỗi bước chọn biến có đóng góp cần biên lớn nhất cho tập hiện tại.
    """
    def __init__(self, top_k=3):
        self.top_k = top_k
        self.selected_indices = None   # thứ tự được chọn (numpy array)
        self.weights = None
        self.bias = None

    def _ols_rss(self, X_sub, y):
        """Tính RSS của OLS trên X_sub (dùng lstsq)."""
        X_b = np.c_[np.ones(len(X_sub)), X_sub]
        theta, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
        residual = y - X_b.dot(theta)
        return residual.dot(residual)

    def fit(self, X, y):
        n_features = X.shape[1]
        top_k = min(self.top_k, n_features)

        remaining = list(range(n_features))
        selected  = []

        for _ in range(top_k):
            best_rss, best_feat = np.inf, None
            for feat in remaining:
                candidate = selected + [feat]
                rss = self._ols_rss(X[:, candidate], y)
                if rss < best_rss:
                    best_rss, best_feat = rss, feat
            selected.append(best_feat)
            remaining.remove(best_feat)

        self.selected_indices = np.array(selected)

        # Huấn luyện OLS trên tập biến đã chọn (dùng lstsq để ổn định)
        X_sel = X[:, self.selected_indices]
        X_b   = np.c_[np.ones(len(X_sel)), X_sel]
        theta, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
        self.bias    = theta[0]
        self.weights = theta[1:]
        return self

    def predict(self, X):
        X_sel = X[:, self.selected_indices]
        return X_sel.dot(self.weights) + self.bias

    @property
    def selected_features(self):
        """Trả về danh sách chỉ số biến được chọn (sorted)."""
        return sorted(self.selected_indices.tolist())


class BayesianLinearRegression:
    """
    Bayesian Linear Regression (Bishop PRML, Ch.3).

    Tham số:
        alpha : precision của prior Gaussian  p(w) = N(0, alpha^{-1} I)
        beta  : precision của noise  p(y|x,w) = N(w^T x, beta^{-1})

    Sau khi fit:
        w_mu  : posterior mean  m_N  = beta * S_N * X^T * y
        w_cov : posterior covariance  S_N = (alpha*I + beta*X^T*X)^{-1}
    """
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta  = beta
        self.w_mu  = None
        self.w_cov = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        n_features = X_b.shape[1]

        # S_N^{-1} = alpha*I + beta * X^T * X
        precision = self.alpha * np.eye(n_features) + self.beta * X_b.T.dot(X_b)

        # Posterior covariance S_N = inv(precision)
        # Dùng solve thay inv để ổn định số học:
        #   S_N = precision^{-1}  <==>  precision @ S_N = I
        self.w_cov = np.linalg.solve(precision, np.eye(n_features))

        # Posterior mean: m_N = beta * S_N * X^T * y
        # Tính qua solve: precision @ m_N = beta * X^T * y
        rhs = self.beta * X_b.T.dot(y)
        self.w_mu = np.linalg.solve(precision, rhs)

        return self

    def predict(self, X, return_std=False):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Kỳ vọng hậu nghiệm: E[y] = X_b @ m_N
        y_pred = X_b.dot(self.w_mu)

        if return_std:
            # Phương sai dự đoán: Var[y] = 1/beta + x^T S_N x
            y_var = 1.0/self.beta + np.sum(X_b.dot(self.w_cov) * X_b, axis=1)
            return y_pred, np.sqrt(np.maximum(y_var, 0))  # clip âm do lỗi số học

        return y_pred


# ---------------------------------------------------------------------------

def cross_validate_model(model_class, X, y, lambda_values, k=5,
                         lr_values=None, n_iterations=3000):
    """
    K-fold cross-validation để tìm (lambda, learning_rate) tối ưu.

    Tham số:
        lambda_values : danh sách các giá trị lambda cần thử
        lr_values     : danh sách learning_rate cần thử
                        (mặc định [0.01, 0.05, 0.1, 0.5])
        n_iterations  : số vòng lặp tối đa cho gradient descent

    Trả về:
        results        : dict { (lambda, lr): avg_val_rmse }
        optimal_lambda : lambda cho RMSE validation nhỏ nhất
        optimal_lr     : learning_rate tương ứng
    """
    if lr_values is None:
        raise ValueError("lr_values must be provided for Ridge and Lasso")

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results = {}

    for lam in lambda_values:
        for lr in lr_values:
            val_rmse_scores = []
            for train_idx, val_idx in kf.split(X):
                X_tr_cv, X_val_cv = X[train_idx], X[val_idx]
                y_tr_cv, y_val_cv = y[train_idx], y[val_idx]

                model = model_class(lambda_=lam, learning_rate=lr,
                                    n_iterations=n_iterations)
                model.fit(X_tr_cv, y_tr_cv)
                y_pred_cv = model.predict(X_val_cv)
                rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
                val_rmse_scores.append(rmse)

            results[(lam, lr)] = np.mean(val_rmse_scores)

    optimal_key = min(results, key=results.get)
    optimal_lambda, optimal_lr = optimal_key
    return results, optimal_lambda, optimal_lr


def cross_validate_ols(X, y, k=5):
    """K-fold cross-validation cho OLS (không có hyperparameter)"""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    val_rmse_scores = []
    for train_idx, val_idx in kf.split(X):
        model = OLSRegression()
        model.fit(X[train_idx], y[train_idx])
        rmse = np.sqrt(mean_squared_error(y[val_idx], model.predict(X[val_idx])))
        val_rmse_scores.append(rmse)
    return np.mean(val_rmse_scores)


def cross_validate_ols_fs(X, y, top_k_values=None, k=5):
    """
    K-fold cross-validation để tìm top_k tối ưu cho OLSFeatureSelection.

    Tham số:
        top_k_values : danh sách số biến cần thử
                       (mặc định: từ 1 đến số cột của X)

    Trả về:
        results       : dict { top_k: avg_val_rmse }
        optimal_top_k : top_k cho CV-RMSE nhỏ nhất
    """
    if top_k_values is None:
        top_k_values = list(range(1, X.shape[1] + 1))

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results = {}

    for top_k in top_k_values:
        val_rmse_scores = []
        for train_idx, val_idx in kf.split(X):
            model = OLSFeatureSelection(top_k=top_k)
            model.fit(X[train_idx], y[train_idx])
            y_pred_cv = model.predict(X[val_idx])
            rmse = np.sqrt(mean_squared_error(y[val_idx], y_pred_cv))
            val_rmse_scores.append(rmse)
        results[top_k] = np.mean(val_rmse_scores)

    optimal_top_k = min(results, key=results.get)
    return results, optimal_top_k


def cross_validate_bayesian(X, y, alpha_values=None, beta_values=None, k=5):
    """
    K-fold cross-validation để tìm (alpha, beta) tối ưu cho BayesianLinearRegression.

    Tham số:
        alpha_values : danh sách precision prior cần thử
                       (mặc định [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0])
        beta_values  : danh sách precision noise cần thử
                       (mặc định [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])

    Trả về:
        results       : dict { (alpha, beta): avg_val_rmse }
        optimal_alpha : alpha cho RMSE validation nhỏ nhất
        optimal_beta  : beta tương ứng
    """
    if alpha_values is None:
        alpha_values = [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0]
    if beta_values is None:
        beta_values  = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results = {}

    for alpha in alpha_values:
        for beta in beta_values:
            val_rmse_scores = []
            for train_idx, val_idx in kf.split(X):
                model = BayesianLinearRegression(alpha=alpha, beta=beta)
                model.fit(X[train_idx], y[train_idx])
                y_pred_cv = model.predict(X[val_idx])
                rmse = np.sqrt(mean_squared_error(y[val_idx], y_pred_cv))
                val_rmse_scores.append(rmse)

            results[(alpha, beta)] = np.mean(val_rmse_scores)

    optimal_key   = min(results, key=results.get)
    optimal_alpha, optimal_beta = optimal_key
    return results, optimal_alpha, optimal_beta


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Đánh giá mô hình trên tập test"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'y_pred': y_pred
    }


def compare_models(models_dict, X_train, y_train, X_test, y_test):
    """So sánh nhiều mô hình, trả về DataFrame kết quả"""
    results = {}
    for name, model in models_dict.items():
        eval_result = evaluate_model(model, X_train, y_train, X_test, y_test)
        results[name] = {k: v for k, v in eval_result.items() if k != 'y_pred'}
    return pd.DataFrame(results).T