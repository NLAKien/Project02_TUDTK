"""
model_comparison.py
-------------------
Cài đặt từ đầu:
- OLS (Ordinary Least Squares)
- Ridge Regression + Cross-Validation tìm λ tối ưu
- Lasso Regression (coordinate descent) + CV
- So sánh MAE, RMSE, R² trên test set
- Vẽ 4 biểu đồ phần dư (Residual Analysis)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from itertools import combinations

np.random.seed(42)


# ================================================================
# 1. OLS TỪ ĐẦU
# ================================================================

def ols_fit(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Cài đặt OLS từ đầu.
    β̂ = (XᵀX)⁻¹Xᵀy

    Returns
    -------
    dict với keys: beta, y_hat, residuals, sigma2, rss, tss, r2, r2_adj
    """
    n, p = X.shape

    # Nghiệm OLS
    XtX     = X.T @ X
    Xty     = X.T @ y
    beta    = np.linalg.solve(XtX, Xty)

    y_hat      = X @ beta
    residuals  = y - y_hat
    rss        = np.sum(residuals ** 2)
    tss        = np.sum((y - y.mean()) ** 2)
    r2         = 1 - rss / tss
    p_feat     = p - 1  # Trừ intercept
    r2_adj     = 1 - (n - 1) / (n - p_feat - 1) * (1 - r2)
    sigma2     = rss / (n - p_feat - 1)

    return {
        "beta": beta, "y_hat": y_hat, "residuals": residuals,
        "sigma2": sigma2, "rss": rss, "tss": tss,
        "r2": r2, "r2_adj": r2_adj,
    }


# ================================================================
# 2. RIDGE REGRESSION TỪ ĐẦU
# ================================================================

def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Ridge Regression từ đầu.
    β̂_ridge = (XᵀX + λI)⁻¹Xᵀy
    """
    n, p    = X.shape
    I       = np.eye(p)
    I[0, 0] = 0  # Không regularize intercept
    beta    = np.linalg.solve(X.T @ X + lam * I, X.T @ y)
    return beta


# ================================================================
# 3. LASSO REGRESSION (Coordinate Descent)
# ================================================================

def _soft_threshold(z: float, lam: float) -> float:
    """Toán tử soft-thresholding cho Lasso."""
    if z > lam:
        return z - lam
    elif z < -lam:
        return z + lam
    return 0.0


def lasso_fit(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Lasso Regression bằng Coordinate Descent.
    Không có nghiệm closed-form, giải bằng coordinate descent.
    """
    n, p   = X.shape
    beta   = np.zeros(p)
    beta[0] = y.mean()  # Khởi tạo intercept

    for _ in range(max_iter):
        beta_old = beta.copy()

        for j in range(p):
            # Tính residual không tính đóng góp của cột j
            r_j = y - X @ beta + X[:, j] * beta[j]
            z_j = X[:, j] @ r_j / n

            if j == 0:
                # Intercept không bị regularize
                beta[j] = z_j / (X[:, j] @ X[:, j] / n)
            else:
                beta[j] = _soft_threshold(z_j, lam / 2) / (X[:, j] @ X[:, j] / n)

        if np.max(np.abs(beta - beta_old)) < tol:
            break

    return beta


# ================================================================
# 4. K-FOLD CROSS-VALIDATION TÌM λ TỐI ƯU
# ================================================================

def kfold_cv_ridge(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    k: int = 5,
) -> tuple[float, np.ndarray]:
    """
    K-fold CV để tìm λ tối ưu cho Ridge.

    Returns
    -------
    best_lambda : float
    cv_scores   : np.ndarray, MSE trung bình cho mỗi lambda
    """
    n = len(y)
    fold_size = n // k
    indices   = np.random.permutation(n)

    cv_scores = np.zeros(len(lambdas))

    for i, lam in enumerate(lambdas):
        mse_folds = []

        for fold in range(k):
            val_idx   = indices[fold * fold_size: (fold + 1) * fold_size]
            train_idx = np.concatenate([indices[:fold * fold_size],
                                        indices[(fold + 1) * fold_size:]])

            X_train, y_train = X[train_idx], y[train_idx]
            X_val,   y_val   = X[val_idx],   y[val_idx]

            beta  = ridge_fit(X_train, y_train, lam)
            y_pred = X_val @ beta
            mse_folds.append(np.mean((y_val - y_pred) ** 2))

        cv_scores[i] = np.mean(mse_folds)

    best_lambda = lambdas[np.argmin(cv_scores)]
    return best_lambda, cv_scores


def kfold_cv_lasso(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    k: int = 5,
) -> tuple[float, np.ndarray]:
    """K-fold CV để tìm λ tối ưu cho Lasso."""
    n = len(y)
    fold_size = n // k
    indices   = np.random.permutation(n)
    cv_scores = np.zeros(len(lambdas))

    for i, lam in enumerate(lambdas):
        mse_folds = []
        for fold in range(k):
            val_idx   = indices[fold * fold_size: (fold + 1) * fold_size]
            train_idx = np.concatenate([indices[:fold * fold_size],
                                        indices[(fold + 1) * fold_size:]])

            X_train, y_train = X[train_idx], y[train_idx]
            X_val,   y_val   = X[val_idx],   y[val_idx]

            beta   = lasso_fit(X_train, y_train, lam)
            y_pred = X_val @ beta
            mse_folds.append(np.mean((y_val - y_pred) ** 2))

        cv_scores[i] = np.mean(mse_folds)

    best_lambda = lambdas[np.argmin(cv_scores)]
    return best_lambda, cv_scores


# ================================================================
# 5. METRICS ĐÁNH GIÁ
# ================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Tính MAE, RMSE, R² trên tập test."""
    residuals = y_true - y_pred
    mae  = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    tss  = np.sum((y_true - y_true.mean()) ** 2)
    rss  = np.sum(residuals ** 2)
    r2   = 1 - rss / tss
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# ================================================================
# 6. SO SÁNH 3+ MÔ HÌNH
# ================================================================

def compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lambdas: np.ndarray = None,
    k: int = 5,
) -> pd.DataFrame:
    """
    Huấn luyện OLS, Ridge, Lasso và so sánh trên test set.
    """
    if lambdas is None:
        lambdas = np.logspace(-4, 4, 50)

    results = {}

    # --- OLS ---
    ols_result  = ols_fit(X_train, y_train)
    y_pred_ols  = X_test @ ols_result["beta"]
    results["OLS"] = compute_metrics(y_test, y_pred_ols)

    # --- Ridge với CV ---
    best_lam_ridge, cv_ridge = kfold_cv_ridge(X_train, y_train, lambdas, k)
    beta_ridge = ridge_fit(X_train, y_train, best_lam_ridge)
    y_pred_ridge = X_test @ beta_ridge
    results[f"Ridge (λ={best_lam_ridge:.4f})"] = compute_metrics(y_test, y_pred_ridge)

    # --- Lasso với CV ---
    best_lam_lasso, cv_lasso = kfold_cv_lasso(X_train, y_train, lambdas, k)
    beta_lasso = lasso_fit(X_train, y_train, best_lam_lasso)
    y_pred_lasso = X_test @ beta_lasso
    results[f"Lasso (λ={best_lam_lasso:.4f})"] = compute_metrics(y_test, y_pred_lasso)

    df_results = pd.DataFrame(results).T.round(4)
    print("\n=== SO SÁNH MÔ HÌNH TRÊN TEST SET ===")
    print(df_results.to_string())

    # Vẽ CV curves
    _plot_cv_curves(lambdas, cv_ridge, best_lam_ridge, cv_lasso, best_lam_lasso)

    return df_results, {
        "ols": ols_result["beta"],
        "ridge": beta_ridge,
        "lasso": beta_lasso,
        "best_lam_ridge": best_lam_ridge,
        "best_lam_lasso": best_lam_lasso,
    }


def _plot_cv_curves(lambdas, cv_ridge, best_lam_ridge, cv_lasso, best_lam_lasso):
    """Vẽ đường cong CV MSE theo λ cho Ridge và Lasso."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, cv_scores, best_lam, title in zip(
        axes,
        [cv_ridge, cv_lasso],
        [best_lam_ridge, best_lam_lasso],
        ["Ridge", "Lasso"],
    ):
        ax.semilogx(lambdas, cv_scores, color="#E8623A", linewidth=2)
        ax.axvline(best_lam, color="blue", linestyle="--",
                   label=f"Best λ = {best_lam:.4f}")
        ax.set_xlabel("λ (log scale)")
        ax.set_ylabel("CV MSE")
        ax.set_title(f"{title}: Cross-Validation MSE vs λ")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("part2_cv_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[Saved] part2_cv_curves.png")


# ================================================================
# 7. 4 BIỂU ĐỒ PHẦN DƯ (RESIDUAL ANALYSIS)
# ================================================================

def residual_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "OLS",
):
    """
    Vẽ 4 biểu đồ chẩn đoán phần dư chuẩn:
    1. Residuals vs Fitted
    2. Normal Q-Q Plot
    3. Scale-Location
    4. Cook's Distance (approximate)
    """
    residuals  = y_true - y_pred
    std_resid  = (residuals - residuals.mean()) / residuals.std()
    n          = len(residuals)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Residual Diagnostics — {model_name}", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # --- Plot 1: Residuals vs Fitted ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_pred, residuals, alpha=0.5, color="#E8623A", s=20)
    ax1.axhline(0, color="black", linestyle="--", linewidth=1)
    ax1.set_xlabel("Fitted Values (ŷ)")
    ax1.set_ylabel("Residuals (y − ŷ)")
    ax1.set_title("1. Residuals vs Fitted")
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Normal Q-Q ---
    ax2 = fig.add_subplot(gs[0, 1])
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals, dist="norm")
    ax2.scatter(osm, osr, alpha=0.5, color="#E8623A", s=20)
    ax2.plot(osm, slope * np.array(osm) + intercept, color="black", linewidth=1.5)
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles")
    ax2.set_title("2. Normal Q-Q Plot")
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Scale-Location ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(y_pred, np.sqrt(np.abs(std_resid)), alpha=0.5, color="#E8623A", s=20)
    ax3.set_xlabel("Fitted Values (ŷ)")
    ax3.set_ylabel("√|Standardized Residuals|")
    ax3.set_title("3. Scale-Location (Homoscedasticity)")
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: Cook's Distance (approximate) ---
    ax4 = fig.add_subplot(gs[1, 1])
    cooks_d = (std_resid ** 2) / (2 * residuals.std() ** 2 + 1e-8)
    ax4.stem(range(n), cooks_d, markerfmt="o", linefmt="C1-", basefmt="k-")
    ax4.axhline(4 / n, color="red", linestyle="--", label=f"Threshold 4/n={4/n:.3f}")
    ax4.set_xlabel("Observation Index")
    ax4.set_ylabel("Cook's Distance")
    ax4.set_title("4. Cook's Distance (Influential Points)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.savefig(f"part2_residual_{model_name.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Saved] part2_residual_{model_name.lower()}.png")


# ================================================================
# 8. RIDGE TRACE
# ================================================================

def plot_ridge_trace(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lambdas: np.ndarray,
    feature_names: list = None,
):
    """Vẽ Ridge Trace — hệ số β thay đổi theo λ."""
    p       = X_train.shape[1]
    betas   = np.zeros((len(lambdas), p))

    for i, lam in enumerate(lambdas):
        betas[i] = ridge_fit(X_train, y_train, lam)

    fig, ax = plt.subplots(figsize=(12, 6))
    for j in range(1, p):  # Bỏ intercept
        label = feature_names[j] if feature_names and j < len(feature_names) else f"β{j}"
        ax.semilogx(lambdas, betas[:, j], label=label)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("λ (log scale)")
    ax.set_ylabel("Hệ số β")
    ax.set_title("Ridge Trace: Hệ số hồi quy theo λ")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig("part2_ridge_trace.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[Saved] part2_ridge_trace.png")


# ================================================================
# UNIT TEST
# ================================================================

def _test_models():
    """Unit test cho OLS, Ridge, Lasso."""
    np.random.seed(42)
    n, p = 100, 5
    X    = np.random.randn(n, p)
    X    = np.hstack([np.ones((n, 1)), X])  # Thêm intercept
    beta_true = np.array([1.0, 2.0, -1.5, 0.5, 3.0, -0.8])
    y    = X @ beta_true + np.random.randn(n) * 0.5

    # Test OLS
    result = ols_fit(X, y)
    assert abs(result["r2"] - 1.0) < 0.05, "Test OLS R² FAILED"
    print(f"Test OLS PASSED: R² = {result['r2']:.4f}")

    # Test Ridge
    beta_ridge = ridge_fit(X, y, lam=0.01)
    assert beta_ridge.shape == beta_true.shape, "Test Ridge shape FAILED"
    print(f"Test Ridge PASSED: β̂[1] = {beta_ridge[1]:.4f} (true: {beta_true[1]})")

    # Test Lasso
    beta_lasso = lasso_fit(X, y, lam=0.01)
    assert beta_lasso.shape == beta_true.shape, "Test Lasso shape FAILED"
    print(f"Test Lasso PASSED: β̂[1] = {beta_lasso[1]:.4f} (true: {beta_true[1]})")

    # Test CV
    lambdas = np.logspace(-3, 2, 20)
    best_lam, _ = kfold_cv_ridge(X, y, lambdas, k=5)
    assert best_lam > 0, "Test CV Ridge FAILED"
    print(f"Test CV Ridge PASSED: best λ = {best_lam:.4f}")

    print("\nAll tests PASSED!")


if __name__ == "__main__":
    _test_models()