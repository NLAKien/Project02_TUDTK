"""
advanced_methods.py
-------------------
Kỹ thuật nâng cao (Bonus):
1. Kernel Ridge Regression (RBF kernel)
2. Bayesian Linear Regression (conjugate prior)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)


# ================================================================
# 1. KERNEL RIDGE REGRESSION
# ================================================================

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float = 1.0) -> np.ndarray:
    """
    RBF (Gaussian) kernel:
    k(x, x') = exp(-||x - x'||² / (2ℓ²))
    """
    # Tính ||x - x'||² hiệu quả
    X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
    X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
    sq_dist = X1_sq + X2_sq.T - 2 * X1 @ X2.T
    return np.exp(-sq_dist / (2 * length_scale ** 2))


def polynomial_kernel(X1: np.ndarray, X2: np.ndarray, degree: int = 2, c: float = 1.0) -> np.ndarray:
    """
    Polynomial kernel:
    k(x, x') = (xᵀx' + c)^d
    """
    return (X1 @ X2.T + c) ** degree


class KernelRidgeRegression:
    """
    Kernel Ridge Regression:
    ŷ(x) = k(x)ᵀ(K + λI)⁻¹y

    Mở rộng OLS sang không gian đặc trưng phi tuyến qua kernel trick.
    """

    def __init__(self, kernel: str = "rbf", lam: float = 1.0, length_scale: float = 1.0, degree: int = 2):
        self.kernel       = kernel
        self.lam          = lam
        self.length_scale = length_scale
        self.degree       = degree
        self._X_train     = None
        self._alpha       = None

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.kernel == "rbf":
            return rbf_kernel(X1, X2, self.length_scale)
        elif self.kernel == "poly":
            return polynomial_kernel(X1, X2, self.degree)
        else:
            raise ValueError(f"Kernel '{self.kernel}' không hỗ trợ. Chọn: 'rbf', 'poly'")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KernelRidgeRegression":
        """
        Giải: α = (K + λI)⁻¹y
        với K_ij = k(x_i, x_j)
        """
        self._X_train = X.copy()
        K             = self._compute_kernel(X, X)
        n             = len(y)
        self._alpha   = np.linalg.solve(K + self.lam * np.eye(n), y)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """ŷ(x) = k(x)ᵀα"""
        K_test = self._compute_kernel(X_test, self._X_train)
        return K_test @ self._alpha

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """R² trên tập test."""
        y_pred = self.predict(X_test)
        rss    = np.sum((y_test - y_pred) ** 2)
        tss    = np.sum((y_test - y_test.mean()) ** 2)
        return 1 - rss / tss


# ================================================================
# 2. BAYESIAN LINEAR REGRESSION
# ================================================================

class BayesianLinearRegression:
    """
    Bayesian Linear Regression với conjugate Gaussian prior:

    Prior:    β ~ N(m₀, S₀)
    Likelihood: y | X, β ~ N(Xβ, σ²I)

    Posterior:
        Sₙ = (S₀⁻¹ + (1/σ²)XᵀX)⁻¹
        mₙ = Sₙ(S₀⁻¹m₀ + (1/σ²)Xᵀy)

    Ưu điểm: Cung cấp uncertainty quantification qua credible intervals.
    """

    def __init__(self, sigma2: float = 1.0, prior_precision: float = 1.0):
        """
        Parameters
        ----------
        sigma2 : float
            Phương sai nhiễu (giả sử biết trước)
        prior_precision : float
            Độ chính xác của prior (= 1/α, α là phương sai prior)
        """
        self.sigma2          = sigma2
        self.prior_precision = prior_precision
        self._m_n            = None  # Posterior mean
        self._S_n            = None  # Posterior covariance
        self._fitted         = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianLinearRegression":
        """
        Tính posterior:
        Sₙ = (αI + (1/σ²)XᵀX)⁻¹
        mₙ = (1/σ²)Sₙ Xᵀy
        """
        n, p = X.shape
        alpha = self.prior_precision

        # Prior: m₀ = 0, S₀ = (1/α)I
        S0_inv = alpha * np.eye(p)

        # Posterior precision
        S_n_inv   = S0_inv + (1 / self.sigma2) * X.T @ X
        self._S_n = np.linalg.inv(S_n_inv)

        # Posterior mean (m₀ = 0 nên bỏ S₀⁻¹m₀)
        self._m_n = (1 / self.sigma2) * self._S_n @ X.T @ y

        self._fitted = True
        return self

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Dự đoán với uncertainty:

        Returns
        -------
        y_mean : np.ndarray — Posterior predictive mean
        y_std  : np.ndarray — Posterior predictive std (credible interval)
        """
        if not self._fitted:
            raise RuntimeError("Gọi fit() trước predict()")

        y_mean = X_test @ self._m_n

        # Phương sai dự đoán: σ²_pred = σ² + xᵀSₙx
        y_var  = np.array([
            self.sigma2 + x @ self._S_n @ x
            for x in X_test
        ])
        y_std = np.sqrt(y_var)

        return y_mean, y_std

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """R² trên tập test."""
        y_pred, _ = self.predict(X_test)
        rss = np.sum((y_test - y_pred) ** 2)
        tss = np.sum((y_test - y_test.mean()) ** 2)
        return 1 - rss / tss

    def plot_credible_intervals(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_points: int = 50,
        title: str = "Bayesian Prediction với Credible Intervals 95%",
    ):
        """Vẽ dự đoán kèm khoảng tin cậy Bayesian 95%."""
        y_mean, y_std = self.predict(X_test)
        idx = np.argsort(y_mean)[:n_points]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.scatter(range(n_points), y_test[idx], color="#2C1810", s=20,
                   label="Thực tế", zorder=3)
        ax.plot(range(n_points), y_mean[idx], color="#E8623A", linewidth=2,
                label="Posterior Mean")
        ax.fill_between(
            range(n_points),
            y_mean[idx] - 1.96 * y_std[idx],
            y_mean[idx] + 1.96 * y_std[idx],
            alpha=0.25, color="#E8623A", label="95% Credible Interval",
        )
        ax.set_xlabel("Quan sát (sắp xếp theo ŷ)")
        ax.set_ylabel("Giá trị")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("part2_bayesian_prediction.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("[Saved] part2_bayesian_prediction.png")


# ================================================================
# SO SÁNH KERNEL vs OLS vs BAYESIAN
# ================================================================

def compare_advanced_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    sigma2:  float = 1.0,
) -> None:
    """So sánh Kernel Ridge, Bayesian và OLS thông thường."""
    from model_comparison import ols_fit, compute_metrics

    results = {}

    # OLS
    ols_res      = ols_fit(X_train, y_train)
    y_pred_ols   = X_test @ ols_res["beta"]
    results["OLS"] = compute_metrics(y_test, y_pred_ols)

    # Kernel Ridge (RBF)
    krr = KernelRidgeRegression(kernel="rbf", lam=1.0, length_scale=1.0)
    krr.fit(X_train[:, 1:], y_train)  # Bỏ cột intercept cho kernel
    y_pred_krr = krr.predict(X_test[:, 1:])
    results["Kernel Ridge (RBF)"] = compute_metrics(y_test, y_pred_krr)

    # Bayesian
    blr = BayesianLinearRegression(sigma2=sigma2, prior_precision=1.0)
    blr.fit(X_train, y_train)
    y_pred_blr, _ = blr.predict(X_test)
    results["Bayesian LR"] = compute_metrics(y_test, y_pred_blr)

    import pandas as pd
    df = pd.DataFrame(results).T.round(4)
    print("\n=== SO SÁNH MÔ HÌNH NÂNG CAO ===")
    print(df.to_string())

    blr.plot_credible_intervals(X_test, y_test)


# ================================================================
# UNIT TEST
# ================================================================

def _test_advanced():
    """Unit test cho Kernel Ridge và Bayesian LR."""
    np.random.seed(42)
    n = 100

    X = np.random.randn(n, 3)
    X_int = np.hstack([np.ones((n, 1)), X])
    y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n) * 0.3

    # Test Kernel Ridge
    krr = KernelRidgeRegression(kernel="rbf", lam=0.1)
    krr.fit(X, y)
    r2_krr = krr.score(X, y)
    assert r2_krr > 0.8, f"Test Kernel Ridge FAILED: R²={r2_krr:.4f}"
    print(f"Test Kernel Ridge PASSED: R² = {r2_krr:.4f}")

    # Test Bayesian LR
    blr = BayesianLinearRegression(sigma2=0.09, prior_precision=1.0)
    blr.fit(X_int, y)
    r2_blr = blr.score(X_int, y)
    assert r2_blr > 0.8, f"Test Bayesian FAILED: R²={r2_blr:.4f}"
    print(f"Test Bayesian LR PASSED: R² = {r2_blr:.4f}")

    print("\nAll advanced tests PASSED!")


if __name__ == "__main__":
    _test_advanced()