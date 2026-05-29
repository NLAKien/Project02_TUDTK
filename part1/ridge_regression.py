"""
6. ridge_fit(X, y, lam) — Ridge Regression (L2 Regularization) + Ridge Trace.

Công thức nghiệm closed-form:
    β̂_ridge = (X^T X + λI)^{-1} X^T y

Thuần logic: chỉ dùng linalg.Matrix / linalg.Vector.
Không dùng sklearn, scipy, statsmodels, hay bất kỳ thư viện ML/stats nào.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

try:
    from .linalg import Matrix, Vector
except ImportError:
    from linalg import Matrix, Vector


# ─────────────────────────────────────────────────────────────────────────────
# Hàm tiện ích nội bộ
# ─────────────────────────────────────────────────────────────────────────────

def _mat_vec_mul(A: Matrix, v: Vector) -> Vector:
    """Nhân ma trận A với vector cột v."""
    n = A.shape[0]
    result = [
        sum(A[i][k] * v[k] for k in range(A.shape[1]))
        for i in range(n)
    ]
    return Vector(result)


def _mat_mul(A: Matrix, B: Matrix) -> Matrix:
    """Nhân hai ma trận A và B."""
    n, m = A.shape[0], B.shape[1]
    K = A.shape[1]
    data = [
        [sum(A[i][k] * B[k][j] for k in range(K)) for j in range(m)]
        for i in range(n)
    ]
    return Matrix(data)


def _transpose(A: Matrix) -> Matrix:
    """Chuyển vị ma trận A."""
    return Matrix([[A[i][j] for i in range(A.shape[0])] for j in range(A.shape[1])])


def _add_ridge_penalty(A: Matrix, lam: float, penalize_intercept: bool = False) -> Matrix:
    """
    Trả về A + λI (hoặc A + λI' nếu không phạt intercept).
    Mặc định: phần tử (0,0) của I' = 0 (không phạt bias).
    """
    n = A.shape[0]
    new_data = [[A[i][j] for j in range(n)] for i in range(n)]
    for i in range(n):
        if i == 0 and not penalize_intercept:
            continue            # không phạt intercept
        new_data[i][i] += lam
    return Matrix(new_data)


def _solve_ridge(Xb: Matrix, y: Vector, lam: float) -> Vector:
    """
    Giải hệ Ridge: β = (X^T X + λI)^{-1} X^T y
    với X đã bao gồm cột intercept ở vị trí 0.
    """
    Xt = _transpose(Xb)
    XtX = _mat_mul(Xt, Xb)                         # (p+1, p+1)
    A = _add_ridge_penalty(XtX, lam)               # (X^T X + λI)
    A_inv = A.inverse()
    if A_inv is None:
        raise ValueError(f"Ma trận (X^TX + {lam}·I) suy biến.")
    Xty = _mat_vec_mul(Xt, y)                      # (p+1,)
    beta = _mat_vec_mul(A_inv, Xty)
    return beta


def _rss(Xb: Matrix, y: Vector, beta: Vector) -> float:
    """Tính RSS = ||y - Xb β||²."""
    n = Xb.shape[0]
    y_hat = _mat_vec_mul(Xb, beta)
    return sum((y[i] - y_hat[i]) ** 2 for i in range(n))


def _add_intercept(X: Matrix) -> Matrix:
    """Thêm cột 1 (intercept) vào đầu ma trận X."""
    n = X.shape[0]
    return Matrix([[1.0] + [X[i][j] for j in range(X.shape[1])] for i in range(n)])


# ─────────────────────────────────────────────────────────────────────────────
# 6. ridge_fit(X, y, lam) — Ridge Regression + Ridge Trace
# ─────────────────────────────────────────────────────────────────────────────

def ridge_fit(X: Matrix, y: Vector, lam):
    """
    Cài đặt Ridge Regression (L2 regularization).

        β̂_ridge = (X^T X + λI)^{-1} X^T y

    Cột intercept được thêm tự động và KHÔNG bị phạt (thông lệ thống kê).

    Tham số
    -------
    X   : Matrix (n, p)       — ma trận đặc trưng (chưa thêm cột bias)
    y   : Vector (n,)         — vector nhãn
    lam : float hoặc list     — tham số điều chỉnh λ (một giá trị hoặc danh sách)

    Trả về
    ------
    - Nếu lam là số   : Vector (p+1,)           — hệ số Ridge (bias + p hệ số)
    - Nếu lam là list : list[Vector] (len_lam,) — danh sách hệ số tương ứng
    Vẽ Ridge Trace + RSS khi lam là danh sách.
    """
    n, p = X.shape

    # Thêm cột intercept
    Xb = _add_intercept(X)                         # (n, p+1)

    # Chuẩn hoá đầu vào λ
    scalar_input = isinstance(lam, (int, float))
    lam_list = [float(lam)] if scalar_input else [float(l) for l in lam]

    results: list[Vector] = []
    for l in lam_list:
        beta = _solve_ridge(Xb, y, l)
        results.append(beta)

    # ── Vẽ Ridge Trace khi lam là danh sách ──────────────────────────────────
    if not scalar_input and len(lam_list) > 1:
        _plot_ridge_trace(Xb, y, lam_list, results, p)

    if scalar_input:
        return results[0]
    return results


def _plot_ridge_trace(Xb: Matrix, y: Vector, lam_list: list[float],
                      results: list[Vector], p: int) -> None:
    """Vẽ Ridge Trace (hệ số theo λ) và RSS theo λ."""
    import math

    lam_log = [math.log10(l) if l > 0 else -10 for l in lam_list]

    # Thu thập giá trị hệ số (bỏ intercept – vị trí 0)
    coefs = [[beta[j] for beta in results] for j in range(1, p + 1)]

    # Tính RSS cho từng λ
    n = Xb.shape[0]
    rss_vals = [_rss(Xb, y, beta) for beta in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Ridge Regression – Ridge Trace & RSS", fontsize=14, fontweight='bold')

    # ── Biểu đồ 1: Ridge Trace ───────────────────────────────────────────────
    ax1 = axes[0]
    for j, coef_series in enumerate(coefs):
        ax1.plot(lam_log, coef_series, label=f"β{j + 1}")
    ax1.axhline(0, color='k', linewidth=0.7, linestyle='--')
    ax1.set_xlabel("log₁₀(λ)")
    ax1.set_ylabel("Giá trị hệ số")
    ax1.set_title("Ridge Trace: Hệ số theo λ")
    ax1.legend(fontsize=8, ncol=min(p, 4))
    ax1.grid(True, alpha=0.3)

    # ── Biểu đồ 2: RSS theo λ ────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(lam_log, rss_vals, color='crimson', linewidth=1.8)
    ax2.set_xlabel("log₁₀(λ)")
    ax2.set_ylabel("RSS")
    ax2.set_title("RSS theo λ")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Demo / Kiểm thử
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    import math
    random.seed(42)

    # Tạo dữ liệu tổng hợp: y = 2 + 1.5*x1 - 0.8*x2 + noise
    n_demo = 60
    x1 = [random.gauss(0, 1) for _ in range(n_demo)]
    x2 = [random.gauss(0, 1) for _ in range(n_demo)]
    y_data = [2.0 + 1.5 * x1[i] - 0.8 * x2[i] + random.gauss(0, 0.5)
              for i in range(n_demo)]

    X_demo = Matrix([[x1[i], x2[i]] for i in range(n_demo)])
    y_demo = Vector(y_data)

    # ── Scalar λ ──────────────────────────────────────────────────────────────
    print("[Ridge] λ = 1.0 (scalar)")
    beta_r = ridge_fit(X_demo, y_demo, lam=1.0)
    print(f"  β = [{', '.join(f'{beta_r[j]:.4f}' for j in range(len(beta_r.data)))}]")

    # ── Lambda grid → Ridge Trace ─────────────────────────────────────────────
    print("\n[Ridge] Vẽ Ridge Trace với λ ∈ [10⁻³, 10³]")
    lam_grid = [10 ** (e / 10) for e in range(-30, 31)]   # 61 giá trị
    ridge_fit(X_demo, y_demo, lam=lam_grid)
