"""
7. residual_plots(X, y, beta_hat) — 4 biểu đồ phân tích phần dư.

Các biểu đồ:
  1. Residuals vs Fitted    — kiểm tra tính tuyến tính & đồng phương sai
  2. Normal Q-Q Plot        — kiểm tra tính chuẩn của phần dư
  3. Scale-Location         — kiểm tra phương sai đồng đều (homoscedasticity)
  4. Cook's Distance        — xác định các quan sát có ảnh hưởng lớn

Thuần logic: chỉ dùng linalg.Matrix / linalg.Vector.
Không dùng sklearn, scipy, statsmodels, hay bất kỳ thư viện ML/stats nào.
"""

from __future__ import annotations

import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    from .linalg import Matrix, Vector
except ImportError:
    from linalg import Matrix, Vector


# ─────────────────────────────────────────────────────────────────────────────
# Hàm tiện ích nội bộ (thuần logic)
# ─────────────────────────────────────────────────────────────────────────────

def _mat_vec_mul(A: Matrix, v: Vector) -> Vector:
    """Nhân ma trận A (n×m) với vector cột v (m,)."""
    return Vector([
        sum(A[i][k] * v[k] for k in range(A.shape[1]))
        for i in range(A.shape[0])
    ])


def _mat_mul(A: Matrix, B: Matrix) -> Matrix:
    """Nhân hai ma trận A và B."""
    n, m, K = A.shape[0], B.shape[1], A.shape[1]
    return Matrix([
        [sum(A[i][k] * B[k][j] for k in range(K)) for j in range(m)]
        for i in range(n)
    ])


def _transpose(A: Matrix) -> Matrix:
    """Chuyển vị ma trận A."""
    return Matrix([[A[i][j] for i in range(A.shape[0])] for j in range(A.shape[1])])


def _add_intercept(X: Matrix) -> Matrix:
    """Thêm cột 1 (intercept) vào đầu ma trận X."""
    n = X.shape[0]
    return Matrix([[1.0] + [X[i][j] for j in range(X.shape[1])] for i in range(n)])


def _hat_diagonal(Xb: Matrix) -> list[float]:
    """
    Tính đường chéo của Hat matrix H = X(X^TX)^{-1}X^T.
    h_ii = x_i^T (X^TX)^{-1} x_i
    """
    Xt = _transpose(Xb)
    XtX = _mat_mul(Xt, Xb)
    XtX_inv = XtX.inverse()
    if XtX_inv is None:
        n = Xb.shape[0]
        return [0.0] * n
    n = Xb.shape[0]
    h = []
    for i in range(n):
        xi = Vector([Xb[i][k] for k in range(Xb.shape[1])])
        # h_ii = xi^T (X^TX)^{-1} xi
        tmp = _mat_vec_mul(XtX_inv, xi)          # (p+1,)
        h_ii = sum(xi[k] * tmp[k] for k in range(len(xi.data)))
        h.append(h_ii)
    return h


def _cook_distance(Xb: Matrix, e: list[float], mse: float) -> list[float]:
    """
    Tính Cook's Distance:
        D_i = (e_i² / (p · MSE)) · h_ii / (1 - h_ii)²
    """
    n, p = Xb.shape
    h = _hat_diagonal(Xb)
    cook = []
    for i in range(n):
        h_ii = min(h[i], 1.0 - 1e-12)           # tránh chia cho 0
        d = (e[i] ** 2) / (p * mse + 1e-14) * (h_ii / max((1 - h_ii) ** 2, 1e-14))
        cook.append(d)
    return cook


def _normal_quantiles(n: int) -> list[float]:
    """
    Tính quantile lý thuyết N(0,1) bằng xấp xỉ Abramowitz & Stegun 26.2.17.
    Công thức Filliben: p_i = (i - 0.375) / (n + 0.25)
    """
    def _ppf(p_val: float) -> float:
        """Nghịch đảo CDF chuẩn tắc (xấp xỉ rational)."""
        if p_val <= 0:
            return -8.0
        if p_val >= 1:
            return 8.0
        q = p_val - 0.5
        if abs(q) <= 0.425:
            r = 0.180625 - q * q
            num = (((((((2.5090809287301226727e3 * r +
                          3.3430575583588128105e4) * r +
                          6.7265770927008700853e4) * r +
                          4.5921953931549871457e4) * r +
                          1.3731693765509461125e4) * r +
                          1.9715909503065514427e3) * r +
                          1.3314616702987527976e2) * r +
                          3.3871328727963666080e0) * q
            den = (((((((5.2264952788528545610e3 * r +
                          2.8729085735721942674e4) * r +
                          3.9307895800092710610e4) * r +
                          2.1213794301586595867e4) * r +
                          5.3941960214247511077e3) * r +
                          6.8718700749205790830e2) * r +
                          4.2313330701600911252e1) * r + 1.0)
            return num / den
        else:
            r = math.sqrt(-math.log(p_val if q < 0 else 1 - p_val))
            if r <= 5:
                r -= 1.6
                num = (((((((7.7133361990583708786e-5 * r +
                              1.2256372724086830321e-3) * r +
                              1.1077232987691765757e-2) * r +
                              4.6647109958629042715e-2) * r +
                              1.0434994865674728051e-1) * r +
                              1.1388547798021914979e-1) * r +
                              5.5855823801564311830e-2) * r +
                              1.3512093740528449716e-2)
                den = (((((((1.0507500716444522877e-4 * r +
                              1.4810143742274127665e-3) * r +
                              1.1353661682543052591e-2) * r +
                              3.7235316048253478289e-2) * r +
                              5.9467982927001562100e-2) * r +
                              3.9874436174979073851e-2) * r +
                              8.5567418059665564941e-3) * r + 1.0)
            else:
                r -= 5.0
                num = (((((((2.0103343992922633202e-9 * r +
                              2.7115555687715776124e-8) * r +
                              1.5421227249498880095e-7) * r +
                              4.8631919794604162036e-7) * r +
                              1.1369459477328752498e-6) * r +
                              1.8583304968782649573e-6) * r +
                              1.7554455698302039099e-6) * r +
                              8.5969888095818499671e-7)
                den = (((((((2.0440959022448609428e-10 * r +
                              1.2889159700487696453e-9) * r +
                              4.7874374165151700694e-9) * r +
                              1.2018741164651614978e-8) * r +
                              2.2193089649453830509e-8) * r +
                              2.7818602669688047703e-8) * r +
                              2.2148417523247148521e-8) * r + 1.0)
            val = num / den
            return -val if q < 0 else val

    quantiles = []
    for i in range(1, n + 1):
        p_i = (i - 0.375) / (n + 0.25)
        quantiles.append(_ppf(p_i))
    return quantiles


def _running_median(x_sorted: list[float], y_sorted: list[float],
                    win: int) -> tuple[list[float], list[float]]:
    """
    Đường xu hướng đơn giản bằng running median (cửa sổ trượt).
    Dùng để làm smooth trên Residuals vs Fitted và Scale-Location.
    """
    n = len(x_sorted)
    sm_x, sm_y = [], []
    for i in range(n):
        lo = max(0, i - win // 2)
        hi = min(n, lo + win)
        window = sorted(y_sorted[lo:hi])
        mid = len(window) // 2
        median = window[mid] if len(window) % 2 == 1 else (window[mid - 1] + window[mid]) / 2
        sm_x.append(x_sorted[i])
        sm_y.append(median)
    return sm_x, sm_y


def _sort_by(keys: list[float], *lists):
    """Sắp xếp tất cả lists theo thứ tự tăng dần của keys."""
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    sorted_keys = [keys[i] for i in order]
    sorted_lists = [[lst[i] for i in order] for lst in lists]
    return (sorted_keys, *sorted_lists)


# ─────────────────────────────────────────────────────────────────────────────
# 7. residual_plots(X, y, beta_hat) — 4 biểu đồ phân tích phần dư
# ─────────────────────────────────────────────────────────────────────────────

def residual_plots(X: Matrix, y: Vector, beta_hat: Vector) -> dict:
    """
    Vẽ 4 biểu đồ phân tích phần dư chuẩn:
      1. Residuals vs Fitted   — tính tuyến tính & đồng phương sai
      2. Normal Q-Q Plot       — tính chuẩn của phần dư
      3. Scale-Location        — phương sai đồng đều (homoscedasticity)
      4. Cook's Distance       — quan sát ảnh hưởng lớn (influential points)

    Tham số
    -------
    X        : Matrix (n, p)   — ma trận đặc trưng (không bao gồm cột bias)
    y        : Vector (n,)     — nhãn thực tế
    beta_hat : Vector (p+1,)   — hệ số ước lượng (bias ở vị trí 0)

    Trả về
    ------
    dict gồm:
      'residuals'              : list[float]  — phần dư thô e = y − ŷ
      'standardised_residuals' : list[float]  — phần dư chuẩn hoá
      'cooks_distance'         : list[float]  — Cook's Distance
      'influential_points'     : list[int]    — chỉ số quan sát có ảnh hưởng lớn
    """
    n, p = X.shape
    Xb = _add_intercept(X)                        # (n, p+1)

    # ── Tính phần dư ─────────────────────────────────────────────────────────
    y_hat_vec = _mat_vec_mul(Xb, beta_hat)
    y_hat = [y_hat_vec[i] for i in range(n)]
    e     = [y[i] - y_hat[i] for i in range(n)]

    # MSE (dùng bậc tự do n - p - 1)
    df_res = max(n - p - 1, 1)
    mse    = sum(ei ** 2 for ei in e) / df_res
    sigma  = math.sqrt(mse) if mse > 0 else 1.0

    # Phần dư chuẩn hoá (studentised đơn giản)
    e_std = [ei / sigma for ei in e]

    # Cook's Distance
    cook = _cook_distance(Xb, e, mse)

    # Q-Q plot: sắp xếp phần dư chuẩn hoá + quantile lý thuyết
    e_sorted = sorted(e_std)
    q_theory = _normal_quantiles(n)

    # Sắp xếp theo ŷ cho Residuals vs Fitted và Scale-Location
    y_hat_s, e_s, e_std_s = _sort_by(y_hat, e, e_std)[0], \
                             _sort_by(y_hat, e, e_std)[1], \
                             _sort_by(y_hat, e, e_std)[2]
    # Tính lại sắp xếp đúng cách
    order_yhat = sorted(range(n), key=lambda i: y_hat[i])
    y_hat_s = [y_hat[i] for i in order_yhat]
    e_s     = [e[i]     for i in order_yhat]
    e_std_s = [e_std[i] for i in order_yhat]

    win = max(3, n // 8)
    sm_x1, sm_y1 = _running_median(y_hat_s, e_s, win)
    sqrt_abs_e_s  = [math.sqrt(abs(v)) for v in e_std_s]
    sm_x3, sm_y3  = _running_median(y_hat_s, sqrt_abs_e_s, win)

    # Ngưỡng Cook
    threshold = 4.0 / n
    influential = [i for i, d in enumerate(cook) if d > threshold]

    # ── Vẽ 4 biểu đồ ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("Phân Tích Phần Dư (Residual Analysis)", fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── 1. Residuals vs Fitted ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_hat, e, alpha=0.6, edgecolors='k', linewidths=0.4,
                color='steelblue', s=40)
    ax1.axhline(0, color='red', linewidth=1.2, linestyle='--')
    ax1.plot(sm_x1, sm_y1, color='red', linewidth=1.4)
    ax1.set_xlabel("Fitted values (ŷ)")
    ax1.set_ylabel("Residuals (e)")
    ax1.set_title("Residuals vs Fitted")
    ax1.grid(True, alpha=0.25)

    # ── 2. Normal Q-Q Plot ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(q_theory, e_sorted, alpha=0.6, edgecolors='k', linewidths=0.4,
                color='darkorange', s=40)
    lo_q, hi_q = q_theory[0], q_theory[-1]
    ax2.plot([lo_q, hi_q], [lo_q, hi_q], 'r--', linewidth=1.4)
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles")
    ax2.set_title("Normal Q-Q Plot")
    ax2.grid(True, alpha=0.25)

    # ── 3. Scale-Location ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    sqrt_abs_e = [math.sqrt(abs(v)) for v in e_std]
    ax3.scatter(y_hat, sqrt_abs_e, alpha=0.6, edgecolors='k', linewidths=0.4,
                color='seagreen', s=40)
    ax3.plot(sm_x3, sm_y3, color='red', linewidth=1.4)
    ax3.set_xlabel("Fitted values (ŷ)")
    ax3.set_ylabel("√|Standardised Residuals|")
    ax3.set_title("Scale-Location")
    ax3.grid(True, alpha=0.25)

    # ── 4. Cook's Distance ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(list(range(n)), cook, color='mediumpurple', alpha=0.7,
            edgecolor='k', linewidth=0.3)
    ax4.axhline(threshold, color='red', linewidth=1.2, linestyle='--',
                label=f"Ngưỡng 4/n = {threshold:.4f}")
    for idx in influential:
        ax4.annotate(str(idx), (idx, cook[idx]),
                     textcoords="offset points", xytext=(0, 4),
                     fontsize=7, color='red', ha='center')
    ax4.set_xlabel("Chỉ số quan sát")
    ax4.set_ylabel("Cook's Distance")
    ax4.set_title("Cook's Distance")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.25, axis='y')

    plt.show()

    return {
        "residuals":              e,
        "standardised_residuals": e_std,
        "cooks_distance":         cook,
        "influential_points":     influential,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Demo / Kiểm thử
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(42)

    # Tạo dữ liệu tổng hợp: y = 2 + 1.5*x1 - 0.8*x2 + noise
    n_demo = 80
    x1 = [random.gauss(0, 1) for _ in range(n_demo)]
    x2 = [random.gauss(0, 1) for _ in range(n_demo)]
    y_data = [2.0 + 1.5 * x1[i] - 0.8 * x2[i] + random.gauss(0, 0.8)
              for i in range(n_demo)]

    X_demo = Matrix([[x1[i], x2[i]] for i in range(n_demo)])
    y_demo = Vector(y_data)

    # Tính OLS (cần thêm cột intercept trước khi gọi ols_fit)
    try:
        from ols_implementation import ols_fit
    except ImportError:
        from .ols_implementation import ols_fit

    Xb_demo = Matrix([[1.0, x1[i], x2[i]] for i in range(n_demo)])
    beta_hat_demo, _ = ols_fit(Xb_demo, y_demo)

    print("[7] Vẽ 4 biểu đồ phân tích phần dư")
    info = residual_plots(X_demo, y_demo, beta_hat_demo)
    print(f"Influential points (Cook > 4/n): {info['influential_points']}")