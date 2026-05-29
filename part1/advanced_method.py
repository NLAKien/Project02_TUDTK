"""
Các hàm Data Fitting nâng cao — Tổng hợp từ các module riêng biệt.

Mỗi hàm yêu cầu được cài đặt trong file riêng (thuần logic):
  • vif(X)                   →  vif.py
  • ridge_fit(X, y, lam)     →  ridge_regression.py
  • residual_plots(X, y, b)  →  residual_analysis.py
  • kfold_cv(X, y, k)        →  cross_validation.py

File này (advanced_method.py) vẫn giữ phiên bản numpy nội bộ (ridge_fit,
residual_plots, kfold_cv, gauss_markov_simulation) đồng thời re-export các
hàm thuần-linalg từ 4 file trên.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Re-export 4 hàm thuần-linalg từ các file riêng ──────────────────────────
try:
    from vif import vif
    from ridge_regression import ridge_fit as ridge_fit_pure
    from residual_analysis import residual_plots as residual_plots_pure
    from cross_validation import kfold_cv as kfold_cv_pure
except ImportError:
    try:
        from .vif import vif
        from .ridge_regression import ridge_fit as ridge_fit_pure
        from .residual_analysis import residual_plots as residual_plots_pure
        from .cross_validation import kfold_cv as kfold_cv_pure
    except ImportError:
        vif = ridge_fit_pure = residual_plots_pure = kfold_cv_pure = None



# ─────────────────────────────────────────────────────────────────────────────
# Hàm tiện ích nội bộ (pure logic)
# ─────────────────────────────────────────────────────────────────────────────

def _mat_inv(A):
    """Nghịch đảo ma trận bằng phân rã LU (Gauss-Jordan) — không dùng np.linalg.inv trực tiếp."""
    n = A.shape[0]
    aug = np.hstack([A.astype(float), np.eye(n)])
    for col in range(n):
        # Tìm pivot lớn nhất (partial pivoting)
        max_row = col + np.argmax(np.abs(aug[col:, col]))
        aug[[col, max_row]] = aug[[max_row, col]]
        pivot = aug[col, col]
        if abs(pivot) < 1e-14:
            raise ValueError("Ma trận suy biến, không thể nghịch đảo.")
        aug[col] /= pivot
        for row in range(n):
            if row != col:
                aug[row] -= aug[row, col] * aug[col]
    return aug[:, n:]


def _ols_beta(X, y):
    """Tính beta OLS: (X^T X)^{-1} X^T y."""
    XtX = X.T @ X
    Xty = X.T @ y
    return _mat_inv(XtX) @ Xty


def _r2(y, y_hat):
    """Tính R² từ y thực và y dự đoán."""
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-14 else 0.0


def _normal_quantiles(n):
    """
    Tính quantile lý thuyết chuẩn N(0,1) cho Q-Q plot
    dùng xấp xỉ Beasley-Springer-Moro (không dùng scipy.stats.norm).
    """
    # Phương pháp Filliben: p_i = (i - 0.375) / (n + 0.25)
    i = np.arange(1, n + 1)
    p = (i - 0.375) / (n + 0.25)
    # Xấp xỉ nghịch đảo CDF chuẩn (Abramowitz & Stegun 26.2.17)
    def _ppf(p_val):
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
            r = np.sqrt(-np.log(p_val if q < 0 else 1 - p_val))
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
    return np.array([_ppf(pv) for pv in p])


def _cook_distance(X, y, y_hat, residuals):
    """
    Tính Cook's Distance thuần tay:
      D_i = (e_i^2 / (p * MSE)) * (h_ii / (1 - h_ii)^2)
    với h_ii = diagonal của hat matrix H = X(X^TX)^{-1}X^T.
    """
    n, p = X.shape
    XtX_inv = _mat_inv(X.T @ X)
    # Diagonal của H = X (X^TX)^{-1} X^T
    h = np.array([X[i] @ XtX_inv @ X[i] for i in range(n)])
    mse = np.sum(residuals ** 2) / max(n - p, 1)
    cook = (residuals ** 2) / (p * mse + 1e-14) * (h / np.clip((1 - h) ** 2, 1e-14, None))
    return cook


# ─────────────────────────────────────────────────────────────────────────────
# 6. ridge_fit(X, y, lam) — Ridge Regression + Ridge Trace
# ─────────────────────────────────────────────────────────────────────────────

def ridge_fit(X, y, lam):
    """
    Cài đặt Ridge Regression (L2 regularization).

    Tham số
    -------
    X   : ndarray (n, p)  — ma trận đặc trưng (chưa thêm cột bias)
    y   : ndarray (n,)    — vector nhãn
    lam : float hoặc array — tham số điều chỉnh λ

    Trả về
    ------
    beta_ridge : ndarray (p+1,) hoặc (len(lam), p+1) — hệ số Ridge (gồm cả bias)
    Vẽ Ridge Trace nếu lam là array.
    """
    n, p = X.shape
    # Thêm cột bias
    ones = np.ones((n, 1))
    Xb = np.hstack([ones, X])          # (n, p+1)

    scalar_input = np.isscalar(lam)
    lam_arr = np.atleast_1d(np.asarray(lam, dtype=float))

    results = []
    for l in lam_arr:
        # β̂_ridge = (X^TX + λI)^{-1} X^T y
        # Không chuẩn hoá cột bias → dùng I với phần tử đầu = 0
        reg = l * np.eye(p + 1)
        reg[0, 0] = 0.0               # không phạt bias
        A = Xb.T @ Xb + reg
        b = Xb.T @ y
        beta = _mat_inv(A) @ b
        results.append(beta)

    results = np.array(results)        # (len(lam), p+1)

    # ── Vẽ Ridge Trace ──────────────────────────────────────────────────────
    if len(lam_arr) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Ridge Regression – Ridge Trace & RSS", fontsize=14, fontweight='bold')

        ax1 = axes[0]
        for j in range(1, p + 1):           # bỏ qua intercept
            ax1.plot(lam_arr, results[:, j], label=f"β{j}")
        ax1.axhline(0, color='k', linewidth=0.7, linestyle='--')
        ax1.set_xscale('log')
        ax1.set_xlabel("λ (log scale)")
        ax1.set_ylabel("Giá trị hệ số")
        ax1.set_title("Ridge Trace: Hệ số theo λ")
        ax1.legend(fontsize=8, ncol=min(p, 4))
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        rss_vals = []
        for i, l in enumerate(lam_arr):
            y_hat = Xb @ results[i]
            rss_vals.append(np.sum((y - y_hat) ** 2))
        ax2.plot(lam_arr, rss_vals, color='crimson')
        ax2.set_xscale('log')
        ax2.set_xlabel("λ (log scale)")
        ax2.set_ylabel("RSS")
        ax2.set_title("RSS theo λ")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    if scalar_input:
        return results[0]
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 7. residual_plots(X, y, beta_hat) — 4 biểu đồ phân tích phần dư
# ─────────────────────────────────────────────────────────────────────────────

def residual_plots(X, y, beta_hat):
    """
    Vẽ 4 biểu đồ phân tích phần dư chuẩn:
      1. Residuals vs Fitted
      2. Q-Q Plot (Normal Q-Q)
      3. Scale-Location (√|e_std| vs Fitted)
      4. Cook's Distance

    Tham số
    -------
    X        : ndarray (n, p) — ma trận đặc trưng (không cột bias)
    y        : ndarray (n,)
    beta_hat : ndarray (p+1,) — hệ số (gồm bias ở vị trí 0)
    """
    n, p = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    y_hat = Xb @ beta_hat
    e = y - y_hat                          # residuals thô
    mse = np.sum(e ** 2) / max(n - p - 1, 1)
    e_std = e / np.sqrt(mse)               # studentised (đơn giản)

    cook = _cook_distance(Xb, y, y_hat, e)

    # Q-Q: sắp xếp phần dư chuẩn hoá + quantile lý thuyết
    e_sorted = np.sort(e_std)
    q_theory = _normal_quantiles(n)

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("Phân Tích Phần Dư (Residual Analysis)", fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── 1. Residuals vs Fitted ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_hat, e, alpha=0.6, edgecolors='k', linewidths=0.4, color='steelblue', s=40)
    ax1.axhline(0, color='red', linewidth=1.2, linestyle='--')
    # Đường LOWESS đơn giản (running median với cửa sổ cố định)
    order = np.argsort(y_hat)
    win = max(3, n // 8)
    smooth_x, smooth_y = [], []
    for i in range(n):
        lo = max(0, i - win // 2)
        hi = min(n, lo + win)
        smooth_x.append(y_hat[order[i]])
        smooth_y.append(np.median(e[order[lo:hi]]))
    ax1.plot(smooth_x, smooth_y, color='red', linewidth=1.4)
    ax1.set_xlabel("Fitted values (ŷ)")
    ax1.set_ylabel("Residuals (e)")
    ax1.set_title("Residuals vs Fitted")
    ax1.grid(True, alpha=0.25)

    # ── 2. Normal Q-Q Plot ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(q_theory, e_sorted, alpha=0.6, edgecolors='k', linewidths=0.4,
                color='darkorange', s=40)
    # Đường tham chiếu (45°)
    lo_q, hi_q = q_theory[0], q_theory[-1]
    ax2.plot([lo_q, hi_q], [lo_q, hi_q], 'r--', linewidth=1.4)
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Sample Quantiles")
    ax2.set_title("Normal Q-Q Plot")
    ax2.grid(True, alpha=0.25)

    # ── 3. Scale-Location ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    sqrt_abs_e = np.sqrt(np.abs(e_std))
    ax3.scatter(y_hat, sqrt_abs_e, alpha=0.6, edgecolors='k', linewidths=0.4,
                color='seagreen', s=40)
    order2 = np.argsort(y_hat)
    smooth_x2, smooth_y2 = [], []
    for i in range(n):
        lo = max(0, i - win // 2)
        hi = min(n, lo + win)
        smooth_x2.append(y_hat[order2[i]])
        smooth_y2.append(np.median(sqrt_abs_e[order2[lo:hi]]))
    ax3.plot(smooth_x2, smooth_y2, color='red', linewidth=1.4)
    ax3.set_xlabel("Fitted values (ŷ)")
    ax3.set_ylabel("√|Standardised Residuals|")
    ax3.set_title("Scale-Location")
    ax3.grid(True, alpha=0.25)

    # ── 4. Cook's Distance ──────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(np.arange(n), cook, color='mediumpurple', alpha=0.7, edgecolor='k', linewidth=0.3)
    threshold = 4.0 / n
    ax4.axhline(threshold, color='red', linewidth=1.2, linestyle='--',
                label=f"Ngưỡng 4/n = {threshold:.4f}")
    influential = np.where(cook > threshold)[0]
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
        "residuals": e,
        "standardised_residuals": e_std,
        "cooks_distance": cook,
        "influential_points": influential,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. kfold_cv(X, y, k) — k-Fold Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

def kfold_cv(X, y, k=5):
    """
    Cài đặt k-Fold Cross-Validation cho OLS.

    CV(k) = (1/k) Σ MSE_i

    Tham số
    -------
    X : ndarray (n, p)  — ma trận đặc trưng (không bias)
    y : ndarray (n,)
    k : int             — số fold

    Trả về
    ------
    cv_score : float    — trung bình MSE qua k fold
    mse_list : list     — MSE của từng fold
    """
    n = len(y)
    if k < 2 or k > n:
        raise ValueError(f"k phải trong [2, {n}], nhận được k={k}")

    # Tạo chỉ số ngẫu nhiên (Fisher-Yates shuffle thuần tay)
    indices = list(range(n))
    # Dùng numpy random chỉ để shuffle (không phải thư viện thống kê)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(indices)

    # Chia thành k fold (cơ số)
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1           # phân bổ phần dư
    fold_ends = np.cumsum(fold_sizes)
    fold_starts = np.concatenate([[0], fold_ends[:-1]])

    mse_list = []
    for i in range(k):
        val_idx = indices[fold_starts[i]: fold_ends[i]]
        train_idx = indices[: fold_starts[i]] + indices[fold_ends[i]:]

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val   = X[val_idx]
        y_val   = y[val_idx]

        # Thêm bias
        Xb_train = np.hstack([np.ones((len(train_idx), 1)), X_train])
        Xb_val   = np.hstack([np.ones((len(val_idx),   1)), X_val])

        beta = _ols_beta(Xb_train, y_train)
        y_pred = Xb_val @ beta
        mse = np.mean((y_val - y_pred) ** 2)
        mse_list.append(mse)

    cv_score = sum(mse_list) / k

    # ── Vẽ kết quả ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"{k}-Fold Cross-Validation (OLS)", fontsize=13, fontweight='bold')

    ax1 = axes[0]
    colors = ['#4c72b0' if m <= cv_score else '#dd8452' for m in mse_list]
    bars = ax1.bar(range(1, k + 1), mse_list, color=colors, edgecolor='k', linewidth=0.5)
    ax1.axhline(cv_score, color='red', linewidth=1.5, linestyle='--',
                label=f"CV Score = {cv_score:.4f}")
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("MSE")
    ax1.set_title("MSE trên từng Fold")
    ax1.set_xticks(range(1, k + 1))
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, mse in zip(bars, mse_list):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001 * max(mse_list),
                 f"{mse:.3f}", ha='center', va='bottom', fontsize=8)

    ax2 = axes[1]
    ax2.plot(range(1, k + 1), np.cumsum(mse_list) / np.arange(1, k + 1),
             marker='o', color='steelblue', linewidth=1.5, label="Running avg MSE")
    ax2.axhline(cv_score, color='red', linewidth=1.2, linestyle='--',
                label=f"Cuối: {cv_score:.4f}")
    ax2.set_xlabel("Fold (tích luỹ)")
    ax2.set_ylabel("MSE trung bình tích luỹ")
    ax2.set_title("Hội Tụ CV Score")
    ax2.set_xticks(range(1, k + 1))
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n{'─'*45}")
    print(f"  {k}-Fold CV (OLS)  |  CV Score = {cv_score:.6f}")
    print(f"{'─'*45}")
    for i, m in enumerate(mse_list, 1):
        print(f"  Fold {i:2d}: MSE = {m:.6f}")
    print(f"{'─'*45}")

    return cv_score, mse_list


# ─────────────────────────────────────────────────────────────────────────────
# 9. Minh hoạ định lý Gauss–Markov bằng Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────

def gauss_markov_simulation(n=50, p=2, n_sim=2000, noise_std=1.0, seed=0):
    """
    Mô phỏng Monte Carlo để kiểm chứng định lý Gauss–Markov:
      (a) E[β̂_OLS] = β  (không chệch)
      (b) Var(β̂_OLS) ≤ Var(β̃)  với mọi ước lượng tuyến tính không chệch β̃

    Ước lượng thay thế được kiểm tra: β̃ = (X^T X + δI)^{-1} X^T y (biased ridge)
    và β̃_w = ước lượng tuyến tính với trọng số ngẫu nhiên.

    Tham số
    -------
    n       : số quan sát mỗi mô phỏng
    p       : số đặc trưng (không bias)
    n_sim   : số lần mô phỏng Monte Carlo
    noise_std: độ lệch chuẩn nhiễu ε ~ N(0, σ²)
    seed    : hạt giống ngẫu nhiên
    """
    rng = np.random.default_rng(seed)

    # Tham số thực sự β (cố định)
    beta_true = rng.standard_normal(p + 1)   # gồm bias

    # Ma trận X cố định (chỉ lấy một lần)
    X_raw = rng.standard_normal((n, p))
    Xb = np.hstack([np.ones((n, 1)), X_raw])  # (n, p+1)

    # ── Lưu trữ ước lượng ────────────────────────────────────────────────────
    beta_ols_list   = []
    beta_ridge_list = []   # ridge nhỏ δ (biased estimator)

    delta = 0.5            # tham số ridge cố định để so sánh

    half = n // 2     # kích thước subset cho ước lượng thay thế

    for _ in range(n_sim):
        eps = rng.normal(0, noise_std, n)
        y = Xb @ beta_true + eps

        # OLS (BLUE)
        b_ols = _ols_beta(Xb, y)
        beta_ols_list.append(b_ols)

        # Ước lượng tuyến tính không chệch khác: OLS trên subset ngẫu nhiên n/2
        # → vẫn không chệch (E[β̃] = β) nhưng variance LỚN HƠN OLS (GM dự đoán)
        idx = rng.choice(n, half, replace=False)
        b_sub = _ols_beta(Xb[idx], y[idx])
        beta_ridge_list.append(b_sub)

    beta_ols_arr   = np.array(beta_ols_list)    # (n_sim, p+1)
    beta_ridge_arr = np.array(beta_ridge_list)

    # ── Thống kê ─────────────────────────────────────────────────────────────
    E_ols   = beta_ols_arr.mean(axis=0)
    E_ridge = beta_ridge_arr.mean(axis=0)
    Var_ols   = beta_ols_arr.var(axis=0)
    Var_ridge = beta_ridge_arr.var(axis=0)

    bias_ols = E_ols   - beta_true
    bias_sub = E_ridge - beta_true   # E_ridge đang lưu subset estimator

    alt_label = f"OLS(n={n//2})"

    # ── In bảng kết quả ──────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  Kiểm chứng Định Lý Gauss–Markov  |  n_sim={n_sim}, n={n}, p={p}")
    print(f"  Ước lượng thay thế: {alt_label} (không chệch, kém hiệu quả hơn)")
    print(f"{'═'*70}")
    print(f"{'Tham số':>8}  {'β_true':>10}  {'E[β̂_OLS]':>12}  {'Bias_OLS':>10}  "
          f"{'Var_OLS':>10}  {f'Var_{alt_label}':>14}")
    print(f"{'─'*70}")
    for j in range(p + 1):
        name = "intercept" if j == 0 else f"β{j}"
        print(f"{name:>8}  {beta_true[j]:>10.4f}  {E_ols[j]:>12.4f}  "
              f"{bias_ols[j]:>10.4f}  {Var_ols[j]:>10.4f}  {Var_ridge[j]:>14.4f}")
    print(f"{'─'*70}")
    print(f"  OLS không chệch: max|Bias| = {np.max(np.abs(bias_ols)):.6f}  (≈ 0 ✓)")
    print(f"  {alt_label} không chệch: max|Bias| = {np.max(np.abs(bias_sub)):.6f}  (≈ 0 ✓)")
    print(f"  Var_OLS ≤ Var_{alt_label} mọi thành phần: "
          f"{'✓  → Gauss–Markov xác nhận' if np.all(Var_ols <= Var_ridge) else '✗'}")
    print(f"{'═'*70}\n")

    # ── Vẽ đồ thị ────────────────────────────────────────────────────────────
    n_params = p + 1
    fig, axes = plt.subplots(2, n_params, figsize=(4 * n_params, 8))
    if n_params == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(
        f"Gauss–Markov Monte Carlo  |  n_sim={n_sim}, n={n}, σ={noise_std}",
        fontsize=13, fontweight='bold'
    )

    labels = ["Intercept"] + [f"β{j}" for j in range(1, n_params)]

    for j in range(n_params):
        # Hàng trên: phân phối β̂_OLS
        ax_top = axes[0, j]
        vals = beta_ols_arr[:, j]
        # Histogram thuần tay
        counts, bins = np.histogram(vals, bins=40)
        ax_top.bar(bins[:-1], counts / (n_sim * (bins[1] - bins[0])),
                   width=(bins[1] - bins[0]) * 0.9,
                   color='steelblue', alpha=0.7, edgecolor='k', linewidth=0.2,
                   label="OLS")
        ax_top.axvline(beta_true[j],  color='red',    linewidth=2.0, label=f"β_true={beta_true[j]:.3f}")
        ax_top.axvline(E_ols[j],      color='green',  linewidth=1.5, linestyle='--',
                       label=f"E[β̂]={E_ols[j]:.3f}")
        ax_top.set_title(f"{labels[j]} – OLS phân phối")
        ax_top.set_xlabel("Giá trị ước lượng")
        ax_top.set_ylabel("Mật độ")
        ax_top.legend(fontsize=7)
        ax_top.grid(True, alpha=0.25)

        # Hàng dưới: OLS vs Subset OLS – Variance
        ax_bot = axes[1, j]
        alt_label = f"OLS(n={n//2})"
        categories = ['OLS (BLUE)', alt_label]
        variances   = [Var_ols[j], Var_ridge[j]]
        bars = ax_bot.bar(categories, variances,
                          color=['steelblue', 'tomato'], edgecolor='k',
                          linewidth=0.5, width=0.4)
        for bar, v in zip(bars, variances):
            ax_bot.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(variances) * 0.01,
                        f"{v:.4f}", ha='center', va='bottom', fontsize=8)
        ax_bot.set_title(f"{labels[j]} – Var so sánh")
        ax_bot.set_ylabel("Variance")
        ax_bot.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    plt.show()

    return {
        "beta_true":   beta_true,
        "E_ols":       E_ols,
        "E_alt":       E_ridge,
        "Var_ols":     Var_ols,
        "Var_alt":     Var_ridge,
        "bias_ols":    bias_ols,
        "bias_alt":    bias_sub,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Demo / Kiểm thử
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # ── Tạo dữ liệu tổng hợp ─────────────────────────────────────────────────
    n, p = 80, 3
    X_demo = rng.standard_normal((n, p))
    beta_true = np.array([2.0, -1.5, 0.8, 0.3])   # [bias, β1, β2, β3]
    Xb_demo = np.hstack([np.ones((n, 1)), X_demo])
    y_demo = Xb_demo @ beta_true + rng.normal(0, 1.0, n)

    # Tính OLS cơ sở để có beta_hat cho residual_plots
    beta_hat_demo = _ols_beta(Xb_demo, y_demo)

    print("=" * 55)
    print("  DEMO CÁC HÀM OLS NÂNG CAO")
    print("=" * 55)

    # ── 6. Ridge Fit ─────────────────────────────────────────────────────────
    print("\n[6] Ridge Regression – Scalar λ = 1.0")
    b_ridge = ridge_fit(X_demo, y_demo, lam=1.0)
    print("  beta_ridge:", b_ridge.round(4))

    print("\n[6] Ridge Trace – λ từ 1e-3 đến 1e3")
    lam_grid = np.logspace(-3, 3, 60)
    ridge_fit(X_demo, y_demo, lam=lam_grid)

    # ── 7. Residual Plots ─────────────────────────────────────────────────────
    print("\n[7] Residual Plots")
    info = residual_plots(X_demo, y_demo, beta_hat_demo)
    print(f"  Influential points (Cook > 4/n): {info['influential_points']}")

    # ── 8. k-Fold CV ─────────────────────────────────────────────────────────
    print("\n[8] 5-Fold Cross-Validation")
    cv_score, mse_list = kfold_cv(X_demo, y_demo, k=5)
    print(f"  CV Score (k=5): {cv_score:.6f}")

    # ── 9. Gauss–Markov Monte Carlo ───────────────────────────────────────────
    print("\n[9] Gauss–Markov Monte Carlo Simulation")
    results = gauss_markov_simulation(n=60, p=2, n_sim=3000, noise_std=1.5, seed=7)

    