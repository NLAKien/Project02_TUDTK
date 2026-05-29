"""
8. kfold_cv(X, y, k) — k-Fold Cross-Validation cho OLS.

Công thức:
    CV(k) = (1/k) Σ_{i=1}^{k} MSE_i

Thuần logic: chỉ dùng linalg.Matrix / linalg.Vector và ols_fit từ ols_implementation.
Không dùng sklearn, scipy, statsmodels, hay bất kỳ thư viện ML/stats nào.
"""

from __future__ import annotations

import math
import random
import matplotlib.pyplot as plt

try:
    from .linalg import Matrix, Vector
    from .ols_implementation import ols_fit
except ImportError:
    from linalg import Matrix, Vector
    from ols_implementation import ols_fit


# ─────────────────────────────────────────────────────────────────────────────
# Hàm tiện ích nội bộ (thuần logic)
# ─────────────────────────────────────────────────────────────────────────────

def _mat_vec_mul(A: Matrix, v: Vector) -> Vector:
    """Nhân ma trận A (n×m) với vector cột v (m,)."""
    return Vector([
        sum(A[i][k] * v[k] for k in range(A.shape[1]))
        for i in range(A.shape[0])
    ])


def _add_intercept_rows(rows: list[list[float]]) -> list[list[float]]:
    """Thêm giá trị 1.0 vào đầu mỗi hàng (thêm cột intercept)."""
    return [[1.0] + row for row in rows]


def _shuffle_indices(n: int, seed: int = 42) -> list[int]:
    """
    Trả về danh sách [0, 1, ..., n-1] được xáo trộn ngẫu nhiên.
    Dùng thuật toán Fisher-Yates thuần Python (không dùng numpy).
    """
    indices = list(range(n))
    rng = random.Random(seed)
    for i in range(n - 1, 0, -1):
        j = rng.randint(0, i)
        indices[i], indices[j] = indices[j], indices[i]
    return indices


def _split_folds(indices: list[int], k: int) -> list[list[int]]:
    """
    Chia danh sách chỉ số thành k fold có kích thước gần bằng nhau.
    Các phần tư dư (n % k phần tử) được phân bổ vào k fold đầu.
    """
    n = len(indices)
    fold_size, remainder = divmod(n, k)
    folds: list[list[int]] = []
    start = 0
    for i in range(k):
        extra = 1 if i < remainder else 0
        end = start + fold_size + extra
        folds.append(indices[start:end])
        start = end
    return folds


def _mse(y_true: list[float], y_pred: list[float]) -> float:
    """Tính Mean Squared Error giữa y_true và y_pred."""
    n = len(y_true)
    return sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n


# ─────────────────────────────────────────────────────────────────────────────
# 8. kfold_cv(X, y, k) — k-Fold Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

def kfold_cv(X: Matrix, y: Vector, k: int = 5, seed: int = 42) -> tuple[float, list[float]]:
    """
    Cài đặt k-Fold Cross-Validation cho mô hình OLS.

        CV(k) = (1/k) Σ_{i=1}^{k} MSE_i

    Tham số
    -------
    X    : Matrix (n, p)  — ma trận đặc trưng (không bao gồm cột bias)
    y    : Vector (n,)    — vector nhãn
    k    : int            — số fold (mặc định 5)
    seed : int            — hạt giống cho xáo trộn ngẫu nhiên (mặc định 42)

    Trả về
    ------
    cv_score : float      — trung bình MSE qua k fold
    mse_list : list[float]— MSE của từng fold (từ fold 1 đến fold k)

    In ra
    -----
    Bảng MSE từng fold và CV score tổng hợp.
    Vẽ 2 biểu đồ: MSE từng fold + Running average MSE.
    """
    n = X.shape[0]
    p = X.shape[1]

    if k < 2 or k > n:
        raise ValueError(f"k phải trong [2, {n}], nhận được k={k}")

    # Trích dữ liệu thô (list) để thao tác dễ dàng
    X_data = [[X[i][j] for j in range(p)] for i in range(n)]
    y_data = [y[i] for i in range(n)]

    # Xáo trộn chỉ số (Fisher-Yates thuần Python)
    indices = _shuffle_indices(n, seed=seed)

    # Chia thành k fold
    folds = _split_folds(indices, k)

    mse_list: list[float] = []

    for fold_idx in range(k):
        # Chỉ số validation và training
        val_idx   = folds[fold_idx]
        train_idx = [i for fi in range(k) if fi != fold_idx for i in folds[fi]]

        # Xây dựng tập train (có intercept)
        X_train_rows = _add_intercept_rows([X_data[i] for i in train_idx])
        y_train_data = [y_data[i] for i in train_idx]
        X_train = Matrix(X_train_rows)
        y_train = Vector(y_train_data)

        # Xây dựng tập validation (có intercept)
        X_val_rows = _add_intercept_rows([X_data[i] for i in val_idx])
        y_val_data = [y_data[i] for i in val_idx]
        X_val  = Matrix(X_val_rows)

        # Huấn luyện OLS trên tập train
        try:
            beta, _ = ols_fit(X_train, y_train)
        except (ValueError, ZeroDivisionError):
            # Fold này có ma trận suy biến (thường do fold quá nhỏ)
            mse_list.append(float('inf'))
            continue

        # Dự đoán trên tập validation
        y_pred_vec = _mat_vec_mul(X_val, beta)
        y_pred = [y_pred_vec[i] for i in range(len(val_idx))]

        # Tính MSE fold
        mse_fold = _mse(y_val_data, y_pred)
        mse_list.append(mse_fold)

    # Lọc inf trước khi tính trung bình
    valid_mse = [m for m in mse_list if not math.isinf(m)]
    cv_score = sum(valid_mse) / len(valid_mse) if valid_mse else float('inf')

    # ── In bảng kết quả ───────────────────────────────────────────────────────
    _print_cv_table(k, mse_list, cv_score)

    # ── Vẽ biểu đồ ───────────────────────────────────────────────────────────
    _plot_cv(k, mse_list, cv_score)

    return cv_score, mse_list


def _print_cv_table(k: int, mse_list: list[float], cv_score: float) -> None:
    """In bảng kết quả cross-validation."""
    print(f"\n{'─' * 45}")
    print(f"  {k}-Fold CV (OLS)  |  CV Score = {cv_score:.6f}")
    print(f"{'─' * 45}")
    for i, m in enumerate(mse_list, 1):
        flag = "  ← tốt nhất" if m == min(mse_list) else ""
        print(f"  Fold {i:2d}: MSE = {m:.6f}{flag}")
    print(f"{'─' * 45}")
    print(f"  Trung bình MSE (CV Score) = {cv_score:.6f}")
    print(f"  RMSE tương ứng            = {math.sqrt(cv_score):.6f}")
    print(f"{'─' * 45}\n")


def _plot_cv(k: int, mse_list: list[float], cv_score: float) -> None:
    """Vẽ 2 biểu đồ: MSE từng fold + Running average MSE."""
    max_mse = max(m for m in mse_list if not math.isinf(m)) if mse_list else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"{k}-Fold Cross-Validation (OLS)", fontsize=13, fontweight='bold')

    # ── Biểu đồ 1: MSE từng fold ─────────────────────────────────────────────
    ax1 = axes[0]
    colors = ['#4c72b0' if m <= cv_score else '#dd8452' for m in mse_list]
    bars = ax1.bar(range(1, k + 1), mse_list, color=colors,
                   edgecolor='k', linewidth=0.5)
    ax1.axhline(cv_score, color='red', linewidth=1.5, linestyle='--',
                label=f"CV Score = {cv_score:.4f}")
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("MSE")
    ax1.set_title("MSE trên từng Fold")
    ax1.set_xticks(range(1, k + 1))
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, mse in zip(bars, mse_list):
        offset = max_mse * 0.01
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + offset,
                 f"{mse:.3f}", ha='center', va='bottom', fontsize=8)

    # ── Biểu đồ 2: Running average MSE ───────────────────────────────────────
    ax2 = axes[1]
    running_avg = [
        sum(mse_list[:i + 1]) / (i + 1)
        for i in range(k)
    ]
    ax2.plot(range(1, k + 1), running_avg, marker='o',
             color='steelblue', linewidth=1.5, label="Running avg MSE")
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


# ─────────────────────────────────────────────────────────────────────────────
# Demo / Kiểm thử
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(0)

    # Tạo dữ liệu tổng hợp: y = 3 + 2*x1 - x2 + noise
    n_demo = 80
    x1 = [random.gauss(0, 1) for _ in range(n_demo)]
    x2 = [random.gauss(0, 1) for _ in range(n_demo)]
    y_data = [3.0 + 2.0 * x1[i] - 1.0 * x2[i] + random.gauss(0, 0.5)
              for i in range(n_demo)]

    X_demo = Matrix([[x1[i], x2[i]] for i in range(n_demo)])
    y_demo = Vector(y_data)

    print("[8] 5-Fold Cross-Validation")
    cv_score, mse_list = kfold_cv(X_demo, y_demo, k=5)
    print(f"CV Score (k=5): {cv_score:.6f}")

    print("\n[8] 10-Fold Cross-Validation")
    cv_score10, _ = kfold_cv(X_demo, y_demo, k=10)
    print(f"CV Score (k=10): {cv_score10:.6f}")
