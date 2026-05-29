"""
5. vif(X) — Tính Variance Inflation Factor cho từng biến.

Công thức:
    VIF_j = 1 / (1 - R²_j)
với R²_j là R² khi hồi quy biến X_j theo tất cả các biến còn lại.

Thuần logic: chỉ dùng linalg.Matrix / linalg.Vector và ols_fit từ ols_implementation.
Không dùng sklearn, scipy, statsmodels, hay bất kỳ thư viện ML/stats nào.
"""

from __future__ import annotations

try:
    from .linalg import Matrix, Vector
    from .ols_implementation import ols_fit
except ImportError:
    from linalg import Matrix, Vector
    from ols_implementation import ols_fit


# ─────────────────────────────────────────────────────────────────────────────
# Hàm tiện ích nội bộ
# ─────────────────────────────────────────────────────────────────────────────

def _r_squared(y: Vector, y_hat: Vector, n: int) -> float:
    """
    Tính R² = 1 - RSS/TSS.

    Tham số
    -------
    y     : Vector nhãn thực
    y_hat : Vector nhãn dự đoán
    n     : số quan sát

    Trả về
    ------
    float trong [0, 1] (hoặc âm nếu mô hình rất tệ)
    """
    y_mean = sum(y[i] for i in range(n)) / n
    rss = sum((y[i] - y_hat[i]) ** 2 for i in range(n))
    tss = sum((y[i] - y_mean) ** 2 for i in range(n))
    if tss < 1e-14:
        return 0.0
    return 1.0 - rss / tss


def _col_to_vector(X: Matrix, col_idx: int) -> Vector:
    """Trích xuất cột col_idx của ma trận X thành một Vector cột."""
    return Vector([X[i][col_idx] for i in range(X.shape[0])])


def _matrix_without_col(X: Matrix, col_idx: int) -> Matrix:
    """
    Trả về ma trận X sau khi bỏ cột col_idx.
    Nếu X có 1 cột, trả về ma trận cột hằng số (intercept) để tránh lỗi.
    """
    n, p = X.shape
    remaining = [j for j in range(p) if j != col_idx]
    if not remaining:
        # Degenerate: chỉ có 1 cột → trả về vector hằng 1 làm intercept
        return Matrix([[1.0] for _ in range(n)])
    new_data = [[X[i][j] for j in remaining] for i in range(n)]
    return Matrix(new_data)


# ─────────────────────────────────────────────────────────────────────────────
# 5. vif(X) — Tính VIF cho từng biến
# ─────────────────────────────────────────────────────────────────────────────

def vif(X: Matrix) -> list[float]:
    """
    Tính Variance Inflation Factor (VIF) cho từng cột của X.

        VIF_j = 1 / (1 - R²_j),   j = 0, 1, ..., p-1

    với R²_j là hệ số xác định khi hồi quy biến X_j theo tất cả các cột
    còn lại của X (có thêm cột intercept tự động).

    VIF > 10  →  đa cộng tuyến nghiêm trọng.

    Tham số
    -------
    X : Matrix (n, p) — ma trận đặc trưng
        * Không cần (và không nên) bao gồm cột intercept.
        * Nếu X đã có cột intercept (tất cả giá trị = 1) thì cột đó
          sẽ được bỏ qua trong phần tính VIF (VIF của intercept
          không có ý nghĩa thống kê).

    Trả về
    ------
    vif_values : list[float] — danh sách VIF tương ứng với từng cột X.

    In ra
    -----
    Bảng VIF cho từng biến, kèm chú thích mức độ đa cộng tuyến.
    """
    n, p = X.shape
    if p < 2:
        print("VIF cần ít nhất 2 cột đặc trưng.")
        return [float('inf')]

    vif_values: list[float] = []

    for j in range(p):
        # Biến phụ thuộc: cột j
        y_j = _col_to_vector(X, j)

        # Ma trận dự báo: tất cả các cột còn lại + thêm intercept
        X_rest = _matrix_without_col(X, j)         # (n, p-1)
        # Thêm cột intercept vào đầu
        X_reg_data = [[1.0] + list(X_rest[i]) for i in range(n)]
        X_reg = Matrix(X_reg_data)                 # (n, p)  [intercept | cột còn lại]

        # Hồi quy X_j ~ X_rest (OLS)
        try:
            beta_j, _ = ols_fit(X_reg, y_j)
            y_hat_j = Vector([
                sum(X_reg[i][k] * beta_j[k] for k in range(X_reg.shape[1]))
                for i in range(n)
            ])
            r2_j = _r_squared(y_j, y_hat_j, n)
        except (ValueError, ZeroDivisionError):
            # Ma trận suy biến → đa cộng tuyến hoàn hảo
            r2_j = 1.0

        # Tránh chia cho 0 (đa cộng tuyến hoàn hảo)
        denominator = 1.0 - r2_j
        if abs(denominator) < 1e-12:
            vif_j = float('inf')
        else:
            vif_j = 1.0 / denominator

        vif_values.append(vif_j)

    # ── In bảng kết quả ───────────────────────────────────────────────────────
    _print_vif_table(vif_values, p)

    return vif_values


def _print_vif_table(vif_values: list[float], p: int) -> None:
    """In bảng VIF với chú thích mức độ đa cộng tuyến."""
    W = 55
    print("=" * W)
    print(f"{'Variance Inflation Factor (VIF)':^{W}}")
    print("=" * W)
    print(f"  {'Biến':<10} {'VIF':>10}    {'Chẩn đoán'}")
    print("-" * W)
    for j, v in enumerate(vif_values):
        name = f"X{j + 1}"
        if v == float('inf'):
            diag = "⚠  Đa cộng tuyến HOÀN HẢO"
            v_str = "    ∞"
        elif v > 10:
            diag = "⚠  Đa cộng tuyến NGHIÊM TRỌNG"
            v_str = f"{v:10.4f}"
        elif v > 5:
            diag = "△  Đa cộng tuyến VỪA PHẢI"
            v_str = f"{v:10.4f}"
        else:
            diag = "✓  Chấp nhận được"
            v_str = f"{v:10.4f}"
        print(f"  {name:<10} {v_str}    {diag}")
    print("=" * W)
    print("  Ngưỡng: VIF > 10 → đa cộng tuyến nghiêm trọng")
    print("=" * W)


# ─────────────────────────────────────────────────────────────────────────────
# Demo / Kiểm thử
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- Trường hợp 1: Không có đa cộng tuyến ---
    print("\n[Test 1] Dữ liệu không đa cộng tuyến")
    X1 = Matrix([
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 4.0],
        [4.0, 3.0],
        [5.0, 6.0],
        [6.0, 2.0],
        [7.0, 8.0],
        [8.0, 1.0],
    ])
    v1 = vif(X1)
    print("VIF:", [f"{x:.4f}" for x in v1])

    # --- Trường hợp 2: Đa cộng tuyến cao (X3 ≈ 2*X1 + 0.01*noise) ---
    print("\n[Test 2] Đa cộng tuyến cao")
    import random
    random.seed(0)
    n_demo = 30
    x1 = [float(i) for i in range(1, n_demo + 1)]
    x2 = [v + random.gauss(0, 0.5) for v in x1]          # x2 ≈ x1 (tương quan cao)
    x3 = [random.gauss(5, 2) for _ in range(n_demo)]      # biến độc lập

    X2 = Matrix([[x1[i], x2[i], x3[i]] for i in range(n_demo)])
    v2 = vif(X2)
    print("VIF:", [f"{x:.4f}" for x in v2])
