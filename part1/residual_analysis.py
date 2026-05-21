# test_residual_analysis.py
import numpy as np
import matplotlib.pyplot as plt
from residual_analysis import residual_plots

def test_with_numpy():
    """Kiểm chứng với NumPy"""
    print("=== KIỂM CHỨNG VỚI NUMPY ===")
    
    # Tạo dữ liệu
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 3 * X.flatten() + 2 + np.random.randn(100) * 0.5
    
    # Tính beta từ đầu
    X_design = np.c_[np.ones(100), X.flatten()]
    beta_scratch = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
    
    # Tính beta với numpy polyfit
    beta_numpy = np.polyfit(X.flatten(), y, 1)
    
    print(f"Beta from scratch: {beta_scratch}")
    print(f"Beta from numpy: [{beta_numpy[1]}, {beta_numpy[0]}]")
    
    # Vẽ biểu đồ
    fig, res = residual_plots(X_design, y, beta_scratch)
    plt.show()
    
    return np.allclose(beta_scratch, [beta_numpy[1], beta_numpy[0]])

def test_with_sklearn():
    """Kiểm chứng với sklearn"""
    print("\n=== KIỂM CHỨNG VỚI SKLEARN ===")
    
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    
    # Tạo dữ liệu
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 3 * X.flatten() + 2 + np.random.randn(100) * 0.5
    
    # Cài đặt từ đầu
    X_design = np.c_[np.ones(100), X.flatten()]
    beta_scratch = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
    
    # Sklearn
    model = LinearRegression()
    model.fit(X, y)
    
    print(f"MSE from scratch: {mean_squared_error(y, X_design @ beta_scratch):.6f}")
    print(f"MSE from sklearn: {mean_squared_error(y, model.predict(X)):.6f}")
    
    return True

if __name__ == "__main__":
    test_with_numpy()
    test_with_sklearn()
    print("\n✓ Tất cả kiểm chứng đều thành công!")