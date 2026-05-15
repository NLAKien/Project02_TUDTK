from __future__ import annotations
import random
from linalg import Matrix, Vector
from ols_implementation import ols_fit

def mean(values: list[float]) -> float:
    return sum(values) / len(values)

def variance(values: list[float]) -> float:
    x_bar = mean(values)
    return sum((x - x_bar) ** 2 for x in values) / (len(values) - 1)

def generate_X(n_samples: int) -> Matrix:
    """Tạo design matrix với cột intercept và feature x."""
    data = [[1.0, float(i)] for i in range(1, n_samples + 1)]
    return Matrix(data)

def generate_y(X: Matrix, beta0: float, beta1: float, sigma: float) -> Vector:
    """Sinh dữ liệu theo mô hình: y = beta0 + beta1*x + noise."""
    y_data = []
    for row in X.data: # Giả sử Matrix lưu dữ liệu trong thuộc tính .data
        x = row[1]
        noise = random.gauss(0, sigma)
        y = beta0 + beta1 * x + noise
        y_data.append(y)
    return Vector(y_data)

def run_monte_carlo(
    n_simulations: int = 1000,
    n_samples: int = 50,
    beta0: float = 3.0,
    beta1: float = 2.0,
    sigma: float = 1.0
):
    """Mô phỏng Monte Carlo để kiểm chứng tính không chệch E[beta_hat] = beta."""
    beta0_hats = []
    beta1_hats = []

    X = generate_X(n_samples) # X có thể cố định trong mô phỏng Gauss-Markov

    for _ in range(n_simulations):
        y = generate_y(X, beta0, beta1, sigma)
        beta_hat, _ = ols_fit(X, y)
        
        # Giả sử beta_hat trả về một Vector hoặc list
        beta0_hats.append(beta_hat[0])
        beta1_hats.append(beta_hat[1])

    return {
        "beta0_mean": mean(beta0_hats),
        "beta1_mean": mean(beta1_hats),
        "beta0_variance": variance(beta0_hats),
        "beta1_variance": variance(beta1_hats),
        "beta0_hats": beta0_hats,
        "beta1_hats": beta1_hats
    }


