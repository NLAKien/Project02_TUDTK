# stat_inference.py
from linalg import Matrix, Vector
from ols_implementation import ols_fit
from scipy.stats import t as t_dist, f as f_dist
import math


def standard_errors(X: Matrix, squared_sigma_hat: float) -> Vector:
    XtX_inv = (X.transpose() * X).inverse()
    num_params = X.shape[1]          # p+1 (kể intercept)
    se_list = [math.sqrt(squared_sigma_hat * XtX_inv[j][j]) for j in range(num_params)]
    return Vector(se_list)


def t_statistics(beta_hat: Vector, se: Vector) -> Vector:
    num_params = len(beta_hat.data)
    return Vector([beta_hat[j] / se[j] for j in range(num_params)])


def p_values(t_stats: Vector, df: int) -> Vector:
    num_params = len(t_stats.data)
    return Vector([2 * (1 - t_dist.cdf(abs(t_stats[j]), df=df)) for j in range(num_params)])


def confidence_intervals(beta_hat: Vector, se: Vector, df: int, alpha: float = 0.05):
    t_crit = t_dist.ppf(1 - alpha / 2, df=df)
    num_params = len(beta_hat.data)
    return [
        (beta_hat[j] - t_crit * se[j],
         beta_hat[j] + t_crit * se[j])
        for j in range(num_params)
    ]


def r_squared(y: Vector, X: Matrix, beta_hat: Vector) -> float:
    n = X.shape[0]
    y_pred = X * beta_hat
    y_mean = sum(y.data) / n
    rss = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
    tss = sum((y[i] - y_mean)    ** 2 for i in range(n))
    return 1.0 - rss / tss


def adj_r_squared(r2: float, n: int, p: int) -> float:
    """
    p ở đây = số regressor KHÔNG kể intercept
    Adj R² = 1 - (1 - R²)(n - 1) / (n - p - 1)
    """
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)


def f_statistic(r2: float, n: int, p: int):
    """
    p = số regressor KHÔNG kể intercept
    F = [R²/p] / [(1 - R²)/(n - p - 1)]
    df1 = p,   df2 = n - p - 1
    """
    df1 = p
    df2 = n - p - 1
    F_val = (r2 / df1) / ((1.0 - r2) / df2)
    p_val = 1.0 - f_dist.cdf(F_val, dfn=df1, dfd=df2)
    return F_val, p_val


def summary(X: Matrix, y: Vector, feature_names: list[str] = None):
    """
    X đã bao gồm cột intercept (cột 1 đầu tiên).
    feature_names: tên p+1 tham số, vd. ['intercept', 'x1', 'x2']
    """
    n   = X.shape[0]
    p   = X.shape[1] - 1        # số regressor, KHÔNG kể intercept
    df  = n - p - 1             # = df trong ols_fit

    beta_hat, sigma2_hat = ols_fit(X, y)
    se            = standard_errors(X, sigma2_hat)
    t_stats       = t_statistics(beta_hat, se)
    pvals         = p_values(t_stats, df)
    cis           = confidence_intervals(beta_hat, se, df)
    r2            = r_squared(y, X, beta_hat)
    adj_r2        = adj_r_squared(r2, n, p)
    F_val, F_pval = f_statistic(r2, n, p)

    if feature_names is None:
        feature_names = ["intercept"] + [f"x{j}" for j in range(1, p + 1)]

    W = 72
    print("=" * W)
    print(f"{'OLS REGRESSION RESULTS':^{W}}")
    print("=" * W)
    print(f"  Observations : {n:<6}   df residual : {df}")
    print(f"  Regressors   : {p:<6}   σ̂²         : {sigma2_hat:.6f}")
    print(f"  R²           : {r2:.6f}   Adj R²      : {adj_r2:.6f}")
    print(f"  F-statistic  : {F_val:.4f}    Prob(F)     : {F_pval:.4e}")
    print("-" * W)
    print(f"  {'Name':<12} {'Coef':>9} {'Std Err':>9} {'t':>9} {'P>|t|':>9} {'[0.025':>9} {'0.975]':>9}")
    print("-" * W)
    for j in range(p + 1):
        lo, hi = cis[j]
        sig = "*" if pvals[j] < 0.05 else " "
        print(f"  {feature_names[j]:<12} "
              f"{beta_hat[j]:>9.4f} "
              f"{se[j]:>9.4f} "
              f"{t_stats[j]:>9.4f} "
              f"{pvals[j]:>9.4f} "
              f"{lo:>9.4f} "
              f"{hi:>9.4f} {sig}")
    print("=" * W)
    print("  * p < 0.05")


# ── QUICK TEST ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X = Matrix([
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],
        [1, 5]
    ])
    y = Vector([5, 8, 11, 14, 17])

    summary(X, y, feature_names=["intercept", "x1"])