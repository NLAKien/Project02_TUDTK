import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

def plot_residual_analysis(y_true, y_pred, title="Residual Analysis"):
    """
    Vẽ 4 biểu đồ chẩn đoán phần dư:
    1. Residuals vs Fitted
    2. Q-Q plot
    3. Histogram of Residuals
    4. Residuals vs Order
    """
    residuals = y_true - y_pred
    standardized_residuals = residuals / np.std(residuals)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)

    # 1. Residuals vs Fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')

    # 2. Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')

    # 3. Histogram
    axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Histogram of Residuals')

    # 4. Residuals vs Order
    axes[1, 1].plot(residuals, 'o-', alpha=0.5, markersize=3)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Observation Order')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals vs Order')

    plt.tight_layout()
    return fig