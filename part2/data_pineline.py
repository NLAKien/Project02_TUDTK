"""
data_pipeline.py
----------------
DataPipeline class xử lý:
- Missing values
- Encoding biến phân loại
- Chuẩn hóa (z-score standardization)
- Fit trên train, transform trên test
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataPipeline:
    """
    Pipeline tiền xử lý dữ liệu hoàn chỉnh.
    Quy trình: missing values → encoding → chuẩn hóa
    """

    def __init__(
        self,
        missing_strategy: str = "median",
        scale: bool = True,
        encode_categoricals: bool = True,
    ):
        """
        Parameters
        ----------
        missing_strategy : str
            Chiến lược xử lý missing values.
            Chọn: 'mean', 'median', 'mode', 'drop'
        scale : bool
            Có chuẩn hóa z-score không
        encode_categoricals : bool
            Có one-hot encode biến phân loại không
        """
        self.missing_strategy    = missing_strategy
        self.scale               = scale
        self.encode_categoricals = encode_categoricals

        # Lưu các thống kê từ tập train để transform test
        self._fill_values   = {}   # {col: value} để impute
        self._means         = {}   # {col: mean} để chuẩn hóa
        self._stds          = {}   # {col: std} để chuẩn hóa
        self._cat_columns   = []   # Tên cột phân loại
        self._num_columns   = []   # Tên cột số
        self._encoded_cols  = []   # Tên cột sau khi encode
        self._fitted        = False

    # ================================================================
    # FIT
    # ================================================================

    def fit(self, X: pd.DataFrame) -> "DataPipeline":
        """
        Học các thống kê từ tập train.
        Không thay đổi dữ liệu, chỉ lưu tham số.
        """
        X = X.copy()

        # Phân loại cột số và phân loại
        self._cat_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self._num_columns = X.select_dtypes(include=[np.number]).columns.tolist()

        # --- Học fill values cho missing ---
        for col in self._num_columns:
            if self.missing_strategy == "mean":
                self._fill_values[col] = X[col].mean()
            elif self.missing_strategy == "median":
                self._fill_values[col] = X[col].median()
            elif self.missing_strategy == "mode":
                self._fill_values[col] = X[col].mode()[0]

        for col in self._cat_columns:
            self._fill_values[col] = X[col].mode()[0] if not X[col].mode().empty else "Unknown"

        # --- Học mean/std cho chuẩn hóa (dùng sau khi impute) ---
        if self.scale:
            X_temp = X[self._num_columns].fillna(
                {col: self._fill_values.get(col, 0) for col in self._num_columns}
            )
            for col in self._num_columns:
                self._means[col] = X_temp[col].mean()
                self._stds[col]  = X_temp[col].std()
                if self._stds[col] == 0:
                    self._stds[col] = 1.0  # Tránh chia cho 0

        self._fitted = True
        return self

    # ================================================================
    # TRANSFORM
    # ================================================================

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Áp dụng pipeline đã fit lên dữ liệu mới.
        """
        if not self._fitted:
            raise RuntimeError("Gọi fit() trước khi transform()")

        X = X.copy()

        # --- Bước 1: Xử lý missing values ---
        if self.missing_strategy == "drop":
            X = X.dropna()
        else:
            for col, val in self._fill_values.items():
                if col in X.columns:
                    X[col] = X[col].fillna(val)

        # --- Bước 2: Encode biến phân loại ---
        if self.encode_categoricals and self._cat_columns:
            cat_cols_present = [c for c in self._cat_columns if c in X.columns]
            X = pd.get_dummies(X, columns=cat_cols_present, drop_first=True)

        # --- Bước 3: Chuẩn hóa z-score ---
        if self.scale:
            for col in self._num_columns:
                if col in X.columns:
                    X[col] = (X[col] - self._means[col]) / self._stds[col]

        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit rồi transform trên cùng một tập dữ liệu."""
        return self.fit(X).transform(X)

    # ================================================================
    # THỐNG KÊ MISSING VALUES
    # ================================================================

    @staticmethod
    def missing_report(df: pd.DataFrame) -> pd.DataFrame:
        """
        Báo cáo tỉ lệ missing values theo từng cột.
        """
        missing_count = df.isnull().sum()
        missing_pct   = (missing_count / len(df) * 100).round(2)
        report = pd.DataFrame({
            "missing_count": missing_count,
            "missing_pct":   missing_pct,
        })
        return report[report["missing_count"] > 0].sort_values("missing_pct", ascending=False)


# ================================================================
# UNIT TEST
# ================================================================

def _test_pipeline():
    """Unit test cơ bản cho DataPipeline."""
    np.random.seed(42)

    # Tạo dữ liệu giả với missing values
    df = pd.DataFrame({
        "x1": [1.0, 2.0, np.nan, 4.0, 5.0],
        "x2": [10.0, np.nan, 30.0, 40.0, 50.0],
        "cat": ["A", "B", np.nan, "A", "C"],
    })

    pipeline = DataPipeline(missing_strategy="median", scale=True)
    df_train = pipeline.fit_transform(df)

    # Test 1: Không còn missing values
    assert df_train.isnull().sum().sum() == 0, "Test 1 FAILED: vẫn còn missing values"
    print("Test 1 PASSED: Không còn missing values")

    # Test 2: Transform trên test set không bị data leakage
    df_test = pd.DataFrame({
        "x1": [3.0, np.nan],
        "x2": [20.0, 45.0],
        "cat": ["B", "D"],
    })
    df_test_transformed = pipeline.transform(df_test)
    assert df_test_transformed.isnull().sum().sum() == 0, "Test 2 FAILED"
    print("Test 2 PASSED: Transform test set không lỗi")

    print("\nMissing report:")
    print(DataPipeline.missing_report(df))


if __name__ == "__main__":
    _test_pipeline()