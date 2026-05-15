import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class DataPipeline:
    """
    Pipeline xử lý dữ liệu hoàn chỉnh:
    - Xử lý missing values
    - Mã hóa biến phân loại
    - Chuẩn hóa biến số
    """
    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.preprocessor = None
        self.is_fitted = False

    def build_preprocessor(self):
        """Xây dựng bộ tiền xử lý"""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),  # Điền missing value bằng median
            ('scaler', StandardScaler())  # Chuẩn hóa về mean=0, std=1
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )
        return self.preprocessor

    def fit(self, X_train):
        """Fit pipeline trên tập train"""
        if self.preprocessor is None:
            self.build_preprocessor()
        self.preprocessor.fit(X_train)
        self.is_fitted = True
        return self

    def transform(self, X):
        """Transform dữ liệu"""
        if not self.is_fitted:
            raise ValueError("Pipeline chưa được fit. Hãy gọi fit() trước.")
        return self.preprocessor.transform(X)

    def fit_transform(self, X_train):
        """Fit và transform trên tập train"""
        self.fit(X_train)
        return self.transform(X_train)

    def get_feature_names(self):
        """Lấy tên các feature sau khi transform"""
        if not self.is_fitted:
            return None
        cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_feature_names = cat_encoder.get_feature_names_out(self.categorical_features)
        return list(self.numeric_features) + list(cat_feature_names)