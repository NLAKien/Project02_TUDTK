import preprocess as pp
import pandas as pd
import numpy as np

class DataPipeline:
    """
    Pipeline tiền xử lý dữ liệu theo thứ tự:
    1. Xử lý missing values
    2. Encoding
    3. Chuẩn hóa 
    """
    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.is_fitted = False
        self.encoder = pp.OneHotEncoder()

    def fit(self, X: pd.DataFrame):
        """
        Fit pipeline 
        """
        # split X into numeric and categorical
        X_numeric = X[self.numeric_features]
        X_categorical = X[self.categorical_features]
        
        # 1. Imputer
        X_numeric_imputed = pp.knn_imputer(X_numeric)
        X_categorical_imputed = X_categorical.fillna("unknown")
        
        # 3. Scaler
        X_scaled, self.means_, self.stds_ = pp.standard_scaler(pd.DataFrame(X_numeric_imputed, columns=self.numeric_features))
        
        self.is_fitted = True

        # 2. Encoder (chỉ thực hiện khi có cột categorical)
        if self.categorical_features:
            X_encoded = self.encoder.fit_transform(X_categorical_imputed)
            self.train_encoded_columns_ = X_encoded.columns.values
            # ghép encoded dataframe với scaled dataframe
            X_processed = pd.concat([X_scaled, X_encoded], axis=1)
        else:
            self.train_encoded_columns_ = np.array([])
            X_processed = X_scaled

        return X_processed

    def transform(self, X: pd.DataFrame):
        """
        Transform data theo pipeline
        """
        if not self.is_fitted:
            raise ValueError("Pipeline has not been fitted yet.")
        
        # 1. Imputer
        X_numeric_imputed = pp.knn_imputer(X[self.numeric_features])
        X_categorical_imputed = X[self.categorical_features].fillna("unknown")

        # 3. Scaler
        # Correct approach — apply train's mean/std manually
        X_numeric_df = pd.DataFrame(X_numeric_imputed, columns=self.numeric_features)
        X_scaled = (X_numeric_df - pd.Series(self.means_)) / pd.Series(self.stds_)

        # 2. Encoder (chỉ thực hiện khi có cột categorical)
        if self.categorical_features:
            X_encoded = self.encoder.transform(X_categorical_imputed)
            X_encoded = X_encoded[self.train_encoded_columns_]
            # ghép encoded dataframe với scaled dataframe
            X_processed = pd.concat([X_scaled, X_encoded], axis=1)
        else:
            X_processed = X_scaled

        return X_processed

if __name__ == "__main__":
    # Tạo dữ liệu mẫu
    df_train = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': ['X', 'Y', 'Z', 'X'],
        'C': [5, np.nan, 7, 8]
    })

    df_test = pd.DataFrame({
        'A': [10, np.nan, 30],
        'B': ['Y', 'Z', 'X'],
        'C': [15, 20, np.nan]
    })

    # Định nghĩa các cột số và cột danh mục
    numeric_cols = ['A', 'C']
    categorical_cols = ['B']

    # Khởi tạo và fit pipeline
    pipeline = DataPipeline(numeric_features=numeric_cols, categorical_features=categorical_cols)
    
    print("Train Data:")
    print(df_train)
    print("\nTest Data:")
    print(df_test)

    # Fit và transform train data
    df_train_processed = pipeline.fit(df_train)
    print("\nTrain Data Processed:")
    print(df_train_processed)

    # Transform test data
    df_test_processed = pipeline.transform(df_test)
    print("\nTest Data Processed:")
    print(df_test_processed)