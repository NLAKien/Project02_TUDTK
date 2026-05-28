from preprocess import OneHotEncoder

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
        self.train_encoded_columns_ = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame):
        """
        Fit pipeline 
        """
        pass

    def transform(self, X: pd.DataFrame):
        """
        Transform data theo pipeline
        """
        pass

    def fit_transform(self, X: pd.DataFrame):
        """
        Fit and transform data
        """
        return self.fit(X).transform(X)