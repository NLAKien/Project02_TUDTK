import pandas as pd
import numpy as np

class OneHotEncoder:
    def __init__(self, drop_first=True):
        self.drop_first = drop_first
        self.categories_ = {} # Nơi lưu các giá trị duy nhất của từng cột từ tập Train
        self.feature_names_ = [] # Lưu tên cột sau khi encode để tiện tracking

    def fit(self, X):
        X_df = pd.DataFrame(X) # Ép về DataFrame cho dễ thao tác
        
        for col in X_df.columns:
            unique_cats = sorted(X_df[col].dropna().unique())
            if self.drop_first:
                self.categories_[col] = unique_cats[1:]
            else:
                self.categories_[col] = unique_cats
            # Tạo tên cột mới dạng: col_category
            for cat in self.categories_[col]:
                self.feature_names_.append(f"{col}_{cat}")
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X)
        X_encoded = []
        
        for col in self.categories_:
            # Tạo một ma trận toàn số 0 với kích thước (n_samples, n_categories_kept)
            cats = self.categories_[col]
            dummy_matrix = np.zeros((len(X_df), len(cats)))
            
            # Điền số 1 vào vị trí tương ứng
            for i, cat in enumerate(cats):
                dummy_matrix[:, i] = (X_df[col] == cat).astype(int)
            
            X_encoded.append(dummy_matrix)
            
        # Ghép các khối ma trận lại theo chiều ngang
        X_transformed = np.hstack(X_encoded)
        return pd.DataFrame(X_transformed, columns=self.feature_names_)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def get_feature_names_out(self):
        return self.feature_names_

if __name__ == "__main__":
    data = {'col1': ['A', 'B', 'A', 'C', 'B'], 'col2': ['X', 'Y', 'X', 'Z', 'Y']}
    df = pd.DataFrame(data)
    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(df)
    print("Original Data:")
    print(df)
    print("\nEncoded Data:")
    print(encoded_data)
    print("\nFeature Names:")
    print(encoder.get_feature_names_out())