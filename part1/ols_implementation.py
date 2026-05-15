from linalg import Matrix, Vector

def ols_fit(X: Matrix, y: Vector) -> Tuple[Vector, float]:
	"""
	X là ma trận mang giá trị của các thuộc tính được chọn tương ứng với mỗi dòng quan trắc
	y là vector mang giá trị kết quả (giá trị cần được dự đoán)
	Return: 2-tuple (Beta hat, square of sigma hat)
		với: 'Beta hat' là vector tham số ước lượng,
		 	 'square of sigma hat' là ước lượng không chệch của phương sai nhiễu
	"""
	X_transpose = X.transpose()
	X_transpose__X__inv = (X_transpose*X).inverse()
	beta_hat_ols = X_transpose__X__inv*X_transpose*y

	n = X.shape[0]
	p = X.shape[1]
	rss = (y - X*beta_hat_ols).squared_norm()
	squared_sigma_hat = rss/(n - p - 1)

	return (beta_hat_ols, squared_sigma_hat)



if __name__ == "__main__":
	X = Matrix([
	    [1, 1],
	    [1, 2],
	    [1, 3],
	    [1, 4],
	    [1, 5]
	])
	
	y = Vector([5, 8, 11, 14, 17])
	
	beta_hat, sigma2_hat = ols_fit(X, y)
	
	print(f"beta_hat = {beta_hat}")
	
	print(f"sigma^2_hat = {sigma2_hat}")
