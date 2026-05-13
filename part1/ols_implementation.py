from linalg import Matrix, Vector

def ols_fit(X: Matrix, y: Vector) -> Tuple[Vector, float]:
	"""
	X là ma trận mang giá trị của các thuộc tính được chọn tương ứng với mỗi dòng quan trắc
	y là vector mang giá trị kết quả (giá trị cần được dự đoán)
	Return: 2-tuple (Beta hat, square of sigma hat)
		với: 'Beta hat' là vector tham số ước lượng,
		 	 'square of sigma hat' là ước lượng không chệch của phương sai nhiễu
	"""
	pass
