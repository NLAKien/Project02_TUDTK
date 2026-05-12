def is_zero(x: float) -> bool:
	return abs(x) < 1e-10


from operator import index

class Vector:
	"""
	Vector objects by default represent column vectors, if row vector is wanted, let `is_row=True`
	"""
	def __init__(self, components, is_row=False):
		self.data = list(components)
		self.shape = (len(self.data), 1) if is_row == False else len(1, self.data)
	
	def transpose(self) -> Vector:
		ret = Vector(self.data)
		ret.shape = (self.shape[1], self.shape[0])
		return ret

	def __repr__(self):
		sep = ",    " if self.shape[0] == 1 else "    "
		s = sep.join(f"{x:.4f}" for x in self.data)
		return f"({s})"
	
	def __str__(self):
		sep = ",    " if self.shape[1] == 1 else "    "
		s = sep.join(f"{x:.2f}" for x in self.data)
		return f"({s})"
	
	# -- OPERATOR OVERLOADING --
	def __getitem__(self, id):
		return self.data[id]

	def __add__(self, other):
		if self.shape != other.shape:
				raise ValueError(f"Không thể cộng 2 vector khác kích thước: {self.shape[0]}x{self.shape[1]} và {other.shape[0]}x{other.shape[1]}")
		num = max(self.shape[0], self.shape[1])
		data = [self[i] + other[i] for i in range(num)]
		return Vector(data)

	def __sub__(self, other):
		if self.shape != other.shape:
				raise ValueError(f"Không thể trừ 2 vector khác kích thước: {self.shape[0]}x{self.shape[1]} và {other.shape[0]}x{other.shape[1]}")
		num = max(self.shape[0], self.shape[1])
		data = [self[i] - other[i] for i in range(num)]
		return Vector(data)

	def __mul__(self, other):
		if self.shape != other.shape:
				raise ValueError(f"Không thể nhân vô hướng 2 vector khác kích thước: {self.shape[0]}x{self.shape[1]} và {other.shape[0]}x{other.shape[1]}")
		num = max(self.shape[0], self.shape[1])
		ret = 0
		for i in range(num):
			ret += self[i] * other[i]
		return ret


class Matrix:
	""" START CONSTRUCTORS """
	def __init__(self, data: list[list[float]]):
		num_row = len(data)
		num_col = len(data[0])
		self.shape = (num_row, num_col)
		self.rows = [Vector(row[:]) for row in data]
		print([row[j] for row in data] for j in range(num_col))
		# self.cols = [Vector([row[j] for row in data] for j in range(num_col))]

	@classmethod
	def diag(cls, data: list[float]):
		n = len(data)
		return cls([
			[data[i] if i == j else 0 for j in range(n)]
			for i in range(n)
		])

	@classmethod
	def identity(cls, n: int):
		return cls([
			[1 if i == j else 0 for j in range(n)]
			for i in range(n)
		])
	""" END CONSTRUCTORS """

	def __str__(self):
		return "\n".join(
			"[ " + " ".join(f"{x:.2f}" for x in row) + " ]"
			for row in self.data
		)

	""" START ATTRIBUTE CHECKING """
	def is_ref(self):
		# return True if `self` is REF
		""" 
		a matrix is not ref
		if later pivot column index < max column index of all previous pivot column
		"""
		pivot_col = -1
		mat = self.data
		for row in mat:
			for j in range(self.shape[1]):
				if row[j]:
					if j <= pivot_col:
						return False
					pivot_col = j
					continue
		return True

	def is_diagonal(self):
		mat = self.data
		for i in range(self.shape[0]):
			for j in range(self.shape[1]):
				if i != j and mat[i][j]:
					return False
		return True
	
	def __getitem__(self, index):
		return self.data[index]

	def transpose(self):
		# Tạo dữ liệu mới bằng cách đổi hàng thành cột
		new_data = [[self.data[j][i] for j in range(self.shape[0])] for i in range(self.shape[1])]
		# Trả về một đối tượng Matrix mới (đảm bảo file này đã import Matrix hoặc dùng chính class này)
		return Matrix(new_data)

	def __mul__(self, other):
		# 1. Kiểm tra điều kiện kích thước
		if self.shape[1] != other.shape[0]:
			raise ValueError(f"Không thể nhân ma trận: {self.shape[0]}x{self.shape[1]} và {other.shape[0]}x{other.shape[1]}")

		# 2. Tạo ma trận rỗng (m hàng của A x p cột của B)
		# Sử dụng list comprehension để tối ưu tốc độ trong Python thuần
		result_data = [
			[
				sum(self.data[i][k] * other.data[k][j] for k in range(self.shape[1]))
				for j in range(other.shape[1])
			]
			for i in range(self.shape[0])
		]

		# 3. Trả về đối tượng Matrix mới
		return Matrix(result_data)

	def is_identity(self):
		mat = self.data
		for i in range(self.shape[0]):
			for j in range(self.shape[1]):
				if (i != j and mat[i][j]) or (i == j and mat[i][j] != 1):
					return False
		return True
	""" END ATTRIBUTE CHECKING """

	""" START : GENERATE NEW MATRIX """
	def augment(self, other = None):
		"""
		return augmented matrix [self | other]
		if other == None then let other = Identity matrix
		"""
		if other != None and self.shape[0] != other.shape[0]:
			raise ValueError("Matrices must have the same number of rows")

		n = self.shape[0]
		if other == None:
			other = Matrix.identity(n)

		new_data = [self.data[i] + other.data[i] for i in range(n)]
		return Matrix(new_data)

	def take_cols(self, *args):
		# case 1: list of targeted cols
		if len(args) == 1 and isistance(args[0], list):
			selected_cols = args[0]
		# case 2: <<first_col_id>>, <<last_col_id>> -> identify range [<<first_col_id>>, <<last_col_id>>]
		elif len(args) == 2:
			start_col, end_col = args
			selected_cols = list(range(start_col, end_col + 1))
		
		selected_cols.sort()
		return Matrix([[row[j] for j in selected_cols] for row in self.data])

	def inverse(self):
		# return self^{-1}
		from part1.inverse import inverse
		return inverse(self)

	def gaussian_eliminate(self, b):
		# return (REF matrix of self, solution, number of steps)
		return gauss.gaussian_eliminate(self.data, b)
	
	def gauss_jordan_eliminate(self):
		# return RREF matrix of self
		from part1.inverse import gauss_jordan_eliminate as gj
		return gj(self)
	""" END : GENERATE NEW MATRIX """

	""" START : CALCULATE ON MATRIX """



# Testing
if __name__ == "__main__":
	mat = Matrix([
		[1,2,3],
		[6,5,4]
	])
print(f"mat.rows = {mat.rows}", sep="\n")
# print(f"mat.cols = {mat.cols}", sep="\n")
