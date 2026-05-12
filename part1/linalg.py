def is_zero(x: float) -> bool:
	return abs(x) < 1e-10

""" START : ELEMENTARY ROW OPERATIONS """
def row_multiply(row: list, k):
	"""
	row <- row * k
	"""
	n = len(row)
	for i in range(n):
		row[i] *= k
	return

def row_add(row: list, k, another: list):
	"""
	row <- row + (k * another)
	"""
	if len(row) != len(another):
		raise ValueError("Both rows must have the same length")

	n = len(row)
	for i in range(n):
		row[i] += k*another[i]
""" END   : ELEMENTARY ROW OPERATIONS """


from operator import index

class Matrix:
	""" START CONSTRUCTORS """
	def __init__(self, data: list[list[float]]):
		self.data = [row[:] for row in data]
		self.num_row = len(data)
		self.num_col = len(data[0])

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

	def shape(self):
		return (self.rows, self.cols)

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
			for j in range(self.num_col):
				if row[j]:
					if j <= pivot_col:
						return False
					pivot_col = j
					continue
		return True

	def is_diagonal(self):
		mat = self.data
		for i in range(self.num_row):
			for j in range(self.num_col):
				if i != j and mat[i][j]:
					return False
		return True
	
	def __getitem__(self, index):
		return self.data[index]

	def transpose(self):
		# Tạo dữ liệu mới bằng cách đổi hàng thành cột
		new_data = [[self.data[j][i] for j in range(self.num_row)] for i in range(self.num_col)]
		# Trả về một đối tượng Matrix mới (đảm bảo file này đã import Matrix hoặc dùng chính class này)
		return Matrix(new_data)

	def matmul(self, other):
		# 1. Kiểm tra điều kiện kích thước
		if self.num_col != other.num_row:
			raise ValueError(f"Không thể nhân ma trận: {self.num_row}x{self.num_col} và {other.num_row}x{other.num_col}")

		# 2. Tạo ma trận rỗng (m hàng của A x p cột của B)
		# Sử dụng list comprehension để tối ưu tốc độ trong Python thuần
		result_data = [
			[
				sum(self.data[i][k] * other.data[k][j] for k in range(self.num_col))
				for j in range(other.num_col)
			]
			for i in range(self.num_row)
		]

		# 3. Trả về đối tượng Matrix mới
		return Matrix(result_data)

	def is_identity(self):
		mat = self.data
		for i in range(self.num_row):
			for j in range(self.num_col):
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
		if other != None and self.num_row != other.num_row:
			raise ValueError("Matrices must have the same number of rows")

		n = self.num_row
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

class Vector:
	def __init__(self, components, is_column=True):
		self.data = list(components)
		self.num_row = self.num_col = 1
		if is_column == True:
			self.num_row = len(self.data)
		else:
			self.num_col = len(self.data)

	def transpose(self):
		return Vector(self.data, is_column=not self.is_column)

	def __repr__(self):
		direction = "^T" if self.num_col < self.num_row else ""
		s = ", ".join(f"{x:.2f}" for x in self.data)
		return f"({s}){direction}"
	
	def __str__(self):
		direction = "^T" if self.num_col < self.num_row else ""
		s = ", ".join(f"{x:.2f}" for x in self.data)
		return f"({s}){direction}"