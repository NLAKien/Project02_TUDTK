def is_zero(x: float) -> bool:
	return abs(x) < 1e-10


from operator import index

class Vector:
	"""
	Vector objects by default represent column vectors, if row vector is wanted, let `is_row=True`
	Display of a column vector: (x_1, x_2, ..., x_n)
	Display of a row vector: (x_1 x_2 ... x_n)
	"""
	def __init__(self, components, is_row=False):
		self.data = list(components)
		self.shape = (len(self.data), 1) if is_row == False else (1, len(self.data))
	
	def transpose(self) -> Vector:
		ret = Vector(self.data)
		ret.shape = (self.shape[1], self.shape[0])
		return ret

	def __repr__(self):
		sep = ",    " if self.shape[1] == 1 else "    "
		s = sep.join(f"{x:.4f}" for x in self.data)
		return f"({s})"
	
	def __str__(self):
		sep = ",    " if self.shape[1] == 1 else "    "
		s = sep.join(f"{x:.2f}" for x in self.data)
		return f"({s})"

	def __iter__(self):
		return iter(self.data)

	# -- OPERATOR OVERLOADING --
	def __getitem__(self, idx):
		return self.data[idx]

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
		self.rows = [Vector(row[:], is_row=True) for row in data]
		self.cols = [Vector(row[j] for row in data) for j in range(num_col)]

	@classmethod
	def diag(cls, data: list[float]):
	    n = len(data)
	    # create raw 2D list of floats
	    grid = [[data[i] if i == j else 0 for j in range(n)] for i in range(n)]
	    return cls(grid)
	
	@classmethod
	def identity(cls, n: int):
	    # just call diag with ones
	    return cls.diag([1] * n)

	""" END CONSTRUCTORS """

	def __str__(self):
		return "[" + ",\n ".join(f"{row}" for row in self.rows) + "]"

	def __getitem__(self, idx):
		return self.rows[idx]
	
	def __iter__(self):
		return iter(self.rows)


	""" START ATTRIBUTE CHECKING """
	def is_ref(self):
		# return True if `self` is REF
		""" 
		a matrix is not ref
		if later pivot column index < max column index of all previous pivot column
		"""
		pivot_col = -1
		for row in self:
			for j in range(self.shape[1]):
				if row[j]:
					if j <= pivot_col:
						return False
					pivot_col = j
					continue
		return True

	def is_diagonal(self):
		for i in range(self.shape[0]):
			for j in range(self.shape[1]):
				if i != j and self[i][j]:
					return False
		return True
	
	def is_identity(self):
		for i in range(self.shape[0]):
			for j in range(self.shape[1]):
				if (i != j and self[i][j]) or (i == j and self[i][j] != 1):
					return False
		return True
	""" END ATTRIBUTE CHECKING """

	""" START : GENERATE NEW MATRIX """
	def transpose(self):
		new_data = [[self[j][i] for j in range(self.shape[0])] for i in range(self.shape[1])]
		return Matrix(new_data)

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
		return Matrix([[row[j] for j in selected_cols] for row in self.data])	# NEED CHANGE

	# --- OPERATOR OVERLOADING ---
	def __mul__(self, other):
		if self.shape[1] != other.shape[0]:
			raise ValueError(f"Không thể nhân ma trận: {self.shape[0]}x{self.shape[1]} và {other.shape[0]}x{other.shape[1]}")

		result_data = [
			[
				sum(self[i][k] * other[k][j] for k in range(self.shape[1]))
				for j in range(other.shape[1])
			]
			for i in range(self.shape[0])
		]
		return Matrix(result_data)



# Testing
if __name__ == "__main__":
	mat = Matrix([
		[1,2,3],
		[6,5,4]
	])
	print(f"mat.rows = {mat.rows}", sep="\n")
	print(f"mat.cols = {mat.cols}", sep="\n")
	print(f"mat =\n{mat}", sep="\n")
	print(f"mat^T =\n{mat.transpose()}", sep="\n")
	print(f"mat is_ref = {mat.is_ref()}", sep="\n")
	print(f"mat is_identity = {mat.is_identity()}", sep="\n")
	print(f"mat is_diagonal = {mat.is_diagonal()}", sep="\n")

	iden = Matrix.identity(4)
	print(f"iden =\n{iden}", sep="\n")
	print(f"iden is_ref = {iden.is_ref()}", sep="\n")
	print(f"iden is_identity = {iden.is_identity()}", sep="\n")
	print(f"iden is_diagonal = {iden.is_diagonal()}", sep="\n")
