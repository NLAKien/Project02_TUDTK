from math import sqrt

def is_zero(x: float) -> bool:
	return abs(x) < 1e-10


from operator import index

class Vector:
	"""
	Vector objects by default represent column vectors, if row vector is wanted, let `is_col=False`
	Display of a column vector: (x_1, x_2, ..., x_n)
	Display of a row vector: (x_1 x_2 ... x_n)
	"""
	def __init__(self, components, is_col=True, copy=True):
		self.data = list(components) if copy else components
		self.shape = (len(self.data), 1) if is_col == True else (1, len(self.data))
	
	def transpose(self) -> Vector:
		return Vector(self.data, is_col=(self.shape[1] != 1))
	
	def squared_norm(self) -> float:
		return sum(entry*entry for entry in self.data)
	
	def norm(self) -> float:
		return sqrt(squared_norm)
	
	def scalar_mul(self, other) -> float:
		if not isinstance(other, Vector):
			raise TypeError("Vector chỉ có thể nhân vô hướng với Vector")
		num = max(self.shape[0], self.shape[1])
		ret = 0
		for i in range(num):
			ret += self[i] * other[i]
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

	def __getitem__(self, idx):
		return self.data[idx]
	
	def __setitem__(self, idx, value):
		self.data[idx] = value

	# -- OPERATOR OVERLOADING --
	def __add__(self, other):
		if self.shape != other.shape:
				raise ValueError(f"Không thể cộng 2 vector khác kích thước: {self.shape[0]}x{self.shape[1]} và {other.shape[0]}x{other.shape[1]}")
		num = max(self.shape[0], self.shape[1])
		data = [self[i] + other[i] for i in range(num)]
		return Vector(data, is_col=(self.shape[1] == 1))

	def __sub__(self, other):
		if self.shape != other.shape:
				raise ValueError(f"Không thể trừ 2 vector khác kích thước: {self.shape[0]}x{self.shape[1]} và {other.shape[0]}x{other.shape[1]}")
		num = max(self.shape[0], self.shape[1])
		data = [self[i] - other[i] for i in range(num)]
		return Vector(data, is_col=(self.shape[1] == 1))

	def __mul__(self, other):
		if isinstance(other, Vector):
			if self.shape[1] != other.shape[0]:
					raise ValueError(f"Không thể nhân 2 vector kích thước không phù hợp: {self.shape[0]}x{self.shape[1]} và {other.shape[0]}x{other.shape[1]}")
			num = max(self.shape[0], self.shape[1])
			ret = 0
			for i in range(num):
				ret += self[i] * other[i]
			return ret
		elif isinstance(other, int | float):
			v_ret = Vector([entry * other for entry in self.data], is_col=(self.shape[1] == 1))
			return v_ret
		else:
			raise TypeError("Kiểu của hạng tử không được hỗ trợ cho toán tử *")
	def __rmul__(self, other):
		return self.__mul__(other)


class Matrix:
	""" START CONSTRUCTORS """
	def __init__(self, data: list[list[float]], by_cols=False):
		if not data:
			raise ValueError("Cần tồn tại giá trị để khởi tạo ma trận")
		if by_cols == True:
			tmp = [[data[j][i] for j in range(len(data))] for i in range(len(data[0]))]
			data = tmp

		num_row = len(data)
		num_col = len(data[0])
		self.data = [[entry for entry in row] for row in data]
		self.shape = (num_row, num_col)
		self.rows = [Vector(row, is_col=False, copy=False) for row in self.data]

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
	
	@classmethod
	def from_vecs(cls, vecs: list[Vector], is_col=True):
		"""
		Create matrix from vectors
		"""
		vec_len = 0
		for vec in vecs:
			if not vec_len:
				vec_shape = vec.shape
			if vec.shape != vec_shape:
				raise ValueError("Các vector phải có cùng kích thước")

		vec_len = max(vec_shape[0], vec_shape[1])
		if vec_len == 0:
			raise ValueError("Các vector phải tốn tại giá trị")
		# create raw 2D list of floats
		if is_col:
			grid = [[col[i] for col in vecs] for i in range(vec_len)]
		else:
			grid = [[entry for entry in row] for row in vecs]
		return cls(grid)

	""" END CONSTRUCTORS """

	def __str__(self):
		return "[" + ",\n ".join(f"{row}" for row in self.rows) + "]"

	def __getitem__(self, key):
		# key dang (slice(None), col_idx) tuong ung Matrix[:, j]
		if isinstance(key, tuple) and isinstance(key[0], slice):
			col_idx = key[1]
			return [self.rows[i].data[col_idx] for i in range(self.shape[0])]
		else:
			return self.rows[key]

	def __setitem__(self, key, value):
		# key dang (slice(None), col_idx) tuong ung Matrix[:, j]
		if isinstance(key, tuple) and isinstance(key[0], slice):
			col_idx = key[1]
			for i in range(self.shape[0]):
				self.rows[i].data[col_idx] = value[i]
		else:
			# gan dong
			self.rows[key] = value
			self.data[key] = value
	
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
			raise ValueError("Hai ma trận phải có cùng số dòng để có thể ghép")

		n = self.shape[0]
		if other == None:
			other = Matrix.identity(n)

		self_cols = [Vector(row[j] for row in self.data) for j in range(self.shape[1])]
		other_cols = [Vector(row[j] for row in other.data) for j in range(other.shape[1])]
		col_vecs = self_cols + other_cols
		return Matrix.from_vecs(col_vecs, is_col=True)

	def take_cols(self, *args):
		# case 1: list of targeted cols
		if len(args) == 1 and isinstance(args[0], list):
			selected_cols = args[0]
		# case 2: <<first_col_id>>, <<last_col_id>> -> identify range [<<first_col_id>>, <<last_col_id>>]
		elif len(args) == 2:
			start_col, end_col = args
			selected_cols = list(range(start_col, end_col + 1))
		
		selected_cols.sort()
		self_cols = [Vector(row[j] for row in self.data) for j in range(self.shape[1])]
		col_list = [self_cols[col_id] for col_id in selected_cols]
		return Matrix.from_vecs(col_list, is_col=True)	# NEED CHANGE

	def gauss_jordan_eliminate(self) -> Matrix:
		"""
		Khu Gauss-Jordan co chon phan tu chot (Partial Pivoting)
		Return: ma tran rref
		"""
		# clone: mat <- self
		mat = Matrix.from_vecs(self.rows, is_col=False)
		# bien count duoc dung de dem so lan hoan doi dong
		count = 0

		cur_row = 0
		# buoc khu $k \in [0, so cot)$
		for k in range(mat.shape[1]):
			if cur_row >= mat.shape[0]:	# ket thuc thuat toan khi da duyet het tat ca cac dong cua ma tran
				break

			# Find desired `p` (max row)
			p = cur_row
			# iterrate `i`: `cur_row <= i < num_row` to find max row
			for i in range(cur_row, mat.shape[0]):
				if abs(mat[p][k]) < abs(mat[i][k]):
					p = i	# update max row

			# swap `r_k` <-> `r_p`
			if cur_row != p:
				mat[cur_row], mat[p] = mat[p], mat[cur_row]

			# `mat[cur_row][k] == 0` ->> skip column `k`
			if is_zero(mat[cur_row][k]):
				continue
			
			# if `pivot != 1` (`mat[cur_row][k]` is `pivot`) then divide the current row by `pivot`, the `pivot`'s value after will be 1
			pivot = mat[cur_row][k]
			if not is_zero(pivot - 1):	
				mat[cur_row] = (1/pivot)*mat[cur_row]

			# eliminate values in `pivot`'s column for all rows except the row of current pivot
			for r in range(mat.shape[0]):
				if r == cur_row:	# don't have to eliminate row of pivot
					continue
				factor = mat[r][k]/mat[cur_row][k]
				mat[r] = mat[r] + (-factor*mat[cur_row])
			cur_row += 1
		return mat
	
	def gauss_eliminate(self) -> Tuple[Matrix, int]:
		"""
		Khu Gauss co chon phan tu chot (Partial Pivoting)
		Return 2-tuple: (ma tran ref, so lan hoan doi dong)
		"""
		# clone: mat <- self
		mat = Matrix.from_vecs(self.rows, is_col=False)
		# bien count duoc dung de dem so lan hoan doi dong
		count = 0

		cur_row = 0
		# buoc khu $k \in [0, so cot)$
		for k in range(mat.shape[1]):
			if cur_row >= mat.shape[0]:	# ket thuc thuat toan khi da duyet het tat ca cac dong cua ma tran
				break

			# Find desired `p` (max row)
			p = cur_row
			# iterrate `i`: `cur_row <= i < num_row` to find max row
			for i in range(cur_row, mat.shape[0]):
				if abs(mat[p][k]) < abs(mat[i][k]):
					p = i	# update max row

			# swap `r_k` <-> `r_p`
			if cur_row != p:
				mat[cur_row], mat[p] = mat[p], mat[cur_row]
				count += 1

			# `mat[cur_row][k] == 0` ->> skip column `k`
			if is_zero(mat[cur_row][k]):
				continue
			
			pivot = mat[cur_row][k]

			# eliminate values in `pivot`'s column for all rows below current_row
			for r in range(cur_row + 1, mat.shape[0]):
				factor = mat[r][k]/mat[cur_row][k]
				mat[r] = mat[r] + (-factor*mat[cur_row])
			cur_row += 1
		return (mat, count)

	def det(self):
		# check 
		if self.shape[0] != self.shape[1]:
			return None
		
		mat, s = self.gauss_eliminate() 
	
		n = self.shape[0]
		res = 1.0
	
		for i in range(n):
			res *= mat[i][i]
		res *= (-1)**s
		return res

	def inverse(self):
		"""
		Tra ve ma tran nghich dao cua `self`
		Su dung phuong phap khu Gauss-Jordan
		"""
		if is_zero(self.det()):
			return None
		
		aug_mat = self.augment()
		rref = aug_mat.gauss_jordan_eliminate()
		return rref.take_cols(self.shape[1], self.shape[1]*2 - 1)

	# --- OPERATOR OVERLOADING ---
	def __mul__(self, other):
		if self.shape[1] != other.shape[0]:
			raise ValueError(f"Không thể nhân ma trận: {self.shape[0]}x{self.shape[1]} và {other.shape[0]}x{other.shape[1]}")
		
		if isinstance(other, Matrix):
			result_data = [
				[
					sum(self[i][k] * other[k][j] for k in range(self.shape[1]))
					for j in range(other.shape[1])
				]
				for i in range(self.shape[0])
			]
			return Matrix(result_data)
		elif isinstance(other, Vector):
			result_data = [
				sum(self[i][k] * other[k] for k in range(self.shape[1]))
				for i in range(self.shape[0])
			]
			return Vector(result_data)
		else:
			raise TypeError("Kiểu của hạng tử không được hỗ trợ cho toán tử *")

	def __add__(self, other):
		if self.shape != other.shape:
			raise ValueError(f"Không thể cộng 2 ma trận khác kích thước: {self.shape[0]}x{self.shape[1]} và {other.shape[0]}x{other.shape[1]}")

		rvecs = [self[i] + other[i] for i in range(self.shape[0])]	
		return Matrix.from_vecs(rvecs, is_col=False)
	
	def __sub__(self, other):
		if self.shape != other.shape:
			raise ValueError(f"Không thể cộng 2 ma trận khác kích thước: {self.shape[0]}x{self.shape[1]} và {other.shape[0]}x{other.shape[1]}")

		rvecs = [self[i] - other[i] for i in range(self.shape[0])]	
		return Matrix.from_vecs(rvecs, is_col=False)


# <<-----TESTING----->> #
if __name__ == "__main__":
	mat = Matrix([
		[1,2,3],
		[6,5,4]
	])

	print(f"mat.rows = {mat.rows}", sep="\n")
	print(f"mat =\n{mat}", sep="\n")
	print(f"mat^T =\n{mat.transpose()}", sep="\n")
	print(f"mat * mat^T =\n{mat * mat.transpose()}", sep="\n")
	print(f"mat is_ref = {mat.is_ref()}", sep="\n")
	print(f"mat is_identity = {mat.is_identity()}", sep="\n")
	print(f"mat is_diagonal = {mat.is_diagonal()}", sep="\n")
	print(f"rref of mat^t =\n{mat.transpose().gauss_jordan_eliminate()}")

	iden = Matrix.identity(4)
	print(f"iden =\n{iden}", sep="\n")
	print(f"iden is_ref = {iden.is_ref()}", sep="\n")
	print(f"iden is_identity = {iden.is_identity()}", sep="\n")
	print(f"iden is_diagonal = {iden.is_diagonal()}", sep="\n")

	mat_cols = Matrix.from_vecs([Vector((1,2,3)), Vector((6,7,4))])
	print(f"mat_cols =\n{mat_cols}", sep="\n")

	mat_cols = Matrix([[1,2,3], [6,7,4]], by_cols=True)
	print(f"mat_cols =\n{mat_cols}", sep="\n")

	aug = mat_cols.augment()
	print(f"[mat_cols | I] =\n{aug}", sep="\n")

	aug = mat_cols.augment(mat.transpose())
	print(f"[mat_cols | mat^T] =\n{aug}", sep="\n")

	aug_cols_2_4 = aug.take_cols([1,3])
	print(f"A = matrix from column 2 and 4 of [mat_cols | mat^T] =\n{aug_cols_2_4}", sep="\n")

	sub_A_matT = aug_cols_2_4 - mat.transpose()
	print(f"A - mat^T =\n{sub_A_matT}")

	add_mat_mat = mat + mat
	print(f"mat + mat =\n{add_mat_mat}")

	v1 = Vector((1,2,3))
	v2 = Vector((3,2,1))
	print(f"v1 = {v1}")
	print(f"v2 = {v2}")
	print(f"v1 + v2 = {v1+v2}")
	print(f"v1 - v2 = {v1-v2}")
	print(f"v1 . v2 = {v1*v2}")
	print(f"2 * v1 = {2*v1}")

	inv = Matrix([
		[-1/3, 2/3],
		[1, 1]
	])
	print(f"[inv | I]=\n{inv.augment()}")
	print(f"inv^(-1) =\n{inv.inverse()}")
	v_r2 = Vector((3,3))
	print(f"inv * v_r2 =\n{inv*v_r2}")
