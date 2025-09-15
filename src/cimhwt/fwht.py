import numpy as np


def fwht(x: np.ndarray) -> np.ndarray:
	"""In-place Fast Walsh-Hadamard Transform (FWHT).

	Returns a new array with the transform (unnormalized).
	Accepts 1D or 2D (batch as rows).
	"""
	x = np.asarray(x, dtype=np.float64)
	if x.ndim == 1:
		a = x.copy()
		_fwht_inplace(a)
		return a
	elif x.ndim == 2:
		a = x.copy()
		for i in range(a.shape[0]):
			_fwht_inplace(a[i])
		return a
	else:
		raise ValueError("Input must be 1D or 2D array")


def _fwht_inplace(a: np.ndarray) -> None:
	n = a.shape[0]
	if n == 0 or (n & (n - 1)) != 0:
		raise ValueError("Length must be power of two and > 0")
	h = 1
	while h < n:
		for i in range(0, n, h * 2):
			for j in range(i, i + h):
				x = a[j]
				y = a[j + h]
				a[j] = x + y
				a[j + h] = x - y
		h *= 2
