import numpy as np


def generate_hadamard_matrix(n: int) -> np.ndarray:
	"""Generate Sylvester-type Hadamard matrix of order n (n must be power of two).

	Entries are +1 and -1, dtype float64.
	"""
	if n <= 0 or (n & (n - 1)) != 0:
		raise ValueError("n must be a power of two and > 0")
	H = np.array([[1.0]])
	while H.shape[0] < n:
		H = np.block([[H, H], [H, -H]])
	return H
