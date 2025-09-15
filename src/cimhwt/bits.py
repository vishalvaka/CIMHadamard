import numpy as np
from typing import Tuple


def uniform_fixed_point_encode(x: np.ndarray, num_int_bits: int, num_frac_bits: int, clip: float | None = None) -> Tuple[np.ndarray, float]:
	"""Encode x into signed fixed-point with given integer and fractional bits.

	Returns (quantized_values, scale). Values are in units of LSB (integers), scale maps back to real: real ≈ values * scale.
	If clip is provided, values are clipped to [-clip, clip] before quantization.
	"""
	x = np.asarray(x, dtype=np.float64)
	if clip is not None:
		x = np.clip(x, -clip, clip)

	num_levels = 2 ** (num_int_bits + num_frac_bits)
	# signed range: [-(2^{I}) , 2^{I}-2^{-F}]
	lsb = 2.0 ** (-num_frac_bits)
	max_real = (2 ** (num_int_bits - 1)) - lsb
	min_real = - (2 ** (num_int_bits - 1))
	# Work in integer code domain
	min_code = int(np.floor(min_real / lsb))
	max_code = int(np.floor(max_real / lsb))
	codes = np.clip(np.round(x / lsb), min_code, max_code).astype(np.int64)
	return codes, lsb


def decompose_to_bitplanes(values: np.ndarray, total_bits: int) -> np.ndarray:
	"""Decompose signed integers into bitplanes [bits, ...shape] using two's complement.

	values should be np.int64 in the range representable by total_bits.
	"""
	vals = values.astype(np.int64)
	# Mask to total_bits and work in unsigned to avoid sign-propagation on shifts
	mask = (np.uint64(1) << np.uint64(total_bits)) - np.uint64(1)
	uvals = (vals.astype(np.uint64)) & mask
	bitplanes = []
	for b in range(total_bits):
		bitplanes.append(((uvals >> np.uint64(b)) & np.uint64(1)).astype(np.uint8))
	return np.stack(bitplanes, axis=0)
