import numpy as np
from typing import Optional
from .bits import uniform_fixed_point_encode, decompose_to_bitplanes
from .charge import ChargeAccumulator
from .fwht import fwht as fwht_ideal


class ChargeCimHadamard:
	"""Bit-serial charge-sharing engine for Hadamard transform.

	Workflow:
	1) Quantize input to fixed-point, decompose to bitplanes (two's complement).
	2) For each bitplane (LSB->MSB), compute FWHT contribution scaled by that bit's weight and accumulate via ChargeAccumulator.
	3) Return the accumulated readout (analog sum of weighted contributions).

	This is a pedagogical model; it is not a device-accurate circuit simulator.
	"""

	def __init__(
		self,
		n: int,
		num_int_bits: int = 6,
		num_frac_bits: int = 10,
		capacitance_f: float = 1e-12,
		temperature_k: float = 300.0,
		wordline_alpha: float = 0.0,
		bitline_alpha: float = 0.0,
		leak_decay: float = 0.0,
		rng: Optional[np.random.Generator] = None,
	):
		if n <= 0 or (n & (n - 1)) != 0:
			raise ValueError("n must be a power of two and > 0")
		self.n = n
		self.num_int_bits = int(num_int_bits)
		self.num_frac_bits = int(num_frac_bits)
		self.total_bits = self.num_int_bits + self.num_frac_bits
		self.rng = rng if rng is not None else np.random.default_rng()
		self.accum = ChargeAccumulator(
			capacitance_f=capacitance_f,
			temperature_k=temperature_k,
			wordline_alpha=wordline_alpha,
			bitline_alpha=bitline_alpha,
			leak_decay=leak_decay,
			rng=self.rng,
		)

	def apply(self, x: np.ndarray) -> np.ndarray:
		x = np.asarray(x, dtype=np.float64)
		if x.ndim == 1:
			x = x[None, :]
		if x.shape[1] != self.n:
			raise ValueError("Input length must match n")

		# 1) Fixed-point quantization
		qvals, lsb = uniform_fixed_point_encode(x, self.num_int_bits, self.num_frac_bits)
		bitplanes = decompose_to_bitplanes(qvals, self.total_bits)  # [B, batch, n]

		# Per-bit weights (LSB positive, MSB negative for two's complement)
		weights = np.array([2 ** b for b in range(self.total_bits)], dtype=np.float64) * lsb
		weights[-1] *= -1.0

		# 2) Accumulate weighted FWHT contributions
		self.accum.v_acc = np.zeros_like(x, dtype=np.float64)
		for b in range(self.total_bits):
			plane = bitplanes[b].astype(np.float64)
			v_b = fwht_ideal(plane) * weights[b]
			self.accum.step(v_b)

		# 3) Readout accumulated result
		v_out = self.accum.readout()
		return v_out if v_out.shape[0] > 1 else v_out[0]
