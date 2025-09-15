import numpy as np
from typing import Optional


class CimArray:
	"""Simple CIM crossbar model for applying Hadamard transform.

	Models: gain, offset, additive Gaussian noise, IR drop (scale per row), and ADC quantization.
	"""

	def __init__(
		self,
		n: int,
		gain: float = 1.0,
		offset: float = 0.0,
		noise_sigma: float = 0.0,
		ir_drop_alpha: float = 0.0,
		adc_bits: int = 8,
		adc_clip: Optional[float] = None,
		rng: Optional[np.random.Generator] = None,
	):
		if n <= 0 or (n & (n - 1)) != 0:
			raise ValueError("n must be a power of two and > 0")
		self.n = n
		self.gain = float(gain)
		self.offset = float(offset)
		self.noise_sigma = float(noise_sigma)
		self.ir_drop_alpha = float(ir_drop_alpha)
		self.adc_bits = int(adc_bits)
		if self.adc_bits <= 0:
			raise ValueError("adc_bits must be > 0")
		self.adc_clip = float(adc_clip) if adc_clip is not None else None
		self.rng = rng if rng is not None else np.random.default_rng()

	def apply(self, x: np.ndarray) -> np.ndarray:
		x = np.asarray(x, dtype=np.float64)
		if x.ndim == 1:
			x = x[None, :]
		if x.shape[1] != self.n:
			raise ValueError("Input length must match n")

		y = x.copy()
		h = 1
		stage = 0
		while h < self.n:
			for i in range(0, self.n, h * 2):
				left = y[:, i:i+h]
				right = y[:, i+h:i+2*h]

				sum_out = left + right
				diff_out = left - right

				sum_out = self._nonideal(sum_out, stage)
				diff_out = self._nonideal(diff_out, stage)

				y[:, i:i+h] = sum_out
				y[:, i+h:i+2*h] = diff_out

			h *= 2
			stage += 1

		return y if y.shape[0] > 1 else y[0]

	def _nonideal(self, v: np.ndarray, stage: int) -> np.ndarray:
		n_cols = v.shape[1]
		col_idx = np.arange(n_cols)
		ir_scale = 1.0 - self.ir_drop_alpha * (col_idx / max(1, n_cols - 1))
		v = v * ir_scale

		v = self.gain * v + self.offset

		if self.noise_sigma > 0.0:
			v = v + self.rng.normal(0.0, self.noise_sigma, size=v.shape)

		if self.adc_bits is not None:
			vmax = self.adc_clip if self.adc_clip is not None else np.max(np.abs(v)) + 1e-12
			v = np.clip(v, -vmax, vmax)
			levels = 2 ** self.adc_bits
			step = 2 * vmax / (levels - 1)
			v = np.round((v + vmax) / step) * step - vmax

		return v
