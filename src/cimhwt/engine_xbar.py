import numpy as np
from typing import Optional


class XbarHadamard:
	"""Explicit crossbar-style Hadamard engine with G+/G-, DAC, TIA, ADC.

	For each butterfly pair at each stage, we emulate a 2x2 block:
	- Rows: the two inputs (voltages from DAC)
	- Columns (outputs): sum (+1,+1) and diff (+1,-1) via differential G+/G-
	Pipeline per pair:
	1) DAC: map numeric inputs to row voltages V_in = dac_gain * x
	2) WL attenuation applied across the two rows
	3) Column currents per output from Ohm's law and differential columns:
	   I_sum = g0*(V0 + V1), I_diff = g0*(V0 - V1)
	4) BL attenuation (per output column position within the block)
	5) Sense with TIA: V_out = I * rf
	6) Add Gaussian noise (sense voltage domain)
	7) ADC quantization with bits and clip
	8) Convert back to numeric domain for the next stage by dividing by (rf * g0 * dac_gain)
	"""

	def __init__(
		self,
		n: int,
		g0: float = 10e-6,           # Siemens per +1 entry
		dac_gain: float = 1.0,       # Volts per numeric unit
		rf: float = 100e3,           # Ohms (TIA)
		adc_bits: int = 10,
		adc_clip: Optional[float] = None,
		noise_sigma: float = 0.0,    # Volts, sense domain
		wl_alpha: float = 0.0,
		bl_alpha: float = 0.0,
		rng: Optional[np.random.Generator] = None,
	):
		if n <= 0 or (n & (n - 1)) != 0:
			raise ValueError("n must be a power of two and > 0")
		self.n = n
		self.g0 = float(g0)
		self.dac_gain = float(dac_gain)
		self.rf = float(rf)
		self.adc_bits = int(adc_bits)
		self.adc_clip = float(adc_clip) if adc_clip is not None else None
		self.noise_sigma = float(noise_sigma)
		self.wl_alpha = float(wl_alpha)
		self.bl_alpha = float(bl_alpha)
		self.rng = rng if rng is not None else np.random.default_rng()

	def apply(self, x: np.ndarray) -> np.ndarray:
		x = np.asarray(x, dtype=np.float64)
		if x.ndim == 1:
			x = x[None, :]
		if x.shape[1] != self.n:
			raise ValueError("Input length must match n")

		y = x.copy()
		h = 1
		while h < self.n:
			# For each 2h block, process h butterfly pairs explicitly
			for i in range(0, self.n, h * 2):
				left = y[:, i:i+h]
				right = y[:, i+h:i+2*h]

				# Process each column j within the block as a 2x2 crossbar
				for j in range(h):
					# 1) DAC: numeric to voltage on the two rows
					v0 = self.dac_gain * left[:, j]
					v1 = self.dac_gain * right[:, j]

					# 2) WL attenuation across the two rows
					# scale row 0 by 1.0, row 1 by (1 - wl_alpha)
					v0 = v0 * 1.0
					v1 = v1 * (1.0 - self.wl_alpha)

					# 3) Differential columns implement sum and diff
					# I_sum = g0*(+1*v0 + +1*v1), I_diff = g0*(+1*v0 + -1*v1)
					I_sum = self.g0 * (v0 + v1)
					I_diff = self.g0 * (v0 - v1)

					# 4) BL attenuation across outputs within the block
					# position index 0..(h-1) maps to scale 1 - bl_alpha * (j/(h-1)) if h>1
					if h > 1:
						scale_bl = 1.0 - self.bl_alpha * (j / (h - 1))
					else:
						scale_bl = 1.0
					I_sum *= scale_bl
					I_diff *= scale_bl

					# 5) TIA: current to voltage
					V_sum = I_sum * self.rf
					V_diff = I_diff * self.rf

					# 6) Add noise in sense voltage domain
					if self.noise_sigma > 0.0:
						V_sum = V_sum + self.rng.normal(0.0, self.noise_sigma, size=V_sum.shape)
						V_diff = V_diff + self.rng.normal(0.0, self.noise_sigma, size=V_diff.shape)

					# 7) ADC quantization with optional fixed clip
					V_sum_q = self._quantize(V_sum)
					V_diff_q = self._quantize(V_diff)

					# 8) Convert back to numeric domain for next stage
					gain = self.rf * self.g0 * self.dac_gain
					num_sum = V_sum_q / gain
					num_diff = V_diff_q / gain

					# Write back to the working vector (replace sum/diff slots)
					left[:, j] = num_sum
					right[:, j] = num_diff

			h *= 2

		# Return batch semantics
		return y if y.shape[0] > 1 else y[0]

	def _quantize(self, v: np.ndarray) -> np.ndarray:
		levels = 2 ** self.adc_bits
		if self.adc_clip is None:
			vmax = np.max(np.abs(v)) + 1e-12
		else:
			vmax = self.adc_clip
		v = np.clip(v, -vmax, vmax)
		step = 2 * vmax / (levels - 1)
		return np.round((v + vmax) / step) * step - vmax
