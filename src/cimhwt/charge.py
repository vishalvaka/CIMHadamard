import numpy as np
from typing import Optional


k_B = 1.380649e-23  # J/K


class ChargeAccumulator:
	"""Charge-sharing accumulator for bit-serial compute.

	This models per-step charge updates with:
	- Wordline and bitline attenuation factors
	- Thermal noise ~ N(0, sqrt(kT/C)) in voltage domain
	- Optional leakage decay per step
	"""

	def __init__(
		self,
		capacitance_f: float = 1e-12,
		temperature_k: float = 300.0,
		wordline_alpha: float = 0.0,
		bitline_alpha: float = 0.0,
		leak_decay: float = 0.0,
		rng: Optional[np.random.Generator] = None,
	):
		self.C = float(capacitance_f)
		self.T = float(temperature_k)
		self.wordline_alpha = float(wordline_alpha)
		self.bitline_alpha = float(bitline_alpha)
		self.leak = float(leak_decay)
		self.rng = rng if rng is not None else np.random.default_rng()

	def step(self, v_in: np.ndarray) -> np.ndarray:
		"""Accumulate one step with applied input v_in (shape: [batch, n]).

		Returns updated accumulator voltage.
		"""
		batch, n = v_in.shape
		# Bitline attenuation across columns
		cols = np.arange(n)
		bitline_scale = 1.0 - self.bitline_alpha * (cols / max(1, n - 1))
		v = v_in * bitline_scale
		# Wordline attenuation across rows (batch dimension)
		rows = np.arange(batch)
		wordline_scale = 1.0 - self.wordline_alpha * (rows / max(1, batch - 1))
		v = v * wordline_scale[:, None]

		# Apply leakage decay on stored node
		if not hasattr(self, "v_acc"):
			self.v_acc = np.zeros((batch, n), dtype=np.float64)
		else:
			self.v_acc *= (1.0 - self.leak)

		# Add contribution (voltage-domain approximation)
		v_next = self.v_acc + v

		# Add kT/C thermal noise per node per step
		sigma = np.sqrt(k_B * self.T / max(self.C, 1e-30))
		noise = self.rng.normal(0.0, sigma, size=v_next.shape)
		v_next = v_next + noise

		self.v_acc = v_next
		return self.v_acc

	def readout(self) -> np.ndarray:
		return self.v_acc
