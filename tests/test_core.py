import numpy as np
from cimhwt.fwht import fwht


def test_fwht_roundtrip():
	rng = np.random.default_rng(0)
	for n in [2, 4, 8, 16, 32, 64]:
		x = rng.standard_normal(n)
		y = fwht(x)
		z = fwht(y)
		# Unnormalized FWHT, so applying twice yields N * x
		assert np.allclose(z, n * x, atol=1e-9)


def test_cim_matches_ideal_when_perfect():
	rng = np.random.default_rng(1)
	n = 64
	x = rng.standard_normal(n)

	from cimhwt.cim import CimArray

	cim = CimArray(
		n=n,
		gain=1.0,
		offset=0.0,
		noise_sigma=0.0,
		ir_drop_alpha=0.0,
		adc_bits=16,
		adc_clip=None,
		rng=rng,
	)

	y_ideal = fwht(x)
	y_cim = cim.apply(x)

	assert np.allclose(y_ideal, y_cim, atol=1e-9)
