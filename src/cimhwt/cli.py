import argparse
import numpy as np
from .fwht import fwht
from .cim import CimArray
from .engine_charge import ChargeCimHadamard
from .engine_xbar import XbarHadamard


def main():
	p = argparse.ArgumentParser(description="CIM Hadamard Transform Simulator")
	p.add_argument("--size", type=int, default=256, help="Transform size (power of two)")
	p.add_argument("--engine", type=str, default="adc", choices=["adc", "charge", "xbar"], help="Engine/model to use")

	# ADC engine params
	p.add_argument("--adc-bits", type=int, default=8)
	p.add_argument("--adc-clip", type=float, default=None)
	p.add_argument("--noise-sigma", type=float, default=0.0)
	p.add_argument("--ir-drop-alpha", type=float, default=0.0)
	p.add_argument("--gain", type=float, default=1.0)
	p.add_argument("--offset", type=float, default=0.0)

	# Charge engine params
	p.add_argument("--int-bits", type=int, default=6, help="Integer bits for input encoding")
	p.add_argument("--frac-bits", type=int, default=10, help="Fractional bits for input encoding")
	p.add_argument("--cap-f", type=float, default=1e-12, help="Accumulator capacitance in Farads")
	p.add_argument("--temp-k", type=float, default=300.0, help="Temperature in Kelvin")
	p.add_argument("--wl-alpha", type=float, default=0.0, help="Wordline attenuation factor")
	p.add_argument("--bl-alpha", type=float, default=0.0, help="Bitline attenuation factor")
	p.add_argument("--leak", type=float, default=0.0, help="Leakage decay per step (0..1)")

	# Xbar engine params
	p.add_argument("--x-g0", type=float, default=10e-6, help="Unit conductance per +1 entry (Siemens)")
	p.add_argument("--x-dac-gain", type=float, default=1.0, help="DAC gain (V per numeric unit)")
	p.add_argument("--x-rf", type=float, default=100e3, help="TIA feedback resistance (Ohms)")
	p.add_argument("--x-adc-bits", type=int, default=10, help="ADC bits in sense path")
	p.add_argument("--x-adc-clip", type=float, default=None, help="ADC full-scale (Volts)")
	p.add_argument("--x-noise", type=float, default=0.0, help="Sense voltage noise sigma (Volts)")
	p.add_argument("--x-wl-alpha", type=float, default=0.0, help="WL attenuation (row 1 scaling)")
	p.add_argument("--x-bl-alpha", type=float, default=0.0, help="BL attenuation across j in block")

	p.add_argument("--repeat", type=int, default=1, help="Repeat runs and average metrics")
	p.add_argument("--seed", type=int, default=123)
	args = p.parse_args()

	rng = np.random.default_rng(args.seed)

	if args.engine == "adc":
		engine = CimArray(
			n=args.size,
			gain=args.gain,
			offset=args.offset,
			noise_sigma=args.noise_sigma,
			ir_drop_alpha=args.ir_drop_alpha,
			adc_bits=args.adc_bits,
			adc_clip=args.adc_clip,
			rng=rng,
		)
	elif args.engine == "charge":
		engine = ChargeCimHadamard(
			n=args.size,
			num_int_bits=args.int_bits,
			num_frac_bits=args.frac_bits,
			capacitance_f=args.cap_f,
			temperature_k=args.temp_k,
			wordline_alpha=args.wl_alpha,
			bitline_alpha=args.bl_alpha,
			leak_decay=args.leak,
			rng=rng,
		)
	else:
		engine = XbarHadamard(
			n=args.size,
			g0=args.x_g0,
			dac_gain=args.x_dac_gain,
			rf=args.x_rf,
			adc_bits=args.x_adc_bits,
			adc_clip=args.x_adc_clip,
			noise_sigma=args.x_noise,
			wl_alpha=args.x_wl_alpha,
			bl_alpha=args.x_bl_alpha,
			rng=rng,
		)

	rmses = []

	for _ in range(args.repeat):
		x = rng.standard_normal(args.size)
		y_ideal = fwht(x)
		y_model = engine.apply(x)

		rmse = np.sqrt(np.mean((y_ideal - y_model) ** 2))
		psnr = 20 * np.log10(np.max(np.abs(y_ideal)) / (rmse + 1e-12))
		rmses.append((rmse, psnr))

	mean_rmse = float(np.mean([r for r, _ in rmses]))
	mean_psnr = float(np.mean([p for _, p in rmses]))

	print(f"engine={args.engine} N={args.size}")
	print(f"RMSE={mean_rmse:.6g} PSNR={mean_psnr:.3f} dB")


if __name__ == "__main__":
	main()
