# CIM Hadamard Transform (CIM-HWT)

A simulator and CLI for evaluating compute-in-memory (CIM) execution of the Hadamard transform versus an ideal software fast Walsh–Hadamard transform (FWHT).

- Ideal FWHT baseline (O(N log N))
- Three CIM engines:
  - adc: behavioral analog front-end + ADC per butterfly stage
  - charge: bit-serial, charge-sharing accumulator with kT/C noise
  - xbar: explicit crossbar-style engine with G+/G−, DAC, TIA, ADC
- Error metrics: RMSE and PSNR versus ideal FWHT

## Install
```bash
pip install -e .
```

## Quick start
Run the FWHT vs CIM comparison for size N (power of two). Results print averaged RMSE/PSNR.

- Behavioral ADC engine (near-ideal):
```bash
PYTHONPATH=src python -m cimhwt.cli --engine adc --size 256 \
  --adc-bits 12 --noise-sigma 0 --ir-drop-alpha 0
```

- Charge-sharing engine (very low noise):
```bash
PYTHONPATH=src python -m cimhwt.cli --engine charge --size 64 \
  --int-bits 6 --frac-bits 10 --cap-f 1e-9 --temp-k 1e-6 \
  --wl-alpha 0.0 --bl-alpha 0.0 --leak 0.0
```

- Explicit crossbar engine (near-ideal sensing/quantization):
```bash
PYTHONPATH=src python -m cimhwt.cli --engine xbar --size 32 \
  --x-g0 1e-5 --x-dac-gain 0.05 --x-rf 2e5 \
  --x-adc-bits 18 --x-adc-clip 2.0 --x-noise 0 \
  --x-wl-alpha 0 --x-bl-alpha 0
```

## Engines and what they simulate

### 1) adc engine (behavioral front-end + ADC)
- Pipeline per stage (per butterfly):
  1. Ideal sums/diffs (Hadamard butterfly)
  2. IR-drop-like scaling across columns in the local block
  3. Global gain/offset
  4. Additive Gaussian noise
  5. ADC quantization with uniform levels and optional fixed clip
- Purpose: Study quantization/noise/line-loss impact with a compact model.
- Parameters:
  - `--adc-bits`: ADC resolution (levels = 2^bits)
  - `--adc-clip`: Full-scale ±vmax; if omitted, range auto-adapts per slice
  - `--noise-sigma`: Additive Gaussian noise std (numeric units)
  - `--ir-drop-alpha`: Linear attenuation across columns inside each block
  - `--gain`, `--offset`: Global linear distortion before ADC

### 2) charge engine (bit-serial charge sharing)
- Pipeline:
  1. Fixed-point encode input with two’s-complement: `int_bits + frac_bits`
  2. Decompose to bitplanes (LSB→MSB)
  3. For each bitplane b:
     - Compute its FWHT contribution
     - Weight by bit significance (MSB negative)
     - Accumulate via a charge node (kT/C thermal noise, WL/BL attenuation, leakage per step)
  4. Read out the accumulated result
- Purpose: Pedagogical analog accumulation with realistic noise/leak/attenuation across bit-steps.
- Parameters:
  - `--int-bits`, `--frac-bits`: Fixed-point range/precision (LSB = 2^-frac_bits)
  - `--cap-f`, `--temp-k`: kT/C noise via σ ≈ sqrt(k·T/C)
  - `--wl-alpha`, `--bl-alpha`: Word/bit-line attenuation (row/column scaling)
  - `--leak`: Fractional decay per bit-step (0..1)

### 3) xbar engine (explicit crossbar-style per butterfly pair)
- For each 2×2 butterfly:
  1. DAC: numeric to voltages on two rows (V0, V1) via `x-dac-gain`
  2. WL attenuation across the two rows
  3. Differential columns realize sum/diff:
     - I_sum = g0·(V0 + V1), I_diff = g0·(V0 − V1)
  4. BL attenuation per output position
  5. TIA: V_out = I·Rf
  6. Add sense noise (voltage domain)
  7. ADC quantization (bits, clip)
  8. Convert back to numeric domain for next stage by dividing by (Rf·g0·dac_gain)
- Purpose: Make the physical path (DAC→WL/BL→G+/G− currents→TIA→ADC) explicit.
- Parameters:
  - `--x-g0`: Unit conductance for +1 entries (Siemens)
  - `--x-dac-gain`: DAC gain (V per numeric unit)
  - `--x-rf`: TIA feedback resistance (Ohms)
  - `--x-adc-bits`, `--x-adc-clip`: ADC bits and full-scale in the sense path (Volts)
  - `--x-noise`: Sense voltage noise std (Volts)
  - `--x-wl-alpha`, `--x-bl-alpha`: WL/BL attenuation

## CLI arguments (summary)
- Common: `--size`, `--repeat`, `--seed`
- adc: `--adc-bits`, `--adc-clip`, `--noise-sigma`, `--ir-drop-alpha`, `--gain`, `--offset`
- charge: `--int-bits`, `--frac-bits`, `--cap-f`, `--temp-k`, `--wl-alpha`, `--bl-alpha`, `--leak`
- xbar: `--x-g0`, `--x-dac-gain`, `--x-rf`, `--x-adc-bits`, `--x-adc-clip`, `--x-noise`, `--x-wl-alpha`, `--x-bl-alpha`

## Concepts (short)
- Quantization: Round to nearest ADC code. Step Δ ≈ 2·vmax/(2^bits−1). More bits → smaller Δ.
- PSNR: 20·log10(peak/RMSE). Higher is better. Peak is max|ideal FWHT| for the trial.
- WL/BL attenuation: Signals weaken along long wires; modeled as linear scaling across row/column index.
- kT/C noise: Thermal voltage noise on a capacitor, σ ≈ sqrt(k·T/C). Larger C or lower T reduces noise.
- Fixed-point: `int_bits` sets range ~ ±2^(int_bits−1), `frac_bits` sets LSB = 2^-frac_bits.

## Recommended presets
- Near-ideal adc:
```bash
PYTHONPATH=src python -m cimhwt.cli --engine adc --size 256 \
  --adc-bits 14 --adc-clip 20 --gain 1 --noise-sigma 0 --ir-drop-alpha 0
```
- Near-ideal charge:
```bash
PYTHONPATH=src python -m cimhwt.cli --engine charge --size 64 \
  --int-bits 6 --frac-bits 12 --cap-f 1e-9 --temp-k 1e-6 \
  --wl-alpha 0 --bl-alpha 0 --leak 0
```
- Near-ideal xbar:
```bash
PYTHONPATH=src python -m cimhwt.cli --engine xbar --size 32 \
  --x-g0 1e-5 --x-dac-gain 0.05 --x-rf 2e5 \
  --x-adc-bits 18 --x-adc-clip 2.0 --x-noise 0 \
  --x-wl-alpha 0 --x-bl-alpha 0
```

## Interpreting results
- If RMSE is high or PSNR is low:
  - adc: increase `--adc-bits`, set a sensible `--adc-clip`, adjust `--gain` to use full-scale
  - charge: increase `--cap-f` or lower `--temp-k`, reduce `--wl/--bl`, reduce `--leak`, increase bits
  - xbar: reduce `--x-dac-gain` or `--x-rf`, increase `--x-adc-bits`/`--x-adc-clip`, set `--x-noise 0`, zero `--x-wl/--x-bl`

## Dev notes
- Package: `cimhwt`
  - `fwht.py`: ideal FWHT
  - `cim.py`: adc engine
  - `engine_charge.py`: charge engine
  - `engine_xbar.py`: explicit crossbar engine
  - `bits.py`, `charge.py`: utilities
- Tests (optional):
```bash
PYTHONPATH=src pytest -q
```
