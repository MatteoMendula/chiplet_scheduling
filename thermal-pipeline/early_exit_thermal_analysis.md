# Early Exit vs. Thermal Throttling: MAESTRO + LTI + ThermLeT

This document describes the analysis that checks whether early exit (EE) in LLaVA-7B helps avoid thermal throttling on edge platforms (NVIDIA Jetson Orin Nano and AGX Xavier). It covers how the analysis evolved, the current model, how to run it, the preliminary results and the next steps.

Main script: `lti-maestro-ee.py` · Configuration: `hw_config_ee.json` · Output: `results_ee/`

> **Status:** every hardware parameter is currently an **indicative estimate** (datasheet or literature order of magnitude). The numbers below show how the pipeline behaves. They are not measured results, and should not be quoted until the parameters are calibrated with `tegrastats` (see [Next steps](#6-next-steps)).

---

## 1. Thesis claim under test

LLaVA-7B-EE has three intermediate exit heads (`EARLY_EXIT_HEAD_8/16/24`) and the final head (`FINAL_LM_HEAD`). In the MAESTRO profile, compute power is nearly constant across layers: about 3376 mW for every layer except `PATCH_EMBED`, at 4427 mW. Early exit therefore does **not** lower peak power. What it lowers is the **time and energy** spent heating the chip.

So the claim is framed as:

> With realistic thermal inertia, exiting early shortens the heat exposure. If the exit happens before the time at which the temperature would cross the throttling limit, the inference finishes without throttling. In a continuous stream, early exit lowers the duty cycle and therefore the operating temperature.

Position of each exit along the MAESTRO profile (cycles only, before the memory-bound correction):

| Exit | Layer index | Share of total cycles |
|---|---|---|
| EE-8 | 202 | 27% |
| EE-16 | 259 | 51% |
| EE-24 | 316 | 76% |
| Full | 373 | 100% |

---

## 2. How the analysis evolved

| Step | File | Outcome |
|---|---|---|
| Baseline | `lti-maestro.py` | LTI model with a hand-set ceiling of 72 °C. Left unchanged. |
| ThermLeT checkpoint | `ThermLeT/Fusion_pipeline/experiment/exp_1/checkpoints/best.pth` | The training dataset is not downloadable. Pre-trained weights come from the GitHub release `MicroSystemsDesignLab/ThermLeT`, tag `model_weights` (about 1.2 GB). |
| Ceiling from ThermLeT | `lti-maestro-thermlet.py` | First version gave a ceiling of 91,126 °C because raw mW values were fed to the model. Normalising power to ~1 fixed the scale. The script was later extended with `hw_config.json` and DRAM power; a variable-ordering `NameError` was fixed. |
| Diagnosis | (same) | With `hw_config.json`, **all 374 layers throttled**. The causes are listed below. |
| Redesign | `lti-maestro-ee.py`, `hw_config_ee.json` | Current version, described in this document. |

Why the previous setup throttled every layer:

1. **τ was far too small.** 5e7 cycles is about 80 ms at 625 MHz, while a layer lasts about 1.5e8 cycles on average. Every layer reached steady state, so the model had no thermal memory and the timing of the exit could not matter.
2. **R and DRAM power were too high.** 12 °C/W with 10–27 W total gives a steady state of 160–360 °C. DRAM peaked at 19.2 W because the MAESTRO configuration requests about 200 GB/s, against 68 GB/s available on an Orin Nano.
3. **The ceiling was circular.** It was a ThermLeT prediction on the *same* layer power that was compared against an RC estimate of that same power. Two estimates of the same quantity were compared, not a temperature against a limit.
4. **The scales did not match.** ThermLeT outputs about 46 °C even at zero power, because it was trained on chiplet packages and not on Jetson boards, so its absolute temperatures do not apply to these platforms.

---

## 3. Model (`lti-maestro-ee.py`)

### 3.1 Per-layer time and power

For each MAESTRO layer:

```
t_comp   = cycles / clock_freq_hz
t_mem    = (OffchipBW [elem/cycle] · cycles · 2 B) / dram_bw          # FP16
dt       = max(t_comp, t_mem)                                       # memory-bound if DRAM cannot keep up
E_comp   = Power_MAESTRO [W] · t_comp
E_dram   = bytes · 8 · dram_pj_per_bit
P_total  = E_comp/dt + E_dram/dt + idle_power_w
```

Capping DRAM traffic at the real bandwidth makes many layers memory-bound. As a result the Orin Nano's full inference takes about 271 s, not the 92 s implied by cycles alone.

### 3.2 Thermal model

Single-pole LTI model:

```
T_ss  = T_amb + R · P_total + ΔT_hotspot
T_end = T_ss + (T_start − T_ss) · exp(−dt / τ)
```

- **Fixed limit `T_lim`**, the platform throttling threshold. It replaces the dynamic ceiling.
- **ThermLeT gives only the hotspot correction:** `ΔT_hotspot = max(pred) − mean(pred over the chiplet footprint)`. ThermLeT is used for the spatial gradient, not for absolute temperature. Power is normalised to the most power-hungry layer, because the model was trained on inputs of order ~1. Results are cached by power level, so only a few inferences run. With the current settings the correction is about 2.5–2.7 °C.
- **Throttling (DVFS):** if `T_end > T_lim`, the layer is recomputed with compute power ×0.5 and compute time ×1.5. Memory time is unchanged.

### 3.3 Analytical criterion

At a constant mean power P, the time at which the temperature crosses the limit, starting from T_amb, is:

```
t* = −τ · ln( (T_ss − T_lim) / (T_ss − T_amb) )      (t* = ∞ if T_ss ≤ T_lim)
```

An exit at layer k avoids throttling if its cumulative time is below t*. This closed form is the main analytical result for the thesis. The simulation checks it, including the per-layer variation in power.

### 3.4 Experiments

1. **Single inference.** T(t) for each exit, starting from T_amb, with `T_lim` and `t*` drawn on the plot.
2. **Stream.** N inferences issued at a fixed period, with the platform idling between them. The default period is 1.1 × the Full latency without throttling. The output reports peak temperature, throttled layers and deadline misses.
3. **R × τ sensitivity.** A 16×16 grid over R ∈ [0.5, 2]·R_nom and τ ∈ [0.1, 10]·τ_nom, both around the nominal values. Each cell shows the latest exit that finishes a single inference without throttling.

---

## 4. Configuration (`hw_config_ee.json`)

| Parameter | Unit | Orin Nano | AGX Xavier | Source / status |
|---|---|---|---|---|
| `description` | — | 8GB, 15W mode, passive heatsink in enclosure | 32GB, 30W mode, heatsink + fan | assumed scenario |
| `clock_freq_hz` | Hz | 625e6 | 1377e6 | max GPU clock (datasheet) |
| `dram_bw_gbs` | GB/s | 68 | 137 | datasheet |
| `dram_pj_per_bit` | pJ/bit | 8 | 10 | LPDDR5 / LPDDR4x order of magnitude, **estimate** |
| `idle_power_w` | W | 2.5 | 5.0 | **estimate**, measure |
| `R_c_per_w` | °C/W | 5.0 | 1.55 | **estimate**, fit from tegrastats |
| `tau_s` | s | 150 | 300 | **estimate**, fit from tegrastats |
| `T_amb_c` | °C | 45 | 45 | assumed (enclosure) |
| `T_lim_c` | °C | 95 | 95 | **to verify** in the Jetson Thermal Design Guide |
| `footprint_px` | px on 224×224 | 70 | 84 | carried over from `hw_config.json` |

Units differ from the old `hw_config.json`: power is in W (not mW), R is in °C/W and τ is in seconds (not cycles).

---

## 5. Usage and preliminary results

```bash
conda activate thermlet
cd thermal-pipeline
python lti-maestro-ee.py                          # all platforms
python lti-maestro-ee.py --platforms xavier_agx   # a single platform
python lti-maestro-ee.py --stream-period 60 --stream-n 10
python lti-maestro-ee.py --no-thermlet            # LTI only, no hotspot correction
python lti-maestro-ee.py --hw-config my_measured_config.json
```

Run it on CPU, which is the default. The installed PyTorch build does not support the RTX PRO 4500 Blackwell (sm_120), so `--device cuda` fails. The environment needs `numpy==1.26.4` and `pandas==2.2.3`, because NumPy 2 breaks the installed torch.

Outputs in `results_ee/`: `single_<platform>.png`, `stream_<platform>.png`, `sensitivity_<platform>.png`, `summary.csv`. Each run overwrites the plots of the platforms it runs, and rewrites `summary.csv` with only those platforms.

### Results with the indicative parameters

Single inference:

| Platform | Exit | Latency (s) | Energy (J) | T_peak (°C) | Throttled layers |
|---|---|---|---|---|---|
| Orin Nano | EE-8 | 73.1 | 584 | 61.5 | 0 / 203 |
| Orin Nano | EE-16 | 138.9 | 1112 | 70.8 | 0 / 260 |
| Orin Nano | EE-24 | 204.8 | 1639 | 76.8 | 0 / 317 |
| Orin Nano | Full | 270.7 | 2166 | 80.7 | 0 / 374 |
| AGX Xavier | EE-8 | 36.3 | 617 | 48.3 | 0 / 203 |
| AGX Xavier | EE-16 | 69.0 | 1173 | 50.9 | 0 / 260 |
| AGX Xavier | EE-24 | 101.7 | 1729 | 53.3 | 0 / 317 |
| AGX Xavier | Full | 134.4 | 2285 | 55.4 | 0 / 374 |

Stream of 6 inferences (period 298 s on Orin Nano, 148 s on Xavier), T_peak in °C:

| Platform | EE-8 | EE-16 | EE-24 | Full |
|---|---|---|---|---|
| Orin Nano | 71.0 | 78.7 | 83.6 | 86.8 |
| AGX Xavier | 58.0 | 62.8 | 67.0 | 70.9 |

No configuration throttles and there are no deadline misses.

### What these results show

- Early exit lowers peak temperature consistently. On Orin Nano, EE-8 stays about 19 °C below Full for a single inference and about 16 °C below in the stream. Energy scales with the exit point.
- With the nominal parameters, **neither platform reaches `T_lim`**, so throttling is not triggered and the claim cannot be tested at this operating point.
- The Orin Nano sensitivity map shows that the nominal point (R = 5 °C/W, τ = 150 s) sits just outside the interesting region. Above about 6 °C/W, which corresponds to worse passive cooling or a hotter enclosure, there is a band of τ values in which early exits finish without throttling and Full does not. At small τ and high R, even EE-8 throttles.
- The thesis claim therefore depends on the thermal regime of the platform. The sensitivity map is the right way to present it: it shows where early exit prevents throttling and where it only adds thermal margin.

### Known limitations

- MAESTRO compute power is the same for both platforms. It describes a 256-PE accelerator, not the Jetson GPUs. Only clock, bandwidth and thermal parameters differ.
- The thermal model has a single pole. A real board has at least two time constants: die (ms) and heatsink/package (tens to hundreds of seconds).
- ThermLeT is trained on 2.5D chiplet packages, so its hotspot correction is qualitative only.
- DVFS is crude (×0.5 power, ×1.5 time). Real Jetson throttling follows discrete frequency steps.
- The EE heads are assumed to sit inline on the full path, so Full also computes them. The choice of exit is fixed per run; there is no confidence-based exit policy.

---

## 6. Next steps

1. **Calibrate with `tegrastats` on each board**, which is a priority.
   - Run continuous inference from a cold start at a fixed power mode (`nvpmodel`, with `jetson_clocks` both on and off) and log `tegrastats --interval 100`.
   - Fit `T(t) = T_amb + ΔT·(1 − e^(−t/τ))` to obtain **τ**, then compute **R = ΔT / P_steady**, using the rail power (VDD_IN) from tegrastats.
   - Measure **idle power** with the board idle in the same power mode.
   - Read **T_lim** (software throttling threshold) from the Jetson Thermal Design Guide, or from `/sys/class/thermal/thermal_zone*/trip_point_*`.
   - Record the cooling setup and ambient temperature, then write the values to a new config such as `hw_config_measured.json`.
2. **Replace MAESTRO compute power** with per-layer or per-block power measured on the board, or at least scale MAESTRO power to the measured average.
3. **Validate the model.** Compare the simulated T(t) for a full inference against the measured trace, and report the RMSE.
4. **Throttling model.** Replace the ×0.5/×1.5 rule with the real Jetson DVFS frequency steps.
5. **Two-pole thermal model** (die + heatsink), if the measured trace shows a fast component that the single-pole model misses.
6. **Exit policy.** Move from a fixed exit to a confidence-based or thermally aware exit: choose the exit at run time from the current T and t*. Report the accuracy/temperature trade-off using the accuracy of each EE head.
7. **Thesis figures:** the sensitivity map with the measured operating point, single-inference T(t) with t*, and the stream at different request periods (duty cycles).
