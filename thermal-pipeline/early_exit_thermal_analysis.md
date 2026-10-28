# Early Exit vs. Thermal Throttling: MAESTRO + LTI + ThermLeT

This document describes the analysis that checks whether early exit (EE) in LLaVA-7B helps avoid thermal throttling on edge platforms (NVIDIA Jetson Orin Nano and AGX Xavier). It covers how the analysis evolved, the current model, how to run it, the preliminary results and the next steps.

Main script: `lti-maestro-ee.py` · Configuration: `hw_config_ee.json` · Output: `results_ee/`

> **Status:** every hardware parameter is currently an **indicative estimate** (datasheet or literature order of magnitude). The numbers below show how the pipeline behaves. They are not measured results and should not be quoted.

> **Change of direction (2026-10-28):** the thesis targets **2.5D chiplet accelerators**, not Jetson boards. Sections 1–5 document the Jetson-based phase, which is now a methodology baseline only. Section 6 is the current work plan and section 7 holds everything needed to resume the work without the original conversation.

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

## 6. Work plan: early exit on a 2.5D chiplet accelerator

### 6.1 Why the direction changed

The Jetson model (sections 3–5) treats the chip as a single thermal node. Topology enters only through `footprint_px`: a centred square of uniform power on the 224×224 grid, from which ThermLeT gives a hotspot correction of about 2.5 °C. Jetson Orin Nano and AGX Xavier are monolithic SoCs, so they sit outside ThermLeT's training distribution (2.5D chiplet packages). The thesis instead targets chiplet accelerators, where topology is the core of the argument.

### 6.2 Thesis claim (chiplet version)

On a 2.5D package, each part of LLaVA-7B-EE runs on a specific set of chiplets. Early exit changes **which chiplets stay active and for how long**. It therefore changes not only the total heat exposure (as in the Jetson model), but also the **spatial temperature distribution**: hotspot position and peak, gradients across the interposer, and thermal coupling between neighbouring chiplets. The claim to test:

> Early exit, combined with a layer-to-chiplet mapping, keeps the hottest chiplet below the throttling limit in cases where the full inference throttles. Mapping and exit point together set the thermal outcome.

### 6.3 Pipeline to build

New script, suggested name `chiplet-ee.py`. Keep `lti-maestro-ee.py` unchanged as the single-node baseline.

1. **Layout definition** (new file, for example `chiplet_layout.json`)
   - Package size in mm, mapped onto the 224×224 ThermLeT grid (px/mm scale stated explicitly).
   - Compute chiplets: position and size. The current MAESTRO accelerator has 256 PEs, for example split into 4 × 64-PE chiplets.
   - HBM/DRAM stacks and their positions, with their own power from off-chip traffic.
   - Start from a layout similar to those in the ThermLeT paper (Micro150, Ascend910, MultiGPU), so inputs stay close to the training distribution.

2. **Layer-to-chiplet mapping**
   - Assign each MAESTRO layer (or block of layers) to one or more chiplets. Example: ViT encoder on chiplet A, LLM blocks pipelined across chiplets B–D, EE heads on the chiplet that holds the corresponding block.
   - The three MAESTRO split modes (`no-split`, `part-split`, `full-split`, see 7.4) give per-layer metrics under different partitioning constraints. Use them to define and compare mapping options.
   - Per-chiplet power for each phase: MAESTRO `PE power` + `L1 power` + `L2 power` + `NOC power`, divided by the chiplet area, give a power density. Idle and leakage power for inactive chiplets must be set explicitly.

3. **Steady-state maps from ThermLeT**
   - For each distinct phase (a set of active chiplets with given power levels), build the power map and run ThermLeT. Cache identical phases, because there are few distinct power levels.
   - **Power scaling is the main open problem** (see 6.4).

4. **Transient**
   - ThermLeT only predicts steady state. Apply a first-order LTI filter **per pixel** (or per chiplet) between consecutive steady-state maps, with a package time constant τ:
     `T(t+dt) = T_ss,phase + (T(t) − T_ss,phase) · exp(−dt/τ)`.
   - This assumes the thermal response is close to linear in power. Check it first: the sum of the maps predicted for two chiplets alone should match the map with both chiplets active.
   - Choose τ from the literature on 2.5D packages, then cover it with a sensitivity sweep.

5. **Limit and throttling**
   - Fixed junction limit `T_lim` (for example 95–105 °C, with a cited source), applied to the **hottest pixel of each chiplet**.
   - Throttling per chiplet: only the hot chiplet is slowed. In a pipelined mapping this stalls the following stages, which is the effect worth showing.

6. **Experiments**
   - Single inference per exit (EE-8/16/24/Full): temperature maps at the end, peak per chiplet over time, throttled layers.
   - Stream of inferences at different request periods (duty cycles).
   - Mapping comparison: the same exits under 2–3 mappings (for example, encoder and early LLM blocks on adjacent chiplets vs. far apart).
   - Sensitivity sweep over τ, idle power and power scale.
   - Thesis figures: temperature maps for each exit side by side, peak T per chiplet vs. time with `T_lim`, and a mapping × exit table of throttled layers.

### 6.4 Open problems to solve first

1. **Units of ThermLeT's power channel.** Training data is not available. Evidence so far, from probing the model with a 64×64 px square of constant value on the 224×224 grid:

   | Input value | Peak predicted (°C) |
   |---|---|
   | 0 | 46.8 |
   | 0.5 | 52.9 |
   | 1 | 63.6 |
   | 2 | 87.3 |
   | 5 | 160 |
   | 10 | 294 |

   The response is about linear above 0.5. The training pipeline adds Gaussian noise with std 0.01 to the power channel, which suggests values of order 1. The absolute units (W/mm², normalised, …) are unknown. To resolve:
   - Ask the authors for a few dataset samples or the normalisation used (Varun D. Parekh, `vdp5074@psu.edu`). The SharePoint dataset link in the README is not accessible.
   - Otherwise, calibrate against a figure in the paper: rebuild one of its layouts and match the reported temperatures.
   - Cross-check one or two layouts with an open-source thermal simulator (HotSpot or 3D-ICE). This also gives an independent validation of ThermLeT on our layouts.
2. **Material and cooling context are not available.** The released model only reads channel 0 (power) and channels 3–4 (coordinates). Channels 1–2 (in `test.py`, channel 2 is the "heat" channel, overridden with `--heat_value 950`) are ignored by the model. Package materials and cooling cannot be varied through ThermLeT; they would need a retrained model or be handled outside it.
3. **Coordinate channels.** Inference currently uses x/y maps from `linspace(0, 1, 224)`. The README says "normalized x/y coordinate maps" but does not give the exact range. Confirm with a dataset sample.
4. **`part-split` and `full-split` MAESTRO CSVs are identical** (same md5), even though their accelerator files use different L1/L2 constraints (1200/6000 vs. 120/600). Check whether MAESTRO was run correctly or whether the constraints do not bind. The CSV currently used (`data/llava_7B_ee_correct.csv`) is this same file.
5. **MAESTRO power is nearly flat** (3376 mW for every layer except `PATCH_EMBED`). Per-chiplet differences will come mostly from the mapping and from DRAM traffic, not from layer-specific compute power. Check the per-component power columns, which may vary more.
6. **The `Power` unit** in MAESTRO output is assumed to be mW. Confirm it in the MAESTRO documentation.

### 6.5 What happens to the Jetson work

- `lti-maestro-ee.py` and `hw_config_ee.json` stay as the single-node baseline: they show the time/energy effect of early exit without topology.
- Calibrating Jetson boards with `tegrastats` is no longer on the critical path. It is still useful to validate the LTI transient method (fit τ, compare simulated and measured T(t)), and it links to the existing `../REAL-scheduling/` measurements on Orin Nano.

---

## 7. Resuming the work: reference information

### 7.1 Directory map

All paths are relative to `chiplet_scheduling/`.

| Path | Content |
|---|---|
| `thermal-pipeline/early_exit_thermal_analysis.md` | This document |
| `thermal-pipeline/data/llava_7B_ee_correct.csv` | MAESTRO per-layer results for LLaVA-7B-EE (374 layers), identical to `MAESTRO/results/llava_ee/part-split-…csv` and `full-split-…csv` |
| `thermal-pipeline/lti-maestro.py` | Original baseline, hand-set 72 °C ceiling (unchanged) |
| `thermal-pipeline/lti-maestro-thermlet.py` | Intermediate version: dynamic ThermLeT ceiling + `hw_config.json` (superseded; throttles every layer) |
| `thermal-pipeline/lti-maestro-thermlet-bk.py` | Backup of the above |
| `thermal-pipeline/lti-maestro-ee.py` + `hw_config_ee.json` | Current single-node EE analysis (sections 3–5) |
| `thermal-pipeline/results_ee/` | Plots and `summary.csv` from `lti-maestro-ee.py` |
| `thermal-pipeline/thermal_pipeline_documentation.md` | Older documentation of the `hw_config.json` pipeline (its R and τ units differ from `hw_config_ee.json`) |
| `thermal-pipeline/ThermLeT/` | ThermLeT repository (`ThermLeT-main-bk/` is an untouched copy) |
| `thermal-pipeline/ThermLeT/Fusion_pipeline/experiment/exp_1/checkpoints/best.pth` | Pre-trained ThermLeT weights |
| `MAESTRO/results/llava_ee/` | MAESTRO models (`*.m`), accelerator configs and CSVs for the three split modes |
| `MAESTRO/accelerators/` | Other accelerator configurations |
| `REAL-scheduling/` | Measured k-factor calibration on RTX 5090 Mobile and Orin Nano |
| `motivation/` | Motivation plots (DRAM wall, memory paradox) |

### 7.2 Environment

- Conda environment: `conda activate thermlet` (Python 3.11).
- Pinned packages: `numpy==1.26.4` and `pandas==2.2.3`. NumPy 2 breaks the installed PyTorch ("Numpy is not available"), and `pip install pandas` without a pin pulls in NumPy 2.
- **Run on CPU.** The installed PyTorch supports only up to sm_86, and the workstation GPU (RTX PRO 4500 Blackwell) is sm_120. To use the GPU, install a CUDA 12.8+ build (`pip install torch --index-url https://download.pytorch.org/whl/cu128`) and re-check that NumPy stays at 1.26.4. CPU is enough for the current number of inferences.

### 7.3 ThermLeT facts

- Paper: Parekh, Devulapally et al., *ThermLeT: Transformer-based Temperature Prediction for 2.5D Chiplet Architecture*, VLSID 2026. Trained on 6,221 ANSYS layouts (Micro150, Ascend910, MultiGPU). Reported 3.16 °C RMSE, ~13 ms/sample.
- Repository: `https://github.com/MicroSystemsDesignLab/ThermLeT`.
- Weights: GitHub release `model_weights`, file `best.pth` (1,209,867,530 bytes), at `https://github.com/MicroSystemsDesignLab/ThermLeT/releases/download/model_weights/best.pth`.
- Checkpoint keys: `model_state`, `epoch`, `optimizer_state`. Load with `model.load_state_dict(ckpt["model_state"])`.
- Model configuration (defaults in `main.py`/`test.py`, which load the checkpoint without errors):
  `img_size=[224,224], patch_size=16, hidden_dim=768, num_layers=12, nheads=12, dim_feedforward=2048, mlp_ratio=4.0, qkv_bias=False, dropout=0.1, num_res_blocks=1, filters=64`.
  Build with `build_temperature_model(cfg)` from `Fusion_pipeline/model.py`.
- Input format: H×W×5 `.npy`. Channel 0 is power, channels 3–4 are x/y coordinates. Channels 1–2 are not read by the model.
- Forward call: `model(power[B,1,H,W], coords[B,2,H,W]) → temperature[B,1,H,W]` in °C.
- Reusable helpers in `lti-maestro-ee.py`: `load_thermlet`, `make_coords`, and the power-map construction.

### 7.4 MAESTRO setup

- Accelerator for the current CSV: 256 PEs, `offchip_bw_cstr: 160` elements/cycle (`part-split` and `full-split` configs). `no-split` uses near-ideal resources (L1 120000, L2 600000, NoC 100000, off-chip 5000).
- Useful CSV columns (names must be `.str.strip()`-ed): `Layer Number`, `Runtime (Cycles)`, `Power`, `PE power`, `L1 power`, `L2 power`, `NOC power`, `Offchip BW Req (Elements/cycle)`, `Activity count-based Energy (nJ)`, `Area`.
- Early-exit heads in the layer list: `EARLY_EXIT_HEAD_8` (index 202), `EARLY_EXIT_HEAD_16` (259), `EARLY_EXIT_HEAD_24` (316), `FINAL_LM_HEAD` (373). Layer 0 is `PATCH_EMBED`, layers 1–144 are the ViT encoder (`VIT_BLKxx_*`), layer 145 is `VISION_TO_LLM_PROJ`, and the rest are LLM blocks (`LLM_BLKxx_*`) with the exit heads in between.

### 7.5 Lessons learned (avoid repeating)

- Feeding raw MAESTRO mW values to ThermLeT gives absurd temperatures (91,126 °C). Power must be scaled to the model's input range.
- A thermal time constant much shorter than a layer's duration erases the effect of early exit, because every layer reaches steady state.
- Using ThermLeT on the same power as the RC model as a "dynamic ceiling" is circular. Use a fixed limit and let ThermLeT provide temperature, not the threshold.
- Parameters chosen only so that the claim holds will not survive review. Prefer cited values plus a sensitivity sweep, and mark every estimate as such.
