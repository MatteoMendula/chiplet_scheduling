# Thermal Scheduling Pipeline: MAESTRO + ThermLeT + LTI

This document outlines the architecture of the hybrid thermal simulation pipeline and explains the hardware parameters used to model different edge AI accelerators (e.g., NVIDIA Jetson series).

## 1. Hardware Configuration (`hw_config.json`)

The `hw_config.json` file decouples the physical hardware characteristics from the execution logic. By modifying this file, you can simulate how the same neural network (e.g., LLaVA_7B) behaves thermally on different physical chips without changing the Python code.

### Parameters Explained

*   **`footprint` (Integer):** 
    *   **Definition:** The spatial representation (in pixels) of the active processing elements (PEs) on a fixed $224 \times 224$ grid used by the ThermLeT surrogate model.
    *   **Logic:** It maps the physical die size of the chip. A smaller footprint means the same power is concentrated in a tighter area (higher power density), leading ThermLeT to calculate a *lower* safe temperature ceiling to prevent localized hotspots.
*   **`R` (Float):** 
    *   **Definition:** The total Thermal Resistance ($\theta_{JA}$) of the system, expressed in °C/mW. 
    *   **Logic:** This dictates how effectively the chip dissipates heat into the environment. It is the sum of the Junction-to-Case ($\theta_{JC}$) resistance of the silicon and the Case-to-Ambient ($\theta_{CA}$) resistance of the heatsink/fan. A higher `R` value (e.g., passive cooling) causes the steady-state temperature to spike faster.
*   **`tau` (Float):** 
    *   **Definition:** The thermal time constant ($\tau = R \times C$) converted into execution cycles.
    *   **Logic:** It dictates the "thermal inertia" of the system. It determines how quickly the transient temperature rises toward the steady-state maximum.
*   **`description` (String):** 
    *   **Definition:** A human-readable note to identify the hardware model, die size, and cooling setup.

### Example JSON Block
```json
{
  "orin_nano": {
    "footprint": 70,
    "R": 0.00155,
    "tau": 50000000.0,
    "description": "NVIDIA Jetson Orin Nano (Active Cooling)"
  }
}
```

---

## 2. Pipeline Logic (`lti-maestro-thermlet.py`)

The script simulates the layer-by-layer execution of a neural network, dynamically checking for thermal violations based on the specific topology of the target hardware.

### Execution Flow

1.  **Data Ingestion (MAESTRO):**
    The script reads the CSV output from MAESTRO, extracting the power consumption (`Power`) and execution time (`Runtime Cycles`) for each specific layer of the neural network.

2.  **Dynamic Thermal Ceiling (ThermLeT):**
    *Unlike a static threshold*, the script queries the AI surrogate model (ThermLeT) inside the execution loop. 
    For each layer, it projects the power load onto the spatial `footprint` defined in the JSON. ThermLeT infers the 2D thermal map and returns the maximum safe steady-state temperature (the **Ceiling**) for *that specific layer's power density*.

3.  **Transient Tracking (LTI System):**
    Instead of running a slow, full-physics simulator (like HotSpot) at runtime, the pipeline uses a Linear Time-Invariant (LTI) algebraic equation to track the heat buildup:
    `T_new = T_steady + (T_last - T_steady) * exp(-dt / tau)`
    This calculates the exact temperature of the silicon at the end of the layer's execution time (`dt`), building upon the residual heat (`T_last`) left by the previous layers.

4.  **Circuit Breaker (Scheduling Policy):**
    Before finalizing the layer's execution, the system compares the predicted transient temperature (`T_new`) against the physical limit (`layer_ceiling`).
    *   **If Safe:** The layer executes normally.
    *   **If Violated:** The scheduler triggers a mitigation policy. In this PoC, it simulates a **DVFS (Dynamic Voltage and Frequency Scaling)** intervention, reducing power by 50% but increasing execution time by 50%. The temperature is then recalculated to reflect the throttled state.

5.  **Output Visualization:**
    Finally, Matplotlib generates a layer-by-layer graph (`thermal_profile_layers_[hw_model].png`) plotting the transient temperature curve against the fluctuating dynamic thermal ceiling.