import argparse
import os
import sys
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Setup directory ThermLeT (come nel tuo script originale)
HERE = os.path.dirname(os.path.abspath(__file__))
FUSION_DIR = os.path.join(HERE, "ThermLeT", "Fusion_pipeline")
sys.path.insert(0, FUSION_DIR)
from model import build_temperature_model  # noqa: E402

def parse_args():
    p = argparse.ArgumentParser("MAESTRO + ThermLeT thermal scheduler")
    p.add_argument("--csv", default=os.path.join(HERE, "data", "llava_7B_ee_correct.csv"))
    p.add_argument("--ckpt", default=os.path.join(FUSION_DIR, "experiment", "exp_1", "checkpoints", "best.pth"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--hw-config", default="hw_config.json", help="Path al file JSON delle configurazioni HW")
    p.add_argument("--hw-model", required=True, help="Nome del modello HW nel JSON (es. orin_nano, xavier_agx)")
    p.add_argument("--power-scale", type=float, default=None)
    p.add_argument("--ref-power", type=float, default=None)
    return p.parse_args()

def load_thermlet(ckpt_path, device):
    cfg = SimpleNamespace(img_size=[224, 224], patch_size=16, hidden_dim=768, num_layers=12,
                          nheads=12, dim_feedforward=2048, mlp_ratio=4.0, qkv_bias=False,
                          dropout=0.1, num_res_blocks=1, filters=64)
    model = build_temperature_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()

def make_coords(h=224, w=224):
    ys, xs = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    return np.stack([xs, ys]).astype(np.float32)

@torch.no_grad()
def thermlet_peak_temperature(model, coords, power_value, footprint, device, h=224, w=224):
    pm = np.zeros((h, w), dtype=np.float32)
    t0, l0 = (h - footprint) // 2, (w - footprint) // 2
    pm[t0:t0 + footprint, l0:l0 + footprint] = power_value
    p_t = torch.from_numpy(pm)[None, None].to(device)
    c_t = torch.from_numpy(coords)[None].to(device)
    return float(model(p_t, c_t).max().item())

args = parse_args()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

# --- 1. Caricamento HW Config ---
with open(args.hw_config, 'r') as f:
    hw_configs = json.load(f)

if args.hw_model not in hw_configs:
    raise ValueError(f"Modello '{args.hw_model}' non trovato in {args.hw_config}. Modelli disponibili: {list(hw_configs.keys())}")

hw_params = hw_configs[args.hw_model]
footprint = hw_params["footprint"]
R = hw_params["R"]
tau = hw_params["tau"]
T_amb = 40.0
print(f"[*] Configurazione HW caricata: {hw_params['description']} (Footprint: {footprint}px, R: {R})")

# --- 2. Dati MAESTRO ---
df = pd.read_csv(args.csv)
df.columns = df.columns.str.strip()
power = df['Power'].values
cycles = df['Runtime (Cycles)'].values
layers = df['Layer Number'].values

# --- 3. ThermLeT Ceiling ---
model = load_thermlet(args.ckpt, device)
coords = make_coords()
ref_power = args.ref_power if args.ref_power is not None else float(np.average(power, weights=cycles))

# Parametri dal JSON (servono già per il massimo assoluto qui sotto)
pj_per_bit = hw_params["dram_pj_per_bit"]
clock_hz = hw_params["clock_freq_hz"]
p_idle = hw_params["idle_power_mw"]

# Estrai colonna Offchip da MAESTRO (assicurati che il nome colonna corrisponda al tuo CSV)
offchip_bw = df['Offchip BW Req (Elements/cycle)'].values

# Calcola array potenze reali per trovare il massimo assoluto
p_dram_array = (offchip_bw * 16 * pj_per_bit * 1e-9) * clock_hz
p_reale_array = power + p_dram_array + p_idle

# Mantieni il supporto all'argomento da CLI, altrimenti usa il nuovo massimo
power_scale = args.power_scale if args.power_scale is not None else 1.0 / p_reale_array.max()

THERMLET_CEILING = thermlet_peak_temperature(model, coords, ref_power * power_scale, footprint, device)
print(f"[*] ThermLeT ceiling: {THERMLET_CEILING:.2f}°C")

# --- 4. Variabili di Stato LTI e Tracciamento ---
T_last = T_amb
log_scheduler = []
plot_temps = []
dynamic_ceilings = [] # Nuovo: tetto termico dinamico per layer
layer_indices = range(len(df))

# --- 5. Ciclo di Simulazione (Layer by Layer) ---
for i in layer_indices:
    print(f"\r{i+1}/{len(layer_indices)}", end="", flush=True)
    p_compute_maestro = power[i]
    dt = cycles[i]
    bw_req = offchip_bw[i]
    layer_name = layers[i]

    # Calcolo scientifico della potenza DRAM in mW
    # Assumiamo elementi in FP16 (16 bit per elemento)
    # pJ_per_bit * 10^-12 (Joule) * 10^3 (mW) = 10^-9
    p_dram = (bw_req * 16 * pj_per_bit * 1e-9) * clock_hz

    # Potenza totale reale del layer
    p_reale = p_compute_maestro + p_dram + p_idle

    # A. Tetto Termico Dinamico (Inferenza ThermLeT sul layer corrente)
    layer_power_norm = p_reale * power_scale
    layer_ceiling = thermlet_peak_temperature(model, coords, layer_power_norm, footprint, device)
    dynamic_ceilings.append(layer_ceiling)

    # B. Calcolo LTI Transitorio
    T_steady = T_amb + (R * p_reale)
    exp_factor = np.exp(-dt / tau)
    T_new = T_steady + (T_last - T_steady) * exp_factor

    # C. Circuit Breaker Dinamico
    if T_new > layer_ceiling:
        # DVFS riduce clock/potenza calcolo, ma la memoria trasferisce ancora i dati
        p_compute_ridotta = p_compute_maestro * 0.5
        p_dram_ridotta = p_dram * 0.7  # anche la banda scende, ma non proporzionalmente
        p_reale_mitigata = p_compute_ridotta + p_dram_ridotta + p_idle
        
        dt_aumentato = dt * 1.5
        
        T_steady_mitigato = T_amb + (R * p_reale_mitigata)
        exp_factor_mitigato = np.exp(-dt_aumentato / tau)
        T_new = T_steady_mitigato + (T_last - T_steady_mitigato) * exp_factor_mitigato
        
        log_scheduler.append(f"[WARNING] Layer {i} ({layer_name}): Throttling attivato")
        
    plot_temps.append(T_new)
    T_last = T_new

print(f"\nLayer throttlati: {sum('[WARNING]' in l for l in log_scheduler)}/{len(df)}")

# --- 6. Plotting Matplotlib (Layer-by-Layer) ---
plt.figure(figsize=(14, 6))

# Asse X = Indice del Layer
plt.plot(layer_indices, plot_temps, color='#d62728', linewidth=2, label='Temperatura Transitoria Layer (LTI)')
plt.plot(layer_indices, dynamic_ceilings, color='#1f77b4', linestyle='--', linewidth=2, label='ThermLeT Ceiling Dinamico')

plt.title(f"Profilo Termico Layer-by-Layer - LLaVA_7B su {args.hw_model}", fontsize=14, fontweight='bold')
plt.xlabel("Indice Layer Rete Neurale", fontsize=11)
plt.ylabel("Temperatura (°C)", fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc="upper right")
plt.tight_layout()

out_filename = f"thermal_profile_layers_{args.hw_model}.png"
plt.savefig(out_filename, dpi=300)
print(f"[*] Plot salvato come {out_filename}")