import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

# Il modello ThermLeT vive in ThermLeT/Fusion_pipeline
HERE = os.path.dirname(os.path.abspath(__file__))
FUSION_DIR = os.path.join(HERE, "ThermLeT", "Fusion_pipeline")
sys.path.insert(0, FUSION_DIR)
from model import build_temperature_model  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser("MAESTRO + ThermLeT thermal scheduler")
    p.add_argument("--csv", default=os.path.join(HERE, "data", "llava_7B_ee_correct.csv"))
    p.add_argument("--ckpt", default=os.path.join(FUSION_DIR, "experiment", "exp_1", "checkpoints", "best.pth"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--power-scale", type=float, default=None,
                   help="Fattore applicato a Power prima di costruire la power map "
                        "(default: 1/max(Power), cioe' il layer piu' potente vale 1.0: "
                        "ThermLeT e' stato addestrato su power map normalizzate di ordine ~1)")
    p.add_argument("--footprint", type=int, default=64,
                   help="Lato (in pixel della griglia 224x224) del chiplet target, centrato")
    p.add_argument("--ref-power", type=float, default=None,
                   help="Potenza di riferimento per il ceiling (default: media di Power pesata sui cicli)")
    return p.parse_args()


def load_thermlet(ckpt_path, device):
    # Stessa configurazione di default di test.py (deve combaciare con il training)
    cfg = SimpleNamespace(img_size=[224, 224], patch_size=16, hidden_dim=768, num_layers=12,
                          nheads=12, dim_feedforward=2048, mlp_ratio=4.0, qkv_bias=False,
                          dropout=0.1, num_res_blocks=1, filters=64)
    model = build_temperature_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def make_coords(h=224, w=224):
    ys, xs = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    return np.stack([xs, ys]).astype(np.float32)  # (2, H, W)


@torch.no_grad()
def thermlet_peak_temperature(model, coords, power_value, footprint, device, h=224, w=224):
    """Inferenza ThermLeT: power map con il chiplet target acceso -> temperatura di picco (°C)."""
    pm = np.zeros((h, w), dtype=np.float32)
    t0, l0 = (h - footprint) // 2, (w - footprint) // 2
    pm[t0:t0 + footprint, l0:l0 + footprint] = power_value
    p_t = torch.from_numpy(pm)[None, None].to(device)
    c_t = torch.from_numpy(coords)[None].to(device)
    return float(model(p_t, c_t).max().item())


args = parse_args()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

# 1. Caricamento dati MAESTRO
df = pd.read_csv(args.csv)
df.columns = df.columns.str.strip()  # Pulizia nomi colonne
power = df['Power'].values
cycles = df['Runtime (Cycles)'].values
layers = df['Layer Number'].values

# 2. Parametri del modello topologico
T_amb = 40.0
R = 0.01      # Resistenza termica del chiplet target
tau = 5e7     # Costante di tempo termica in cicli

# 2b. Ceiling dal surrogato ThermLeT (al posto della soglia fissata a mano)
model = load_thermlet(args.ckpt, device)
coords = make_coords()
ref_power = args.ref_power if args.ref_power is not None else float(np.average(power, weights=cycles))
power_scale = args.power_scale if args.power_scale is not None else 1.0 / power.max()
THERMLET_CEILING = thermlet_peak_temperature(model, coords, ref_power * power_scale,
                                             args.footprint, device)
print(f"ThermLeT ceiling: {THERMLET_CEILING:.2f}°C (ref power {ref_power:.1f} -> {ref_power * power_scale:.3f} normalizzata, footprint {args.footprint}px)")

# 3. Inizializzazione variabili di stato
T_last = T_amb
storico_temperature = []
log_scheduler = []

# 4. Ciclo di simulazione e Scheduling Termico (Runtime)
for i in range(len(df)):
    p = power[i]
    dt = cycles[i]
    layer_name = layers[i]

    # Previsione della temperatura per il layer corrente
    T_steady = T_amb + (R * p)
    exp_factor = np.exp(-dt / tau)
    T_new = T_steady + (T_last - T_steady) * exp_factor

    # Logica di controllo (Circuit Breaker)
    if T_new > THERMLET_CEILING:
        # Intervento dello scheduler: simuliamo un DVFS (riduzione potenza del 50%, aumento tempo)
        p_ridotta = p * 0.5
        dt_aumentato = dt * 1.5

        # Ricalcolo con i nuovi parametri mitigati
        T_steady_mitigato = T_amb + (R * p_ridotta)
        exp_factor_mitigato = np.exp(-dt_aumentato / tau)
        T_new = T_steady_mitigato + (T_last - T_steady_mitigato) * exp_factor_mitigato

        log_scheduler.append(f"[WARNING] Layer {i} '{layer_name}': Limite superato. Throttling applicato. Nuova T: {T_new:.2f}°C")
    else:
        log_scheduler.append(f"[OK] Layer {i} '{layer_name}': Esecuzione standard. T: {T_new:.2f}°C")

    storico_temperature.append(T_new)
    T_last = T_new

# Stampa i risultati per i primi 15 layer per verifica
for log in log_scheduler[:15]:
    print(log)
print(f"Layer throttlati: {sum('[WARNING]' in l for l in log_scheduler)}/{len(log_scheduler)}")
