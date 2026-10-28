"""
Early exit vs throttling termico: MAESTRO (potenza/cicli per layer) + modello LTI a un polo
+ correzione di hotspot spaziale da ThermLeT.

Modello termico per layer:
    T_ss  = T_amb + R * P_totale + dT_hotspot
    T_end = T_ss + (T_start - T_ss) * exp(-dt / tau)
Il throttling (DVFS) scatta quando T_end supera il limite fisso T_lim.

Esperimenti:
  1. singola inferenza, uscita a EE-8 / EE-16 / EE-24 / Full
  2. stream di N inferenze a periodo fisso (duty cycle)
  3. sensitivity R x tau: uscita più tardiva che non va in throttling
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FUSION_DIR = os.path.join(HERE, "ThermLeT", "Fusion_pipeline")
sys.path.insert(0, FUSION_DIR)
from model import build_temperature_model  # noqa: E402

EXITS = {"EE-8": "EARLY_EXIT_HEAD_8", "EE-16": "EARLY_EXIT_HEAD_16",
         "EE-24": "EARLY_EXIT_HEAD_24", "Full": "FINAL_LM_HEAD"}
BYTES_PER_ELEM = 2       # FP16
DVFS_POWER = 0.5         # throttling: potenza di calcolo x0.5 ...
DVFS_TIME = 1.5          # ... tempo di calcolo x1.5


def parse_args():
    p = argparse.ArgumentParser("Early exit vs throttling termico")
    p.add_argument("--csv", default=os.path.join(HERE, "data", "llava_7B_ee_correct.csv"))
    p.add_argument("--ckpt", default=os.path.join(FUSION_DIR, "experiment", "exp_1", "checkpoints", "best.pth"))
    p.add_argument("--hw-config", default=os.path.join(HERE, "hw_config_ee.json"))
    p.add_argument("--platforms", nargs="+", default=None, help="Default: tutte quelle nel JSON")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-thermlet", action="store_true", help="Disattiva la correzione di hotspot (solo LTI)")
    p.add_argument("--stream-n", type=int, default=6, help="Numero di inferenze nello stream")
    p.add_argument("--stream-period", type=float, default=None,
                   help="Periodo tra richieste in s (default: 1.1 x latenza Full senza throttling)")
    p.add_argument("--out-dir", default=os.path.join(HERE, "results_ee"))
    return p.parse_args()


# ---------------------------------------------------------------- ThermLeT
def load_thermlet(ckpt_path, device):
    cfg = SimpleNamespace(img_size=[224, 224], patch_size=16, hidden_dim=768, num_layers=12,
                          nheads=12, dim_feedforward=2048, mlp_ratio=4.0, qkv_bias=False,
                          dropout=0.1, num_res_blocks=1, filters=64)
    model = build_temperature_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


class Hotspot:
    """dT di hotspot = picco - media sul footprint della mappa predetta da ThermLeT.
    ThermLeT è addestrato su package chiplet, non su Jetson: lo uso solo per il gradiente
    spaziale (relativo), non per la temperatura assoluta."""

    def __init__(self, model, device, footprint, p_ref, h=224, w=224):
        self.model, self.device, self.p_ref = model, device, p_ref
        ys, xs = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
        self.coords = torch.from_numpy(np.stack([xs, ys]).astype(np.float32))[None].to(device)
        t0, l0 = (h - footprint) // 2, (w - footprint) // 2
        self.region = (slice(t0, t0 + footprint), slice(l0, l0 + footprint))
        self.shape = (h, w)
        self.cache = {}

    @torch.no_grad()
    def delta(self, p_compute_w):
        # potenza normalizzata sul layer più potente: ThermLeT si aspetta valori di ordine ~1
        key = round(p_compute_w / self.p_ref, 3)
        if key not in self.cache:
            pm = np.zeros(self.shape, dtype=np.float32)
            pm[self.region] = key
            pred = self.model(torch.from_numpy(pm)[None, None].to(self.device), self.coords)[0, 0].cpu().numpy()
            self.cache[key] = float(pred.max() - pred[self.region].mean())
        return self.cache[key]


# ---------------------------------------------------------------- Modello per layer
def build_layers(df, hw):
    """Tempo ed energia per layer. Se la DRAM non regge la banda richiesta, il layer è memory-bound."""
    cycles = df["Runtime (Cycles)"].values.astype(float)
    p_comp = df["Power"].values.astype(float) / 1000.0           # MAESTRO: mW -> W
    bw = df["Offchip BW Req (Elements/cycle)"].values.astype(float)
    t_comp = cycles / hw["clock_freq_hz"]
    dram_bytes = bw * cycles * BYTES_PER_ELEM
    return pd.DataFrame({
        "name": df["Layer Number"].astype(str).str.strip().values,
        "t_comp": t_comp,
        "t_mem": dram_bytes / (hw["dram_bw_gbs"] * 1e9),
        "e_comp": p_comp * t_comp,
        "e_dram": dram_bytes * 8 * hw["dram_pj_per_bit"] * 1e-12,
    })


def layer_step(T, row, hw, R, tau, hotspot, throttled):
    t_comp = row.t_comp * (DVFS_TIME if throttled else 1.0)
    e_comp = row.e_comp * (DVFS_POWER * DVFS_TIME if throttled else 1.0)
    dt = max(t_comp, row.t_mem)
    p_comp = e_comp / dt
    P = p_comp + row.e_dram / dt + hw["idle_power_w"]
    dT_hs = hotspot.delta(p_comp) if hotspot else 0.0
    T_ss = hw["T_amb_c"] + R * P + dT_hs
    return dt, P * dt, T_ss + (T - T_ss) * np.exp(-dt / tau)


def run_inference(layers, T0, t0, hw, R, tau, hotspot):
    """Esegue una sequenza di layer. Ritorna traccia (t, T) e statistiche."""
    T, t, energy, n_thr = T0, t0, 0.0, 0
    ts, Ts = [t0], [T0]
    for row in layers.itertuples():
        dt, e, T_new = layer_step(T, row, hw, R, tau, hotspot, throttled=False)
        if T_new > hw["T_lim_c"]:
            dt, e, T_new = layer_step(T, row, hw, R, tau, hotspot, throttled=True)
            n_thr += 1
        T, t, energy = T_new, t + dt, energy + e
        ts.append(t)
        Ts.append(T)
    return dict(t=ts, T=Ts, T_end=T, t_end=t, latency=t - t0, energy=energy,
                throttled=n_thr, T_peak=max(Ts))


def idle(T, duration, hw, R, tau):
    T_ss = hw["T_amb_c"] + R * hw["idle_power_w"]
    return T_ss + (T - T_ss) * np.exp(-duration / tau)


def run_stream(layers, n, period, hw, R, tau, hotspot):
    T, t = hw["T_amb_c"], 0.0
    ts, Ts, thr, misses = [], [], 0, 0
    for _ in range(n):
        r = run_inference(layers, T, t, hw, R, tau, hotspot)
        ts += r["t"]
        Ts += r["T"]
        thr += r["throttled"]
        misses += r["latency"] > period
        gap = max(period - r["latency"], 0.0)
        T, t = idle(r["T_end"], gap, hw, R, tau), r["t_end"] + gap
        ts.append(t)
        Ts.append(T)
    return dict(t=ts, T=Ts, throttled=thr, misses=misses, T_peak=max(Ts))


def t_star(P, hw, R, tau, dT_hs=0.0):
    """Tempo (da T_amb) a cui si supera T_lim a potenza costante P; inf se mai."""
    T_ss = hw["T_amb_c"] + R * P + dT_hs
    if T_ss <= hw["T_lim_c"]:
        return np.inf
    return -tau * np.log((T_ss - hw["T_lim_c"]) / (T_ss - hw["T_amb_c"]))


# ---------------------------------------------------------------- Main
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.hw_config) as f:
        hw_all = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    platforms = args.platforms or list(hw_all)

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()
    names = df["Layer Number"].astype(str).str.strip().tolist()
    exit_idx = {k: names.index(v) for k, v in EXITS.items()}
    p_comp_max = df["Power"].max() / 1000.0

    model = None if args.no_thermlet else load_thermlet(args.ckpt, torch.device(args.device))
    summary = []

    for plat in platforms:
        hw = hw_all[plat]
        R, tau = hw["R_c_per_w"], hw["tau_s"]
        hotspot = None if model is None else Hotspot(model, torch.device(args.device), hw["footprint_px"], p_comp_max)
        layers = build_layers(df, hw)
        paths = {k: layers.iloc[:i + 1] for k, i in exit_idx.items()}
        print(f"\n=== {plat}: {hw['description']} ===")

        # Criterio analitico a potenza media costante (layer di calcolo tipico)
        full = paths["Full"]
        dt_full = np.maximum(full.t_comp, full.t_mem)
        P_mean = (full.e_comp.sum() + full.e_dram.sum()) / dt_full.sum() + hw["idle_power_w"]
        dT_typ = hotspot.delta(float(np.median(full.e_comp / dt_full))) if hotspot else 0.0
        ts_ = t_star(P_mean, hw, R, tau, dT_typ)
        print(f"P media {P_mean:.2f} W, hotspot ThermLeT {dT_typ:.2f} °C, "
              f"T_ss {hw['T_amb_c'] + R * P_mean + dT_typ:.1f} °C, t* = {ts_:.1f} s")

        # 1. Singola inferenza
        fig, ax = plt.subplots(figsize=(10, 5))
        # ogni uscita è un prefisso della Full: disegno dalla più lunga, con il punto d'uscita marcato
        for k, lay in reversed(paths.items()):
            r = run_inference(lay, hw["T_amb_c"], 0.0, hw, R, tau, hotspot)
            line, = ax.plot(r["t"], r["T"], label=f"{k} ({r['throttled']} layer throttlati)")
            ax.plot(r["t"][-1], r["T"][-1], "o", color=line.get_color())
            print(f"  {k:5s}: latenza {r['latency']:6.1f} s, energia {r['energy']:7.1f} J, "
                  f"T_peak {r['T_peak']:5.1f} °C, throttlati {r['throttled']}/{len(lay)}")
            summary.append(dict(platform=plat, experiment="single", exit=k, latency_s=r["latency"],
                                energy_j=r["energy"], T_peak_c=r["T_peak"], throttled_layers=r["throttled"],
                                n_layers=len(lay), t_star_s=ts_))
        ax.axhline(hw["T_lim_c"], color="k", ls="--", label="T_lim")
        if np.isfinite(ts_):
            ax.axvline(ts_, color="gray", ls=":", label="t* (analitico)")
        ax.set(title=f"Singola inferenza - {plat}", xlabel="Tempo (s)", ylabel="Temperatura (°C)")
        ax.grid(ls=":", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"single_{plat}.png"), dpi=200)
        plt.close(fig)

        # 2. Stream a periodo fisso
        lat_full = dt_full.sum()
        period = args.stream_period or 1.1 * lat_full
        fig, ax = plt.subplots(figsize=(10, 5))
        for k, lay in paths.items():
            s = run_stream(lay, args.stream_n, period, hw, R, tau, hotspot)
            ax.plot(s["t"], s["T"], label=f"{k} ({s['throttled']} throttlati, {s['misses']} deadline miss)")
            print(f"  stream {k:5s}: T_peak {s['T_peak']:5.1f} °C, throttlati {s['throttled']}, "
                  f"deadline miss {s['misses']}/{args.stream_n}")
            summary.append(dict(platform=plat, experiment="stream", exit=k, T_peak_c=s["T_peak"],
                                throttled_layers=s["throttled"], deadline_miss=s["misses"],
                                n_layers=len(lay) * args.stream_n, period_s=period))
        ax.axhline(hw["T_lim_c"], color="k", ls="--", label="T_lim")
        ax.set(title=f"Stream di {args.stream_n} inferenze, periodo {period:.0f} s - {plat}",
               xlabel="Tempo (s)", ylabel="Temperatura (°C)")
        ax.grid(ls=":", alpha=0.6)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"stream_{plat}.png"), dpi=200)
        plt.close(fig)

        # 3. Sensitivity R x tau: uscita più tardiva senza throttling (singola inferenza)
        R_grid = R * np.linspace(0.5, 2.0, 16)
        tau_grid = tau * np.logspace(-1, 1, 16)
        safe = np.zeros((len(R_grid), len(tau_grid)))
        for i, Rg in enumerate(R_grid):
            for j, tg in enumerate(tau_grid):
                for level, (k, lay) in enumerate(paths.items(), start=1):
                    if run_inference(lay, hw["T_amb_c"], 0.0, hw, Rg, tg, hotspot)["throttled"]:
                        break
                    safe[i, j] = level
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(tau_grid, R_grid, safe, cmap="viridis", vmin=0, vmax=len(paths), shading="nearest")
        ax.set_xscale("log")
        ax.plot(tau, R, "r*", ms=14, label="valori nominali")
        cb = fig.colorbar(im, ax=ax, ticks=range(len(paths) + 1))
        cb.ax.set_yticklabels(["nessuna"] + list(paths))
        cb.set_label("Uscita più tardiva senza throttling")
        ax.set(title=f"Sensitivity R x tau - {plat}", xlabel="tau (s)", ylabel="R (°C/W)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"sensitivity_{plat}.png"), dpi=200)
        plt.close(fig)

    out_csv = os.path.join(args.out_dir, "summary.csv")
    pd.DataFrame(summary).to_csv(out_csv, index=False)
    print(f"\n[*] Risultati in {args.out_dir}")


if __name__ == "__main__":
    main()
