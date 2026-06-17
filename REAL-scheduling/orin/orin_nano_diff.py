import math

def calculate_universal_k():
    print("==================================================")
    print("  Calcolo Statistico Fattori di Scala (k) Universale")
    print("==================================================")

    # ==================================================
    # 1. PARAMETRI HARDWARE TEORICI (Modifica per Jetson)
    # ==================================================
    # Dati RTX 5090 Mobile
    THEORETICAL_BANDWIDTH_GBS = 68.0  # GB/s
    THEORETICAL_PEAK_TFLOPS = 1.28  # TFLOPS
    N = 16384  # Dimensione matrice

    # Calcolo costanti analitiche ideali
    matrix_size_mb = (N * N * 4) / (1024 * 1024)
    theoretical_tflops = (2 * (N**3)) / 1e12

    ideal_compute_ms = (theoretical_tflops / THEORETICAL_PEAK_TFLOPS) * 1000
    ideal_comm_ms = (matrix_size_mb / (THEORETICAL_BANDWIDTH_GBS * 1024)) * 1000
    ideal_total_ms = ideal_compute_ms + ideal_comm_ms

    # ==================================================
    # 2. DATI SPERIMENTALI (Media, StdDev)
    # ==================================================
    # Dati dal Run Python (CUDA Events)
    pytorch_comp_mean, pytorch_comp_std = 14221.8602, 18.6811
    pytorch_comm_mean, pytorch_comm_std = 711.6393, 64.0162

    # Dati dal Run NSYS (Driver Overhead)
    nsys_malloc_mean, nsys_malloc_std = 2121.8749, 223.9523

    # ==================================================
    # 3. FUSIONE DEI DATI E PROPAGAZIONE DELL'ERRORE
    # ==================================================
    # Calcolo puro
    comp_mean = pytorch_comp_mean
    comp_std = pytorch_comp_std

    # Comunicazione (Host Malloc + Device Transfer)
    comm_mean = nsys_malloc_mean + pytorch_comm_mean
    comm_std = math.sqrt(nsys_malloc_std**2 + pytorch_comm_std**2)

    # Totale di Sistema
    total_mean_real = comp_mean + comm_mean
    total_std_real = math.sqrt(comp_std**2 + comm_std**2)

    # ==================================================
    # 4. CALCOLO FATTORI K CALIBRATI
    # ==================================================
    k_compute_mean = comp_mean / ideal_compute_ms
    k_compute_std = comp_std / ideal_compute_ms

    k_comm_mean = comm_mean / ideal_comm_ms
    k_comm_std = comm_std / ideal_comm_ms

    k_total_mean = total_mean_real / ideal_total_ms
    k_total_std = total_std_real / ideal_total_ms

    # ==================================================
    # 5. STAMPA RISULTATI
    # ==================================================
    print("[Latenze Ideali del Modello Analitico]:")
    print(f"  - Compute Ideale: {ideal_compute_ms:.4f} ms")
    print(f"  - Comm Ideale:    {ideal_comm_ms:.4f} ms")
    print(f"  - Totale Ideale:  {ideal_total_ms:.4f} ms")

    print("\n[Latenze Reali Complessive (Host + Device)]:")
    print(f"  - Compute Reale:  {comp_mean:.4f} ms (± {comp_std:.4f} ms)")
    print(f"  - Comm Reale:     {comm_mean:.4f} ms (± {comm_std:.4f} ms)  <-- (Malloc + Transfer)")
    print(f"  - Totale Reale:   {total_mean_real:.4f} ms (± {total_std_real:.4f} ms)")

    print("\n==================================================")
    print("          FATTORI K CALIBRATI CON INCERTEZZA      ")
    print("==================================================")
    print(f"  👉 k_compute: {k_compute_mean:.4f}x (± {k_compute_std:.4f}x)")
    print(f"  👉 k_comm:    {k_comm_mean:.4f}x (± {k_comm_std:.4f}x)")
    print(f"  👉 k_total:   {k_total_mean:.4f}x (± {k_total_std:.4f}x)")
    print("==================================================")

if __name__ == "__main__":
    calculate_universal_k()

