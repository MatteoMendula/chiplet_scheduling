import time
import torch

def run_rtx5090_calibration_fixed():
    print("==================================================")
    print("   RTX 5090 Mobile - Simulation Calibration Tool  ")
    print("==================================================")
    
    # 1. Hardware Profiling Settings (Specifiche teoriche RTX 5090 Mobile)
    THEORETICAL_BANDWIDTH_GBS = 896.0  # Banda GDDR7 in GB/s
    THEORETICAL_PEAK_TFLOPS = 33.52    # FP32 Peak TFLOPS
    
    if not torch.cuda.is_available():
        print("[ERRORE] CUDA non è disponibile. Installa PyTorch con supporto CUDA.")
        return
        
    device = torch.device("cuda")
    print(f"--> Rilevata GPU: {torch.cuda.get_device_name(0)}")
    
    # 2. Definizione del carico di lavoro (Matrice 16k x 16k float32)
    N = 16384  
    matrix_size_bytes = N * N * 4  
    matrix_size_mb = matrix_size_bytes / (1024 * 1024)
    
    # Calcolo dei FLOPs teorici (2 * N^3)
    theoretical_flops = 2 * (N ** 3)
    theoretical_tflops = theoretical_flops / 1e12
    
    # 3. Calcolo delle latenze ideali CORRETTO
    # Convertiamo la banda teorica in MB/s per uniformarla al peso del tensore
    theoretical_bandwidth_mbs = THEORETICAL_BANDWIDTH_GBS * 1024
    
    ideal_compute_latency_ms = (theoretical_tflops / THEORETICAL_PEAK_TFLOPS) * 1000
    ideal_comm_latency_ms = (matrix_size_mb / theoretical_bandwidth_mbs) * 1000
    ideal_total_latency_ms = ideal_compute_latency_ms + ideal_comm_latency_ms
    
    print(f"\n[Calcoli Ideali Modello Analitico]:")
    print(f"  - Dimensione Dati Tensor: {matrix_size_mb:.2f} MB")
    print(f"  - Complessità del Blocco: {theoretical_tflops:.4f} TFLOPs")
    print(f"  - Latenza Compute Ideale: {ideal_compute_latency_ms:.4f} ms")
    print(f"  - Latenza Comm/Transfer Ideale: {ideal_comm_latency_ms:.4f} ms")
    print(f"  - Latenza Totale Stimata: {ideal_total_latency_ms:.4f} ms")
    
    # Warm-up per stabilizzare i clock della GPU
    print("\nRiscaldamento dei core della GPU in corso...")
    for _ in range(10):
        A_dummy = torch.randn(1024, 1024, device=device)
        _ = torch.matmul(A_dummy, A_dummy)
    torch.cuda.synchronize()
        
    print("Esecuzione delle misurazioni empiriche...")
    
    # 4. Misurazioni reali hardware
    # Creiamo gli eventi CUDA
    start_comm = torch.cuda.Event(enable_timing=True)
    end_comm = torch.cuda.Event(enable_timing=True)
    start_compute = torch.cuda.Event(enable_timing=True)
    end_compute = torch.cuda.Event(enable_timing=True)

    # Misura Comm/Setup PURA
    start_comm.record()
    A = torch.randn(N, N, device=device)
    B = torch.randn(N, N, device=device)
    end_comm.record()
    
    # Sincronizziamo e leggiamo il tempo hardware (in millisecondi)
    torch.cuda.synchronize()
    actual_comm_latency_ms = start_comm.elapsed_time(end_comm)

    # Misura Compute PURA
    start_compute.record()
    C = torch.matmul(A, B)
    end_compute.record()
    
    torch.cuda.synchronize()
    actual_compute_latency_ms = start_compute.elapsed_time(end_compute)

    actual_total_latency_ms = actual_comm_latency_ms + actual_compute_latency_ms
    
    print(f"\n[Misure Reali Hardware Laptop]:")
    print(f"  - Latenza Compute Misurata: {actual_compute_latency_ms:.4f} ms")
    print(f"  - Latenza Comm/Setup Misurata: {actual_comm_latency_ms:.4f} ms")
    print(f"  - Latenza Totale Effettiva: {actual_total_latency_ms:.4f} ms")
    
    # 5. Estrazione dei corretti fattori di scala (k)
    k_compute = actual_compute_latency_ms / ideal_compute_latency_ms
    k_comm = actual_comm_latency_ms / ideal_comm_latency_ms
    k_total = actual_total_latency_ms / ideal_total_latency_ms
    
    print("\n==================================================")
    print("      FATTORI DI SCALA DI CALIBRAZIONE (k)        ")
    print("==================================================")
    print(f"  👉 Fattore Compute (k_compute): {k_compute:.2f}x")
    print(f"  👉 Fattore Comm/Interconnect (k_comm): {k_comm:.2f}x")
    print(f"  👉 Moltiplicatore di Overhead Globale (k_total): {k_total:.2f}x")
    print("==================================================")
    # Aggiungi questo in fondo al tuo laptop5090.py
    print(f"RAW_RESULTS: {actual_compute_latency_ms:.4f} , {actual_comm_latency_ms:.4f}")

if __name__ == "__main__":
    run_rtx5090_calibration_fixed()

# ==================================================
#           RISULTATI STATISTICI FINALI             
# ==================================================
# [Misure da Script Python (CUDA Events)]:
#   🔹 Compute (PyTorch): 483.8035 ms (± 18.6811 ms)
#   🔹 Comm/Setup (PyTorch): 4.1327 ms (± 0.0219 ms)

# [Misure Hardware Pure da Profiler NSYS]:
#   🔹 Kernel GPU Puro (Cutlass): 483.6005 ms (± 18.6830 ms)
#   🔹 Driver Overhead (cudaMalloc): 0.8988 ms (± 0.1746 ms)

# PyTorch comm/setup contiene:
# allocazione tensori GPU
# inizializzazione random (kernel GPU!)
# eventuali memset
# caching allocator PyTorch
# possibili kernel ausiliari

# NSYS “cudaMalloc” invece è solo:
# chiamata API per allocare memoria GPU