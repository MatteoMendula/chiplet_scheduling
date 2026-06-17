#!/bin/bash

# File temporanei per salvare i dati grezzi
COMPUTE_PY_FILE="py_compute.txt"
COMM_PY_FILE="py_comm.txt"
COMPUTE_GPU_FILE="gpu_compute.txt"
MALLOC_API_FILE="gpu_malloc.txt"

# Pulizia file precedenti
rm -f $COMPUTE_PY_FILE $COMM_PY_FILE $COMPUTE_GPU_FILE $MALLOC_API_FILE
touch $COMPUTE_GPU_FILE
touch $MALLOC_API_FILE

# Modifica qui il numero di run desiderati
NUM_RUNS=100

echo "=================================================="
echo "  Avvio Benchmark Automatizzato: $NUM_RUNS Iterazioni"
echo "=================================================="

for i in $(seq 1 $NUM_RUNS); do
    echo -ne "Esecuzione run $i/$NUM_RUNS...\r"
    
    # Esegue nsys e forza la sovrascrittura
    nsys profile --trace=cuda,osrt --stats=false -o temp_report_run --force-overwrite=true python laptop5090.py > temp_output.txt
    
    # 1. Estrai i risultati RAW dallo script Python
    py_raw=$(grep "RAW_RESULTS:" temp_output.txt | awk '{print $2, $4}')
    py_comp=$(echo $py_raw | awk '{print $1}')
    py_comm=$(echo $py_raw | awk '{print $2}')
    echo $py_comp >> $COMPUTE_PY_FILE
    echo $py_comm >> $COMM_PY_FILE
    
    # 2. Usa 'nsys stats' per esportare i report in CSV
    nsys stats --report=cuda_gpu_kern_sum --format=csv --force-export=true temp_report_run.nsys-rep > temp_kern.csv
    nsys stats --report=cuda_api_sum --format=csv --force-export=true temp_report_run.nsys-rep > temp_api.csv
    
    # 3. Isola il tempo del kernel CUTLASS (Max time in ns, colonna 7, pulito da spazi e VIRGOLETTE)
    # Cerchiamo "cutlass" o "gemm" (ignorando maiuscole/minuscole) per assicurarci di prendere il kernel di calcolo
    gpu_kern_ns=$(grep -iE "cutlass|gemm" temp_kern.csv | head -n 1 | cut -d',' -f7 | tr -d '[:space:]"')
    
    if [[ "$gpu_kern_ns" =~ ^[0-9.]+$ ]]; then
        gpu_kern_ms=$(echo "$gpu_kern_ns / 1000000" | bc -l)
        echo $gpu_kern_ms >> $COMPUTE_GPU_FILE
    fi
    
    # 4. Isola il tempo di cudaMalloc (Total Time in ns, colonna 2, pulito da spazi e VIRGOLETTE)
    malloc_ns=$(grep "cudaMalloc" temp_api.csv | head -n 1 | cut -d',' -f2 | tr -d '[:space:]"')

    if [[ "$malloc_ns" =~ ^[0-9.]+$ ]]; then
        malloc_ms=$(echo "$malloc_ns / 1000000" | bc -l)
        echo $malloc_ms >> $MALLOC_API_FILE
    fi

done

echo -e "\nEsecuzioni completate! Calcolo delle statistiche in corso...\n"

# Funzione interna in Python rapida per calcolare Media e StdDev
calculate_stats() {
    local file=$1
    local label=$2
    python -c "
import math, sys
label = '$label'
try:
    lines = [float(x.strip()) for x in open('$file').readlines() if x.strip()]
    if not lines: 
        print(f'  🔹 {label}: Dati non trovati!')
        sys.exit()
    n = len(lines)
    mean = sum(lines) / n
    variance = sum((x - mean) ** 2 for x in lines) / n
    stddev = math.sqrt(variance)
    print(f'  🔹 {label}: {mean:.4f} ms (± {stddev:.4f} ms)')
except Exception as e:
    print(f'  🔹 {label}: Errore nella lettura dati ({e})')
"
}

echo "=================================================="
echo "          RISULTATI STATISTICI FINALI             "
echo "=================================================="
echo "[Misure da Script Python (CUDA Events)]:"
calculate_stats $COMPUTE_PY_FILE "Compute (PyTorch)"
calculate_stats $COMM_PY_FILE "Comm/Setup (PyTorch)"

echo -e "\n[Misure Hardware Pure da Profiler NSYS]:"
calculate_stats $COMPUTE_GPU_FILE "Kernel GPU Puro (Cutlass)"
calculate_stats $MALLOC_API_FILE "Driver Overhead (cudaMalloc)"
echo "=================================================="

# Pulizia finale dei file temporanei pesanti
rm -f temp_report_run.nsys-rep temp_report_run.sqlite temp_output.txt temp_kern.csv temp_api.csv