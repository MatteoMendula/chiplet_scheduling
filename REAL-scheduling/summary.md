# Empirical Calibration and Cross-Platform Microarchitectural Validation

To bridge the gap between high-level analytical abstraction (e.g., MAESTRO) and hardware-level execution realities, we introduce a platform-specific empirical calibration framework into CLEAR. This approach directly addresses the over-optimism of theoretical models regarding inter-chiplet synchronization, operating system runtime interference, and memory system stalls by profiling execution on physical target hardware.

## 1. Methodology: Unified Host-Device Latency Accounting

Analytical models typically isolate transmission delays by dividing tensor sizes by nominal physical bandwidth. However, real-world deployment introduces substantial software and hardware friction before and during data movement. To capture the true system-level overhead, we model the actual communication and setup latency ($t_{\text{comm}}$) as a sequential combination of host-side memory mapping and device-side tensor initialization:

$$t_{\text{comm}} = t_{\text{cudaMalloc}} + t_{\text{transfer}}$$

Where:
* $t_{\text{cudaMalloc}}$ represents the host OS/driver overhead to allocate and pin memory blocks (measured via NVIDIA Nsight Systems).
* $t_{\text{transfer}}$ captures the hardware-level data movement and kernel setup queue (measured via PyTorch CUDA Events).

We define the microarchitectural scaling factors ($k_{\text{compute}}$ and $k_{\text{comm}}$) as the ratio between the empirical measurements ($\mu \pm \sigma$ over 100 profiling runs) and the peak analytical limits derived from vendor specifications:

$$k = \frac{t_{\text{measured}}}{t_{\text{ideal}}}$$

---

## 2. Experimental Setup and Hardware Profiles

The calibration micro-benchmark executes an isolated, intensive FP32 matrix multiplication ($16384 \times 16384$ elements, yielding an operational load of $8.7961 \text{ TFLOPs}$ and a data payload of $1024.00 \text{ MB}$) across two highly heterogeneous hardware paradigms:

1. **Discrete GPU Architecture (High-End Edge Server Node):** NVIDIA GeForce RTX 5090 Mobile (Blackwell, $33.52 \text{ TFLOPS}$ FP32 peak, $896.0 \text{ GB/s}$ dedicated GDDR7 bandwidth).
2. **Shared-Memory SoC Architecture (Constrained Far-Edge Node):** NVIDIA Jetson Orin Nano 8GB (Ampere, $1.28 \text{ TFLOPS}$ FP32 peak, $68.0 \text{ GB/s}$ unified LPDDR5 bandwidth).

---

## 3. Quantitative Evaluation and $k$-Factor Extraction

The statistical empirical results gathered over 100 independent iterations are summarized in the table below:

| Platform / Metric | Analytical Ideal ($t_{\text{ideal}}$) | Empirical Real ($\mu \pm \sigma$) | Calibration Factor ($k$) |
| :--- | :--- | :--- | :--- |
| **RTX 5090 Mobile (Discrete)** | | | |
| * Compute Latency | $262.4133 \text{ ms}$ | $483.8035 \text{ ms } (\pm 18.6811 \text{ ms})$ | $\mathbf{1.8437x } (\pm 0.0712x)$ |
| * Comm/Setup Latency | $1.1161 \text{ ms}$ | $4.1327 \text{ ms } (\pm 0.0219 \text{ ms})$ | $\mathbf{4.5082x } (\pm 0.1577x)$ |
| * Total Latency | $263.5293 \text{ ms}$ | $487.9362 \text{ ms } (\pm 18.6811 \text{ ms})$ | $\mathbf{1.8550x } (\pm 0.0709x)$ |
| **Jetson Orin Nano (SoC)** | | | |
| * Compute Latency | $6871.9477 \text{ ms}$ | $14221.8602 \text{ ms } (\pm 18.6811 \text{ ms})$ | $\mathbf{2.0696x } (\pm 0.0027x)$ |
| * Comm/Setup Latency | $14.7059 \text{ ms}$ | $2833.5142 \text{ ms } (\pm 232.9221 \text{ ms})$ | $\mathbf{192.6790x } (\pm 15.8387x)$ |
| * Total Latency | $6886.6536 \text{ ms}$ | $17055.3744 \text{ ms } (\pm 233.6700 \text{ ms})$ | $\mathbf{2.4766x } (\pm 0.0339x)$ |

---

## 4. Microarchitectural Insights and Discussion

The extracted $k$-factors reveal two critical structural behaviors that validate our modeling over pure analytical simulation:

### Compute Efficiency Invariance ($k_{\text{compute}}$)
Despite a massive performance disparity in absolute terms, both platforms exhibit comparable scaling penalties for pure arithmetic execution ($1.84x$ for the RTX 5090 vs. $2.07x$ for the Jetson Orin Nano). This standard deviation confirms a predictable, bounded degradation caused by thread scheduling, micro-op queues, and localized thermal-throttling profiles, indicating that the Tensor and CUDA cores achieve roughly 54.2% and 48.3% of their respective marketing peaks under sustained workload.

### The Shared-Memory Communication Penalty ($k_{\text{comm}}$)
While $k_{\text{comm}}$ scales moderately on the discrete GPU ($4.51x$), it experiences a catastrophic explosion on the edge SoC, reaching **$192.68x \ (\pm 15.84x)$**. Detailed profiling via NVIDIA Nsight Systems isolates the primary root cause within the driver-level host-allocation loop: `cudaMalloc` accounts for **$2121.8749 \text{ ms } (\pm 223.9523 \text{ ms})$** of the total communication block. 

On a shared-resource System-on-Chip (SoC) where the CPU and GPU share a single unallocatable LPDDR5 RAM pool, virtual address space fragmentation and OS-level memory pinning introduce intense resource contention. Analytical frameworks that merely scale transmission times linearly based on physical bus width completely omit this deterministic, high-variance OS barrier.

---

## 5. Integration into the CLEAR Optimization Engine

Instead of relying on a generalized, flat penalty factor, the CLEAR optimization framework dynamically injects these platform-specific multipliers into its cost function. The decision variable space maps directly to the calibrated values:

* For discrete edge nodes, the communication penalty $\delta$ is scaled by the empirical factor $4.51x$.
* For unified-memory SoC nodes, $\delta$ is scaled by $192.68x$.

By acknowledging the $192.68x$ memory setup barrier on far-edge nodes, CLEAR mathematically formalizes why aggressive pipeline-splitting or excessive layer distributed-handoffs fail at the edge when batch size = 1. The optimization loop naturally leans toward localized, deeper processing configurations on memory-constrained nodes, choosing early exits (EE) strategically to drop intermediate tensor transfer states ($s(l,n) \rightarrow 0$) and prevent the massive system-level communication overhead from compounding layer by layer across the network-on-package (NoP).