# CLEAR Project: Hardware Calibration and Early Exits (EE)

This text summarizes the state of the CLEAR framework, the analysis of microarchitectural criticisms raised by Reviewer 1 (IEEE TMC), and the results of an empirical calibration experiment conducted on a local GPU to align the analytical simulator with real silicon behavior.

---

## 1. Context and Reviewer 1 Criticism

Reviewer 1 questioned the accuracy of the analytical simulator **MAESTRO**. The main objection is that high-level mathematical models fail to capture real silicon frictions in Multi-Chip Module (MCM) systems, particularly:

* **Synchronization and Switching Costs:** The assumption that accelerators operate independently from the host processor (CPU) overlooks driver overhead and control-flow stalls.
* **Memory System Limitations:** Modern workloads such as Transformers, ViTs, and LVLMs are memory-bound rather than compute-bound.
* **Internal SRAM Frictions:** Phenomena like SRAM banking conflicts and tensor rematerialization introduce latency and energy overheads that MAESTRO ignores, leading to overly optimistic estimates.

---

## 2. The Two-Parameter Profiling Paradox

To address the critique regarding the ineffectiveness of Early Exits (EE) with batch size = 1, it is necessary to map the impact of scheduling decisions onto two key parameters defined in the CLEAR model:

1. 
**$\kappa(l,n)$ (Compute Intensity):** Denotes the number of units of computational capabilities needed by a block to process one unit of traffic.


2. 
**$s(l,n)$ (Data Size):** Denotes the size of the data originating from a node after it has been processed by a block.



### Misalignment of Perspectives

* **Reviewer’s perspective:** Focuses exclusively on $\kappa(l,n)$. The claim is that with batch size = 1, the additional compute overhead required to evaluate early-exit confidence cancels out the benefit of skipping deeper network layers.
* **Our perspective (MCM reality):** Focuses on $s(l,n)$. In a distributed chiplet architecture, the real bottleneck is data movement across the network-on-package (NoP).


If an Early Exit succeeds, intermediate computation is terminated early without executing the remaining deeper layers , and the data size to be transmitted to the next chiplet drops significantly or completely bypasses deeper transfers.


The reduction in latency and energy consumption from avoiding the transfer of large activation matrices across the NoP more than compensates for the small compute overhead added to $\kappa(l,n)$, even when batch size = 1.

---

## 3. Calibration Experiment (RTX 5090 Mobile)

To demonstrate the impact of silicon friction and calibrate the simulator, a micro-benchmark was executed on an **NVIDIA GeForce RTX 5090 Laptop GPU** (Blackwell architecture, theoretical memory bandwidth of **896.0 GB/s**, and estimated FP32 peak performance of **33.52 TFLOPS**).

The test consisted of an isolated large matrix multiplication run across 100 iterations to establish robust empirical data:

* Size: $16384 \times 16384$ (Float32)
* Data payload: **1024.00 MB**
* Computational complexity: **8.7961 TFLOPs**

### Benchmark Results (100-Run Statistics)

| Metric | Ideal Model | Real Measurement (PyTorch/NSYS) | Scaling Factor (k) |
| --- | --- | --- | --- |
| **Compute Latency** | **262.4133 ms** | **483.8035 ms (± 18.6811 ms)** | **1.8437x (± 0.0712x)** |
| **Comm/Setup Latency** | **1.1161 ms** | **4.1327 ms (± 0.0219 ms)** | **3.7029x (± 0.0196x)** |
| **Total Latency** | **263.5293 ms** | **487.9362 ms (± 18.6811 ms)** | **1.8515x (± 0.0709x)** |

---

## 4. Analysis of Scaling Factors (k)

The extracted multipliers represent empirical correction coefficients for the simulator:

* **k_compute = 1.8437x (± 0.0712x)** The real GPU requires 84.37% more time than ideal computation. This reflects the gap between theoretical TFLOPS and actual execution due to thermal limits, power-throttling in mobile profiles, and CUDA kernel configuration efficiency. Under this massive workload, the GPU operates at roughly 54.23% of its theoretical peak efficiency.
* **k_comm = 3.7029x (± 0.0196x)** This directly addresses the reviewer’s concern. Memory setup, API binding, and driver invocation take nearly 3.7× longer than the raw physical bandwidth would suggest. In-depth profiling with NVIDIA Nsight Systems confirms that underlying driver processes like `cudaMalloc` consume **0.8988 ms (± 0.1746 ms)** of pure system overhead before physical data initialization even begins. This hidden friction represents system-level interconnect overhead that MAESTRO does not model natively.


* **Why is total latency 1.8515x instead of averaging 1.8437x and 3.7029x?** The overall factor is a **time-weighted average**. In this macro-level experiment, the workload is heavily compute-bound:
* **99.15%** of the total time is spent on pure computation (~483.80 ms)
* **0.85%** of the total time is spent on memory setup (~4.13 ms)


As a result, the global scaling multiplier is dominated by compute efficiency.

However, in distributed MCM systems, workloads are split and pipelined, making them inherently communication-bound. Frequent block handoffs across the NoP cause the **3.7029x communication penalty** to accumulate layer by layer, making it the hidden dominant factor in overall performance.

---

## 5. Next Steps Strategy (Call Preparation)

To align the team and prepare for resubmission:

1. 
**Preserve mathematical formulation** The deterministic optimization structure remains unchanged. The probabilistic nature of Early Exits will be handled by updating input parameters ($\kappa$ and $s$) using expected values weighted by dataset difficulty before running the optimization.


2. **Jetson validation** Replicate the benchmark on the target Jetson platform using its theoretical hardware specifications (TOPS and memory bandwidth).
As a shared-resource edge SoC, it is expected that k_comm will exceed 3.7029x due to tighter memory contention.
3. 
**Simulator calibration** Inject the measured k factors into the analytical model by scaling communication bandwidth and latency parameters ($\delta$).


This produces a simulator grounded in real hardware measurements, strengthening its credibility.