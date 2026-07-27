import matplotlib.pyplot as plt
import numpy as np

# 1. Definizione dei modelli in ordine di complessita'
models = ['B-LeNet\n(Activity Filter)', 'B-ResNet\n(Object Detection)', 'DoViT\n(Semantic Segm.)', 'LLaVA\n(Visual Reasoning)']

# 2. Dati Esatti di Energia (in mJ) estratti da MAESTRO
energy_no_ee = [0.015, 1084.17, 367.10, 20311.0]   
energy_with_ee = [0.0004, 21.20, 7.58, 5531.0]     

# 3. Dati di Latenza (in Secondi)
# Calcolati usando i cicli estratti da MAESTRO divisi per 30 GHz, 
# e stimati per LLaVA secondo il constraint del paper (< 5s)
lat_no_ee = [1.55e-6, 0.255, 0.051, 4.8]  
lat_with_ee = [5.78e-7, 0.0039, 0.016, 1.2]    

# 4. Parametri per il raggruppamento delle barre
x = np.arange(len(models))
width = 0.2  # Larghezza delle singole barre

# 5. Setup della figura con doppio asse Y
fig, ax1 = plt.subplots(figsize=(12, 7))
ax2 = ax1.twinx()  # Crea il secondo asse che condivide lo stesso asse X

# 6. Tracciamento delle barre
# Gruppo Energia (Asse Y Sinistro)
rects1 = ax1.bar(x - 1.5*width, energy_no_ee, width, label='Energy: Monolithic', 
                 color='#8B0000', edgecolor='black', zorder=3)
rects2 = ax1.bar(x - 0.5*width, energy_with_ee, width, label='Energy: CLEAR (EE-1)', 
                 color='#FF6347', edgecolor='black', zorder=3)

# Gruppo Latenza (Asse Y Destro)
rects3 = ax2.bar(x + 0.5*width, lat_no_ee, width, label='Latency: Monolithic', 
                 color='#006400', edgecolor='black', zorder=3)
rects4 = ax2.bar(x + 1.5*width, lat_with_ee, width, label='Latency: CLEAR (EE-1)', 
                 color='#3CB371', edgecolor='black', zorder=3)

# 7. Impostazione della Scala Logaritmica per entrambi gli assi
ax1.set_yscale('log')
ax2.set_yscale('log')

# Limiti assi per spaziatura
ax1.set_ylim(0.0001, 100000)
ax2.set_ylim(1e-7, 100)

# 8. Creazione della "Unfeasible Zone" (Sull'asse dell'energia)
threshold_energy = 100.0 
ax1.axhline(y=threshold_energy, color='red', linestyle='--', linewidth=2, zorder=2)
ax1.text(2.0, threshold_energy * 1.5, 'UNFEASIBLE REGION (DRAM-Bound)', 
         color='red', fontsize=12, fontweight='bold', alpha=0.8, ha='center')

# 9. Formattazione e Stile Accademico
ax1.set_xlabel('AI Application & Model Architecture', fontsize=14, fontweight='bold', color='black')
ax1.set_ylabel('Total Inference Energy [mJ] (Log Scale)', fontsize=14, fontweight='bold', color='#8B0000')
ax2.set_ylabel('Inference Latency [s] (Log Scale)', fontsize=14, fontweight='bold', color='#006400')
plt.title('Energy & Latency Scaling: Monolithic vs CLEAR', fontsize=16, fontweight='bold', color='black', pad=15)

ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=12, fontweight='bold')

# Unione delle legende dei due assi in un unico riquadro in alto a sinistra
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11, framealpha=0.9, edgecolor='black')

ax1.grid(True, which="major", axis='y', ls="-", alpha=0.3, zorder=0)

# Mostra e Salva
plt.tight_layout()
plt.savefig('energy_latency_grouped_scaling.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()