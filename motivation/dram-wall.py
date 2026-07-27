import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 1. Configurazione dei file e dei parametri grafici
files = {
    'full-split': 'llava_ee/full-split-llava_7B_ee_correct.csv',
    'part-split': 'llava_ee/part-split-llava_7B_ee_correct.csv',
    'no-split': 'llava_ee/no-split-llava_7B_ee_correct.csv'
}

# check if files exist
for split_type, filepath in files.items():
    try:
        with open(filepath, 'r') as f:
            pass
    except FileNotFoundError:
        print(f"File non trovato: {filepath}")

markers = {'full-split': 's', 'part-split': 'D', 'no-split': '^'}
colors = {'full-split': 'purple', 'part-split': 'gray', 'no-split': 'teal'}

# FATTORI DI CONVERSIONE (Modifica in base ai clock/scalers del tuo esperimento)
# Assumiamo che l'energia in MAESTRO sia in nJ e la convertiamo in J
ENERGY_CONVERT = 1e9  
# Assumiamo un fattore di scala per i cicli (sostituisci con il tuo 30000000000 se necessario)
CYCLES_TO_SEC = 30000000000  

fig, ax = plt.subplots(figsize=(10, 7))

# 2. Elaborazione dati e Plotting
for split_type, filepath in files.items():
    try:
        # Molte versioni di MAESTRO usano il tab o la virgola, gestiamo gli spazi
        df = pd.read_csv(filepath, sep=None, engine='python', skipinitialspace=True)
        df.columns = df.columns.str.strip()
        
        # MAESTRO aggiunge una riga finale di "Total" (con NaN nei nomi/layer). La rimuoviamo per il calcolo cumulativo.
        df_layers = df.dropna(subset=['Layer Number']).copy()
        
        # Calcolo somme cumulative per tracciare la traiettoria di esecuzione
        df_layers['Cum_Latency_s'] = df_layers['Runtime (Cycles)'].cumsum() / CYCLES_TO_SEC
        df_layers['Cum_Energy_J'] = df_layers['Activity count-based Energy (nJ)'].cumsum() / ENERGY_CONVERT
        
        # Plot della linea di traiettoria
        ax.plot(df_layers['Cum_Latency_s'], df_layers['Cum_Energy_J'], 
                color=colors[split_type], linestyle='-', linewidth=2, alpha=0.6, zorder=1)
        
        # Identifichiamo il punto finale (Modello Completo, 32 blocchi)
        final_lat = df_layers['Cum_Latency_s'].iloc[-1]
        final_ene = df_layers['Cum_Energy_J'].iloc[-1]
        
        # Marker per il modello completo
        ax.scatter(final_lat, final_ene, 
                   s=150, marker=markers[split_type], color=colors[split_type], 
                   edgecolor='k', linewidth=1.5, zorder=3, label=f'{split_type} (Full)')
        
        # Ipotizziamo dei punti di Early Exit (es. a 1/4, 2/4 e 3/4 del modello)
        # Se sai esattamente a che riga/layer si trova l'EE, sostituisci gli indici
        num_layers = len(df_layers)
        ee_points = [num_layers//4, num_layers//2, int(num_layers*0.75)]
        
        for idx in ee_points:
            ee_lat = df_layers['Cum_Latency_s'].iloc[idx]
            ee_ene = df_layers['Cum_Energy_J'].iloc[idx]
            ax.scatter(ee_lat, ee_ene, 
                       s=80, marker=markers[split_type], color=colors[split_type], 
                       alpha=0.7, zorder=2)
            
    except FileNotFoundError:
        print(f"File non trovato: {filepath}")

# 3. Estetica e Annotazioni (Il "DRAM Wall")
# Creiamo un'area sfumata rossa nella parte alta per evidenziare la zona DRAM-bound
ax.axhspan(15, 25, color='coral', alpha=0.15, zorder=0)
ax.text(0.01, 20, 'DRAM-Bound Execution Zone\n(High Off-Chip Memory Traffic)', 
        fontsize=12, color='darkred', alpha=0.8, weight='bold',
        transform=ax.get_yaxis_transform())

# Area sicura (SRAM / Early Exit)
ax.axhspan(0, 5, color='green', alpha=0.07, zorder=0)
ax.text(0.01, 2, 'SRAM / Early-Exit Zone\n(High Efficiency)', 
        fontsize=12, color='darkgreen', alpha=0.8, weight='bold',
        transform=ax.get_yaxis_transform())

# Labels e Titolo
ax.set_xlabel('Latency [s]', fontsize=14)
ax.set_ylabel('Total Energy [J]', fontsize=14)
ax.set_title('LLaVA-7B Energy Trajectory: The DRAM Wall', fontsize=16, weight='bold')

ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=12, loc='upper left')

# Se l'energia diverge esponenzialmente, decommenta la linea seguente per asse logaritmico
# ax.set_yscale('log')

plt.tight_layout()
plt.savefig('llava_dram_wall.pdf', format='pdf', dpi=300)
plt.show()