import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Carichiamo il file
filepath = 'llava_ee/part-split-llava_7B_ee_correct.csv'

try:
    df = pd.read_csv(filepath, sep=None, engine='python', skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df_layers = df.dropna(subset=['Layer Number']).copy()
    
    # --- NUOVO CALCOLO LATENZA ---
    df_layers['Cum_Latency_s'] = df_layers['Runtime (Cycles)'].cumsum() / 30000000000
    
    # --- ASSE X ---
    # Usiamo il numero progressivo del layer
    layers_x = range(1, len(df_layers) + 1)
    
    # --- ASSE Y SINISTRO (SRAM) ---
    # Calcolo della Peak L2 SRAM in KB
    l2_sram_kb = df_layers['L2 SRAM Size Req (Bytes)'] / 1024
    
    # --- ASSE Y DESTRO (Memoria Cumulativa DRAM) ---
    # Assumiamo formato FP16 (2 byte). Convertiamo in GB.
    layer_memory_bytes = (df_layers.get('Num Total Weight Pixels', 0) + df_layers.get('Num Total Input Pixels', 0)) * 2
    cum_memory_gb = layer_memory_bytes.cumsum() / (1024**3)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 1. Plot della SRAM attiva (Asse Sinistro)
    color_sram = 'tab:blue'
    ax1.set_xlabel('Layer Number', fontsize=14, weight='bold')
    ax1.set_ylabel('Active L2 SRAM Req. [KB]', fontsize=14, weight='bold', color=color_sram)
    line1 = ax1.plot(layers_x, l2_sram_kb, color=color_sram, linestyle='-', linewidth=2.5, label='Active SRAM (Tile Size)')
    ax1.tick_params(axis='y', labelcolor=color_sram)

    # Aggiungiamo una linea orizzontale per indicare la capacità massima della SRAM L2 (6 MB = 6000 KB)
    ax1.axhline(y=6000/1024, color='tab:blue', linestyle=':', linewidth=2, label='L2 SRAM Capacity')
    
    # Fissiamo un limite Y che mantenga la linea piatta visibilmente in basso (es. 50 KB)
    ax1.set_ylim(0, max(50, l2_sram_kb.max() * 1.5)) 
    
    # 2. Plot della Memoria Cumulativa (Asse Destro)
    ax2 = ax1.twinx()
    color_dram = 'tab:red'
    ax2.set_ylabel('Cumulative DRAM Footprint [GB]', fontsize=14, weight='bold', color=color_dram)
    line2 = ax2.plot(layers_x, cum_memory_gb, color=color_dram, linestyle='--', linewidth=2.5, label='Cumulative Memory Fetched')
    ax2.tick_params(axis='y', labelcolor=color_dram)
    ax2.set_ylim(-0.2, 18)
    
    # 3. Posizionamento Marker di Early Exit (Sulla linea della Memoria Cumulativa)
    total_layers = len(df_layers)
    idx_ee1 = int(total_layers * 0.40)
    idx_ee2 = int(total_layers * 0.70)
    idx_ee3 = int(total_layers * 0.90)
    idx_full = total_layers - 1
    
    # Removed hardcoded colors, kept markers
    points = [
        (idx_ee1, 'CLEAR: EE 1', 'o'),
        (idx_ee2, 'CLEAR: EE 2', 'o'),
        (idx_ee3, 'CLEAR: EE 3', 'o'),
        (idx_full, 'Full Model', 'D')
    ]

    # --- CONFIGURAZIONE HEATMAP (LATENZA) ---
    latencies = [df_layers['Cum_Latency_s'].iloc[idx] for idx, _, _ in points]
    norm = mcolors.Normalize(vmin=min(latencies), vmax=max(latencies))
    cmap = plt.cm.coolwarm # Blue (low latency) to Red (high latency)

    memory_e1 = None
    
    for idx, label, marker in points:
        x_val = layers_x[idx]
        y_val = cum_memory_gb.iloc[idx]
        latency_val = df_layers['Cum_Latency_s'].iloc[idx]
        
        # Calculate color dynamically based on latency
        point_color = cmap(norm(latency_val))
        
        ax2.scatter(x_val, y_val, s=130, color=point_color, marker=marker, edgecolor='black', linewidth=1.5, zorder=5)
        
        # Posizionamento etichette (Text forced to black for clear visibility against the heatmap)
        if 'Full' in label:
            ax2.text(x_val - 1, y_val + 0.5, label, fontsize=12, weight='bold', color='black', ha='right', va='bottom')
        elif 'EE 1' in label:
            ax2.text(x_val - 1, y_val + 0.25, label, fontsize=12, weight='bold', color='black', ha='right', va='bottom')
        else:
            ax2.text(x_val - 1, y_val + 0.5, label, fontsize=12, weight='bold', color='black', ha='right', va='bottom')


    # --- AGGIUNTA COLORBAR PER LA LATENZA ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # The pad prevents the colorbar from overlapping the right Y-axis labels
    cbar = fig.colorbar(sm, ax=ax2, pad=0.15) 
    cbar.set_label('Cumulative Latency [s]', fontsize=13, weight='bold')

    # --- FEASIBILITY / UNFEASIBILITY ZONES (Tied to ax2 - DRAM Footprint) ---
    
    # Define the threshold where cumulative memory fetching breaks the energy/latency SLA.
    # Looking at the plot, the ramp-up becomes aggressive after ~4-5 GB.
    SLA_MEMORY_THRESHOLD_GB = 2
    MAX_MEMORY_GB = 18.0 # Just slightly above your max y-limit for ax2
    
    # Unfeasible Zone (Red)
    ax2.axhspan(SLA_MEMORY_THRESHOLD_GB, MAX_MEMORY_GB, color='coral', alpha=0.15, zorder=0)
    ax2.text(10, SLA_MEMORY_THRESHOLD_GB + 2.5, 
             'Unfeasible Zone\n(Massive DRAM fetch)', 
             fontsize=13, color='darkred', weight='bold')

    # Feasible Zone (Green)
    ax2.axhspan(-10, SLA_MEMORY_THRESHOLD_GB, color='green', alpha=0.1, zorder=0)
    ax2.text(200, SLA_MEMORY_THRESHOLD_GB - 0.7, 
             'Feasible Zone', 
             fontsize=13, color='darkgreen', weight='bold')

    # 5. Titolo e Legenda
    # plt.title('The SRAM Paradox: Flat Active Memory vs. Massive Cumulative DRAM Fetch', fontsize=14, weight='bold')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    # Posizioniamo la legenda in un punto vuoto (centro-sinistra)
    ax1.legend(lines, labels, loc='upper left', fontsize=11)

    ax1.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('llava_memory_paradox.pdf', format='pdf', dpi=300)
    plt.show()

except Exception as e:
    print(f"Errore: {e}")