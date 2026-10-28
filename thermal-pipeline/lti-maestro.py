import pandas as pd
import numpy as np

# 1. Caricamento dati MAESTRO
df = pd.read_csv("./data/llava_7B_ee_correct.csv")
df.columns = df.columns.str.strip() # Pulizia nomi colonne
power = df['Power'].values  
cycles = df['Runtime (Cycles)'].values
layers = df['Layer Number'].values

# 2. Parametri del modello topologico e ThermLeT
T_amb = 40.0
R = 0.01      # Resistenza termica del chiplet target
tau = 5e7     # Costante di tempo termica in cicli
THERMLET_CEILING = 72.0 # Soglia critica dettata dal modello surrogato

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