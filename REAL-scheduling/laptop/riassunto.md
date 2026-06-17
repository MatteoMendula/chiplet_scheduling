# Progetto CLEAR: Calibrazione dell'Hardware ed Early Exits (EE)

Questo documento riassume lo stato dell'arte del framework CLEAR, l'analisi delle critiche microarchitetturali sollevate dal Reviewer 1 di IEEE TMC, e i risultati dell'esperimento di calibrazione empirica eseguito su GPU locale per allineare il simulatore analitico alla realtà del silicio.

---

## 1. Il Contesto e le Critiche del Reviewer 1

Il Reviewer 1 ha contestato l'accuratezza del simulatore analitico **MAESTRO**. L'obiezione principale è che i modelli matematici ad alto livello non catturano le frizioni reali del silicio nei sistemi Multi-Chip Module (MCM), in particolare:
* **Costi di Sincronizzazione e Switching:** L'assunzione che gli acceleratori lavorino in modo completamente indipendente dal processore host (CPU) trascura l'overhead dei driver e i blocchi di controllo.
* **Limiti del Sistema di Memoria:** I moderni carichi Transformer, ViT e LVLM sono vincolati dal sistema di memoria (*memory-bound*) e non dal calcolo (*compute-bound*).
* **Frizioni interne all'SRAM:** Fenomeni come i conflitti bancari dell'SRAM (*SRAM banking conflicts*) e la ricalcolazione dei tensori (*tensor rematerialization*) introducono ritardi e consumi energetici che MAESTRO ignora, fornendo stime troppo ottimistiche.

---

## 2. Il Paradosso dei due Parametri di Profilazione

[cite_start]Per rispondere alla critica del reviewer sull'inutilità degli Early Exits (EE) con batch size = 1, occorre mappare l'impatto della scelta di scheduling sui due parametri chiave definiti nel modello di CLEAR[cite: 396]:

1. [cite_start]**$\kappa(l,n)$ (Intensità di Calcolo):** Rappresenta le unità di capacità computazionale necessarie per elaborare un blocco di traffico[cite: 396].
2. [cite_start]**$s(l,n)$ (Dimensione dei Dati):** Rappresenta la dimensione dei dati generati dopo l'elaborazione del blocco $1$ e pronti per essere trasmessi[cite: 396].

### Il Disallineamento di Visione:
* **La prospettiva del Reviewer:** Si concentra solo su $\kappa(l,n)$. Sostiene che con batch size = 1, il costo computazionale per valutare la confidenza dell'uscita anticipata (l'overhead di calcolo aggiunto a $\kappa$) annulla il tempo risparmiato saltando i livelli successivi della rete.
* [cite_start]**La nostra prospettiva (MCM Reality):** Ci concentriamo su $s(l,n)$[cite: 79, 197]. [cite_start]In un'architettura a chiplet distribuita, il vero collo di bottiglia è lo spostamento dei dati sulla rete interna del pacchetto (NoP)[cite: 46, 124, 200]. [cite_start]Se un Early Exit ha successo, **la dimensione dei dati intermedi $s(l,n)$ da trasmettere al chiplet successivo crolla a zero**[cite: 78]. Il risparmio energetico e di latenza derivante dal non dover muovere enormi matrici di attivazione lungo la NoP compensa ampiamente il micro-overhead computazionale aggiunto a $\kappa(l,n)$, anche a batch size = 1.

---

## 3. L'Esperimento di Calibrazione (RTX 5090 Mobile)

Per dimostrare l'impatto di questo "attrito del silicio" e calibrare le formule teoriche del simulatore, è stato eseguito un micro-benchmark su una GPU **NVIDIA GeForce RTX 5090 Laptop** (Architettura Blackwell, banda teorica nominale di **896.0 GB/s** e performance di picco FP32 stimata a **33.52 TFLOPS**).

Il test ha previsto l'esecuzione isolata di una moltiplicazione di matrici massiccia ($16384 \times 16384$ in Float32, pari a un payload di **1024.00 MB** e una complessità di **8.7961 TFLOPs**).

### Risultati del Benchmark:

| Metrica | Calcolo Ideale (Modello Analitico) | Misura Reale (Hardware Laptop) | Fattore di Scala ($k$) |
| :--- | :--- | :--- | :--- |
| **Latenza Compute** | **262.4133 ms** | **426.8219 ms** | **1.63x** |
| **Latenza Comm/Setup** | **1.1161 ms** | **4.2359 ms** | **3.80x** |
| **Latenza Totale** | **263.5293 ms** | **431.0578 ms** | **1.64x** |

---

## 4. Analisi e Spiegazione dei Fattori di Scala ($k$)

I tre moltiplicatori estratti rappresentano i coefficienti di correzione empirica da applicare alle costanti del simulatore:

* **$k_{compute} = 1.63x$**: Indica che la GPU reale impiega il 63% di tempo in più rispetto al calcolo matematico perfetto. Evidenzia che i TFLOPS nominali di marketing non tengono conto dei limiti termici del laptop e dell'efficienza dei kernel CUDA (la scheda viaggia a circa il 61% delle sue capacità teoriche massime).
* **$k_{comm} = 3.80x$**: Questo valore rappresenta la risposta diretta al Reviewer 1. Dimostra empiricamente che l'allocazione di memoria e l'invocazione dei driver hardware richiedono quasi **4 volte il tempo** rispetto alla pura velocità fisica della banda passante. Questo scarto è l'interconnessione reale che MAESTRO non vede.
* **Perché il totale è $1.64x$ (e non la media di 1.63 e 3.80)?**
  Il fattore globale è una **media pesata sui millisecondi**. In questo specifico test, il carico è stato fortemente sbilanciato sul calcolo (*compute-bound*): la GPU ha passato il 99% del tempo totale a fare calcoli duri (426.82 ms) e solo l'1% del tempo a preparare la memoria (4.23 ms). Di conseguenza, il fattore finale (1.64x) è stato quasi interamente guidato dall'efficienza di calcolo (1.63x).

Nel contesto del paper, tuttavia, la rete distribuita su MCM è fortemente vincolata alla comunicazione (*communication-bound*)[cite: 46, 124, 197]. I saltelli continui sulla NoP fanno sì che il fattore di penalità della memoria di **3.80x** si sommi a ogni livello della rete, diventando il vero fattore dominante delle prestazioni generali.

---

## 5. Strategia per i Prossimi Passaggi e Call con Carla

Per allineare il team in vista della call e preparare la risottomissione del testo:

1. **Rassicurare Carla sulla tenuta matematica:** Chiarire che non verrà modificata la formulazione deterministica del problema di ottimizzazione in Eq. 10 (preservando intatte le dimostrazioni di submodularità e programmazione geometrica)[cite: 86, 97]. La natura probabilistica degli EE verrà gestita aggiornando i parametri di input ($\kappa$ e $s$) attraverso i loro valori attesi ponderati sulla difficoltà del dataset *prima* dell'esecuzione dell'algoritmo.
2. **Ancoraggio su Jetson:** Replicare lo stesso script sul modulo Jetson di destinazione, sostituendo le costanti hardware con le specifiche teoriche della scheda embedded (TOPS e Bandwidth della memoria condivisa). Essendo un sistema SoC edge a risorse condivise, ci si aspetta che il fattore $k_{comm}$ risulti nettamente superiore a 3.80x a causa della forte contesa dei banchi di memoria.
3. **Iniezione dei fattori nel simulatore:** Utilizzare i coefficienti $k$ estratti dal Jetson come moltiplicatori correttivi per i parametri analitici di MAESTRO (es. riducendo la banda NoP effettiva o scalando le latenze fisse $\delta$)[cite: 218, 405]. Questo permetterà di presentare ai reviewer un modello solido e calibrato su dati hardware reali.