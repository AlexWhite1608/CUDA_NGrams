import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Stile per i grafici ---
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12

SAVE_PATH = "../../plots/"
LOAD_PATH = "../../results/"

# --- Nomi dei file CSV ---
# Utilizzo i nomi dei file che hai fornito in precedenza
file_a = LOAD_PATH + "experiment_A_20251117_160037.csv"
file_b = LOAD_PATH + "experiment_B_20251117_161136.csv"

print(f"Caricamento file: {file_a}, {file_b}")

# --- Caricamento Dati ---
try:
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)
    print("File CSV caricati correttamente.")
except FileNotFoundError as e:
    print(f"Errore: Impossibile trovare i file CSV. Assicurati che siano nella stessa cartella.")
    print(e)
    # Termina lo script se i file non sono trovati
    exit()

# --- FIX: Standardizza i nomi degli algoritmi in MAIUSCOLO ---
# Questo risolve l'errore KeyError che hai riscontrato
df_a['algorithm'] = df_a['algorithm'].str.upper()
df_b['algorithm'] = df_b['algorithm'].str.upper()

# Sostituisci 'inf' con 'NaN' per una corretta visualizzazione
df_a['gpu_time_mean'] = df_a['gpu_time_mean'].replace(np.inf, np.nan)
df_b['gpu_time_mean'] = df_b['gpu_time_mean'].replace(np.inf, np.nan)
# Per i grafici di speedup, 0.0 è un valore corretto per i fallimenti, non serve sostituire

print("\n--- Inizio Generazione Grafici TEMPI ESECUZIONE (GPU Time) ---")

# --- Grafico 1: Esperimento A (Tempo vs. 'n') ---
print("Generazione Grafico 1 (Barre): exp_A_time_bars.png")
df_a_pivot = df_a.pivot_table(index='n', columns='algorithm', values='gpu_time_mean')
df_a_pivot = df_a_pivot[['V1', 'V2', 'B']]
ax = df_a_pivot.plot(kind='bar', figsize=(12, 7), rot=0)
plt.title("Experiment A: GPU Time vs. N-gram Complexity 'n' (10x Corpus)")
plt.xlabel("N-gram Size (n)")
plt.ylabel("Mean GPU Time (seconds)")
ax.legend(title="Algorithm")
plt.grid(True, axis='y')
plt.savefig(SAVE_PATH + "exp_A_time_bars.png")
plt.close()

# --- Grafico 2: Esperimento B (Tempo vs. 'N' per n=2) ---
print("Generazione Grafico 2 (Linee): exp_B_n2_time.png")
df_b_n2 = df_b[df_b['n'] == 2]
plt.figure()
for alg in ['V1', 'V2', 'B']:
    df_alg = df_b_n2[df_b_n2['algorithm'] == alg].sort_values('amplification_factor')
    plt.plot(df_alg['amplification_factor'], df_alg['gpu_time_mean'], marker='o', linestyle='-', label=f'Algorithm {alg}')
plt.title("Experiment B: GPU Time vs. Data Size 'N' (for n=2)")
plt.xlabel("Corpus Amplification Factor (N)")
plt.ylabel("Mean GPU Time (seconds)")
plt.legend()
plt.grid(True)
plt.savefig(SAVE_PATH + "exp_B_n2_time.png")
plt.close()

# --- Grafico 3: Esperimento B (Tempo vs. 'N' per n=3) ---
print("Generazione Grafico 3 (Linee): exp_B_n3_time.png")
df_b_n3 = df_b[df_b['n'] == 3]
plt.figure()
for alg in ['V1', 'V2', 'B']:
    df_alg = df_b_n3[df_b_n3['algorithm'] == alg].sort_values('amplification_factor')
    plt.plot(df_alg['amplification_factor'], df_alg['gpu_time_mean'], marker='o', linestyle='-', label=f'Algorithm {alg}')
plt.title("Experiment B: GPU Time vs. Data Size 'N' (for n=3)")
plt.xlabel("Corpus Amplification Factor (N)")
plt.ylabel("Mean GPU Time (seconds)")
plt.legend()
plt.grid(True)
plt.savefig(SAVE_PATH + "exp_B_n3_time.png")
plt.close()

print("--- Grafici TEMPI ESECUZIONE generati. ---")
print("\n--- Inizio Generazione Grafici SPEEDUP ---")

# --- Grafico 4: Esperimento A (Speedup vs. 'n') ---
print("Generazione Grafico 4 (Barre): exp_A_speedup_bars.png")
df_a_pivot_speedup = df_a.pivot_table(index='n', columns='algorithm', values='speedup_mean')
df_a_pivot_speedup = df_a_pivot_speedup[['V1', 'V2', 'B']]
ax = df_a_pivot_speedup.plot(kind='bar', figsize=(12, 7), rot=0)
plt.title("Experiment A: Speedup vs. N-gram Complexity 'n' (10x Corpus)")
plt.xlabel("N-gram Size (n)")
plt.ylabel("Mean Speedup (CPU/GPU)")
ax.legend(title="Algorithm")
plt.grid(True, axis='y')
plt.savefig(SAVE_PATH + "exp_A_speedup_bars.png")
plt.close()

# --- Grafico 5: Esperimento B (Speedup vs. 'N' per n=2) ---
print("Generazione Grafico 5 (Linee): exp_B_n2_speedup.png")
df_b_n2_speedup = df_b[df_b['n'] == 2]
plt.figure()
for alg in ['V1', 'V2', 'B']:
    df_alg = df_b_n2_speedup[df_b_n2_speedup['algorithm'] == alg].sort_values('amplification_factor')
    plt.plot(df_alg['amplification_factor'], df_alg['speedup_mean'], marker='o', linestyle='-', label=f'Algorithm {alg}')
plt.title("Experiment B: Speedup vs. Data Size 'N' (for n=2)")
plt.xlabel("Corpus Amplification Factor (N)")
plt.ylabel("Mean Speedup (CPU/GPU)")
plt.legend()
plt.grid(True)
plt.savefig(SAVE_PATH + "exp_B_n2_speedup.png")
plt.close()

# --- Grafico 6: Esperimento B (Speedup vs. 'N' per n=3) ---
print("Generazione Grafico 6 (Linee): exp_B_n3_speedup.png")
df_b_n3_speedup = df_b[df_b['n'] == 3]
plt.figure()
for alg in ['V1', 'V2', 'B']:
    df_alg = df_b_n3_speedup[df_b_n3_speedup['algorithm'] == alg].sort_values('amplification_factor')
    plt.plot(df_alg['amplification_factor'], df_alg['speedup_mean'], marker='o', linestyle='-', label=f'Algorithm {alg}')
plt.title("Experiment B: Speedup vs. Data Size 'N' (for n=3)")
plt.xlabel("Corpus Amplification Factor (N)")
plt.ylabel("Mean Speedup (CPU/GPU)")
plt.legend()
plt.grid(True)
plt.savefig(SAVE_PATH + "exp_B_n3_speedup.png")
plt.close()

print("--- Grafici SPEEDUP generati. ---")
print("\n--- Inizio Generazione Grafici TEMPI CPU (per Validazione) ---")

# --- Grafico 7: Esperimento B (CPU Time vs. 'N') ---
print("Generazione Grafico 7 (Linee): exp_B_cpu_time.png")
# Prendiamo i dati di V1 come riferimento, tanto i tempi CPU sono simili
df_b_n2_cpu = df_b[(df_b['n'] == 2) & (df_b['algorithm'] == 'V1')].sort_values('amplification_factor')
df_b_n3_cpu = df_b[(df_b['n'] == 3) & (df_b['algorithm'] == 'V1')].sort_values('amplification_factor')
plt.figure()
plt.plot(df_b_n2_cpu['amplification_factor'], df_b_n2_cpu['cpu_time_mean'], marker='o', linestyle='-', label='CPU Time (n=2)')
plt.plot(df_b_n3_cpu['amplification_factor'], df_b_n3_cpu['cpu_time_mean'], marker='x', linestyle='--', label='CPU Time (n=3)')
plt.title("CPU Time vs. Data Size 'N' (Baseline)")
plt.xlabel("Corpus Amplification Factor (N)")
plt.ylabel("Mean CPU Time (seconds)")
plt.legend()
plt.grid(True)
plt.savefig(SAVE_PATH + "exp_B_cpu_time.png")
plt.close()

print("--- Grafico CPU Time generato. ---")

print("\n\n--- Generazione Grafici Completata ---")
print("File PNG generati:")
print("- exp_A_time_bars.png")
print("- exp_B_n2_time.png")
print("- exp_B_n3_time.png")
print("- exp_A_speedup_bars.png")
print("- exp_B_n2_speedup.png")
print("- exp_B_n3_speedup.png")
print("- exp_B_cpu_time.png")