import os
import subprocess
import re
import sys

# ==============================================================================
# CONFIGURAZIONE INIZIALE
# ==============================================================================
SCRIPT_ORIGINALE = "satellite_osr.py"  # Assicurati che questo sia il nome esatto del tuo script satellite
SCRIPT_TEMPORANEO = "satellite_osr_ombra.py"
FILE_MATRICE_CSV = "Rolling_Window_OSR_S1A_1990_2009.csv"

# Finestra temporale dinamica: iniziamo dal 1990 per garantire almeno 30 osservazioni al VECM
ANNO_START_ROLLING = 1990
ANNO_MAX_FISSO = 2009

# 1. Pulizia preventiva
if os.path.exists(FILE_MATRICE_CSV):
    os.remove(FILE_MATRICE_CSV)

# 2. Lettura del file sorgente
if not os.path.exists(SCRIPT_ORIGINALE):
    print(f"[ERRORE] File '{SCRIPT_ORIGINALE}' non trovato nella directory corrente!")
    sys.exit(1)

with open(SCRIPT_ORIGINALE, "r", encoding="utf-8") as f:
    codice_sorgente_master = f.read()

print(f"[INFO] Script base caricato: {SCRIPT_ORIGINALE}")
print(f"[INFO] Avvio Rolling Window dal {ANNO_START_ROLLING} al {ANNO_MAX_FISSO}...")

# ==============================================================================
# CICLO DINAMICO ANNO PER ANNO
# ==============================================================================
for anno_target in range(ANNO_START_ROLLING, ANNO_MAX_FISSO + 1):
    print("\n" + "="*65)
    print(f"=== ESECUZIONE ROLLING WINDOW: [1960 - {anno_target}] ===")
    print("="*65)
    
    # 1. Iniezione della Barriera Temporale e Silenziatore Grafico
    barriera_temporale = f"""df = df_raw[['OSR', 'S1A']].copy()

# --- INIEZIONE DINAMICA: TRONCAMENTO DATASET ---
df = df[df.index <= {anno_target}].copy()

# Silenziatore grafico: impedisce a plt.show() di bloccare il ciclo
import matplotlib.pyplot as plt
plt.ioff()
def mock_show(*args, **kwargs): pass
plt.show = mock_show

# Patch per stabilità matriciale sui lag
import statsmodels.tsa.vector_ar.vecm as patched_vecm
import statsmodels.tsa.vector_ar.var_model as patched_var

_old_vecm_select = patched_vecm.select_order
def _safe_vecm_select(data, maxlags=4, *args, **kwargs):
    for lags in range(maxlags, 0, -1):
        try: return _old_vecm_select(data, maxlags=lags, *args, **kwargs)
        except: continue
    try: return _old_vecm_select(data, maxlags=1, *args, **kwargs)
    except: return _old_vecm_select(data, maxlags=0, *args, **kwargs)
patched_vecm.select_order = _safe_vecm_select
"""
    # Applichiamo il troncamento iniziale sostituendo la creazione di df originale
    codice_ombra = codice_sorgente_master.replace("df = df_raw[['OSR', 'S1A']].copy()", barriera_temporale, 1)
    
    # 2. Iniezione del Blocco di Esportazione CSV
    # Si aggancia alla fine del codice, raccogliendo le variabili generate nelle Fasi 2, 3, 4 e 5
    blocco_esportazione = f"""
# ==============================================================================
# FASE ESPORTAZIONE META-MODELLO
# ==============================================================================
import csv
import os
import numpy as np

file_csv_meta = "{FILE_MATRICE_CSV}"
file_exists_meta = os.path.isfile(file_csv_meta)

headers_meta = [
    'Anno_Target', 'Traccia_Johansen', 'Valore_Critico_95', 'Cointegrato',
    'Lag_Ottimale', 'Alpha_OSR', 'Alpha_S1A', 'p_value_Alpha_S1A', 'Granger_p_value_OSR_S1A',
    'IRF_Impatto_t1', 'IRF_Picco_Max'
]

# Protezione in caso di fallimento dell'algoritmo per scarsità di gradi di libertà
try:
    tr_j = tr_stat
    vc_95 = crit_val_95
    is_coint = "SI" if tr_j > vc_95 else "NO"
    
    lag_opt = k_ar_diff_ottimale + 1
    a_osr = alpha_OSR
    a_s1a = alpha_S1A
    pval_a_s1a = alpha_pvals[1] if 'alpha_pvals' in locals() else np.nan
    
    # Estrazione p-value Granger ultra-sicura
    if 'pvals_osr_su_s1a' in locals():
        pvals_array = np.atleast_1d(pvals_osr_su_s1a)
        if len(pvals_array) > 0:
            granger_pval = np.min(pvals_array) # Prendiamo il lag più significativo
        else:
            granger_pval = np.nan
    else:
        granger_pval = np.nan
        
    irf_t1 = istantaneo
    irf_picco = picco_max

except Exception as e:
    tr_j, vc_95, is_coint, lag_opt, a_osr, a_s1a, pval_a_s1a, granger_pval, irf_t1, irf_picco = [np.nan]*10

row_meta = [
    {anno_target}, tr_j, vc_95, is_coint, lag_opt, a_osr, a_s1a, pval_a_s1a, granger_pval, irf_t1, irf_picco
]

# Formattazione per Excel Italiano (Punto -> Virgola)
row_formattata = [str(x).replace('.', ',') if isinstance(x, float) else x for x in row_meta]

with open(file_csv_meta, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=';')
    if not file_exists_meta:
        writer.writerow(headers_meta)
    writer.writerow(row_formattata)

# Generazione del cruscotto a schermo
granger_str = f"{{granger_pval:.4e}}" if not np.isnan(granger_pval) else "N/A (k=0)"

print(f"\\n  > CRUSCOTTO META-VALIDAZIONE PER IL {anno_target} <")
print(f"  > Cointegrazione (Traccia)     : {{tr_j:.2f}} (Soglia: {{vc_95:.2f}}) -> {{is_coint}}")
print(f"  > Lag Ottimale (VECM)          : {{lag_opt}}")
print(f"  > p-value Alpha S1A            : {{pval_a_s1a:.4e}}")
print(f"  > p-value Granger (OSR->S1A)   : {{granger_str}}")
print(f"  > Picco Risposta S1A (IRF)     : {{irf_picco:.4f}}")
"""
    codice_ombra += blocco_esportazione

    # 3. Scrittura temporanea ed Esecuzione
    with open(SCRIPT_TEMPORANEO, "w", encoding="utf-8") as f:
        f.write(codice_ombra)
        
    processo = subprocess.run(
        ["python", SCRIPT_TEMPORANEO], 
        capture_output=True, text=True, encoding="utf-8", errors="ignore"
    )
    
    output_log = processo.stdout
    output_errori = processo.stderr
    
    if "Traceback" in output_errori:
        print(" [!] Anomalia rilevata nel sub-processo:")
        print("\n".join([line for line in output_errori.split('\n') if "ValueWarning" not in line and line.strip()]))
    
    if "> CRUSCOTTO META-VALIDAZIONE" in output_log:
        cruscotto = output_log[output_log.find("> CRUSCOTTO META-VALIDAZIONE"):]
        print(cruscotto.strip())
    else:
        print(" [?] Cruscotto non generato. Controllo log...")
    
    if os.path.exists(SCRIPT_TEMPORANEO):
        os.remove(SCRIPT_TEMPORANEO)
        
    print("-" * 65)
    input(f"Premere [INVIO] per avanzare alla simulazione dell'anno {anno_target + 1}...")

print("\n" + "="*65)
print(f"[OK] META-VALIDAZIONE COMPLETATA. CSV salvato: '{FILE_MATRICE_CSV}'")
print("="*65)