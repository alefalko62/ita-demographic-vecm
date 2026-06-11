import os
import subprocess
import re
import sys

SCRIPT_ORIGINALE = "20_totale_x_meta_v.py"
SCRIPT_TEMPORANEO = "20_totale_x_meta_v_ombra.py"
FILE_MATRICE_CSV = "Validazione_Strutturale_2006_2025.csv"

# 1. Pulizia preventiva (unica e definitiva)
if os.path.exists(FILE_MATRICE_CSV):
    os.remove(FILE_MATRICE_CSV)

# 2. Lettura UNICA e SICURA del file sorgente
if not os.path.exists(SCRIPT_ORIGINALE):
    print(f"[ERRORE] File '{SCRIPT_ORIGINALE}' non trovato!")
    sys.exit(1)

with open(SCRIPT_ORIGINALE, "r", encoding="utf-8") as f:
    codice_sorgente_master = f.read()

# 3. Estrazione intelligente dell'anno massimo
prima_riga = codice_sorgente_master.splitlines()[0]
match = re.search(r"#(\d{4})", prima_riga)
if match:
    anno_max = int(match.group(1))
else:
    # Se manca il commento, cerchiamo l'anno più recente in tutto il file
    anni = re.findall(r"\b(19\d{2}|20\d{2})\b", codice_sorgente_master)
    anno_max = int(max(anni))

print(f"[INFO] Dataset rilevato fino all'anno: {anno_max}")


# 3. Ciclo dinamico
for anno_target in range(2006, anno_max + 1):
    print("\n" + "="*60)
    print(f"=== ESECUZIONE SIMULAZIONE: [1960 - {anno_target}] ===")
    print("="*60)
    
    # 1. Resezione chirurgica della vecchia Fase 10 (Esportazione Strascico)
    parti_codice = re.split(r"(#\s*=+\s*FASE 10.*)", codice_sorgente_master, flags=re.IGNORECASE)
    codice_base = parti_codice[0]
    
    # 2. Iniezione della Barriera Temporale sull'Indice Pandas e Scudo Try-Except sui Lag
    barriera_temporale = f"""df.set_index('Anno', inplace=True)

# --- INIEZIONE DINAMICA: TRONCAMENTO DATASET ---
df = df[df.index <= {anno_target}].copy()

import statsmodels.tsa.vector_ar.vecm as patched_vecm
import statsmodels.tsa.vector_ar.var_model as patched_var

_old_var_select = patched_var.VAR.select_order
def _safe_var_select(self, maxlags=5, *args, **kwargs):
    for lags in range(maxlags, 0, -1):
        try:
            return _old_var_select(self, maxlags=lags, *args, **kwargs)
        except ValueError:
            continue
    try: return _old_var_select(self, maxlags=1, *args, **kwargs)
    except: return _old_var_select(self, maxlags=0, *args, **kwargs)
patched_var.VAR.select_order = _safe_var_select

_old_vecm_select = patched_vecm.select_order
def _safe_vecm_select(data, maxlags=5, *args, **kwargs):
    for lags in range(maxlags, 0, -1):
        try:
            return _old_vecm_select(data, maxlags=lags, *args, **kwargs)
        except ValueError:
            continue
    try: return _old_vecm_select(data, maxlags=1, *args, **kwargs)
    except: return _old_vecm_select(data, maxlags=0, *args, **kwargs)
patched_vecm.select_order = _safe_vecm_select
# ------------------------------------------------------------------------
"""
    # Applichiamo il troncamento iniziale sostituendo il set_index originale
    codice_ombra = codice_base.replace("df.set_index('Anno', inplace=True)", barriera_temporale, 1)
    
    # 3. Iniezione del Modulo Predittivo e Salvataggio Completo (Nuova Fase 10 Dinamica)
    blocco_esportazione = f"""
# ==============================================================================
# FASE 10 (MODIFICATA DAL METAMODELLO): PREVISIONE E ESPORTAZIONE DINAMICA
# ==============================================================================

try:
    # Sganciamo il VECM dal rumore primordiale: addestramento dinamico da epoca_start fino ad anno_target
    df_vecm_dinamico = df.loc[epoca_start:, ['S1', 'S2']].dropna()
    
    # Ristimiamo i lag ottimali sul dataset esteso
    lag_res_dinamico = select_order(df_vecm_dinamico, maxlags=5, deterministic="co")
    p_opt_dinamico = lag_res_dinamico.aic
    if p_opt_dinamico == 0:
        p_opt_dinamico = 1
        
    k_ar_diff_dinamico = max(0, p_opt_dinamico - 1)
    modello_vecm_dinamico = VECM(df_vecm_dinamico, deterministic="co", k_ar_diff=k_ar_diff_dinamico, coint_rank=1)
    risultati_vecm_dinamico = modello_vecm_dinamico.fit()
    
    alpha_S2_dinamico = risultati_vecm_dinamico.alpha[0, 0]
    
    # Previsione in avanti partendo esattamente dal fronte dell'anno_target corrente
    forecast_tuple = risultati_vecm_dinamico.predict(steps=5, alpha=0.05)
    forecast_array = forecast_tuple[0] if isinstance(forecast_tuple, tuple) else forecast_tuple
    
    idx_s2 = list(df_vecm_dinamico.columns).index('S2')
    f_1y = forecast_array[0, idx_s2]
    f_3y = forecast_array[2, idx_s2]  # <-- AGGIUNTA: Previsione a 3 anni (t+3)
    f_5y = forecast_array[4, idx_s2]
    
    # Allineiamo le variabili locali per l'esportazione nello strascico
    p_opt = p_opt_dinamico
    alpha_S2 = alpha_S2_dinamico
except Exception as e:
    f_1y = np.nan
    f_3y = np.nan
    f_5y = np.nan
    p_opt = np.nan
    alpha_S2 = np.nan

file_csv_meta = "{FILE_MATRICE_CSV}"
file_exists_meta = os.path.isfile(file_csv_meta)

# Integriamo le colonne dello strascico originario con quelle di forecast (ora a 1, 3 e 5 anni) e dinamica VECM
headers_meta = ['Anno_Simulato', 'Nazione', 'Alt_Media_LC', 'Epoca_Inizio', 'Epoca_Fine', 'Var_Guida', 'Tipo_Cointegrazione', 
           'Lag_VECM', 'Alpha_S2', 'k1_L1', 'k2_OLS', 'R2_OLS', 'IoU_Geom', 'Max_CUSUM', 'Picco_IRF', 'Half_Life', 'EWS_AR1', 'Forecast_S2_1Y', 'Forecast_S2_3Y', 'Forecast_S2_5Y']

for s in serie_chiave:
    headers_meta.extend([f"{{s}}_Max", f"{{s}}_Max_Anno", f"{{s}}_Min", f"{{s}}_Min_Anno", f"{{s}}_Media", f"{{s}}_Mediana"])

row_meta = [
    {anno_target},
    nazione,
    df['LC'].mean() if 'LC' in df.columns else np.nan,
    epoca_start,
    epoca_end,
    var_vincente_definitiva,
    locals().get('tipo_cointegrazione', 'ND'),
    p_opt,
    alpha_S2,
    locals().get('k_strutturale', np.nan),
    locals().get('k_ols', np.nan),
    locals().get('r2_classico', np.nan),
    locals().get('indice_sovrapposizione', np.nan),
    locals().get('max_cusum', np.nan),
    locals().get('valore_picco', np.nan),
    locals().get('half_life', np.nan),
    locals().get('coefficiente_ar1', np.nan),
    f_1y,
    f_3y,
    f_5y
]

for s in serie_chiave:
    if s in stats_dict:
        d = stats_dict[s]
        row_meta.extend([d['Max'], d['Max_Anno'], d['Min'], d['Min_Anno'], d['Media'], d['Mediana']])
    else:
        row_meta.extend([np.nan] * 6)

# Formattazione per Excel Italiano (Punto -> Virgola)
row_formattata = [str(x).replace('.', ',') if isinstance(x, float) else x for x in row_meta]

import csv
with open(file_csv_meta, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=';')
    if not file_exists_meta:
        writer.writerow(headers_meta)
    writer.writerow(row_formattata)

# Generazione del cruscotto aggiornato con le metriche sul fronte
print(f"  > CRUSCOTTO META-VALIDAZIONE PER IL {anno_target} <")
print(f"  > Epoca Stabile Rilevata       : {{epoca_start}}-{{epoca_end}}")
print(f"  > VECM Lag Ottimale (Dinamico) : {{p_opt}}")
print(f"  > Alpha S2 (Dinamico)          : {{alpha_S2:.6f}}")
if locals().get('r2_classico') is not None:
    print(f"  > R² Strutturale (OLS)         : {{locals().get('r2_classico')*100:.2f}}%")
print(f"  > Deviazione CUSUM             : {{locals().get('max_cusum', 'N/D')}}")
print(f"  > Previsione S2 (1 Anno)       : {{f_1y}}  [Proiezione per il {anno_target + 1}]")
print(f"  > Previsione S2 (3 Anni)       : {{f_3y}}  [Proiezione per il {anno_target + 3}]")
print(f"  > Previsione S2 (5 Anni)       : {{f_5y}}  [Proiezione per il {anno_target + 5}]")
"""
    codice_ombra += blocco_esportazione

    # 4. Scrittura temporanea ed Esecuzione del processo isolato
    with open(SCRIPT_TEMPORANEO, "w", encoding="utf-8") as f:
        f.write(codice_ombra)
        
    processo = subprocess.run(
        ["python", SCRIPT_TEMPORANEO], 
        capture_output=True, text=True, encoding="utf-8", errors="ignore"
    )
    
    output_log = processo.stdout
    output_errori = processo.stderr
    
    if "Traceback" in output_errori or "Traceback" in output_log:
        print(" [!] Anomalia critica rilevata nel sub-processo:")
        print("\n".join([line for line in output_errori.split('\n') if "ValueWarning" not in line and line.strip()]))
    
    # Estraiamo e mostriamo a schermo il cruscotto
    if "> CRUSCOTTO META-VALIDAZIONE" in output_log:
        cruscotto = output_log[output_log.find("> CRUSCOTTO META-VALIDAZIONE"):]
        print(cruscotto.strip())
    else:
        print(" [?] L'esecuzione si è conclusa ma non è stato generato il cruscotto. Controllo log...")
        print("\n".join(output_log.split("\n")[-10:]))
    
    if os.path.exists(SCRIPT_TEMPORANEO):
        os.remove(SCRIPT_TEMPORANEO)
        
    print("-" * 60)
    
    # Mantenuta la pausa interattiva richiesta
    input(f"Premere [INVIO] per avanzare alla simulazione dell'anno {anno_target + 1}...")

print("\n" + "="*60)
print(f"[OK] VALIDAZIONE COMPLETATA. Dati strutturali, descrittivi e previsionali esportati in: '{FILE_MATRICE_CSV}'")
print("="*60)