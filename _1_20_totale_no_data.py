import csv
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import statsmodels.api as sm
import sys
import warnings
from scipy.optimize import minimize
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM, select_order
from statsmodels.tsa.stattools import adfuller

# === INIZIALIZZAZIONE AMBIENTE E LOGGER ===
nazione = input("Inserisci la sigla della Nazione (es. IT, DE, US, UK, KR): ").strip().upper()
if not nazione:
    nazione = "XX"

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Il file di log prenderà il nome della Nazione
sys.stdout = Logger(f"report_completo_{nazione}.txt")
print(f"\n>>> AVVIO PROTOCOLLO V2.0 PER: {nazione} <<<")

warnings.filterwarnings("ignore")

print("="*40)
print("INIZIALIZZAZIONE FASE ZERO: IMPORTAZIONE RAW E TRASFORMAZIONE MATRICIALE")
print("="*40)

# 1 & 2. Blocco dati raw orizzontale (simulazione copia-incolla Excel italiano)
# Struttura: prima cella nome variabile, a seguire i valori. Separatore decimale: virgola.
dati_raw = """Anno
C
L
N
U
"""

# 3. Lettura raw, trasposizione e conversione decimali
# Il separatore r'\s+' legge correttamente sia i tab (nativi del copia-incolla Excel) che gli spazi
df_t = pd.read_csv(io.StringIO(dati_raw), sep=r'\s+', header=None, index_col=0)

# Trasponiamo: le variabili tornano colonne, il tempo scorre sulle righe
df = df_t.T

# Pulizia dell'indice vettoriale e rimozione del nome colonne
df.columns.name = None

# 1. Prima convertiamo TUTTE le colonne (inclusa 'Anno') per pulire stringhe/virgole e renderle numeriche
for col in df.columns:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

# 2. Ora che è un numero pulito, castiamo 'Anno' a intero puro (senza decimali)
df['Anno'] = df['Anno'].astype(int)

# 3. Infine, lo impostiamo come spina dorsale (indice) del DataFrame
df.set_index('Anno', inplace=True)

# 4 & 5. Calcolo variabili derivate 
df['LC'] = df['L'] / df['C'] # 1.0104543 Fattore moltiplicativo per attribuire la spesa assistenziale
df['CL'] = df['C'] / df['L'] # 0.9896538 Fattore moltiplicativo per detrarre la spesa assistenziale

# Estrazione limiti strutturali assoluti (Fase Zero Matematizzata)
M = df['LC'].max()
m_param = 1/ M 

# Generazione metriche 
df['S1A'] = (M - df['LC']) / (M + 1)
df['S1B'] = (df['CL'] - m_param) / (1 + m_param)
df['S2'] = (2.1 / df['N']) - 1
df['S3'] = 1 - (316.91 / df['U'])

print("\n[OK] Trasformazione vettoriale completata.")
print(f"Altitudine Media L/C: {df['LC'].mean():.4f}")
print(f"Variabili operative caricate: {list(df.columns)}")
print("-" *40)

# =============================================================================
# [ AGGIUNTA FASE ZERO: STATISTICHE DESCRITTIVE DELLE SERIE CHIAVE ]
# =============================================================================
print("\n" + "="*40)
print("STATISTICHE DESCRITTIVE DELLE SERIE CHIAVE")
print("="*40)
serie_chiave = ['CL', 'LC', 'S1A', 'S1B', 'N', 'S2']
stats_dict = {}

print(f"{'Serie':<5} | {'Max':<8} {'Anno':<5} | {'Min':<8} {'Anno':<5} | {'Media':<8} | {'Mediana':<8}")
print("-" * 40)

for s in serie_chiave:
    if s in df.columns:
        s_max = df[s].max()
        anno_max = df[s].idxmax()
        s_min = df[s].min()
        anno_min = df[s].idxmin()
        s_mean = df[s].mean()
        s_median = df[s].median()
        
        # Salviamo nel dizionario per passarle poi al database CSV finale
        stats_dict[s] = {
            'Max': s_max, 'Max_Anno': anno_max,
            'Min': s_min, 'Min_Anno': anno_min,
            'Media': s_mean, 'Mediana': s_median
        }
        
        print(f"{s:<5} | {s_max:<8.4f} {int(anno_max):<5} | {s_min:<8.4f} {int(anno_min):<5} | {s_mean:<8.4f} | {s_median:<8.4f}")
print("=" * 40)

print("\n" + "="*40)
print("FASE 1: VERIFICHE METODOLOGICHE (PASSAPORTO I(1) ")
print("="*40)

# ----------------------------------------------------------------------------------------
# Sotto-Fase 1: Filtri ADF (Augmented Dickey-Fuller)
# ----------------------------------------------------------------------------------------
print("\n[--- Sotto-Fase 1: Check qualifica I(1) ---]")

variabili_da_testare = ['S1A', 'S1B', 'S2', 'S3' ]

def adf_report(series, name):
    # Test sui Livelli
    res_livelli = adfuller(series.dropna(), autolag='AIC')
    pval_lvl = res_livelli[1]
    
    # Test sulle Differenze Prime
    res_diff = adfuller(series.diff().dropna(), autolag='AIC')
    pval_diff = res_diff[1]
    
    print(f"\nVariabile: {name}")
    print(f"  - Livelli (p-value):     {pval_lvl:.4f}", end=" -> ")
    
    # Check Livelli: Vogliamo p-value > 0.05 per confermare la non-stazionarietà
    pass_lvl = pval_lvl > 0.05
    print("[PASS] Radice unitaria presente" if pass_lvl else "[FAIL] Già stazionaria!")
    
    print(f"  - Diff. Prime (p-value): {pval_diff:.4f}", end=" -> ")
    
    # Check Differenze Prime: Vogliamo p-value < 0.05 per confermare l'integrazione I(1)
    pass_diff = pval_diff < 0.05
    print("[PASS] Stazionaria in diff 1a" if pass_diff else "[FAIL] Radice unitaria residua")
    
    if pass_lvl and pass_diff:
        print(f"  => ESITO: {name} è I(1) .")
    else:
        print(f"  => ESITO: Attenzione, {name} presenta anomalie di integrazione.")

for var in variabili_da_testare:
    adf_report(df[var], var)



print("\n" + "="*40)
print("FASE 2 E FASE 3: SELETTORE INTELLIGENTE ED EPOCA D'ORO")
print("="*40)

# ----------------------------------------------------------------------------------------
# 1. Motore Statistica F (Invariato)
# ----------------------------------------------------------------------------------------
def calcola_chow_F(y, x, split_idx):
    x_with_const = sm.add_constant(x)
    mod_tot = sm.OLS(y, x_with_const).fit()
    ssr_tot = mod_tot.ssr
    y1, x1 = y.iloc[:split_idx], x_with_const.iloc[:split_idx]
    ssr1 = sm.OLS(y1, x1).fit().ssr
    y2, x2 = y.iloc[split_idx:], x_with_const.iloc[split_idx:]
    ssr2 = sm.OLS(y2, x2).fit().ssr
    k = x_with_const.shape[1]
    N = len(y)
    denominatore = (ssr1 + ssr2) / (N - 2 * k)
    if denominatore == 0: return 0
    return ((ssr_tot - (ssr1 + ssr2)) / k) / denominatore

# ----------------------------------------------------------------------------------------
# 2. Identificazione Segmenti (Escape Route inclusa)
# ----------------------------------------------------------------------------------------
from scipy.stats import f as f_dist

def identifica_segmenti(var_name):
    y_full, x_full = df['S2'], df[var_name]
    N_tot = len(df)
    margine = 5
    k = 2
    
    max_f1, bp1_idx = -1, -1
    for i in range(margine, N_tot - margine):
        f_stat = calcola_chow_F(y_full, x_full, i)
        if f_stat > max_f1:
            max_f1 = f_stat
            bp1_idx = i
            
    # Clausola di Salvaguardia: Nessun taglio se il p-value di Chow non è significativo
    if bp1_idx == -1 or (1 - f_dist.cdf(max_f1, dfn=k, dfd=N_tot - 2 * k)) > 0.05:
        return [(0, N_tot)]
        
    max_f2, bp2_relative_idx = -1, -1
    y_sub, x_sub = y_full.iloc[bp1_idx:].reset_index(drop=True), x_full.iloc[bp1_idx:].reset_index(drop=True)
    N_sub = len(y_sub)
    
    for j in range(margine, N_sub - margine):
        f_stat2 = calcola_chow_F(y_sub, x_sub, j)
        if f_stat2 > max_f2:
            max_f2 = f_stat2
            bp2_relative_idx = j
            
    if bp2_relative_idx != -1 and (1 - f_dist.cdf(max_f2, dfn=k, dfd=N_sub - 2 * k)) <= 0.05:
        bp2_idx = bp1_idx + bp2_relative_idx
        return [(0, bp1_idx), (bp1_idx, bp2_idx), (bp2_idx, N_tot)]
    return [(0, bp1_idx), (bp1_idx, N_tot)]

# ----------------------------------------------------------------------------------------
# 3. Criterio 1 e 2: Filtro Sopravvivenza e Score
# ----------------------------------------------------------------------------------------
def calcola_candidati():
    candidati = []
    for var_name in ['S1A', 'S1B']:
        indici_segmenti = identifica_segmenti(var_name)
        for start_idx, end_idx in indici_segmenti:
            df_seg = df.iloc[start_idx:end_idx]
            lunghezza = len(df_seg)
            
            if lunghezza < 12: # Criterio 1
                continue
                
            corr = df_seg[var_name].corr(df_seg['S2'])
            score = abs(corr) * np.log(lunghezza) # Criterio 2
            
            anno_start = df.index[start_idx]
            anno_end = df.index[end_idx - 1] if end_idx < len(df) else df.index[-1]
            
            candidati.append({'var': var_name, 'start': anno_start, 'end': anno_end, 'L': lunghezza, 'corr': corr, 'score': score})
    
    # Ordinamento gerarchico per Score
    return sorted(candidati, key=lambda x: x['score'], reverse=True)

classifica = calcola_candidati()
print("\n[ Classifica Segmenti ]")
for i, c in enumerate(classifica):
    print(f"{i+1}. {c['var']} [{c['start']}-{c['end']}] | L={c['L']} | Corr={c['corr']:.4f} | Score={c['score']:.4f}")

# ----------------------------------------------------------------------------------------
# 4. Criterio 3: Setaccio Johansen a Cascata (Fase Elezione) - CON LAG DINAMICO
# ----------------------------------------------------------------------------------------
epoca_start, epoca_end, var_vincente_definitiva = None, None, None

for c in classifica:
    v_name, a_start, a_end = c['var'], c['start'], c['end']
    df_test = df.loc[a_start:a_end]
    
    # Calcolo dinamico del lag massimo per proteggere i gradi di libertà (max 4)
    maxlags_dinamico = min(4, max(1, len(df_test) // 5))
    
    print(f"\n[ Test Johansen: {v_name} ({a_start}-{a_end}) | maxlags={maxlags_dinamico} ]")
    
    # 3.1 Test Bivariato (S1, S2)
    df_biv = df_test[[v_name, 'S2']].dropna()
    p_opt = max(1, int(VAR(df_biv).select_order(maxlags=maxlags_dinamico).aic))
    joh_biv = coint_johansen(df_biv, det_order=1, k_ar_diff=max(1, p_opt - 1))
    
    if joh_biv.lr1[0] > joh_biv.cvt[0, 1]:
        print(" -> [PASS] Cointegrazione Bivariata attiva (r>0).")
        epoca_start, epoca_end, var_vincente_definitiva = a_start, a_end, v_name
        tipo_cointegrazione = "Bivariata"
        break
        
    # 3.2 Soccorso Trivariato (S1, S2, CO2)
    print(" -> [FAIL] Bivariato insufficiente. Soccorso Trivariato (CO2)...")
    df_triv = df_test[[v_name, 'S2', 'S3']].dropna()
    p_opt_t = max(1, int(VAR(df_triv).select_order(maxlags=maxlags_dinamico).aic))
    joh_triv = coint_johansen(df_triv, det_order=1, k_ar_diff=max(1, p_opt_t - 1))
    
    if joh_triv.lr1[0] > joh_triv.cvt[0, 1]:
        print(" -> [PASS] Cointegrazione Trivariata attiva (r>0).")
        epoca_start, epoca_end, var_vincente_definitiva = a_start, a_end, v_name
        tipo_cointegrazione = "Trivariata_CO2"
        break
    print(" -> [FAIL] Niente da fare. Avanti il prossimo.")

# Validazione globale e preparazione dataframes per Fase 4
print("\n" + "="*40)
print(f">>> L'elezione è conclusa. Variabile promossa: {var_vincente_definitiva} [{epoca_start}-{epoca_end}]")
df['S1'] = df[var_vincente_definitiva]
mask_stabile = (df.index >= epoca_start) & (df.index <= epoca_end)
df_stabile = df[mask_stabile].copy()
df_stabile['S1'] = df_stabile[var_vincente_definitiva]
print("="*40)

# ==============================================================================
# FASE 4: a ) STIMA DEL PARAMETRO STRUTTURALE (k); b )TEST FINESTRE ESPANSIVE (BACKTESTING OUT-OF-SAMPLE)
# ==============================================================

# ==============================================================================
# FASE 4a ) STIMA DEL PARAMETRO STRUTTURALE (k)
# ==============================================================

print("\n" + "="*40)
print("FASE 4: STIMA DEL PARAMETRO STRUTTURALE (k1)")
print("="*40)

# 1. Partizionamento del dataset in base all'Epoca Stabile
mask_stabile = (df.index >= epoca_start) & (df.index <= epoca_end)
df_stabile = df[mask_stabile]
df_instabile = df[~mask_stabile]

# Estrazione dei vettori operativi per la regressione
x_stabile = df_stabile['S1'].to_numpy()
y_stabile = df_stabile['S2'].to_numpy()

# 2. Definizione della Funzione Obiettivo (L1 Loss / Errore Assoluto)
# La minimizzazione della norma L1 garantisce maggiore robustezza contro eventuali outlier interni
def funzione_obiettivo_L1(k, x, y):
    y_stimato = k[0] * x
    errore_assoluto = np.abs(y - y_stimato)
    return np.sum(errore_assoluto)

# 3. Ottimizzazione e convergenza del parametro strutturale
# Inizializzazione del parametro (guess iniziale k=1.0)
k_iniziale = [1.0]

# Procedura di minimizzazione (utilizziamo Nelder-Mead per funzioni non derivabili come il valore assoluto)
risultato_ottimizzazione = minimize(funzione_obiettivo_L1, k_iniziale, args=(x_stabile, y_stabile), method='Nelder-Mead')
k_strutturale = risultato_ottimizzazione.x[0]

print(f"Stima parametro strutturale (k1) nell'Epoca Stabile: {k_strutturale:.4f}")

# ====================================================================================
# [ AGGIUNTA: CALCOLO DELLE AREE GEOMETRICHE (METRICA L1) SUL NUCLEO STABILE ]
# ====================================================================================
print("\n[--- ANALISI GEOMETRICA DELLE AREE (INDICI L1) ---]")

# 1. Calcolo del Rapporto 1 (Rapporto di Scala globale)
max_s1 = df_stabile['S1'].max()
max_s2 = df_stabile['S2'].max()
min_s1 = df_stabile['S1'].min()
min_s2 = df_stabile['S2'].min()

max_assoluto = max(max_s1, max_s2)
min_assoluto = min(min_s1, min_s2)

rapporto_1 = (max_assoluto - min_assoluto) / max_assoluto

# 2. Calcolo del Rapporto 2 (Area Approssimazione / Area Striscia Vera)
# Estrazione vettoriale del minimo anno per anno tra S1 ed S2
minimo_annuo_serie = np.minimum(df_stabile['S1'], df_stabile['S2'])

# Numeratore: spazio strutturale generato dal modello teorico k
area_approssimazione = np.sum(np.abs((k_strutturale * df_stabile['S1']) - minimo_annuo_serie))

# Denominatore: scostamento fisico e grezzo tra S1 ed S2
area_striscia_vera = np.sum(np.abs(df_stabile['S2'] - df_stabile['S1']))

rapporto_2 = area_approssimazione / area_striscia_vera

# 3. Calcolo del Rapporto 3 (Percentuale di occupazione del Bounding Box)
N_anni = len(df_stabile)
rapporto_3 = area_striscia_vera / ((max_assoluto - min_assoluto) * N_anni)

# ------------------------------------------------------------------------------------
# 4. Prodotto dei Rapporti (Rapporto 2 * Rapporto 3)
# Misura l'impronta strutturale dell'approssimazione rispetto all'area totale disponibile
# ------------------------------------------------------------------------------------
rapporto_prodotto = rapporto_2 * rapporto_3

# ------------------------------------------------------------------------------------
# 5. Indice di Sovrapposizione Geometrica (Intersezione su Unione - IoU)
# Valuta la sovrapposizione visiva esatta tra S2 e k*S1
# ------------------------------------------------------------------------------------
intersezione_curve = np.sum(np.minimum(df_stabile['S2'], k_strutturale * df_stabile['S1']))
unione_curve = np.sum(np.maximum(df_stabile['S2'], k_strutturale * df_stabile['S1']))
indice_sovrapposizione = intersezione_curve / unione_curve

# ------------------------------------------------------------------------------------
# 6. Baricentro dello Scostamento (Centro di Massa Temporale in L1)
# ------------------------------------------------------------------------------------
anni_vettore = df_stabile.index.values

# Baricentro della Striscia Vera (divario strutturale grezzo |S2 - S1|)
massa_striscia = np.abs(df_stabile['S2'] - df_stabile['S1'])
baricentro_vero = np.sum(anni_vettore * massa_striscia) / np.sum(massa_striscia)

# Baricentro dell'Approssimazione (errore del modello |S2 - k*S1|)
massa_modello = np.abs(df_stabile['S2'] - (k_strutturale * df_stabile['S1']))
baricentro_modello = np.sum(anni_vettore * massa_modello) / np.sum(massa_modello)

# Ritardo positivo in mesi (Anni di S2 - Anni di kS1) * 12
ritardo_mesi = (baricentro_vero - baricentro_modello) * 12

# ====================================================================================
# OUTPUT ESTESO
# ====================================================================================
print(f"1) Rapporto Bounding Box ([Max-Min] / Max): {rapporto_1:.4f}")
print(f"2) Rapporto Aree L1 (Approssimazione / Striscia Vera): {rapporto_2:.4f}")
print(f"3) Rapporto Aree L1 (Occupazione del Bounding Box): {rapporto_3:.4f}")
print(f"4) Prodotto Rapporti (Occupazione Approssimata su Bounding Box): {rapporto_prodotto:.4f}")
print(f"5) Indice di Sovrapposizione Geometrica (IoU): {indice_sovrapposizione:.4f}")
print(f"6a) Baricentro Striscia Vera: Anno {baricentro_vero:.2f}")
print(f"6b) Baricentro Modello k1*S1: Anno {baricentro_modello:.2f}")
print(f"6c) \"Ritardo\" di k1*S1 : {ritardo_mesi:.1f} mesi")
print("=" * 40)

# ====================================================================================
# [ AGGIUNTA: "SCUDO ACCADEMICO" - STATISTICA CLASSICA (NORMA L2 / OLS) ]
# ====================================================================================

print("\n[--- SCUDO ACCADEMICO: METRICHE CLASSICHE SUI MINIMI QUADRATI (OLS) ---]")

# 1. Preparazione delle variabili (Modello: S2 = k_ols * S1)
# Attenzione: Omettiamo intenzionalmente l'aggiunta di una costante (intercetta)
# per mantenere l'equazione strutturale pura (S2 proporzionale a S1) come nel calcolo L1.
X_ols = df_stabile['S1']
y_ols = df_stabile['S2']

# Esecuzione della regressione lineare OLS (Ordinary Least Squares)
modello_ols = sm.OLS(y_ols, X_ols)
risultati_ols = modello_ols.fit()

# 2. Estrazione dei 4 parametri di difesa classica
# Punto 1: Il parametro k calcolato con i quadrati (anziché con le aree)
k_ols = risultati_ols.params['S1']

# Punto 2: R^2 Classico (Varianza spiegata)
r2_classico = risultati_ols.rsquared

# Punto 3: RMSE (Root Mean Square Error - Radice dell'Errore Quadratico Medio)
# Calcolato estraendo i residui del modello OLS e applicando la formula L2
residui_ols = risultati_ols.resid
rmse = np.sqrt(np.mean(residui_ols**2))

# Punto 4: p-value del parametro k (Significatività statistica del moltiplicatore)
p_value_k = risultati_ols.pvalues['S1']

# ====================================================================================
# OUTPUT ESTESO CLASSICO
# ====================================================================================
print(f"1) Parametro 'k2' Classico (OLS): {k_ols:.4f} [parametro k1: {k_strutturale:.4f}]")
print(f"2) Indice R² (Varianza Spiegata): {r2_classico:.4f} (ovvero il {r2_classico*100:.2f}%)")
print(f"3) RMSE (Root Mean Sq. Error L2): {rmse:.4f}")
print(f"4) p-value del parametro k2: {p_value_k:.4e}", end="")
if p_value_k < 0.05:
    print(" -> [ALTAMENTE SIGNIFICATIVO]")
else:
    print(" -> [NON SIGNIFICATIVO]")
print("=" * 40)

# ==============================================================================
# FASE 4b: TEST FINESTRE ESPANSIVE (BACKTESTING OUT-OF-SAMPLE)
# ==============================================================================

print("\n" + "="*40)
print("TEST FINESTRE ESPANSIVE (TUTTI i quinquenni dell'Epoca Stabile):")

# Funzione di supporto per l'output in stile Excel italiano (virgola per i decimali)
def formatta_ita(valore, decimali=4):
    return f"{valore:.{decimali}f}".replace('.', ',')

# Estrazione degli anni per le etichette di stampa
anni_stabile = df_stabile.index.values

# Il test parte solo se abbiamo uno storico di almeno 15 anni nell'epoca stabile
n_stabile = len(df_stabile)
if n_stabile >= 15:
    train_size = 10  # Base di partenza: addestramento sui primi 10 anni
    
    while train_size <= n_stabile - 5:
        # Selezione dati di addestramento (storico passato)
        x_train, y_train = x_stabile[:train_size], y_stabile[:train_size]
        
        # Calcolo bendato del moltiplicatore (solo sui dati storici)
        c_iter = minimize(funzione_obiettivo_L1, x0=[1.0], args=(x_train, y_train), method='Nelder-Mead').x[0]
        
        # Definizione della finestra di proiezione (i 5 anni successivi)
        test_end = min(train_size + 5, n_stabile)
        anni_test = anni_stabile[train_size:test_end]
        x_test, y_test = x_stabile[train_size:test_end], y_stabile[train_size:test_end]
        
        # Calcolo delle previsioni e dell'errore
        y_proj = x_test * c_iter
        errore_iter = y_proj - y_test
        
        # Stampa a video formattata (punto e virgola per CSV/Excel)
        print(f"\n[Train {int(anni_stabile[0])}-{int(anni_stabile[train_size-1])} | Moltiplicatore k: {formatta_ita(c_iter)}]")
        print("Anno ; " + " ; ".join([str(int(a)) for a in anni_test]))
        print("Reale ; " + " ; ".join([formatta_ita(y, 2) for y in y_test]))
        print("Previsto ; " + " ; ".join([formatta_ita(y, 2) for y in y_proj]))
        print("Errore % ; " + " ; ".join([formatta_ita((e / r) * 100, 2) + "%" for e, r in zip(errore_iter, y_test)]))
        
        # Allargamento della finestra temporale per il ciclo successivo
        train_size += 5
else:
    print("Dataset dell'Epoca Stabile troppo corto (minimo 15 anni richiesti).")


# ==============================================================================
# FASE 5: DINAMICA VECM 
# ==============================================================================

print("\n" + "="*40)
print("FASE 5: DINAMICA VECM ")
print("="*40)

# ----------------------------------------------------------------------------------------
# 2. Ricerca del Lag Ottimale e Modello a Correzione d'Errore (VECM)
# ----------------------------------------------------------------------------------------

df_vecm = df_stabile[['S1', 'S2']].dropna() 

# Estrazione del ritardo temporale ottimale (criterio AIC)
lag_res_vecm = select_order(df_vecm, maxlags=5, deterministic="co")
p_opt = lag_res_vecm.aic

# Costrizione strutturale del ritardo inferiore
if p_opt == 0:
    p_opt = 1

# Stima del modello VECM
# N.B. In statsmodels, k_ar_diff corrisponde a (p - 1) dove p è l'ordine del VAR in livelli
k_ar_diff_ottimale = max(0, p_opt - 1)
modello_vecm = VECM(df_vecm, deterministic="co", k_ar_diff=k_ar_diff_ottimale, coint_rank=1)
risultati_vecm = modello_vecm.fit()

# Estrazione del coefficiente di correzione d'errore (alpha) per la variabile dipendente S2
# L'indice [0, 0] corrisponde alla prima equazione (S2) e al primo vettore di cointegrazione
alpha_S2 = risultati_vecm.alpha[0, 0]

# ----------------------------------------------------------------------------------------
# 4. Report Accademico Conclusivo
# ----------------------------------------------------------------------------------------
print("\n[ REPORT ANALISI VECM ]")
print(f"  - Ritardo temporale ottimale bivariato (p_opt) : {p_opt}")
print(f"  - Coefficiente di correzione d'errore (alpha)  : {alpha_S2:.4f}")

print("-" *40)

# -------------------------------------------
# "FASE 6: DIAGNOSTICA STRUTTURALE DEI RESIDUI"
# -------------------------------------------

print("\n" + "="*40)
print("FASE 6: DIAGNOSTICA STRUTTURALE DEI RESIDUI")
print("="*40)

# 1. Test di Normalità dei Residui (Jarque-Bera)
print("\n[ Test di Normalità dei Residui (Jarque-Bera) ]")
try:
    norm_test = risultati_vecm.test_normality()
    print(norm_test.summary())
except Exception as e:
    print("Impossibile calcolare Jarque-Bera multivariato sui dati correnti.")

# 2. Test di Autocorrelazione dei Residui (Whiteness / Ljung-Box)
print("\n[ Test di Autocorrelazione dei Residui (Whiteness / Ljung-Box) ]")
try:
    auto_test = risultati_vecm.test_whiteness(nlags=5)
    print(auto_test.summary())
except Exception as e:
    print("Impossibile calcolare il test di Whiteness.")


print("\n" + "="*40)

# 3. Test di Eteroschedasticità (ARCH/Breusch-Pagan)
from statsmodels.stats.diagnostic import het_arch

print("\n[ Test di Eteroschedasticità dei Residui (Test ARCH) ]")
try:
    # Estraiamo i residui della prima equazione (S2)
    resid_S2 = risultati_vecm.resid[:, 0]
    
    # Eseguiamo l'ARCH test (H0: I residui sono omoschedastici / varianza costante)
    arch_test_res = het_arch(resid_S2, nlags=5)
    
    pval_arch = arch_test_res[1]
    print(f"  - Statistica LM (Lagrange Multiplier) : {arch_test_res[0]:.4f}")
    print(f"  - Valore-p (p-value)                  : {pval_arch:.4f}")
    
    if pval_arch > 0.05:
        print("  -> [PASS] Omoschedasticità confermata: la varianza degli errori è costante.")
    else:
        print("  -> [FAIL] Eteroschedasticità rilevata.")
except Exception as e:
    print(f"Impossibile calcolare il test ARCH sui dati correnti: {e}")
    
# -------------------------------------------
# ====================================================================================
# FASE 7: VALIDAZIONE E CONO DI ROTTURA POST-2018 (CAMPIONAMENTO EPOCA D'ORO 1978-2018)
# ====================================================================================

print("\n" + "="*40)
print("FASE 7: VALIDAZIONE POST-2018 (DOPPIO DIAGNOSTICO S1 E S2)")
print("="*40)

df_completo = df.copy()

# 1. Recupero brutale e sicuro dell'asse dei tempi
if 'Anno' in df_completo.columns:
    df_completo = df_completo.set_index('Anno')
elif 'Year' in df_completo.columns:
    df_completo = df_completo.set_index('Year')

try:
    if isinstance(df_completo.index, pd.DatetimeIndex):
        df_completo.index = df_completo.index.year
    else:
        df_completo.index = df_completo.index.astype(int)
    print(f"[ Info ] Indice ripristinato. Orizzonte reale: {df_completo.index.min()} - {df_completo.index.max()}")
except Exception as e:
    print(f"[ Attenzione ] Errore cast indice: {e}")

col_S1 = 'S1' if 'S1' in df_completo.columns else ('S1A' if 'S1A' in df_completo.columns else None)

if col_S1 and 'S2' in df_completo.columns:
    df_analisi = df_completo[[col_S1, 'S2']].dropna()
    df_analisi.columns = ['S1', 'S2']
    
    # =====================================================================
    # MODIFICA METODOLOGICA: RECINTO DINAMICO RELATIVO
    # =====================================================================
    anno_max = df_analisi.index.max()
    # Il vertice di rottura si posiziona a 7 anni dalla fine disponibile, 
    # oppure alla fine naturale dell'Epoca stabile se si è frantumata prima.
    vertice_rottura = int(min(epoca_end, anno_max - 7))
    
    # Circoscriviamo l'addestramento alla VERA Epoca Stabile (da epoca_start a vertice_rottura)
    df_train = df_analisi[(df_analisi.index >= epoca_start) & (df_analisi.index <= vertice_rottura)]
    df_test = df_analisi[df_analisi.index >= vertice_rottura]
    
    print(f"[ Conteggio ] Righe Storiche Epoca d'Oro ({epoca_start}-{vertice_rottura}): {len(df_train)} | Righe Validazione (>={vertice_rottura}): {len(df_test)}")
    
    if not df_test.empty and len(df_train) > 10:
        print("[ Simulazione VECM Incondizionata (Cieca) in corso... ]\n")
        
        # Protezione dinamica sui ritardi per dataset corti
        lag_order_train = select_order(df_train, maxlags=min(4, len(df_train)//5), deterministic="co")
        opt_lag_train = max(1, lag_order_train.aic)
        
        vecm_train = VECM(df_train, deterministic="co", k_ar_diff=opt_lag_train, coint_rank=1).fit()
        
        # Previsione a partire dal vertice dinamico
        steps = len(df_test) - 1
        forecast, lower, upper = vecm_train.predict(steps=steps, alpha=0.05)
        
        idx_s1 = 0
        idx_s2 = 1
        
        # Estrazione previsioni future
        s1_fc_fut = forecast[:, idx_s1]
        s2_fc_fut = forecast[:, idx_s2]
        
        # Estrazione vertici reali per ancoraggio geometrico continuo
        vertice_s1_base = df_train.loc[vertice_rottura, 'S1']
        vertice_s2_base = df_train.loc[vertice_rottura, 'S2']
        
        # Costruzione array definitivi
        s1_forecast = np.insert(s1_fc_fut, 0, vertice_s1_base)
        s1_lower = np.insert(lower[:, idx_s1], 0, vertice_s1_base)
        s1_upper = np.insert(upper[:, idx_s1], 0, vertice_s1_base)
        
        s2_forecast = np.insert(s2_fc_fut, 0, vertice_s2_base)
        s2_lower = np.insert(lower[:, idx_s2], 0, vertice_s2_base)
        s2_upper = np.insert(upper[:, idx_s2], 0, vertice_s2_base)
        
        # =====================================================================
        # OUTPUT GRAFICO E TESTUALE
        # =====================================================================
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        df_storico_visivo = df_analisi[df_analisi.index <= vertice_rottura]
        
        ax1.plot(df_storico_visivo.index, df_storico_visivo['S1'], label=f'S1 Storico Integrale (fino al {vertice_rottura})', color='blue')
        ax1.plot(df_test.index, df_test['S1'], label=f'S1 Reale ({vertice_rottura}+)', color='red', marker='o')
        ax1.plot(df_test.index, s1_forecast, label='S1 Previsto (Inerziale)', color='green', linestyle='--')
        ax1.fill_between(df_test.index, s1_lower, s1_upper, color='green', alpha=0.2, label='Cono 95%')
        ax1.set_title("1. L'Innesco: Deviazione del substrato economico (S1)")
        ax1.set_ylabel("S1")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(df_storico_visivo.index, df_storico_visivo['S2'], label=f'S2 Storico Integrale (fino al {vertice_rottura})', color='blue')
        ax2.plot(df_test.index, df_test['S2'], label=f'S2 Reale ({vertice_rottura}+)', color='red', marker='o')
        ax2.plot(df_test.index, s2_forecast, label='S2 Previsto (Inerziale)', color='green', linestyle='--')
        ax2.fill_between(df_test.index, s2_lower, s2_upper, color='green', alpha=0.2, label='Cono 95%')
        ax2.set_title("2. L'Impatto: Conseguente deviazione (S2)")
        ax2.set_xlabel("Anno")
        ax2.set_ylabel("S2")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax1.set_xlim(int(df_analisi.index.min()), int(df_analisi.index.max()))
        ax2.set_xlim(int(df_analisi.index.min()), int(df_analisi.index.max()))
        
        plt.tight_layout()
        plt.savefig(f"doppio_cono_confidenza_{nazione}.png", dpi=300)
        plt.close(fig)
        
        print("[ LOG DIAGNOSTICO: SCARTI STRUTTURALI REALI VS MODELLO INERZIALE ]")
        print(f"{'Anno':<6} | {'S1 Reale':<10} | {'S1 Prev':<10} | {'Δ S1 (Err)':<12} || {'S2 Reale':<10} | {'S2 Prev':<10} | {'Δ S2 (Err)':<12}")
        print("-" * 40)
        
        for i, anno in enumerate(df_test.index):
            s1_real, s1_prev = df_test.loc[anno, 'S1'], s1_forecast[i]
            s2_real, s2_prev = df_test.loc[anno, 'S2'], s2_forecast[i]
            delta_s1, delta_s2 = s1_real - s1_prev, s2_real - s2_prev
            
            if anno == vertice_rottura:
                print(f"{anno:<6} | {s1_real:<10.4f} | {s1_prev:<10.4f} | {delta_s1:<12.4f} || {s2_real:<10.4f} | {s2_prev:<10.4f} | {delta_s2:<12.4f}  <-- [VERTICE]")
            else:
                print(f"{anno:<6} | {s1_real:<10.4f} | {s1_prev:<10.4f} | {delta_s1:<12.4f} || {s2_real:<10.4f} | {s2_prev:<10.4f} | {delta_s2:<12.4f}")
        
        print("=" * 40)
        print("  -> FATTO: Grafico 'doppio_cono_confidenza.png' salvato e matrice scarti calcolata.")
    else:
        print("  -> ERRORE: Dati validazione insufficienti.")
        print("\n" + "-" *40)

# ====================================================================================
# FASE 8: DINAMICA DELL'ISTERESI E STABILITÀ STRUTTURALE (ESTRAZIONE QUANTITATIVA)
# ====================================================================================
print("\nFASE 8: DINAMICA DELL'ISTERESI E STABILITÀ STRUTTURALE")
print("="*85)

# Forzatura indici geometrici fissi per allineamento polarità VECM
idx_s1 = 0
idx_s2 = 1

# [A] DIAGNOSTICA CUSUM NUMERICA
# Estraiamo i residui della forma strutturale lineare L1 su df_train
residui_strutturali = df_train['S2'] - (k_strutturale * df_train['S1'])
cusum_stat = np.cumsum(residui_strutturali) / np.std(residui_strutturali)
max_cusum = np.max(np.abs(cusum_stat))

# Soglia critica standard al 95% per il test CUSUM (approssimazione analitica su N osservazioni)
N_osservazioni = len(df_train)
soglia_critica_cusum = 0.9479 * np.sqrt(N_osservazioni)
valore_normalizzato_cusum = max_cusum / soglia_critica_cusum

print("[ Test CUSUM di Stabilità dei Parametri ]")
print(f"  - Statistica Max CUSUM Rilevata  : {max_cusum:.4f}")
print(f"  - Soglia Critica di Tolleranza   : {soglia_critica_cusum:.4f}")
if max_cusum < soglia_critica_cusum:
    print("  -> [PASS] Stabilità strutturale confermata nell'Epoca d'Oro.")
else:
    print("  -> [AVVISO] Rilevata instabilità o derive strutturali nei parametri storici.")

# [B] ANALISI QUANTITATIVA IRF (S1 -> S2)
# Estraiamo i coefficienti dinamici dal VECM stimato nella fase precedente (vecm_train)
irf_obj = vecm_train.irf(periods=10)
# irf_obj.irfs ha dimensione (periods+1, variabili, impulsi)
# Identifichiamo la risposta di S2 a un impulso di S1
irf_scambio = irf_obj.irfs[:, idx_s2, idx_s1]

impatto_istantaneo = np.abs(irf_scambio[1])  # Risposta al tempo t+1 (primo anno utile)
anno_picco = np.argmax(np.abs(irf_scambio))
valore_picco = np.abs(irf_scambio[anno_picco])

# Calcolo empirico dell'Half-Life (Tempo di dimezzamento dello shock rispetto al picco)
coda_post_picco = np.abs(irf_scambio[anno_picco:])
meta_valore_picco = np.abs(valore_picco) / 2
anno_dimezzamento = np.where(coda_post_picco <= meta_valore_picco)[0]
half_life = (anno_dimezzamento[0] + anno_picco) if len(anno_dimezzamento) > 0 else 10

print("\n[ Analisi IRF Strutturale: Impulso S1  -> Risposta S2 ]")
print(f"  - Moltiplicatore Dinamico Istantaneo (Anno t+1) : {impatto_istantaneo:.4f}")
print(f"  - Picco Massimo della Risposta d'Impatto         : {valore_picco:.4f} (Raggiunto all'Anno t+{anno_picco})")
print(f"  - Persistenza del Trauma (Tempo di Half-Life)   : {half_life:.1f} anni per riassorbire il 50% dello shock")

# Generazione dei grafici fisici per l'archivio
fig_cusum, ax_cusum = plt.subplots(figsize=(10, 4))
ax_cusum.plot(df_train.index, cusum_stat, color='purple', label='Linea CUSUM empirica')
ax_cusum.axhline(soglia_critica_cusum, color='red', linestyle='--', label='Banda Superiore 95%')
ax_cusum.axhline(-soglia_critica_cusum, color='red', linestyle='--', label='Banda Inferiore 95%')
ax_cusum.set_title("Test CUSUM sui residui strutturali storici")
ax_cusum.legend()
plt.tight_layout()
plt.savefig(f"cusum_{nazione}.png", dpi=300)
plt.close(fig_cusum)

fig_irf = irf_obj.plot(orth=False); plt.gcf().set_size_inches(10, 6)
plt.savefig(f"irf_{nazione}.png", dpi=300)
plt.close()

# ====================================================================================
# ====================================================================================
# FASE 9: DINAMICA EVOLUTIVA (ROLLING PEARSON E INDICI QUANTITATIVI EWS)
# ====================================================================================
print("\nFASE 9: DINAMICA EVOLUTIVA (ROLLING PEARSON E EWS)")
print("="*85)

# [A] DIAGNOSTICA NUMERICA ROLLING PEARSON
# Calcoliamo la stabilità locale del legame a finestre mobili con scudo per serie corte
print("[ Analisi di Stabilità Locale (Rolling Pearson) ]")

if len(df_train) >= 10:
    rolling_10 = df_train['S2'].rolling(window=10).corr(df_train['S1']).dropna()
    if not rolling_10.empty:
        print(f"  - Finestra Mobile 10 Anni : Intervallo [{rolling_10.min():.4f} , {rolling_10.max():.4f}] | Varianza Interna: {rolling_10.var():.6f}")
    else:
        print("  - Finestra Mobile 10 Anni : [INFO] Dati validi insufficienti.")
else:
    print("  - Finestra Mobile 10 Anni : [INFO] Serie troppo corta (< 10 anni).")

if len(df_train) >= 15:
    rolling_15 = df_train['S2'].rolling(window=15).corr(df_train['S1']).dropna()
    if not rolling_15.empty:
        print(f"  - Finestra Mobile 15 Anni : Intervallo [{rolling_15.min():.4f} , {rolling_15.max():.4f}] | Varianza Interna: {rolling_15.var():.6f}")
    else:
        print("  - Finestra Mobile 15 Anni : [INFO] Dati validi insufficienti.")
else:
    print("  - Finestra Mobile 15 Anni : [INFO] Serie troppo corta (< 15 anni).")

# [B] SEGNALI DI ALLARME PRECOCE (EARLY WARNING SIGNALS) SULLA RESILIENZA DI S2
# Analizziamo gli ultimi 7 anni storici prima del punto di rottura dinamico
inizio_ews = vertice_rottura - 7
finestra_ews = df_train.loc[inizio_ews:vertice_rottura, 'S2']

print(f"\n[ Diagnostica EWS (Early Warning Signals) su Struttura S2 prima del collasso {vertice_rottura} ]")

# Scudo protettivo per AutoReg su dataset frammentati (es. UK)
try:
    # Calcolo della varianza mobile negli ultimi anni pre-rottura
    varianza_pre_collasso = finestra_ews.var()
    if inizio_ews > df_train.index.min():
        varianza_storica_base = df_train.loc[:inizio_ews-1, 'S2'].var()
        incremento_percentuale_varianza = ((varianza_pre_collasso - varianza_storica_base) / varianza_storica_base) * 100
    else:
        incremento_percentuale_varianza = 0

    # Calcolo del coefficiente di Autocorrelazione Lag-1 [AR(1)]
    res_ar = AutoReg(finestra_ews, lags=1, trend='c').fit()
    coefficiente_ar1 = res_ar.params.get('S2.L1', res_ar.params.iloc[-1])
    
    print(f"  - Autocorrelazione Seriale AR(1) Locale ({inizio_ews}-{vertice_rottura}) : {coefficiente_ar1:.4f}")
    
    if coefficiente_ar1 > 0.70:
        print("  -> [ALLERTA CRITICAL SLOWING DOWN]: S2 mostrava una perdita di resilienza interna (memoria cinetica) prima dello strappo.")
    else:
        print("  -> [INFO] Gli indici di tensione strutturale locale si mantengono entro i parametri di assorbimento del rumore.")

except Exception as e:
    print(f"  -> [INFO] Dati insufficienti o matrice troppo instabile per calcolare il Critical Slowing Down locale. (Errore: {type(e).__name__})")

# [C] ESPORTAZIONE GRAFICI (con protezione d'errore incorporata)
try:
    fig_roll, ax_roll = plt.subplots(figsize=(10, 4))
    if 'rolling_10' in locals() and not rolling_10.empty:
        ax_roll.plot(rolling_10.index, rolling_10, label='Pearson Rolling (10 anni)', color='orange')
    if 'rolling_15' in locals() and not rolling_15.empty:
        ax_roll.plot(rolling_15.index, rolling_15, label='Pearson Rolling (15 anni)', color='red')
    
    ax_roll.axhline(0.80, color='gray', linestyle=':', label='Soglia Robustezza Standard')
    ax_roll.set_title("Evoluzione locale della correlazione strutturale S1-S2")
    ax_roll.legend()
    plt.tight_layout()
    plt.savefig(f"rolling_pearson_{nazione}.png", dpi=300)
    plt.close(fig_roll)
except Exception:
    plt.close()

try:
    fig_ews, ax_ews = plt.subplots(figsize=(10, 4))
    ax_ews.plot(finestra_ews.index, finestra_ews, marker='o', color='darkred', label='S2 Finestra EWS')
    ax_ews.set_title(f"EWS: Dettaglio della perdita di resilienza di S2 (Pre-{vertice_rottura})")
    ax_ews.legend()
    plt.tight_layout()
    plt.savefig(f"ews_s2_{nazione}.png", dpi=300)
    plt.close(fig_ews)
except Exception:
    plt.close()

print("-" * 40)
print("  -> FATTO: Metriche numeriche calcolate (con scudi attivi) e grafici diagnostici esportati.")

print(">>> PIPELINE ANALITICA V2.0 COMPLETATA CON SUCCESSO <<<")
print("-" *40)

# ====================================================================================
# FASE 10: ESPORTAZIONE NEL DATABASE A STRASCICO (CSV)
# ====================================================================================
print("\n[ Salvataggio nel Database a Strascico... ]")
file_csv = "database_strascico.csv"
file_exists = os.path.isfile(file_csv)

# Definizione della struttura matrice (Colonne)
headers = ['Nazione', 'Alt_Media_LC', 'Epoca_Inizio', 'Epoca_Fine', 'Var_Guida', 'Tipo_Cointegrazione', 
           'k1_L1', 'k2_OLS', 'R2_OLS', 'IoU_Geom', 'Max_CUSUM', 'Picco_IRF', 'Half_Life', 'EWS_AR1']

# Aggiunta dinamica delle colonne statistiche (5 serie x 6 metriche)
for s in serie_chiave:
    headers.extend([f"{s}_Max", f"{s}_Max_Anno", f"{s}_Min", f"{s}_Min_Anno", f"{s}_Media", f"{s}_Mediana"])

# Estrazione sicura delle variabili calcolate durante l'esecuzione
row = [
    nazione,
    df['LC'].mean() if 'LC' in df.columns else np.nan,
    epoca_start,
    epoca_end,
    var_vincente_definitiva,
    locals().get('tipo_cointegrazione', 'ND'),
    locals().get('k_strutturale', np.nan),
    locals().get('k_ols', np.nan),
    locals().get('r2_classico', np.nan),
    locals().get('indice_sovrapposizione', np.nan),
    locals().get('max_cusum', np.nan),
    locals().get('valore_picco', np.nan),
    locals().get('half_life', np.nan),
    locals().get('coefficiente_ar1', np.nan)
]

# Estrazione dei dati dal dizionario delle statistiche
for s in serie_chiave:
    if s in stats_dict:
        d = stats_dict[s]
        row.extend([d['Max'], d['Max_Anno'], d['Min'], d['Min_Anno'], d['Media'], d['Mediana']])
    else:
        row.extend([np.nan] * 6)

# Formattazione "europea" opzionale: se serve la virgola decimale nel CSV per Excel, de-commenta la riga sotto
row_formattata = [str(x).replace('.', ',') if isinstance(x, float) else x for x in row]
# Altrimenti usa semplicemente: row_formattata = row

# Append al database a strascico (una riga per esecuzione)
with open(file_csv, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=';') # Usa il punto e virgola come separatore
    if not file_exists:
        writer.writerow(headers)
    writer.writerow(row_formattata)

print(f"  -> FATTO: Riga matrice per [{nazione}] aggiunta con successo a {file_csv}")
print("=" * 40)