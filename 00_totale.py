import numpy as np
import pandas as pd
import io

print("="*40)
print("INIZIALIZZAZIONE FASE ZERO: IMPORTAZIONE RAW E TRASFORMAZIONE MATRICIALE")
print("="*40)

# 1 & 2. Blocco dati raw orizzontale (simulazione copia-incolla Excel italiano)
# Struttura: prima cella nome variabile, a seguire i valori. Separatore decimale: virgola.
dati_raw = """Anno 		1960	1961	1962	1963	1964	1965	1966	1967	1968	1969	1970	1971	1972	1973	1974	1975	1976	1977	1978	1979	1980	1981	1982	1983	1984	1985	1986	1987	1988	1989	1990	1991	1992	1993	1994	1995	1996	1997	1998	1999	2000	2001	2002	2003	2004	2005	2006	2007	2008	2009	2010	2011	2012	2013	2014	2015	2016	2017	2018	2019	2020	2021	2022	2023	2024	2025
C 		7,3565	8,14054	8,9144	9,63813	10,38489	11,38577	12,54929	13,61251	14,88377	16,60024	17,91826	18,64968	20,34152	24,86974	32,17664	36,58798	46,473	55,61329	66,90408	83,84374	105,54284	125,46556	148,76966	173,15082	203,07352	229,02483	258,31916	280,91492	310,06073	341,03729	366,80131	395,37515	415,64639	427,27694	464,72954	513,8847	541,3639	557,0593	566,9051	586,0251	629,7106	667,5879	685,3494	707,0019	735,5773	743,944	763,5659	794,8201	805,411	764,0703	777,5571	801,653	773,6578	775,473	782,0098	792,3852	827,5922	839,7632	850,33	855,6919	797,0807	882,7508	982,957	1067,9487	1062,5713	1076,9
L 		5,17009	5,74665	6,68624	8,1347	9,06117	9,6073	10,27461	11,36476	12,35734	13,61589	15,92158	18,3014	20,39236	24,51054	30,63586	36,94576	45,38532	55,35193	64,11212	77,74364	95,69352	116,3686	135,52988	155,69935	173,68246	193,96479	209,58026	227,33493	249,44069	273,2777	305,79937	335,06692	351,10915	357,98432	366,12884	380,4652	406,1484	425,6654	424,0727	439,1286	456,6956	481,2742	503,5289	524,389	543,3779	566,7351	592,1013	614,8985	636,7028	632,3164	640,1786	649,0672	640,1097	634,1138	635,9379	649,3151	665,2235	682,7548	703,6504	718,9568	677,9612	738,2206	783,5975	823,169	866,0952	900,4
N 		2,41	2,41	2,46	2,55	2,7	2,66	2,62	2,53	2,49	2,51	2,42	2,41	2,36	2,34	2,33	2,2	2,11	1,97	1,87	1,76	1,68	1,6	1,6	1,54	1,48	1,45	1,37	1,35	1,38	1,35	1,36	1,32	1,32	1,26	1,22	1,19	1,22	1,23	1,21	1,23	1,26	1,25	1,27	1,29	1,34	1,33	1,37	1,39	1,44	1,44	1,44	1,42	1,42	1,39	1,38	1,36	1,36	1,34	1,31	1,27	1,24	1,25	1,24	1,2	1,18	1,14
U 		316,91	317,64	318,45	318,99	319,62	320,04	321,37	322,18	323,05	324,62	325,68	326,32	327,46	329,68	330,19	331,13	332,03	333,84	335,41	336,84	338,76	340,12	341,48	343,15	344,87	346,35	347,61	349,31	351,69	353,2	354,45	355,7	356,54	357,21	358,96	360,97	362,74	363,88	366,84	368,54	369,71	371,32	373,45	375,98	377,7	379,98	382,09	384,02	385,83	387,64	390,1	391,85	394,06	396,74	398,81	401,01	404,41	406,76	408,72	411,65	414,21	416,41	418,53	421,08	424,61	427,35
"""

# 3. Lettura raw, trasposizione e conversione decimali
# Il separatore r'\s+' legge correttamente sia i tab (nativi del copia-incolla Excel) che gli spazi
df_t = pd.read_csv(io.StringIO(dati_raw), sep=r'\s+', header=None, index_col=0)

# Trasponiamo: le variabili tornano colonne, il tempo scorre sulle righe
df = df_t.T

# Pulizia dell'indice vettoriale
df.columns.name = None
df.reset_index(drop=True, inplace=True)

# Conversione della virgola italiana in punto decimale anglosassone e casting a float
for col in df.columns:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

# Cast dell'asse temporale a intero
df['Anno'] = df['Anno'].astype(int)

# 4 & 5. Calcolo variabili derivate del Protocollo Ale-Gemini
df['LC'] = df['L'] / df['C']
df['CL'] = df['C'] / df['L']

# Estrazione limiti strutturali assoluti (Fase Zero Matematizzata)
M = df['LC'].max()
m_param = df['CL'].min()

# Generazione metriche di pressione e deficit demografico
df['S1A'] = (M - df['LC']) / (M + 1)
df['S1B'] = (df['CL'] - m_param) / (1 + m_param)
df['S2'] = (2.1 / df['N']) - 1
# La variabile U (CO2) è già presente nel dataset e pronta all'uso

print("\n[OK] Trasformazione vettoriale completata.")
print(f"Altitudine Media L/C: {df['LC'].mean():.4f}")
print(f"Variabili operative caricate: {list(df.columns)}")
print("-" *40)

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

print("\n" + "="*40)
print("FASE 1: VERIFICHE METODOLOGICHE (PASSAPORTO I(1) E CONFRONTO DI JOHANSEN)")
print("="*40)

# ----------------------------------------------------------------------------------------
# Sotto-Fase 1: Filtri ADF (Augmented Dickey-Fuller)
# ----------------------------------------------------------------------------------------
print("\n[--- Sotto-Fase 1: Check qualifica I(1) ---]")

variabili_da_testare = ['S1A', 'S1B', 'S2']

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

# ----------------------------------------------------------------------------------------
# Sotto-Fase 2:  Johansen (Doppio Cieco con Lag Dinamico AIC)
# ----------------------------------------------------------------------------------------
print("\n[--- Sotto-Fase 2:  Johansen su S1A ed S1B ---]")

# Pulizia dei NaN per le variabili coinvolte nel test
df_joh = df[['S1A', 'S1B', 'S2']].dropna()

from statsmodels.tsa.vector_ar.var_model import VAR

def confronto(data, coppia_nome):
    # 1. Ricerca del Lag Ottimale per la coppia specifica
    modello_var_temp = VAR(data)
    lag_res = modello_var_temp.select_order(maxlags=5)
    p_ottimale = lag_res.aic
    
    # In Johansen, l'argomento k_ar_diff corrisponde a p - 1
    k_ar_diff_joh = max(1, p_ottimale - 1)
    
    # 2. Esecuzione del test di Johansen con lag dinamico
    joh_res = coint_johansen(data, det_order=0, k_ar_diff=k_ar_diff_joh)
    
    # Estrazione Traccia per r=0 e Valore Critico al 95%
    traccia = joh_res.lr1[0]
    soglia_95 = joh_res.cvt[0, 1]
    margine = traccia - soglia_95
    
    print(f"\n  [ Cointegrazione {coppia_nome} ]")
    print(f"  - Lag ottimale (AIC) utilizzato: {p_ottimale}")
    print(f"  - Statistica Traccia: {traccia:.4f}")
    print(f"  - Soglia Crit. (95%): {soglia_95:.4f}")
    
    if margine > 0:
        print(f"  -> Cointegrazione TROVATA. Margine di forza: +{margine:.4f}")
    else:
        print(f"  -> Cointegrazione ASSENTE. Distanza dalla soglia: {margine:.4f}")
        
    return margine

# Test Elastico Lineare (S1A, S2)
A_2 = confronto(df_joh[['S1A', 'S2']], "S1A - S2 ")

# Test Elastico Iperbolico (S1B, S2)
B_2 = confronto(df_joh[['S1B', 'S2']], "S1B - S2 ")

print("\n[ ESITO DEL CONFRONTO ]")
if A_2 > B_2:
    print("» E' più forte il legame strutturale con S1A.")
elif B_2 > A_2:
    print("» E' più forte il legame strutturale con S1B.")
else:
    print("» Pareggio termodinamico tra i due elastici.")
print("-" *40)

import statsmodels.api as sm

print("\n" + "="*40)
print("FASE 2 E FASE 3: IDENTIFICAZIONE FRATTURE E CONFRONTO CORRELAZIONI")
print("="*40)

# ----------------------------------------------------------------------------------------
# 1. Definizione della funzione unificata per la Statistica F di Chow
# ----------------------------------------------------------------------------------------
def calcola_chow_F(y, x, split_idx):
    x_with_const = sm.add_constant(x)
    mod_tot = sm.OLS(y, x_with_const).fit()
    ssr_tot = mod_tot.ssr
    
    y1 = y.iloc[:split_idx]
    x1 = x_with_const.iloc[:split_idx]
    ssr1 = sm.OLS(y1, x1).fit().ssr
    
    y2 = y.iloc[split_idx:]
    x2 = x_with_const.iloc[split_idx:]
    ssr2 = sm.OLS(y2, x2).fit().ssr
    
    k = x_with_const.shape[1]
    N = len(y)
    
    denominatore = (ssr1 + ssr2) / (N - 2 * k)
    if denominatore == 0:
        return 0
    numeratore = (ssr_tot - (ssr1 + ssr2)) / k
    return numeratore / denominatore

# ----------------------------------------------------------------------------------------
# 2. Motore di scansione strutturale e calcolo correlazioni
# ----------------------------------------------------------------------------------------
def analizza_fratture_e_correlazioni(var_name):
    print(f"\n[--- Analisi Strutturale Indipendente: {var_name} ---]")
    y_full = df['S2']
    x_full = df[var_name]
    
    N_tot = len(df)
    mezzogiorno = int(N_tot * 0.60)
    margine = 5
    
    # Prima Scansione (BP1)
    max_f1 = -1
    bp1_idx = -1
    
    for i in range(margine, N_tot - margine):
        f_stat = calcola_chow_F(y_full, x_full, i)
        if f_stat > max_f1:
            max_f1 = f_stat
            bp1_idx = i
            
    anno_bp1 = df['Anno'].iloc[bp1_idx]
    anno_zero = df['Anno'].iloc[0]
    anno_fine = df['Anno'].iloc[-1]
    
    # Bivio Logico Termodinamico e Deep Scan
    if bp1_idx > mezzogiorno:
        epoca_start = anno_zero
        epoca_end = anno_bp1
        print(f"  > Singola rottura terminale trovata nel {anno_bp1}.")
        
        # Calcolo correlazioni sui due segmenti
        corr_1 = df.iloc[:bp1_idx][var_name].corr(df.iloc[:bp1_idx]['S2'])
        corr_2 = df.iloc[bp1_idx:][var_name].corr(df.iloc[bp1_idx:]['S2'])
        
        correlazioni = {
            f"0-BP1 ({anno_zero}-{anno_bp1})": corr_1,
            f"BP1-Fine ({anno_bp1}-{anno_fine})": corr_2
        }
    else:
        print(f"  > Trauma precoce rilevato nel {anno_bp1}. Avvio Deep Scan...")
        max_f2 = -1
        bp2_relative_idx = -1
        
        y_sub = y_full.iloc[bp1_idx:].reset_index(drop=True)
        x_sub = x_full.iloc[bp1_idx:].reset_index(drop=True)
        N_sub = len(y_sub)
        
        for j in range(margine, N_sub - margine):
            f_stat2 = calcola_chow_F(y_sub, x_sub, j)
            if f_stat2 > max_f2:
                max_f2 = f_stat2
                bp2_relative_idx = j
                
        if max_f2 != -1:
            bp2_idx = bp1_idx + bp2_relative_idx
            anno_bp2 = df['Anno'].iloc[bp2_idx]
            epoca_start = anno_bp1
            epoca_end = anno_bp2
            print(f"  > Deep Scan completato. BP2 trovato nel {anno_bp2}.")
            
            # Calcolo correlazioni sui tre segmenti
            corr_1 = df.iloc[:bp1_idx][var_name].corr(df.iloc[:bp1_idx]['S2'])
            corr_2 = df.iloc[bp1_idx:bp2_idx][var_name].corr(df.iloc[bp1_idx:bp2_idx]['S2'])
            corr_3 = df.iloc[bp2_idx:][var_name].corr(df.iloc[bp2_idx:]['S2'])
            
            correlazioni = {
                f"0-BP1 ({anno_zero}-{anno_bp1})": corr_1,
                f"BP1-BP2 ({anno_bp1}-{anno_bp2})": corr_2,
                f"BP2-Fine ({anno_bp2}-{anno_fine})": corr_3
            }
        else:
            epoca_start = anno_zero
            epoca_end = anno_bp1
            print("  > Deep Scan interrotto (gradi di libertà insufficienti).")
            corr_1 = df.iloc[:bp1_idx][var_name].corr(df.iloc[:bp1_idx]['S2'])
            corr_2 = df.iloc[bp1_idx:][var_name].corr(df.iloc[bp1_idx:]['S2'])
            correlazioni = {
                f"0-BP1 ({anno_zero}-{anno_bp1})": corr_1,
                f"BP1-Fine ({anno_bp1}-{anno_fine})": corr_2
            }
            
    print(f"  > Epoca Stabile Definita: {epoca_start} - {epoca_end}")
    for periodo, valore in correlazioni.items():
        print(f"    - Correlazione su {periodo}: {valore:.4f}")
        
    # Estrazione del valore massimo in valore assoluto per il confronto
    # (assumiamo la concordanza di segno come secondaria all'intensità del legame)
    picco_max = max(correlazioni.values(), key=abs)
    return abs(picco_max), correlazioni, epoca_start, epoca_end

# ----------------------------------------------------------------------------------------
# 3. Esecuzione Doppio Screening
# ----------------------------------------------------------------------------------------
picco_A, correlazioni_A, start_A, end_A = analizza_fratture_e_correlazioni('S1A')
picco_B, correlazioni_B, start_B, end_B = analizza_fratture_e_correlazioni('S1B')

# ----------------------------------------------------------------------------------------
# 4. Confronto e Promozione Variabile Definitiva
# ----------------------------------------------------------------------------------------
print("\n[--- ELEZIONE DELLA VARIABILE DEFINITIVA ---]")

var_vincente = None
picco_assoluto = -1

if picco_A > picco_B:
    picco_assoluto = picco_A
    candidata = 'S1A'
    epoca_start = start_A
    epoca_end = end_A
else:
    picco_assoluto = picco_B
    candidata = 'S1B'
    epoca_start = start_B
    epoca_end = end_B

print(f"Picco correlazione assoluto S1A: {picco_A:.4f}")
print(f"Picco correlazione assoluto S1B: {picco_B:.4f}")

if picco_assoluto > 0.5:
    var_vincente = candidata
    df['S1'] = df[var_vincente]
    print(f"\n>>> La variabile più adeguata è {var_vincente} e passa al calcolo di k ed al VECM.")
else:
    var_vincente = candidata
    df['S1'] = df[var_vincente]
    print(f"\n[ATTENZIONE] Nessun picco supera la soglia critica di 0.5 (Picco rilevato: {picco_assoluto:.4f}).")
    print(f">>> La variabile {var_vincente} risulta il best-fit relativo ed è promossa al calcolo di k e al VECM.")

print("-" *40)

from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# FASE 4: a ) STIMA DEL PARAMETRO STRUTTURALE (k); b )TEST FINESTRE ESPANSIVE (BACKTESTING OUT-OF-SAMPLE); c) DIAGRAMMA DI FASE";
# ==============================================================

# ==============================================================================
# FASE 4a ) STIMA DEL PARAMETRO STRUTTURALE (k)
# ==============================================================

print("\n" + "="*40)
print("FASE 4: STIMA DEL PARAMETRO STRUTTURALE (k) E DIAGRAMMA DI FASE")
print("="*40)

# 1. Partizionamento del dataset in base all'Epoca Stabile
mask_stabile = (df['Anno'] >= epoca_start) & (df['Anno'] <= epoca_end)
df_stabile = df[mask_stabile]
df_instabile = df[~mask_stabile]

# Estrazione dei vettori operativi per la regressione
x_stabile = df_stabile['S1'].values
y_stabile = df_stabile['S2'].values

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

print(f"Stima parametro strutturale (k) nell'Epoca Stabile: {k_strutturale:.4f}")

# ==============================================================================
# FASE 4b: TEST FINESTRE ESPANSIVE (BACKTESTING OUT-OF-SAMPLE)
# ==============================================================================

print("\n" + "="*40)
print("TEST FINESTRE ESPANSIVE (TUTTI i quinquenni dell'Epoca Stabile):")

# Funzione di supporto per l'output in stile Excel italiano (virgola per i decimali)
def formatta_ita(valore, decimali=4):
    return f"{valore:.{decimali}f}".replace('.', ',')

# Estrazione degli anni per le etichette di stampa
anni_stabile = df_stabile['Anno'].values

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
        print("Errore ; " + " ; ".join([formatta_ita(e, 2) for e in errore_iter]))
        
        # Allargamento della finestra temporale per il ciclo successivo
        train_size += 5
else:
    print("Dataset dell'Epoca Stabile troppo corto (minimo 15 anni richiesti).")

# ==============================================================================
# FASE 4c ) Generazione del Diagramma di Fase (Spazio delle configurazioni)
# ===============================================================

plt.figure(figsize=(10, 6))

# Plot dei vettori di stato nell'Epoca Stabile (Regime Lineare di Hooke)
plt.scatter(df_stabile['S1'], df_stabile['S2'], color='blue', label='Attrazione Strutturale (Epoca Stabile)', alpha=0.7)

# Plot delle osservazioni esterne (Dinamiche Caotiche e Hysteresis)
plt.scatter(df_instabile['S1'], df_instabile['S2'], color='red', marker='x', label='Divergenza (Isteresi/Caos)', alpha=0.8)

# Vettore di proiezione per la retta teorica
x_min_assoluto = df['S1'].min()
x_max_assoluto = df['S1'].max()
asse_x_teorico = np.linspace(x_min_assoluto, x_max_assoluto, 200)
asse_y_teorico = k_strutturale * asse_x_teorico

# Tracciamento dell'invariante strutturale
plt.plot(asse_x_teorico, asse_y_teorico, color='black', linestyle='--', linewidth=2, label=f'Relazione Teorica (k = {k_strutturale:.4f})')

# Impostazioni morfologiche del grafico
plt.title("Diagramma di Fase: Spazio delle Configurazioni S1-S2", fontsize=14, fontweight='bold')
plt.xlabel(f"Indice di Estrazione ({var_vincente})", fontsize=12)
plt.ylabel("Deficit Demografico (S2)", fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

print("-" * 40)

from statsmodels.tsa.vector_ar.vecm import select_order, VECM
from statsmodels.tsa.vector_ar.var_model import VAR


# ==============================================================================
# FASE 5: DINAMICA VECM E CAUSALITÀ DI TODA-YAMAMOTO (CONFIGURAZIONE A 4 ANNI)
# ==============================================================================

print("\n" + "="*40)
print("FASE 5: DINAMICA VECM E CAUSALITÀ DI TODA-YAMAMOTO")
print("="*40)

# ----------------------------------------------------------------------------------------
# 1. Inizializzazione Indice Ambientale (S3)
# ----------------------------------------------------------------------------------------
# S3 è definito come il complemento a 1 del rapporto tra la baseline (316.91) e le emissioni U
df['S3'] = 1 - (316.91 / df['U'])

# ----------------------------------------------------------------------------------------
# 2. Ricerca del Lag Ottimale e Modello a Correzione d'Errore (VECM)
# ----------------------------------------------------------------------------------------
df_vecm = df[['S2', 'S1']].dropna() 

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
# 3. Test di Causalità di Toda-Yamamoto (Configurazione a 4 Anni)
# ----------------------------------------------------------------------------------------
df_ty = df[['S2', 'S1', 'S3']].dropna() # Aggiornato a S1

# Ricerca dell'ordine ottimale per il sistema trivariato (p_ty)
modello_var_base = VAR(df_ty)
lag_res_ty = modello_var_base.select_order(maxlags=5)
p_ty = lag_res_ty.aic

if p_ty == 0:
    p_ty = 1

# Integrazione del ritardo massimo (d_max = 1 per serie storiche I(1))
d_max = 1
lag_ty_totale = p_ty + d_max

# Costruzione manuale e trasparente della matrice dei ritardi
df_lags = pd.DataFrame()
df_lags['S2'] = df_ty['S2']
df_lags['S1'] = df_ty['S1']

# Generiamo tutte le colonne sfalsate temporalmente
for var in ['S2', 'S1', 'S3']:
    for i in range(1, lag_ty_totale + 1):
        df_lags[f'L{i}_{var}'] = df_ty[var].shift(i)
        
df_lags = df_lags.dropna()

# Matrice delle variabili indipendenti (X) con l'aggiunta della costante
colonne_lag = [c for c in df_lags.columns if c.startswith('L')]
X = sm.add_constant(df_lags[colonne_lag])

# Esecuzione OLS indipendente per l'equazione di S2 (Nascite)
y_S2 = df_lags['S2']
res_ols_S2 = sm.OLS(y_S2, X).fit()

# Esecuzione OLS indipendente per l'equazione di S1 (Salari)
y_S1 = df_lags['S1']
res_ols_S1 = sm.OLS(y_S1, X).fit()

# Costruzione stringhe di restrizione per il test Wald (su TUTTI i lag, incluso il crollo del 4° anno)
restr_S1_causa_S2 = ", ".join([f"L{i}_S1 = 0" for i in range(1, lag_ty_totale + 1)])
restr_S3_causa_S1 = ", ".join([f"L{i}_S3 = 0" for i in range(1, lag_ty_totale + 1)])

wald_S1_S2 = res_ols_S2.wald_test(restr_S1_causa_S2, scalar=True)
wald_S3_S1 = res_ols_S1.wald_test(restr_S3_causa_S1, scalar=True)

# ----------------------------------------------------------------------------------------
# 4. Report Accademico Conclusivo
# ----------------------------------------------------------------------------------------
print("\n[ REPORT ANALISI VECM ]")
print(f"  - Ritardo temporale ottimale bivariato (p_opt) : {p_opt}")
print(f"  - Coefficiente di correzione d'errore (alpha)  : {alpha_S2:.4f}")

print("\n[ REPORT CAUSALITÀ DI TODA-YAMAMOTO (Impatto a 4 Anni) ]")
print(f"  - Ritardo temporale ottimale trivariato (p_ty) : {p_ty}")
print(f"  - Ritardo stimato nel VAR aumentato (p_ty + d_max) : {lag_ty_totale}")
print(f"  - Restrizione Wald test applicata su             : TUTTI i {lag_ty_totale} lag (incluso crollo strutturale)")
print(f"  - Valore-p (H0: S1 non causa Granger S2)         : {wald_S1_S2.pvalue:.4f}")
print(f"  - Valore-p (H0: S3 non causa Granger S1)         : {wald_S3_S1.pvalue:.4f}")

print("-" *40)

import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

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
# "FASE 7: DINAMICA DELL'ISTERESI E STABILITÀ STRUTTURALE"
# -------------------------------------------

print("FASE 7: DINAMICA DELL'ISTERESI E STABILITÀ STRUTTURALE")
print("="*40)

# 3. Test di Stabilità CUSUM (Residui Ricorsivi OLS)
print("\n[ Generazione Test CUSUM (Residui Ricorsivi OLS) in corso... ]")
# Utilizziamo df_vecm per garantire la perfetta corrispondenza dimensionale tra S1 e S2
X_ols = sm.add_constant(df_vecm['S1'])
res_rls = sm.RecursiveLS(df_vecm['S2'], X_ols).fit()

fig_cusum = res_rls.plot_cusum()
plt.title("Test CUSUM sui residui ricorsivi (S2 ~ S1)")
plt.tight_layout()
plt.savefig("cusum_italia.png", dpi=300) # Salvataggio su file ad alta risoluzione
plt.close(fig_cusum)                     # Chiusura e svuotamento della memoria grafica
print("  -> Grafico CUSUM salvato con successo in 'cusum_italia.png'.")

# 4. Funzioni di Risposta agli Impulsi (IRF)
print("\n[ Generazione Funzioni di Risposta agli Impulsi (IRF) in corso... ]")
# Sfruttiamo l'oggetto risultati_vecm già stimato in Fase 5
irf = risultati_vecm.irf(periods=15)
fig_irf = irf.plot(orth=True)
fig_irf.suptitle("Funzioni di Risposta agli Impulsi Ortogonalizzate (15 periodi)", fontsize=12)
plt.tight_layout()
fig_irf.subplots_adjust(top=0.90)
plt.savefig("irf_italia.png", dpi=300)   # Salvataggio su file
plt.close(fig_irf)                       # Chiusura e svuotamento memoria grafica
print("  -> Grafico IRF salvato con successo in 'irf_italia.png'.")

# -------------------------------------------
# "FASE 8: DINAMICA EVOLUTIVA (ROLLING PEARSON E EWS)"
# -------------------------------------------

print("\n" + "="*40)
print("FASE 8: DINAMICA EVOLUTIVA (ROLLING PEARSON E EWS)")
print("="*40)

# 1. Rolling Pearson (10 e 15 anni)
print("[ Calcolo Rolling Pearson 10 e 15 anni in corso... ]")
window_10 = 10
window_15 = 15

# Calcolo della correlazione mobile tra S1 e S2
rolling_corr_10 = df_vecm['S1'].rolling(window=window_10).corr(df_vecm['S2'])
rolling_corr_15 = df_vecm['S1'].rolling(window=window_15).corr(df_vecm['S2'])

# --- FIX: Estrazione dell'asse temporale reale (Anni) ---
# Recuperiamo gli anni dal dataframe originale 'df' allineandoli all'indice di df_vecm
asse_x_anni = df.loc[df_vecm.index, 'Anno']

fig_rolling, ax = plt.subplots(figsize=(10, 5))

# ATTENZIONE AL FIX: Inserito asse_x_anni come primo argomento per l'asse X
ax.plot(asse_x_anni, rolling_corr_10, label='Rolling Pearson (10 anni)', color='blue')
ax.plot(asse_x_anni, rolling_corr_15, label='Rolling Pearson (15 anni)', color='orange', linestyle='--')

ax.axhline(0, color='black', linewidth=1)
ax.axvline(1977, color='red', linestyle=':', label='BP1 (1977)')
ax.axvline(2018, color='red', linestyle=':', label='BP2 (2018)')

ax.set_title("Evoluzione Dinamica della Correlazione (S1 vs S2)")
ax.legend()
plt.tight_layout()
plt.savefig("rolling_pearson_italia.png", dpi=300)
plt.close(fig_rolling)
print("  -> Grafico Rolling Pearson salvato in 'rolling_pearson_italia.png'.")

# 2. Early Warning Signals (Varianza Mobile su S2)
print("[ Analisi EWS su Variabile Ambientale S2 in corso... ]")
# Assumendo che S2 sia nel df originale e allineato
if 'S2' in df.columns:
    rolling_var_S2 = df['S2'].rolling(window=10).var()
    fig_ews, ax_ews = plt.subplots(figsize=(10, 5))
    
    # --- FIX: Inserito df['Anno'] come asse X per l'allineamento temporale ---
    ax_ews.plot(df['Anno'], rolling_var_S2, color='purple', label='Varianza Mobile S2 (10 anni)')
    
    ax_ews.axvline(1977, color='red', linestyle=':')
    ax_ews.set_title("Early Warning Signals: Varianza di S2 (Critical Slowing Down)")
    ax_ews.legend()
    plt.tight_layout()
    plt.savefig("ews_s2_italia.png", dpi=300)
    plt.close(fig_ews)
    print("  -> Grafico EWS S2 salvato in 'ews_s2_italia.png'.")

# -------------------------------------------
# "FASE 9: VALIDAZIONE E CONO DI ROTTURA POST-2018 (FIX DEFINITIVO)"
# ------------------------------------------

print("\n" + "="*40)
print("FASE 9: VALIDAZIONE E CONO DI ROTTURA POST-2018 (FIX DEFINITIVO)")
print("="*40)

df_completo = df.copy()

# 1. Recupero brutale e sicuro dell'asse dei tempi (Anno)
if 'Anno' in df_completo.columns:
    df_completo = df_completo.set_index('Anno')
elif 'Year' in df_completo.columns:
    df_completo = df_completo.set_index('Year')

# Cast diretto a numero intero per eliminare ogni ambiguità
try:
    if isinstance(df_completo.index, pd.DatetimeIndex):
        df_completo.index = df_completo.index.year
    else:
        df_completo.index = df_completo.index.astype(int)
    print(f"[ Info ] Indice ripristinato. Orizzonte reale: {df_completo.index.min()} - {df_completo.index.max()}")
except Exception as e:
    print(f"[ Attenzione ] Errore cast indice: {e}")

# 2. Selezione Colonne S1 e S2
col_S1 = 'S1' if 'S1' in df_completo.columns else ('S1A' if 'S1A' in df_completo.columns else None)

if col_S1 and 'S2' in df_completo.columns:
    # Creazione dataset pulito
    df_analisi = df_completo[[col_S1, 'S2']].dropna()
    df_analisi.columns = ['S1', 'S2']
    
    df_train = df_analisi[df_analisi.index <= 2018]
    df_test = df_analisi[df_analisi.index >= 2019]
    
    print(f"[ Conteggio ] Righe Storiche (<=2018): {len(df_train)} | Righe Validazione (>=2019): {len(df_test)}")
    
    from statsmodels.tsa.vector_ar.vecm import select_order

    if not df_test.empty:
        print("[ Simulazione Cono di Confidenza 95% in corso... ]")
        
        # Ricalcoliamo il lag ottimale in modo indipendente, usando SOLO i dati storici
        lag_order_train = select_order(df_train, maxlags=4, deterministic="co")
        opt_lag_train = max(1, lag_order_train.aic)
        
        # Ri-stimiamo il VECM sull'epoca storica con il suo lag specifico
        vecm_train = VECM(df_train, deterministic="co", k_ar_diff=opt_lag_train, coint_rank=1).fit()
        
        steps = len(df_test)
        forecast, lower, upper = vecm_train.predict(steps=steps, alpha=0.05)
        
        idx_s2 = list(df_train.columns).index('S2')
        s2_forecast = forecast[:, idx_s2]
        s2_lower = lower[:, idx_s2]
        s2_upper = upper[:, idx_s2]
        
        fig_cono, ax_cono = plt.subplots(figsize=(10, 5))
        ax_cono.plot(df_train.index, df_train['S2'], label='S2 Storico (1977-2018)', color='blue')
        ax_cono.plot(df_test.index, df_test['S2'], label='S2 Reale (2019+)', color='red', marker='o')
        ax_cono.plot(df_test.index, s2_forecast, label='S2 Previsto (VECM)', color='green', linestyle='--')
        ax_cono.fill_between(df_test.index, s2_lower, s2_upper, color='green', alpha=0.2, label='Cono di Confidenza 95%')
        
        ax_cono.set_title(" Sfondamento del Cono di Confidenza Post-2018")
        ax_cono.set_xlabel("Anno")
        ax_cono.set_ylabel("S2 (Deficit Demografico)")
        ax_cono.legend()
        plt.tight_layout()
        plt.savefig("cono_confidenza_italia.png", dpi=300)
        plt.close(fig_cono)
        print("  -> TRIONFO: Grafico Cono di Confidenza salvato in 'cono_confidenza_italia.png'.")
    else:
        print("  -> ERRORE: Dati post-2018 ancora non rintracciabili.")
        
        print("\n" + "-" *40)
print(">>> PIPELINE ANALITICA V2.0 COMPLETATA CON SUCCESSO <<<")
print("-" *40)

import numpy as np
import pandas as pd

# ==============================================================================
# FASE 9: TERMODINAMICA INTEGRALE (AREE DI LEGAME E DEFICIT STRUTTURALE)
# ==============================================================================
print("\n" + "="*40)
print("FASE 9: CALCOLO DELLE AREE INTEGRALI (TERMODINAMICA SOCIALE)")
print("="*40)

try:
    # 1. Creazione di un dataframe isolato per evitare problemi con i NaN della finestra mobile
    df_integrali = pd.DataFrame({
        'Anno': asse_x_anni,
        'RP10': rolling_corr_10,
        'S1': df_vecm['S1'],
        'S2': df_vecm['S2']
    }).dropna()

    # Estrazione dell'asse temporale pulito per l'integrazione geometrica
    x_clean = df_integrali['Anno'].values

    # 2. Calcolo Integrale A: L'Energia di Legame (Massa di Coesione)
    # Filtriamo l'onda: poniamo a 0 tutti i valori dove la correlazione è < 0.5
    y_legame = np.where(df_integrali['RP10'] > 0.5, df_integrali['RP10'], 0)
    
    # Integrazione col metodo dei trapezi sull'asse degli anni reali
    area_legame = np.trapz(y_legame, x=x_clean)

    # 3. Calcolo Integrale B: Il Lavoro Strutturale (Deficit Cumulato S1 - S2)
    y_deficit = df_integrali['S1'] - df_integrali['S2']
    area_deficit = np.trapz(y_deficit, x=x_clean)

    # 4. Calcolo della Costante Elastica Storica (Rapporto A / B)
    if area_deficit != 0:
        costante_elastica = area_legame / area_deficit
    else:
        costante_elastica = np.nan

    print("[ REPORT INTEGRALI DEFINITI ]")
    print(f"  - Massa di Coesione (Area RP10 > 0.5)  : {area_legame:.2f} [Unità Correlazione-Anno]")
    print(f"  - Deficit Cumulato (Area S1 - S2)      : {area_deficit:.2f} [Unità Spread-Anno]")
    print(f"  - Costante Elastica (Rapporto A/B)     : {costante_elastica:.4f}")
    
    # Salviamo i risultati in un piccolo file di testo per averli sempre a portata di mano
    with open("report_termodinamica.txt", "w") as f:
        f.write("=== REPORT TERMODINAMICA SOCIALE (INTEGRALI) ===\n")
        f.write(f"Massa di Coesione (Area RP10 > 0.5) : {area_legame:.2f}\n")
        f.write(f"Deficit Cumulato (Area S1 - S2)     : {area_deficit:.2f}\n")
        f.write(f"Costante Elastica                   : {costante_elastica:.4f}\n")
    print("  -> Risultati esportati con successo in 'report_termodinamica.txt'.")
    print("-" *40)

except Exception as e:
    print(f"[ERRORE] Calcolo degli integrali fallito: {e}")
