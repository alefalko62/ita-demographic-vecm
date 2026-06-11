import csv
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import statsmodels.api as sm
import sys
import traceback
import warnings
from scipy import stats
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

# Pulizia dell'indice vettoriale e rimozione del nome colonne
df.columns.name = None

# 1. Prima convertiamo TUTTE le colonne (inclusa 'Anno') per pulire stringhe/virgole e renderle numeriche
for col in df.columns:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

# 2. Ora che è un numero pulito, castiamo 'Anno' a intero puro (senza decimali)
df['Anno'] = df['Anno'].astype(int)

# 3. Infine, lo impostiamo come spina dorsale (indice) del DataFrame
df.set_index('Anno', inplace=True)

# 4 & 5. Calcolo variabili derivate del Protocollo Ale-Gemini
df['LC'] = df['L'] / df['C']
df['CL'] = df['C'] / df['L']

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

# =====================================================================
# MODULO FOTOGRAFICO STORICO: TEST DI CHOW SU DATE SPECIFICHE
# =====================================================================

# Scegliamo S1A come guida (il "TUTTE NOSTRE" del protocollo)
variabile_x = 'S1A' 
variabile_y = 'S2'
anni_storici = [1984, 1992, 2003, 2008]

print("\n" + "="*60)
print(" RADIOGRAFIA STORICA: IMPATTO DELLE RIFORME SUL MOLTIPLICATORE")
print("="*60)

# 0. Capiamo dove sono gli anni nel tuo dataframe senza alterare l'originale
df_chow = df.copy()

if 'Anno' in df_chow.columns:
    colonna_tempo = 'Anno'
elif 'Year' in df_chow.columns:
    colonna_tempo = 'Year'
elif 'TIME' in df_chow.columns:
    colonna_tempo = 'TIME'
else:
    # Se non c'è una colonna esplicita, l'anno è l'indice! Lo trasformiamo in colonna.
    df_chow = df_chow.reset_index()
    colonna_tempo = df_chow.columns[0] # Prende il nome del vecchio indice

# 1. Pulizia di sicurezza
df_clean = df_chow[[colonna_tempo, variabile_x, variabile_y]].dropna()

# Assicuriamoci che l'anno sia un numero intero per fare i confronti (< e >)
df_clean[colonna_tempo] = df_clean[colonna_tempo].astype(int)

# 2. Modello su tutto il campione (SSR_pool)
X_tot = sm.add_constant(df_clean[variabile_x])
y_tot = df_clean[variabile_y]
mod_tot = sm.OLS(y_tot, X_tot).fit()
ssr_tot = mod_tot.ssr
k = len(mod_tot.params) # Numero di parametri (intercetta + pendenza)
N = len(df_clean)

# 3. Ciclo sulle date storiche
report_chow = [] # <--- NUOVA: Scatola per memorizzare i testi per l'esportazione
for anno_break in anni_storici:
    # Dividiamo il campione
    df_prima = df_clean[df_clean[colonna_tempo] <= anno_break]
    df_dopo = df_clean[df_clean[colonna_tempo] > anno_break]
    
    # Se il taglio è troppo vicino ai bordi, saltiamo l'anno per evitare errori
    if len(df_prima) <= k or len(df_dopo) <= k:
        print(f"\n--- Anno: {anno_break} --- Dati insufficienti per il calcolo.")
        continue
        
    # Modello PRIMA della riforma
    X_prima = sm.add_constant(df_prima[variabile_x])
    y_prima = df_prima[variabile_y]
    mod_prima = sm.OLS(y_prima, X_prima).fit()
    ssr_prima = mod_prima.ssr
    k2_prima = mod_prima.params[variabile_x]
    
    # Modello DOPO la riforma
    X_dopo = sm.add_constant(df_dopo[variabile_x])
    y_dopo = df_dopo[variabile_y]
    mod_dopo = sm.OLS(y_dopo, X_dopo).fit()
    ssr_dopo = mod_dopo.ssr
    k2_dopo = mod_dopo.params[variabile_x]
    
    # Calcolo Test di Chow
    numeratore = (ssr_tot - (ssr_prima + ssr_dopo)) / k
    denominatore = (ssr_prima + ssr_dopo) / (N - 2*k)
    f_stat = numeratore / denominatore
    p_value = 1 - stats.f.cdf(f_stat, k, N - 2*k)
    
    # Formattazione dell'output
    significativo = "SÌ" if p_value < 0.05 else "NO"
    delta = k2_dopo - k2_prima
    segno = "Diminuzione" if delta < 0 else "Aumento"
    
    # Prepariamo il blocco di testo
    blocco_testo = f"""--- Anno della Riforma/Crisi: {anno_break} ---
Rottura Strutturale Significativa?  [{significativo}] (p-value: {p_value:.4f})
Forza di k2 PRIMA del {anno_break}:        {k2_prima:.4f}
Forza di k2 DOPO il {anno_break}:         {k2_dopo:.4f}
Impatto sul moltiplicatore:         {segno} di {abs(delta):.4f}"""
    
    print(f"\n{blocco_testo}") # Stampa a video come prima
    report_chow.append(blocco_testo) # Salva in memoria per il file txt

print("\n" + "="*60)

import traceback

# =====================================================================
# MODULO TVP: FILTRO DI KALMAN (RECURSIVE LEAST SQUARES) E ESPORTAZIONE
# =====================================================================
print("\n" + "="*60)
print(" MODULO TVP: TRAIETTORIA STORICA DELLA CINGHIA DI TRASMISSIONE")
print("="*60)

# Assicuriamoci di avere i dati puliti e ordinati cronologicamente
df_tvp = df_clean.sort_values(by=colonna_tempo).copy()
anni_tvp = df_tvp[colonna_tempo].values

# Definiamo le variabili per l'algoritmo
y_kalman = df_tvp[variabile_y]
X_kalman = sm.add_constant(df_tvp[[variabile_x]])

try:
    # 1. CALCOLO FILTRO DI KALMAN
    modello_rls = sm.RecursiveLS(y_kalman, X_kalman)
    risultati_rls = modello_rls.fit()

    nomi_colonne = list(X_kalman.columns)
    idx_param = nomi_colonne.index(variabile_x)
    stati_filtrati = risultati_rls.filtered_state
    k2_traiettoria = stati_filtrati[idx_param, :]

    print(f"\nEvoluzione del moltiplicatore k2 ({variabile_x}) nel tempo:")
    for i, anno in enumerate(anni_tvp):
        if anno % 10 == 0 or anno in anni_storici or anno == anni_tvp[-1]:
            print(f"Anno {anno}: k2 = {k2_traiettoria[i]:.4f}")

    # 2. ESPORTAZIONE DATI (CHOW + KALMAN)
    nome_export = f"Dati_Traiettoria_k2_{nazione}.txt"
    
    # Sicurezza: se report_chow non esiste (es. blocco Chow saltato), creiamo un avviso
    if 'report_chow' not in locals():
        report_chow = ["(Risultati Chow non eseguiti o non disponibili in questa run)"]

    df_export = pd.DataFrame({
        'Anno': anni_tvp,
        'k2_Kalman': k2_traiettoria
    })
    
    with open(nome_export, 'w', encoding='utf-8') as f:
        f.write("============================================================\n")
        f.write(" 1. RADIOGRAFIA STORICA: IMPATTO DELLE RIFORME (TEST DI CHOW)\n")
        f.write("============================================================\n\n")
        for blocco in report_chow:
            f.write(blocco + "\n\n")
            
        f.write("============================================================\n")
        f.write(" 2. TRAIETTORIA STORICA k2 (FILTRO DI KALMAN)\n")
        f.write("============================================================\n\n")
        df_export.to_csv(f, sep='\t', index=False, float_format='%.4f', lineterminator='\n')
        
    print(f"\n[OK] Risultati Chow e dati Kalman esportati nel file: {nome_export}")
    print(f"Percorso esatto di salvataggio: {os.path.abspath(nome_export)}")

    # 3. GENERAZIONE E SALVATAGGIO GRAFICO
    plt.figure(figsize=(12, 6))
    plt.plot(anni_tvp, k2_traiettoria, marker='o', linestyle='-', color='firebrick', linewidth=2, label='Traiettoria k2 (Kalman)')
    plt.axhline(0, color='black', linewidth=1.5, linestyle='--')

    colori_riforme = ['blue', 'orange', 'green', 'purple']
    for anno_break, colore in zip(anni_storici, colori_riforme):
        if anno_break in anni_tvp:
            plt.axvline(x=anno_break, color=colore, linestyle=':', linewidth=2, label=f'Riforma/Crisi {anno_break}')

    plt.title(f'Traiettoria Storica di k2 (Filtro di Kalman)\nDecadimento della trasmissione {variabile_x} -> {variabile_y}', fontsize=14, fontweight='bold')
    plt.xlabel('Anno', fontsize=12)
    plt.ylabel('Forza del Moltiplicatore k2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    nome_grafico = f"TVP_Kalman_{nazione}.png"
    plt.savefig(nome_grafico, dpi=300)
    print(f"[OK] Grafico TVP salvato con successo come: {nome_grafico}")

    # 4. MOSTRA GRAFICO A SCHERMO
    plt.show()

except Exception as e:
    print("\n[ERRORE] Calcolo TVP o Esportazione falliti. Dettagli tecnici:")
    traceback.print_exc()

print("\n>>> DIAGNOSTICA STORICA COMPLETATA <<<")
print("="*60)