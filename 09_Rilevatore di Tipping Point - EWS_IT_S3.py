import pandas as pd
import numpy as np
import scipy.stats as stats

# ==========================================
# 1. INSERIMENTO DATI (COPIA E INCOLLA QUI I DATI ITALIANI DI S3)
# ==========================================

# Incolla SOLO la riga di S3 (la demografia) qui sotto:
S3_raw = """
-0,128630705	-0,128630705	-0,146341463	-0,176470588	-0,222222222	-0,210526316	-0,198473282	-0,169960474	-0,156626506	-0,163346614	-0,132231405	-0,128630705	-0,110169492	-0,102564103	-0,098712446	-0,045454545	-0,004739336	0,065989848	0,122994652	0,193181818	0,25	0,3125	0,3125	0,363636364	0,418918919	0,448275862	0,532846715	0,555555556	0,52173913	0,555555556	0,544117647	0,590909091	0,590909091	0,666666667	0,721311475	0,764705882	0,721311475	0,707317073	0,73553719	0,707317073	0,666666667	0,68	0,653543307	0,627906977	0,567164179	0,578947368	0,532846715	0,510791367	0,458333333	0,458333333	0,458333333	0,478873239	0,478873239	0,510791367	0,52173913	0,544117647	0,544117647	0,567164179	0,603053435	0,653543307	0,693548387	0,68	0,693548387	0,75	0,779661017	0,842105263								
"""

# ==========================================
# 2. CALCOLO DEGLI EARLY WARNING SIGNALS
# ==========================================

def clean_data(raw_string):
    clean_string = raw_string.replace(',', '.').strip()
    return [float(x) for x in clean_string.split() if x]

s3 = clean_data(S3_raw)
df = pd.DataFrame({'S3': s3})

# Definiamo la finestra mobile (es. 15 anni) per calcolare l'evoluzione del sistema
window_size = 15

# Calcolo 1: Varianza Mobile (Rolling Variance)
df['Varianza'] = df['S3'].rolling(window=window_size).var()

# Calcolo 2: Autocorrelazione Mobile (Rolling AR1)
df['AR1'] = df['S3'].rolling(window=window_size).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)

# Rimuoviamo i primi anni (NaN) dovuti alla finestra mobile
df_clean = df.dropna()

# Calcoliamo il Trend (Kendall Tau) per vedere se Varianza e AR1 stanno salendo
# Un Tau > 0 e un p-value < 0.05 significa che stiamo andando dritti verso il burrone
tempo = np.arange(len(df_clean))

tau_var, p_var = stats.kendalltau(tempo, df_clean['Varianza'])
tau_ar1, p_ar1 = stats.kendalltau(tempo, df_clean['AR1'])

print("\n" + "="*80)
print(">>> DIAGNOSTICA DEL TIPPING POINT (CRITICAL SLOWING DOWN) <<<")
print("="*80)

print("\n1. ANALISI DELL'AUTOCORRELAZIONE (Inerzia del Sistema):")
print(f"  - Trend (Tau): {tau_ar1:.4f}")
print(f"  - P-value: {p_ar1:.5f}")
if tau_ar1 > 0 and p_ar1 < 0.05:
    print("  -> ALLARME ROSSO: L'autocorrelazione è in aumento significativo. Il sistema sta perdendo resilienza.")
else:
    print("  -> Nessun segnale di Rallentamento Critico sull'inerzia.")

print("\n2. ANALISI DELLA VARIANZA (Instabilità del Sistema):")
print(f"  - Trend (Tau): {tau_var:.4f}")
print(f"  - P-value: {p_var:.5f}")
if tau_var > 0 and p_var < 0.05:
    print("  -> ALLARME ROSSO: La varianza è in aumento significativo. Il sistema sta sbandando verso la rottura.")
else:
    print("  -> Nessun segnale di instabilità crescente.")
    
print("\n" + "="*80)
print("VALORI DEGLI ULTIMI 5 ANNI (Punto di caduta attuale):")
print(df_clean[['Varianza', 'AR1']].tail())
print("="*80)