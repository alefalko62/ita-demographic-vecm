import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import select_order, VECM
import warnings
warnings.filterwarnings("ignore") # Nasconde i warning noiosi di formattazione

# ==========================================
# 1. INSERIMENTO DATI (COPIA E INCOLLA QUI I DATI ITALIANI (le righe complete))
# ==========================================

# Incolla la riga di S2 o S8 qui sotto:
S_estrazione_raw = """
0,152746442	0,151185205	0,129232772	0,082479905	0,068288409	0,08258608	0,095054389	0,087025738	0,089324553	0,094316927	0,06031078	0,014157516	0,003621941	0,012052012	0,028691553	-1,8782E-15	0,016510863	0,007203887	0,025629395	0,041066315	0,051298789	0,04094195	0,049146599	0,055014171	0,07687899	0,081035111	0,098745046	0,099768447	0,102145	0,103725538	0,087614863	0,08076154	0,082122577	0,085557303	0,110433328	0,134048531	0,129142007	0,122226995	0,130228167	0,129588654	0,141573711	0,143729023	0,136868153	0,133383134	0,134875141	0,123387001	0,116598075	0,117498674	0,109090164	0,090664465	0,092775517	0,099571953	0,090755122	0,095565916	0,09780629	0,094704396	0,102485274	0,097894409	0,09069451	0,08437417	0,079224317	0,086330747	0,105780166	0,118910662	0,096868826	0,086414986								
"""

# Incolla la riga di S3 qui sotto:
S3_raw = """
-0,128630705	-0,128630705	-0,146341463	-0,176470588	-0,222222222	-0,210526316	-0,198473282	-0,169960474	-0,156626506	-0,163346614	-0,132231405	-0,128630705	-0,110169492	-0,102564103	-0,098712446	-0,045454545	-0,004739336	0,065989848	0,122994652	0,193181818	0,25	0,3125	0,3125	0,363636364	0,418918919	0,448275862	0,532846715	0,555555556	0,52173913	0,555555556	0,544117647	0,590909091	0,590909091	0,666666667	0,721311475	0,764705882	0,721311475	0,707317073	0,73553719	0,707317073	0,666666667	0,68	0,653543307	0,627906977	0,567164179	0,578947368	0,532846715	0,510791367	0,458333333	0,458333333	0,458333333	0,478873239	0,478873239	0,510791367	0,52173913	0,544117647	0,544117647	0,567164179	0,603053435	0,653543307	0,693548387	0,68	0,693548387	0,75	0,779661017	0,842105263								
"""

# ==========================================
# 2. ELABORAZIONE VECM
# ==========================================

def clean_data(raw_string):
    clean_string = raw_string.replace(',', '.').strip()
    return [float(x) for x in clean_string.split() if x]

s_estrazione = clean_data(S_estrazione_raw)
s3 = clean_data(S3_raw)

if len(s_estrazione) != len(s3):
    raise ValueError(f"Errore: Lunghezze diverse. S_estrazione: {len(s_estrazione)}, S3: {len(s3)}.")

df = pd.DataFrame({
    'S_estrazione': s_estrazione,
    'S3': s3
})

# 1. Selezione automatica dei lag ottimali tramite Criterio AIC
lag_order = select_order(df, maxlags=5, deterministic="ci")
ottimo_lag = lag_order.aic

print("\n" + "="*80)
print(f"MODELLO VECM (Vector Error Correction Model) - Ritardi ottimali AIC: {ottimo_lag}")
print("="*80)

# 2. Costruzione e fitting del Modello VECM
# Assumiamo coint_rank=1 (una relazione di cointegrazione come validato dal tuo Johansen)
vecm_model = VECM(df, k_ar_diff=ottimo_lag, coint_rank=1, deterministic="ci")
vecm_res = vecm_model.fit()

# 3. Estrazione dei risultati cruciali: I coefficienti Alpha (Causalità di lungo periodo)
print("\n>>> CAUSALITÀ DI LUNGO PERIODO (Error Correction Term - Alpha) <<<")
print("Se il P>|z| è < 0.05, significa che la variabile è CAUSATA dalla relazione di lungo termine.\n")

# 3. Estrazione DIRETTA dei veri coefficienti Alpha (Causalità di lungo periodo)
print("\n" + "="*80)
print(">>> I VERI COEFFICIENTI ALPHA (LUNGO PERIODO) E P-VALUES <<<")
print("="*80)

# Estraiamo i nomi delle variabili per chiarezza
nomi_variabili = vecm_model.endog_names

# I coefficienti alpha (velocità di aggiustamento) e i loro p-values
alphas = vecm_res.alpha
pvalues = vecm_res.pvalues_alpha

for i in range(len(nomi_variabili)):
    var_name = nomi_variabili[i]
    p_val = pvalues[i][0]
    alpha_val = alphas[i][0]
    
    print(f"Variabile: {var_name}")
    print(f"  - Coefficiente Alpha: {alpha_val:.5f}")
    print(f"  - P-value: {p_val:.5f}")
    
    if p_val < 0.05:
        print(f"  -> RISULTATO: SIGNIFICATIVO! {var_name} subisce la gravità del sistema.")
    else:
        print(f"  -> RISULTATO: NON significativo. {var_name} si muove in modo indipendente.")
    print("-" * 40)