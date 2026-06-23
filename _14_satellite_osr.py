import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM, select_order

warnings.filterwarnings("ignore")

print("="*50)
print("SATELLITE VECM: TEOREMA DEL PRIMO DOMINO (OSR => S1A)")
print("="*50)

# =============================================================================
# 1. INIZIALIZZAZIONE FASE ZERO: IMPORTAZIONE RAW E TRASFORMAZIONE MATRICIALE
# =============================================================================
# Struttura: prima cella nome variabile, a seguire i valori. Separatore decimale: virgola.
dati_raw = """Anno 1960 1961 1962 1963 1964 1965 1966 1967 1968 1969 1970 1971 1972 1973 1974 1975 1976 1977 1978 1979 1980 1981 1982 1983 1984 1985 1986 1987 1988 1989 1990 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009
OS 46289 79127 181732 91158 104709 55943 115788 68548 73918 302597 146212 103590 136480 163935 136267 181381 131711 78767 49032 164914 75214 42802 114889 82626 31786 9969 36742 20147 17086 21001 36269 11573 5605 8796 7651 6365 13509 8149 3807 6364 6114 7038 6104 5730 4890 6347 3883 6508 5059 2601
S1A	0,152746442	0,151185205	0,129232772	0,082479905	0,068288409	0,08258608	0,095054389	0,087025738	0,089324553	0,094316927	0,06031078	0,014157516	0,003621941	0,012052012	0,028691553	-1,76771E-15	0,016510863	0,007203887	0,025629395	0,041066315	0,051298789	0,04094195	0,049146599	0,055014171	0,07687899	0,081035111	0,098745046	0,099768447	0,102145	0,103725538	0,087614863	0,08076154	0,082122577	0,085557303	0,110433328	0,134048531	0,129142007	0,122226995	0,130228167	0,129588654	0,141573711	0,143729023	0,136868153	0,133383134	0,134875141	0,123387001	0,116598075	0,117498674	0,109090164	0,090664465
"""

# Lettura raw, trasposizione e conversione decimali
df_t = pd.read_csv(io.StringIO(dati_raw), sep=r'\s+', header=None, index_col=0)

# Trasponiamo: le variabili tornano colonne, il tempo scorre sulle righe
df_raw = df_t.T

# Pulizia dell'indice vettoriale e rimozione del nome colonne
df_raw.columns.name = None

# Convertiamo TUTTE le colonne per pulire stringhe/virgole e renderle numeriche
for col in df_raw.columns:
    df_raw[col] = df_raw[col].astype(str).str.replace(',', '.').astype(float)

# Castiamo 'Anno' a intero puro e lo impostiamo come spina dorsale (indice)
df_raw['Anno'] = df_raw['Anno'].astype(int)
df_raw.set_index('Anno', inplace=True)

# Creazione dell'Indice Relativo di Intensità (OSR)
df_raw['OSR'] = df_raw['OS'] / 302597.0
df = df_raw[['OSR', 'S1A']].copy()

print("\n[OK] Trasformazione vettoriale completata. Dati caricati in riga.")
print(f"Variabili operative caricate: {list(df.columns)}")

# ==========================================
# 2. FASE 1: PASSAPORTO I(1) CON SOGLIA 10%
# ==========================================
print("\n[--- Sotto-Fase 1: Check qualifica I(1) ---]")
def adf_report(series, name):
    res_livelli = adfuller(series.dropna(), autolag='AIC')
    pval_lvl = res_livelli[1]
    res_diff = adfuller(series.diff().dropna(), autolag='AIC')
    pval_diff = res_diff[1]
    
    print(f"\nVariabile: {name}")
    print(f"  - Livelli (p-value):     {pval_lvl:.4f} -> ", end="")
    pass_lvl = pval_lvl > 0.10
    print("[PASS] Radice unitaria" if pass_lvl else "[FAIL] Stazionaria")
    
    print(f"  - Diff. Prime (p-value): {pval_diff:.4f} -> ", end="")
    pass_diff = pval_diff < 0.10
    print("[PASS] Stazionaria in diff" if pass_diff else "[FAIL] Radice residua")
    
    if pass_lvl and pass_diff:
        print(f"  => ESITO: {name} è I(1).")
        return True
    return False

if not (adf_report(df['OSR'], 'OSR') and adf_report(df['S1A'], 'S1A')):
    print("\n[!] ATTENZIONE: Requisiti I(1) non superati. Il modello potrebbe essere instabile.")

# ==========================================
# 3. FASE 2: COINTEGRAZIONE JOHANSEN
# ==========================================
print("\n" + "="*50)
print("FASE 2: TEST DI COINTEGRAZIONE (JOHANSEN)")
print("="*50)
res_johansen = coint_johansen(df, det_order=0, k_ar_diff=1)
tr_stat = res_johansen.lr1[0]
crit_val_95 = res_johansen.cvt[0, 1]

print(f"Statistica Traccia (r=0): {tr_stat:.2f}")
print(f"Valore Critico (95%)    : {crit_val_95:.2f}")

if tr_stat > crit_val_95:
    print("=> [PASS] Trovato almeno 1 vettore di cointegrazione. Equilibrio confermato.")
else:
    print("=> [FAIL] Nessuna cointegrazione rilevata statisticamente al 95%.")

# ==========================================
# 4. FASE 3: VECM ASIMMETRICO
# ==========================================
print("\n" + "="*50)
print("FASE 3: DINAMICA VECM E ALPHA ASIMMETRICO")
print("="*50)

# Selezione lag ottimale
lag_res = select_order(df, maxlags=4, deterministic="co")
p_opt = lag_res.aic
k_ar_diff_ottimale = max(0, p_opt - 1) if p_opt > 0 else 1

modello_vecm = VECM(df, deterministic="co", k_ar_diff=k_ar_diff_ottimale, coint_rank=1)
risultati_vecm = modello_vecm.fit()

# Estrazione coefficienti Error Correction (Alpha)
alpha_OSR = risultati_vecm.alpha[0, 0]
alpha_S1A = risultati_vecm.alpha[1, 0]

print(f"  - Ritardo ottimale (p_opt) : {k_ar_diff_ottimale + 1}")
print(f"  - Alpha [OSR, indipendente]: {alpha_OSR:.4f}")
print(f"  - Alpha [S1A, dipendente]  : {alpha_S1A:.4f}")

if abs(alpha_S1A) > abs(alpha_OSR):
    print("\n=> [CONFERMA STRUTTURALE] S1A agisce come ammortizzatore e assorbe gli shock di OSR.")

# ==========================================
# 5. FASE 4: IMPULSE RESPONSE FUNCTION (IRF)
# ==========================================
print("\n" + "="*50)
print("FASE 4: ANALISI IRF (Impulso OSR -> Risposta S1A)")
print("="*50)

periodi = 15
irf = risultati_vecm.irf(periods=periodi)

# Estrazione della risposta numerica di S1A a uno shock su OSR
risposta_S1A = irf.irfs[:, 1, 0]

istantaneo = risposta_S1A[1]
picco_max = max(risposta_S1A, key=abs)
anno_picco = np.argmax(np.abs(risposta_S1A))

print(f"  - Moltiplicatore Dinamico Istantaneo (Anno t+1) : {istantaneo:.4f}")
print(f"  - Picco Massimo della Risposta d'Impatto        : {picco_max:.4f} (Raggiunto all'Anno t+{anno_picco})")

# Grafico
plt.figure(figsize=(10, 6))
plt.plot(range(periodi+1), risposta_S1A, color='darkred', linewidth=2, marker='o')
plt.axhline(0, color='black', linestyle='--')
plt.title("Risposta di S1A a uno shock strutturale in OSR", fontsize=14)
plt.xlabel("Anni dallo shock", fontsize=12)
plt.ylabel("Variazione S1A", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

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
    resid_S1A = risultati_vecm.resid[:, 1]
    
    # Eseguiamo l'ARCH test (H0: I residui sono omoschedastici / varianza costante)
    arch_test_res = het_arch(resid_S1A, nlags=5)
    
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

# ==========================================
# 6BIS. FASE 5: TEST FORMALI DI CAUSALITÀ
# ==========================================
print("\n" + "="*50)
print("FASE 5: TEST DI CAUSALITÀ (LUNGO E BREVE PERIODO)")
print("="*50)

# -------------------------------------------------------------
# A) CAUSALITÀ DI LUNGO PERIODO: significatività di Alpha
# -------------------------------------------------------------
# H0: alpha = 0  =>  la variabile NON si aggiusta all'equilibrio
#                     (nessuna causalità di lungo periodo verso di essa)
print("\n[ A. Causalità di Lungo Periodo (significatività di Alpha) ]")

alpha_vals    = risultati_vecm.alpha.flatten()
alpha_se      = risultati_vecm.stderr_alpha.flatten()
alpha_tvals   = risultati_vecm.tvalues_alpha.flatten()
alpha_pvals   = risultati_vecm.pvalues_alpha.flatten()

nomi_eq = df.columns.tolist()  # ['OSR', 'S1A']

for i, nome in enumerate(nomi_eq):
    sig = "[PASS] Significativo" if alpha_pvals[i] < 0.05 else "[FAIL] Non significativo"
    print(f"  - Alpha[{nome}]: {alpha_vals[i]:.4f}  "
          f"(SE={alpha_se[i]:.4f}, t={alpha_tvals[i]:.3f}, p={alpha_pvals[i]:.4f}) -> {sig}")

if alpha_pvals[1] < 0.05:
    print("\n  => OSR esercita causalità di lungo periodo su S1A")
    print("     (S1A si corregge significativamente in risposta agli scostamenti dall'equilibrio).")
else:
    print("\n  => Non si rigetta H0: nessuna evidenza di causalità di lungo periodo OSR -> S1A.")

# -------------------------------------------------------------
# B) CAUSALITÀ DI BREVE PERIODO: Wald test sui lag di Gamma
# -------------------------------------------------------------
# H0: tutti i coefficienti di breve periodo (Gamma) di OSR
#     nell'equazione di S1A sono congiuntamente zero
print("\n[ B. Causalità di Breve Periodo (Granger, via Gamma) ]")

try:
    # Indici dei parametri Gamma relativi ai lag di OSR (colonna 0)
    # nell'equazione di S1A (riga 1) — dipende dal numero di lag k_ar_diff
    k = k_ar_diff_ottimale
    n_vars = df.shape[1]

    if k > 0:
        # Nome dei parametri nel modello: 'L1.OSR', 'L1.S1A', 'L2.OSR', ...
        param_names = risultati_vecm.model.exog_names if hasattr(risultati_vecm.model, 'exog_names') else None

        # Estrazione diretta via gamma matrix: shape (n_vars, n_vars*k)
        gamma = risultati_vecm.gamma          # coefficienti
        gamma_se = risultati_vecm.stderr_gamma
        gamma_p = risultati_vecm.pvalues_gamma

        # Equazione di S1A = riga 1; colonne relative a OSR = 0, n_vars, 2*n_vars, ...
        idx_osr_lags = [0 + lag*n_vars for lag in range(k)]
        pvals_osr_su_s1a = gamma_p[1, idx_osr_lags]

        print(f"  - P-value dei lag di OSR nell'equazione di S1A: {np.round(pvals_osr_su_s1a, 4)}")

        if np.any(pvals_osr_su_s1a < 0.05):
            print("  => [PASS] Almeno un lag di OSR è significativo: evidenza di Granger-causalità")
            print("     di breve periodo da OSR verso S1A.")
        else:
            print("  => [FAIL] Nessun lag di OSR significativo nell'equazione di S1A.")
    else:
        print("  k_ar_diff = 0: nessun termine di breve periodo (Gamma) da testare.")
        print("  La causalità, se presente, è interamente di lungo periodo (via Alpha).")

except Exception as e:
    print(f"  Impossibile estrarre/testare i coefficienti Gamma: {e}")

print("\n" + "="*50)
