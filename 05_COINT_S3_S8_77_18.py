import numpy as np
from scipy.stats import pearsonr, linregress
try:
    from statsmodels.tsa.stattools import adfuller, coint
    STATMODELS_INSTALLED = True
except ImportError:
    STATMODELS_INSTALLED = False

# ==========================================
# FUNZIONE HELPER PER EXCEL ITALIANO
# ==========================================
def formatta_ita(val, dec=4):
    if isinstance(val, (int, np.integer)): return str(val)
    return f"{val:.{dec}f}".replace('.', ',')

# ==========================================
# INSERISCI QUI I TUOI DATI (es. 1977-2018)
# ==========================================
# Sostituisci questi array con i tuoi dati REALI di S8 e S3
anni = np.array([1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]) # 42 anni
# DATI SIMULATI PER ESEMPIO (sostituiscili con i tuoi!)
S8 = np.array([0.007203887, 0.025629395, 0.041066315, 0.051298789, 0.04094195, 0.049146599, 0.055014171, 0.07687899, 0.081035111, 0.098745046, 0.099768447, 0.102145, 0.103725538, 0.087614863, 0.08076154, 0.082122577, 0.085557303, 0.110433328, 0.134048531, 0.129142007, 0.122226995, 0.130228167, 0.129588654, 0.141573711, 0.143729023, 0.136868153, 0.133383134, 0.134875141, 0.123387001, 0.116598075, 0.117498674, 0.109090164, 0.090664465, 0.092775517, 0.099571953, 0.090755122, 0.095565916, 0.09780629, 0.094704396, 0.102485274, 0.097894409, 0.09069451])  
S3 = np.array([0.065989848, 0.122994652, 0.193181818, 0.25, 0.3125, 0.3125, 0.363636364, 0.418918919, 0.448275862, 0.532846715, 0.555555556, 0.52173913, 0.555555556, 0.544117647, 0.590909091, 0.590909091, 0.666666667, 0.721311475, 0.764705882, 0.721311475, 0.707317073, 0.73553719, 0.707317073, 0.666666667, 0.68, 0.653543307, 0.627906977, 0.567164179, 0.578947368, 0.532846715, 0.510791367, 0.458333333, 0.458333333, 0.458333333, 0.478873239, 0.478873239, 0.510791367, 0.52173913, 0.544117647, 0.544117647, 0.567164179, 0.603053435])
n = len(anni)

print("="*80)
print(" TRIBUNALE ECONOMETRICO: TEST DI ROBUSTEZZA S8 vs S3")
print("="*80)

# ---------------------------------------------------------
# 1) CORRELAZIONE BASE (I Livelli)
# ---------------------------------------------------------
corr_base, pval_base = pearsonr(S8, S3)
print("\n[PROVA 1] CORRELAZIONE SUI LIVELLI (Il tuo 0.92 originale)")
print(f" -> Pearson (r): {formatta_ita(corr_base, 4)}")
print(f" -> Indice di Determinazione (R^2): {formatta_ita(corr_base**2, 4)}")
print(" (Se R^2 è alto, S8 spiega la maggior parte della varianza di S3).")

# ---------------------------------------------------------
# 2) CORRELAZIONE SULLE DIFFERENZE PRIME (Δ)
# ---------------------------------------------------------
# Rimuoviamo il trend di lungo periodo calcolando (Anno_X - Anno_X-1)
dS8 = np.diff(S8)
dS3 = np.diff(S3)
corr_diff, pval_diff = pearsonr(dS8, dS3)

print("\n[PROVA 2] TEST ANTI-CORRELAZIONE SPURIA (Differenze Prime)")
print(" (Correlando gli 'scatti' annuali ΔS8 e ΔS3, depurati dal trend storico)")
print(f" -> Pearson (r) sulle differenze: {formatta_ita(corr_diff, 4)}")
print(f" -> p-value: {formatta_ita(pval_diff, 6)}")
if pval_diff < 0.05:
    print(" -> VERDETTO: SUPERATO. La correlazione NON è solo frutto del tempo che passa!")
else:
    print(" -> VERDETTO: ATTENZIONE. Senza il trend temporale, il legame annuo è debole.")

# ---------------------------------------------------------
# 3) LAG TEMPORALI (Test di Causalità Temporale)
# ---------------------------------------------------------
print("\n[PROVA 3] LAG TEMPORALI (L'Effetto Eco)")
print(" (Se S8 si muove oggi, dopo quanti anni risponde S3?)")
print(" Lag ; Pearson(r) ; R^2")

# Lag 0 (Sincrono)
print(f"  0  ; {formatta_ita(corr_base,4)} ; {formatta_ita(corr_base**2,4)}")

miglior_lag = 0
miglior_r = corr_base

for lag in range(1, 6): # Testiamo fino a 5 anni di ritardo
    S8_lag = S8[:-lag] # S8 di "ieri"
    S3_fut = S3[lag:]  # S3 di "domani"
    r_lag, _ = pearsonr(S8_lag, S3_fut)
    print(f" +{lag}  ; {formatta_ita(r_lag,4)} ; {formatta_ita(r_lag**2,4)}")
    
    if abs(r_lag) > abs(miglior_r):
        miglior_r = r_lag
        miglior_lag = lag

print(f" -> VERDETTO: La sovrapposizione migliore si ha considerando l'effetto di S8 su S3 dopo {miglior_lag} anni.")

# ---------------------------------------------------------
# 4) TEST DI COINTEGRAZIONE E STAZIONARIETÀ
# ---------------------------------------------------------
print("\n[PROVA 4] TEST DI COINTEGRAZIONE (L'Elastico Invisibile)")
if not STATMODELS_INSTALLED:
    print(" [!] ATTENZIONE: Libreria 'statsmodels' non trovata.")
    print("     Per fare questo test cruciale, installa statsmodels su Thonny.")
    print("     (Strumenti -> Gestisci pacchetti -> cerca 'statsmodels')")
else:
    # 4a) Test ADF sulle singole serie
    adf_S8 = adfuller(S8)
    adf_S3 = adfuller(S3)
    
    print(" - Controllo Stazionarietà (Augmented Dickey-Fuller):")
    print(f"   S8 p-value: {formatta_ita(adf_S8[1],4)} (Se > 0.05, la serie ha un trend, è I(1))")
    print(f"   S3 p-value: {formatta_ita(adf_S3[1],4)} (Se > 0.05, la serie ha un trend, è I(1))")
    
    # 4b) Test Engle-Granger di Cointegrazione
    # Attenzione: l'ordine conta. coint(y, x) testa se y e x sono cointegrati.
    score, pval_coint, _ = coint(S3, S8, autolag='AIC')
    
    print("\n - Test di Cointegrazione di Engle-Granger:")
    print(f"   Statistica: {formatta_ita(score,4)}")
    print(f"   p-value:    {formatta_ita(pval_coint,6)}")
    
    if pval_coint < 0.05:
        print(" -> VERDETTO DEFINITIVO: SUPERATO AL 100%!")
        print("    S8 e S3 SONO COINTEGRATE. Esiste un legame strutturale di lungo periodo.")
        print("    Nessuno potrà più dirti che è una correlazione spuria.")
    else:
        print(" -> VERDETTO: Le serie non risultano statisticamente cointegrate.")
        print("    Il legame è fortissimo (Pearson), ma matematicamente i loro trend")
        print("    potrebbero essere spinti da una terza variabile esterna.")

print("\n" + "="*80)