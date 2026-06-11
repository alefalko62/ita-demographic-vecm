import numpy as np
from scipy.stats import f, pearsonr, linregress

# ==========================================
# FUNZIONE HELPER: Calcolo della Somma dei Quadrati dei Residui (SSR)
# ==========================================
def calcola_ssr(x, y):
    # Esegue una regressione lineare e calcola quanto i dati reali 
    # si discostano dalla linea teorica
    slope, intercept, _, _, _ = linregress(x, y)
    previsioni = intercept + slope * x
    residui = y - previsioni
    return np.sum(residui**2)

def formatta_ita(val, dec=4):
    return f"{val:.{dec}f}".replace('.', ',')

# ==========================================
# INSERISCI QUI I DATI DELLA SERIE COMPLETA (es. 1960-2023)
# ==========================================
# Sostituisci questi array con le tue serie storiche integrali

anni_full = np.array([1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]) # Sostituisci con i tuoi anni reali
S8_full = np.array([0.152746442, 0.151185205, 0.129232772, 0.082479905, 0.068288409, 0.08258608, 0.095054389, 0.087025738, 0.089324553, 0.094316927, 0.06031078, 0.014157516, 0.003621941, 0.012052012, 0.028691553, -1.8782E-15, 0.016510863, 0.007203887, 0.025629395, 0.041066315, 0.051298789, 0.04094195, 0.049146599, 0.055014171, 0.07687899, 0.081035111, 0.098745046, 0.099768447, 0.102145, 0.103725538, 0.087614863, 0.08076154, 0.082122577, 0.085557303, 0.110433328, 0.134048531, 0.129142007, 0.122226995, 0.130228167, 0.129588654, 0.141573711, 0.143729023, 0.136868153, 0.133383134, 0.134875141, 0.123387001, 0.116598075, 0.117498674, 0.109090164, 0.090664465, 0.092775517, 0.099571953, 0.090755122, 0.095565916, 0.09780629, 0.094704396, 0.102485274, 0.097894409, 0.09069451, 0.08437417, 0.079224317, 0.086330747, 0.105780166, 0.118910662, 0.096868826, 0.086414986]) # <-- Inserisci qui tutta la serie storica di S8
S3_full = np.array([-0.128630705, -0.128630705, -0.146341463, -0.176470588, -0.222222222, -0.210526316, -0.198473282, -0.169960474, -0.156626506, -0.163346614, -0.132231405, -0.128630705, -0.110169492, -0.102564103, -0.098712446, -0.045454545, -0.004739336, 0.065989848, 0.122994652, 0.193181818, 0.25, 0.3125, 0.3125, 0.363636364, 0.418918919, 0.448275862, 0.532846715, 0.555555556, 0.52173913, 0.555555556, 0.544117647, 0.590909091, 0.590909091, 0.666666667, 0.721311475, 0.764705882, 0.721311475, 0.707317073, 0.73553719, 0.707317073, 0.666666667, 0.68, 0.653543307, 0.627906977, 0.567164179, 0.578947368, 0.532846715, 0.510791367, 0.458333333, 0.458333333, 0.458333333, 0.478873239, 0.478873239, 0.510791367, 0.52173913, 0.544117647, 0.544117647, 0.567164179, 0.603053435, 0.653543307, 0.693548387, 0.68, 0.693548387, 0.75, 0.779661017, 0.842105263]) # <-- Inserisci qui tutta la serie storica di S3

# --- BLOCCO DI SICUREZZA SE I DATI SONO VUOTI ---
# Genero una rottura strutturale finta nel 2018 per farti vedere l'output.
# CANCELLA QUESTE RIGHE QUANDO INSERISCI I TUOI DATI VERI!
# ------------------------------------------------

print("="*85)
print(" CACCIATORE DI BREAKPOINT (TEST DI CHOW AUTOMATIZZATO)")
print(" Scansione della serie storica alla ricerca della frattura strutturale")
print("="*85)

N_totale = len(anni_full)
ssr_totale = calcola_ssr(S8_full, S3_full)

best_F = 0
anno_breakpoint = None
indice_breakpoint = 0

# Buffer di sicurezza: non testiamo i primissimi o gli ultimissimi 5 anni 
# per avere un campione statisticamente valido in entrambi i tronconi.
buffer = 5

for i in range(buffer, N_totale - buffer):
    anno_test = anni_full[i]
    
    # Troncone 1 (Fino all'anno di test incluso)
    S8_pre = S8_full[:i+1]
    S3_pre = S3_full[:i+1]
    ssr_pre = calcola_ssr(S8_pre, S3_pre)
    
    # Troncone 2 (Dall'anno successivo in poi)
    S8_post = S8_full[i+1:]
    S3_post = S3_full[i+1:]
    ssr_post = calcola_ssr(S8_post, S3_post)
    
    # Parametri per la formula di Chow
    k = 2 # Numero di parametri della regressione (Intercetta + Pendenza)
    
    # Calcolo del Numeratore e Denominatore per la Statistica F
    numeratore = (ssr_totale - (ssr_pre + ssr_post)) / k
    denominatore = (ssr_pre + ssr_post) / (N_totale - 2 * k)
    
    F_stat = numeratore / denominatore
    
    # Cerchiamo l'anno in cui la Statistica F (l'anomalia) tocca il suo massimo assoluto
    if F_stat > best_F:
        best_F = F_stat
        anno_breakpoint = anno_test
        indice_breakpoint = i

# Calcolo p-value per la Statistica F trovata
p_value = f.sf(best_F, k, N_totale - 2*k)

print(f"\n[RISULTATO DELLA SCANSIONE STORICA]")
print(f" -> L'algoritmo ha individuato il PUNTO DI ROTTURA nell'anno: {anno_breakpoint}")
print(f" -> Statistica F (Picco dell'anomalia): {formatta_ita(best_F, 2)}")
print(f" -> p-value: {formatta_ita(p_value, 6)}")

if p_value < 0.05:
    print(" -> VERDETTO: La rottura è STRUTTURALE E STATISTICAMENTE SIGNIFICATIVA.")
    print(f"    L'equazione economica che governava il sistema prima del {anno_breakpoint}")
    print("    non è più valida per descrivere gli anni successivi.")
else:
    print(" -> VERDETTO: Nessuna rottura strutturale significativa rilevata.")

print("\n[ANALISI DEI DUE TRONCONI STORICI]")
# Ricalcoliamo le metriche usando il breakpoint trovato
S8_epoca1 = S8_full[:indice_breakpoint+1]
S3_epoca1 = S3_full[:indice_breakpoint+1]
corr_epoca1, _ = pearsonr(S8_epoca1, S3_epoca1)

S8_epoca2 = S8_full[indice_breakpoint+1:]
S3_epoca2 = S3_full[indice_breakpoint+1:]
corr_epoca2, _ = pearsonr(S8_epoca2, S3_epoca2)

print(f" FASE 1 ({anni_full[0]} - {anno_breakpoint}): L'Era dell'Accumulazione Standard")
print(f" -> Correlazione S8-S3: {formatta_ita(corr_epoca1, 4)}")

print(f"\n FASE 2 ({anni_full[indice_breakpoint+1]} - {anni_full[-1]}): L'Era del Collasso / Accelerazione")
print(f" -> Correlazione S8-S3: {formatta_ita(corr_epoca2, 4)}")

print("\n" + "="*85)
