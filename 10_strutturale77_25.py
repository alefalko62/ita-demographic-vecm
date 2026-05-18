import numpy as np
from scipy.stats import linregress, f, pearsonr
from scipy.optimize import minimize

# ==========================================
# FUNZIONI HELPER
# ==========================================
def parse_dati(raw_string):
    pulita = raw_string.replace(',', '.')
    return np.array([float(x) for x in pulita.split()])

def formatta_ita(val, dec=4):
    if isinstance(val, (int, np.integer)): return str(val)
    return f"{val:.{dec}f}".replace('.', ',')

# ==========================================
# 1. INSERIMENTO DATI (COPIA E INCOLLA QUI)
# ==========================================
Anni_raw = """
1977 1978 1979 1980 1981 1982 1983 1984 1985 1986 1987 1988 1989 1990 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
"""

S8_raw = """
0,007203887	0,025629395	0,041066315	0,051298789	0,04094195	0,049146599	0,055014171	0,07687899	0,081035111	0,098745046	0,099768447	0,102145	0,103725538	0,087614863	0,08076154	0,082122577	0,085557303	0,110433328	0,134048531	0,129142007	0,122226995	0,130228167	0,129588654	0,141573711	0,143729023	0,136868153	0,133383134	0,134875141	0,123387001	0,116598075	0,117498674	0,109090164	0,090664465	0,092775517	0,099571953	0,090755122	0,095565916	0,09780629	0,094704396	0,102485274	0,097894409	0,09069451	0,08437417	0,079224317	0,086330747	0,105780166	0,118910662	0,096868826	0,086414986
"""

S3_raw = """
0,065989848	0,122994652	0,193181818	0,25	0,3125	0,3125	0,363636364	0,418918919	0,448275862	0,532846715	0,555555556	0,52173913	0,555555556	0,544117647	0,590909091	0,590909091	0,666666667	0,721311475	0,764705882	0,721311475	0,707317073	0,73553719	0,707317073	0,666666667	0,68	0,653543307	0,627906977	0,567164179	0,578947368	0,532846715	0,510791367	0,458333333	0,458333333	0,458333333	0,478873239	0,478873239	0,510791367	0,52173913	0,544117647	0,544117647	0,567164179	0,603053435	0,653543307	0,693548387	0,68	0,693548387	0,75	0,779661017	0,842105263	
"""

anni = parse_dati(Anni_raw)
S8 = parse_dati(S8_raw)
S3 = parse_dati(S3_raw)

print("="*80)
print(" REPORT ANALISI STRUTTURALE - FORMATO ESPORTAZIONE EXCEL")
print(" Copia i blocchi sottostanti e usa 'Testo in colonne' con separatore ';'")
print("="*80)

# ---------------------------------------------------------
# IMPOSTAZIONE FORZATA BREAKPOINT (2018) E CALCOLI
# ---------------------------------------------------------
n = len(anni)
anno_x = 2018

# Trova l'indice esatto del 2018
if anno_x in anni:
    idx_x = list(anni).index(anno_x)
else:
    print(f"\nERRORE: L'anno {anno_x} non è presente nei dati.")
    exit()

anni_pre, S8_pre, S3_pre = anni[:idx_x], S8[:idx_x], S3[:idx_x]
anni_post, S8_post, S3_post = anni[idx_x:], S8[idx_x:], S3[idx_x:]

# Calcolo Statistica F sul 2018 (per darti comunque il valore di robustezza)
def calcola_rss(x, y):
    if len(x) < 3: return float('inf')
    slope, intercept, _, _, _ = linregress(x, y)
    return np.sum((y - (intercept + slope * x))**2)

rss_pooled = calcola_rss(S8, S3)
rss_unrestricted = calcola_rss(S8_pre, S3_pre) + calcola_rss(S8_post, S3_post)
f_stat_2018 = 0
p_val_2018 = 1
if rss_unrestricted > 0:
    f_stat_2018 = ((rss_pooled - rss_unrestricted) / 2) / (rss_unrestricted / (n - 4))
    p_val_2018 = 1 - f.cdf(f_stat_2018, 2, n - 4)

corr_pre, _ = pearsonr(S8_pre, S3_pre)

def l1_loss(c, x, y): return np.sum(np.abs(y - (c * x)))
res = minimize(l1_loss, x0=1.0, args=(S8_pre, S3_pre), method='Nelder-Mead')
molt_L1 = res.x[0]

S3_proiettato_post = S8_post * molt_L1
errore_assoluto_post = S3_proiettato_post - S3_post

# ---------------------------------------------------------
# STAMPE FORMATTATE ORIZZONTALMENTE PER EXCEL
# ---------------------------------------------------------

print("\n1) BREAKPOINT E DATI STORICI (FORZATO SUL TIPPING POINT):")
print("Anno_X ; Statistica_F ; p-value ; Corr_Pearson_Pre ; Molt_L1_Pre")
print(f"{int(anno_x)} ; {formatta_ita(f_stat_2018,2)} ; {formatta_ita(p_val_2018,6)} ; {formatta_ita(corr_pre)} ; {formatta_ita(molt_L1)}")

print("\n4) PROIEZIONE (TENSIONE IDEALE) E ERRORE POST-2018:")
print("Anno ; " + " ; ".join([str(int(a)) for a in anni_post]))
print("Reale_S3 ; " + " ; ".join([formatta_ita(x,2) for x in S3_post]))
print("Teorico_Elastico ; " + " ; ".join([formatta_ita(x,2) for x in S3_proiettato_post]))
print("Diff_Strappo ; " + " ; ".join([formatta_ita(x,2) for x in errore_assoluto_post]))

print("\n5) TEST FINESTRE ESPANSIVE (TUTTI i quinquenni pre-rottura):")
if idx_x >= 15:
    train_size = 10
    while train_size <= idx_x - 5:
        x_train, y_train = S8[:train_size], S3[:train_size]
        c_iter = minimize(l1_loss, x0=1.0, args=(x_train, y_train), method='Nelder-Mead').x[0]
        test_end = min(train_size + 5, idx_x)
        anni_test = anni[train_size:test_end]
        x_test, y_test = S8[train_size:test_end], S3[train_size:test_end]
        y_proj = x_test * c_iter
        errore_iter = y_proj - y_test
        
        print(f"\n[Train {int(anni[0])}-{int(anni[train_size-1])} | Moltiplicatore: {formatta_ita(c_iter)}]")
        print("Anno ; " + " ; ".join([str(int(a)) for a in anni_test]))
        print("Reale ; " + " ; ".join([formatta_ita(x,2) for x in y_test]))
        print("Previsto ; " + " ; ".join([formatta_ita(x,2) for x in y_proj]))
        print("Errore ; " + " ; ".join([formatta_ita(x,2) for x in errore_iter]))
        
        train_size += 5

print("\n6) CODE: INTERPOLAZIONE POST-2018:")
if len(anni_post) >= 3:
    slope_S8, _, _, _, _ = linregress(anni_post, S8_post)
    slope_S3, _, _, _, _ = linregress(anni_post, S3_post)
    print("Pendenza_S8 ; Pendenza_S3")
    print(f"{formatta_ita(slope_S8,4)} ; {formatta_ita(slope_S3,4)}")

print("\n7) ROLLING PEARSON (10 e 15 ANNI) - ELETTROCARDIOGRAMMA POST-ROTTURA:")
for window in [10, 15]:
    if n >= window:
        corrs, anni_end = [], []
        for i in range(n - window + 1):
            x_win, y_win = S8[i:i+window], S3[i:i+window]
            c = 0.0 if np.std(x_win)==0 or np.std(y_win)==0 else pearsonr(x_win, y_win)[0]
            corrs.append(c)
            anni_end.append(int(anni[i+window-1]))
        
        corrs, anni_end = np.array(corrs), np.array(anni_end)
        mask_post = anni_end >= anno_x
        
        print(f"\n[Finestra {window} anni]")
        print("Anno ; " + " ; ".join([str(a) for a in anni_end[mask_post]]))
        print("Pearson ; " + " ; ".join([formatta_ita(c,4) for c in corrs[mask_post]]))

# ==========================================
# NUOVA SEZIONE 8: PROIEZIONE 2019-2025
# ==========================================
print("\n8) ESTENSIONE TRAIETTORIA DI COLLASSO (2019 - 2025):")
if len(anni_post) >= 3:
    slope_8, int_8, _, _, _ = linregress(anni_post, S8_post)
    slope_3, int_3, _, _, _ = linregress(anni_post, S3_post)
    
    anni_futuri = np.arange(2019, 2026)
    S8_futuro = int_8 + slope_8 * anni_futuri
    S3_futuro = int_3 + slope_3 * anni_futuri
    
    print("Anno_Proiezione ; " + " ; ".join([str(int(a)) for a in anni_futuri]))
    print("Trend_S8 (Economia) ; " + " ; ".join([formatta_ita(x,4) for x in S8_futuro]))
    print("Trend_S3 (Deficit) ; " + " ; ".join([formatta_ita(x,4) for x in S3_futuro]))

print("\n" + "="*80)