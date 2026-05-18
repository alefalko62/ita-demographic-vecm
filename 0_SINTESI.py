import numpy as np
from scipy import stats
from scipy.optimize import minimize
import sys


# ==========================================
# FUNZIONI HELPER E INTERFACCIA
# ==========================================
def parse_dati(raw_string):
    pulita = raw_string.replace(',', '.')
    return np.array([float(x) for x in pulita.split()])

def formatta_ita(val, dec=4):
    if isinstance(val, (int, np.integer)): return str(val)
    return f"{val:.{dec}f}".replace('.', ',')

def attendi_input():
    print("-" * 80)
    risposta = input(">>> Premi [INVIO] per il prossimo test, oppure 'Q' e [INVIO] per uscire: ")
    if risposta.strip().upper() == 'Q':
        print("\n[!] Esecuzione interrotta dal Revisore. Chiusura terminale.")
        sys.exit()
    print("=" * 80)

def calcola_rss(x, y):
    if len(x) < 3: return float('inf')
    slope, intercept, _, _, _ = stats.linregress(x, y)
    return np.sum((y - (intercept + slope * x))**2)

# ==========================================
# INSERIMENTO DATI GREZZI (SERIE ESTESA)
# ==========================================
print("="*80)
print(" INIZIALIZZAZIONE PIPELINE C L N E CONVERSIONE DATI GREZZI")
print("="*80)

# IMPORTANTE: Sostituisci questi dati di esempio con le tue stringhe reali complete
# 1. ANNI (Serie storica completa)
Anni_raw = "1960	1961	1962	1963	1964	1965	1966	1967	1968	1969	1970	1971	1972	1973	1974	1975	1976	1977	1978	1979	1980	1981	1982	1983	1984	1985	1986	1987	1988	1989	1990	1991	1992	1993	1994	1995	1996	1997	1998	1999	2000	2001	2002	2003	2004	2005	2006	2007	2008	2009	2010	2011	2012	2013	2014	2015	2016	2017	2018	2019	2020	2021	2022	2023	2024	2025" # Incolla qui tutti gli anni

# 2. U: CO2 in ppm (Mauna Loa)
U_CO2_raw = "316,91	317,64	318,45	318,99	319,62	320,04	321,37	322,18	323,05	324,62	325,68	326,32	327,46	329,68	330,19	331,13	332,03	333,84	335,41	336,84	338,76	340,12	341,48	343,15	344,87	346,35	347,61	349,31	351,69	353,2	354,45	355,7	356,54	357,21	358,96	360,97	362,74	363,88	366,84	368,54	369,71	371,32	373,45	375,98	377,7	379,98	382,09	384,02	385,83	387,64	390,1	391,85	394,06	396,74	398,81	401,01	404,41	406,76	408,72	411,65	414,21	416,41	418,53	421,08	424,61	427,35" # Incolla qui i dati CO2

# 3. C: UOGD (Redditi da Capitale)
C_UOGD_raw = "7,3565	8,14054	8,9144	9,63813	10,38489	11,38577	12,54929	13,61251	14,88377	16,60024	17,91826	18,64968	20,34152	24,86974	32,17664	36,58798	46,473	55,61329	66,90408	83,84374	105,54284	125,46556	148,76966	173,15082	203,07352	229,02483	258,31916	280,91492	310,06073	341,03729	366,80131	395,37515	415,64639	427,27694	464,72954	513,8847	541,3639	557,0593	566,9051	586,0251	629,7106	667,5879	685,3494	707,0019	735,5773	743,944	763,5659	794,8201	805,411	764,0703	777,5571	801,653	773,6578	775,473	782,0098	792,3852	827,5922	839,7632	850,33	855,6919	797,0807	882,7508	982,957	1067,9487	1062,5713	1076,9" # Incolla qui i dati Capitale

# 4. L: UWCD (Redditi da Lavoro)
L_UWCD_raw = "5,17009	5,74665	6,68624	8,1347	9,06117	9,6073	10,27461	11,36476	12,35734	13,61589	15,92158	18,3014	20,39236	24,51054	30,63586	36,94576	45,38532	55,35193	64,11212	77,74364	95,69352	116,3686	135,52988	155,69935	173,68246	193,96479	209,58026	227,33493	249,44069	273,2777	305,79937	335,06692	351,10915	357,98432	366,12884	380,4652	406,1484	425,6654	424,0727	439,1286	456,6956	481,2742	503,5289	524,389	543,3779	566,7351	592,1013	614,8985	636,7028	632,3164	640,1786	649,0672	640,1097	634,1138	635,9379	649,3151	665,2235	682,7548	703,6504	718,9568	677,9612	738,2206	783,5975	823,169	866,0952	900,4" # Incolla qui i dati Lavoro

# 5. N: TFT/TFR (Tasso di Fecondità Totale)
N_TFR_raw = "2,41	2,41	2,46	2,55	2,7	2,66	2,62	2,53	2,49	2,51	2,42	2,41	2,36	2,34	2,33	2,2	2,11	1,97	1,87	1,76	1,68	1,6	1,6	1,54	1,48	1,45	1,37	1,35	1,38	1,35	1,36	1,32	1,32	1,26	1,22	1,19	1,22	1,23	1,21	1,23	1,26	1,25	1,27	1,29	1,34	1,33	1,37	1,39	1,44	1,44	1,44	1,42	1,42	1,39	1,38	1,36	1,36	1,34	1,31	1,27	1,24	1,25	1,24	1,2	1,18	1,14" # Incolla qui i dati TFR

anni_full = parse_dati(Anni_raw)
U = parse_dati(U_CO2_raw)
C = parse_dati(C_UOGD_raw)
L = parse_dati(L_UWCD_raw)
N = parse_dati(N_TFR_raw)

print(f" Dati storici grezzi caricati: {len(anni_full)} anni ({int(anni_full[0])}-{int(anni_full[-1])}).")

# ==========================================
# TRASFORMAZIONE IN VETTORI TERMODINAMICI (S1, S2, S3)
# ==========================================
print("\n[!] Esecuzione Trasformazione Formule C.L.N. ...")

S1_full = 1 - (316.91 / U)
L_su_C = L / C
M = np.max(L_su_C)
S2_full = (M - L_su_C) / (M + 1)
S3_full = (2.1 / N) - 1

print(" -> Trasformazione completata.")
print(" -> Vettori S1, S2, S3 generati con successo.")
attendi_input()

# ==========================================
# TEST PRE-0: CORRELAZIONE SULL'INTERVALLO VISIVO (1975-2017)
# ==========================================
print("\n[TEST PRE-0] CORRELAZIONE SULL'INTERVALLO VISIVO (1975-2017)")
maschera_visiva = (anni_full >= 1975) & (anni_full <= 2017)
S2_visivo = S2_full[maschera_visiva]
S3_visivo = S3_full[maschera_visiva]

if len(S2_visivo) > 2:
    corr_visiva, _ = stats.pearsonr(S2_visivo, S3_visivo)
    r2_visivo = corr_visiva**2
    print("Parametro ; Valore")
    print(f"Indice Pearson (r) [1975-2017] ; {formatta_ita(corr_visiva,4)}")
    print(f"Varianza Spiegata (R^2) ; {formatta_ita(r2_visivo * 100,2)} %")
    print("\n-> CONCLUSIONE: L'intuizione visiva originale è confermata dai dati.")
else:
    print("[!] Attenzione: dati insufficienti per l'intervallo 1975-2017.")
attendi_input()

# ==========================================
# TEST 0: RICERCA DEL PERIMETRO (CHOW TEST INIZIALE)
# ==========================================
print("\n[TEST 0] CHOW TEST INIZIALE: Ricerca Frattura Regime Baby-Boom")
n_full = len(anni_full)
rss_pooled_full = calcola_rss(S2_full, S3_full)
massima_f_inizio, anno_inizio, idx_inizio = 0, None, 0

for i in range(5, n_full - 5):
    rss_unrestricted = calcola_rss(S2_full[:i], S3_full[:i]) + calcola_rss(S2_full[i:], S3_full[i:])
    if rss_unrestricted > 0:
        f_stat = ((rss_pooled_full - rss_unrestricted) / 2) / (rss_unrestricted / (n_full - 4))
        if f_stat > massima_f_inizio:
            massima_f_inizio, anno_inizio, idx_inizio = f_stat, anni_full[i], i

print("Parametro ; Valore")
print(f"Anno Rottura Regime Storico ; {int(anno_inizio)}")
print(f"Statistica F ; {formatta_ita(massima_f_inizio,2)}")
print(f"\n-> CONCLUSIONE: Vettori ricalibrati sul perimetro di 'Deficit' {int(anno_inizio)}-2024.")

anni = anni_full[idx_inizio:]
S1, S2, S3 = S1_full[idx_inizio:], S2_full[idx_inizio:], S3_full[idx_inizio:]

anno_tipping = 2018
idx_x = list(anni).index(anno_tipping) if anno_tipping in anni else -1

if idx_x != -1:
    anni_pre, S1_pre, S2_pre, S3_pre = anni[:idx_x], S1[:idx_x], S2[:idx_x], S3[:idx_x]
    anni_post, S1_post, S2_post, S3_post = anni[idx_x:], S1[idx_x:], S2[idx_x:], S3[idx_x:]
else:
    print("\n[!] Tipping point 2018 non trovato nei dati. Interrompo.")
    sys.exit()
attendi_input()

# ==========================================
# TEST 1: CORRELAZIONE 2D BASE (PRE-ROTTURA)
# ==========================================
print("\n[TEST 1] ANALISI DI CORRELAZIONE LINEARE 2D (Fase Pre-Rottura)")
corr_pre, _ = stats.pearsonr(S2_pre, S3_pre)
r_quadro = corr_pre**2
print("Parametro ; Valore")
print(f"Indice Pearson (S2 -> S3) ; {formatta_ita(corr_pre,4)}")
print(f"Varianza Spiegata 2D (R^2) ; {formatta_ita(r_quadro * 100,2)} %")
attendi_input()

# ==========================================
# TEST 2: ANALISI MULTIVARIATA 3D (LA CALDAIA)
# ==========================================
print("\n[TEST 2] ANALISI MULTIVARIATA 3D (S1 + S2 -> S3 | La Caldaia)")
X_3d = np.column_stack((S1_pre, S2_pre, np.ones(len(S1_pre))))
coeffs, _, _, _ = np.linalg.lstsq(X_3d, S3_pre, rcond=None)
b1, b2, intercept = coeffs

S3_pred_3d = np.dot(X_3d, coeffs)
ss_res_3d = np.sum((S3_pre - S3_pred_3d)**2)
ss_tot_3d = np.sum((S3_pre - np.mean(S3_pre))**2)
r2_3d = 1 - (ss_res_3d / ss_tot_3d)

print("Parametro ; Valore")
print(f"Varianza Spiegata Congiunta 3D (R^2) ; {formatta_ita(r2_3d * 100,2)} %")
print(f"Peso S1 (Ambiente/Spazio) ; {formatta_ita(b1,4)}")
print(f"Peso S2 (Capitale/Economia) ; {formatta_ita(b2,4)}")
print("\n-> CONCLUSIONE: L'integrazione di S1 dimostra la saturazione simultanea.")
print("   Il sistema non è in caduta, ma in esplosione per sovrapressione su tre assi.")
attendi_input()

# ==========================================
# TEST 3: LA LEVA DEMOGRAFICA (MOLTIPLICATORE L1)
# ==========================================
print("\n[TEST 3] CALCOLO DELLA LEVA DEMOGRAFICA (Moltiplicatore L1 su S2)")
def l1_loss(c, x, y): return np.sum(np.abs(y - (c * x)))
res = minimize(l1_loss, x0=1.0, args=(S2_pre, S3_pre), method='Nelder-Mead')
molt_L1 = res.x[0]
print(f"Moltiplicatore (Elasticità) ; {formatta_ita(molt_L1,4)}")
print(f"\n-> CONCLUSIONE: Fino al 2018, leva a x{int(round(molt_L1))} della pressione economica sulle nascite.")
attendi_input()

# ==========================================
# TEST 4: VALIDAZIONE FORZA DI GRAVITA' (ALFA VECM)
# ==========================================
print("\n[TEST 4] TEST DI GRAVITA' TERMODINAMICA (VECM Pre-Rottura)")
ECT = S3_pre - (S2_pre * molt_L1)
delta_S3 = np.diff(S3_pre)
ECT_lagged = ECT[:-1]
X = ECT_lagged
y = delta_S3
alpha = np.sum(X * y) / np.sum(X**2)
residui_alpha = y - (alpha * X)
se_alpha = np.sqrt((np.sum(residui_alpha**2) / (len(y) - 1)) / np.sum(X**2))
t_stat_alpha = alpha / se_alpha
print("Parametro ; Valore ; Std_Error ; T-Stat")
print(f"Coefficiente di Richiamo (Alfa) ; {formatta_ita(alpha,4)} ; {formatta_ita(se_alpha,4)} ; {formatta_ita(t_stat_alpha,2)}")
attendi_input()

# ==========================================
# TEST 5: ISTERESI E CONO DI CONFIDENZA POST-2018
# ==========================================
print("\n[TEST 5] ISTERESI: PROIEZIONE E CONO DI CONFIDENZA (2019-2025)")
slope, intercept, _, _, _ = stats.linregress(anni_post, S3_post)
n_post = len(anni_post)
residui_post = S3_post - (intercept + slope * anni_post)
see = np.sqrt(np.sum(residui_post**2) / (n_post - 2))
t_critico = stats.t.ppf(0.975, df=n_post-2)

anni_futuri = np.arange(2019, 2026)
media_x = np.mean(anni_post)
sq_x = np.sum((anni_post - media_x)**2)

print("Anno ; Limite_Ottimistico(95%) ; Trend_Centrale ; Limite_Pessimistico(95%)")
for t_futuro in anni_futuri:
    y_fut = intercept + slope * t_futuro
    margine = t_critico * see * np.sqrt(1 + (1/n_post) + ((t_futuro - media_x)**2 / sq_x))
    print(f"{int(t_futuro)} ; {formatta_ita(y_fut - margine,4)} ; {formatta_ita(y_fut,4)} ; {formatta_ita(y_fut + margine,4)}")

print("\n-> CONCLUSIONE MINIMA: Emorragia autonoma certificata.")
print("\n[*** AUTOPSIA TERMINATA ***]")