import numpy as np
from scipy import stats

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
# 1. INSERIMENTO DATI (Usa i tuoi dati reali)
# ==========================================
Anni_raw = "1977 1978 1979 1980 1981 1982 1983 1984 1985 1986 1987 1988 1989 1990 1991 1992 1993 1994 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025"

# IMPORTANTE: Usa i tuoi dati reali per S8 (Economia) e S3 (Deficit Demografico)
S8_raw = "0,007203887	0,025629395	0,041066315	0,051298789	0,04094195	0,049146599	0,055014171	0,07687899	0,081035111	0,098745046	0,099768447	0,102145	0,103725538	0,087614863	0,08076154	0,082122577	0,085557303	0,110433328	0,134048531	0,129142007	0,122226995	0,130228167	0,129588654	0,141573711	0,143729023	0,136868153	0,133383134	0,134875141	0,123387001	0,116598075	0,117498674	0,109090164	0,090664465	0,092775517	0,099571953	0,090755122	0,095565916	0,09780629	0,094704396	0,102485274	0,097894409	0,09069451	0,08437417	0,079224317	0,086330747	0,105780166	0,118910662	0,096868826	0,086414986"

S3_raw = "0,065989848	0,122994652	0,193181818	0,25	0,3125	0,3125	0,363636364	0,418918919	0,448275862	0,532846715	0,555555556	0,52173913	0,555555556	0,544117647	0,590909091	0,590909091	0,666666667	0,721311475	0,764705882	0,721311475	0,707317073	0,73553719	0,707317073	0,666666667	0,68	0,653543307	0,627906977	0,567164179	0,578947368	0,532846715	0,510791367	0,458333333	0,458333333	0,458333333	0,478873239	0,478873239	0,510791367	0,52173913	0,544117647	0,544117647	0,567164179	0,603053435	0,653543307	0,693548387	0,68	0,693548387	0,75	0,779661017	0,842105263	"
anni = parse_dati(Anni_raw)
S8 = parse_dati(S8_raw)
S3 = parse_dati(S3_raw)

idx_x = list(anni).index(2018)
anni_pre, S8_pre, S3_pre = anni[:idx_x], S8[:idx_x], S3[:idx_x]
anni_post, S8_post, S3_post = anni[idx_x:], S8[idx_x:], S3[idx_x:]

print("="*80)
print(" VALIDAZIONI FINALI - Errore su Alfa e Intervalli di Confidenza")
print("="*80)

# ---------------------------------------------------------
# VALIDAZIONE 1: STIMA DELL'ERRORE SU ALFA (Meccanismo VECM)
# ---------------------------------------------------------
# 1. Calcoliamo il moltiplicatore di lungo periodo (Elasticità) pre-2018
molt_L1 = np.sum(np.abs(S3_pre)) / np.sum(np.abs(S8_pre)) # Approssimazione L1 veloce

# 2. Calcoliamo l'ECT (Error Correction Term) = S3 - (S8 * molt_L1)
ECT = S3_pre - (S8_pre * molt_L1)

# 3. Calcoliamo le differenze prime (Delta S3)
delta_S3 = np.diff(S3_pre)
ECT_lagged = ECT[:-1] # L'errore dell'anno prima causa la correzione dell'anno dopo

# 4. Regressione OLS senza intercetta: Delta S3 = Alfa * ECT_lagged
# Usiamo l'algebra matriciale per l'errore standard
X = ECT_lagged
y = delta_S3
alpha = np.sum(X * y) / np.sum(X**2)

# Residui e Standard Error di Alfa
residui_alpha = y - (alpha * X)
gradi_liberta = len(y) - 1
varianza_residua = np.sum(residui_alpha**2) / gradi_liberta
se_alpha = np.sqrt(varianza_residua / np.sum(X**2))
t_stat_alpha = alpha / se_alpha

print("\n1) ROBUSTEZZA DELLA FORZA DI GRAVITA' (ALFA) PRE-2018:")
print("Parametro ; Valore_Stimato ; Errore_Standard (SE) ; Statistica_T")
print(f"Coefficiente Alfa ; {formatta_ita(alpha,4)} ; {formatta_ita(se_alpha,4)} ; {formatta_ita(t_stat_alpha,2)}")

if abs(t_stat_alpha) > 2:
    print(" -> RISULTATO: T-stat > 2. La gravità del sistema è statisticamente INOPPUGNABILE.")

# ---------------------------------------------------------
# VALIDAZIONE 2: CONO DI CONFIDENZA 95% PROIEZIONE POST-2018
# ---------------------------------------------------------
slope, intercept, r_value, p_value, std_err = stats.linregress(anni_post, S3_post)

# Calcolo dell'Errore Standard della Previsione (SEE)
n_post = len(anni_post)
y_hat_storico = intercept + slope * anni_post
residui_post = S3_post - y_hat_storico
see = np.sqrt(np.sum(residui_post**2) / (n_post - 2))

anni_futuri = np.arange(2019, 2026)
media_anni_post = np.mean(anni_post)
somma_quadrati_x = np.sum((anni_post - media_anni_post)**2)
t_critico = stats.t.ppf(0.975, df=n_post-2) # Valore t per 95% confidenza

trend_centrale = []
limite_inferiore = []
limite_superiore = []

for t_futuro in anni_futuri:
    y_futuro = intercept + slope * t_futuro
    # Formula esatta del margine di errore per la previsione
    margine_errore = t_critico * see * np.sqrt(1 + (1/n_post) + ((t_futuro - media_anni_post)**2 / somma_quadrati_x))
    
    trend_centrale.append(y_futuro)
    limite_inferiore.append(y_futuro - margine_errore)
    limite_superiore.append(y_futuro + margine_errore)

print("\n2) CONO DI CONFIDENZA AL 95% PER LA NECROSI (2019-2025):")
print("Anno_Proiez ; Limite_Ottimistico (Basso) ; Trend_Centrale ; Limite_Pessimistico (Alto)")
for i, anno in enumerate(anni_futuri):
    print(f"{int(anno)} ; {formatta_ita(limite_inferiore[i],4)} ; {formatta_ita(trend_centrale[i],4)} ; {formatta_ita(limite_superiore[i],4)}")

print("\n" + "="*80)