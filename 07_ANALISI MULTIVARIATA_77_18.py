import numpy as np
from scipy.stats import pearsonr, linregress
try:
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    STATMODELS_INSTALLED = True
except ImportError:
    STATMODELS_INSTALLED = False

# ==========================================
# FUNZIONE HELPER: CORRELAZIONE PARZIALE
# ==========================================
def correlazione_parziale(x, y, z):
    """
    Calcola la correlazione tra x (S8) e y (S3) RIMUOVENDO l'effetto di z (S1).
    Come? Regredendo x su z e y su z, e calcolando Pearson sui residui.
    """
    # Quanto di x è spiegato da z? Troviamo l'errore (residuo)
    slope_xz, intercept_xz, _, _, _ = linregress(z, x)
    residui_x = x - (slope_xz * z + intercept_xz)
    
    # Quanto di y è spiegato da z? Troviamo l'errore (residuo)
    slope_yz, intercept_yz, _, _, _ = linregress(z, y)
    residui_y = y - (slope_yz * z + intercept_yz)
    
    # Correlazione pura tra x e y, senza l'ombra di z
    corr_parziale, pval = pearsonr(residui_x, residui_y)
    return corr_parziale, pval

def formatta_ita(val, dec=4):
    return f"{val:.{dec}f}".replace('.', ',')

# ==========================================
# INSERISCI QUI I TUOI DATI VERI (es. 1977-2018)
# ==========================================
# Sostituisci gli array vuoti con i tuoi dati estratti da dbnomics / Mauna Loa
# DEVONO avere tutti la stessa lunghezza (es. 42 anni)

S1 = np.array([-0.050713, -0.055156, -0.059168, -0.064500, -0.068241, -0.071952, -0.076468, -0.081074, -0.085001, -0.088317, -0.092754, -0.098894, -0.102746, -0.105911, -0.109053, -0.111152, -0.112819, -0.117144, -0.122060, -0.126344, -0.129081, -0.136108, -0.140093, -0.142815, -0.146531, -0.151399, -0.157109, -0.160948, -0.165982, -0.170588, -0.174757, -0.178628, -0.182463, -0.187619, -0.191247, -0.195782, -0.201215, -0.205361, -0.209720, -0.216365, 0.220891926, 0.224628107]) # <-- Inserisci qui lo scostamento relativo PPM CO2
S8 = np.array([0.007203887, 0.025629395, 0.041066315, 0.051298789, 0.04094195, 0.049146599, 0.055014171, 0.07687899, 0.081035111, 0.098745046, 0.099768447, 0.102145, 0.103725538, 0.087614863, 0.08076154, 0.082122577, 0.085557303, 0.110433328, 0.134048531, 0.129142007, 0.122226995, 0.130228167, 0.129588654, 0.141573711, 0.143729023, 0.136868153, 0.133383134, 0.134875141, 0.123387001, 0.116598075, 0.117498674, 0.109090164, 0.090664465, 0.092775517, 0.099571953, 0.090755122, 0.095565916, 0.09780629, 0.094704396, 0.102485274, 0.097894409, 0.09069451])  # <-- Inserisci qui l'Aumento del Capitale
S3 = np.array([0.065989848, 0.122994652, 0.193181818, 0.25, 0.3125, 0.3125, 0.363636364, 0.418918919, 0.448275862, 0.532846715, 0.555555556, 0.52173913, 0.555555556, 0.544117647, 0.590909091, 0.590909091, 0.666666667, 0.721311475, 0.764705882, 0.721311475, 0.707317073, 0.73553719, 0.707317073, 0.666666667, 0.68, 0.653543307, 0.627906977, 0.567164179, 0.578947368, 0.532846715, 0.510791367, 0.458333333, 0.458333333, 0.458333333, 0.478873239, 0.478873239, 0.510791367, 0.52173913, 0.544117647, 0.544117647, 0.567164179, 0.603053435])  # <-- Inserisci qui il Deficit Demografico

# --- BLOCCO DI SICUREZZA SE I DATI SONO VUOTI ---
# Se non metti i dati, genero numeri a caso solo per farti vedere l'impaginazione.
# CANCELLA QUESTE 3 RIGHE DOPO AVER MESSO I TUOI DATI VERI!
# ------------------------------------------------

print("="*85)
print(" SUPREMA CORTE ECONOMETRICA: ANALISI MULTIVARIATA TERMODINAMICA")
print(" Modello: S1 (Entropia) -> S8 (Capitale) -> S3 (Demografia)")
print("="*85)

# ---------------------------------------------------------
# PROVA 1: IL TEST DEL BURATTINAIO (Correlazione Parziale)
# ---------------------------------------------------------
corr_semplice, _ = pearsonr(S8, S3)
corr_parz, pval_parz = correlazione_parziale(S8, S3, S1)

print("\n[PROVA 1] RICERCA DEL BURATTINAIO (Correlazione Parziale)")
print(f" -> Correlazione Semplice (S8 vs S3):    {formatta_ita(corr_semplice, 4)}")
print(f" -> Correlazione Parziale (S8 vs S3 | depurata da S1): {formatta_ita(corr_parz, 4)}")

print("\n INTERPRETAZIONE:")
if abs(corr_parz) < abs(corr_semplice) * 0.5:
    print(" -> BINGO! La correlazione crolla senza S1.")
    print("    Significa che l'Entropia (S1) è il vero motore nascosto che sta")
    print("    schiacciando contemporaneamente sia l'economia (S8) che le nascite (S3).")
else:
    print(" -> La correlazione resiste forte anche senza S1.")
    print("    S8 e S3 hanno una loro dinamica indipendente fortissima.")

# ---------------------------------------------------------
# PROVA 2: TEST DI COINTEGRAZIONE DI JOHANSEN (Il Sistema a 3)
# ---------------------------------------------------------
print("\n[PROVA 2] IL LIMITE BIOFISICO (Test di Johansen a 3 Variabili)")
if not STATMODELS_INSTALLED:
    print(" [!] statsmodels non installato. Impossibile eseguire Johansen.")
else:
    # Uniamo le tre serie in una matrice (42 righe, 3 colonne)
    dati_matrice = np.column_stack((S1, S8, S3))
    
    # Eseguiamo il test di Johansen (k_ar_diff=1 è standard per serie storiche annuali)
    # det_order = -1 (nessun trend deterministico), 0 (costante), 1 (trend lineare)
    # Usiamo 1 perché le tue serie hanno palesemente un trend lineare/asintotico.
    risultato = coint_johansen(dati_matrice, det_order=1, k_ar_diff=1)
    
    # La statistica della Traccia (Trace Statistic) e i valori critici al 95%
    trace_stat = risultato.lr1
    crit_vals_95 = risultato.cvt[:, 1] # colonna 1 è il livello di significatività del 5%
    
    print(" Verifica se esiste almeno una 'forza di gravità' che tiene unite S1, S8, S3:")
    print(" Ipotesi Null (R=0): Zero vettori di cointegrazione (Nessun legame).")
    
    print(f"\n -> Statistica della Traccia: {formatta_ita(trace_stat[0], 2)}")
    print(f" -> Valore Critico (95%):     {formatta_ita(crit_vals_95[0], 2)}")
    
    if trace_stat[0] > crit_vals_95[0]:
        print("\n -> VERDETTO DEFINITIVO: SISTEMA COINTEGRATO!")
        print("    Statistica > Valore Critico. Hai dimostrato che Termodinamica (S1),")
        print("    Estrazione del Capitale (S8) e Crollo Biologico (S3) NON POSSONO")
        print("    separarsi. Formano un unico blocco sistemico nel lungo periodo.")
    else:
        print("\n -> VERDETTO: Nessuna cointegrazione a 3.")
        print("    Statistica < Valore Critico. Le serie tendono ad allontanarsi")
        print("    strutturalmente col tempo.")

print("\n" + "="*85)