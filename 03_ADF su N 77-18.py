import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# ==========================================
# INSERISCI QUI I TUOI DATI 
# ==========================================
# Sostituisci questi array con i tuoi dati REALI
anni = np.array([1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]) # 42 anni
 

# Inserisci i dati grezzi delle nascite (o del tasso di fecondità TFR), NON la formula S3
N_array = np.array([1.97, 1.87, 1.76, 1.68, 1.6, 1.6, 1.54, 1.48, 1.45, 1.37, 1.35, 1.38, 1.35, 1.36, 1.32, 1.32, 1.26, 1.22, 1.19, 1.22, 1.23, 1.21, 1.23, 1.26, 1.25, 1.27, 1.29, 1.34, 1.33, 1.37, 1.39, 1.44, 1.44, 1.44, 1.42, 1.42, 1.39, 1.38, 1.36, 1.36, 1.34, 1.31])
# ==========================================

print("AVVIO DIAGNOSTICA ADF SU VARIABILE BIOLOGICA GREZZA\n" + "="*50)

# Controllo di sicurezza sulle lunghezze degli array
if len(anni) == 0 or len(N_array) == 0:
    print("ERRORE: Inserire i dati negli array prima di eseguire lo script.")
elif len(anni) != len(N_array):
    print("ERRORE: L'array degli anni e quello della variabile N hanno lunghezze diverse!")
else:
    # Trasformazione in serie Pandas 
    N = pd.Series(N_array, index=anni)
    
    # Calcolo della differenza prima (ΔN) 
    delta_N = N.diff().dropna()

    # Funzione diagnostica standardizzata
    def esegui_adf(serie, nome_variabile):
        print(f"--- Risultati Test ADF per: {nome_variabile} ---")
        risultato = adfuller(serie, autolag='AIC')
        adf_stat = risultato[0]
        p_value = risultato[1]
        
        print(f"Statistica del test: {adf_stat:.4f}")
        print(f"P-value: {p_value:.6f}")
        print("Valori Critici di riferimento:")
        for chiave, valore in risultato[4].items():
            print(f"  {chiave}: {valore:.4f}")
        
        if p_value < 0.05:
            print(f">>> VERDETTO: La serie {nome_variabile} è STAZIONARIA (L'ipotesi nulla è rifiutata).\n")
        else:
            print(f">>> VERDETTO: La serie {nome_variabile} NON È STAZIONARIA (Presenza di radice unitaria).\n")

    # Esecuzione sequenziale
    esegui_adf(N, "N (Livelli del dato grezzo Nascite/TFR)")
    esegui_adf(delta_N, "ΔN (Differenze prime di N)")