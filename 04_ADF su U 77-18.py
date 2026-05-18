import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# ==========================================
# INSERISCI QUI I TUOI DATI 
# ==========================================
# Sostituisci questi array con i tuoi dati REALI
anni = np.array([1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]) # 42 anni

    # Esempio: 1977, 1978, 1979...
 

U_array = np.array([333.84, 335.41, 336.84, 338.76, 340.12, 341.48, 343.15, 344.87, 346.35, 347.61, 349.31, 351.69, 353.2, 354.45, 355.7, 356.54, 357.21, 358.96, 360.97, 362.74, 363.88, 366.84, 368.54, 369.71, 371.32, 373.45, 375.98, 377.7, 379.98, 382.09, 384.02, 385.83, 387.64, 390.1, 391.85, 394.06, 396.74, 398.81, 401.01, 404.41, 406.76, 408.72])
# ==========================================

print("AVVIO DIAGNOSTICA ADF\n" + "="*40)

# Controllo di sicurezza sulle lunghezze degli array
if len(anni) == 0 or len(U_array) == 0:
    print("ERRORE: Inserire i dati negli array prima di eseguire lo script.")
elif len(anni) != len(U_array):
    print("ERRORE: L'array degli anni e quello della variabile U hanno lunghezze diverse!")
else:
    # Trasformazione in serie Pandas per facilitare i calcoli
    U = pd.Series(U_array, index=anni)
    
    # Calcolo della differenza prima (ΔU) e pulizia dei valori nulli (NaN generato dal primo lag)
    delta_U = U.diff().dropna()

    # Funzione diagnostica per l'esecuzione e l'interpretazione del test ADF
    def esegui_adf(serie, nome_variabile):
        print(f"--- Risultati Test ADF per: {nome_variabile} ---")
        
        # Esecuzione del test con ottimizzazione automatica dei lag tramite criterio AIC
        risultato = adfuller(serie, autolag='AIC')
        
        # Estrazione delle metriche chiave
        adf_stat = risultato[0]
        p_value = risultato[1]
        
        print(f"Statistica del test: {adf_stat:.4f}")
        print(f"P-value: {p_value:.6f}")
        
        print("Valori Critici di riferimento:")
        for chiave, valore in risultato[4].items():
            print(f"  {chiave}: {valore:.4f}")
        
        # Interpretazione logica (Soglia di confidenza al 95%, alpha = 0.05)
        if p_value < 0.05:
            print(f">>> VERDETTO: La serie {nome_variabile} è STAZIONARIA (L'ipotesi nulla è rifiutata).\n")
        else:
            print(f">>> VERDETTO: La serie {nome_variabile} NON È STAZIONARIA (Presenza di radice unitaria).\n")

    # Esecuzione sequenziale
    esegui_adf(U, "U (Livelli di CO2 grezza)")
    esegui_adf(delta_U, "ΔU (Differenze prime di CO2)")