import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- CONFIGURAZIONE FILE ---
file_input_originale = 'risultati_sentiment_1974_2024.csv'
file_output_clean = 'risultati_emozioni_1974_2024_NoLyrics.csv'
file_output_analisi = 'analisi_emozioni_top5.xlsx'

print("--- FASE 1: Caricamento e Pulizia ---")

try:
    # 1. Caricamento aggressivo: trattiamo tutto come stringa all'inizio per evitare errori di parsing immediati
    df = pd.read_csv(file_input_originale, sep=None, engine='python', dtype=str)
    print(f"File caricato. Righe totali: {len(df)}")

    # 2. Rimuoviamo la colonna lyrics se esiste
    if 'lyrics' in df.columns:
        df = df.drop(columns=['lyrics'])
        print("Colonna 'lyrics' rimossa.")

    # 3. Identifichiamo le colonne metadati (che NON devono essere convertite in numeri decimali)
    colonne_fisse = ['anno', 'rank', 'titolo', 'artista']

    # Tutte le altre sono potenziali emozioni
    potenziali_emozioni = [col for col in df.columns if col not in colonne_fisse]

    print("Conversione forzata dei valori numerici in corso...")

    # 4. CICLO DI CONVERSIONE FORZATA
    for col in potenziali_emozioni:
        # Sostituisce la virgola con il punto (per i decimali italiani)
        df[col] = df[col].str.replace(',', '.', regex=False)
        # Converte in numero. 'coerce' trasforma gli errori (testo non convertibile) in NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convertiamo anche Anno e Rank in numeri (ma interi)
    df['anno'] = pd.to_numeric(df['anno'], errors='coerce')
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')

    # 5. FILTRO DI SICUREZZA (Il fix per il tuo errore)
    # Ora ridefiniamo 'colonne_emozioni' prendendo SOLO quelle che sono effettivamente diventate numeri (float o int)
    # Questo esclude automaticamente qualsiasi colonna rimasta "object" che causava il crash.
    colonne_numeriche = df.select_dtypes(include=[np.number]).columns.tolist()
    colonne_emozioni = [col for col in colonne_numeriche if col not in colonne_fisse]

    # Riempiamo i NaN nelle emozioni con 0.0 per non rompere i calcoli
    df[colonne_emozioni] = df[colonne_emozioni].fillna(0.0)

    print(f"Colonne emozioni valide identificate: {len(colonne_emozioni)}")

    # Salviamo il file pulito
    df.to_csv(file_output_clean, index=False, sep=',', decimal='.')
    print(f"File pulito salvato: {file_output_clean}")

except FileNotFoundError:
    print(f"ERRORE: Il file '{file_input_originale}' non esiste.")
    exit()
except Exception as e:
    print(f"ERRORE GENERICO DURANTE IL CARICAMENTO: {e}")
    exit()

# ---------------------------------------------------------
# FASE 2: ANALISI (TOP 5 EMOZIONI)
# ---------------------------------------------------------
print("\n--- FASE 2: Analisi Top 5 Emozioni ---")

if len(colonne_emozioni) == 0:
    print("ERRORE: Non sono state trovate colonne numeriche per le emozioni. Controlla il CSV.")
    exit()


def get_top_5_emotions(row):
    # Seleziona i valori usando la lista sicura 'colonne_emozioni'
    emozioni_row = row[colonne_emozioni]

    # Sicurezza extra: ci assicuriamo che siano float
    emozioni_row = emozioni_row.astype(float)

    # Trova i 5 valori più alti
    top_5 = emozioni_row.nlargest(5)

    result = {}
    for i, (emozione, valore) in enumerate(top_5.items(), 1):
        result[f'Top_{i}_Nome'] = emozione
        result[f'Top_{i}_Valore'] = valore
    return pd.Series(result)


print("Calcolo in corso...")
try:
    # Eseguiamo l'applicazione
    df_results = df.apply(get_top_5_emotions, axis=1)
    df_analysis = df.join(df_results)

    df_analysis.to_excel(file_output_analisi, index=False)
    print(f"Analisi completata. Report salvato come: {file_output_analisi}")

except Exception as e:
    print(f"ERRORE DURANTE L'ANALISI: {e}")
    # Stampiamo il tipo di dati per debug se fallisce ancora
    print("Tipi di dati attuali:")
    print(df[colonne_emozioni].dtypes.head())
    exit()

# ---------------------------------------------------------
# FASE 3: GRAFICI
# ---------------------------------------------------------
print("\n--- FASE 3: Generazione Grafici ---")

try:
    # Raggruppa per anno
    evoluzione_annuale = df.groupby('anno')[colonne_emozioni].mean()

    # Grafico Linee
    plt.figure(figsize=(14, 7))
    target_emotions = ['joy', 'sadness', 'anger', 'love', 'optimism', 'fear']

    for emotion in target_emotions:
        if emotion in evoluzione_annuale.columns:
            sns.lineplot(data=evoluzione_annuale, x=evoluzione_annuale.index, y=emotion, label=emotion, linewidth=2)

    plt.title('Evoluzione delle Emozioni (Trend Temporale)', fontsize=16)
    plt.xlabel('Anno')
    plt.ylabel('Valore Medio')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Heatmap
    plt.figure(figsize=(18, 10))
    sns.heatmap(evoluzione_annuale.T, cmap='viridis', cbar_kws={'label': 'Intensità'})
    plt.title('Heatmap Emozioni per Anno', fontsize=16)
    plt.xlabel('Anno')
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Errore nella generazione dei grafici: {e}")

print("\nProcedura terminata.")