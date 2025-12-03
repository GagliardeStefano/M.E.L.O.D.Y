import pandas as pd
import lyricsgenius
from transformers import pipeline
import billboard
import numpy as np
import time
import os

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================

# -- RANGE ANNI DA ANALIZZARE --
# Per fare una prova veloce metti un range piccolo (es. 2020-2021)
# Per il progetto finale metti: 1970, 2023
START_YEAR = 2020
END_YEAR = 2021


def load_token(path="token.txt"):
    """Carica il token da file, o usa una stringa diretta se il file non c'è"""
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    else:
        # Fallback: Inserisci qui il token se non vuoi usare il file txt
        return "INSERISCI_QUI_IL_TUO_TOKEN_SE_NON_HAI_IL_FILE"


GENIUS_TOKEN = load_token()

# Configurazione Genius
genius = lyricsgenius.Genius(GENIUS_TOKEN,
                             remove_section_headers=True,
                             skip_non_songs=True,
                             verbose=False)

# Configurazione Modello AI (Hugging Face)
print(">>> Caricamento del modello RoBERTa (potrebbe richiedere un minuto)...")
emotion_classifier = pipeline(
    task="text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    truncation=True,
    max_length=512
)
print(">>> Modello caricato con successo!")


# ==========================================
# 2. FUNZIONI DI SUPPORTO
# ==========================================

def clean_artist_name(artist_str):
    """
    Pulisce il nome dell'artista per aiutare la ricerca su Genius.
    Es: "Dua Lipa feat. DaBaby" -> "Dua Lipa"
    """
    separators = [" feat.", " Feat.", " featuring", " Featuring", " with ", " With ", " x ", " X "]
    cleaned = artist_str
    for sep in separators:
        if sep in cleaned:
            cleaned = cleaned.split(sep)[0]
    return cleaned.strip()


def scarica_testo(titolo, artista):
    """Cerca la canzone su Genius e restituisce il testo pulito."""
    try:
        # Tentativo 1: Ricerca esatta
        song = genius.search_song(titolo, artista)

        # Tentativo 2: Se fallisce, prova a pulire il nome artista (es. togliere 'feat.')
        if not song and "feat" in artista.lower():
            clean_art = clean_artist_name(artista)
            print(f"      (Riprovo con artista pulito: {clean_art}...)")
            song = genius.search_song(titolo, clean_art)

        if song:
            return song.lyrics
        return None
    except Exception as e:
        print(f"Errore Genius su {titolo}: {e}")
        return None


def analizza_emozioni(text, chunk_size=350):
    """Divide il testo in blocchi, analizza e fa la media dei punteggi."""
    if not text or not isinstance(text, str):
        return None

    words = text.split()
    if len(words) < 5:
        return None

    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i: i + chunk_size])
        chunks.append(chunk_text)

    try:
        results_list = emotion_classifier(chunks)
        all_scores = []
        for result in results_list:
            chunk_score = {item['label']: item['score'] for item in result}
            all_scores.append(chunk_score)

        df_temp = pd.DataFrame(all_scores)
        mean_scores = df_temp.mean().to_dict()
        return mean_scores

    except Exception as e:
        print(f"Errore AI: {e}")
        return None


# ==========================================
# 3. IL PROCESSO PRINCIPALE (MAIN)
# ==========================================

def main():
    if "INSERISCI" in GENIUS_TOKEN and not os.path.exists("token.txt"):
        print("ERRORE: Manca il Token di Genius! Crea un file token.txt o inseriscilo nel codice.")
        return

    # A. SCARICAMENTO DATI DA BILLBOARD
    print(f"\n>>> 1. Scaricamento Classifiche Billboard ({START_YEAR}-{END_YEAR})...")

    songs_list = []

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"   Fetching Year-End Chart: {year}...")
        try:
            # 'hot-100-songs' è l'identificativo per la classifica di FINE ANNO
            chart = billboard.ChartData('hot-100-songs', year=year)

            for entry in chart:
                songs_list.append({
                    'anno': year,
                    'rank': entry.rank,
                    'titolo': entry.title,
                    'artista': entry.artist
                })

            # Pausa per non essere bloccati da Billboard
            time.sleep(1)

        except Exception as e:
            print(f"   Errore scaricando l'anno {year}: {e}")

    # Creazione DataFrame Iniziale
    df = pd.DataFrame(songs_list)
    print(f"\n>>> Dataset base creato! Totale canzoni: {len(df)}")
    print(df.head())

    # B. SCARICAMENTO TESTI GENIUS
    print("\n>>> 2. Scaricamento Testi da Genius...")

    testi = []
    found_count = 0

    # Usiamo un contatore per monitorare il progresso
    total = len(df)

    for index, row in df.iterrows():
        print(f"[{index + 1}/{total}] Searching: {row['titolo']} - {row['artista']}...")

        lyrics = scarica_testo(row['titolo'], row['artista'])

        if lyrics:
            testi.append(lyrics)
            found_count += 1
        else:
            testi.append(None)
            print("      [X] Testo non trovato.")

    df['lyrics'] = testi

    # Filtriamo le canzoni senza testo
    df_clean = df.dropna(subset=['lyrics']).copy()
    print(f"\n>>> Testi trovati: {found_count} su {total}")

    # C. ANALISI EMOTIVA (AI)
    print("\n>>> 3. Analisi Emotiva in corso (RoBERTa)...")

    # Se il dataset è grande, questo passaggio impiegherà tempo
    emotion_results = df_clean['lyrics'].apply(analizza_emozioni)

    # Convertiamo i risultati in colonne
    emotion_df = emotion_results.apply(pd.Series)

    # Uniamo tutto
    df_finale = pd.concat([df_clean, emotion_df], axis=1)

    # D. OUTPUT E SALVATAGGIO
    print("\n>>> 4. Salvataggio Risultati...")

    filename = f'billboard_analysis_{START_YEAR}_{END_YEAR}.csv'

    # Salviamo su CSV
    df_finale.to_csv(filename, index=False)

    print(f"COMPLETATO! File salvato come: {filename}")

    # Anteprima veloce
    cols = ['anno', 'titolo', 'joy', 'sadness', 'anger']
    # Controlliamo se le colonne esistono (in caso di errori nell'apply)
    existing_cols = [c for c in cols if c in df_finale.columns]
    if existing_cols:
        print(df_finale[existing_cols].head())


# Avvio dello script
if __name__ == "__main__":
    main()