import pandas as pd
import lyricsgenius
from transformers import pipeline
import billboard
import time
import os
import re
from langdetect import detect, LangDetectException

# ==========================================
# 1. CONFIGURAZIONE
# ==========================================

START_YEAR = 2020
END_YEAR = 2020


def load_token(path="token.txt"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    else:
        return "NON TROVATO"


GENIUS_TOKEN = load_token()

# Configurazione Genius
genius = lyricsgenius.Genius(GENIUS_TOKEN,
                             remove_section_headers=True,
                             skip_non_songs=True,
                             verbose=False,
                             excluded_terms=["(Remix)", "(Live)", "Translation", "Traducción", "Perevod", "Version"]
                             )
genius.timeout = 20
genius.retries = 3

# --- MODIFICA AI: CAMBIO MODELLO ---
print(">>> Caricamento modello RoBERTa (GoEmotions)...")
emotion_classifier = pipeline(
    task="text-classification",
    # Qui abbiamo inserito il nuovo modello di SamLowe
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,
    truncation=True,
    max_length=512
)
print(">>> Modello GoEmotions caricato! (Output atteso: 28 emozioni)")


# ==========================================
# 2. FUNZIONI DI SUPPORTO (LOGGING AVANZATO)
# ==========================================

def clean_artist_name_advanced(artist_str):
    """Pulizia nome artista (toglie feat, with, &, ecc)"""
    pattern = r"(?i)(\sfeat\.|\sfeaturing|\swith|\s&|\sx\s|\s\(|\s\/|\sduet).*"
    cleaned = re.sub(pattern, "", artist_str)
    return cleaned.strip()


def check_language_verbose(text):
    """
    Controlla la lingua e STAMPA il motivo esatto dell'accettazione o rifiuto.
    Ritorna: (Esito booleano, Motivo stringa)
    """
    text_lower = text.lower()
    words = set(re.findall(r"\b\w+\b", text_lower))

    # 1. VOCABOLARIO STRETTO (Strict English)
    strict_english_vocab = {
        'the', 'and', 'with', 'that', 'this', 'what', 'know',
        'from', 'have', 'they', 'but', 'not', 'would', 'could',
        'should', 'there', 'their', 'when', 'which', 'make',
        'just', 'your', 'about', 'some', 'time', 'out', 'into',
        'year', 'people', 'take', 'good', 'them', 'see', 'other',
        'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',
        'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work',
        'first', 'well', 'way', 'even', 'new', 'want', 'because'
    }

    vocab_matches = words.intersection(strict_english_vocab)

    # 2. LANG DETECT
    try:
        detected_lang = detect(text)
    except LangDetectException:
        detected_lang = "unknown"

    # --- LOGICA DI DECISIONE ---

    # CASO A: LangDetect è sicuro sia Inglese
    if detected_lang == 'en':
        return True, f"Validato da LangDetect ('en')"

    # CASO B: LangDetect fallisce, ma il Vocabolario ci salva
    # (Richiediamo almeno 2 parole uniche esclusive inglesi)
    if len(vocab_matches) >= 2:
        found_words = list(vocab_matches)[:3]  # Ne mostriamo solo 3 per pulizia
        return True, f"Validato da Vocabolario (Trovate: {found_words}, nonostante Lang='{detected_lang}')"

    # CASO C: Rifiutato
    return False, f"Rifiutato (Lang='{detected_lang}', Vocab Matches={len(vocab_matches)})"


def scarica_testo_verbose(titolo, artista):
    """
    Prova diverse strategie di ricerca e stampa un log dettagliato passo-passo.
    """

    # Creiamo le varianti del nome artista
    nome_originale = artista
    nome_pulito = clean_artist_name_advanced(artista)

    tentativi = [
        ("Originale", nome_originale),
        ("Pulito", nome_pulito)
    ]

    # Rimuoviamo duplicati mantenendo l'ordine (se originale == pulito)
    tentativi_unici = []
    visti = set()
    for tipo, nome in tentativi:
        if nome not in visti:
            tentativi_unici.append((tipo, nome))
            visti.add(nome)

    # --- INIZIO RICERCA ---
    for tipo_ricerca, nome_art in tentativi_unici:
        print(f"      > Tentativo ({tipo_ricerca}): '{titolo}' by '{nome_art}'")

        try:
            song = genius.search_song(titolo, nome_art)
        except Exception as e:
            print(f"        [!] Errore connessione Genius: {e}")
            time.sleep(2)
            continue

        if not song:
            print(f"        [X] Genius: Nessun risultato trovato.")
            continue

        # Se Genius ha trovato qualcosa, controlliamo la lingua
        is_valid, reason = check_language_verbose(song.lyrics)

        if is_valid:
            print(f"        [V] ACCETTATO: {reason}")
            return song.lyrics
        else:
            print(f"        [!] TROVATO MA SCARTATO: {reason}")
            # Non ritorniamo None, ma continuiamo il ciclo sperando che
            # il prossimo tentativo (nome pulito) trovi una versione migliore/inglese
            continue

    # Se finisce il ciclo e non ha ritornato nulla
    return None


def analizza_emozioni(text, chunk_size=350):
    if not text or not isinstance(text, str): return None
    words = text.split()
    if len(words) < 5: return None
    chunks = [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size)]
    try:
        results_list = emotion_classifier(chunks)
        all_scores = []
        for result in results_list:
            # result è una lista di dizionari [{'label': 'admiration', 'score': 0.9}, ...]
            # La convertiamo in un dizionario singolo {label: score} per il chunk
            chunk_score = {item['label']: item['score'] for item in result}
            all_scores.append(chunk_score)

        df_temp = pd.DataFrame(all_scores)
        # Media di tutti i chunk per ottenere il profilo della canzone
        return df_temp.mean().to_dict()
    except Exception as e:
        print(f"Errore analisi: {e}")
        return None


# ==========================================
# 3. MAIN
# ==========================================

def main():
    if GENIUS_TOKEN == "NON TROVATO" and not os.path.exists("token.txt"):
        print("ERRORE: Token Genius mancante.")
        return

    # A. BILLBOARD
    print(f"\n==========================================")
    print(f" 1. SCARICAMENTO BILLBOARD ({START_YEAR}-{END_YEAR})")
    print(f"==========================================")

    songs_list = []
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"   Recupero classifica anno {year}...")
        try:
            chart = billboard.ChartData('hot-100-songs', year=year)
            for entry in chart:
                songs_list.append({'anno': year, 'rank': entry.rank, 'titolo': entry.title, 'artista': entry.artist})
            time.sleep(1)
        except Exception as e:
            print(f"   Errore critico anno {year}: {e}")

    df = pd.DataFrame(songs_list)
    df = df.drop_duplicates(subset=['titolo', 'artista'], keep='first')
    print(f">>> Totale canzoni uniche da analizzare: {len(df)}")

    # B. GENIUS E PROCESSING
    print(f"\n==========================================")
    print(f" 2. RICERCA TESTI E ANALISI")
    print(f"==========================================")

    testi = []
    found_count = 0
    total = len(df)

    for index, row in df.iterrows():
        print(f"\n[{index + 1}/{total}] {row['titolo']} - {row['artista']}")

        # Funzione Verbose
        lyrics = scarica_testo_verbose(row['titolo'], row['artista'])

        if lyrics:
            testi.append(lyrics)
            found_count += 1
        else:
            testi.append(None)
            print("      [FAIL] Canzone saltata definitivamente.")

        time.sleep(2)  # Pausa di cortesia

    df['lyrics'] = testi
    df_clean = df.dropna(subset=['lyrics']).copy()

    print(f"\n------------------------------------------")
    print(f"STATISTICHE FINALI RECUPERO:")
    print(f"Canzoni Totali: {total}")
    print(f"Testi Trovati e Validi: {found_count}")
    print(f"Scartati: {total - found_count}")
    print(f"------------------------------------------")

    # C. ANALISI EMOTIVA (Solo su quelle trovate)
    print(f"\n==========================================")
    print(f" 3. CALCOLO EMOZIONI (GoEmotions)")
    print(f"==========================================")

    if not df_clean.empty:
        print("Inizio analisi AI sui testi scaricati...")
        emotion_results = df_clean['lyrics'].apply(analizza_emozioni)
        emotion_df = emotion_results.apply(pd.Series)

        df_finale = pd.concat([df_clean, emotion_df], axis=1)

        # Output ordinato dinamico (si adatta alle colonne del nuovo modello)
        base_cols = ['anno', 'rank', 'titolo', 'artista']
        emotion_cols = sorted(emotion_df.columns.tolist())  # Ordiniamo le emozioni alfabeticamente

        final_cols = [c for c in base_cols if c in df_finale.columns] + emotion_cols

        df_finale = df_finale[final_cols]

        filename = f'billboard_goemotions_{START_YEAR}_{END_YEAR}.csv'
        df_finale.to_csv(filename, index=False)
        print(f"\n>>> COMPLETATO! File salvato: {filename}")
    else:
        print("Nessun dato valido per l'analisi.")


if __name__ == "__main__":
    main()