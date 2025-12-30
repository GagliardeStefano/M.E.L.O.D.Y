import pandas as pd
import lyricsgenius
from transformers import pipeline
import time
import os
import re
from langdetect import detect, LangDetectException
import torch
import sys

# ==========================================
# CONFIGURAZIONE ANALISI
# ==========================================
# DEFINISCI QUI IL RANGE DI ANNI DA ANALIZZARE
ANNO_INIZIO = 1974
ANNO_FINE = 2024

INPUT_FILE = "dataset_grezzo.csv"
OUTPUT_FILE = f"risultati_sentiment_{ANNO_INIZIO}_{ANNO_FINE}.csv"
ERROR_FILE = f"errori_non_trovati_{ANNO_INIZIO}_{ANNO_FINE}.csv"

def load_token(path="token.txt"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    return "NON TROVATO"

GENIUS_TOKEN = load_token()

# Inizializzazione Genius
genius = lyricsgenius.Genius(GENIUS_TOKEN, remove_section_headers=True, skip_non_songs=True, verbose=False, retries=3, timeout=20)
genius.excluded_terms = ["(Remix)", "(Live)", "Translation", "Traducción", "Version"]

# Inizializzazione Modello AI
device = 0 if torch.cuda.is_available() else -1
print(f">>> Caricamento modello GoEmotions su {'GPU' if device == 0 else 'CPU'}...")
emotion_classifier = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None, truncation=True, max_length=512, device=device
)

# ==========================================
# FUNZIONI DI SUPPORTO
# ==========================================

def clean_artist_name(artist_str):
    """Rimuove feat, with, &, ecc."""
    pattern = r"(?i)(\sfeat\.|\sfeaturing|\swith|\s&|\sx\s|\s\(|\s\/|\sduet).*"
    return re.sub(pattern, "", artist_str).strip()

def clean_title_name(title_str):
    """Rimuove qualsiasi contenuto tra parentesi tonde ()."""
    return re.sub(r'\(.*?\)', '', title_str).strip()

def check_language(text):
    """Verifica se il testo è inglese usando langdetect + vocabolario di sicurezza."""
    if not text: return False, "Testo vuoto"
    
    strict_vocab = {'the', 'and', 'with', 'that', 'this', 'what', 'know', 'from', 'have', 'but'}
    words = set(re.findall(r"\b\w+\b", text.lower()))
    matches = words.intersection(strict_vocab)

    try:
        lang = detect(text)
    except:
        lang = "unknown"

    if lang == 'en': return True, "LangDetect: OK"
    if len(matches) >= 2: return True, "VocabCheck: OK"
    
    return False, f"Lingua non inglese (Rilevato: {lang})"

def scarica_testo(titolo, artista):
    """
    Prova tutte le combinazioni possibili tra varianti del Titolo e varianti dell'Artista.
    """
    
    # 1. Generiamo le varianti dell'ARTISTA
    raw_artist = artista
    clean_art = clean_artist_name(artista)
    very_clean_art = clean_art.split(',')[0].split(' x ')[0].strip()
    
    artist_variants = [raw_artist]
    if clean_art != raw_artist and clean_art: 
        artist_variants.append(clean_art)
    if very_clean_art != clean_art and very_clean_art != raw_artist and very_clean_art:
        artist_variants.append(very_clean_art)

    # 2. Generiamo le varianti del TITOLO
    raw_title = titolo
    clean_title = clean_title_name(titolo)
    
    title_variants = [raw_title]
    if clean_title != raw_title and len(clean_title) > 1:
        title_variants.append(clean_title)

    # 3. Creiamo la MATRICE DI TENTATIVI
    tentativi = []
    for t_var in title_variants:
        for a_var in artist_variants:
            label = "Originale"
            if t_var == clean_title and t_var != raw_title: label = "Titolo Pulito"
            if a_var == clean_art and a_var != raw_artist: label += " + Artista Pulito"
            elif a_var == very_clean_art and a_var != raw_artist: label += " + Solo Main Artist"
            
            tentativi.append((label, t_var, a_var))

    ultimo_errore = "Nessun risultato trovato dopo tutti i tentativi combinati"

    for idx, (label, t_query, a_query) in enumerate(tentativi):
        print(f"      > Tentativo {idx+1}/{len(tentativi)} [{label}]: '{t_query}' by '{a_query}'")
        
        try:
            song = genius.search_song(t_query, a_query)
        except Exception as e:
            msg = f"Errore API Genius: {e}"
            print(f"        [!] {msg}")
            ultimo_errore = msg
            time.sleep(2)
            continue

        if song:
            ok, motivo = check_language(song.lyrics)
            if ok:
                print(f"        [V] TROVATO ({motivo})")
                return song.lyrics, None 
            else:
                msg = f"Trovato ma scartato ({motivo})"
                print(f"        [X] {msg}")
                ultimo_errore = msg 
        else:
            msg = f"Nessun risultato"
            print(f"        [.] {msg}")
            if "scartato" not in ultimo_errore:
                ultimo_errore = msg
            
    return None, ultimo_errore

def analizza_emozioni(text):
    if not text or len(text.split()) < 5: return None
    words = text.split()
    chunk_size = 350
    chunks = [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size)]
    
    try:
        results = emotion_classifier(chunks)
        all_scores = [{item['label']: item['score'] for item in res} for res in results]
        return pd.DataFrame(all_scores).mean().to_dict()
    except Exception as e:
        print(f"Errore AI: {e}")
        return None

# ==========================================
# MAIN
# ==========================================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERRORE: File '{INPUT_FILE}' non trovato. Esegui prima scraper_wiki.py")
        return

    print(f"Lettura dataset completo: {INPUT_FILE}")
    df_completo = pd.read_csv(INPUT_FILE)
    
    # --- FILTRO ANNI ---
    print(f"Filtraggio range: {ANNO_INIZIO} - {ANNO_FINE}")
    df = df_completo[(df_completo['anno'] >= ANNO_INIZIO) & (df_completo['anno'] <= ANNO_FINE)].copy()
    
    if df.empty:
        print(f"❌ NESSUNA CANZONE TROVATA NEL RANGE {ANNO_INIZIO}-{ANNO_FINE}.")
        return

    print(f"Canzoni selezionate per l'analisi: {len(df)}")
    
    testi = []
    lista_errori = [] 
    interrotto_manualmente = False
    
    # 1. SCARICAMENTO TESTI (CON GESTIONE INTERRUZIONE)
    print("\n--- FASE 1: SCARICAMENTO TESTI (LOGICA COMBINATORIA) ---")
    print("ℹ️  NOTA: Puoi interrompere lo script con CTRL+C e i dati verranno salvati.")
    
    total = len(df)
    
    try:
        for i, row in df.iterrows():
            idx_visuale = list(df.index).index(i) + 1
            print(f"\n[{idx_visuale}/{total}] (Anno {row['anno']}) {row['titolo']} - {row['artista']}")
            
            lyrics, errore = scarica_testo(row['titolo'], row['artista'])
            
            if lyrics:
                testi.append(lyrics)
            else:
                testi.append(None)
                lista_errori.append({
                    'anno': row['anno'],
                    'rank': row['rank'],
                    'titolo': row['titolo'],
                    'artista': row['artista'],
                    'motivo_errore': errore
                })
                
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 INTERRUZIONE RILEVATA! (CTRL+C)")
        print("Salvataggio dei dati raccolti finora in corso... attendere.")
        interrotto_manualmente = True

    # --- FASE DI SALVATAGGIO (Eseguita anche se interrotto) ---
    
    # Pareggiamo la lunghezza della lista testi con il dataframe originale processato finora
    # Se abbiamo interrotto a metà, 'testi' è più corta di 'df'
    righe_processate = len(testi)
    
    if righe_processate == 0:
        print("Nessuna canzone elaborata prima dell'interruzione. Esco.")
        return

    # Tagliamo il DataFrame originale per farlo combaciare con i testi scaricati
    df_parziale = df.iloc[:righe_processate].copy()
    df_parziale['lyrics'] = testi
    
    print(f"\n--- SALVATAGGIO PROGRESSI ({righe_processate} canzoni elaborate) ---")
    
    # 2. SALVATAGGIO ERRORI
    if lista_errori:
        df_errori = pd.DataFrame(lista_errori)
        df_errori.to_csv(ERROR_FILE, index=False)
        print(f"⚠️ Report errori salvato in: '{ERROR_FILE}'")
    else:
        print("✨ Nessun errore trovato finora.")

    # 3. ANALISI EMOZIONI E SALVATAGGIO FINALE
    df_clean = df_parziale.dropna(subset=['lyrics']).copy()
    print(f"Testi validi pronti per analisi sentiment: {len(df_clean)}")

    if not df_clean.empty:
        print("\n--- FASE 2: ANALISI EMOZIONI (Sui dati raccolti) ---")
        try:
            emotion_results = df_clean['lyrics'].apply(analizza_emozioni)
            emotion_df = emotion_results.apply(pd.Series)
            
            df_finale = pd.concat([df_clean, emotion_df], axis=1)
            
            cols = ['anno', 'rank', 'titolo', 'artista', 'lyrics']
            emo_cols = sorted(emotion_df.columns.tolist())
            final_cols = cols + emo_cols
            final_cols = [c for c in final_cols if c in df_finale.columns]
            
            df_finale[final_cols].to_csv(OUTPUT_FILE, index=False)
            print(f"\n✅ COMPLETATO! Dati parziali salvati in '{OUTPUT_FILE}'")
            
        except Exception as e:
            print(f"❌ Errore critico durante il salvataggio finale: {e}")
            # Salvataggio di emergenza senza sentiment se l'analisi fallisce
            df_clean.to_csv(f"EMERGENZA_{OUTPUT_FILE}", index=False)
            print("Dati grezzi salvati in file di emergenza.")
            
    else:
        print("Nessun testo valido trovato prima dell'interruzione.")

    if interrotto_manualmente:
        print("\n🚫 Script terminato dall'utente.")
        sys.exit(0)

if __name__ == "__main__":
    main()