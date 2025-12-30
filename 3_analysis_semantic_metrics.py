import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os

# --- CONFIGURAZIONE ---
INPUT_CSV = "risultati_sentiment_1974_2024.csv"  # Il tuo file completo
OUTPUT_YEARLY = "metriche_semantiche_annuali.csv"
OUTPUT_DECADES = "metriche_semantiche_decenni_full.csv"

# Modello
MODELLO_NAME = 'sentence-transformers/all-mpnet-base-v2'

# Parametri Chunking (per leggere tutto il testo)
CHUNK_SIZE_WORDS = 250  # Parole per blocco (stima sicura per stare nei 384 token)

def check_gpu():
    if torch.cuda.is_available():
        print(f">>> GPU RILEVATA: {torch.cuda.get_device_name(0)}")
        return 'cuda'
    return 'cpu'

def get_song_embedding_full(model, text):
    """
    Spezza il testo in chunk, calcola l'embedding per ognuno e ne fa la media.
    Garantisce che tutto il testo venga analizzato.
    """
    if not isinstance(text, str) or len(text) < 5:
        # Ritorna un vettore di zeri se il testo è vuoto (caso limite)
        return np.zeros(model.get_sentence_embedding_dimension())
        
    words = text.split()
    
    # Se il testo è breve, encoding diretto (più veloce)
    if len(words) <= CHUNK_SIZE_WORDS:
        return model.encode(text)
    
    # Altrimenti, chunking
    chunks = [" ".join(words[i: i + CHUNK_SIZE_WORDS]) for i in range(0, len(words), CHUNK_SIZE_WORDS)]
    
    # Calcola embeddings per tutti i chunk
    chunk_embeddings = model.encode(chunks)
    
    # Media dei vettori (Pooling)
    # axis=0 fa la media colonna per colonna, restituendo un unico vettore
    song_vector = np.mean(chunk_embeddings, axis=0)
    
    return song_vector

def calculate_metrics():
    if not os.path.exists(INPUT_CSV):
        print(f"ERRORE: {INPUT_CSV} non trovato.")
        return

    print(f"Caricamento dati...")
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=['lyrics'])
    print(f"Canzoni totali: {len(df)}")
    
    device = check_gpu()
    print(f"Caricamento modello '{MODELLO_NAME}'...")
    model = SentenceTransformer(MODELLO_NAME, device=device)
    
    # --- FASE 1: CALCOLO EMBEDDINGS COMPLETI ---
    print("\n>>> Calcolo vettori semantici (Full Text)...")
    # Nota: Non possiamo usare model.encode(lista) direttamente per il full text custom.
    # Dobbiamo iterare. Con la 4060 sarà veloce comunque.
    
    all_embeddings = []
    
    # Iteriamo con progress bar semplice
    total = len(df)
    for i, row in df.iterrows():
        if i % 100 == 0: print(f"Processing {i}/{total}...", end='\r')
        emb = get_song_embedding_full(model, row['lyrics'])
        all_embeddings.append(emb)
    
    print(f"\nEncoding completato. Creazione matrice numpy...")
    embedding_matrix = np.vstack(all_embeddings) # Matrice (N_canzoni x 768)
    
    # Aggiungiamo i vettori al dataframe per facilitare i raggruppamenti dopo
    # (Non salviamo questo nel CSV perché sarebbe enorme, lo usiamo in memoria)
    df['embedding'] = list(embedding_matrix)
    
    # --- FASE 2: ANALISI ANNUALE ---
    print("\n>>> Analisi Annuale (Omogeneità e Stabilità)...")
    yearly_metrics = []
    years = sorted(df['anno'].unique())
    prev_centroid = None
    
    for year in years:
        # Filtriamo i vettori dell'anno corrente
        # np.stack serve a convertire la lista di array in una matrice 2D
        year_embs = np.stack(df[df['anno'] == year]['embedding'].values)
        
        # 1. Omogeneità Interna
        sim_matrix = cosine_similarity(year_embs)
        upper_tri = np.triu_indices_from(sim_matrix, k=1)
        homogeneity = np.mean(sim_matrix[upper_tri])
        
        # 2. Similarità col passato
        current_centroid = np.mean(year_embs, axis=0).reshape(1, -1)
        sim_past = None
        if prev_centroid is not None:
            sim_past = cosine_similarity(current_centroid, prev_centroid)[0][0]
            
        yearly_metrics.append({
            'anno': year,
            'omogeneita_interna': homogeneity,
            'similarita_anno_prev': sim_past
        })
        prev_centroid = current_centroid
        
    pd.DataFrame(yearly_metrics).to_csv(OUTPUT_YEARLY, index=False)
    print(f"Salvato: {OUTPUT_YEARLY}")

    # --- FASE 3: ANALISI DECENNI ---
    print("\n>>> Analisi per Decenni...")
    
    # Creiamo colonna decennio
    # Es. 1974 -> 1970, 1986 -> 1980
    df['decennio'] = (df['anno'] // 10) * 10
    
    decade_metrics = []
    decades = sorted(df['decennio'].unique())
    prev_dec_centroid = None
    prev_dec_label = None
    
    for dec in decades:
        label = f"Anni {dec}"
        print(f"Elaborazione {label}...")
        
        # Vettori del decennio
        dec_embs = np.stack(df[df['decennio'] == dec]['embedding'].values)
        
        # 1. Omogeneità del decennio (quanto sono varie le canzoni in 10 anni?)
        # Nota: su 1000 canzoni la matrice è grande (1000x1000), può richiedere un attimo
        if len(dec_embs) > 5000: # Se troppe canzoni, campioniamo per velocità
             indices = np.random.choice(len(dec_embs), 5000, replace=False)
             sim_matrix_dec = cosine_similarity(dec_embs[indices])
        else:
             sim_matrix_dec = cosine_similarity(dec_embs)
             
        upper_tri_dec = np.triu_indices_from(sim_matrix_dec, k=1)
        homogeneity_dec = np.mean(sim_matrix_dec[upper_tri_dec])
        
        # 2. Confronto con decennio precedente
        curr_dec_centroid = np.mean(dec_embs, axis=0).reshape(1, -1)
        sim_prev_dec = None
        
        if prev_dec_centroid is not None:
            sim_prev_dec = cosine_similarity(curr_dec_centroid, prev_dec_centroid)[0][0]
            print(f"   Vs {prev_dec_label}: {sim_prev_dec:.4f}")
            
        decade_metrics.append({
            'decennio': label,
            'num_canzoni': len(dec_embs),
            'omogeneita_totale': homogeneity_dec,
            'similarita_decennio_precedente': sim_prev_dec
        })
        
        prev_dec_centroid = curr_dec_centroid
        prev_dec_label = label
        
    pd.DataFrame(decade_metrics).to_csv(OUTPUT_DECADES, index=False)
    print(f"Salvato: {OUTPUT_DECADES}")
    print("\n✅ Tutto completato.")

if __name__ == "__main__":
    calculate_metrics()