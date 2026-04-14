
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import os
import sys
from typing import Dict, Any, TypedDict


class PredictionResult(TypedDict):
    """Struttura del risultato della predizione."""
    department: str
    department_confidence: float
    department_probabilities: Dict[str, float]
    sentiment: str
    sentiment_confidence: float
    sentiment_probabilities: Dict[str, float]

# Aggiungi la directory src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessor import pulisci_testo

# Configurazione pagina
st.set_page_config(
    page_title="Hotel Review Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths modelli
MODELS_DIR = "models"
DEPT_MODEL_PATH = f"{MODELS_DIR}/department_model.pkl"
SENT_MODEL_PATH = f"{MODELS_DIR}/sentiment_model.pkl"
DEPT_VECTORIZER_PATH = f"{MODELS_DIR}/vectorizer_dept.pkl"
SENT_VECTORIZER_PATH = f"{MODELS_DIR}/vectorizer_sent.pkl"


@st.cache_resource
def carica_modelli():
    try:
        dept_model = joblib.load(DEPT_MODEL_PATH)
        sent_model = joblib.load(SENT_MODEL_PATH)
        dept_vectorizer = joblib.load(DEPT_VECTORIZER_PATH)
        sent_vectorizer = joblib.load(SENT_VECTORIZER_PATH)
        return dept_model, sent_model, dept_vectorizer, sent_vectorizer
    except FileNotFoundError:
        st.error("Modelli non trovati. Eseguire pipeline.py per addestrare i modelli.")
        st.stop()


def preprocessa_testo(titolo, corpo):
    testo_combinato = f"{titolo} {titolo} {corpo}"
    testo_pulito = pulisci_testo(testo_combinato)
    return testo_pulito


def predici_recensione(titolo, corpo, dept_model, sent_model, dept_vec, sent_vec) -> PredictionResult:
    # Preprocessa
    testo = preprocessa_testo(titolo, corpo)
    
    # Vectorize
    X_dept = dept_vec.transform([testo])
    X_sent = sent_vec.transform([testo])
    
    # Predizioni
    dept_pred = dept_model.predict(X_dept)[0]
    dept_proba = dept_model.predict_proba(X_dept)[0]
    
    sent_pred = sent_model.predict(X_sent)[0]
    sent_proba = sent_model.predict_proba(X_sent)[0]
    
    # Costruisci risultato
    dept_idx = list(dept_model.classes_).index(dept_pred)
    sent_idx = list(sent_model.classes_).index(sent_pred)
    
    return {
        'department': dept_pred,
        'department_confidence': dept_proba[dept_idx] * 100,
        'department_probabilities': dict(zip(dept_model.classes_, dept_proba * 100)),
        'sentiment': sent_pred,
        'sentiment_confidence': sent_proba[sent_idx] * 100,
        'sentiment_probabilities': dict(zip(sent_model.classes_, sent_proba * 100))
    }


def main():
    
    # Header
    st.title("Hotel Review Classifier")
    st.markdown("### Sistema di smistamento recensioni hotel")
    st.markdown("---")
    
    # Carica modelli
    with st.spinner("Caricamento modelli..."):
        dept_model, sent_model, dept_vec, sent_vec = carica_modelli()
    
    # Sidebar con info
    with st.sidebar:
        st.header("Informazioni Sistema")
        st.metric("Reparti", "3")
        st.metric("Accuracy Classificatore", "100%")
        st.metric("Accuracy Sentiment", "100%")
        
        st.markdown("---")
        st.markdown("### Come usare")
        st.markdown("""
        **Modalità Singola:**
        - Inserisci titolo e testo recensione
        - Clicca 'Analizza Recensione'
        - Visualizza reparto e sentiment
        
        **Modalità Batch:**
        - Carica file CSV con colonne `title` e `body`
        - Visualizza preview predizioni
        - Scarica risultati completi
        """)
        
    # Tab per le due modalità
    tab1, tab2 = st.tabs(["Predizione Singola", "Batch Processing"])
    
    # TAB 1: Predizione singola
    with tab1:
        st.header("Analisi rapida")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            titolo = st.text_input(
                "Titolo recensione",
                placeholder="Es: Camera pulitissima",
                help="Inserisci il titolo della recensione"
            )
            
            corpo = st.text_area(
                "Testo recensione",
                placeholder="Es: La camera era impeccabile, lenzuola fresche e bagno splendido.",
                help="Inserisci il corpo della recensione",
                height=150
            )
            
            analyze_button = st.button("Analizza", type="primary", use_container_width=True)
        
        with col2:
            if analyze_button and titolo and corpo:
                with st.spinner("Analisi in corso..."):
                    risultato = predici_recensione(titolo, corpo, dept_model, sent_model, dept_vec, sent_vec)
                
                # Mostra risultati
                st.success("Analisi completata!")
                
                # Department
                st.markdown("#### Reparto")
                st.markdown(f"## {risultato['department']}")
                st.progress(risultato['department_confidence'] / 100)
                st.caption(f"Confidenza: {risultato['department_confidence']:.1f}%")
                
                with st.expander("Vedi probabilità per tutti i reparti"):
                    for dept, prob in risultato['department_probabilities'].items():
                        st.write(f"**{dept}:** {prob:.1f}%")
                
                st.markdown("---")
                
                # Sentiment
                st.markdown("#### Sentiment")
                sent_color = 'green' if risultato['sentiment'] == 'positive' else 'red'
                st.markdown(f"## :{sent_color}[{risultato['sentiment'].upper()}]")
                st.progress(risultato['sentiment_confidence'] / 100)
                st.caption(f"Confidenza: {risultato['sentiment_confidence']:.1f}%")
                
                with st.expander("Vedi probabilità per entrambi i sentiment"):
                    for sent, prob in risultato['sentiment_probabilities'].items():
                        st.write(f"**{sent}:** {prob:.1f}%")
                        
            elif analyze_button:
                st.warning("Inserisci titolo e corpo della recensione.")
    
    # TAB 2: Batch processing
    with tab2:
        st.header("Processa un batch di recensioni")
        
        uploaded_file = st.file_uploader(
            "Carica file CSV",
            type=['csv'],
            help="Il file deve contenere le colonne 'title' e 'body'"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validazione
                if 'title' not in df.columns or 'body' not in df.columns:
                    st.error("Il file deve contenere le colonne 'title' e 'body'.")
                    st.stop()
                
                st.success(f"File caricato ({len(df)} righe)")
                
                with st.expander("Preview dataset"):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Pulsante predizione
                if st.button("Esegui Predizioni", type="primary"):
                    with st.spinner(f"Processando {len(df)} recensioni..."):
                        risultati = []
                        
                        progress_bar = st.progress(0)
                        for idx, row in df.iterrows():
                            res = predici_recensione(
                                row['title'], row['body'],
                                dept_model, sent_model, dept_vec, sent_vec
                            )
                            risultati.append(res)
                            progress_bar.progress((idx + 1) / len(df))
                    
                    # Crea DataFrame risultati
                    df_risultati = df.copy()
                    df_risultati['predicted_department'] = [r['department'] for r in risultati]
                    df_risultati['department_confidence'] = [f"{r['department_confidence']:.1f}%" for r in risultati]
                    df_risultati['predicted_sentiment'] = [r['sentiment'] for r in risultati]
                    df_risultati['sentiment_confidence'] = [f"{r['sentiment_confidence']:.1f}%" for r in risultati]
                    
                    st.success("Elaborazione completata.")
                    st.markdown("### Risultati")
                    st.dataframe(df_risultati, use_container_width=True)
                    
                    # Statistiche
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        dept_counts = df_risultati['predicted_department'].value_counts()
                        st.metric("Reparto più frequente", dept_counts.index[0], f"{dept_counts.iloc[0]} recensioni")
                    with col2:
                        sent_counts = df_risultati['predicted_sentiment'].value_counts()
                        st.metric("Sentiment prevalente", sent_counts.index[0], f"{sent_counts.iloc[0]} recensioni")
                    with col3:
                        st.metric("Totale processate", len(df_risultati))
                    
                    # Download
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv = df_risultati.to_csv(index=False, encoding='utf-8')
                    st.download_button(
                        label="Scarica File",
                        data=csv,
                        file_name=f"predictions_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"Errore caricamento: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Sistema ML per Smistamento Recensioni Hotel | "
        "Project Work L-31 | 2026"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
