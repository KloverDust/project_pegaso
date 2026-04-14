# Hotel Review Classifier

Sistema di Machine Learning per lo smistamento automatico delle recensioni di strutture ricettive verso i reparti appropriati (Housekeeping, Reception, F&B) con analisi del sentiment.

## Descrizione

Questo progetto implementa una soluzione completa per:
- **Classificazione Automatica**: Assegnazione recensioni ai reparti competenti
- **Analisi Sentiment**: Identificazione recensioni positive/negative  
- **Dashboard Interattiva**: Interfaccia web per predizioni in tempo reale
- **Batch Processing**: Elaborazione massive di recensioni da file CSV

## Quick Start

### 1. Installazione Dipendenze

```bash
# Crea ambiente virtuale
python3 -m venv venv
source venv/bin/activate

# Installa dipendenze
pip install -r requirements.txt
```

### 2. Genera Dataset Sintetico

```bash
python src/data_generator.py
```

Output: `data/generated/reviews.csv` con 400 recensioni sintetiche

### 3. Addestra Modelli ML

```bash
cd src
python pipeline.py
```

Questo comando:
- Preprocessa i testi
- Addestra classificatore reparto e ​​analizzatore sentiment
- Genera metriche e visualizzazioni
- Salva modelli in `models/`

### 4. Lancia Dashboard

```bash
cd ..
streamlit run app.py
```

La dashboard sarà disponibile su `http://localhost:8501`

## Struttura Progetto

```
project_pegaso/
├── README.md                           # Questo file
├── requirements.txt                    # Dipendenze Python
├── app.py                             # Dashboard Streamlit
├── data/
│   ├── generated/
│   │   └── reviews.csv                # Dataset sintetico
│   └── predictions/                   # Risultati batch
├── src/
│   ├── data_generator.py              # Generatore dataset
│   ├── preprocessor.py                # Preprocessing testi
│   ├── department_classifier.py       # Classificatore reparto
│   ├── sentiment_analyzer.py          # Analizzatore sentiment
│   ├── pipeline.py                    # Pipeline training
│   └── evaluation.py                  # Valutazione e grafici
├── models/
│   ├── department_model.pkl           # Modello reparto
│   ├── sentiment_model.pkl            # Modello sentiment
│   ├── vectorizer_dept.pkl            # TF-IDF vectorizer reparto
│   └── vectorizer_sent.pkl            # TF-IDF vectorizer sentiment
├── outputs/
│   ├── confusion_matrix_department.png
│   ├── confusion_matrix_sentiment.png
│   └── performance_by_class_department.png
└── docs/
    ├── REPORT.md                      # Report tecnico
    ├── specifics.docx                 # Specifica progetto
    └── project_esempio.pdf            # Esempio riferimento
```

## Utilizzo Dashboard

### Modalità Predizione Singola

1. Inserisci titolo e testo della recensione
2. Clicca "Analizza Recensione"
3. Visualizza:
   - Reparto consigliato con confidenza
   - Sentiment (positivo/negativo) con probabilità
   - Distribuzione probabilità per tutte le classi

### Modalità Batch Processing

1. Prepara file CSV con colonne `title` e `body`
2. Carica il file tramite l'interfaccia
3. Clicca "Esegui Predizioni"
4. Scarica risultati con timestamp

Esempio formato CSV:
```csv
title,body
Camera pulita,La stanza era impeccabile e profumata
Check-in lento,Attesa troppo lunga alla reception
Colazione scarsa,Buffet limitato e cibo freddo
```

## Tecnologie Utilizzate

- **Python 3.8+**: Linguaggio principale
- **Scikit-learn**: Modelli ML (Logistic Regression, TF-IDF)
- **Pandas/NumPy**: Manipolazione dati
- **Matplotlib/Seaborn**: Visualizzazioni
- **Streamlit**: Dashboard interattiva
- **Joblib**: Serializzazione modelli

## Performance Modelli

| Modello | Accuracy | F1 Score |
|---------|----------|----------|
| Department Classifier | 100% | 1.000 |
| Sentiment Analyzer | 100% | 1.000 |

*Valutati su test set (80/20 split)*

## Dettagli Tecnici

### Preprocessing
- Lowercasing
- Rimozione punteggiatura
- TF-IDF vectorization (bi-grammi, max 500 features)
- Combinazione titolo + corpo (peso maggiore al titolo)

### Modelli
- **Reparto**: Logistic Regression multi-classe
- **Sentiment**: Logistic Regression binaria
- Regolarizzazione L2 (C=1.0)
- Random state fisso (42) per riproducibilità

### Dataset
- 400 recensioni sintetiche
- Distribuzione bilanciata:
  - Reparti: ~33% ciascuno
  - Sentiment: ~50% positivo, 50% negativo
- 10% recensioni ambigue per robustezza

## Comandi Utili

```bash
# Test singoli moduli
python src/preprocessor.py
python src/department_classifier.py
python src/sentiment_analyzer.py

# Rigenera dataset (nuovo seed)
python src/data_generator.py

# Visualizza prime righe dataset
head -20 data/generated/reviews.csv

# Check modelli salvati
ls -lh models/
```

## Note

- **Dati Sintetici**: Il dataset è completamente artificiale, nessun dato personale
- **Riproducibilità**: Seed fisso garantisce risultati consistenti
- **Estensibilità**: Architettura modulare facilita aggiunta nuovi modelli
- **Lingua**: Recensioni in italiano, ottimizzato per il contesto hospitality

## Troubleshooting

**Problema**: ModuleNotFoundError
```bash
# Soluzione: Verifica ambiente virtuale attivo
source venv/bin/activate
pip install -r requirements.txt
```

**Problema**: Modelli non trovati nella dashboard
```bash
# Soluzione: Addestra prima i modelli
cd src && python pipeline.py
```

**Problema**: Port già in uso (Streamlit)
```bash
# Soluzione: Usa port diverso
streamlit run app.py --server.port 8502
```

## Autori
Creatore: Oleksandr Chumak
Project Work - Corso di Laurea L-31  
Tema: Machine Learning per Processi Aziendali  
Anno: 2026

## Licenza

Progetto didattico - Tutti i diritti riservatigit 