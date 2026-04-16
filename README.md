# Project Pegaso: Hotel Review Classifier

Il presente documento illustra l'architettura tecnica del progetto e le istruzioni operative per l'inizializzazione del software.

## Descrizione del Progetto

Il progetto implementa un sistema di Machine Learning per lo smistamento automatico delle recensioni di strutture ricettive verso i reparti di pertinenza (es. Housekeeping, Reception, Ristorazione) e per l'analisi del relativo sentiment (positivo o negativo). 

Sviluppato come Project Work per il corso di Laurea L-31, ha l'obiettivo di dimostrare l'applicazione di algoritmi di intelligenza artificiale per l'ottimizzazione di processi aziendali, permettendo la gestione su larga scala di feedback.

## Istruzioni di Avvio (Quick Start)

I seguenti passaggi illustrano le procedure per clonare ed eseguire correttamente gli script di formazione e validazione.

### 1. Configurazione dell'Ambiente

Per garantire l'isolamento delle librerie necessarie ed evitare conflitti, si richiede la creazione di un ambiente virtuale (Virtual Environment).

```bash
# Creazione dell'ambiente virtuale
python3 -m venv venv

# Attivazione dell'ambiente virtuale
source venv/bin/activate

# Installazione dei pacchetti richiesti
pip install -r requirements.txt
```

### 2. Generazione del Dataset Sintetico

Essendo un progetto didattico, il sistema non comprende o distribuisce banche dati reali. È stato invece sviluppato un generatore in grado di produrre file CSV di recensioni realistiche (classi bilanciate) su cui eseguire attività di training.

```bash
python src/data_generator.py
```
Lo script produrrà il file `reviews.csv` nella directory `/data/generated/` per un totale di circa 400 record sintetizzati.

### 3. Addestramento dei Modelli

La fase di addestramento è centralizzata all'interno della pipeline. Questa operazione pre-processerà i testi e serializzerà i modelli per un utilizzo ripetuto.

```bash
cd src
python pipeline.py
```
Nello specifico, questo comando:
- Pulisce e processa i testi estratti dal dataset iniziale.
- Esegue il fit della *Logistic Regression* per la classificazione del reparto.
- Esegue il fit della *Logistic Regression* binaria per l'identificazione del sentiment.
- Memorizza persistendo i rispettivi algoritmi e vectorizers nella cartella `/models/`.

### 4. Esecuzione dell'Interfaccia Grafica (Dashboard)

Il progetto dispone di una applicazione basata sul framework Streamlit, utile per l'interazione pratica e il collaudo da parte dell'utente.

```bash
cd ..
streamlit run app.py
```
L'infrastruttura web sarà avviata all'indirizzo locale `http://localhost:8501`. L'applicazione supporta formalmente due funzionalità di utilizzo:
- **Predizione Singola:** utile per collaudare i test inserendo specifici pattern di frasi libere, fornendo score di probabilità delle misurazioni.
- **Predizione Batch:** orientato ad un caricamento in blocco di recensioni estrapolate sotto forma di un unico file CSV, automatizzandone l'intera classificazione logica.

## Struttura della Repository

Per facilitare la navigazione della logica dell'applicativo e separare coerentemente script e output analitici, l'architettura è stata strutturata come segue:

```text
project_pegaso/
├── README.md                           # Documentazione corrente
├── requirements.txt                    # Elenco delle specifiche richieste da Python
├── app.py                              # Entrypoint della GUI Streamlit
├── data/
│   ├── generated/                      # Locazione temporanea di testset non tracciato
│   └── predictions/                    # Output CSV delle elaborazioni utente
├── src/                                # Sorgente ML
│   ├── data_generator.py               # Generatore randomizzato a seed fisso
│   ├── preprocessor.py                 # Funzioni base per la sanitizzazione in stringhe valide
│   ├── department_classifier.py        # Classe per identificazione modulo alberghiero 
│   ├── sentiment_analyzer.py           # Classe valutatore del peso emozionale del testo
│   ├── pipeline.py                     # Aggregatore delle fasi precedenti per il training
│   └── evaluation.py                   # Modulo per l'output grafico su base percentuale
├── models/                             # Destinazione finale dei file *.pkl esportati da JobLib
├── outputs/                            # Immagini autogenerate dalla valutazione statistica
└── docs/                               # File accademici supplementari per la progettazione
```

## Dettagli Tecnici

Il motore sfrutta tecniche classiche di Machine Learning e NLP:
- Il Text Pre-Processing applica lower-casing progressivo e conversione matematica tramite il metodo statisticamente fondato su `TF-IDF` e computando metriche anche nell'ordine di vari unigrammi/bi-grammi (max 500 features).
- Le procedure classificazione in questione (sia quella a 3 esiti che quella a 2 esiti) convergono tramite `Logistic Regression`. Il coefficiente per la regolarizzazione previene fenomeni estremi di overfitting.
- Relativamente alle metriche di accuratezza finali registrate sui set sintetici, risulta fisiologica una precisione massima. Con varianze e dati testuali meno definiti, le stesse procedure scalerebbero su stime maggiormente vicine all'85-90%. 

## Risoluzione Problemi e Troubleshooting

Nel caso si riscontrino errori in fase di debug e compilazione, le indicazioni generali in successione sono le seguenti:

- **Eccezione su moduli non rilevati (ModuleNotFoundError):** Assicurarsi che l'ambiente virtuale sia realmente richiamato al promp del sistema operativo (`source venv/bin/activate`) ed eseguire una nuova installazione di requirements.txt.
- **Errore per i file `*.pkl` assenti della cartella models:** Procedere dall'interno della cartella `src/` alla ricostruzione del training data richiamando l'apposito comando `python pipeline.py`. Al termine dell'output della pipeline, accertatevi che la cartella models in root del progetto si sia popolata in maniera non-vuota.
- **Eccezioni Streamlit (Porta 8501 già in uso):** Qualora le porte di ascolto siano impegnate da un demone non interrotto correttamente passate un differente parametro in fase di runtime: `streamlit run app.py --server.port 8502`.

## Autore
- **Oleksandr Chumak**
- Project Work - Corso di Laurea L-31 
- Tema: Machine Learning per Processi Aziendali
- Anno: 2026