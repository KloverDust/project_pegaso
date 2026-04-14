
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys
import joblib

# Import moduli locali
from preprocessor import preprocessa_dataset, crea_vectorizer
from department_classifier import DepartmentClassifier
from sentiment_analyzer import SentimentAnalyzer
from evaluation import (
    plot_confusion_matrix, 
    plot_performance_by_class,
    analizza_errori,
    stampa_metriche_complete,
    estrai_metriche_per_classe
)


# Configurazione paths
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data/generated/reviews.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Parametri
TEST_SIZE = 0.2
RANDOM_STATE = 42


def carica_dataset():
    print("Caricamento dataset...")
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    print(f"Dataset caricato: {len(df)} recensioni")
    return df


def prepara_dati(df):
    print("Preprocessing e split dei dati...")
    testi_puliti = preprocessa_dataset(df)
    print(f"{len(testi_puliti)} testi preprocessati")
    
    # Estrai labels
    y_department = df['department'].tolist()
    y_sentiment = df['sentiment'].tolist()
    
    # Train/test split
    # Usiamo lo stesso split per entrambi i task per consistency
    X_train_text, X_test_text, y_dept_train, y_dept_test, y_sent_train, y_sent_test, idx_train, idx_test = train_test_split(
        testi_puliti.tolist(), y_department, y_sentiment, df.index.tolist(),
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y_department  # Stratify per bilanciamento reparti
    )
    
    print(f"Training set: {len(X_train_text)} campioni")
    print(f"Test set: {len(X_test_text)} campioni")
    
    return X_train_text, X_test_text, y_dept_train, y_dept_test, y_sent_train, y_sent_test, idx_train, idx_test


def train_department_model(X_train_text, X_test_text, y_train, y_test):
    print("\nTraining Department Classifier...")
    vectorizer = crea_vectorizer(ngram_range=(1, 2), max_features=500, min_df=2)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"Vectorizer creato. Features shape: {X_train.shape}")
    
    # Training
    classifier = DepartmentClassifier(model_type='logistic')
    classifier.train(X_train, y_train)
    
    # Valutazione
    print("Valutazione modello...")
    metrics = classifier.evaluate(X_test, y_test)
    stampa_metriche_complete(metrics, "Department Classifier")
    
    classifier.save(f"{MODELS_DIR}/department_model.pkl")
    joblib.dump(vectorizer, f"{MODELS_DIR}/vectorizer_dept.pkl")
    print(f"Vectorizer salvato: {MODELS_DIR}/vectorizer_dept.pkl")
    
    return classifier, vectorizer, metrics


def train_sentiment_model(X_train_text, X_test_text, y_train, y_test):
    print("\nTraining Sentiment Analyzer...")
    vectorizer = crea_vectorizer(ngram_range=(1, 2), max_features=500, min_df=2)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"Vectorizer creato. Features shape: {X_train.shape}")
    
    # Training
    analyzer = SentimentAnalyzer()
    analyzer.train(X_train, y_train)
    
    # Valutazione
    print("Valutazione modello...")
    metrics = analyzer.evaluate(X_test, y_test)
    stampa_metriche_complete(metrics, "Sentiment Analyzer")
    
    analyzer.save(f"{MODELS_DIR}/sentiment_model.pkl")
    joblib.dump(vectorizer, f"{MODELS_DIR}/vectorizer_sent.pkl")
    print(f"Vectorizer salvato: {MODELS_DIR}/vectorizer_sent.pkl")
    
    return analyzer, vectorizer, metrics


def genera_visualizzazioni(dept_metrics, sent_metrics):
    print("\nGenerazione visualizzazioni...")
    
    # Confusion matrix - Department
    cm_dept = dept_metrics['confusion_matrix']
    classes_dept = ['F&B', 'Housekeeping', 'Reception']  # Ordine alfabetico
    plot_confusion_matrix(
        cm_dept, classes_dept, 
        f"{OUTPUTS_DIR}/confusion_matrix_department.png",
        "Confusion Matrix - Department Classification"
    )
    
    # Confusion matrix - Sentiment
    cm_sent = sent_metrics['confusion_matrix']
    classes_sent = ['negative', 'positive']
    plot_confusion_matrix(
        cm_sent, classes_sent,
        f"{OUTPUTS_DIR}/confusion_matrix_sentiment.png",
        "Confusion Matrix - Sentiment Analysis"
    )
    
    # Performance by class - Department
    dept_class_metrics = estrai_metriche_per_classe(dept_metrics['classification_report'])
    plot_performance_by_class(
        dept_class_metrics,
        f"{OUTPUTS_DIR}/performance_by_class_department.png",
        "Performance by Department"
    )
    
    print("Tutte le visualizzazioni generate")


def analizza_tutti_errori(df, idx_test, dept_metrics, sent_metrics):
    print("\nAnalisi errori...")
    
    df_test = df.iloc[idx_test].copy().reset_index(drop=True)
    
    # Errori department
    analizza_errori(
        df_test, 
        df_test['department'].values,
        dept_metrics['predictions'],
        f"{OUTPUTS_DIR}/errors_department.csv",
        task_type='department'
    )
    
    # Errori sentiment
    analizza_errori(
        df_test,
        df_test['sentiment'].values,
        sent_metrics['predictions'],
        f"{OUTPUTS_DIR}/errors_sentiment.csv",
        task_type='sentiment'
    )
    
    print("Analisi errori completata")


def main():
    print("Avvio pipeline ML...")
    
    # 1. Carica dataset
    df = carica_dataset()
    
    # 2. Prepara dati
    X_train_text, X_test_text, y_dept_train, y_dept_test, y_sent_train, y_sent_test, idx_train, idx_test = prepara_dati(df)
    
    # 3. Train department classifier
    dept_classifier, dept_vectorizer, dept_metrics = train_department_model(
        X_train_text, X_test_text, y_dept_train, y_dept_test
    )
    
    # 4. Train sentiment analyzer
    sent_analyzer, sent_vectorizer, sent_metrics = train_sentiment_model(
        X_train_text, X_test_text, y_sent_train, y_sent_test
    )
    
    # 5. Genera visualizzazioni
    genera_visualizzazioni(dept_metrics, sent_metrics)
    
    # 6. Analizza errori
    analizza_tutti_errori(df, idx_test, dept_metrics, sent_metrics)
    
    print("\nPipeline completata.")
    print("Risultati finali:")
    print("Department Classifier:")
    print(f"- Accuracy: {dept_metrics['accuracy']:.2%}")
    print(f"- F1 Macro: {dept_metrics['f1_macro']:.4f}")
    print("Sentiment Analyzer:")
    print(f"- Accuracy: {sent_metrics['accuracy']:.2%}")
    print(f"- F1 Score: {sent_metrics['f1_score']:.4f}")
    
    print("\nProssimo passaggio: Lanciare la dashboard con 'streamlit run app.py'")


if __name__ == "__main__":
    main()
