
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


def plot_confusion_matrix(cm, classes, filepath, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix salvata: {filepath}")


def plot_performance_by_class(metrics_dict, filepath, title='Performance by Class'):
    classes = list(metrics_dict.keys())
    precision = [metrics_dict[c]['precision'] for c in classes]
    recall = [metrics_dict[c]['recall'] for c in classes]
    f1 = [metrics_dict[c]['f1-score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precision, width, label='Precision', color='skyblue')
    ax.bar(x, recall, width, label='Recall', color='lightcoral')
    ax.bar(x + width, f1, width, label='F1-Score', color='lightgreen')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance chart salvato: {filepath}")


def analizza_errori(df, y_true, y_pred, filepath, task_type='department'):
    errori_idx = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true != pred]
    
    if len(errori_idx) == 0:
        print(f"Nessun errore trovato per {task_type}.")
        return
    
    # Crea DataFrame errori
    errori_df = df.iloc[errori_idx].copy()
    errori_df['true_label'] = [y_true[i] for i in errori_idx]
    errori_df['predicted_label'] = [y_pred[i] for i in errori_idx]
    
    # Salva
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    errori_df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"Analisi errori salvata in: {filepath}")
    print(f"Totale errori {task_type}: {len(errori_idx)}")
    
    # Stampa alcuni esempi
    print(f"\n  Esempi di errori ({task_type}):")
    for i, row in errori_df.head(3).iterrows():
        print(f"    - '{row['title']}' → Vero: {row['true_label']}, Predetto: {row['predicted_label']}")


def stampa_metriche_complete(metrics, label="Model"):
    print(f"\n--- {label.upper()} METRICS ---")
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    if 'f1_macro' in metrics:
        print(f"F1 Macro: {metrics['f1_macro']:.4f}")
        print(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
    
    if 'f1_score' in metrics:
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
    
    print("\nClassification Report:")
    print(metrics['classification_report'])


def estrai_metriche_per_classe(classification_report_str):
    lines = classification_report_str.strip().split('\n')
    metrics = {}
    
    for line in lines[2:-5]:  # Salta header e summary
        parts = line.split()
        if len(parts) >= 5:
            class_name = parts[0]
            precision = float(parts[1])
            recall = float(parts[2])
            f1 = float(parts[3])
            
            metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1
            }
    
    return metrics



