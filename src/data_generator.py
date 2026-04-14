

import pandas as pd
import random
import os
from datetime import datetime

# Seed per riproducibilità
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Configurazione
NUM_REVIEWS = 400  # Numero target di recensioni (tra 200-500)
OUTPUT_PATH = "data/generated/reviews.csv"

# Dizionari di parole chiave e frasi per reparto

HOUSEKEEPING_KEYWORDS = {
    'positivo': {
        'aggettivi': ['pulita', 'pulitissima', 'splendida', 'impeccabile', 'ordinata', 'profumata', 'fresca'],
        'sostantivi': ['camera', 'bagno', 'lenzuola', 'asciugamani', 'pulizia', 'igiene', 'ordine'],
        'verbi': ['pulire', 'sistemare', 'cambiare', 'riordinare', 'profumare'],
        'template_titoli': [
            'Camera impeccabile',
            'Pulizia perfetta',
            'Stanza pulitissima',
            'Igiene eccellente',
            'Ordine e pulizia'
        ],
        'template_corpi': [
            'La camera era {agg}, {sost} {agg2} e tutto perfettamente in ordine.',
            '{sost} {agg} e {sost2} {agg2}, veramente soddisfatto della pulizia.',
            'Personale delle pulizie eccellente, {sost} sempre {agg}.',
            'Non ho mai visto una {sost} così {agg}, complimenti!',
            '{sost} {agg2}, {sost2} profumate, tutto perfetto.'
        ]
    },
    'negativo': {
        'aggettivi': ['sporca', 'trascurata', 'vecchia', 'macchiata', 'puzzolente', 'disordinata', 'mal tenuta'],
        'sostantivi': ['camera', 'bagno', 'lenzuola', 'asciugamani', 'pavimento', 'tappeto'],
        'template_titoli': [
            'Camera sporca',
            'Pulizia pessima',
            'Igiene scarsa',
            'Stanza trascurata',
            'Disordine ovunque'
        ],
        'template_corpi': [
            'La camera era {agg}, {sost} {agg2} e un odore sgradevole.',
            '{sost} {agg} e {sost2} {agg2}, non accettabile.',
            'Pulizia inesistente, {sost} completamente {agg}.',
            '{sost} con macchie, {sost2} {agg2}, molto deluso.',
            'Standard igienico bassissimo, {sost} {agg}.'
        ]
    }
}

RECEPTION_KEYWORDS = {
    'positivo': {
        'aggettivi': ['gentile', 'disponibile', 'cortese', 'professionale', 'efficiente', 'cordiale', 'veloce'],
        'sostantivi': ['personale', 'reception', 'check-in', 'check-out', 'receptionist', 'servizio', 'staff'],
        'template_titoli': [
            'Personale eccellente',
            'Check-in velocissimo',
            'Reception cordiale',
            'Servizio impeccabile',
            'Staff professionale'
        ],
        'template_corpi': [
            'Il {sost} è stato {agg}, {sost2} gestito in modo {agg2}.',
            '{sost} molto {agg}, hanno risolto ogni problema con {agg2}.',
            '{sost2} {agg}, procedura di {sost} molto rapida.',
            'Complimenti al {sost}, sempre {agg} e {agg2}.',
            '{sost} {agg2}, ottima esperienza al momento del {sost2}.'
        ]
    },
    'negativo': {
        'aggettivi': ['scortese', 'lento', 'inefficiente', 'disorganizzato', 'maleducato', 'arrogante', 'incompetente'],
        'sostantivi': ['personale', 'reception', 'check-in', 'check-out', 'receptionist', 'prenotazione', 'pagamento'],
        'template_titoli': [
            'Reception pessima',
            'Check-in lentissimo',
            'Personale scortese',
            'Servizio disorganizzato',
            'Staff maleducato'
        ],
        'template_corpi': [
            'Il {sost} è stato {agg}, {sost2} gestito malissimo.',
            '{sost} molto {agg}, problemi con la {sost2}.',
            '{sost2} {agg}, attesa interminabile per il {sost}.',
            '{sost} completamente {agg}, esperienza negativa.',
            'Errori nella {sost2}, {sost} {agg} e {agg2}.'
        ]
    }
}

FB_KEYWORDS = {
    'positivo': {
        'aggettivi': ['ottima', 'buonissima', 'eccellente', 'deliziosa', 'varia', 'abbondante', 'fresca'],
        'sostantivi': ['colazione', 'ristorante', 'cibo', 'buffet', 'caffè', 'menu', 'cucina'],
        'template_titoli': [
            'Colazione fantastica',
            'Ristorante eccellente',
            'Cibo delizioso',
            'Buffet ricchissimo',
            'Cucina ottima'
        ],
        'template_corpi': [
            'La {sost} era {agg}, {sost2} {agg2} e di qualità.',
            '{sost} molto {agg}, ampia scelta al {sost2}.',
            '{sost2} {agg2}, {sost} veramente {agg}.',
            'Complimenti per la {sost}, tutto {agg} e {agg2}.',
            '{sost} {agg}, il {sost2} era {agg2}.'
        ]
    },
    'negativo': {
        'aggettivi': ['scarsa', 'pessima', 'fredda', 'limitata', 'cattiva', 'scadente', 'povera'],
        'sostantivi': ['colazione', 'ristorante', 'cibo', 'buffet', 'caffè', 'menu', 'cucina'],
        'template_titoli': [
            'Colazione scarsa',
            'Ristorante deludente',
            'Cibo pessimo',
            'Buffet limitato',
            'Cucina scadente'
        ],
        'template_corpi': [
            'La {sost} era {agg}, {sost2} {agg2} e poco invitante.',
            '{sost} molto {agg}, scelta al {sost2} {agg2}.',
            '{sost2} di qualità {agg}, {sost} {agg2}.',
            'Deluso dalla {sost}, tutto {agg} e {agg2}.',
            '{sost} {agg2}, il {sost2} era {agg}.'
        ]
    }
}


def genera_recensione(reparto, sentiment):
    if reparto == 'Housekeeping':
        keywords = HOUSEKEEPING_KEYWORDS
    elif reparto == 'Reception':
        keywords = RECEPTION_KEYWORDS
    else:  # F&B
        keywords = FB_KEYWORDS
    
    sentiment_key = 'positivo' if sentiment == 'positive' else 'negativo'
    data = keywords[sentiment_key]
    
    titolo = random.choice(data['template_titoli'])
    template = random.choice(data['template_corpi'])
    
    corpo = template
    if '{agg}' in corpo:
        corpo = corpo.replace('{agg}', random.choice(data['aggettivi']), 1)
    if '{agg2}' in corpo:
        corpo = corpo.replace('{agg2}', random.choice(data['aggettivi']), 1)
    if '{sost}' in corpo:
        corpo = corpo.replace('{sost}', random.choice(data['sostantivi']), 1)
    if '{sost2}' in corpo:
        corpo = corpo.replace('{sost2}', random.choice(data['sostantivi']), 1)
    
    return titolo, corpo


def genera_recensione_ambigua():
    casi_ambigui = [
        {
            'titolo': 'Bella struttura ma problemi vari',
            'corpo': 'Camera pulita ma personale poco cortese al check-in',
            'reparto': 'Reception',
            'sentiment': 'negative'
        },
        {
            'titolo': 'Esperienza mista',
            'corpo': 'Ottima colazione e camera ordinata, peccato per il check-out lento',
            'reparto': 'F&B',
            'sentiment': 'positive'
        },
        {
            'titolo': 'Pulizia ok, resto no',
            'corpo': 'Stanza pulita ma colazione scarsa e personale freddo',
            'reparto': 'Housekeeping',
            'sentiment': 'negative'
        },
        {
            'titolo': 'Buon soggiorno nel complesso',
            'corpo': 'Reception gentile e buffet ricco, solo il bagno un po\' datato',
            'reparto': 'Reception',
            'sentiment': 'positive'
        },
        {
            'titolo': 'Servizio altalenante',
            'corpo': 'Check-in veloce e camera spaziosa, ma ristorante deludente',
            'reparto': 'Reception',
            'sentiment': 'positive'
        },
        {
            'titolo': 'Potrebbe migliorare',
            'corpo': 'Colazione buona ma lenzuola macchiate e reception disorganizzata',
            'reparto': 'F&B',
            'sentiment': 'negative'
        }
    ]
    
    caso = random.choice(casi_ambigui)
    return caso['titolo'], caso['corpo'], caso['reparto'], caso['sentiment']


def genera_dataset(num_reviews=NUM_REVIEWS):
    reparti = ['Housekeeping', 'Reception', 'F&B']
    sentiments = ['positive', 'negative']
    
    reviews = []
    
    num_standard = int(num_reviews * 0.9)
    num_ambigue = num_reviews - num_standard
    
    print(f"Generazione di {num_standard} recensioni standard...")
    for i in range(num_standard):
        reparto = reparti[i % 3]  # Distribuisci equamente tra reparti
        sentiment = sentiments[i % 2]  # Alterna pos/neg
        
        titolo, corpo = genera_recensione(reparto, sentiment)
        
        reviews.append({
            'id': f'REV_{i+1:04d}',
            'title': titolo,
            'body': corpo,
            'department': reparto,
            'sentiment': sentiment
        })
    
    print(f"Generazione di {num_ambigue} recensioni ambigue...")
    for i in range(num_ambigue):
        titolo, corpo, reparto, sentiment = genera_recensione_ambigua()
        
        reviews.append({
            'id': f'REV_{num_standard + i + 1:04d}',
            'title': titolo,
            'body': corpo,
            'department': reparto,
            'sentiment': sentiment
        })
    
    random.shuffle(reviews)
    
    for i, review in enumerate(reviews):
        review['id'] = f'REV_{i+1:04d}'
    
    df = pd.DataFrame(reviews)
    return df


def valida_dataset(df):
    print("Validazione dataset...")
    
    print(f"\nNumero totale recensioni: {len(df)}")
    print(f"Colonne: {list(df.columns)}")
    print(f"Valori null: {df.isnull().sum().sum()}")
    
    print("\nDistribuzione per Reparto:")
    dept_dist = df['department'].value_counts()
    for dept, count in dept_dist.items():
        pct = (count / len(df)) * 100
        print(f"{dept}: {count} ({pct:.1f}%)")
    
    print("\nDistribuzione per Sentiment:")
    sent_dist = df['sentiment'].value_counts()
    for sent, count in sent_dist.items():
        pct = (count / len(df)) * 100
        print(f"{sent}: {count} ({pct:.1f}%)")
    
    print("\nDistribuzione Sentiment per Reparto:")
    for dept in df['department'].unique():
        dept_df = df[df['department'] == dept]
        pos_count = len(dept_df[dept_df['sentiment'] == 'positive'])
        neg_count = len(dept_df[dept_df['sentiment'] == 'negative'])
        print(f"{dept}: {pos_count} pos, {neg_count} neg")
    
    print("\nEsempi di Recensioni:")
    for dept in df['department'].unique():
        print(f"\n{dept}:")
        sample = df[df['department'] == dept].iloc[0]
        print(f"  Titolo: {sample['title']}")
        print(f"  Corpo: {sample['body']}")
        print(f"  Sentiment: {sample['sentiment']}")
    
    print("")


def main():
    df = genera_dataset(NUM_REVIEWS)
    valida_dataset(df)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
    print(f"\nDataset salvato in: {OUTPUT_PATH}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nPrime 5 righe del dataset:")
    print(df.head().to_string())


if __name__ == "__main__":
    main()
