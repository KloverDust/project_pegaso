
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer


def pulisci_testo(testo):
    testo = testo.lower()
    testo = testo.translate(str.maketrans('', '', string.punctuation))
    testo = re.sub(r'\s+', ' ', testo)
    testo = testo.strip()
    return testo

def crea_vectorizer(ngram_range=(1, 2), max_features=500, min_df=2):
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        lowercase=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{2,}',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    )


def preprocessa_dataset(df):
    testi_combinati = df['title'] + ' ' + df['title'] + ' ' + df['body']
    testi_puliti = testi_combinati.apply(pulisci_testo)
    return testi_puliti



