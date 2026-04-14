
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os


class SentimentAnalyzer:
    def __init__(self):
        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        )
        self.classes_ = None
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        print(f"Sentiment analyzer training completed. Classes: {list(self.classes_)}")
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='positive')
        recall = recall_score(y_test, y_pred, pos_label='positive')
        f1 = f1_score(y_test, y_pred, pos_label='positive')
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': y_pred
        }
        
        return metrics
    
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Modello salvato: {filepath}")
    
    def load(self, filepath):
        self.model = joblib.load(filepath)
        self.classes_ = self.model.classes_
        print(f"Modello caricato: {filepath}")



