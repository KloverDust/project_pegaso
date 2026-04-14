
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

class DepartmentClassifier:
    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        
        if model_type == 'logistic':
            self.model = LogisticRegression(
                C=1.0,  # Regolarizzazione
                max_iter=1000,
                random_state=42,
                solver='lbfgs'
            )
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=1.0)
        else:
            raise ValueError(f"Tipo modello non supportato: {model_type}")
        
        self.classes_ = None
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        print(f"Department classifier training completed. Classes: {list(self.classes_)}")
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
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



