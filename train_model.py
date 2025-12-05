import pandas as pd
import numpy as np
import pickle
import json
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from preprocessing import TextPreprocessor
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    from transformers import pipeline
    import torch
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("Warning: Transformers library not available. Will only train traditional ML models.")


class CommentClassifier:
    
    def __init__(self, use_transformer=False):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = {}
        self.reverse_label_encoder = {}
        self.use_transformer = use_transformer and TRANSFORMER_AVAILABLE
    
    def extract_features(self, texts):
        features = []
        for text in texts:
            original_text = str(text)
            text_lower = original_text.lower()
            
            feat = {
                'length': len(original_text),
                'word_count': len(original_text.split()),
                'exclamation_count': original_text.count('!'),
                'question_count': original_text.count('?'),
                'uppercase_ratio': sum(1 for c in original_text if c.isupper()) / max(len(original_text), 1),
                'has_question': 1 if '?' in original_text else 0,
                'has_exclamation': 1 if '!' in original_text else 0,
                'has_negative_words': 1 if any(word in text_lower for word in ['not', 'no', 'never', 'bad', 'worst', 'hate', 'trash', 'terrible', 'awful']) else 0,
                'has_positive_words': 1 if any(word in text_lower for word in ['good', 'great', 'amazing', 'love', 'best', 'excellent', 'wonderful', 'fantastic']) else 0,
                'has_suggestion_words': 1 if any(word in text_lower for word in ['can', 'could', 'would', 'should', 'suggest', 'maybe', 'try', 'consider']) else 0,
                'has_threat_words': 1 if any(word in text_lower for word in ['report', 'legal', 'action', 'warning', 'violate', 'escalate', 'complain']) else 0,
            }
            features.append(list(feat.values()))
        return np.array(features)
        
    def load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        
        print("Preprocessing comments...")
        df['processed_comment'] = df['comment'].apply(self.preprocessor.preprocess)
        
        unique_labels = sorted(df['label'].unique())
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_encoder = {idx: label for label, idx in self.label_encoder.items()}
        
        X = df['processed_comment'].values
        y = df['label'].map(self.label_encoder).values
        
        return X, y, unique_labels
    
    def train_traditional_ml(self, X_train, y_train, X_test, y_test, model_type='logistic'):
        print(f"\nTraining {model_type.upper()} model...")
        
        print("Extracting additional features...")
        X_train_features = self.extract_features(X_train)
        X_test_features = self.extract_features(X_test)
        
        print("Vectorizing text with TF-IDF...")
        all_words = set()
        for text in X_train:
            all_words.update(str(text).split())
        vocab_size = min(2500, len(all_words))
        
        self.vectorizer = TfidfVectorizer(
            max_features=vocab_size,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            use_idf=True,
            analyzer='word',
            norm='l2'
        )
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        X_train_features_scaled = self.scaler.fit_transform(X_train_features)
        X_test_features_scaled = self.scaler.transform(X_test_features)
        
        X_train_combined = hstack([X_train_vec, X_train_features_scaled])
        X_test_combined = hstack([X_test_vec, X_test_features_scaled])
        
        if model_type == 'logistic':
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'solver': ['lbfgs', 'liblinear']
            }
            base_model = LogisticRegression(
                max_iter=3000,
                random_state=42,
                multi_class='multinomial',
                class_weight='balanced'
            )
            print("Performing grid search for best parameters...")
            grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
            grid_search.fit(X_train_combined, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            
        elif model_type == 'svm':
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'kernel': ['linear', 'rbf']
            }
            base_model = SVC(
                random_state=42,
                probability=True,
                class_weight='balanced'
            )
            print("Performing grid search for best parameters...")
            grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
            grid_search.fit(X_train_combined, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            
        elif model_type == 'ensemble':
            lr = LogisticRegression(max_iter=3000, random_state=42, C=10.0, solver='lbfgs', 
                                   multi_class='multinomial', class_weight='balanced')
            svm = SVC(kernel='linear', C=10.0, random_state=42, probability=True, class_weight='balanced')
            rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, 
                                       class_weight='balanced', n_jobs=-1)
            self.model = VotingClassifier(
                estimators=[('lr', lr), ('svm', svm), ('rf', rf)],
                voting='soft',
                weights=[2, 2, 1]
            )
            print("Training ensemble model...")
            self.model.fit(X_train_combined, y_train)
        else:
            raise ValueError("model_type must be 'logistic', 'svm', or 'ensemble'")
        
        print("Training model...")
        if model_type != 'ensemble' and model_type != 'logistic' and model_type != 'svm':
            self.model.fit(X_train_combined, y_train)
        
        y_pred = self.model.predict(X_test_combined)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{model_type.upper()} Model Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=[self.reverse_label_encoder[i] for i in range(len(self.reverse_label_encoder))]))
        
        return accuracy
    
    def train_transformer(self, X_train, y_train, X_test, y_test):
        if not TRANSFORMER_AVAILABLE:
            print("Transformers library not available. Skipping transformer training.")
            return None
        
        print("\nUsing pre-trained DistilBERT for zero-shot classification...")
        return None
    
    def save_model(self, model_path='model.pkl', vectorizer_path='vectorizer.pkl', 
                   label_encoder_path='label_encoder.json', scaler_path='scaler.pkl'):
        if self.model and self.vectorizer:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            with open(label_encoder_path, 'w') as f:
                json.dump(self.reverse_label_encoder, f, indent=2)
            
            print(f"\nModel saved to {model_path}")
            print(f"Vectorizer saved to {vectorizer_path}")
            print(f"Scaler saved to {scaler_path}")
            print(f"Label encoder saved to {label_encoder_path}")


def main():
    print("=" * 60)
    print("Comment Categorization Model Training")
    print("=" * 60)
    
    classifier = CommentClassifier()
    X, y, labels = classifier.load_data('project_data.csv')
    
    print(f"\nDataset loaded: {len(X)} comments")
    print(f"Categories: {labels}")
    print(f"Label distribution:\n{pd.Series(y).value_counts().sort_index()}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    classifier_lr = CommentClassifier()
    classifier_lr.label_encoder = classifier.label_encoder
    classifier_lr.reverse_label_encoder = classifier.reverse_label_encoder
    accuracy_lr = classifier_lr.train_traditional_ml(
        X_train, y_train, X_test, y_test, model_type='logistic'
    )
    classifier_lr.save_model('model_lr.pkl', 'vectorizer_lr.pkl', 'label_encoder.json', 'scaler_lr.pkl')
    
    classifier_svm = CommentClassifier()
    classifier_svm.label_encoder = classifier.label_encoder
    classifier_svm.reverse_label_encoder = classifier.reverse_label_encoder
    accuracy_svm = classifier_svm.train_traditional_ml(
        X_train, y_train, X_test, y_test, model_type='svm'
    )
    classifier_svm.save_model('model_svm.pkl', 'vectorizer_svm.pkl', 'label_encoder.json', 'scaler_svm.pkl')
    
    classifier_ensemble = CommentClassifier()
    classifier_ensemble.label_encoder = classifier.label_encoder
    classifier_ensemble.reverse_label_encoder = classifier.reverse_label_encoder
    accuracy_ensemble = classifier_ensemble.train_traditional_ml(
        X_train, y_train, X_test, y_test, model_type='ensemble'
    )
    classifier_ensemble.save_model('model_ensemble.pkl', 'vectorizer_ensemble.pkl', 'label_encoder.json', 'scaler_ensemble.pkl')
    
    print("\n" + "=" * 60)
    print("Model Comparison:")
    print(f"Logistic Regression Accuracy: {accuracy_lr:.4f}")
    print(f"SVM Accuracy: {accuracy_svm:.4f}")
    print(f"Ensemble Accuracy: {accuracy_ensemble:.4f}")
    
    best_accuracy = max(accuracy_lr, accuracy_svm, accuracy_ensemble)
    if accuracy_ensemble == best_accuracy:
        print("\nUsing Ensemble model as default model.")
        import shutil
        shutil.copy('model_ensemble.pkl', 'model.pkl')
        shutil.copy('vectorizer_ensemble.pkl', 'vectorizer.pkl')
        shutil.copy('scaler_ensemble.pkl', 'scaler.pkl')
    elif accuracy_lr == best_accuracy:
        print("\nUsing Logistic Regression as default model.")
        import shutil
        shutil.copy('model_lr.pkl', 'model.pkl')
        shutil.copy('vectorizer_lr.pkl', 'vectorizer.pkl')
        shutil.copy('scaler_lr.pkl', 'scaler.pkl')
    else:
        print("\nUsing SVM as default model.")
        import shutil
        shutil.copy('model_svm.pkl', 'model.pkl')
        shutil.copy('vectorizer_svm.pkl', 'vectorizer.pkl')
        shutil.copy('scaler_svm.pkl', 'scaler.pkl')
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
