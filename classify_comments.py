import pandas as pd
import json
import pickle
import argparse
import sys
import numpy as np
from pathlib import Path
from scipy.sparse import hstack
from preprocessing import TextPreprocessor


class CommentCategorizer:
    
    def __init__(self, model_path='model.pkl', vectorizer_path='vectorizer.pkl', 
                 label_encoder_path='label_encoder.json', scaler_path='scaler.pkl'):
        self.preprocessor = TextPreprocessor()
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            except FileNotFoundError:
                self.scaler = None
            
            with open(label_encoder_path, 'r') as f:
                self.label_encoder = json.load(f)
                self.reverse_label_encoder = {int(k): v for k, v in self.label_encoder.items()}
            
            print(f"Model loaded successfully from {model_path}")
        except FileNotFoundError as e:
            print(f"Error: Model files not found. Please train the model first using train_model.py")
            print(f"Missing file: {e}")
            sys.exit(1)
    
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
    
    def classify(self, text):
        processed_text = self.preprocessor.preprocess(text)
        text_vec = self.vectorizer.transform([processed_text])
        
        features = self.extract_features([text])
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
            combined = hstack([text_vec, features_scaled])
        else:
            combined = text_vec
        
        prediction = self.model.predict(combined)[0]
        probabilities = self.model.predict_proba(combined)[0]
        
        category = self.reverse_label_encoder[prediction]
        confidence = probabilities[prediction]
        
        return category, confidence
    
    def classify_batch(self, texts):
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        texts_vec = self.vectorizer.transform(processed_texts)
        
        features = self.extract_features(texts)
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
            combined = hstack([texts_vec, features_scaled])
        else:
            combined = texts_vec
        
        predictions = self.model.predict(combined)
        probabilities = self.model.predict_proba(combined)
        
        results = []
        for i, pred in enumerate(predictions):
            category = self.reverse_label_encoder[pred]
            confidence = probabilities[i][pred]
            results.append((category, confidence))
        
        return results
    
    def classify_from_csv(self, csv_path, output_path=None, comment_column='comment'):
        df = pd.read_csv(csv_path)
        
        if comment_column not in df.columns:
            raise ValueError(f"Column '{comment_column}' not found in CSV. Available columns: {df.columns.tolist()}")
        
        print(f"Classifying {len(df)} comments...")
        results = self.classify_batch(df[comment_column].tolist())
        
        df['predicted_category'] = [r[0] for r in results]
        df['confidence'] = [r[1] for r in results]
        
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        else:
            return df
    
    def classify_from_json(self, json_path, output_path=None, comment_key='comment'):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            comments = [item[comment_key] for item in data]
        else:
            comments = [data[comment_key]]
        
        print(f"Classifying {len(comments)} comments...")
        results = self.classify_batch(comments)
        
        if isinstance(data, list):
            for i, (category, confidence) in enumerate(results):
                data[i]['predicted_category'] = category
                data[i]['confidence'] = confidence
        else:
            data['predicted_category'] = results[0][0]
            data['confidence'] = results[0][1]
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_path}")
        else:
            return data
    
    def get_category_summary(self, df):
        summary = df.groupby('predicted_category').agg({
            'predicted_category': 'count',
            'confidence': ['mean', 'min', 'max']
        }).round(4)
        
        summary.columns = ['count', 'avg_confidence', 'min_confidence', 'max_confidence']
        summary = summary.sort_values('count', ascending=False)
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Classify comments into categories')
    parser.add_argument('--input', '-i', type=str, help='Input file path (CSV or JSON)')
    parser.add_argument('--output', '-o', type=str, help='Output file path')
    parser.add_argument('--text', '-t', type=str, help='Single comment text to classify')
    parser.add_argument('--model', '-m', type=str, default='model.pkl', help='Path to model file')
    parser.add_argument('--vectorizer', '-v', type=str, default='vectorizer.pkl', help='Path to vectorizer file')
    parser.add_argument('--labels', '-l', type=str, default='label_encoder.json', help='Path to label encoder file')
    
    args = parser.parse_args()
    
    categorizer = CommentCategorizer(args.model, args.vectorizer, args.labels)
    
    if args.text:
        category, confidence = categorizer.classify(args.text)
        print(f"\nComment: {args.text}")
        print(f"Category: {category}")
        print(f"Confidence: {confidence:.4f}")
        return
    
    if args.input:
        input_path = Path(args.input)
        
        if not input_path.exists():
            print(f"Error: Input file '{args.input}' not found.")
            return
        
        if input_path.suffix == '.csv':
            df = categorizer.classify_from_csv(args.input, args.output)
            if not args.output:
                print("\nClassification Results:")
                print(df[['comment', 'predicted_category', 'confidence']].to_string())
                print("\nCategory Summary:")
                print(categorizer.get_category_summary(df))
        
        elif input_path.suffix == '.json':
            data = categorizer.classify_from_json(args.input, args.output)
            if not args.output:
                print("\nClassification Results:")
                print(json.dumps(data, indent=2, ensure_ascii=False))
        
        else:
            print(f"Error: Unsupported file format '{input_path.suffix}'. Use CSV or JSON.")
    
    else:
        print("Comment Categorization Tool")
        print("Enter comments to classify (type 'quit' to exit):\n")
        
        while True:
            comment = input("Comment: ").strip()
            if comment.lower() in ['quit', 'exit', 'q']:
                break
            
            if comment:
                category, confidence = categorizer.classify(comment)
                print(f"Category: {category} (Confidence: {confidence:.4f})\n")


if __name__ == "__main__":
    main()
