import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


class TextPreprocessor:
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        stopwords_list = stopwords.words('english')
        very_common = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        self.stop_words = set([w for w in stopwords_list if w in very_common])
        self.stop_words.update(['u', 'ur', 'youre'])
    
    def clean_text(self, text):
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        
        punct_to_keep = "!?."
        text = ''.join([char if char.isalnum() or char.isspace() or char in punct_to_keep else ' ' for char in text])
        text = re.sub(r'\s+', ' ', text)
        
        text = text.lower().strip()
        
        return text.strip()
    
    def tokenize(self, text):
        if not text:
            return []
        return word_tokenize(text)
    
    def lemmatize_tokens(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stop_words and len(token) > 0]
    
    def preprocess(self, text):
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize_tokens(tokens)
        processed = ' '.join(tokens)
        if not processed.strip():
            return cleaned.lower()
        return processed
    
    def preprocess_batch(self, texts):
        return [self.preprocess(text) for text in texts]


if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "Amazing work! The animation looks stunning.",
        "This is trash, stop posting.",
        "Can you make one about space exploration?",
        "Follow me for instant followers! https://example.com"
    ]
    
    print("Testing Text Preprocessor:")
    print("-" * 50)
    for text in test_texts:
        processed = preprocessor.preprocess(text)
        print(f"Original: {text}")
        print(f"Processed: {processed}\n")
