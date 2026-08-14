import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK packages are downloaded
def download_nltk_resources():
    resources = {
        'corpora/stopwords': 'stopwords',
        'tokenizers/punkt': 'punkt',
        'corpora/wordnet': 'wordnet',
        'corpora/omw-1.4': 'omw-1.4'
    }
    for resource_path, download_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Downloading NLTK resource: {download_name}...")
            nltk.download(download_name, quiet=True)

download_nltk_resources()

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """
        Cleans the input text by:
        1. Converting to lowercase
        2. Removing HTML tags
        3. Removing punctuation and special characters
        4. Removing numbers
        5. Removing stopwords
        6. Tokenizing and Lemmatizing
        7. Normalizing whitespace
        """
        if not isinstance(text, str):
            return ""
            
        # 1. Lowercase conversion
        text = text.lower()
        
        # 2. Remove HTML tags (common in drug review datasets, e.g., &#039;, &amp;)
        text = re.sub(r'&#\d+;', '', text)  # remove HTML entities
        text = re.sub(r'<[^>]*>', '', text)   # remove HTML tags
        
        # 3 & 4. Remove special characters, punctuation, and numbers
        # Keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # 5. Tokenization and 6. Lemmatization / Stopword removal
        tokens = text.split()  # Fast tokenization by whitespace
        cleaned_tokens = []
        
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                # Lemmatize verbs, nouns, and adjectives
                lemma = self.lemmatizer.lemmatize(token, pos='v')
                lemma = self.lemmatizer.lemmatize(lemma, pos='n')
                cleaned_tokens.append(lemma)
                
        # 7. Whitespace normalization
        cleaned_text = " ".join(cleaned_tokens).strip()
        
        return cleaned_text

# Reusable preprocess function
def preprocess_review(text):
    preprocessor = TextPreprocessor()
    return preprocessor.clean_text(text)

if __name__ == "__main__":
    test_text = "The medicine cured my acne! But it caused headaches and dizziness... 10/10."
    print("Original Text:", test_text)
    print("Preprocessed Text:", preprocess_review(test_text))
