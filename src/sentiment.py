import os
import joblib
import numpy as np
from preprocess import preprocess_review

class SentimentAnalyzer:
    def __init__(self, vectorizer_path=None, model_path=None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        if vectorizer_path is None:
            vectorizer_path = os.path.join(base_dir, "models", "tfidf_vectorizer.pkl")
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "sentiment_model.pkl")
            
        self.vectorizer = None
        self.model = None
        self.classes = []
        
        # Load models
        if os.path.exists(vectorizer_path) and os.path.exists(model_path):
            self.vectorizer = joblib.load(vectorizer_path)
            meta = joblib.load(model_path)
            self.model = meta["model"]
            self.classes = meta["classes"]
        else:
            print("Warning: Sentiment model or vectorizer not found. Call train.py first.")

    def analyze(self, text):
        """
        Analyzes the sentiment of the input review.
        Returns a dictionary with:
        - sentiment: 'Positive', 'Neutral', or 'Negative'
        - confidence: probability score of the prediction (0.0 to 1.0)
        - probabilities: dictionary of probabilities for each class
        """
        if self.model is None or self.vectorizer is None:
            return {"sentiment": "Unknown", "confidence": 0.0, "probabilities": {}}
            
        # Preprocess text
        cleaned_text = preprocess_review(text)
        if not cleaned_text:
            return {
                "sentiment": "Neutral",
                "confidence": 1.0,
                "probabilities": {"Positive": 0.0, "Neutral": 1.0, "Negative": 0.0}
            }
            
        # Vectorize
        X = self.vectorizer.transform([cleaned_text])
        
        # Predict class and probabilities
        pred_label = self.model.predict(X)[0]
        probs = self.model.predict_proba(X)[0]
        
        # Map class to probability
        prob_dict = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        confidence = float(np.max(probs))
        
        return {
            "sentiment": pred_label,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in prob_dict.items()}
        }

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    reviews = [
        "This medicine worked wonders! I feel amazing and my symptoms are gone.",
        "It was okay, did the job but nothing special.",
        "Absolutely horrible. The side effects made me extremely sick and it did not help at all."
    ]
    for r in reviews:
        print(f"\nReview: '{r}'")
        print("Analysis:", analyzer.analyze(r))
