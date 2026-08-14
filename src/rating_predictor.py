import os
import joblib
import numpy as np
from preprocess import preprocess_review

class RatingPredictor:
    def __init__(self, vectorizer_path=None, model_path=None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        if vectorizer_path is None:
            vectorizer_path = os.path.join(base_dir, "models", "tfidf_vectorizer.pkl")
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "rating_model.pkl")
            
        self.vectorizer = None
        self.model = None
        
        # Load models
        if os.path.exists(vectorizer_path) and os.path.exists(model_path):
            self.vectorizer = joblib.load(vectorizer_path)
            self.model = joblib.load(model_path)
        else:
            print("Warning: Rating model or vectorizer not found. Call train.py first.")

    def predict(self, text):
        """
        Predicts rating (on a scale from 1.0 to 10.0) from the input review text.
        """
        if self.model is None or self.vectorizer is None:
            return 5.0  # Default neutral score if model is not loaded
            
        # Preprocess text
        cleaned_text = preprocess_review(text)
        if not cleaned_text:
            return 5.0
            
        # Vectorize
        X = self.vectorizer.transform([cleaned_text])
        
        # Predict continuous rating
        pred_rating = self.model.predict(X)[0]
        
        # Clip to valid rating range (1.0 to 10.0)
        pred_rating = np.clip(pred_rating, 1.0, 10.0)
        
        return round(float(pred_rating), 1)

if __name__ == "__main__":
    predictor = RatingPredictor()
    reviews = [
        "This medicine worked wonders! I feel amazing and my symptoms are gone.",
        "It was okay, did the job but nothing special.",
        "Absolutely horrible. The side effects made me extremely sick and it did not help at all."
    ]
    for r in reviews:
        print(f"\nReview: '{r}'")
        print("Predicted Rating:", predictor.predict(r))
