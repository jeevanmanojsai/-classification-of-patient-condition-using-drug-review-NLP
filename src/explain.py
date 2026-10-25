import os
import re
import joblib
import numpy as np
from preprocess import TextPreprocessor

class PredictExplainer:
    def __init__(self, vectorizer_path=None, model_path=None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        if vectorizer_path is None:
            vectorizer_path = os.path.join(base_dir, "models", "tfidf_vectorizer.pkl")
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "best_model.pkl")
            
        self.vectorizer = None
        self.model_meta = None
        self.feature_names = []
        self.preprocessor = TextPreprocessor()
        
        if os.path.exists(vectorizer_path) and os.path.exists(model_path):
            self.vectorizer = joblib.load(vectorizer_path)
            self.feature_names = self.vectorizer.get_feature_names_out()
            self.model_meta = joblib.load(model_path)
        else:
            print("Warning: Explainer models or vectorizer not found. Train them first.")

    def explain(self, text, predicted_condition, top_n=8):
        """
        Explains why a review was classified into a specific condition.
        Returns:
        - top_positive: list of dicts with {'word': word, 'weight': weight} (pushing towards class)
        - top_negative: list of dicts with {'word': word, 'weight': weight} (pulling away from class)
        - highlighted_html: original text with positive keywords highlighted
        """
        if self.model_meta is None or self.vectorizer is None or not self.feature_names.any():
            return [], [], text
            
        # 1. Preprocess and vectorise
        cleaned_text = self.preprocessor.clean_text(text)
        if not cleaned_text:
            return [], [], text
            
        X = self.vectorizer.transform([cleaned_text])
        feature_indices = X.nonzero()[1]
        
        if len(feature_indices) == 0:
            return [], [], text

        tfidf_values = X.toarray()[0]
        words_in_text = [self.feature_names[idx] for idx in feature_indices]
        word_tfidf_map = {self.feature_names[idx]: tfidf_values[idx] for idx in feature_indices}
        
        classes = self.model_meta["classes"]
        try:
            class_idx = classes.index(predicted_condition)
        except ValueError:
            # Fallback if class not found
            top_positive = [{"word": w, "weight": float(word_tfidf_map[w])} for w in words_in_text]
            top_positive = sorted(top_positive, key=lambda x: x["weight"], reverse=True)[:top_n]
            return top_positive, [], text

        # 2. Extract feature weights for explanation
        # We will use surrogate Logistic Regression coefficients saved in metadata (explainer_coef)
        # to ensure we always have high-quality, linear linear explanations even if the best model is Random Forest!
        has_surrogate = "explainer_coef" in self.model_meta
        word_weights = []

        if has_surrogate:
            coefs = self.model_meta["explainer_coef"]
            classes_surr = self.model_meta["explainer_classes"]
            class_idx_surr = classes_surr.index(predicted_condition)
            
            if coefs.ndim > 1:
                class_coefs = coefs[class_idx_surr]
            else:
                class_coefs = coefs if class_idx_surr == 1 else -coefs
                
            for word in words_in_text:
                word_idx = np.where(self.feature_names == word)[0][0]
                coef_weight = class_coefs[word_idx]
                tfidf_val = word_tfidf_map[word]
                # combination score
                importance = coef_weight * tfidf_val
                word_weights.append({"word": word, "weight": float(importance)})
        else:
            # Fallback to direct model if it has coef_ or feature_log_prob_
            model = self.model_meta["model"]
            if hasattr(model, "coef_"):
                coefs = model.coef_
                if coefs.ndim > 1:
                    class_coefs = coefs[class_idx]
                else:
                    class_coefs = coefs if class_idx == 1 else -coefs
                    
                for word in words_in_text:
                    word_idx = np.where(self.feature_names == word)[0][0]
                    coef_weight = class_coefs[word_idx]
                    tfidf_val = word_tfidf_map[word]
                    word_weights.append({"word": word, "weight": float(coef_weight * tfidf_val)})
            elif hasattr(model, "feature_log_prob_"):
                log_probs = model.feature_log_prob_[class_idx]
                avg_log_probs = np.mean(model.feature_log_prob_, axis=0)
                for word in words_in_text:
                    word_idx = np.where(self.feature_names == word)[0][0]
                    importance = (log_probs[word_idx] - avg_log_probs[word_idx]) * word_tfidf_map[word]
                    word_weights.append({"word": word, "weight": float(importance)})
            else:
                # Fallback to TF-IDF weights
                for word in words_in_text:
                    word_weights.append({"word": word, "weight": float(word_tfidf_map[word])})

        # Separate positive and negative weights
        pos_weights = [w for w in word_weights if w["weight"] > 0.001]
        neg_weights = [w for w in word_weights if w["weight"] < -0.001]

        # Sort
        top_positive = sorted(pos_weights, key=lambda x: x["weight"], reverse=True)[:top_n]
        top_negative = sorted(neg_weights, key=lambda x: x["weight"], reverse=False)[:top_n] # Sorted from most negative to least negative
        
        # Absolute weights for negative keywords display
        for nw in top_negative:
            nw["abs_weight"] = abs(nw["weight"])

        # Highlight positive keywords in text
        highlighted_html = text
        if top_positive:
            max_weight = max([w["weight"] for w in top_positive])
            sorted_hl_words = sorted([w["word"] for w in top_positive], key=len, reverse=True)
            
            for word in sorted_hl_words:
                weight = next(w["weight"] for w in top_positive if w["word"] == word)
                opacity = 0.2 + (weight / (max_weight + 1e-9)) * 0.6
                
                # Case-insensitive whole word regex matching
                pattern = re.compile(rf"\b({word}\w*)\b", re.IGNORECASE)
                replacement = rf'<span class="explain-highlight" style="background-color: rgba(13, 110, 253, {opacity:.2f}); font-weight: 500; padding: 2px 6px; border-radius: 4px; border-bottom: 2px solid rgba(13, 110, 253, 0.8);" title="AI Weight: {weight:.3f}">\1</span>'
                highlighted_html = pattern.sub(replacement, highlighted_html)
                
        return top_positive, top_negative, highlighted_html

if __name__ == "__main__":
    explainer = PredictExplainer()
    test_review = "I was prescribed medicine for high blood pressure. It helped stabilize my readings."
    print(explainer.explain(test_review, "Obesity"))
