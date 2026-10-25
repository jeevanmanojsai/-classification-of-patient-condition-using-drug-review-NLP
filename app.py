import os
import sys
import json
import joblib
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash

# Add src folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocess import TextPreprocessor, preprocess_review
from side_effects import SideEffectDetector
from recommend import DrugRecommender
from sentiment import SentimentAnalyzer
from rating_predictor import RatingPredictor
from explain import PredictExplainer

app = Flask(__name__)
app.secret_key = "nlp_healthcare_secret_key"

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "models")
vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
best_model_path = os.path.join(models_dir, "best_model.pkl")

# Initialize global components
recommender = DrugRecommender()
side_effect_detector = SideEffectDetector()

# Medical review keywords list for validation
MEDICAL_KEYWORDS = [
    "medicine", "drug", "treatment", "doctor", "patient", "prescribed", "symptom", "pain",
    "depression", "diabetes", "blood pressure", "headache", "anxiety", "acne", "infection",
    "review", "side effects", "side-effects", "bipolar", "insomnia", "obesity", "adhd", 
    "weight loss", "yeast", "pill", "capsule", "tablet", "dose", "dosage", "efficacy"
]

# Greetings to check and reject
GREETINGS = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings", "sup", "hola"]

def check_models_exist():
    required = [
        vectorizer_path, 
        best_model_path, 
        os.path.join(models_dir, "sentiment_model.pkl"),
        os.path.join(models_dir, "rating_model.pkl"),
        os.path.join(models_dir, "evaluation_metrics.json")
    ]
    return all(os.path.exists(p) for p in required)

@app.route("/", methods=["GET"])
def index():
    models_ready = check_models_exist()
    return render_template("index.html", models_ready=models_ready)

@app.route("/dashboard", methods=["GET"])
def dashboard():
    metrics_path = os.path.join(models_dir, "evaluation_metrics.json")
    if not os.path.exists(metrics_path):
        flash("Dashboard metrics are not available. Please train models first.", "danger")
        return redirect(url_for("index"))
        
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)
        
    return render_template(
        "dashboard.html",
        best_model_name=metrics_data["best_model_name"],
        model_comparison=metrics_data["model_comparison"],
        classification_report=metrics_data["classification_report"],
        confusion_matrix=metrics_data["confusion_matrix"],
        classes=metrics_data["classes"]
    )

@app.route("/predict", methods=["POST"])
def predict():
    if not check_models_exist():
        flash("Models are not trained yet! Please run python src/train.py first.", "danger")
        return redirect(url_for("index"))
        
    review_text = request.form.get("review_text", "").strip()
    
    # --- Input Validation (Issue 8) ---
    # 1. Reject empty text
    if not review_text:
        flash("Please enter a valid review to analyze.", "warning")
        return redirect(url_for("index"))
        
    review_text_lower = review_text.lower()
    words = [w.strip(".,!?\"'()[]") for w in review_text_lower.split() if w.strip()]
    
    # 2. Reject greeting or single word
    if len(words) == 1:
        if words[0] in GREETINGS:
            flash("Input rejected: greetings are not valid patient reviews. Please describe a clinical experience.", "warning")
            return redirect(url_for("index"))
        else:
            flash("Input rejected: single-word inputs are not valid treatment reviews.", "warning")
            return redirect(url_for("index"))
            
    # 3. Check for greetings under 15 words
    if len(words) < 15:
        # Check if they are just greetings
        if all(w in GREETINGS for w in words):
            flash("Input rejected: greetings are not valid patient reviews.", "warning")
            return redirect(url_for("index"))
        # General word count failure
        flash("Review is too short. Please provide a more detailed patient review of at least 15 words to ensure reliable analysis.", "warning")
        return redirect(url_for("index"))

    # --- Medical Review Detection (Issue 1) ---
    is_medical = any(keyword in review_text_lower for keyword in MEDICAL_KEYWORDS)
    if not is_medical:
        flash("No valid patient treatment review detected. Please enter a review describing a medical condition, treatment experience, medication effectiveness, or side effects.", "warning")
        return redirect(url_for("index"))

    try:
        # 1. Load condition classification model and vectorizer
        vectorizer = joblib.load(vectorizer_path)
        cond_meta = joblib.load(best_model_path)
        cond_model = cond_meta["model"]
        model_name = cond_meta.get("model_name", "Unknown Model")
        classes = cond_meta["classes"]

        # 2. Preprocess and vectorize
        cleaned_text = preprocess_review(review_text)
        if not cleaned_text:
            flash("Cleaned text contains no meaningful words. Please write a longer, descriptive review.", "warning")
            return redirect(url_for("index"))
            
        X = vectorizer.transform([cleaned_text])

        # 3. Predict condition
        pred_condition = cond_model.predict(X)[0]

        # 4. Calculate probabilities / confidence for all classes
        if hasattr(cond_model, "predict_proba"):
            probs = cond_model.predict_proba(X)[0]
        elif hasattr(cond_model, "decision_function"):
            scores = cond_model.decision_function(X)[0]
            if scores.ndim == 0:  # binary
                p = 1 / (1 + np.exp(-scores))
                probs = np.array([1 - p, p])
            else:
                exp_scores = np.exp(scores - np.max(scores))
                probs = exp_scores / np.sum(exp_scores)
        else:
            probs = np.ones(len(classes)) / len(classes)

        condition_probs = {classes[i]: float(probs[i]) for i in range(len(classes))}
        confidence = float(np.max(probs))

        # --- Confidence Threshold Check (Issue 2) ---
        is_low_confidence = confidence < 0.30

        # 5. Run sentiment analysis
        sent_analyzer = SentimentAnalyzer()
        sentiment_res = sent_analyzer.analyze(review_text)

        # 6. Run rating prediction
        rating_pred = RatingPredictor()
        predicted_rating = rating_pred.predict(review_text)

        # 7. Detect side effects
        detected_side_effects = side_effect_detector.detect(review_text)

        # 8. Recommender system (only retrieved if NOT low confidence)
        top_drugs = []
        alternative_drugs = []
        mentioned_drug = None
        
        if not is_low_confidence:
            top_drugs = recommender.get_recommendations(pred_condition, top_n=5)
            
            if recommender.df is not None:
                for d in recommender.df[recommender.df["condition"].str.lower() == pred_condition.lower()]["drugName"].unique():
                    if d.lower() in review_text.lower():
                        mentioned_drug = d
                        break
            
            exclude_drug = mentioned_drug if mentioned_drug else (top_drugs[0]["drugName"] if top_drugs else None)
            if detected_side_effects:
                alternative_drugs = recommender.get_alternatives(pred_condition, current_drug=exclude_drug, top_n=3)

        # 9. Explainable AI: highlight influential review terms
        explainer = PredictExplainer()
        _, _, highlighted_review = explainer.explain(review_text, pred_condition)

        # Return results to template
        return render_template(
            "result.html",
            original_review=review_text,
            highlighted_review=highlighted_review,
            predicted_condition=pred_condition,
            confidence=round(confidence * 100, 1),
            is_low_confidence=is_low_confidence,
            sentiment=sentiment_res["sentiment"],
            sentiment_confidence=round(sentiment_res["confidence"] * 100, 1),
            sentiment_probs=sentiment_res["probabilities"],
            predicted_rating=predicted_rating,
            detected_side_effects=detected_side_effects,
            top_drugs=top_drugs,
            alternative_drugs=alternative_drugs,
            condition_probs=condition_probs,
            model_used=model_name,
            mentioned_drug=mentioned_drug
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"An error occurred during prediction: {str(e)}", "danger")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
