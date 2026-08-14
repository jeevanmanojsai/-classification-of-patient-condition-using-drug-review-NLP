import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error

from preprocess import TextPreprocessor

def main():
    print("=" * 60)
    print("EVALUATING TRAINED MACHINE LEARNING MODELS")
    print("=" * 60)

    # File Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(base_dir, "data", "drug_reviews.csv")
    models_dir = os.path.join(base_dir, "models")
    
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    sentiment_model_path = os.path.join(models_dir, "sentiment_model.pkl")
    rating_model_path = os.path.join(models_dir, "rating_model.pkl")
    metrics_path = os.path.join(models_dir, "evaluation_metrics.json")

    # Check that everything exists
    for path in [data_path, vectorizer_path, best_model_path, sentiment_model_path, rating_model_path, metrics_path]:
        if not os.path.exists(path):
            print(f"Error: Required file not found at {path}. Make sure to run train.py first.")
            return

    # Load data
    print("Loading test data...")
    df = pd.read_csv(data_path)
    
    # Preprocess
    preprocessor = TextPreprocessor()
    df["cleaned_review"] = df["review"].apply(preprocessor.clean_text)
    df = df[df["cleaned_review"].str.strip() != ""]
    df = df.dropna(subset=["cleaned_review", "condition"])

    # Load Vectorizer and Models
    print("Loading vectorizer and trained models...")
    vectorizer = joblib.load(vectorizer_path)
    condition_meta = joblib.load(best_model_path)
    sentiment_meta = joblib.load(sentiment_model_path)
    rating_model = joblib.load(rating_model_path)

    # Extract components
    cond_model = condition_meta["model"]
    cond_model_name = condition_meta.get("model_name", "Condition Classifier")
    sent_model = sentiment_meta["model"]

    # Transform text to TF-IDF features
    X = vectorizer.transform(df["cleaned_review"])

    # 1. Show Model Comparison Summary
    print("\n--- Model Performance Comparison Table ---")
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)
        
    comp_df = pd.DataFrame(metrics_data["model_comparison"])
    print(comp_df.to_string(index=False))
    print(f"\nAuto-selected Best Model: {metrics_data['best_model_name']}")

    # 2. Condition Classification Evaluation
    print(f"\n2. Evaluation for Condition Classification (Model: {cond_model_name})")
    y_cond = df["condition"]
    
    _, X_test_c, _, y_test_c = train_test_split(
        X, y_cond, test_size=0.2, random_state=42, stratify=y_cond
    )
    
    c_preds = cond_model.predict(X_test_c)
    
    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_test_c, c_preds))
    
    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_test_c, c_preds)
    classes = cond_model.classes_
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print(cm_df.to_string())

    # 3. Sentiment Classification Evaluation
    print("\n" + "-" * 50)
    print("3. Evaluation for Sentiment Analysis")
    
    def map_rating_to_sentiment(r):
        if r <= 4:
            return "Negative"
        elif r <= 6:
            return "Neutral"
        else:
            return "Positive"
            
    df["sentiment"] = df["rating"].apply(map_rating_to_sentiment)
    y_sent = df["sentiment"]
    
    _, X_test_s, _, y_test_s = train_test_split(
        X, y_sent, test_size=0.2, random_state=42, stratify=y_sent
    )
    
    s_preds = sent_model.predict(X_test_s)
    
    print("\n--- Sentiment Classification Report ---")
    print(classification_report(y_test_s, s_preds))

    # 4. Rating Regressor Evaluation
    print("\n" + "-" * 50)
    print("4. Evaluation for Rating Regressor")
    y_rating = df["rating"]
    
    _, X_test_r, _, y_test_r = train_test_split(
        X, y_rating, test_size=0.2, random_state=42
    )
    
    r_preds = rating_model.predict(X_test_r)
    r_preds_clipped = np.clip(r_preds, 1.0, 10.0)
    
    mae = mean_absolute_error(y_test_r, r_preds_clipped)
    mse = mean_squared_error(y_test_r, r_preds_clipped)
    rmse = np.sqrt(mse)
    
    print(f"\nMean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    print("\n" + "=" * 60)
    print("EVALUATION PIPELINE COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
