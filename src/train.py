import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from preprocess import TextPreprocessor

def main():
    print("=" * 60)
    print("STARTING IMPROVED MACHINE LEARNING MODEL TRAINING PIPELINE")
    print("=" * 60)

    # File Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(base_dir, "data", "drug_reviews.csv")
    models_dir = os.path.join(base_dir, "models")
    
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Error: Preprocessed data not found at {data_path}. Please run prepare_data.py first.")
        return

    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records.")

    # 2. Text Preprocessing
    print("Preprocessing review texts...")
    preprocessor = TextPreprocessor()
    df["cleaned_review"] = df["review"].apply(preprocessor.clean_text)
    
    # Drop rows where cleaned review became empty or null
    df = df[df["cleaned_review"].str.strip() != ""]
    df = df.dropna(subset=["cleaned_review", "condition"])
    print(f"Cleaned dataset size: {len(df)} records.")

    # 3. TF-IDF Feature Engineering (optimized with bi-grams, min_df)
    print("Fitting optimized TF-IDF Vectorizer with bi-grams...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # unigrams and bigrams
        min_df=3,            # ignore terms appearing in fewer than 3 reviews
        max_df=0.90          # ignore terms appearing in more than 90% of reviews
    )
    
    X = vectorizer.fit_transform(df["cleaned_review"])
    
    # Save TF-IDF Vectorizer
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Saved TF-IDF Vectorizer to {vectorizer_path}")

    # =========================================================================
    # Task A: Condition Classification (Logistic Regression vs NB vs SVM vs RF)
    # =========================================================================
    print("\n--- Training & Comparing Condition Classification Models ---")
    y_cond = df["condition"]
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_cond, test_size=0.2, random_state=42, stratify=y_cond
    )
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=25, random_state=42, n_jobs=-1)
    }
    
    best_f1 = 0.0
    best_model_name = ""
    best_model = None
    model_comparison = []
    
    lr_model = None  # to keep track for surrogate explanations

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_c, y_train_c)
        preds = model.predict(X_test_c)
        
        acc = accuracy_score(y_test_c, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_c, preds, average='weighted')
        
        # Save Logistic Regression model for explainer surrogate
        if name == "Logistic Regression":
            lr_model = model
            
        print(f"-> Accuracy: {acc:.4f} | Weighted F1 Score: {f1:.4f}")
        
        model_comparison.append({
            "model": name,
            "accuracy": round(float(acc), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4)
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    print(f"\nBest Condition Model Selected: {best_model_name} (F1 Score: {best_f1:.4f})")
    
    # Save Best Model metadata along with surrogate explainer coefficients
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    model_meta = {
        "model": best_model,
        "model_name": best_model_name,
        "classes": list(best_model.classes_)
    }
    
    # Save LR coefficients as explanation surrogate if best model is RF or SVM
    if lr_model is not None:
        model_meta["explainer_coef"] = lr_model.coef_
        model_meta["explainer_intercept"] = lr_model.intercept_
        model_meta["explainer_classes"] = list(lr_model.classes_)
        
    joblib.dump(model_meta, best_model_path)
    print(f"Saved best model metadata to {best_model_path}")

    # Generate Evaluation Reports for the Best Model
    best_preds = best_model.predict(X_test_c)
    clf_report = classification_report(y_test_c, best_preds, output_dict=True)
    conf_mat = confusion_matrix(y_test_c, best_preds)
    classes_list = list(best_model.classes_)

    # Save metrics JSON for the dashboard
    metrics_path = os.path.join(models_dir, "evaluation_metrics.json")
    metrics_data = {
        "best_model_name": best_model_name,
        "model_comparison": model_comparison,
        "classification_report": clf_report,
        "confusion_matrix": conf_mat.tolist(),
        "classes": classes_list
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Saved evaluation dashboard metrics to {metrics_path}")

    # =========================================================================
    # Task B: Sentiment Analysis Model
    # =========================================================================
    print("\n--- Training Sentiment Classification Model ---")
    def map_rating_to_sentiment(r):
        if r <= 4:
            return "Negative"
        elif r <= 6:
            return "Neutral"
        else:
            return "Positive"
            
    df["sentiment"] = df["rating"].apply(map_rating_to_sentiment)
    y_sent = df["sentiment"]
    
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y_sent, test_size=0.2, random_state=42, stratify=y_sent
    )
    
    sentiment_classifier = LogisticRegression(max_iter=1000, random_state=42)
    sentiment_classifier.fit(X_train_s, y_train_s)
    
    s_preds = sentiment_classifier.predict(X_test_s)
    s_acc = accuracy_score(y_test_s, s_preds)
    s_precision, s_recall, s_f1, _ = precision_recall_fscore_support(y_test_s, s_preds, average='weighted')
    print(f"Sentiment Model -> Accuracy: {s_acc:.4f} | F1 Score: {s_f1:.4f}")
    
    sentiment_model_path = os.path.join(models_dir, "sentiment_model.pkl")
    joblib.dump({
        "model": sentiment_classifier,
        "classes": list(sentiment_classifier.classes_)
    }, sentiment_model_path)
    print(f"Saved sentiment model to {sentiment_model_path}")

    # =========================================================================
    # Task C: Rating Prediction Model
    # =========================================================================
    print("\n--- Training Rating Regressor Model ---")
    y_rating = df["rating"]
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_rating, test_size=0.2, random_state=42
    )
    
    rating_regressor = Ridge(alpha=1.0)
    rating_regressor.fit(X_train_r, y_train_r)
    
    r_preds = rating_regressor.predict(X_test_r)
    mae = np.mean(np.abs(y_test_r - r_preds))
    rmse = np.sqrt(np.mean((y_test_r - r_preds) ** 2))
    print(f"Rating Regressor -> MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    
    rating_model_path = os.path.join(models_dir, "rating_model.pkl")
    joblib.dump(rating_regressor, rating_model_path)
    print(f"Saved rating model to {rating_model_path}")

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
