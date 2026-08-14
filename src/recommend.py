import os
import pandas as pd
import numpy as np

class DrugRecommender:
    def __init__(self, data_path=None):
        if data_path is None:
            data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "drug_reviews.csv"))
        
        self.data_path = data_path
        self.df = None
        self.load_data()
        
    def load_data(self):
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
            # Clean column names and types
            self.df['rating'] = pd.to_numeric(self.df['rating'], errors='coerce')
            self.df['usefulCount'] = pd.to_numeric(self.df['usefulCount'], errors='coerce')
        else:
            print(f"Warning: Dataset not found at {self.data_path}. Recommender initialized empty.")
            self.df = pd.DataFrame(columns=["drugName", "condition", "rating", "usefulCount"])

    def get_recommendations(self, condition, exclude_drugs=None, top_n=5):
        """
        Recommends top N drugs for a given condition based on custom score.
        Score = (Avg Rating * 0.6) + (min(Avg Useful / 50, 1.0) * 2.0) + (min(Num Reviews / 20, 1.0) * 2.0)
        """
        if self.df is None or self.df.empty:
            return []
            
        # Case-insensitive match for condition
        cond_mask = self.df["condition"].str.lower() == condition.lower()
        cond_df = self.df[cond_mask]
        
        if cond_df.empty:
            # Try fuzzy matching if exact fails
            cond_mask = self.df["condition"].str.contains(condition, case=False, na=False)
            cond_df = self.df[cond_mask]
            
        if cond_df.empty:
            return []
            
        if exclude_drugs:
            exclude_drugs_lower = [d.lower() for d in exclude_drugs]
            cond_df = cond_df[~cond_df["drugName"].str.lower().isin(exclude_drugs_lower)]
            
        # Group by drugName and aggregate
        grouped = cond_df.groupby("drugName").agg(
            avg_rating=("rating", "mean"),
            avg_useful=("usefulCount", "mean"),
            total_useful=("usefulCount", "sum"),
            num_reviews=("rating", "count")
        ).reset_index()
        
        # Calculate recommendation score
        # Rating is out of 10. Avg useful normalized up to 50. Review count normalized up to 20.
        # Max theoretical score: 6.0 + 2.0 + 2.0 = 10.0
        grouped["score"] = (
            (grouped["avg_rating"] * 0.6) +
            (grouped["avg_useful"].clip(upper=50) / 50.0 * 2.0) +
            (grouped["num_reviews"].clip(upper=20) / 20.0 * 2.0)
        )
        
        # Sort by score descending and round values
        grouped = grouped.sort_values(by="score", ascending=False).head(top_n)
        grouped["avg_rating"] = grouped["avg_rating"].round(1)
        grouped["score"] = grouped["score"].round(2)
        
        return grouped.to_dict(orient="records")

    def get_alternatives(self, condition, current_drug, top_n=3):
        """
        Suggests alternative drugs for the same condition, excluding the currently used drug.
        """
        exclude = [current_drug] if current_drug else []
        return self.get_recommendations(condition, exclude_drugs=exclude, top_n=top_n)

if __name__ == "__main__":
    recommender = DrugRecommender()
    print("Top drugs for Depression:")
    for i, r in enumerate(recommender.get_recommendations("Depression"), 1):
        print(f"{i}. {r['drugName']} - Rating: {r['avg_rating']}, Useful: {r['total_useful']}, Score: {r['score']}")
        
    print("\nAlternative drugs for Depression excluding 'Lexapro':")
    for i, r in enumerate(recommender.get_alternatives("Depression", "Lexapro"), 1):
        print(f"{i}. {r['drugName']} - Rating: {r['avg_rating']}, Useful: {r['total_useful']}, Score: {r['score']}")
