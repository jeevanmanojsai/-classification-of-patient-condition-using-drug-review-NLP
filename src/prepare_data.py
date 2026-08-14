import os
import pandas as pd

def main():
    print("Preparing dataset...")
    # Paths
    raw_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dataset-nlp", "drugsComTrain_raw.csv"))
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    output_path = os.path.join(output_dir, "drug_reviews.csv")
    
    if not os.path.exists(raw_data_path):
        print(f"Error: Raw dataset not found at {raw_data_path}")
        return

    # Load dataset
    print(f"Loading raw data from {raw_data_path}...")
    df = pd.read_csv(raw_data_path)
    print(f"Original shape: {df.shape}")

    # Drop missing values in critical columns
    df = df.dropna(subset=["condition", "review", "rating"])
    print(f"Shape after dropping NaNs: {df.shape}")

    # Remove duplicates
    df = df.drop_duplicates(subset=["review"])
    print(f"Shape after dropping duplicate reviews: {df.shape}")

    # Filter out invalid conditions (e.g., HTML artifacts like '</span>')
    df = df[~df["condition"].str.contains("</span>|users found|helpful", case=False, na=False)]
    print(f"Shape after filtering invalid conditions: {df.shape}")

    # Find the top 10 most common conditions (excluding non-conditions if any, but standard top conditions are fine)
    # Let's list the top conditions in the cleaned set
    top_conditions = df["condition"].value_counts().head(12).index.tolist()
    print(f"Top 12 conditions selected: {top_conditions}")

    # Filter dataset for only these top conditions
    df_filtered = df[df["condition"].isin(top_conditions)]
    print(f"Shape after filtering for top conditions: {df_filtered.shape}")

    # Perform stratified sampling: sample up to 2500 reviews per condition
    sampled_dfs = []
    samples_per_condition = 2500
    
    for cond in top_conditions:
        cond_df = df_filtered[df_filtered["condition"] == cond]
        if len(cond_df) > samples_per_condition:
            sampled_dfs.append(cond_df.sample(n=samples_per_condition, random_state=42))
        else:
            sampled_dfs.append(cond_df)
            
    df_sampled = pd.concat(sampled_dfs, ignore_index=True)
    
    # Shuffle the final dataset
    df_sampled = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Final sampled dataset shape: {df_sampled.shape}")
    print("Class distribution in sampled dataset:")
    print(df_sampled["condition"].value_counts())

    # Create output directory and save
    os.makedirs(output_dir, exist_ok=True)
    df_sampled.to_csv(output_path, index=False)
    print(f"Sampled dataset successfully saved to {output_path}")

if __name__ == "__main__":
    main()
