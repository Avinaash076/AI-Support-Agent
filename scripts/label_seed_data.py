import os
import pandas as pd
from src.intent_classifier import KeywordBaselineClassifier

def create_labeled_dataset(sample_size: int = 2500):
    input_path = os.path.join("data", "processed", "applesupport_pairs.csv")
    output_path = os.path.join("data", "processed", "applesupport_labeled.csv")
    
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
        
    kw_clf = KeywordBaselineClassifier()
    print("Labeling dataset using baseline heuristics...")
    df["intent"] = df["customer_query"].apply(kw_clf.predict)
    
    print("\n--- Intent Distribution ---")
    print(df["intent"].value_counts())
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved labeled dataset ({len(df)} rows) to {output_path}")

if __name__ == "__main__":
    create_labeled_dataset()
