import pandas as pd
from pathlib import Path

df = pd.read_csv(Path(__file__).resolve().parents[1] / "data/processed/applesupport_pairs.csv")

print("=== 10 Real Customer Queries & Apple Support Responses ===\n")
for i, row in df.head(10).iterrows():
    q = str(row['customer_query']).replace('\n', ' ')
    r = str(row['brand_response_text']).replace('\n', ' ')
    print(f"[{i+1}] Customer: {q}")
    print(f"    AppleSupport: {r}\n")
