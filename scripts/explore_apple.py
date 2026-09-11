import pandas as pd
import os

apple_path = os.path.join("data", "processed", "applesupport_pairs.csv")
amazon_path = os.path.join("data", "processed", "amazonhelp_pairs.csv")

df_apple = pd.read_csv(apple_path)
df_amazon = pd.read_csv(amazon_path)

print(f"=== Apple Support dataset size: {len(df_apple)} pairs ===")
print("\n--- Sample Apple Support Customer Queries ---")
for i, row in df_apple.sample(8, random_state=42).iterrows():
    print(f"Customer: {row['customer_query']}")
    print(f"AppleReply: {row['brand_response_text']}")
    print("-" * 60)
