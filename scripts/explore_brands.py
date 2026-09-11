import os
import pandas as pd

RAW_PATH = os.path.join("data", "raw", "twcs.csv")

print(f"Loading dataset from: {RAW_PATH} ...")
df = pd.read_csv(
    RAW_PATH, 
    dtype={"tweet_id": str, "in_response_to_tweet_id": str, "response_tweet_id": str, "author_id": str}
)

print("\n--- Data Info & Dtypes ---")
print(df.info())

print("\nParsing created_at timestamps...")
df["created_at"] = pd.to_datetime(df["created_at"], format="%a %b %d %H:%M:%S %z %Y", utc=True)

print("\n--- Top 25 Brands (Outbound Customer Support Handles) ---")
outbound_counts = df.loc[df["inbound"] == False, "author_id"].value_counts().head(25)
print(outbound_counts.to_string())

print("\n--- Total Inbound vs Outbound Counts ---")
print(df["inbound"].value_counts())
