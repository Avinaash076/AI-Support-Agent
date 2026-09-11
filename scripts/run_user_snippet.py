import os
import pandas as pd

RAW = os.path.join("data", "raw", "twcs.csv")

print(f"Reading dataset from {RAW}...")
df = pd.read_csv(RAW, dtype={"tweet_id": str, "in_response_to_tweet_id": str,
                             "response_tweet_id": str, "author_id": str})

print("\n--- DataFrame dtypes ---")
print(df.dtypes)   # verify: inbound should be bool (pandas auto-parses True/False here)

print("\nParsing created_at datetime...")
df["created_at"] = pd.to_datetime(
    df["created_at"], format="%a %b %d %H:%M:%S %z %Y", utc=True
)  # Twitter's format — explicit parse, don't trust inference

print("\n--- Top 20 Outbound Brand Handles ---")
print(df.loc[df["inbound"] == False, "author_id"].value_counts().head(20))
