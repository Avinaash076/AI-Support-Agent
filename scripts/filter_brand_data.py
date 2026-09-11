import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "twcs.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

def filter_brand_conversations(brand_handle: str = "AmazonHelp", sample_size: int = 50000):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    print(f"Loading {RAW_PATH} to extract conversations for @{brand_handle}...")
    
    df = pd.read_csv(
        RAW_PATH, 
        dtype={"tweet_id": str, "in_response_to_tweet_id": str, "response_tweet_id": str, "author_id": str}
    )
    
    # Identify outbound tweets from the target brand
    brand_tweets = df[(df["inbound"] == False) & (df["author_id"].str.lower() == brand_handle.lower())]
    print(f"Found {len(brand_tweets)} outbound tweets from @{brand_handle}.")
    
    # Identify incoming customer tweets answered by the brand
    # inbound tweets whose response_tweet_id contains brand tweet ids or in_response_to_tweet_id points to customer
    brand_response_ids = set(brand_tweets["tweet_id"].dropna())
    
    # Find direct inbound tweets that received a response from the brand
    inbound_tweets = df[df["inbound"] == True].copy()
    
    # Match inbound tweets where response_tweet_id overlaps with brand responses
    def receives_brand_response(response_str):
        if pd.isna(response_str):
            return False
        ids = [x.strip() for x in str(response_str).split(",")]
        return any(rid in brand_response_ids for rid in ids)
    
    inbound_for_brand = inbound_tweets[inbound_tweets["response_tweet_id"].apply(receives_brand_response)].copy()
    print(f"Found {len(inbound_for_brand)} incoming customer tweets answered by @{brand_handle}.")
    
    # Join inbound customer query with brand response
    # For fast lookup, map brand responses
    brand_tweet_map = brand_tweets.set_index("tweet_id")["text"].to_dict()
    
    def get_brand_reply(response_str):
        if pd.isna(response_str):
            return ""
        ids = [x.strip() for x in str(response_str).split(",")]
        for rid in ids:
            if rid in brand_tweet_map:
                return brand_tweet_map[rid]
        return ""
    
    inbound_for_brand["brand_response_text"] = inbound_for_brand["response_tweet_id"].apply(get_brand_reply)
    
    # Filter non-empty replies
    pairs_df = inbound_for_brand[inbound_for_brand["brand_response_text"] != ""].copy()
    pairs_df = pairs_df[["tweet_id", "author_id", "created_at", "text", "brand_response_text"]].rename(
        columns={"text": "customer_query", "author_id": "customer_id"}
    )
    
    if sample_size and len(pairs_df) > sample_size:
        pairs_df = pairs_df.head(sample_size)
        
    out_path = os.path.join(PROCESSED_DIR, f"{brand_handle.lower()}_pairs.csv")
    pairs_df.to_csv(out_path, index=False)
    print(f"Saved {len(pairs_df)} query-response pairs to {out_path}!")

if __name__ == "__main__":
    import sys
    brand = sys.argv[1] if len(sys.argv) > 1 else "AmazonHelp"
    filter_brand_conversations(brand)
