# Data Sampling Methodology

## Source
Primary dataset: `customer-support-on-twitter` (Kaggle).

## Filtering
1. We filtered the 3 million tweet dataset strictly for outbound tweets authored by `@AppleSupport`.
2. We mapped these to inbound tweets where the `response_tweet_id` matched our target.
3. We dropped empty or mismatched pairs, leaving a clean set of paired (customer query -> brand response) interactions.

## Subsampling
To build the offline index, we subsampled the first 50,000 interactions to keep memory usage low for local TF-IDF vectorization.

To build the golden evaluation set (`eval/answer_quality.json`), we drew 40 handcrafted examples covering all core intent categories. We later augmented this with synthetic cases (tracked separately in eval) to reach a larger n-count for stability testing.
