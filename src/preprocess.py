import pandas as pd

# Paths
INPUT_PATH = "/Users/mac/Desktop/domain_adaptive_transformer_customer_feedback/data/raw/amazon_fine_food_reviews.csv"
OUTPUT_PATH = "/Users/mac/Desktop/domain_adaptive_transformer_customer_feedback/data/processed/reviews_clean.csv"

def map_sentiment(score):
    if score <= 2:
        return "Negative"
    elif score == 3:
        return "Neutral"
    else:
        return "Positive"

def main():
    print("Loading raw dataset...")
    df = pd.read_csv(INPUT_PATH)

    print("Initial shape:", df.shape)

    # Keep only needed columns
    df = df[["Text", "Score"]].dropna()

    # Map score to sentiment label
    df["label"] = df["Score"].apply(map_sentiment)

    # Rename for consistency
    df = df.rename(columns={"Text": "text"})

    # Final dataset
    df = df[["text", "label"]]

    print("Final shape:", df.shape)
    print("Label distribution:")
    print(df["label"].value_counts())

    # Save processed data
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Clean dataset saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()