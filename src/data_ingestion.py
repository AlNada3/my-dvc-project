import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
URL = "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"
print("📥 Downloading dataset from:", URL, flush=True)
df = pd.read_csv(URL)

# Step 2: Verify dataset
if df.empty:
    print("❌ ERROR: Dataset failed to load or is empty!", flush=True)
    exit()

print(f"✅ Dataset loaded! Total rows: {len(df)}", flush=True)

# Step 3: Keep only relevant columns
df = df[['sentiment', 'content']]
print(f"✅ Relevant columns kept! Data now has {len(df)} rows.", flush=True)

# Step 4: Define emotion mappings
happy_labels = ["happiness", "love"]
sad_labels = ["sadness", "fear", "anger"]

# Step 5: Map emotions to binary values
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x in happy_labels else (0 if x in sad_labels else None))

# Step 6: Drop rows with undefined labels
df.dropna(inplace=True)
print(f"✅ Data filtered! Rows after removing neutral sentiments: {len(df)}", flush=True)

# Step 7: Split into training and testing sets
if len(df) > 0:
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    # Print preview before saving
    print("\n🔹 Train Data Sample:\n", train_data.head(), flush=True)
    print("\n🔹 Test Data Sample:\n", test_data.head(), flush=True)

    # Step 8: Save processed data
    train_data.to_csv("data/raw/train.csv", index=False)
    test_data.to_csv("data/raw/test.csv", index=False)

    print(f"✅ Train set size: {len(train_data)} rows", flush=True)
    print(f"✅ Test set size: {len(test_data)} rows", flush=True)
    print("✅ Data ingestion completed successfully! Files saved in 'data/raw/'.", flush=True)
else:
    print("❌ ERROR: No data left after filtering! Check sentiment mapping.", flush=True)
