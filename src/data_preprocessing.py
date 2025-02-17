import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Load data
train_data = pd.read_csv("data/raw/train.csv")
test_data = pd.read_csv("data/raw/test.csv")

# Apply text cleaning
train_data["content"] = train_data["content"].apply(clean_text)
test_data["content"] = test_data["content"].apply(clean_text)

# Save cleaned data
train_data.to_csv("data/processed/train_processed.csv", index=False)
test_data.to_csv("data/processed/test_processed.csv", index=False)

print("✅ Data preprocessing completed. Processed files saved in 'data/processed/'.")

