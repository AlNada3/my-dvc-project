import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Step 1: Load preprocessed data
train_data = pd.read_csv("data/processed/train_processed.csv")
test_data = pd.read_csv("data/processed/test_processed.csv")

# Step 2: Remove NaN values (Fix for ValueError)
train_data.dropna(subset=["content"], inplace=True)
test_data.dropna(subset=["content"], inplace=True)

# Step 3: Initialize CountVectorizer
vectorizer = CountVectorizer()

# Step 4: Transform text into numerical features
X_train = vectorizer.fit_transform(train_data["content"].astype(str))  # Convert to string
X_test = vectorizer.transform(test_data["content"].astype(str))  # Convert to string

# Step 5: Save features
train_features = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
test_features = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out())

train_features.to_csv("data/features/train_bow.csv", index=False)
test_features.to_csv("data/features/test_bow.csv", index=False)

print("✅ Feature extraction completed! Files saved in 'data/features/'.")
