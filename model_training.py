import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load the CSV with no headers (since it’s broken)
df = pd.read_csv("merged_data.csv", header=None, low_memory=False)

# 2. Define correct column names (you can adjust as needed)
columns = [
    "App Name", "App Id", "Category", "Rating", "Rating Count", "Installs", "Minimum Installs",
    "Maximum Installs", "Free", "Price", "Currency", "Size", "Minimum Android", "Developer Id",
    "Developer Website", "Developer Email", "Released", "Last Updated", "Content Rating",
    "Privacy Policy", "Ad Supported", "In App Purchases", "Editors Choice", "Scraped Time",
    "scam"  # Final label column (0 = safe, 1 = scam)
]

# 3. Assign column names to the dataframe
df.columns = columns

# 4. Remove rows with missing label
df = df[df["scam"].notna()]

# 5. Optional: Print shape and preview
print("Shape after cleanup:", df.shape)
print(df.head(2))

# 6. Encode categorical features (simplified)
df_encoded = pd.get_dummies(df.drop(["App Id", "Developer Email", "App Name"], axis=1), drop_first=True)

# 7. Separate features and label
X = df_encoded.drop("scam", axis=1)
y = df_encoded["scam"].astype(int)

# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 10. Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 11. Save model for later use
import joblib
joblib.dump(model, "scam_detector_model.pkl")
print("✅ Model saved as scam_detector_model.pkl")
