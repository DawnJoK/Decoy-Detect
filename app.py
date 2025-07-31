from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from metadata_utils import fetch_app_metadata
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# --- Configuration ---
MODEL_PATH = "decoy_detect.joblib"
MODEL_FEATURES_PATH = "model_features.pkl"

model_features = []
try:
    if os.path.exists(MODEL_FEATURES_PATH):
        model_features = joblib.load(MODEL_FEATURES_PATH)
        print(f"✅ Model features loaded successfully from '{MODEL_FEATURES_PATH}'!")
        print(f"Loaded {len(model_features)} features. First 5: {model_features[:5]}")
    else:
        print(f"WARNING: '{MODEL_FEATURES_PATH}' not found. Attempting to get features from model or using fallback.")

except Exception as e:
    print(f"❌ Error loading model features from '{MODEL_FEATURES_PATH}': {e}")

try:
    with open(MODEL_PATH, "rb") as file:
        model = joblib.load(file)

    if not model_features and hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
        model_features = model.feature_names_in_
        print("✅ Model features obtained from model.feature_names_in_!")
    elif not model_features:
        print("CRITICAL WARNING: Model features could not be loaded or inferred. Predictions may be unreliable.")

    print(f"✅ Model loaded successfully from {MODEL_PATH}!")
    if len(model_features) > 0:
        print(f"Model expects {len(model_features)} features.")
    else:
        print("Model features list is empty. This will likely lead to prediction errors.")

except Exception as e:
    print(f"❌ Error loading model or its features: {e}")
    exit()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get("url", "")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    metadata = fetch_app_metadata(url)
    print(f"\n--- Fetched Metadata for {url}: ---")
    print(metadata)

    if "error" in metadata:
        print(f"Error fetching metadata: {metadata['error']} - Details: {metadata.get('details', '')}")
        return jsonify({"error": metadata["error"], "details": metadata.get("details", "")}), 500

    input_data = [{
        "Category": metadata.get('genre'),
        "Installs": metadata.get('installs'),
        "Free": metadata.get('free'),
        "Size": metadata.get('size'),
        "Minimum Android": metadata.get('minAndroidVersion'),
        "Content Rating": metadata.get('contentRating'),
        "Ad Supported": metadata.get('adSupported'),
        "In App Purchases": metadata.get('inAppPurchases'),
        "Editors Choice": metadata.get('editorsChoice'),
        "Rating": metadata.get('score'),
        "Rating Count": metadata.get('ratings'),
        "description_len": metadata.get('description_len'),
        "Minimum Installs": metadata.get('minInstalls'),
        "Maximum Installs": metadata.get('maxInstalls'),
        "Price": metadata.get('price'),
        "Currency": metadata.get('currency'),
        "Developer Id": metadata.get('developerId'),
        "Developer Website": metadata.get('developerWebsite'),
        "Released": metadata.get('released'),
        "Last Updated": metadata.get('updated'),
        "Privacy Policy": metadata.get('privacyPolicy'),
        "App Name": metadata.get('title'),
        "App Id": metadata.get('appId'),
        "Developer Email": metadata.get('developerEmail'),
        "Scraped Time": None
    }]
    df = pd.DataFrame(input_data)
    print(f"\n--- DataFrame after initial metadata load: ---")
    print(df)

    df['is_scam_rule_based'] = 0

    rule1 = (df['Rating'] == 5.0) & (df['Maximum Installs'] < 1000)
    rule2 = df['Developer Website'].isna() | df['Developer Email'].isna()
    rule3 = df['Developer Email'].astype(str).str.contains('@gmail', na=False)
    rule4 = df['App Name'].astype(str).str.contains(r'(?i)gift|cash|free|win|earn|money|rich|giveaway', na=False)
    rule5 = (df['Price'] == 0) & (df['In App Purchases'].astype(str).str.lower() == 'true')

    df.loc[(rule1 & rule2 & rule3) | rule4 | rule5, 'is_scam_rule_based'] = 1

    df['Released'] = pd.to_datetime(df['Released'], errors='coerce')
    df['Last Updated'] = pd.to_datetime(df['Last Updated'], errors='coerce')

    fixed_current_timestamp = pd.Timestamp(datetime.now().date())
    df['app_age'] = (fixed_current_timestamp - df['Released']).dt.days / 365
    df['days_since_update'] = (fixed_current_timestamp - df['Last Updated']).dt.days

    df['has_policy'] = df['Privacy Policy'].notna().astype(int)

    for col in ['Free', 'Ad Supported', 'In App Purchases', 'Editors Choice']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    numerical_cols = ['Installs', 'Size', 'Minimum Android', 'Rating', 'Rating Count', 'description_len',
                      'Minimum Installs', 'Maximum Installs', 'Price', 'app_age', 'days_since_update']
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    cols_to_drop_training = [
        'App Name', 'App Id', 'Developer Email', 'Developer Website',
        'Released', 'Last Updated', 'Scraped Time', 'Privacy Policy',
        'Developer Id', 'Currency'
    ]
    columns_to_remove_from_training = [col for col in df.columns if
                                        col.startswith("20") or
                                        col.startswith("Apr") or
                                        col.startswith("Dec") or
                                        "http" in str(col) or
                                        "False" in str(col) or
                                        col == "Everyone"]

    all_cols_to_drop = list(set(cols_to_drop_training + columns_to_remove_from_training))
    df.drop(columns=all_cols_to_drop, inplace=True, errors='ignore')

    categorical_cols_for_ohe = [col for col in df.columns if df[col].dtype == 'object']
    df_encoded = pd.get_dummies(df, columns=categorical_cols_for_ohe, drop_first=True)

    if 'Free' in df_encoded.columns and df_encoded['Free'].dtype == 'bool':
        df_encoded['Free'] = df_encoded['Free'].astype(int)

    if len(model_features) == 0:
        print("ERROR: model_features is empty. Cannot align input DataFrame. This is a critical error.")
        return jsonify({"error": "Model features not loaded. Cannot make prediction."}), 500

    missing_cols = set(model_features) - set(df_encoded.columns)
    for c in missing_cols:
        df_encoded[c] = 0

    extra_cols = set(df_encoded.columns) - set(model_features)
    df_encoded = df_encoded.drop(columns=list(extra_cols))

    final_input_df = df_encoded[model_features]

    print(f"\n--- Final Input DataFrame for Prediction: ---")
    print(final_input_df)
    print(f"\n--- Final Input DataFrame Columns: ---")
    print(final_input_df.columns.tolist())

    try:
        prediction = model.predict(final_input_df)[0]
        proba = model.predict_proba(final_input_df)[0]
        print(f"\n--- Prediction Probabilities: {proba} ---")

        probability_safe = proba[0]
        confidence_in_predicted_class = proba[prediction]

        # --- Recommendation Logic (using rule-based flag and model prediction) ---
        recommendation = ""
        recommendation_category = "GREEN" # Default category
        fetched_score = metadata.get('score', 0.0)
        fetched_installs = metadata.get('installs', 0)
        fetched_ratings_count = metadata.get('ratings', 0)

        is_very_low_rating = False
        if isinstance(fetched_score, (int, float)) and fetched_score < 2.0:
            is_very_low_rating = True

        is_suspicious_downloads_ratings = False
        if isinstance(fetched_installs, (int, float)) and isinstance(fetched_score, (int, float)):
            if fetched_installs < 10000 and fetched_score >= 4.0:
                is_suspicious_downloads_ratings = True

        is_discrepant_installs_ratings_count = False
        if isinstance(fetched_installs, (int, float)) and isinstance(fetched_ratings_count, (int, float)):
            if fetched_installs >= 1_000_000 and fetched_ratings_count < 10000:
                is_discrepant_installs_ratings_count = True

        # Determine the recommendation and its category
        if is_very_low_rating:
            recommendation = "🔴 RED FLAG: This app has a very low rating. It is NOT RECOMMENDED to download."
            recommendation_category = "RED"
        elif is_suspicious_downloads_ratings:
            recommendation = "🟠 ORANGE FLAG: This app has relatively low downloads but a surprisingly high rating. It is RECOMMENDED to proceed with caution."
            recommendation_category = "ORANGE"
        elif is_discrepant_installs_ratings_count:
            recommendation = "🟠 ORANGE FLAG: This app has a very high number of installs but disproportionately few ratings. It is RECOMMENDED to proceed with caution."
            recommendation_category = "ORANGE"
        elif prediction == 1: # If model predicts SCAM (class 1)
            recommendation = "🔴 RED FLAG: Based on the model's prediction, it is NOT RECOMMENDED to download."
            recommendation_category = "RED"
        else: # If model predicts SAFE (class 0)
            recommendation = "🟢 GREEN FLAG: Based on the model's prediction, it appears SAFE to download."
            recommendation_category = "GREEN"

        # FIX: Make trust_score deterministic based on recommendation_category
        # Use fixed values or a simple derivation for consistency
        if recommendation_category == "RED":
            trust_score = 20 # Fixed low score for RED
        elif recommendation_category == "ORANGE":
            trust_score = 60 # Fixed medium score for ORANGE
        else: # GREEN
            # For GREEN, use the model's probability of being safe, rounded
            trust_score = round(probability_safe * 100)
            # Ensure it's at least 70 if it's green, to align with color logic
            if trust_score < 70:
                trust_score = 70


        return jsonify({
            "prediction": int(prediction),
            "trust_score": trust_score, # Now this will be deterministic based on category
            "recommendation": recommendation,
            "confidence": round(confidence_in_predicted_class, 4),
            "recommendation_category": recommendation_category # New field to send to frontend
        })

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
