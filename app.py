import joblib
import pandas as pd
from metadata_utils import fetch_app_metadata, clean_numeric_string, clean_size_string, extract_min_android_version
import numpy as np # Import numpy for numerical operations

# --- Configuration ---
# Make sure this matches the model file actually saved by your training script
MODEL_PATH = "decoy_detect.joblib" # Or "scam_detector_model.pkl" if you're using RandomForest

# Load trained model
try:
    with open(MODEL_PATH, "rb") as file:
        model = joblib.load(file)
    # print(f"✅ Model loaded successfully from {MODEL_PATH}!") # Commented out
    # Model expects 30 features. # Commented out
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_
    else:
        # print("Warning: Model does not have 'feature_names_in_'. Ensure 'dummy_cols' and feature list below are accurate.") # Commented out
        model_features = np.array([
            'Installs', 'Free', 'Size', 'Minimum Android', 'Ad Supported', 'In App Purchases',
            'Editors Choice', 'Rating', 'Rating Count', 'description_len', 'Minimum Installs',
            'Maximum Installs', 'Price', 'Privacy Policy', 'Category_Auto & Vehicles', 'Category_Beauty',
            'Category_Books & Reference', 'Category_Business', 'Category_Comics', 'Category_Communication',
            'Category_Dating', 'Category_Education', 'Category_Entertainment', 'Category_Events',
            'Category_Finance', 'Category_Food & Drink', 'Category_Health & Fitness', 'Category_House & Home',
            'Category_Libraries & Demo', 'Category_Lifestyle', 'Category_Maps & Navigation', 'Category_Medical',
            'Category_Music & Audio', 'Category_News & Magazines', 'Category_Parenting', 'Category_Personalization',
            'Category_Photography', 'Category_Productivity', 'Category_Shopping', 'Category_Social',
            'Category_Sports', 'Category_Tools', 'Category_Travel & Local', 'Category_Video Players & Editors',
            'Category_Weather', 'Content Rating_Everyone', 'Content Rating_Mature 17+', 'Content Rating_Teen',
            'Currency_EUR', 'Currency_INR', 'Currency_USD', 'Developer Id_5445778848498877028',
            'Developer Id_6023530737119560417', 'Developer Id_6507661036081498687', 'Developer Id_7060370428587635697',
            'Developer Id_8003666014451006509', 'Developer Website_google.com', 'Released_2016-01-01',
            'Released_2017-01-01', 'Released_2018-01-01', 'Released_2019-01-01', 'Released_2020-01-01',
            'Released_2021-01-01', 'Released_2022-01-01', 'Released_2023-01-01', 'Released_2024-01-01',
            'Last Updated_2024-01-01', 'Last Updated_2024-02-01', 'Last Updated_2024-03-01',
            'Last Updated_2024-04-01', 'Last Updated_2024-05-01', 'Last Updated_2024-06-01'
        ])


    # print(f"Model expects {len(model_features)} features.") # Commented out
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit() # Exit if model cannot be loaded

# Input URL from user
user_url = input("Paste a Play Store URL to scan: ")

# Fetch metadata
metadata = fetch_app_metadata(user_url)

# Check for errors
if "error" in metadata:
    print("❌ Error while fetching metadata:", metadata["error"])
    if "details" in metadata:
        print("Details:", metadata["details"])
else:
    # print("✅ Metadata fetched successfully!") # Commented out
    # print("Raw metadata:", metadata) # Uncomment to see all raw fetched data if needed

    # Prepare data for prediction
    # Create a DataFrame from the fetched metadata
    # Ensure keys match expected column names (e.g., 'genre' from scraper is 'Category')
    # Use a list of dicts to handle single row DataFrames correctly
    input_data = [{
        "Category": metadata.get('genre'), # 'genre' from scraper usually maps to 'Category'
        "Installs": metadata.get('installs'),
        "Free": metadata.get('free'),
        "Size": metadata.get('size'),
        "Minimum Android": metadata.get('minAndroidVersion'),
        "Content Rating": metadata.get('contentRating'),
        "Ad Supported": metadata.get('adSupported'),
        "In App Purchases": metadata.get('inAppPurchases'),
        "Editors Choice": metadata.get('editorsChoice'),
        # Add other numerical features that were part of your original training,
        # e.g., 'score', 'ratings', 'description_len', if they were used without get_dummies
        "Rating": metadata.get('score'), # Mapping 'score' to 'Rating'
        "Rating Count": metadata.get('ratings'), # Mapping 'ratings' to 'Rating Count'
        "description_len": metadata.get('description_len'),
        # Other numericals, if they were used directly:
        "Minimum Installs": metadata.get('minInstalls'),
        "Maximum Installs": metadata.get('maxInstalls'),
        "Price": metadata.get('price'),
        # For 'Currency', 'Developer Id', 'Developer Website', 'Released', 'Last Updated', 'Privacy Policy'
        # these would also need to be included if they were used for get_dummies
        "Currency": metadata.get('currency'),
        "Developer Id": metadata.get('developerId'),
        "Developer Website": metadata.get('developerWebsite'),
        "Released": metadata.get('released'),
        "Last Updated": metadata.get('updated'),
        "Privacy Policy": metadata.get('privacyPolicy') # This is a URL, probably needs to be boolean (has_policy)
    }]
    input_df = pd.DataFrame(input_data)

    # --- Replicate Preprocessing from Training Script ---
    # 1. Handle 'Privacy Policy': Convert URL to boolean (presence of URL means True)
    input_df['Privacy Policy'] = input_df['Privacy Policy'].apply(lambda x: x is not None and x != '')

    # 2. Convert boolean-like columns to 0/1 (int)
    for col in ['Free', 'Ad Supported', 'In App Purchases', 'Editors Choice', 'Privacy Policy']:
        if col in input_df.columns: # Check if column exists before converting
            input_df[col] = input_df[col].astype(int)

    # 3. Handle potential NaN values and ensure numerical types
    numerical_cols = ['Installs', 'Size', 'Minimum Android', 'Rating', 'Rating Count', 'description_len',
                      'Minimum Installs', 'Maximum Installs', 'Price']
    for col in numerical_cols:
        if col in input_df.columns:
            # Convert to numeric, errors='coerce' will turn non-convertible values into NaN
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0) # Fill NaN with 0

    # Handle categorical columns: fill NaNs with 'Unknown'
    categorical_cols = ['Category', 'Content Rating', 'Currency', 'Developer Id', 'Developer Website', 'Released', 'Last Updated']
    for col in categorical_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].fillna('Unknown')

    # Important: Drop columns that were excluded from get_dummies in training
    # Based on your model_training.py, these were 'App Id', 'Developer Email', 'App Name', 'Scraped Time'
    columns_to_drop_before_dummies = ["appId", "developerEmail", "title", "Scraped Time", "developer", "reviews"] # Adjusted based on fetch_app_metadata output and original training logic
    for col in columns_to_drop_before_dummies:
        if col in input_df.columns: # Check if column exists before dropping
            input_df = input_df.drop(columns=[col])


    # 4. Apply One-Hot Encoding (pd.get_dummies)
    # Get list of columns to dummify based on the current DataFrame, excluding numerical and boolean-converted ones
    # Make sure this list reflects how your training data was processed
    cols_to_dummify = [col for col in input_df.columns if input_df[col].dtype == 'object'] # Object dtype typically means strings/categorical

    # You MUST ensure that the categories seen here are aligned with the categories seen during training.
    # The safest way is to save the actual `columns` of your training X_train AFTER get_dummies,
    # or to use a `ColumnTransformer` with `handle_unknown='ignore'`.
    # For now, we rely on the `model_features` alignment below.
    input_df_encoded = pd.get_dummies(input_df, columns=cols_to_dummify, drop_first=True)


    # 5. Align columns with the model's expected features (model_features)
    # This is the most robust way to ensure feature consistency.
    missing_cols = set(model_features) - set(input_df_encoded.columns)
    for c in missing_cols:
        input_df_encoded[c] = 0 # Add missing one-hot encoded columns as 0

    # Ensure the order of columns and presence matches the training data
    # Filter to only features the model expects and reorder them
    # Make sure all columns in model_features are also in input_df_encoded (after adding missing)
    final_input_df = input_df_encoded[model_features]

    # Debugging: Print final input DataFrame info
    # print("\nFinal input DataFrame for prediction:") # Commented out
    # print(final_input_df.head()) # Commented out
    # print("Shape:", final_input_df.shape) # Commented out
    # print("Columns:", final_input_df.columns.tolist()) # Commented out


    # Predict
    try:
        prediction = model.predict(final_input_df)[0]
        prediction_proba = model.predict_proba(final_input_df)[0] # Get probabilities

        # Display result # Commented out
        # print("\n--- Prediction Result ---") # Commented out
        if prediction == 1:
            model_outcome = "SCAM"
            model_confidence = prediction_proba[1]
            # print(f"🔴 This app is likely a **SCAM** ❌ (Confidence: {model_confidence:.2%})") # Commented out
        else:
            model_outcome = "SAFE"
            model_confidence = prediction_proba[0]
            # print(f"🟢 This app is likely **SAFE** ✅ (Confidence: {model_confidence:.2%})") # Commented out

        # --- Final Download Recommendation ---
        fetched_score = metadata.get('score', 0.0) # Get the fetched score
        fetched_installs = metadata.get('installs', 0) # Get the fetched installs
        fetched_ratings_count = metadata.get('ratings', 0) # Get the fetched ratings count

        # Print the fetched details
        print(f"Fetched Score: {fetched_score}")
        print(f"Fetched Ratings Count: {fetched_ratings_count}")
        print(f"Fetched Installs: {fetched_installs}")
        print("\n--- Download Recommendation ---")


        # Rule 1: Check for very low rating (e.g., less than 2.0 out of 5)
        is_very_low_rating = False
        if isinstance(fetched_score, (int, float)) and fetched_score < 2.0: # You can adjust this threshold (e.g., 2.5)
            is_very_low_rating = True

        # Rule 2: Low downloads but high rating (potential for fake reviews/artificial popularity)
        is_suspicious_downloads_ratings = False
        if isinstance(fetched_installs, (int, float)) and isinstance(fetched_score, (int, float)):
            # Example thresholds: less than 10,000 installs AND rating 4.0 or higher
            if fetched_installs < 10000 and fetched_score >= 4.0:
                is_suspicious_downloads_ratings = True

        # Rule 3: High installs but disproportionately low ratings count
        is_discrepant_installs_ratings_count = False
        if isinstance(fetched_installs, (int, float)) and isinstance(fetched_ratings_count, (int, float)):
            # Example thresholds: 1 Million+ installs AND less than 10,000 ratings
            if fetched_installs >= 1_000_000 and fetched_ratings_count < 10000:
                is_discrepant_installs_ratings_count = True


        # Determine final recommendation to print
        if is_very_low_rating:
            print("🔴 **RED FLAG:** This app has a very low rating. It is **NOT RECOMMENDED** to download.\n")
        elif is_suspicious_downloads_ratings:
            print("🟠 **ORANGE FLAG:** This app has relatively low downloads but a surprisingly high rating.")
            print("It is **RECOMMENDED to proceed with caution**.\n")
        elif is_discrepant_installs_ratings_count:
            print("🟠 **ORANGE FLAG:** This app has a very high number of installs but disproportionately few ratings.")
            print("It is **RECOMMENDED to proceed with caution**.\n")
        elif model_outcome == "SCAM":
            print("🔴 **RED FLAG:** Based on the model's prediction, it is **NOT RECOMMENDED** to download.\n")
        else:
            print("🟢 **GREEN FLAG:** Based on the model's prediction, it appears **SAFE to download.**\n")
            # print("Always exercise caution and check permissions before downloading any app.") # Can be uncommented if desired


    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        # print("This often means the input features do not match the model's expectations.") # Commented out
        # print("Debug Info: Input DataFrame columns:", final_input_df.columns.tolist()) # Commented out
        # print("Debug Info: Expected Model features:", model_features.tolist()) # Commented out
        # print("Missing in Input but expected by model:", set(model_features) - set(final_input_df.columns)) # Commented out
        # print("In Input but not expected by model:", set(final_input_df.columns) - set(model_features)) # Commented out