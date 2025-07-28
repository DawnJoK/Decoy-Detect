# Save this content as metadata_utils.py
from google_play_scraper import app, Sort, reviews
import re

# Helper function to clean numeric strings like '1,000,000+'
def clean_numeric_string(s):
    if isinstance(s, (int, float)):
        return s
    if isinstance(s, str):
        s = s.replace('+', '').replace(',', '').strip()
        if 'M' in s:
            return float(s.replace('M', '')) * 1_000_000
        elif 'K' in s:
            return float(s.replace('K', '')) * 1_000
        try:
            return int(s)
        except ValueError:
            return 0 # Default to 0 if conversion fails
    return 0

# Helper function to clean size strings like '12M'
def clean_size_string(s):
    if isinstance(s, (int, float)):
        return s
    if isinstance(s, str):
        s = s.strip()
        if 'M' in s:
            return float(s.replace('M', '')) # Return in MB
        elif 'K' in s:
            return float(s.replace('K', '')) / 1024 # Convert KB to MB
        elif 'G' in s:
            return float(s.replace('G', '')) * 1024 # Convert GB to MB
        try:
            # If it's just a number, assume bytes and convert to MB
            return int(s) / (1024 * 1024)
        except ValueError:
            return 0.0 # Default to 0 if conversion fails
    return 0.0

# Helper function to extract major version from "4.1 and up"
def extract_min_android_version(s):
    if isinstance(s, (int, float)):
        return s
    if isinstance(s, str):
        match = re.search(r'^(\d+\.?\d*)', s)
        if match:
            return float(match.group(1))
    return 0.0 # Default or indicator for missing/unknown

# Extract package name from URL
def extract_package_name(url):
    match = re.search(r'id=([a-zA-Z0-9_.]+)', url)
    return match.group(1) if match else None

# Fetch metadata from Play Store
def fetch_app_metadata(url):
    package_name = extract_package_name(url)
    if not package_name:
        return {"error": "Fake URL detected", "details": "Could not extract a valid app ID from the URL."}

    try:
        app_data = app(package_name)

        # Extracting required fields, handling potential missing keys
        metadata = {
            "appId": app_data.get('appId', None),
            "title": app_data.get('title', None),
            "developer": app_data.get('developer', None),
            "developerId": app_data.get('developerId', None),
            "developerWebsite": app_data.get('developerWebsite', None),
            "developerEmail": app_data.get('developerEmail', None),
            "score": app_data.get('score', 0.0),
            "ratings": app_data.get('ratings', 0),
            "reviews": app_data.get('reviews', 0), # Total number of reviews
            "installs": clean_numeric_string(app_data.get('installs', '0+')),
            "minInstalls": clean_numeric_string(app_data.get('minInstalls', '0')),
            "maxInstalls": clean_numeric_string(app_data.get('maxInstalls', '0')),
            "free": app_data.get('free', True), # Defaults to True if not specified
            "price": app_data.get('price', 0.0),
            "currency": app_data.get('currency', 'USD'),
            "size": clean_size_string(app_data.get('size', '0M')),
            "minAndroidVersion": extract_min_android_version(app_data.get('minAndroid', '0.0')),
            "genre": app_data.get('genre', 'Unknown'), # This is 'Category' in your model
            "released": app_data.get('released', None),
            "updated": app_data.get('updated', None),
            "contentRating": app_data.get('contentRating', 'Everyone'),
            "privacyPolicy": app_data.get('privacyPolicy', None), # URL, check if not None
            "adSupported": app_data.get('adSupported', False),
            "inAppPurchases": app_data.get('inAppPurchases', False),
            "editorsChoice": app_data.get('editorsChoice', False),
            "description_len": len(app_data.get('description', '')),
            # Add any other fields you found useful in your training data
        }
        return metadata

    except Exception as e:
        import traceback
        # Distinguish between "App not found" (404) and other scraping errors
        if "App not found(404)" in str(e):
            return {"error": "App not found (404)", "details": "The app at this URL could not be found on the Play Store. It might have been removed."}
        else:
            return {"error": str(e), "details": traceback.format_exc()}