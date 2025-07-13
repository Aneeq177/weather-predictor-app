# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# --- Define the file path for the new dataset ---
DATA_FILE_PATH = os.path.join('data', 'Weather Data.csv')

def train_and_save_model():
    """Loads rich weather data, trains an accurate classifier, and saves it."""
    print("Loading final weather dataset...")
    try:
        df = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset not found at '{DATA_FILE_PATH}'.")
        return

    # --- 1. Data Cleaning and Feature Engineering ---
    print("Cleaning data and preparing features...")
    
    # We will predict the general 'Weather' condition
    # Let's clean up the data - we'll focus on single-condition events for simplicity
    df = df[~df['Weather'].str.contains(',')].copy()
    
    # --- Use the powerful, predictive features from this dataset ---
    features = [
        'Temp_C', 
        'Dew Point Temp_C', 
        'Rel Hum_%', 
        'Wind Speed_km/h',
        'Visibility_km',
        'Press_kPa'
    ]
    target = 'Weather'
    
    df.dropna(subset=features + [target], inplace=True)
    
    # Encode our target variable 'Weather' since it's text
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])
    
    # We will save this label encoder to decode the prediction later
    encoders = {'Weather': le}

    X = df[features]
    y = df[target]

    # --- 2. Model Training ---
    print("Splitting data and training the improved model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # --- 3. Model Evaluation ---
    y_pred = model.predict(X_test)
    # Decode the predictions and test values to make the report readable
    y_test_decoded = encoders['Weather'].inverse_transform(y_test)
    y_pred_decoded = encoders['Weather'].inverse_transform(y_pred)
    
    print("\n--- Final Model Evaluation Report ---")
    # Setting zero_division=0 handles cases where a class has no predictions
    print(classification_report(y_test_decoded, y_pred_decoded, zero_division=0))

    # --- 4. Save the new, more accurate Model and Encoders ---
    print("Saving final model and encoders...")
    joblib.dump(model, 'weather_predictor_model.joblib')
    joblib.dump(encoders, 'weather_encoders.joblib')
    print("Final model saved successfully!")

if __name__ == "__main__":
    train_and_save_model()