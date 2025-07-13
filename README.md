Machine Learning Weather Predictor & Alert System
This project is an end-to-end machine learning application that predicts weather conditions based on meteorological data. It features a Python script for data processing and model training, and an interactive web application built with Streamlit that allows users to input live weather data and receive a real-time prediction.

This project demonstrates the complete lifecycle of a data science project, from data acquisition and cleaning to model training, evaluation, and deployment as a user-facing application.

Final Application Screenshot
(This is where you should add a screenshot of your final, working Streamlit app!)

Features
Data Processing: Uses pandas for efficient loading, cleaning, and preparation of weather data.

Machine Learning Model: A RandomForestClassifier is trained using scikit-learn to classify weather events.

Model Persistence: The trained model and data encoders are saved to disk using joblib for later use in the application.

Interactive Web UI: A user-friendly web application built with Streamlit allows for real-time predictions based on user-inputted data.

Secure Configuration: A .env file is used to manage sensitive information like database credentials securely.

Tech Stack
Language: Python

Data Manipulation: Pandas

Machine Learning: Scikit-learn

Web Framework: Streamlit

Model Persistence: Joblib

Development Journey & Challenges Solved
A key part of this project was navigating real-world data science challenges. The initial dataset, while large, proved to be inadequate for accurate predictions, leading to a necessary and successful pivot.

Initial KeyError Debugging: The first version of the script failed with a KeyError. This was diagnosed by programmatically printing the actual column names from the CSV file and comparing them to the feature list in the code. The problem was solved by correcting the feature list to match the available data.

Diagnosing Model Overfitting: After fixing the initial errors, the model was observed to be highly inaccurate, always predicting a single outcome ("Hail"). This was identified as a classic case of overfitting due to the dataset lacking key predictive features (like temperature, pressure, and humidity).

Pivoting to a Richer Dataset: Recognizing the data quality issue, I sourced a new, more detailed dataset that included the necessary meteorological features. This pivot was crucial for improving model accuracy.

Solving the stratify ValueError: The new dataset contained some weather categories with only one sample. This caused a ValueError in scikit-learn's train_test_split function when using the stratify parameter. The issue was resolved by removing stratification, a practical trade-off for handling imbalanced classes in a large dataset.

This iterative process of diagnosing errors, evaluating model performance, and improving data quality was fundamental to the project's success.

How to Run This Project Locally
Clone the Repository

git clone [your-repo-link-here]
cd weather-alert-system

Set Up the Environment

# Create and activate a virtual environment
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

(Note: You will need to create a requirements.txt file. See instructions below.)

Prepare the Data

Download the dataset from Kaggle.

Create a data folder in the project root.

Place the CSV file inside and rename it to Weather Data.csv.

Train the Model

Run the training script to generate the model files:

python train_model.py

Run the Web App

Launch the Streamlit application:

streamlit run app.py

The application will be available at http://localhost:8501.