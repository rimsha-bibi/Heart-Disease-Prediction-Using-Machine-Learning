from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the saved model
model = joblib.load('rf_classifier.joblib')

# Define the route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Extract the features from the JSON
    features = [
        data['Age'],
        data['Cholesterol'],
        data['Blood Pressure'],
        data['Heart Rate'],
        data['Smoking'],
        data['Alcohol Intake'],
        data['Exercise Hours'],
        data['Family History'],
        data['Diabetes'],
        data['Obesity'],
        data['Stress Level'],
        data['Blood Sugar'],
        data['Exercise Induced Angina'],
        data['Chest Pain Type'],
        data['Log Blood Pressure'],
        data['Cholesterol_BloodPressure'],
        data['Exercise_Stress'],
        data['Cholesterol_Ratio'],
        data['Mean Arterial Pressure'],
        data['Risk_Score']
    ]

    # Convert features to a numpy array and reshape for the model
    features_array = np.array(features).reshape(1, -1)

    # Make a prediction using the model
    prediction = model.predict(features_array)

    # Return the prediction result as JSON
    return jsonify({'Heart Disease': bool(prediction[0])})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
