import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, request, jsonify
import train  # Assuming train.consensus_prediction is implemented here

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Flask API setup
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/consensus_predict', methods=['GET'])
def predict():
    try:
        # Extract features from the query parameters
        feature_list = request.args.getlist('features', type=float)
        
        if len(feature_list) != 4:
            return jsonify({"error": "Expected exactly 4 features!"}), 400  # Bad request

        # Convert the extracted values into a NumPy array
        input_features = np.array([feature_list])

        # Make a prediction using the consensus model
        prediction_probs = train.consensus_prediction(input_features)
        return jsonify({"probabilities": prediction_probs.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Internal server error

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
