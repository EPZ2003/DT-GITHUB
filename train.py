import json
import random
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, request, jsonify

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print dataset information
print("Training Data:")
print(X_train)
print("Training Labels:")
print(y_train)
print("Test Data:")
print(X_test)
print("Test Labels:")
print(y_test)

# Define a PyTorch neural network model
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Node class representing a participant in the network
class Node:
    def __init__(self, node_id, model_type="sklearn"):
        self.node_id = node_id
        self.model_type = model_type
        self.balance = 1000  # Initial balance for the node
        self.weight = 1.0
        if model_type == "sklearn":
            self.model = RandomForestClassifier()
        elif model_type == "torch":
            self.model = IrisNet()
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def train(self, X, y):
        if self.model_type == "sklearn":
            self.model.fit(X, y)
        elif self.model_type == "torch":
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)
            for _ in range(100):  # Train for 100 epochs
                self.optimizer.zero_grad()
                outputs = self.model(X_tensor)
                loss = self.criterion(outputs, y_tensor)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        if self.model_type == "sklearn":
            return self.model.predict_proba(X)
        elif self.model_type == "torch":
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                outputs = self.model(X_tensor)
                return torch.softmax(outputs, dim=1).numpy()

# Initialize nodes
nodes = [Node(node_id=i, model_type=random.choice(["sklearn", "torch"])) for i in range(5)]

# Train models on each node
for node in nodes:
    node.train(X_train, y_train)

# Consensus prediction function
def consensus_prediction(X):
    predictions = []
    for node in nodes:
        predictions.append(node.predict(X))
    
    # Average predictions from all nodes
    avg_prediction = np.mean(predictions, axis=0)
    return avg_prediction

# Flask API setup
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Convert input to numpy array
        input_features = np.array(data["features"], dtype=float).reshape(1, -1)
        prediction_probs = consensus_prediction(input_features)
        return jsonify({"probabilities": prediction_probs.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

# Test consensus model
y_pred_probs = consensus_prediction(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
accuracy = np.mean(y_pred == y_test)
print(f"Consensus model accuracy: {accuracy}")

# Print predictions
print("Prediction Probabilities:")
print(y_pred_probs)

# Simulate a JSON database to track balances
balances = {f"node_{node.node_id}": node.balance for node in nodes}
with open("balances.json", "w") as f:
    json.dump(balances, f, indent=4)

print("Balances saved to balances.json")
