import json
import numpy as np

# Initialize model balances and weights
balances = {f"node_{i}": 1000 for i in range(5)}  # Initial deposit of 1000 euros per node
weights = {f"node_{i}": 1.0 for i in range(5)}  # Initial weight of 1.0 per node

# Save initial balances to JSON file
def save_balances():
    with open("balances.json", "w") as f:
        json.dump(balances, f, indent=4)

def save_weights():
    with open("weights.json", "w") as f:
        json.dump(weights, f, indent=4)

save_balances()
save_weights()

# Function to update model weights and apply slashing
def update_weights(predictions, true_labels):
    global balances, weights
    for i, (node, pred) in enumerate(predictions.items()):
        accuracy = np.mean(pred == true_labels)
        weight_adjustment = accuracy * 0.1  # Adjust weight based on accuracy
        weights[node] = max(0.1, min(1.0, weights[node] + weight_adjustment - 0.05))
        if accuracy < 0.5:  # Apply slashing for poor performance
            balances[node] -= 50  # Deduct 50 euros for low accuracy
    save_balances()
    save_weights()

def get_balances():
    with open("balances.json", "r") as f:
        return json.load(f)

def get_weights():
    with open("weights.json", "r") as f:
        return json.load(f)
