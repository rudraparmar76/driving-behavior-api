from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load the trained model, scaler, and label encoder
model = load_model('motion_classification_model.keras')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')  # Save during training

# Feature Columns
feature_columns = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']
sequence_length = 50

# Initialize Flask App
app = Flask(__name__)

# Define a route for the prediction function
@app.route('/predict', methods=['POST'])
def predict_behavior():
    try:
        # Get the data from the request
        data = request.json

        # Convert the data to a DataFrame
        df = pd.DataFrame(data)

        # Validate required columns
        if not all(col in df.columns for col in feature_columns):
            return jsonify({'error': 'Missing columns'}), 400

        # Scale the data
        df[feature_columns] = scaler.transform(df[feature_columns])

        # Create sequences
        sequences = []
        if len(df) >= sequence_length:
            # If enough data for full sequences
            for i in range(len(df) - sequence_length + 1):
                seq = df.iloc[i:i+sequence_length][feature_columns].values
                sequences.append(seq)
        else:
            # Pad the data if it's smaller than the sequence length
            padded_data = np.pad(
                df[feature_columns].values,
                ((sequence_length - len(df), 0), (0, 0)),  # Pad missing rows
                mode='constant',
                constant_values=0
            )
            sequences.append(padded_data)

        # Convert to NumPy array
        X_input = np.array(sequences)

        # Predicting the behavior
        predictions = model.predict(X_input)
        predicted_classes = np.argmax(predictions, axis=1)

        # Convert integers to class labels
        class_labels = label_encoder.inverse_transform(predicted_classes)

        # Calculate class frequencies
        unique_classes, counts = np.unique(class_labels, return_counts=True)
        max_count = np.max(counts)
        most_frequent_classes = unique_classes[counts == max_count]

        # Select the first class in case of ties
        most_frequent_class = most_frequent_classes[0]  # Select the first class alphabetically

        # Return the predicted class labels and the most frequent class
        return jsonify({
            "predicted_classes": list(class_labels),  # Full list of predictions
            "most_frequent_class": most_frequent_class
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
