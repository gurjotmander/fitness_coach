import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from cnn import create_model  # Import the model definition from cnn_model.pys

# Load and Prepare the Data
csv_file_path = 'Model/data/keypoints.csv'
df = pd.read_csv(csv_file_path)

# 0 for squating, 1 for standing
y = df['pose']

# Drop non-feature columns and convert the rest to a NumPy array
X = df.drop(['image_name', 'pose'], axis=1).to_numpy()

# Normalize keypoint values
X = X / np.max(X)

# One-hot encode the labels
y = to_categorical(y, num_classes=2)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Import and Create Model
model = create_model(X_train.shape[1])  # Pass input shape to the model function

# Train Model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save  Model
model.save('Model/data/cnn_squat_detection_model.h5')