import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from cnn import create_model

# Load the CSV file
csv_file_path = 'C:/Users/gurjo/Documents/term 8/major project/fitness app/Model/data/keypoints.csv'
df = pd.read_csv(csv_file_path)

# Load images and labels
def load_data(df, img_size=(128, 128)):
    images = []
    labels = []
    for index, row in df.iterrows():
        img_path = f'C:/Users/gurjo/Documents/term 8/major project/fitness app/Model/data/images/person/{row["image_name"]}'
        image = load_img(img_path, target_size=img_size)
        image = img_to_array(image) / 255.0  # Normalize the image
        images.append(image)
        labels.append(row['label'])
    
    return np.array(images), np.array(labels)

# Prepare data
X, y = load_data(df)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert labels to numeric values

# Check if there are exactly 2 unique labels
num_classes = len(np.unique(y_encoded))
if num_classes != 2:
    raise ValueError(f"Expected 2 classes for binary classification, but found {num_classes} classes.")

# One-hot encode the labels
y_encoded = to_categorical(y_encoded, num_classes=num_classes)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define input shape for the CNN
input_shape = (128, 128, 3)

# Create and compile the model
model = create_model(input_shape)  # Use the model from cnn.py
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model
model.save('C:/Users/gurjo/Documents/term 8/major project/fitness app/Model/data/cnn_pose_detection_model.h5')