import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from cnn import create_model
#import tensorflowjs as tfjs

# Load the CSV file
csv_file_path = 'C:/Users/gurjo/Documents/term 8/major project/fitness app/Model/data/dataset.csv'
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define input shape for the CNN
input_shape = (128, 128, 3)

# Create and compile the model
model = create_model(input_shape)  # model from cnn.py
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Make prediction
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int).flatten()

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy: {accuracy}")

# Save the model
model.save('C:/Users/gurjo/Documents/term 8/major project/fitness app/Model/data/cnn_pose_detection_model.keras')
#tfjs.converters.save_keras_model(model, 'C:/Users/gurjo/Documents/term 8/major project/fitness app/Model/data/tfjs_model')