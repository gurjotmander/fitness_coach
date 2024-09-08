import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout

def create_model(input_shape):
    model = Sequential([
        Dense(128, input_shape=(input_shape,), activation='relu'),  # Input layer
        Dropout(0.3),  # Dropout layer for regularization
        Dense(64, activation='relu'),  # Hidden layer
        Dropout(0.3),  # Another dropout layer
        Dense(2, activation='softmax')  # Output layer with 2 classes ('standing' and 'squatting')
    ])

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model