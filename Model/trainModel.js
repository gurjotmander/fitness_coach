import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import { createModel } from './cnn.js';

// Load the JSON file
const filePath = 'C:/Users/gurjo/Documents/term 8/major project/fitness app/Model/data/dataset.json';
const jsonData = JSON.parse(fs.readFileSync(filePath, 'utf8'));

// Load images and labels
async function loadData(jsonData, imgSize = [128, 128]) {
    const images = [];
    const labels = [];
    for (const entry of jsonData) {
        const imgPath = `C:/Users/gurjo/Documents/term 8/major project/fitness app/Model/data/images/person/${entry.image_name}`;
        const image = await tf.node.decodeImage(fs.readFileSync(imgPath), 3);
        const resizedImage = tf.image.resizeBilinear(image, imgSize);
        const normalizedImage = resizedImage.div(255.0);
        images.push(normalizedImage);
        labels.push(entry.label);
    }
    return [tf.stack(images), labels];
}

// Prepare data
const [X, y] = await loadData(jsonData);

// Encode labels
const labelEncoder = new Map();
const yEncoded = y.map(label => {
    if (!labelEncoder.has(label)) {
        labelEncoder.set(label, labelEncoder.size);
    }
    return labelEncoder.get(label);
});

// Split data
const splitIndex = Math.floor(X.shape[0] * 0.8);
const [X_train, X_test] = tf.split(X, [splitIndex, X.shape[0] - splitIndex]);
const y_train_encoded = yEncoded.slice(0, splitIndex);
const y_test_encoded = yEncoded.slice(splitIndex);
const y_train_tensor = tf.tensor1d(y_train_encoded, 'int32');
const y_test_tensor = tf.tensor1d(y_test_encoded, 'int32');

// Define input shape for the CNN
const inputShape = [128, 128, 3];

// Create and compile the model with Dropout for regularization
const model = createModel(inputShape);
model.compile({
    optimizer: 'adam',
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
});

// Define the early stopping callback without restoreBestWeights
const earlyStopping = tf.callbacks.earlyStopping({
    monitor: 'val_loss',
    patience: 10, 
    mode: 'min', 
});

// Training the model
await model.fit(X_train, y_train_tensor, {
    epochs: 30,
    batchSize: 32,
    validationData: [X_test, y_test_tensor],
    callbacks: [earlyStopping],
});

// Make prediction
const yPred = model.predict(X_test);
const yPredClasses = yPred.greater(0.5).toInt().dataSync();

// Evaluate the model
const accuracy = tf.metrics.sparseCategoricalAccuracy(y_test_tensor, yPredClasses).dataSync();
console.log(`Accuracy: ${accuracy}`);

// Save the model
await model.save('file://C:/Users/gurjo/Documents/term 8/major project/fitness app/Model/data/cnn_pose_detection_model');