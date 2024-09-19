// This class defines the CNN model

import * as tf from '@tensorflow/tfjs';

export function createModel(inputShape) {
    const model = tf.sequential();

    // First convolutional layer
    model.add(tf.layers.conv2d({
        inputShape: inputShape,
        filters: 32,
        kernelSize: 3,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }) // Add L2 regularization
    }));

    // Add L2 regularization to all convolutional and dense layers
    model.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
    }));
    
    model.add(tf.layers.flatten());

    // Dense layer with L2 regularization
    model.add(tf.layers.dense({
        units: 128,
        activation: 'relu',
        kernelRegularizer: tf.regularizers.l2({ l2: 0.01 })
    }));

    // Output layer
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));

    return model;
}