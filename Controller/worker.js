importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs');

console.log("Worker script loaded");

let model;

async function loadModel() {
  try {
    model = await tf.loadLayersModel('/Model/data/cnn_pose_detection_model/model.json');
    console.log("Worker: Model loaded successfully");
  } catch (error) {
    console.error("Worker: Error loading model:", error);
  }
}

self.onmessage = async function(event) {
  if (event.data.modelData) {
    try {
      console.log("Worker: Model loaded successfully from main thread data");
    } catch (error) {
      console.error("Worker: Error loading model from JSON:", error);
    }
  } else {
    const { imageData } = event.data;
    console.log('Worker received:', imageData);
    if (!model) {
      console.error("Worker: Model is not loaded yet.");
      return;
    }
    try {
      const input = tf.tensor(imageData, [1, 128, 128, 3]); // Ensure the shape matches the resized image
      console.log("Input tensor shape:", input.shape);

      const predictions = await model.predict(input).array();
      const squatConfidence = predictions[0][0];
      const isSquatting = squatConfidence > 0.5;

      console.log(`Worker: isSquatting=${isSquatting}, confidence=${squatConfidence}`);
      self.postMessage({ isSquatting, confidence: squatConfidence });

      tf.dispose(input);
    } catch (error) {
      console.error("Worker: Error during prediction:", error);
    }
  }
};

loadModel();