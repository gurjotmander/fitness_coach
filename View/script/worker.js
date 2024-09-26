importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs');

console.log("Worker script loaded");

let model;

self.onmessage = async function(event) {
  if (event.data.modelData) {
    model = await tf.models.modelFromJSON(event.data.modelData);
    console.log("Worker: Model loaded successfully from main thread data");
  } else {
    const { imageData } = event.data;
    console.log('Worker received:', imageData);

    if (!model) {
      console.error("Worker: Model is not loaded yet.");
      return;
    }
    try {
      const input = tf.tensor(imageData, [1, 128, 128, 3]);
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