var video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const canvasContext = canvas.getContext('2d');

let model;
let frameCount = 0;
let lastSquatDetectedTime = 0;
const squatCooldown = 2000;
let isProcessing = false;

tf.setBackend('webgl');

const worker = new Worker('/Controller/worker.js');

worker.onmessage = function(event) {
  const { isSquatting, confidence } = event.data;
  console.log(`Received from worker: isSquatting=${isSquatting}, confidence=${confidence}`);
  drawFeedback(isSquatting, confidence);
};

async function cameraSetup() {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      video.play();
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      console.log("Video metadata loaded, playing video.");
      detectFrame(); // Start the detection process
    };
  }
}


async function reloadResources() {
  // Stop previous video stream
  if (video.srcObject) {
    const tracks = video.srcObject.getTracks();
    tracks.forEach(track => track.stop());
    video.srcObject = null;
  }
}

async function loadModel() {
  console.log("Loading model...");
  try {
    model = await tf.loadLayersModel('/Model/data/cnn_pose_detection_model/model.json');
    console.log("Model loaded successfully");
    
  } catch (error) {
    console.error("Error loading model:", error);
  }
}

function drawFeedback(isSquatting, confidence) {
  const currentTime = Date.now();
  if (isSquatting && (currentTime - lastSquatDetectedTime) > squatCooldown) {
    lastSquatDetectedTime = currentTime; // Update the time when the squat was detected
  }
  canvasContext.clearRect(0, 0, canvas.width, canvas.height);

  canvasContext.globalAlpha = 0.3;
  canvasContext.fillStyle = isSquatting ? 'green' : 'red';
  canvasContext.fillRect(0, 0, canvas.width, canvas.height);

  canvasContext.globalAlpha = 1.0;
  canvasContext.font = '30px Arial';
  canvasContext.fillStyle = 'white';
  canvasContext.textAlign = 'center';
  const feedbackText = isSquatting ? `Squat Detected (Confidence: ${Math.round(confidence * 100)}%)` : `No Squat (Confidence: ${Math.round(confidence * 100)}%)`;
  canvasContext.fillText(feedbackText, canvas.width / 2, canvas.height / 2);

  if (isSquatting) {
    canvasContext.strokeStyle = 'blue';
    canvasContext.lineWidth = 5;
    canvasContext.strokeRect(50, 50, canvas.width - 100, canvas.height - 100);
  }
}


async function detectFrame() {
  try {
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      requestAnimationFrame(detectFrame);
      return;
    }
    if (frameCount % 5 === 0 && !isProcessing) {
      isProcessing = true;
      const input = tf.browser.fromPixels(video);
      const resized = tf.image.resizeBilinear(input, [128, 128]);
      const normalized = resized.div(255.0).expandDims(0);

      if (model) {
        const imageData = await normalized.data();
        console.log('Sending to worker:', imageData);
        worker.postMessage({ imageData });

        tf.dispose([input, resized, normalized]);
      } else {
        console.log("Model is not loaded.");
      }

      isProcessing = false;
    }
    frameCount++;
  } catch (error) {
    console.error("Error during prediction:", error);
    await reloadResources();
  }

  requestAnimationFrame(detectFrame);
}

async function detectPose() {
  await reloadResources();
  await loadModel();
  await cameraSetup();

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  detectFrame();
}

detectPose();
