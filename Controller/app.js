/**
 * This class uses model.json to count user squat reps in real-time while
 * drawing feedback on the camera for the user to see if the squat was performed.
 * 
 * This class uses worker.js to load the model in the background of the browser.
 */

var video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const canvasContext = canvas.getContext('2d');
const startButton = document.getElementById("button");
const repDisplay = document.getElementById("currentSet");
const setListDisplay = document.getElementById("listSets");
const timer = document.getElementById("timer");

let model;
let frameCount = 0;
let lastSquatDetectedTime = 0;
const squatCooldown = 2000;
let isProcessing = false;
let isDetecting = false;
let reps = 0;
let sets = 1;
let timerInterval;
let startTime;
let startedTimer = false;

tf.setBackend('webgl');

startButton.addEventListener("click", () => {
  if (!isDetecting) {
    reps = 0;
    updateRepDisplay();
    startSquatDetection();
  } else {
    currentReps();
    stopTimer();
    stopSquatDetection();
  }
  startButton.textContent = isDetecting ? "Start Workout" : "Stop Workout";
  isDetecting = !isDetecting;
});

const worker = new Worker('/Controller/worker.js');

worker.onmessage = function(event) {
  const { isSquatting, confidence } = event.data;
  console.log(`Received from worker: isSquatting=${isSquatting}, confidence=${confidence}`);
  if (isSquatting && (Date.now() - lastSquatDetectedTime) > squatCooldown) {
    reps++;
    updateRepDisplay();
  }
  drawFeedback(isSquatting, confidence);
};

function updateRepDisplay() {
  repDisplay.textContent = `Set ${sets}: ${reps} reps`;
}

function currentReps() {
  const timeElapsed = formatTime(Date.now() - startTime);
  const previousSet = document.createElement("div");
  previousSet.classList.add("previous-set");
  if (!startedTimer) {
    previousSet.textContent = `Set ${sets}: ${reps} Reps (Time: 00:00)`;
  } else {
    previousSet.textContent = `Set ${sets}: ${reps} Reps (Time: ${timeElapsed})`;
  }
  setListDisplay.appendChild(previousSet);
  
  sets++;
}

function startTimer() {
  startTime = Date.now();
  timerInterval = setInterval(() => {
    const timeElapsed = Date.now() - startTime;
    timer.textContent = `Time: ${formatTime(timeElapsed)}`;
  }, 1000);
}

function stopTimer() {
  clearInterval(timerInterval);
  timer.textContent = 'Time: 00:00';
  startedTimer = false;
}

function formatTime(time) {
  const totalSeconds = Math.floor(time / 1000);
  const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
  const seconds = String(totalSeconds % 60).padStart(2, '0');
  return `${minutes}:${seconds}`;
} 

async function cameraSetup() {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      video.play();
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      console.log("Video metadata loaded, playing video.");
      detectFrame(); // Start detection
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
    lastSquatDetectedTime = currentTime; // Update the time when squat was detected
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

  //Start timer when squat is performed
  if (!startedTimer && isSquatting) {
    startTimer();
    startedTimer = true;
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

async function startSquatDetection() {
  await reloadResources();
  await loadModel();
  await cameraSetup();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  detectFrame();
}

function stopSquatDetection() {
  console.log("Squat detection stopped");
  if (video.srcObject) {
      const tracks = video.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      video.srcObject = null;
  }
  stopTimer();
  startedTimer = false;
}
