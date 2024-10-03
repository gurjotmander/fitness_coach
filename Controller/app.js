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

const worker = new Worker('/Controller/worker.js');

tf.setBackend('webgl');

/**
 * This function activates the squat detection when the "Start Workout" button 
 * is clicked.
 */
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

/**
 * Listens for squat detection data from the Web Worker class and calls functions to 
 * update the html display.
 * @param {*} event data sent from Web Worker
 */
worker.onmessage = function(event) {
  const { isSquatting, confidence } = event.data;
  console.log(`Received from worker: isSquatting=${isSquatting}, confidence=${confidence}`);
  if (isSquatting && (Date.now() - lastSquatDetectedTime) > squatCooldown) {
    reps++;
    updateRepDisplay();
  }
  drawFeedback(isSquatting, confidence);
};

/**
 * Displays the updated number of sets and reps completed by the user.
 */
function updateRepDisplay() {
  repDisplay.textContent = `Set ${sets}: ${reps} reps`;
}

/**
 * Displays the completed set and number of reps in a list beneath the previously 
 * completed set and adds it to the DOM.
 */
function currentReps() {
  const timeElapsed = formatTime(Date.now() - startTime);
  const previousSet = document.createElement("div");
  previousSet.classList.add("previous-set");

  //Format time for completed set
  previousSet.textContent = startedTimer? `Set ${sets}: ${reps} Reps (Time: ${timeElapsed})` : 
  `Set ${sets}: ${reps} Reps (Time: 00:00)`;
  setListDisplay.appendChild(previousSet);

  //Update set for next set
  sets++;
}

/**
 * Starts the timer and displays it in the html.
 */
function startTimer() {
  startTime = Date.now();
  timerInterval = setInterval(() => {
    const timeElapsed = Date.now() - startTime;
    timer.textContent = `Time: ${formatTime(timeElapsed)}`;
  }, 1000);
}

/**
 * Stops and clears the timer.
 */
function stopTimer() {
  clearInterval(timerInterval);
  timer.textContent = 'Time: 00:00';
  startedTimer = false;
}

/**
 * Formats the time by converting it to seconds and Stringifies the extracted minutes 
 * and seconds.
 * @param {*} time 
 * @returns String
 */
function formatTime(time) {
  const totalSeconds = Math.floor(time / 1000);
  const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
  const seconds = String(totalSeconds % 60).padStart(2, '0');
  return `${minutes}:${seconds}`;
} 

/**
 * Gets the device camera started and displays the video feed on the home page.
 */
async function cameraSetup() {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      video.play();
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      console.log("Video metadata loaded, playing video.");
      detectFrame(); // start detection
    };
  }
}

/**
 * Stops the previous video stream and frees camera resources.
 */
async function reloadResources() {
  if (video.srcObject) {
    const tracks = video.srcObject.getTracks();
    tracks.forEach(track => track.stop());
    video.srcObject = null; // deactivates video stream
  }
}

/**
 * Loads the custom CNN Model.
 */
async function loadModel() {
  console.log("Loading model...");
  try {
    model = await tf.loadLayersModel('/Model/data/cnn_pose_detection_model/model.json');
    console.log("Model loaded successfully");
    
  } catch (error) {
    console.error("Error loading model:", error);
  }
}

/**
 * Displays feedback to user regarding squat detection to let the user know if the squat was detected.
 * @param {*} isSquatting Boolean
 */
function drawFeedback(isSquatting) {
  const currentTime = Date.now();
  if (isSquatting && (currentTime - lastSquatDetectedTime) > squatCooldown) {
    lastSquatDetectedTime = currentTime; // Update the time when squat was detected
  }

  // Clear canvas between feedback
  canvasContext.clearRect(0, 0, canvas.width, canvas.height);

  canvasContext.globalAlpha = 0.3;
  canvasContext.fillStyle = isSquatting ? 'green' : 'red';
  canvasContext.fillRect(0, 0, canvas.width, canvas.height);

  const wordList = ["Good Job", "Awesome", "Keep Going", "You're Doing Great!", "Look at you go!", 
    "Amazing", "Don't Stop now!", "One rep at a time", "A healthier you will thank you", "Dont give up",
    "You've got this!", "Stay determined", "Keep working hard"];

  canvasContext.globalAlpha = 1.0;
  canvasContext.font = '30px Arial';
  canvasContext.fillStyle = 'white';
  canvasContext.textAlign = 'center';
  //Display encourgement words when squat is performed.
  if (isSquatting) {
    const randomWord = Math.floor(Math.random() * wordList.length);
    const feedbackText = `${wordList[randomWord]}`;
    canvasContext.fillText(feedbackText, canvas.width / 2, canvas.height / 2);
  }

  //Start timer when squat is performed
  if (!startedTimer && isSquatting) {
    startTimer();
    startedTimer = true;
  }
}

/**
 * Processes every 5th frame and convert it to a tensor that is suitable for the model and then
 * normalizes pixel values and adds a batch dimension to make tensor shape which matches models 
 * expected input. Data is then sent to Worker.
 * @returns 
 */
async function detectFrame() {
  try {
    // Check if video is ready
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

        tf.dispose([input, resized, normalized]); // dispose tensor to free up space
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

  //schedule request next frame
  requestAnimationFrame(detectFrame); // process runs in loop
}

/**
 * Begins the squat detection. First this function frees resources, 
 * loads the CNN model, sets up the camera, and then begins the frame detection.
 */
async function startSquatDetection() {
  await reloadResources();
  await loadModel();
  await cameraSetup();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  detectFrame();
}

/**
 * Stops the squat detection. Frees camera resources and then stops the timer.
 */
function stopSquatDetection() {
  console.log("Squat detection stopped");
  reloadResources();
  stopTimer();
  startedTimer = false;
}
