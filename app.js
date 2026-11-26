
import {
  FilesetResolver,
  FaceLandmarker,
  HandLandmarker,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const video = document.querySelector("#video");
const canvas = document.querySelector("#overlay");
const ctx = canvas.getContext("2d");

const startBtn = document.querySelector("#startBtn");
const stopBtn = document.querySelector("#stopBtn");
const flipBtn = document.querySelector("#flipBtn");

const statusEl = document.querySelector("#status");
const placeholder = document.querySelector("#placeholder");
const livePill = document.querySelector("#livePill");

const faceCountEl = document.querySelector("#faceCount");
const handCountEl = document.querySelector("#handCount");
const gestureResults = document.querySelector("#gestureResults");

let faceLandmarker, handLandmarker;
let stream, running = false;
let facingMode = "user";
let lastVideoTime = -1;
let animationId;

async function loadModels() {
  statusEl.textContent = "Loading AI models...";

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  const base = "https://storage.googleapis.com/mediapipe-models";

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `${base}/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
      delegate: "CPU"
    },
    runningMode: "VIDEO",
    numFaces: 5
  });

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: `${base}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
      delegate: "CPU"
    },
    runningMode: "VIDEO",
    numHands: 2
  });
}

async function startCamera() {
  try {
    startBtn.disabled = true;

    if (!faceLandmarker || !handLandmarker) {
      await loadModels();
    }

    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode,
        width: { ideal: 1280 },
        height: { ideal: 960 }
      }
    });

    video.srcObject = stream;
    await video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    running = true;
    lastVideoTime = -1;

    placeholder.hidden = true;
    livePill.classList.add("show");

    stopBtn.disabled = false;
    flipBtn.disabled = false;

    statusEl.textContent = "Camera is running!";
    predict();

  } catch (error) {
    console.error(error);
    startBtn.disabled = false;
    statusEl.textContent = "Camera error: " + error.message;
  }
}

function countFingers(points, handedness) {
  let count = 0;
  const side = handedness?.[0]?.categoryName || "";

  if (
    side === "Left"
      ? points[4].x > points[3].x
      : points[4].x < points[3].x
  ) count++;

  for (const [tip, pip] of [
    [8, 6], [12, 10], [16, 14], [20, 18]
  ]) {
    if (points[tip].y < points[pip].y) count++;
  }

  return count;
}

function gestureName(n) {
  return [
    "Closed Hand / Fist",
    "One Finger",
    "Two Fingers",
    "Three Fingers",
    "Four Fingers",
    "Open Palm"
  ][n];
}

function predict() {
  if (!running) return;

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    const now = performance.now();
    const faces = faceLandmarker.detectForVideo(video, now);
    const hands = handLandmarker.detectForVideo(video, now);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const points of faces.faceLandmarks) {
      const xs = points.map(p => p.x * canvas.width);
      const ys = points.map(p => p.y * canvas.height);

      const x = Math.min(...xs);
      const y = Math.min(...ys);
      const w = Math.max(...xs) - x;
      const h = Math.max(...ys) - y;

      ctx.strokeStyle = "#34d399";
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, w, h);

      ctx.fillStyle = "#34d399";
      ctx.font = "bold 20px Arial";
      ctx.fillText("Face", x, Math.max(22, y - 8));
    }

    const drawing = new DrawingUtils(ctx);
    const rows = [];

    hands.landmarks.forEach((points, i) => {
      drawing.drawConnectors(
        points,
        HandLandmarker.HAND_CONNECTIONS,
        { color: "#fbbf24", lineWidth: 4 }
      );

      drawing.drawLandmarks(
        points,
        { color: "#fb7185", radius: 3 }
      );

      const n = countFingers(points, hands.handedness[i]);
      rows.push(`${gestureName(n)} — ${n}/5 fingers`);
    });

    faceCountEl.textContent = faces.faceLandmarks.length;
    handCountEl.textContent = hands.landmarks.length;

    gestureResults.innerHTML = rows.length
      ? rows.map(r => `<div class="gesture-row">${r}</div>`).join("")
      : '<p class="empty">No hands detected.</p>';
  }

  animationId = requestAnimationFrame(predict);
}

function stopCamera() {
  running = false;
  cancelAnimationFrame(animationId);

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
  }

  stream = null;
  video.srcObject = null;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  placeholder.hidden = false;
  livePill.classList.remove("show");

  faceCountEl.textContent = "0";
  handCountEl.textContent = "0";

  gestureResults.innerHTML =
    '<p class="empty">No hands detected.</p>';

  startBtn.disabled = false;
  stopBtn.disabled = true;
  flipBtn.disabled = true;

  statusEl.textContent = "Camera stopped.";
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);

flipBtn.addEventListener("click", async () => {
  facingMode = facingMode === "user" ? "environment" : "user";
  stopCamera();
  await startCamera();
});

window.addEventListener("pagehide", stopCamera);
        
