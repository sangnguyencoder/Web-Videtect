// Global variables
let currentTab = "upload";
let webcamStream = null;
let webcamInterval = null;
let isWebcamActive = false;
let analysisStartTime = null;
let uploadInProgress = false;

// DOM elements
const statusIndicator = document.getElementById("statusIndicator");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

// Initialize app
document.addEventListener("DOMContentLoaded", () => {
  initializeApp();
  setupEventListeners();
  checkSystemStatus();
  updateUptime();
});

// Initialize app
function initializeApp() {
  console.log("🚀 Initializing Violence Detection AI System...");
  checkModelStatus();
  loadConfiguration();
  analysisStartTime = new Date();
  initializeSmoothScrolling();
}

// Setup event listeners
function setupEventListeners() {
  // Tab navigation
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      switchTab(tab);
    });
  });

  // File upload with improved error handling
  const videoInput = document.getElementById("videoInput");
  const uploadArea = document.getElementById("uploadArea");

  if (videoInput) {
    videoInput.addEventListener("change", handleFileSelect);
  }

  if (uploadArea) {
    // Drag and drop with better feedback
    uploadArea.addEventListener("dragover", handleDragOver);
    uploadArea.addEventListener("dragleave", handleDragLeave);
    uploadArea.addEventListener("drop", handleDrop);

    // Remove the click event listener from uploadArea to prevent double trigger
    // uploadArea.addEventListener("click", (e) => {
    //   if (!uploadInProgress && videoInput) {
    //     videoInput.click()
    //   }
    // })

    // Keyboard accessibility
    uploadArea.addEventListener("keydown", (e) => {
      if ((e.key === "Enter" || e.key === " ") && !uploadInProgress) {
        e.preventDefault();
        if (videoInput) videoInput.click();
      }
    });
  }

  // Upload button event listener
  const uploadBtn = document.getElementById("uploadBtn");
  if (uploadBtn) {
    uploadBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!uploadInProgress && videoInput) {
        videoInput.click();
      }
    });
  }

  // Webcam controls - IMPORTANT: Setup webcam event listeners here
  const startWebcamBtn = document.getElementById("startWebcam");
  const stopWebcamBtn = document.getElementById("stopWebcam");

  if (startWebcamBtn) {
    startWebcamBtn.addEventListener("click", startWebcam);
  }

  if (stopWebcamBtn) {
    stopWebcamBtn.addEventListener("click", stopWebcam);
  }

  // Configuration
  const saveConfigBtn = document.getElementById("saveConfig");
  if (saveConfigBtn) {
    saveConfigBtn.addEventListener("click", saveConfiguration);
  }
}

// Initialize smooth scrolling
function initializeSmoothScrolling() {
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute("href"));
      if (target) {
        target.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }
    });
  });
}

// Smooth scrolling for navigation
function scrollToDemo() {
  document.getElementById("demo").scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
}

// Tab switching
function switchTab(tabName) {
  if (uploadInProgress && tabName !== currentTab) {
    showNotification(
      "Please wait for the current upload to complete",
      "warning"
    );
    return;
  }

  // Update tab buttons
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.remove("active");
    btn.setAttribute("aria-selected", "false");
  });

  const activeBtn = document.querySelector(`[data-tab="${tabName}"]`);
  if (activeBtn) {
    activeBtn.classList.add("active");
    activeBtn.setAttribute("aria-selected", "true");
  }

  // Update tab content
  document.querySelectorAll(".tab-content").forEach((content) => {
    content.classList.remove("active");
  });

  const activeContent = document.getElementById(`${tabName}-tab`);
  if (activeContent) {
    activeContent.classList.add("active");
  }

  currentTab = tabName;

  // Handle tab-specific actions
  if (tabName !== "webcam" && isWebcamActive) {
    stopWebcamAnalysis();
  }

  console.log(`📊 Tab switched to: ${tabName}`);
}

// System status check
async function checkSystemStatus() {
  try {
    const response = await fetch("/status");
    const data = await response.json();

    if (data.model_loaded) {
      updateStatusIndicator(true, "Model Ready");
      updateSystemInfo(data);
    } else {
      updateStatusIndicator(false, "Model Loading");
      setTimeout(checkSystemStatus, 3000);
    }
  } catch (error) {
    console.error("❌ Error checking system status:", error);
    updateStatusIndicator(false, "Connection Error");
    setTimeout(checkSystemStatus, 5000);
  }
}

// Model status check
async function checkModelStatus() {
  try {
    const response = await fetch("/status");
    const data = await response.json();

    if (data.model_loaded) {
      updateStatusIndicator(true, "Model Ready");
      updateSystemInfo(data);
    } else {
      updateStatusIndicator(false, "Model Loading");
      setTimeout(checkModelStatus, 3000);
    }
  } catch (error) {
    console.error("❌ Error checking model status:", error);
    updateStatusIndicator(false, "Connection Error");
    setTimeout(checkModelStatus, 5000);
  }
}

// Update status indicator
function updateStatusIndicator(connected, text) {
  if (statusText) {
    statusText.textContent = text;
  }

  const statusIndicator = document.getElementById("statusIndicator");
  if (statusIndicator) {
    if (connected) {
      statusIndicator.style.background = "#f0fff4";
      statusIndicator.style.borderColor = "#9ae6b4";
      statusIndicator.style.color = "#2f855a";
    } else {
      statusIndicator.style.background = "#fef5e7";
      statusIndicator.style.borderColor = "#f6ad55";
      statusIndicator.style.color = "#c05621";
    }
  }
}

// Update system information
function updateSystemInfo(data) {
  const elements = {
    modelStatus: data.model_loaded ? "✅ Loaded Successfully" : "❌ Not Loaded",
    modelPath: data.model_path || "-",
    framesPerClip: data.frames_per_clip || "-",
    imgSize: data.img_size ? `${data.img_size}x${data.img_size}px` : "-",
  };

  Object.entries(elements).forEach(([id, value]) => {
    const element = document.getElementById(id);
    if (element) {
      element.textContent = value;
    }
  });
}

// Update uptime
function updateUptime() {
  const uptimeElement = document.getElementById("uptime");
  if (uptimeElement && analysisStartTime) {
    const now = new Date();
    const diff = now - analysisStartTime;
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    const seconds = Math.floor((diff % (1000 * 60)) / 1000);

    uptimeElement.textContent = `${hours.toString().padStart(2, "0")}:${minutes
      .toString()
      .padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
  }

  setTimeout(updateUptime, 1000);
}

// File handling
function handleFileSelect(event) {
  event.preventDefault();
  event.stopPropagation();

  const file = event.target.files[0];
  if (file && !uploadInProgress) {
    // Clear the input value to allow selecting the same file again if needed
    setTimeout(() => {
      event.target.value = "";
    }, 100);

    if (validateFile(file)) {
      processVideoFile(file);
    }
  }
}

function handleDragOver(event) {
  event.preventDefault();
  if (!uploadInProgress) {
    event.currentTarget.classList.add("dragover");
  }
}

function handleDragLeave(event) {
  event.currentTarget.classList.remove("dragover");
}

function handleDrop(event) {
  event.preventDefault();
  event.currentTarget.classList.remove("dragover");

  if (uploadInProgress) {
    showNotification("Upload already in progress", "warning");
    return;
  }

  const files = event.dataTransfer.files;
  if (files.length > 0) {
    const file = files[0];
    if (validateFile(file)) {
      processVideoFile(file);
    }
  }
}

// Validate file
function validateFile(file) {
  const allowedTypes = [
    "video/mp4",
    "video/avi",
    "video/mov",
    "video/mkv",
    "video/webm",
    "video/x-flv",
  ];
  const allowedExtensions = /\.(mp4|avi|mov|mkv|webm|flv)$/i;
  const maxSize = 100 * 1024 * 1024; // 100MB

  if (!allowedTypes.includes(file.type) && !allowedExtensions.test(file.name)) {
    showErrorModal(
      "Unsupported file format. Please select MP4, AVI, MOV, MKV, WEBM or FLV files.",
      false
    );
    return false;
  }

  if (file.size > maxSize) {
    showErrorModal(
      `File too large (${formatFileSize(
        file.size
      )}). Please select a file smaller than 100MB.`,
      false
    );
    return false;
  }

  if (file.size === 0) {
    showErrorModal(
      "The selected file appears to be empty. Please choose a valid video file.",
      false
    );
    return false;
  }

  return true;
}

// Process video file
async function processVideoFile(file) {
  if (uploadInProgress) {
    showNotification("Upload already in progress", "warning");
    return;
  }

  try {
    uploadInProgress = true;
    showProcessingState(true);
    showVideoPreview(file);
    showAnalysisSection();

    const formData = new FormData();
    formData.append("video", file);

    const response = await uploadWithProgress(formData);
    const result = await response.json();

    if (response.ok) {
      showResult(result);
      showNotification("Video analysis completed successfully!", "success");
    } else {
      throw new Error(result.error || `Server error: ${response.status}`);
    }
  } catch (error) {
    console.error("❌ Error processing video:", error);
    hideAnalysisSection();

    let errorMessage = "Failed to process video. ";
    if (error.message.includes("Failed to fetch")) {
      errorMessage += "Please check your internet connection and try again.";
    } else if (error.message.includes("413")) {
      errorMessage += "File too large. Please select a smaller file.";
    } else {
      errorMessage += error.message || "Unknown error occurred.";
    }

    showErrorModal(errorMessage, true);
  } finally {
    uploadInProgress = false;
    showProcessingState(false);
  }
}

// Upload with progress tracking
async function uploadWithProgress(formData) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) {
        const percentComplete = (e.loaded / e.total) * 100;
        updateUploadProgress(percentComplete);
      }
    });

    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve({
          ok: true,
          status: xhr.status,
          json: () => Promise.resolve(JSON.parse(xhr.responseText)),
        });
      } else {
        reject(new Error(`HTTP ${xhr.status}: ${xhr.statusText}`));
      }
    });

    xhr.addEventListener("error", () => {
      reject(new Error("Network error occurred"));
    });

    xhr.addEventListener("timeout", () => {
      reject(new Error("Upload timeout"));
    });

    xhr.open("POST", "/upload");
    xhr.timeout = 300000; // 5 minutes timeout
    xhr.send(formData);
  });
}

// Update upload progress
function updateUploadProgress(percent) {
  const progressFill = document.getElementById("progressFill");
  const progressText = document.getElementById("progressText");

  if (progressFill) {
    progressFill.style.width = `${percent}%`;
  }

  if (progressText) {
    if (percent < 100) {
      progressText.textContent = `Uploading... ${Math.round(percent)}%`;
    } else {
      progressText.textContent = "Processing video...";
    }
  }
}

function showProcessingState(show) {
  const uploadArea = document.getElementById("uploadArea");
  const processingOverlay = document.getElementById("processingOverlay");
  const uploadBtn = document.getElementById("uploadBtn");

  if (uploadArea) {
    uploadArea.classList.toggle("processing", show);
    // Disable pointer events when processing
    uploadArea.style.pointerEvents = show ? "none" : "auto";
  }

  if (processingOverlay) {
    processingOverlay.classList.toggle("active", show);
  }

  if (uploadBtn) {
    uploadBtn.disabled = show;
    if (show) {
      uploadBtn.innerHTML =
        '<i class="fas fa-spinner fa-spin"></i>Processing...';
    } else {
      uploadBtn.innerHTML = '<i class="fas fa-folder-open"></i>Browse Files';
    }
  }
}

// Show video preview
function showVideoPreview(file) {
  const videoPreview = document.getElementById("videoPreview");
  const previewVideo = document.getElementById("previewVideo");
  const videoInfoGrid = document.getElementById("videoInfoGrid");
  const uploadArea = document.getElementById("uploadArea");

  if (!videoPreview || !previewVideo || !videoInfoGrid || !uploadArea) return;

  const url = URL.createObjectURL(file);
  previewVideo.src = url;

  videoInfoGrid.innerHTML = `
    <div class="video-info-item">
        <div class="video-info-label">📁 File Name</div>
        <div class="video-info-value">${file.name}</div>
    </div>
    <div class="video-info-item">
        <div class="video-info-label">📊 File Size</div>
        <div class="video-info-value">${formatFileSize(file.size)}</div>
    </div>
    <div class="video-info-item">
        <div class="video-info-label">🎬 File Type</div>
        <div class="video-info-value">${file.type || "Video"}</div>
    </div>
    <div class="video-info-item">
        <div class="video-info-label">⏰ Upload Time</div>
        <div class="video-info-value">${new Date().toLocaleTimeString()}</div>
    </div>
  `;

  videoPreview.style.display = "block";
  uploadArea.style.display = "none";

  previewVideo.addEventListener(
    "loadeddata",
    () => {
      URL.revokeObjectURL(url);
      const duration = previewVideo.duration;
      const videoWidth = previewVideo.videoWidth;
      const videoHeight = previewVideo.videoHeight;

      if (duration && videoWidth && videoHeight) {
        videoInfoGrid.innerHTML += `
        <div class="video-info-item">
            <div class="video-info-label">⏱️ Duration</div>
            <div class="video-info-value">${formatDuration(duration)}</div>
        </div>
        <div class="video-info-item">
            <div class="video-info-label">📐 Resolution</div>
            <div class="video-info-value">${videoWidth}x${videoHeight}</div>
        </div>
      `;
      }
    },
    { once: true }
  );
}

// Show analysis section
function showAnalysisSection() {
  const analysisSection = document.getElementById("analysisSection");
  if (!analysisSection) return;

  analysisSection.style.display = "block";

  const progressFill = document.getElementById("progressFill");
  const progressText = document.getElementById("progressText");
  const progressBar = document.querySelector(".progress-bar");

  if (!progressFill || !progressText || !progressBar) return;

  let progress = 0;
  const messages = [
    "Uploading video to server...",
    "Extracting video frames...",
    "Preprocessing data...",
    "Running AI analysis...",
    "Processing results...",
    "Analysis complete!",
  ];
  let messageIndex = 0;

  const interval = setInterval(() => {
    progress += Math.random() * 8 + 2;
    if (progress >= 95) {
      progress = 95;
    }

    progressFill.style.width = progress + "%";
    progressBar.setAttribute("aria-valuenow", Math.round(progress));

    const newMessageIndex = Math.min(
      Math.floor(progress / 20),
      messages.length - 2
    );
    if (newMessageIndex !== messageIndex) {
      messageIndex = newMessageIndex;
      progressText.textContent = messages[messageIndex];
    }
  }, 200);

  analysisSection.dataset.interval = interval;
}

// Hide analysis section
function hideAnalysisSection() {
  const analysisSection = document.getElementById("analysisSection");
  if (analysisSection) {
    const interval = analysisSection.dataset.interval;
    if (interval) {
      clearInterval(Number.parseInt(interval));
    }

    analysisSection.style.display = "none";

    const progressFill = document.getElementById("progressFill");
    if (progressFill) {
      progressFill.style.width = "0%";
    }
  }
}

// Show result
function showResult(result) {
  const resultSection = document.getElementById("resultSection");
  const analysisSection = document.getElementById("analysisSection");

  if (!resultSection) return;

  const progressFill = document.getElementById("progressFill");
  const progressText = document.getElementById("progressText");
  if (progressFill && progressText) {
    progressFill.style.width = "100%";
    progressText.textContent = "Analysis complete!";

    setTimeout(() => {
      if (analysisSection) {
        analysisSection.style.display = "none";
      }
    }, 1000);
  }

  const resultTimestamp = document.getElementById("resultTimestamp");
  if (resultTimestamp) {
    resultTimestamp.textContent = new Date(result.timestamp).toLocaleString();
  }

  const predictionLabel = document.getElementById("predictionLabel");
  const labelText = document.getElementById("labelText");
  const confidenceText = document.getElementById("confidenceText");

  if (labelText && confidenceText) {
    const isViolence = result.label === "Violence";
    labelText.innerHTML = `
      ${isViolence ? "⚠️" : "✅"} 
      ${isViolence ? "Violence Detected" : "No Violence Detected"}
    `;
    confidenceText.textContent = `${result.confidence}%`;
  }

  if (predictionLabel) {
    predictionLabel.className =
      "prediction-label " +
      (result.label === "Violence" ? "violence" : "non-violence");
    predictionLabel.setAttribute("aria-live", "polite");
  }

  const confidenceFill = document.getElementById("confidenceFill");
  if (confidenceFill) {
    confidenceFill.style.width = result.confidence + "%";
    confidenceFill.className =
      "confidence-fill " +
      (result.label === "Violence" ? "violence" : "non-violence");
  }

  if (result.video_info) {
    const videoDetails = document.getElementById("videoDetails");
    if (videoDetails) {
      videoDetails.innerHTML = `
        <div><strong>📊 FPS</strong><span>${result.video_info.fps}</span></div>
        <div><strong>⏱️ Duration</strong><span>${result.video_info.duration}s</span></div>
        <div><strong>📐 Resolution</strong><span>${result.video_info.resolution}</span></div>
        <div><strong>🎬 Frames</strong><span>${result.video_info.frame_count}</span></div>
        <div><strong>🎯 Confidence</strong><span>${result.confidence}%</span></div>
        <div><strong>⚡ Model</strong><span>3D-CNN + LSTM</span></div>
      `;
    }
  }

  resultSection.style.display = "block";
  resultSection.scrollIntoView({ behavior: "smooth", block: "center" });
}

// Reset upload
function resetUpload() {
  if (uploadInProgress) {
    if (!confirm("Upload is in progress. Are you sure you want to reset?")) {
      return;
    }
  }

  const sections = ["videoPreview", "analysisSection", "resultSection"];
  sections.forEach((id) => {
    const element = document.getElementById(id);
    if (element) {
      element.style.display = "none";
    }
  });

  const uploadArea = document.getElementById("uploadArea");
  if (uploadArea) {
    uploadArea.style.display = "block";
  }

  const videoInput = document.getElementById("videoInput");
  if (videoInput) {
    videoInput.value = "";
  }

  uploadInProgress = false;
  showProcessingState(false);
  showNotification("Upload reset successfully", "success");
}

// WEBCAM FUNCTIONS - Fixed implementation
function startWebcam() {
  const startBtn = document.getElementById("startWebcam");
  const stopBtn = document.getElementById("stopWebcam");
  const webcamVideo = document.getElementById("webcamVideo");
  const webcamOverlay = document.getElementById("webcamOverlay");

  if (!startBtn || !stopBtn || !webcamVideo || !webcamOverlay) {
    console.error("❌ Webcam elements not found");
    return;
  }

  console.log("🎬 Starting webcam...");

  // Show loading state
  startBtn.innerHTML =
    '<i class="fas fa-spinner fa-spin"></i><span>Starting...</span>';
  startBtn.disabled = true;

  navigator.mediaDevices
    .getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user",
      },
    })
    .then((stream) => {
      console.log("✅ Camera access granted");
      webcamVideo.srcObject = stream;
      webcamStream = stream;
      webcamOverlay.classList.add("hidden");
      startBtn.style.display = "none";
      stopBtn.style.display = "inline-block";
      startBtn.disabled = false;
      startBtn.innerHTML =
        '<i class="fas fa-play"></i><span>Start Camera</span>';

      // Start analysis after video is ready
      webcamVideo.addEventListener(
        "loadedmetadata",
        () => {
          console.log("📹 Video metadata loaded, starting analysis...");
          startWebcamAnalysis();
        },
        { once: true }
      );

      showNotification("Camera started successfully", "success");
    })
    .catch((err) => {
      console.error("❌ Camera error:", err);
      startBtn.disabled = false;
      startBtn.innerHTML =
        '<i class="fas fa-play"></i><span>Start Camera</span>';

      let errorMessage = "Unable to access camera: ";

      switch (err.name) {
        case "NotAllowedError":
        case "PermissionDeniedError":
          errorMessage +=
            "Permission denied. Please allow camera access and try again.";
          break;
        case "NotFoundError":
          errorMessage += "No camera found on your device.";
          break;
        case "NotReadableError":
          errorMessage += "Camera is being used by another application.";
          break;
        case "OverconstrainedError":
          errorMessage += "Camera does not support the required settings.";
          break;
        case "SecurityError":
          errorMessage +=
            "Camera access is not allowed on this page. Please use HTTPS.";
          break;
        default:
          errorMessage += err.message || "Unknown error occurred.";
      }

      alert(errorMessage);
    });
}

function stopWebcam() {
  console.log("🛑 Stopping webcam...");

  const startBtn = document.getElementById("startWebcam");
  const stopBtn = document.getElementById("stopWebcam");
  const webcamVideo = document.getElementById("webcamVideo");
  const webcamOverlay = document.getElementById("webcamOverlay");
  const webcamResult = document.getElementById("webcamResult");

  if (webcamStream) {
    webcamStream.getTracks().forEach((track) => track.stop());
    webcamStream = null;
  }

  if (webcamVideo) webcamVideo.srcObject = null;
  if (webcamOverlay) webcamOverlay.classList.remove("hidden");
  if (startBtn) startBtn.style.display = "inline-block";
  if (stopBtn) stopBtn.style.display = "none";
  if (webcamResult) webcamResult.style.display = "none";

  stopWebcamAnalysis();
  showNotification("Camera stopped", "success");
}

function startWebcamAnalysis() {
  const video = document.getElementById("webcamVideo");
  const canvas = document.getElementById("webcamCanvas");

  if (!video || !canvas) {
    console.error("❌ Video or canvas element not found");
    return;
  }

  const ctx = canvas.getContext("2d");
  isWebcamActive = true;

  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;

  console.log("🔍 Starting webcam analysis...");

  webcamInterval = setInterval(async () => {
    if (video.readyState === video.HAVE_ENOUGH_DATA && isWebcamActive) {
      // Draw video frame to canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Get frame data
      const frameData = canvas.toDataURL("image/jpeg", 0.8);

      try {
        const response = await fetch("/webcam", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ frame: frameData }),
        });

        if (response.ok) {
          const result = await response.json();
          updateWebcamResult(result);
        } else {
          console.error("Webcam analysis error:", response.status);
        }
      } catch (error) {
        console.error("Error sending frame:", error);
      }
    }
  }, 1000); // Analyze every second
}

function stopWebcamAnalysis() {
  isWebcamActive = false;

  if (webcamInterval) {
    clearInterval(webcamInterval);
    webcamInterval = null;
  }

  const webcamResult = document.getElementById("webcamResult");
  if (webcamResult) {
    webcamResult.style.display = "none";
  }

  console.log("🛑 Webcam analysis stopped");
}

function updateWebcamResult(result) {
  const webcamResult = document.getElementById("webcamResult");
  const predictionIndicator = document.getElementById("webcamPrediction");
  const predictionText = document.getElementById("webcamPredictionText");
  const confidenceValue = document.getElementById("webcamConfidenceValue");

  if (
    !webcamResult ||
    !predictionIndicator ||
    !predictionText ||
    !confidenceValue
  )
    return;

  const isViolence = result.label === "Violence";
  predictionText.innerHTML = `
    ${isViolence ? "⚠️" : "✅"} 
    ${isViolence ? "Violence Detected" : "No Violence Detected"}
  `;
  confidenceValue.textContent = `${result.confidence}%`;

  predictionIndicator.className =
    "prediction-indicator " + (isViolence ? "violence" : "non-violence");
  predictionIndicator.setAttribute("aria-live", "polite");

  webcamResult.style.display = "block";
}

// Configuration functions
async function loadConfiguration() {
  try {
    const response = await fetch("/config");
    if (response.ok) {
      const config = await response.json();

      const elements = {
        threshold: config.default_threshold
          ? config.default_threshold * 100
          : 50,
        frameSkip: config.default_frame_skip || 1,
        showFps: config.show_fps !== undefined ? config.show_fps : true,
        showConfidence:
          config.show_confidence !== undefined ? config.show_confidence : true,
        autoSaveLogs:
          config.auto_save_logs !== undefined ? config.auto_save_logs : true,
      };

      Object.entries(elements).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) {
          if (element.type === "checkbox") {
            element.checked = value;
          } else {
            element.value = value;
          }
        }
      });
    }
  } catch (error) {
    console.error("Error loading configuration:", error);
  }
}

async function saveConfiguration() {
  try {
    const config = {
      default_threshold:
        Number.parseFloat(document.getElementById("threshold")?.value || 50) /
        100,
      default_frame_skip: Number.parseInt(
        document.getElementById("frameSkip")?.value || 1
      ),
      show_fps: document.getElementById("showFps")?.checked || false,
      show_confidence:
        document.getElementById("showConfidence")?.checked || false,
      auto_save_logs: document.getElementById("autoSaveLogs")?.checked || false,
      export_format: "json",
      updated_at: new Date().toISOString(),
    };

    const response = await fetch("/config", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(config),
    });

    const result = await response.json();

    if (response.ok) {
      showNotification("✅ Configuration saved successfully!", "success");
    } else {
      showErrorModal(result.error || "Error saving configuration", false);
    }
  } catch (error) {
    console.error("Error saving configuration:", error);
    showErrorModal("Server connection error while saving configuration", false);
  }
}

// Utility functions
function formatFileSize(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return (
    Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
  );
}

function formatDuration(seconds) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, "0")}:${secs
      .toString()
      .padStart(2, "0")}`;
  } else {
    return `${minutes}:${secs.toString().padStart(2, "0")}`;
  }
}

function showErrorModal(message, showRetry = false) {
  const errorModal = document.getElementById("errorModal");
  const errorMessage = document.getElementById("errorMessage");
  const retryBtn = document.getElementById("retryBtn");

  if (errorModal && errorMessage) {
    errorMessage.textContent = message;
    errorModal.style.display = "flex";

    if (retryBtn) {
      retryBtn.style.display = showRetry ? "inline-block" : "none";
    }
  }
}

function closeErrorModal() {
  const errorModal = document.getElementById("errorModal");
  if (errorModal) {
    errorModal.style.display = "none";
  }
}

function showNotification(message, type = "success") {
  const notification = document.createElement("div");
  notification.className = `notification ${type}`;

  const icon =
    type === "success"
      ? "check-circle"
      : type === "error"
      ? "exclamation-triangle"
      : type === "warning"
      ? "exclamation-triangle"
      : "info-circle";

  notification.innerHTML = `
    <i class="fas fa-${icon}"></i>
    <span>${message}</span>
  `;

  document.body.appendChild(notification);

  setTimeout(() => {
    notification.style.opacity = "0";
    notification.style.transform = "translateX(100%)";
    setTimeout(() => {
      if (notification.parentNode) {
        notification.parentNode.removeChild(notification);
      }
    }, 300);
  }, 4000);
}

// Clean up on page unload
window.addEventListener("beforeunload", () => {
  if (webcamStream) {
    stopWebcam();
  }
});

// Handle visibility change
document.addEventListener("visibilitychange", () => {
  if (document.hidden && webcamStream) {
    stopWebcam();
  }
});

// Close modals when clicking outside
window.addEventListener("click", (event) => {
  const errorModal = document.getElementById("errorModal");
  if (event.target === errorModal) {
    closeErrorModal();
  }
});

// Keyboard shortcuts
document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "u") {
    event.preventDefault();
    switchTab("upload");
  }

  if ((event.ctrlKey || event.metaKey) && event.key === "w") {
    event.preventDefault();
    switchTab("webcam");
  }

  if ((event.ctrlKey || event.metaKey) && event.key === "s") {
    event.preventDefault();
    switchTab("settings");
  }

  if (event.key === "Escape") {
    closeErrorModal();
  }
});

// Export functions for global access
window.ViolenceDetectionApp = {
  switchTab,
  resetUpload,
  showErrorModal,
  showNotification,
  closeErrorModal,
  formatFileSize,
  formatDuration,
};
