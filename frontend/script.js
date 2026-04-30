(() => {
  "use strict";

  const API_URL = "http://localhost:8000/predict";
  const INTERVAL_MS = 1000;

  // ── DOM refs — webcam ─────────────────────────────────────────────────────
  const video = document.getElementById("video");
  const captureCanvas = document.getElementById("capture");
  const labelOverlay = document.getElementById("label-overlay");
  const videoWrapper = document.querySelector(".video-wrapper");
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");

  // ── DOM refs — upload ─────────────────────────────────────────────────────
  const fileInput = document.getElementById("fileInput");
  const imagePreview = document.getElementById("imagePreview");
  const previewWrapper = document.getElementById("previewWrapper");
  const predictUploadBtn = document.getElementById("predictUploadBtn");
  const clearFileBtn = document.getElementById("clearFileBtn");
  const dropZone = document.getElementById("dropZone");

  // ── DOM refs — shared result ───────────────────────────────────────────────
  const resultCard = document.getElementById("result-card");
  const resultSource = document.getElementById("result-source");
  const resultStatus = document.getElementById("result-status");
  const resultAction = document.getElementById("result-action");
  const resultConfidence = document.getElementById("result-confidence");
  const confidenceBar = document.getElementById("confidence-bar");
  const errorMsg = document.getElementById("error-msg");

  let detectionInterval = null;
  let isDetecting = false;

  // ════════════════════════════════════════════════════════════════════════════
  // SHARED API FUNCTION
  // ════════════════════════════════════════════════════════════════════════════

  /**
   * sendToAPI(file)
   * Sends any File or Blob to POST /predict and returns the parsed JSON.
   * Throws on network errors or non-2xx responses.
   */
  async function sendToAPI(file) {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(API_URL, { method: "POST", body: formData });

    if (!res.ok) {
      throw new Error(`Server returned ${res.status} ${res.statusText}`);
    }

    return res.json();
  }

  // ════════════════════════════════════════════════════════════════════════════
  // WEBCAM
  // ════════════════════════════════════════════════════════════════════════════

  async function initWebcam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      hideError();
    } catch (err) {
      showError(`Webcam access denied or unavailable: ${err.message}`);
      startBtn.disabled = true;
    }
  }

  function captureFrame() {
    captureCanvas.width = video.videoWidth || 640;
    captureCanvas.height = video.videoHeight || 480;

    const ctx = captureCanvas.getContext("2d");
    // Un-mirror so the model receives a natural (non-flipped) image
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(
      video,
      -captureCanvas.width,
      0,
      captureCanvas.width,
      captureCanvas.height,
    );
    ctx.restore();

    return new Promise((resolve) =>
      captureCanvas.toBlob(resolve, "image/jpeg", 0.85),
    );
  }

  async function predictFromWebcam() {
    let blob;
    try {
      blob = await captureFrame();
    } catch {
      return; // video not ready yet
    }

    try {
      const data = await sendToAPI(
        new File([blob], "frame.jpg", { type: "image/jpeg" }),
      );
      hideError();
      renderResult(data, "Webcam");
    } catch (err) {
      showError(`Webcam prediction failed: ${err.message}`);
    }
  }

  function startDetection() {
    if (isDetecting) return;
    isDetecting = true;

    startBtn.disabled = true;
    stopBtn.disabled = false;
    videoWrapper.classList.add("detecting");
    labelOverlay.classList.remove("hidden");

    predictFromWebcam();
    detectionInterval = setInterval(predictFromWebcam, INTERVAL_MS);
  }

  function stopDetection() {
    if (!isDetecting) return;
    isDetecting = false;

    clearInterval(detectionInterval);
    detectionInterval = null;

    startBtn.disabled = false;
    stopBtn.disabled = true;
    videoWrapper.classList.remove("detecting");
    labelOverlay.classList.add("hidden");
  }

  // ════════════════════════════════════════════════════════════════════════════
  // IMAGE UPLOAD
  // ════════════════════════════════════════════════════════════════════════════

  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];

    if (!file) {
      predictUploadBtn.disabled = true;
      previewWrapper.classList.add("hidden");
      return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      previewWrapper.classList.remove("hidden");
    };
    reader.readAsDataURL(file);

    predictUploadBtn.disabled = false;
  });

  predictUploadBtn.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) return;

    setUploadLoading(true);
    hideError();

    try {
      const data = await sendToAPI(file);
      renderResult(data, "Image Upload");
    } catch (err) {
      showError(`Upload prediction failed: ${err.message}`);
    } finally {
      setUploadLoading(false);
    }
  });

  function setUploadLoading(loading) {
    predictUploadBtn.disabled = loading;
    predictUploadBtn.textContent = loading ? "Analyzing…" : "Predict Image";
    predictUploadBtn.classList.toggle("loading", loading);
  }

  // Clear button
  clearFileBtn.addEventListener("click", () => {
    fileInput.value = "";
    previewWrapper.classList.add("hidden");
    predictUploadBtn.disabled = true;
  });

  // Drag-and-drop onto the drop zone label
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  ["dragleave", "dragend"].forEach((evt) =>
    dropZone.addEventListener(evt, () =>
      dropZone.classList.remove("drag-over"),
    ),
  );

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (!file || !file.type.startsWith("image/")) return;

    // Inject into the hidden file input so the rest of the flow is identical
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    fileInput.dispatchEvent(new Event("change"));
  });

  // ════════════════════════════════════════════════════════════════════════════
  // SHARED RENDER
  // ════════════════════════════════════════════════════════════════════════════

  /**
   * renderResult(data, source)
   * Populates the shared result card for both webcam and upload results.
   * source: "Webcam" | "Image Upload"
   */
  function renderResult(data, source) {
    const status = (data.status || "").toLowerCase();
    const isMaskOn = status === "mask_on";
    const cls = isMaskOn ? "mask-on" : "mask-off";

    // Normalise confidence to 0-100
    let conf = parseFloat(data.confidence ?? data.confidence_score ?? 0);
    if (conf > 1) conf = conf / 100;
    const confPct = Math.round(conf * 100);

    // Source tag
    resultSource.textContent = source;
    resultSource.className = `source-tag ${source === "Webcam" ? "source-webcam" : "source-upload"}`;

    // Status badge
    resultStatus.textContent = isMaskOn ? "Mask On" : "No Mask";
    resultStatus.className = `result-value badge ${cls}`;

    // Action
    resultAction.textContent =
      data.action || (isMaskOn ? "Allow entry" : "Deny entry");
    resultAction.style.color = isMaskOn ? "var(--mask-on)" : "var(--mask-off)";

    // Confidence
    confidenceBar.style.width = `${confPct}%`;
    confidenceBar.className = `confidence-bar ${cls}`;
    resultConfidence.textContent = `${confPct}%`;

    // Live label on video (only meaningful for webcam, hide for upload)
    if (source === "Webcam") {
      labelOverlay.textContent = isMaskOn ? "✓ Mask On" : "✗ No Mask";
      labelOverlay.className = `label-overlay ${cls}`;
    }

    resultCard.classList.remove("hidden");
  }

  // ════════════════════════════════════════════════════════════════════════════
  // ERROR HELPERS
  // ════════════════════════════════════════════════════════════════════════════

  function showError(msg) {
    errorMsg.textContent = msg;
    errorMsg.classList.remove("hidden");
  }

  function hideError() {
    errorMsg.classList.add("hidden");
  }

  // ════════════════════════════════════════════════════════════════════════════
  // EVENT LISTENERS & BOOT
  // ════════════════════════════════════════════════════════════════════════════

  startBtn.addEventListener("click", startDetection);
  stopBtn.addEventListener("click", stopDetection);

  document.addEventListener("visibilitychange", () => {
    if (document.hidden && isDetecting) stopDetection();
  });

  initWebcam();
})();
