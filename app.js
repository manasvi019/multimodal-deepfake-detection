


"use strict";

// ==========================================
// ELEMENTS
// ==========================================

const videoFile = document.getElementById("videoFile");
const dropZone = document.getElementById("dropZone");
const fileDetails = document.getElementById("fileDetails");
const fileName = document.getElementById("fileName");
const fileInfo = document.getElementById("fileInfo");
const videoPreview = document.getElementById("videoPreview");
const fileMessage = document.getElementById("fileMessage");
const removeVideo = document.getElementById("removeVideo");
const analyzeButton = document.getElementById("analyzeButton");
const analysisNotice = document.getElementById("analysisNotice");
const resultsSection = document.getElementById("resultsSection");

let selectedVideo = null;
let previewUrl = null;

// ==========================================
// HELPERS
// ==========================================

function formatDuration(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);

    return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
}

function formatFileSize(bytes) {
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function resetAnalysis() {
    resultsSection.hidden = true;

    analysisNotice.textContent =
        "Frontend prototype — analysis is not connected yet.";
}

// ==========================================
// CLEAR SELECTED VIDEO
// ==========================================

function clearVideo() {
    videoPreview.pause();
    videoPreview.removeAttribute("src");
    videoPreview.load();

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }

    selectedVideo = null;
    videoFile.value = "";

    fileDetails.hidden = true;
    analyzeButton.disabled = true;

    fileName.textContent = "";
    fileInfo.textContent = "";
    fileMessage.textContent = "No video selected yet.";

    resetAnalysis();
}

// ==========================================
// SELECT VIDEO
// ==========================================

function selectVideo(file) {
    if (!file) return;

    if (!file.type.startsWith("video/")) {
        fileMessage.textContent =
            "Please select a valid video file.";

        return;
    }

    // Release the previous preview before selecting another file.
    clearVideo();

    selectedVideo = file;
    previewUrl = URL.createObjectURL(file);

    fileName.textContent = file.name;
    fileInfo.textContent = formatFileSize(file.size);

    fileDetails.hidden = false;
    analyzeButton.disabled = false;

    fileMessage.textContent = "Video selected successfully.";

    videoPreview.src = previewUrl;

    videoPreview.onloadedmetadata = () => {
        if (selectedVideo !== file) return;

        if (Number.isFinite(videoPreview.duration)) {
            fileInfo.textContent =
                `${formatFileSize(file.size)} · ` +
                formatDuration(videoPreview.duration);
        }
    };

    videoPreview.onerror = () => {
        if (selectedVideo !== file) return;

        fileMessage.textContent =
            "The video preview could not be loaded. " +
            "Try another video.";
    };

    resetAnalysis();
}

// ==========================================
// FILE INPUT
// ==========================================

videoFile.addEventListener("change", () => {
    selectVideo(videoFile.files[0]);
});

// ==========================================
// REMOVE VIDEO
// ==========================================

removeVideo.addEventListener("click", clearVideo);

// ==========================================
// DRAG AND DROP
// ==========================================

dropZone.addEventListener("click", (event) => {
    if (event.target.closest("input, label, button")) return;

    videoFile.click();
});

dropZone.addEventListener("keydown", (event) => {
    if (event.target !== dropZone) return;

    if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        videoFile.click();
    }
});

["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.add("drag-over");
    });
});

["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.remove("drag-over");
    });
});

dropZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];

    selectVideo(file);
});

// ==========================================
// ANALYSIS INTERFACE
// Backend integration will be added later.
// ==========================================

analyzeButton.addEventListener("click", () => {
    if (!selectedVideo) {
        fileMessage.textContent =
            "Please select a video before continuing.";

        return;
    }

    resultsSection.hidden = false;

    analysisNotice.textContent =
        "Preview ready. Connect the detection backend " +
        "to generate actual analysis results.";

    resultsSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
});



