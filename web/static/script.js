const canvas = document.getElementById("digitCanvas");
const ctx = canvas.getContext("2d");

const previewCanvas = document.getElementById("previewCanvas");
const previewCtx = previewCanvas.getContext("2d");

const clearBtn = document.getElementById("clearBtn");
const undoBtn = document.getElementById("undoBtn");
const saveBtn = document.getElementById("saveBtn");
const predictBtn = document.getElementById("predictBtn");

const uploadInput = document.getElementById("uploadInput");
const uploadBtn = document.getElementById("uploadBtn");
const uploadFileName = document.getElementById("uploadFileName");

const resultNumber = document.getElementById("resultNumber");
const resultText = document.getElementById("resultText");
const confidenceText = document.getElementById("confidenceText");

const toolButtons = document.querySelectorAll("[data-tool]");
const sizeButtons = document.querySelectorAll("[data-size]");
const colorButtons = document.querySelectorAll("[data-color]");

let isDrawing = false;
let currentTool = "pencil";
let brushSize = 14;
let brushColor = "#111111";
let lastPoint = null;
let undoStack = [];

function fillCanvasWhite() {
    ctx.save();
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.restore();
}

function saveState() {
    undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));

    if (undoStack.length > 25) {
        undoStack.shift();
    }
}

function updatePreviewFromCanvas() {
    previewCtx.save();
    previewCtx.fillStyle = "#ffffff";
    previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
    previewCtx.drawImage(canvas, 0, 0, previewCanvas.width, previewCanvas.height);
    previewCtx.restore();
}

function updatePreviewFromImage(imageBase64) {
    const img = new Image();

    img.onload = () => {
        previewCtx.save();
        previewCtx.fillStyle = "#ffffff";
        previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
        previewCtx.drawImage(img, 0, 0, previewCanvas.width, previewCanvas.height);
        previewCtx.restore();
    };

    img.src = imageBase64;
}

function resetResult(message = "Draw or upload one or more digits, then click Predict.") {
    resultNumber.textContent = "?";
    resultNumber.classList.remove("long-result");
    resultText.textContent = "Predicted Number";
    confidenceText.textContent = message;
}

function getPoint(event) {
    const rect = canvas.getBoundingClientRect();
    const clientX = event.touches ? event.touches[0].clientX : event.clientX;
    const clientY = event.touches ? event.touches[0].clientY : event.clientY;

    return {
        x: (clientX - rect.left) * (canvas.width / rect.width),
        y: (clientY - rect.top) * (canvas.height / rect.height)
    };
}

function applyBrushStyle() {
    ctx.lineJoin = "round";
    ctx.lineCap = currentTool === "pixel" ? "square" : "round";
    ctx.lineWidth = currentTool === "eraser" ? brushSize * 1.4 : brushSize;
    ctx.strokeStyle = currentTool === "eraser" ? "#ffffff" : brushColor;
    ctx.fillStyle = currentTool === "eraser" ? "#ffffff" : brushColor;
    ctx.globalAlpha = currentTool === "marker" ? 0.74 : 1;
    ctx.shadowBlur = currentTool === "neon" ? 18 : 0;
    ctx.shadowColor = currentTool === "neon" ? brushColor : "transparent";
}

function drawLine(from, to) {
    applyBrushStyle();

    if (currentTool === "pixel") {
        const step = brushSize * 0.75;
        const distance = Math.hypot(to.x - from.x, to.y - from.y);
        const steps = Math.max(1, Math.floor(distance / step));

        for (let i = 0; i <= steps; i++) {
            const x = from.x + ((to.x - from.x) * i) / steps;
            const y = from.y + ((to.y - from.y) * i) / steps;
            ctx.fillRect(x - brushSize / 2, y - brushSize / 2, brushSize, brushSize);
        }
        return;
    }

    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
}

function startDrawing(event) {
    event.preventDefault();
    saveState();
    isDrawing = true;
    lastPoint = getPoint(event);

    applyBrushStyle();
    ctx.beginPath();
    ctx.arc(lastPoint.x, lastPoint.y, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fill();

    updatePreviewFromCanvas();
    resetResult("Drawing detected. Click Predict Number when ready.");
}

function draw(event) {
    if (!isDrawing) return;

    event.preventDefault();
    const point = getPoint(event);

    drawLine(lastPoint, point);
    lastPoint = point;
    updatePreviewFromCanvas();
}

function stopDrawing() {
    if (!isDrawing) return;

    isDrawing = false;
    lastPoint = null;
    ctx.globalAlpha = 1;
    ctx.shadowBlur = 0;
    updatePreviewFromCanvas();
}

function clearCanvas() {
    saveState();
    fillCanvasWhite();
    updatePreviewFromCanvas();
    resetResult("Canvas cleared. Draw a new number.");
}

function undoCanvas() {
    const previousState = undoStack.pop();

    if (!previousState) {
        resetResult("Nothing to undo.");
        return;
    }

    ctx.putImageData(previousState, 0, 0);
    updatePreviewFromCanvas();
    resetResult("Undo completed.");
}

function saveImage() {
    const link = document.createElement("a");
    link.download = "handwritten-number.png";
    link.href = canvas.toDataURL("image/png");
    link.click();
}

function setActiveButton(buttons, selectedButton) {
    buttons.forEach(button => button.classList.remove("active"));
    selectedButton.classList.add("active");
}

function getCanvasImageBase64() {
    return canvas.toDataURL("image/png");
}

function formatDigitBreakdown(digits) {
    if (!digits || digits.length === 0) {
        return "";
    }

    return digits
        .map(item => `${item.digit} (${item.confidence})`)
        .join("  |  ");
}

function renderPrediction(data, sourceName) {
    const predictedNumber = data.number ?? data.digit ?? "?";
    const predictedText = String(predictedNumber);

    resultNumber.textContent = predictedText;

    if (predictedText.length >= 3) {
        resultNumber.classList.add("long-result");
    } else {
        resultNumber.classList.remove("long-result");
    }

    resultText.textContent = predictedText.length > 1 ? "Predicted Number" : "Predicted Digit";

    const breakdown = formatDigitBreakdown(data.digits);

    if (breakdown) {
        confidenceText.textContent = `${sourceName} | Average confidence: ${data.confidence} | Digits: ${breakdown}`;
    } else {
        confidenceText.textContent = data.confidence
            ? `${sourceName} | Confidence: ${data.confidence}`
            : `${sourceName} | Prediction completed.`;
    }
}

async function sendImageForPrediction(imageBase64, sourceName) {
    resultNumber.textContent = "...";
    resultNumber.classList.remove("long-result");
    resultText.textContent = "Predicting";
    confidenceText.textContent = `Preparing ${sourceName.toLowerCase()} image for the Logistic Regression model...`;

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                image: imageBase64
            })
        });

        const data = await response.json();

        if (!response.ok || (data.status && data.status !== "success")) {
            throw new Error(data.message || "Prediction failed.");
        }

        renderPrediction(data, sourceName);
    } catch (error) {
        resultNumber.textContent = "?";
        resultText.textContent = "Prediction Failed";
        confidenceText.textContent = error.message || "Backend not connected.";
    }
}

async function predictCanvasNumber() {
    updatePreviewFromCanvas();
    const imageBase64 = getCanvasImageBase64();
    await sendImageForPrediction(imageBase64, "Canvas");
}

function predictUploadedImage() {
    const file = uploadInput.files && uploadInput.files[0];

    if (!file) {
        resetResult("Please choose an image file first.");
        return;
    }

    const reader = new FileReader();

    reader.onload = async event => {
        const imageBase64 = event.target.result;
        updatePreviewFromImage(imageBase64);
        await sendImageForPrediction(imageBase64, "Uploaded image");
    };

    reader.onerror = () => {
        resetResult("Cannot read this image file.");
    };

    reader.readAsDataURL(file);
}

toolButtons.forEach(button => {
    button.addEventListener("click", () => {
        currentTool = button.dataset.tool;
        setActiveButton(toolButtons, button);
        resetResult(`Selected tool: ${button.textContent.trim()}`);
    });
});

sizeButtons.forEach(button => {
    button.addEventListener("click", () => {
        brushSize = Number(button.dataset.size);
        setActiveButton(sizeButtons, button);
        resetResult(`Brush size changed to ${brushSize}px.`);
    });
});

colorButtons.forEach(button => {
    button.addEventListener("click", () => {
        brushColor = button.dataset.color;
        setActiveButton(colorButtons, button);
        resetResult("Ink color changed.");
    });
});

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
window.addEventListener("mouseup", stopDrawing);

canvas.addEventListener("touchstart", startDrawing, { passive: false });
canvas.addEventListener("touchmove", draw, { passive: false });
canvas.addEventListener("touchend", stopDrawing);

clearBtn.addEventListener("click", clearCanvas);
undoBtn.addEventListener("click", undoCanvas);
saveBtn.addEventListener("click", saveImage);
predictBtn.addEventListener("click", predictCanvasNumber);

if (uploadInput) {
    uploadInput.addEventListener("change", () => {
        const file = uploadInput.files && uploadInput.files[0];

        if (!file) {
            uploadFileName.textContent = "No image selected";
            return;
        }

        uploadFileName.textContent = file.name;

        const reader = new FileReader();

        reader.onload = event => {
            updatePreviewFromImage(event.target.result);
            resetResult("Image selected. Click Predict Uploaded Image.");
        };

        reader.readAsDataURL(file);
    });
}

if (uploadBtn) {
    uploadBtn.addEventListener("click", predictUploadedImage);
}

fillCanvasWhite();
saveState();
updatePreviewFromCanvas();
