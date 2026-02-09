/**
 * app.js — Main application logic
 * Handles drawing canvas, model loading, inference, and UI updates.
 */

// ===== Global State =====
let modelWeights = null;
let network3d = null;

// ===== Model Loading =====
async function loadModel() {
    const response = await fetch('model_weights.json');
    modelWeights = await response.json();
    console.log('Model loaded:', modelWeights.layer_sizes);
    renderLayerInfo(modelWeights);
}

// ===== Forward Pass (Pure JS — no libraries needed) =====

function relu(arr) {
    return arr.map(v => Math.max(0, v));
}

function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / sum);
}

function matMul(input, weights, biases) {
    // input: [input_dim], weights: [input_dim][output_dim], biases: [1][output_dim]
    const outputDim = weights[0].length;
    const result = new Array(outputDim).fill(0);
    for (let j = 0; j < outputDim; j++) {
        for (let i = 0; i < input.length; i++) {
            result[j] += input[i] * weights[i][j];
        }
        result[j] += biases[0][j];
    }
    return result;
}

function predict(pixelData) {
    if (!modelWeights) return null;

    const weights = modelWeights.weights;
    const biases = modelWeights.biases;
    const numLayers = weights.length;

    let activation = pixelData;
    const allActivations = [activation];

    // Hidden layers (ReLU)
    for (let i = 0; i < numLayers - 1; i++) {
        let z = matMul(activation, weights[i], biases[i]);
        activation = relu(z);
        allActivations.push(activation);
    }

    // Output layer (Softmax)
    let outputZ = matMul(activation, weights[numLayers - 1], biases[numLayers - 1]);
    let probabilities = softmax(outputZ);
    allActivations.push(probabilities);

    return { probabilities, allActivations };
}

// ===== Drawing Canvas =====
function setupCanvas() {
    const canvas = document.getElementById('draw-canvas');
    const ctx = canvas.getContext('2d');
    let isDrawing = false;

    // Black background
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, 280, 280);

    // Drawing settings — white brush
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    canvas.addEventListener('mousedown', (e) => {
        isDrawing = true;
        ctx.beginPath();
        const rect = canvas.getBoundingClientRect();
        ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
    });

    canvas.addEventListener('mousemove', (e) => {
        if (!isDrawing) return;
        const rect = canvas.getBoundingClientRect();
        ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
        ctx.stroke();
    });

    canvas.addEventListener('mouseup', () => {
        isDrawing = false;
        runInference();
    });

    canvas.addEventListener('mouseleave', () => {
        if (isDrawing) {
            isDrawing = false;
            runInference();
        }
    });

    // Clear button
    document.getElementById('clear-btn').addEventListener('click', () => {
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, 280, 280);
        clearPredictions();
        if (network3d) network3d.clearActivations();
    });
}

function getPixelData() {
    const canvas = document.getElementById('draw-canvas');
    const ctx = canvas.getContext('2d');

    // Create a temporary 28x28 canvas to downsample
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = 28;
    tmpCanvas.height = 28;
    const tmpCtx = tmpCanvas.getContext('2d');

    // Draw the 280x280 canvas onto the 28x28 canvas (downsampling)
    tmpCtx.drawImage(canvas, 0, 0, 28, 28);

    // Extract pixel data — grayscale values normalized to 0-1
    const imageData = tmpCtx.getImageData(0, 0, 28, 28);
    const pixels = new Array(784);
    for (let i = 0; i < 784; i++) {
        // Use the red channel (since we drew white on black, R=G=B)
        pixels[i] = imageData.data[i * 4] / 255.0;
    }
    return pixels;
}

function runInference() {
    const pixels = getPixelData();
    const result = predict(pixels);
    if (!result) return;

    updateProbabilityBars(result.probabilities);

    if (network3d) {
        network3d.updateActivations(result.allActivations);
    }
}

// ===== UI Updates =====
function renderProbabilityBars() {
    const container = document.getElementById('probability-bars');
    container.innerHTML = '';
    for (let i = 0; i < 10; i++) {
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.id = `prob-row-${i}`;
        row.innerHTML = `
            <span class="prob-label">${i}</span>
            <div class="prob-bar-track">
                <div class="prob-bar-fill" id="prob-fill-${i}"></div>
            </div>
            <span class="prob-value" id="prob-val-${i}">0.0%</span>
        `;
        container.appendChild(row);
    }
}

function updateProbabilityBars(probabilities) {
    const topIdx = probabilities.indexOf(Math.max(...probabilities));

    for (let i = 0; i < 10; i++) {
        const pct = probabilities[i] * 100;
        const fill = document.getElementById(`prob-fill-${i}`);
        const val = document.getElementById(`prob-val-${i}`);
        const row = document.getElementById(`prob-row-${i}`);

        fill.style.width = `${pct}%`;
        val.textContent = `${pct.toFixed(1)}%`;

        if (i === topIdx) {
            row.classList.add('top-prediction');
        } else {
            row.classList.remove('top-prediction');
        }
    }
}

function clearPredictions() {
    for (let i = 0; i < 10; i++) {
        const fill = document.getElementById(`prob-fill-${i}`);
        const val = document.getElementById(`prob-val-${i}`);
        const row = document.getElementById(`prob-row-${i}`);
        fill.style.width = '0%';
        val.textContent = '0.0%';
        row.classList.remove('top-prediction');
    }
}

function renderLayerInfo(model) {
    const container = document.getElementById('layer-info');
    container.innerHTML = '';
    const sizes = model.layer_sizes;
    const layerNames = ['Input', ...sizes.slice(1, -1).map((_, i) => `Hidden ${i + 1}`), 'Output'];
    const activations = ['—', ...sizes.slice(1, -1).map(() => 'relu'), 'softmax'];

    for (let i = 0; i < sizes.length; i++) {
        const row = document.createElement('div');
        row.className = 'layer-row';

        const dims = i < sizes.length - 1
            ? `${sizes[i]} → ${sizes[i + 1]}`
            : `${sizes[i]}`;

        row.innerHTML = `
            <span class="layer-name">${layerNames[i]}</span>
            <span class="layer-dims">${dims}</span>
            <span class="layer-activation">${activations[i]}</span>
        `;
        container.appendChild(row);
    }
}

// ===== Init =====
async function init() {
    await loadModel();
    renderProbabilityBars();
    setupCanvas();

    // Initialize 3D visualization
    const container = document.getElementById('network-3d');
    network3d = new Network3D(container, modelWeights.layer_sizes);
}

window.addEventListener('DOMContentLoaded', init);
