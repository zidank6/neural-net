# MNIST 3D Neural Network Visualizer

A neural network built **from scratch** in Python/NumPy — no ML frameworks — trained on MNIST handwritten digits, with an interactive 3D web visualizer.

![Visualizer Screenshot](https://github.com/user-attachments/assets/placeholder.png)

## What This Is

- **`neural_net.py`** — The engine. Forward pass, backpropagation, ReLU, Softmax, Cross-Entropy loss — all implemented from scratch with NumPy.
- **`train_mnist.py`** — Trains the network on 60,000 MNIST images. Achieves **97.3% test accuracy**.
- **`visualizer/`** — A client-side web app that loads the trained weights and runs inference in the browser. Draw a digit and watch the network think in real time.

## Architecture

```
Input (784) → Hidden (128, ReLU) → Hidden (64, ReLU) → Output (10, Softmax)
```

## Visualizer Features

- **Drawing canvas** — Draw any digit with your mouse
- **Real-time inference** — Forward pass runs in JavaScript using the exported weights
- **3D network** — Three.js scene with neurons, connections, and a 28×28 input pixel grid
- **Activation glow** — Neurons and connections light up based on live signal flow
- **Probability bars** — See the confidence for each digit 0–9

## Run Locally

### Train the model (optional — weights are already exported)

```bash
pip install numpy scikit-learn
python train_mnist.py
```

### Launch the visualizer

```bash
cd visualizer
python3 -m http.server 8080
# open http://localhost:8080
```

## Project Structure

```
neural-net/
├── neural_net.py          # Neural network engine (from scratch)
├── train_mnist.py         # MNIST training script
├── model_weights.json     # Exported trained weights
└── visualizer/
    ├── index.html         # App layout
    ├── style.css          # Dark theme + glassmorphism
    ├── app.js             # Drawing, inference, UI updates
    ├── network3d.js       # Three.js 3D visualization
    └── model_weights.json # Weights (copy for web app)
```

## Tech Stack

| Layer | Tech |
|-------|------|
| Neural network | Python, NumPy |
| Training data | MNIST (scikit-learn) |
| Web frontend | HTML, CSS, JavaScript |
| 3D rendering | Three.js |
| Deployment | Any static host (Vercel, GitHub Pages) |
