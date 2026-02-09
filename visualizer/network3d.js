/**
 * network3d.js — Three.js 3D neural network visualization
 * Renders input image as a pixel grid, neurons as glowing spheres,
 * and connections as weight-colored lines.
 */

class Network3D {
    constructor(container, layerSizes, modelWeights) {
        this.container = container;
        this.layerSizes = layerSizes;
        this.modelWeights = modelWeights;
        this.neurons = [];
        this.connections = [];
        this.inputPixels = [];

        this.initScene();
        this.buildNetwork();
        this.animate();
    }

    initScene() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.scene = new THREE.Scene();

        // Camera
        this.camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 1000);
        this.camera.position.set(0, 0, 24);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setClearColor(0x000000, 0);
        this.container.appendChild(this.renderer.domElement);

        // Orbit controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.enableZoom = true;
        this.controls.autoRotate = false;

        // Lighting
        const ambient = new THREE.AmbientLight(0xffffff, 0.3);
        this.scene.add(ambient);

        const point1 = new THREE.PointLight(0xffa040, 0.6);
        point1.position.set(10, 8, 15);
        this.scene.add(point1);

        const point2 = new THREE.PointLight(0x4060ff, 0.3);
        point2.position.set(-10, -5, 10);
        this.scene.add(point2);

        window.addEventListener('resize', () => this.onResize());
    }

    buildNetwork() {
        const layers = this.layerSizes;
        const numLayers = layers.length;

        // Visual neuron counts per layer (can't show all 784)
        const visibleCounts = [14, 16, 12, 10];

        const layerSpacing = 4;
        const totalWidth = (numLayers - 1) * layerSpacing;
        const startX = -totalWidth / 2 + 3; // shift right for grid space

        // ---- Build input pixel grid (28x28 miniature) ----
        this.buildInputGrid(startX - 4);

        // ---- Build neuron layers ----
        for (let l = 0; l < numLayers; l++) {
            const layerNeurons = [];
            const visibleCount = visibleCounts[l] || 10;
            const neuronSpacing = l === numLayers - 1 ? 0.65 : 0.5;
            const totalHeight = (visibleCount - 1) * neuronSpacing;
            const startY = totalHeight / 2;
            const x = startX + l * layerSpacing;

            for (let n = 0; n < visibleCount; n++) {
                const y = startY - n * neuronSpacing;

                // Neuron size scales by layer
                const radius = l === 0 ? 0.1 : (l === numLayers - 1 ? 0.2 : 0.14);

                const geometry = new THREE.SphereGeometry(radius, 20, 20);
                const material = new THREE.MeshPhongMaterial({
                    color: 0x2a2a3e,
                    emissive: 0x111122,
                    emissiveIntensity: 0.4,
                    transparent: true,
                    opacity: 0.7,
                    shininess: 80,
                });

                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(x, y, 0);
                this.scene.add(mesh);
                layerNeurons.push(mesh);
            }

            this.neurons.push(layerNeurons);
        }

        // ---- Build connections ----
        for (let l = 0; l < numLayers - 1; l++) {
            const layerConns = [];
            const fromNeurons = this.neurons[l];
            const toNeurons = this.neurons[l + 1];
            const weights = this.modelWeights ? this.modelWeights.weights[l] : null;

            const maxConns = 300;
            let connCount = 0;

            // Connect every visible neuron to every visible neuron
            const fromStep = 1;
            const toStep = 1;

            for (let f = 0; f < fromNeurons.length && connCount < maxConns; f += fromStep) {
                for (let t = 0; t < toNeurons.length && connCount < maxConns; t += toStep) {
                    const points = [fromNeurons[f].position, toNeurons[t].position];
                    const geometry = new THREE.BufferGeometry().setFromPoints(points);

                    // Get actual weight value for coloring
                    let weightVal = 0;
                    if (weights) {
                        const actualFrom = Math.floor(f * this.layerSizes[l] / fromNeurons.length);
                        const actualTo = Math.floor(t * this.layerSizes[l + 1] / toNeurons.length);
                        if (weights[actualFrom] && weights[actualFrom][actualTo] !== undefined) {
                            weightVal = weights[actualFrom][actualTo];
                        }
                    }

                    // Orange/yellow color based on weight magnitude
                    const absW = Math.min(Math.abs(weightVal), 1);
                    const material = new THREE.LineBasicMaterial({
                        color: new THREE.Color().setHSL(
                            0.08 + absW * 0.06,  // hue: 0.08 (orange) to 0.14 (yellow)
                            0.9,
                            0.25 + absW * 0.35
                        ),
                        transparent: true,
                        opacity: 0.06 + absW * 0.12,
                    });

                    const line = new THREE.Line(geometry, material);
                    this.scene.add(line);
                    layerConns.push({
                        line, from: f, to: t,
                        weight: weightVal,
                        baseMaterial: material.clone()
                    });
                    connCount++;
                }
            }
            this.connections.push(layerConns);
        }
    }

    buildInputGrid(gridCenterX) {
        // 28x28 grid using InstancedMesh (1 draw call instead of 784)
        const gridSize = 28;
        const cellSize = 0.14;
        const totalSize = gridSize * cellSize;
        const gridStartX = gridCenterX - totalSize / 2;
        const gridStartY = totalSize / 2;
        const count = gridSize * gridSize;

        const geometry = new THREE.BoxGeometry(cellSize * 0.8, cellSize * 0.8, cellSize * 0.3);
        const material = new THREE.MeshPhongMaterial({
            color: 0xffa500,
            emissive: 0x000000,
            transparent: true,
            opacity: 0.15,
        });

        this.inputGrid = new THREE.InstancedMesh(geometry, material, count);
        this.inputGrid.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

        const dummy = new THREE.Object3D();
        const color = new THREE.Color();

        for (let row = 0; row < gridSize; row++) {
            for (let col = 0; col < gridSize; col++) {
                const idx = row * gridSize + col;
                dummy.position.set(
                    gridStartX + col * cellSize,
                    gridStartY - row * cellSize,
                    0
                );
                dummy.updateMatrix();
                this.inputGrid.setMatrixAt(idx, dummy.matrix);
                this.inputGrid.setColorAt(idx, color.setHex(0x0a0a0a));
            }
        }

        this.inputGrid.instanceMatrix.needsUpdate = true;
        this.inputGrid.instanceColor.needsUpdate = true;
        this.scene.add(this.inputGrid);
    }

    updateActivations(allActivations) {
        // ---- Update input pixel grid (InstancedMesh) ----
        const inputActivations = allActivations[0];
        const color = new THREE.Color();
        for (let i = 0; i < inputActivations.length; i++) {
            const val = inputActivations[i];
            if (val > 0.05) {
                const hue = 0.1 - val * 0.05;
                const sat = 1.0 - val * 0.3;
                const light = 0.2 + val * 0.6;
                color.setHSL(hue, sat, light);
            } else {
                color.setHex(0x0a0a0a);
            }
            this.inputGrid.setColorAt(i, color);
        }
        this.inputGrid.instanceColor.needsUpdate = true;

        // ---- Update neurons ----
        for (let l = 0; l < this.neurons.length; l++) {
            const neurons = this.neurons[l];
            const activations = allActivations[l];

            for (let n = 0; n < neurons.length; n++) {
                const actualIdx = Math.floor(n * activations.length / neurons.length);
                const value = Math.min(1, Math.max(0, activations[actualIdx]));
                const mesh = neurons[n];

                if (l === this.neurons.length - 1) {
                    // Output: bright green/cyan for high confidence
                    mesh.material.emissive.setHSL(0.35, 0.9, value * 0.5);
                    mesh.material.emissiveIntensity = 0.3 + value * 2.5;
                    mesh.material.color.setHSL(0.35, 0.5 + value * 0.4, 0.15 + value * 0.5);
                    mesh.material.opacity = 0.5 + value * 0.5;
                } else {
                    // Hidden: warm orange/amber glow
                    const hue = 0.08 + value * 0.04;
                    mesh.material.emissive.setHSL(hue, 0.9, value * 0.45);
                    mesh.material.emissiveIntensity = 0.2 + value * 2.0;
                    mesh.material.color.setHSL(hue, 0.4 + value * 0.4, 0.1 + value * 0.4);
                    mesh.material.opacity = 0.4 + value * 0.6;
                }
            }
        }

        // ---- Update connections ----
        for (let l = 0; l < this.connections.length; l++) {
            const conns = this.connections[l];
            const fromAct = allActivations[l];
            const toAct = allActivations[l + 1];

            for (const conn of conns) {
                const fromIdx = Math.floor(conn.from * fromAct.length / this.neurons[l].length);
                const toIdx = Math.floor(conn.to * toAct.length / this.neurons[l + 1].length);

                const fromVal = Math.min(1, Math.abs(fromAct[fromIdx]));
                const toVal = Math.min(1, Math.abs(toAct[toIdx]));
                const flow = Math.sqrt(fromVal * toVal);

                if (flow > 0.02) {
                    const hue = 0.06 + flow * 0.08;
                    const lightness = 0.3 + flow * 0.4;
                    conn.line.material.color.setHSL(hue, 0.95, lightness);
                    conn.line.material.opacity = 0.04 + flow * 0.45;
                } else {
                    conn.line.material.color.setHSL(0.08, 0.5, 0.15);
                    conn.line.material.opacity = 0.02;
                }
            }
        }
    }

    clearActivations() {
        // Reset input grid
        const color = new THREE.Color(0x0a0a0a);
        for (let i = 0; i < 784; i++) {
            this.inputGrid.setColorAt(i, color);
        }
        this.inputGrid.instanceColor.needsUpdate = true;

        // Reset neurons
        for (const layer of this.neurons) {
            for (const mesh of layer) {
                mesh.material.emissive.setHex(0x111122);
                mesh.material.emissiveIntensity = 0.4;
                mesh.material.color.setHex(0x2a2a3e);
                mesh.material.opacity = 0.7;
            }
        }

        // Reset connections
        for (const layerConns of this.connections) {
            for (const conn of layerConns) {
                conn.line.material.color.setHSL(0.08, 0.5, 0.15);
                conn.line.material.opacity = 0.03;
            }
        }
    }

    onResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}
