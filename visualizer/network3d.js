/**
 * network3d.js — Three.js 3D neural network visualization
 * Renders neurons as spheres and connections as lines.
 * Color-codes connections by weight value and lights up neurons during inference.
 */

class Network3D {
    constructor(container, layerSizes) {
        this.container = container;
        this.layerSizes = layerSizes;
        this.neurons = [];      // [layer][neuron] = mesh
        this.connections = [];  // [layer] = [line meshes]

        this.initScene();
        this.buildNetwork();
        this.animate();
    }

    initScene() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        // Scene
        this.scene = new THREE.Scene();

        // Camera — perspective, looking at center
        this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        this.camera.position.set(0, 0, 18);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setClearColor(0x000000, 0);
        this.container.appendChild(this.renderer.domElement);

        // Orbit controls for mouse rotation
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.enableZoom = true;
        this.controls.autoRotate = false;

        // Ambient light
        const ambient = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambient);

        // Point light from camera direction
        const point = new THREE.PointLight(0xffffff, 0.8);
        point.position.set(5, 5, 15);
        this.scene.add(point);

        // Resize handler
        window.addEventListener('resize', () => this.onResize());
    }

    buildNetwork() {
        const layers = this.layerSizes;
        const numLayers = layers.length;

        // We can't render 784 neurons, so we cap the visual representation
        // Show max ~20 neurons per layer, with dots indicating "more"
        const maxVisible = [20, 20, 20, 10]; // visual caps per layer

        const layerSpacing = 4;
        const totalWidth = (numLayers - 1) * layerSpacing;
        const startX = -totalWidth / 2;

        for (let l = 0; l < numLayers; l++) {
            const layerNeurons = [];
            const actualCount = layers[l];
            const visibleCount = Math.min(actualCount, maxVisible[l] || 20);
            const neuronSpacing = 0.45;
            const totalHeight = (visibleCount - 1) * neuronSpacing;
            const startY = totalHeight / 2;
            const x = startX + l * layerSpacing;

            for (let n = 0; n < visibleCount; n++) {
                const y = startY - n * neuronSpacing;
                const z = 0;

                // Sphere size — smaller for larger layers
                const radius = l === 0 ? 0.08 : (l === numLayers - 1 ? 0.18 : 0.12);

                const geometry = new THREE.SphereGeometry(radius, 16, 16);
                const material = new THREE.MeshPhongMaterial({
                    color: 0x4a5568,
                    emissive: 0x1a1a2e,
                    emissiveIntensity: 0.3,
                    transparent: true,
                    opacity: 0.8,
                });

                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(x, y, z);
                this.scene.add(mesh);
                layerNeurons.push(mesh);
            }

            this.neurons.push(layerNeurons);
        }

        // Draw connections between adjacent layers
        for (let l = 0; l < numLayers - 1; l++) {
            const layerConnections = [];
            const fromNeurons = this.neurons[l];
            const toNeurons = this.neurons[l + 1];

            // Limit connections to avoid overwhelming the scene
            const maxConns = 200;
            let connCount = 0;

            for (let f = 0; f < fromNeurons.length && connCount < maxConns; f++) {
                for (let t = 0; t < toNeurons.length && connCount < maxConns; t++) {
                    const points = [
                        fromNeurons[f].position,
                        toNeurons[t].position
                    ];

                    const geometry = new THREE.BufferGeometry().setFromPoints(points);
                    const material = new THREE.LineBasicMaterial({
                        color: 0x334155,
                        transparent: true,
                        opacity: 0.06,
                    });

                    const line = new THREE.Line(geometry, material);
                    this.scene.add(line);
                    layerConnections.push({ line, from: f, to: t });
                    connCount++;
                }
            }

            this.connections.push(layerConnections);
        }
    }

    updateActivations(allActivations) {
        // Update neuron colors based on activation values
        for (let l = 0; l < this.neurons.length; l++) {
            const neurons = this.neurons[l];
            const activations = allActivations[l];

            for (let n = 0; n < neurons.length; n++) {
                // Map neuron index to actual activation index
                const actualIdx = Math.floor(n * activations.length / neurons.length);
                const value = activations[actualIdx];

                // Clamp value to [0, 1]
                const intensity = Math.min(1, Math.max(0, value));

                const mesh = neurons[n];

                if (l === this.neurons.length - 1) {
                    // Output layer: green for high confidence
                    mesh.material.emissive.setRGB(0, intensity * 0.8, intensity * 0.3);
                    mesh.material.emissiveIntensity = 0.3 + intensity * 2;
                    mesh.material.color.setRGB(0.2, 0.3 + intensity * 0.7, 0.2 + intensity * 0.3);
                } else {
                    // Hidden layers: blue-purple glow
                    mesh.material.emissive.setRGB(intensity * 0.3, intensity * 0.2, intensity * 0.9);
                    mesh.material.emissiveIntensity = 0.2 + intensity * 1.5;
                    mesh.material.color.setRGB(0.2 + intensity * 0.2, 0.2 + intensity * 0.1, 0.3 + intensity * 0.5);
                }

                mesh.material.opacity = 0.4 + intensity * 0.6;
            }
        }

        // Update connection colors based on flow
        for (let l = 0; l < this.connections.length; l++) {
            const conns = this.connections[l];
            const fromActivations = allActivations[l];
            const toActivations = allActivations[l + 1];

            for (const conn of conns) {
                const fromIdx = Math.floor(conn.from * fromActivations.length / this.neurons[l].length);
                const toIdx = Math.floor(conn.to * toActivations.length / this.neurons[l + 1].length);

                const fromVal = Math.min(1, Math.abs(fromActivations[fromIdx]));
                const toVal = Math.min(1, Math.abs(toActivations[toIdx]));
                const flow = fromVal * toVal;

                if (flow > 0.01) {
                    // Active connection: green-to-blue gradient based on strength
                    const r = flow * 0.1;
                    const g = flow * 0.8;
                    const b = flow * 0.4;
                    conn.line.material.color.setRGB(r, g, b);
                    conn.line.material.opacity = 0.05 + flow * 0.5;
                } else {
                    // Inactive connection
                    conn.line.material.color.setHex(0x334155);
                    conn.line.material.opacity = 0.03;
                }
            }
        }
    }

    clearActivations() {
        for (const layer of this.neurons) {
            for (const mesh of layer) {
                mesh.material.emissive.setRGB(0.1, 0.1, 0.18);
                mesh.material.emissiveIntensity = 0.3;
                mesh.material.color.setHex(0x4a5568);
                mesh.material.opacity = 0.8;
            }
        }
        for (const layerConns of this.connections) {
            for (const conn of layerConns) {
                conn.line.material.color.setHex(0x334155);
                conn.line.material.opacity = 0.06;
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
