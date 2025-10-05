/**
 * Main Three.js application for 3D exoplanet system visualization
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Configuration
const API_URL = 'http://localhost:8000';
const AU_SCALE = 50; // Scale factor for visualization (1 AU = 50 units)

class ExoplanetVisualizer {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;

        // Scene objects
        this.star = null;
        this.planet = null;
        this.orbit = null;
        this.hzInner = null;
        this.hzOuter = null;
        this.hzZone = null;

        // System data
        this.systemData = null;

        // Animation
        this.animationSpeed = 1.0;
        this.currentAngle = 0;
    }

    init() {
        console.log('Initializing 3D visualization...');

        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);

        // Create camera
        const container = document.getElementById('canvas-container');
        this.camera = new THREE.PerspectiveCamera(
            60,
            container.clientWidth / container.clientHeight,
            0.1,
            10000
        );
        this.camera.position.set(100, 50, 100);

        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(this.renderer.domElement);

        // Add controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.minDistance = 10;
        this.controls.maxDistance = 500;

        // Add lights
        this.addLights();

        // Add stars background
        this.addStarfield();

        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());

        // Setup UI
        this.setupUI();

        // Hide loading indicator
        document.getElementById('loading').style.display = 'none';

        // Start animation loop
        this.animate();

        console.log('✓ 3D visualization initialized');
    }

    addLights() {
        // Ambient light
        const ambient = new THREE.AmbientLight(0x404040, 0.5);
        this.scene.add(ambient);

        // Point light at star position
        const pointLight = new THREE.PointLight(0xffffff, 2, 1000);
        pointLight.position.set(0, 0, 0);
        this.scene.add(pointLight);
    }

    addStarfield() {
        const geometry = new THREE.BufferGeometry();
        const vertices = [];

        for (let i = 0; i < 10000; i++) {
            const x = (Math.random() - 0.5) * 2000;
            const y = (Math.random() - 0.5) * 2000;
            const z = (Math.random() - 0.5) * 2000;
            vertices.push(x, y, z);
        }

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));

        const material = new THREE.PointsMaterial({
            color: 0xffffff,
            size: 0.5,
            sizeAttenuation: true
        });

        const starfield = new THREE.Points(geometry, material);
        this.scene.add(starfield);
    }

    temperatureToColor(temp) {
        // Convert stellar temperature to RGB color
        // Simplified blackbody color approximation

        if (temp < 3700) {
            return new THREE.Color(0xff6b4a); // Red (M dwarf)
        } else if (temp < 5200) {
            return new THREE.Color(0xffa040); // Orange (K dwarf)
        } else if (temp < 6000) {
            return new THREE.Color(0xfff4e0); // Yellow-white (G dwarf, like Sun)
        } else if (temp < 7500) {
            return new THREE.Color(0xffffff); // White (F star)
        } else if (temp < 10000) {
            return new THREE.Color(0xcce5ff); // Blue-white (A star)
        } else {
            return new THREE.Color(0x9bb0ff); // Blue (B star)
        }
    }

    createStar(radius, temperature) {
        // Remove old star if exists
        if (this.star) {
            this.scene.remove(this.star);
        }

        const color = this.temperatureToColor(temperature);

        // Star geometry - scale to scene units
        // 1 Solar radius = ~0.0047 AU, so use small multiplier to keep visible
        const starRadius = Math.max(radius * 2, 0.5); // Minimum 0.5 units
        const geometry = new THREE.SphereGeometry(starRadius, 32, 32);

        // Star material with glow
        const material = new THREE.MeshBasicMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: 0.8
        });

        this.star = new THREE.Mesh(geometry, material);
        this.star.position.set(0, 0, 0);
        this.scene.add(this.star);

        // Add corona glow
        const glowGeometry = new THREE.SphereGeometry(starRadius * 1.2, 32, 32);
        const glowMaterial = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.2,
            side: THREE.BackSide
        });

        const glow = new THREE.Mesh(glowGeometry, glowMaterial);
        this.star.add(glow);

        console.log(`Created star: radius=${radius.toFixed(2)} R☉, temp=${temperature}K`);
    }

    createPlanet(radius, semiMajorAxis, color = 0x4a90e2) {
        // Remove old planet if exists
        if (this.planet) {
            this.scene.remove(this.planet);
        }

        // Planet geometry - EXAGGERATED for visibility (NOT to scale!)
        // Educational visualization: make planets visible while keeping them smaller than stars
        // Scale: 1 R⊕ ≈ 0.4 units (exaggerated ~1000x from reality for visibility)
        const planetRadius = Math.max(radius * 0.4, 0.3); // Minimum 0.3 units for Earth-sized
        const geometry = new THREE.SphereGeometry(planetRadius, 32, 32);

        // Planet material
        const material = new THREE.MeshStandardMaterial({
            color: color,
            metalness: 0.3,
            roughness: 0.7
        });

        this.planet = new THREE.Mesh(geometry, material);

        // Position at aphelion initially
        const distance = semiMajorAxis * AU_SCALE;
        this.planet.position.set(distance, 0, 0);

        this.scene.add(this.planet);

        console.log(`Created planet: radius=${radius.toFixed(2)} R⊕, orbit=${semiMajorAxis.toFixed(3)} AU`);
    }

    createOrbit(semiMajorAxis) {
        // Remove old orbit if exists
        if (this.orbit) {
            this.scene.remove(this.orbit);
        }

        const distance = semiMajorAxis * AU_SCALE;

        // Create orbit line
        const curve = new THREE.EllipseCurve(
            0, 0,           // center
            distance, distance,  // radii (circular orbit)
            0, 2 * Math.PI, // start/end angle
            false,          // clockwise
            0               // rotation
        );

        const points = curve.getPoints(100);
        const geometry = new THREE.BufferGeometry().setFromPoints(points);

        const material = new THREE.LineBasicMaterial({
            color: 0x4fc3f7,
            transparent: true,
            opacity: 0.5
        });

        this.orbit = new THREE.Line(geometry, material);
        this.orbit.rotation.x = Math.PI / 2; // Rotate to XZ plane
        this.scene.add(this.orbit);
    }

    createHabitableZone(hzInner, hzOuter) {
        // Remove old HZ visualization if exists
        if (this.hzInner) this.scene.remove(this.hzInner);
        if (this.hzOuter) this.scene.remove(this.hzOuter);
        if (this.hzZone) this.scene.remove(this.hzZone);

        const innerDist = hzInner * AU_SCALE;
        const outerDist = hzOuter * AU_SCALE;

        // Inner boundary ring
        const innerGeometry = new THREE.RingGeometry(innerDist - 0.5, innerDist + 0.5, 64);
        const innerMaterial = new THREE.MeshBasicMaterial({
            color: 0xff6b6b,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.6
        });
        this.hzInner = new THREE.Mesh(innerGeometry, innerMaterial);
        this.hzInner.rotation.x = Math.PI / 2;
        this.scene.add(this.hzInner);

        // Outer boundary ring
        const outerGeometry = new THREE.RingGeometry(outerDist - 0.5, outerDist + 0.5, 64);
        const outerMaterial = new THREE.MeshBasicMaterial({
            color: 0x4ecdc4,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.6
        });
        this.hzOuter = new THREE.Mesh(outerGeometry, outerMaterial);
        this.hzOuter.rotation.x = Math.PI / 2;
        this.scene.add(this.hzOuter);

        // Habitable zone band
        const zoneGeometry = new THREE.RingGeometry(innerDist, outerDist, 64);
        const zoneMaterial = new THREE.MeshBasicMaterial({
            color: 0x4caf50,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.15
        });
        this.hzZone = new THREE.Mesh(zoneGeometry, zoneMaterial);
        this.hzZone.rotation.x = Math.PI / 2;
        this.scene.add(this.hzZone);

        console.log(`Created HZ: ${hzInner.toFixed(3)} - ${hzOuter.toFixed(3)} AU`);
    }

    async updateSystem(params) {
        console.log('Updating system...', params);

        try {
            // Call API to get system parameters
            const response = await fetch(`${API_URL}/api/predict-system`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(params)
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            this.systemData = await response.json();
            console.log('System data received:', this.systemData);

            // Update 3D visualization
            this.renderSystem();

            // Update UI
            this.updateUI();

        } catch (error) {
            console.error('Error updating system:', error);
            alert('Error connecting to API. Make sure the backend server is running on port 8000.');
        }
    }

    renderSystem() {
        const data = this.systemData;

        // Create star
        this.createStar(
            data.system_parameters.stellar.radius_rsun,
            data.system_parameters.stellar.teff_K
        );

        // Create planet
        const planetColor = this.getPlanetColor(data.planet_type);
        this.createPlanet(
            data.system_parameters.planet.radius_rearth,
            data.system_parameters.orbit.semi_major_axis_au,
            planetColor
        );

        // Create orbit
        this.createOrbit(data.system_parameters.orbit.semi_major_axis_au);

        // Create habitable zone
        this.createHabitableZone(
            data.system_parameters.habitability.hz_inner_au,
            data.system_parameters.habitability.hz_outer_au
        );
    }

    getPlanetColor(planetType) {
        const colors = {
            'Rocky Planet': 0x8b7355,      // Brown/rocky
            'Super-Earth': 0x4a90e2,       // Blue
            'Mini-Neptune': 0x6a9ee8,      // Light blue
            'Neptune-like': 0x4a7ba7,      // Deep blue
            'Gas Giant': 0xe8c547          // Yellowish
        };

        return colors[planetType] || 0x4a90e2;
    }

    updateUI() {
        const data = this.systemData;

        // ML Prediction
        if (data.prediction) {
            const isPlanet = data.prediction.is_planet;
            const mlIsPlanetEl = document.getElementById('ml-is-planet');
            mlIsPlanetEl.innerHTML = `<span class="status-indicator ${isPlanet ? 'status-good' : 'status-bad'}"></span>${isPlanet ? 'YES' : 'NO'}`;

            document.getElementById('ml-probability').textContent =
                `${(data.prediction.probability * 100).toFixed(1)}%`;
            document.getElementById('ml-confidence').textContent =
                `${(data.prediction.confidence * 100).toFixed(1)}%`;

            // Individual model votes
            if (data.prediction.individual_models) {
                const models = data.prediction.individual_models;
                for (const [modelName, probability] of Object.entries(models)) {
                    const elementId = `vote-${modelName}`;
                    const element = document.getElementById(elementId);
                    if (element) {
                        const percent = (probability * 100).toFixed(1);
                        const isPlanetVote = probability > 0.5;
                        element.innerHTML = `<span class="status-indicator ${isPlanetVote ? 'status-good' : 'status-bad'}"></span>${percent}%`;
                    }
                }
            }

            // SHAP features
            if (data.top_features) {
                const shapContainer = document.getElementById('shap-features');
                shapContainer.innerHTML = '';

                data.top_features.slice(0, 5).forEach(feature => {
                    const div = document.createElement('div');
                    div.style.fontSize = '11px';
                    div.style.marginBottom = '5px';
                    div.innerHTML = `<span style="color: #81c784;">${feature.feature}:</span> <span style="color: #fff;">${feature.importance.toFixed(3)}</span>`;
                    shapContainer.appendChild(div);
                });
            }
        }

        // Planet info
        document.getElementById('planet-type').textContent = data.planet_type;
        document.getElementById('planet-radius').textContent =
            `${data.system_parameters.planet.radius_rearth.toFixed(2)} R⊕`;
        document.getElementById('planet-mass').textContent =
            `${data.system_parameters.planet.mass_mearth.toFixed(2)} M⊕`;
        document.getElementById('planet-temp').textContent =
            `${data.system_parameters.planet.equilibrium_temp_K.toFixed(0)} K (${data.temperature_category})`;
        document.getElementById('planet-density').textContent =
            `${data.validation.density_g_cm3.toFixed(2)} g/cm³`;

        // Orbit info
        document.getElementById('orbit-sma').textContent =
            `${data.system_parameters.orbit.semi_major_axis_au.toFixed(3)} AU`;
        document.getElementById('orbit-period').textContent =
            `${data.system_parameters.orbit.period_days.toFixed(1)} days`;
        document.getElementById('orbit-velocity').textContent =
            `${data.system_parameters.orbit.orbital_velocity_km_s.toFixed(1)} km/s`;

        // Habitability
        const inHz = data.system_parameters.habitability.in_hz_conservative;
        const hzStatusEl = document.getElementById('hz-status');
        hzStatusEl.innerHTML = `<span class="status-indicator ${inHz ? 'status-good' : 'status-bad'}"></span>${inHz ? 'Yes' : 'No'}`;

        document.getElementById('hz-range').textContent =
            `${data.system_parameters.habitability.hz_inner_au.toFixed(3)} - ${data.system_parameters.habitability.hz_outer_au.toFixed(3)} AU`;
        document.getElementById('hz-distance').textContent =
            `${data.system_parameters.habitability.distance_from_hz_au.toFixed(3)} AU`;

        // Validation
        const validated = data.validation.validated;
        const validationEl = document.getElementById('validation-status');
        validationEl.innerHTML = `<span class="status-indicator ${validated ? 'status-good' : 'status-warning'}"></span>${validated ? 'Pass' : 'Warning'}`;

        const messagesEl = document.getElementById('validation-messages');
        messagesEl.innerHTML = '';

        if (data.validation.errors && data.validation.errors.length > 0) {
            data.validation.errors.forEach(error => {
                const div = document.createElement('div');
                div.style.color = '#f44336';
                div.textContent = `❌ ${error}`;
                messagesEl.appendChild(div);
            });
        }

        if (data.validation.warnings && data.validation.warnings.length > 0) {
            data.validation.warnings.forEach(warning => {
                const div = document.createElement('div');
                div.style.color = '#ff9800';
                div.textContent = `⚠️  ${warning}`;
                messagesEl.appendChild(div);
            });
        }

        if (validated && (!data.validation.warnings || data.validation.warnings.length === 0)) {
            const div = document.createElement('div');
            div.style.color = '#4caf50';
            div.textContent = '✓ All checks passed';
            messagesEl.appendChild(div);
        }
    }

    getDemoExamples() {
        return {
            'kepler-442b': {
                name: 'Kepler-442b',
                period_days: 112.3,
                transit_depth_ppm: 376,
                transit_duration_hrs: 4.2,
                stellar_teff: 4402,
                stellar_radius: 0.601,
                stellar_logg: 4.653,
                description: 'Habitable zone super-Earth around K-dwarf'
            },
            'earth': {
                name: 'Earth',
                period_days: 365.25,
                transit_depth_ppm: 84,
                transit_duration_hrs: 13.0,
                stellar_teff: 5778,
                stellar_radius: 1.0,
                stellar_logg: 4.44,
                description: 'Our home planet - weak signal at detection limit'
            },
            'hot-jupiter': {
                name: 'Hot Jupiter Example',
                period_days: 3.5,
                transit_depth_ppm: 15000,
                transit_duration_hrs: 3.2,
                stellar_teff: 6100,
                stellar_radius: 1.2,
                stellar_logg: 4.3,
                description: 'Close-in gas giant with deep transit'
            },
            'mini-neptune': {
                name: 'Mini-Neptune Example',
                period_days: 24.0,
                transit_depth_ppm: 1200,
                transit_duration_hrs: 5.0,
                stellar_teff: 5200,
                stellar_radius: 0.9,
                stellar_logg: 4.5,
                description: 'Small gas planet in temperate zone'
            }
        };
    }

    async loadDemoTargets() {
        try {
            const response = await fetch(`${API_URL}/api/targets`);
            const data = await response.json();

            const selector = document.getElementById('demo-selector');
            const optgroup = selector.querySelector('optgroup[label="Test Set Examples"]');

            data.targets.forEach(target => {
                const option = document.createElement('option');
                option.value = `api-${target.id}`;
                option.textContent = `${target.name} - ${target.disposition} (${target.period_days.toFixed(1)}d, ${target.radius_rearth.toFixed(2)}R⊕)`;
                option.dataset.target = JSON.stringify(target);
                optgroup.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to load demo targets:', error);
        }
    }

    setupUI() {
        // Load demo targets
        this.loadDemoTargets();

        // Demo selector
        document.getElementById('demo-selector').addEventListener('change', async (e) => {
            const selectedValue = e.target.value;
            console.log('Demo selector changed:', selectedValue);

            if (!selectedValue) return;

            const demoExamples = this.getDemoExamples();
            console.log('Available demo examples:', Object.keys(demoExamples));

            // Check if it's a hardcoded example
            if (demoExamples[selectedValue]) {
                const example = demoExamples[selectedValue];
                console.log('Loading hardcoded example:', example.name, example);

                document.getElementById('period').value = example.period_days;
                document.getElementById('depth').value = example.transit_depth_ppm;
                document.getElementById('duration').value = example.transit_duration_hrs;
                document.getElementById('stellar-teff').value = example.stellar_teff;
                document.getElementById('stellar-radius').value = example.stellar_radius;
                document.getElementById('stellar-logg').value = example.stellar_logg;

                console.log('Input fields updated, triggering system update...');

                // Trigger update
                document.getElementById('update-btn').click();
                return;
            }

            // Handle API targets (prefixed with 'api-')
            const selectedOption = e.target.options[e.target.selectedIndex];
            if (!selectedOption.dataset.target) return;

            const target = JSON.parse(selectedOption.dataset.target);

            try {
                // Update input fields from API target
                document.getElementById('period').value = target.period_days || 112.3;
                document.getElementById('depth').value = target.transit_depth_ppm || 376;
                document.getElementById('duration').value = target.transit_duration_hrs || 4.2;
                document.getElementById('stellar-teff').value = target.stellar_teff || 4402;
                document.getElementById('stellar-radius').value = target.stellar_radius || 0.601;
                document.getElementById('stellar-logg').value = target.stellar_logg || 4.653;

                // Trigger update
                document.getElementById('update-btn').click();
            } catch (error) {
                console.error('Failed to load target details:', error);
            }
        });

        // Update button
        document.getElementById('update-btn').addEventListener('click', () => {
            const params = {
                period_days: parseFloat(document.getElementById('period').value),
                transit_depth_ppm: parseFloat(document.getElementById('depth').value),
                transit_duration_hrs: parseFloat(document.getElementById('duration').value),
                stellar_teff: parseFloat(document.getElementById('stellar-teff').value),
                stellar_radius: parseFloat(document.getElementById('stellar-radius').value),
                stellar_logg: parseFloat(document.getElementById('stellar-logg').value),
                impact_parameter: 0.5
            };

            this.updateSystem(params);
        });

        // Reset view button
        document.getElementById('reset-view-btn').addEventListener('click', () => {
            this.camera.position.set(100, 50, 100);
            this.controls.target.set(0, 0, 0);
            this.controls.update();
        });

        // Load initial system (Kepler-442b)
        this.updateSystem({
            period_days: 112.3,
            transit_depth_ppm: 376,
            transit_duration_hrs: 4.2,
            stellar_teff: 4402,
            stellar_radius: 0.601,
            stellar_logg: 4.653,
            impact_parameter: 0.3
        });
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        // Update controls
        this.controls.update();

        // Animate planet orbit
        if (this.planet && this.systemData) {
            const period = this.systemData.system_parameters.orbit.period_days;
            const sma = this.systemData.system_parameters.orbit.semi_major_axis_au * AU_SCALE;

            // Update angle (speed adjusted for visualization)
            this.currentAngle += (0.01 / period) * this.animationSpeed;

            // Update planet position
            this.planet.position.x = sma * Math.cos(this.currentAngle);
            this.planet.position.z = sma * Math.sin(this.currentAngle);

            // Rotate planet
            this.planet.rotation.y += 0.01;
        }

        // Rotate star slowly
        if (this.star) {
            this.star.rotation.y += 0.001;
        }

        // Render scene
        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        const container = document.getElementById('canvas-container');
        this.camera.aspect = container.clientWidth / container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(container.clientWidth, container.clientHeight);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const app = new ExoplanetVisualizer();
    app.init();
});
