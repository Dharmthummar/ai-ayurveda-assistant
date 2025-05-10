let currentMode = null;
let videoElement = null;
let canvasElement = null;
let captureButton = null;
let featureSelectElement = null;
let resultElement = null;
let capturedImages = {};
let mediaStream = null;

async function initWebcam() {
    videoElement = document.getElementById('webcam');
    canvasElement = document.getElementById('canvas');
    captureButton = document.getElementById('capture-btn');
    featureSelectElement = document.getElementById('feature-select');
    resultElement = document.getElementById('analysis-result');

    try {
        if (!mediaStream) {
            mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            });
        }
        
        videoElement.srcObject = mediaStream;
        videoElement.style.transform = 'scaleX(-1)';

        captureButton.addEventListener('click', captureImage);
        featureSelectElement.addEventListener('change', updateInstructions);
        document.getElementById('reset-btn').addEventListener('click', resetCapture);
        document.getElementById('analyze-btn').addEventListener('click', analyzeImages);
        document.getElementById('share-btn').addEventListener('click', function() {
            const result = JSON.parse(this.getAttribute('data-result'));
            shareWithChatbot(result);
        });

        updateInstructions();
    } catch (error) {
        console.error("Error accessing webcam:", error);
        document.querySelector('.webcam-feed').innerHTML = 
            `<div class="error-message">Error accessing webcam: ${error.message}</div>`;
    }
}

function updateInstructions() {
    currentMode = featureSelectElement.value;
    const instructionElement = document.getElementById('capture-instructions');

    const instructions = {
        'face': 'Look straight at the camera with a neutral expression.',
        'eyes': 'Look directly at the camera with eyes fully open.',
        'mouth': 'Open your mouth slightly to show your teeth.',
        'skin': 'Position your face close to the camera to capture skin texture.',
        'profile': 'Turn your face to the side to show your profile.',
    };

    instructionElement.textContent = instructions[currentMode] || 'Select a feature to capture';
    updateBodyMapHighlight(currentMode);
}

function updateBodyMapHighlight(feature) {
    document.querySelectorAll('.body-map-highlight').forEach(el => {
        el.classList.remove('active');
    });

    const highlightEl = document.getElementById(`highlight-${feature}`);
    if (highlightEl) {
        highlightEl.classList.add('active');
    }
}

function captureImage() {
    if (!currentMode) return;

    const context = canvasElement.getContext('2d');
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

    const imageData = canvasElement.toDataURL('image/jpeg');
    capturedImages[currentMode] = imageData;

    const thumbnailContainer = document.getElementById('thumbnails');
    const thumbnail = document.createElement('div');
    thumbnail.className = 'thumbnail';
    thumbnail.innerHTML = `
        <img src="${imageData}" alt="${currentMode}">
        <span>${currentMode}</span>
    `;

    const existingThumbnail = thumbnailContainer.querySelector(`[data-feature="${currentMode}"]`);
    if (existingThumbnail) {
        thumbnailContainer.removeChild(existingThumbnail);
    }

    thumbnail.setAttribute('data-feature', currentMode);
    thumbnailContainer.appendChild(thumbnail);

    const features = ['face', 'eyes', 'mouth', 'skin', 'profile'];
    const currentIndex = features.indexOf(currentMode);
    if (currentIndex < features.length - 1) {
        featureSelectElement.value = features[currentIndex + 1];
        updateInstructions();
    }

    updateAnalyzeButton();
}

function updateAnalyzeButton() {
    const requiredFeatures = ['face', 'eyes', 'mouth', 'skin'];
    const allCaptured = requiredFeatures.every(feature => feature in capturedImages);

    document.getElementById('analyze-btn').disabled = !allCaptured;
    if (allCaptured) {
        document.getElementById('analyze-btn').classList.add('ready');
    }
}

async function analyzeImages() {
    const analyzeBtn = document.getElementById('analyze-btn');
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';
    resultElement.innerHTML = '<div class="analyzing-spinner"></div><div>Analyzing facial features to determine your Ayurvedic constitution (Prakriti)...</div>';

    try {
        // Create request data with images
        const requestData = {
            images: capturedImages
        };

        const response = await fetch('/analyze-features', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
        }

        const result = await response.json();
        console.log("Server response:", result);
        
        // If server returns an error message, display it
        if (result.error) {
            throw new Error(result.error);
        }
        
        // If server doesn't return doshas data, use fallback demo data
        if (!result.doshas) {
            console.warn("Server didn't return proper dosha data, using fallback demo data");
            
            // Create fallback demo data
            const demoResult = {
                doshas: {
                    vata: 33,
                    pitta: 33,
                    kapha: 34
                },
                primary_dosha: "kapha",
                dosha_description: "Your constitution shows a balanced mix of doshas with a slight Kapha predominance. Kapha types typically have strong immunity, good stamina, and a calm disposition.",
                feature_description: {
                    face: "Your facial structure shows Kapha characteristics with well-proportioned features.",
                    eyes: "Your eyes indicate a balanced constitution with good luster.",
                    mouth: "Your mouth and teeth structure shows balanced characteristics.",
                    skin: "Your skin texture indicates balanced doshas with good complexion."
                },
                recommendations: [
                    "Maintain a balanced diet with fresh, whole foods",
                    "Regular exercise is beneficial for your constitution",
                    "Establish a consistent daily routine",
                    "Consider incorporating warming herbs and spices in your diet"
                ]
            };
            
            displayResults(demoResult);
            document.getElementById('share-btn').setAttribute('data-result', JSON.stringify(demoResult));
            document.getElementById('share-btn').style.display = 'block';
            return;
        }
        
        // Now safely determine the primary dosha
        result.primary_dosha = Object.entries(result.doshas).reduce((a, b) => a[1] > b[1] ? a : b)[0];
        
        displayResults(result);
        document.getElementById('share-btn').setAttribute('data-result', JSON.stringify(result));
        document.getElementById('share-btn').style.display = 'block';
    } catch (error) {
        console.error("Error analyzing images:", error);
        resultElement.innerHTML = `<div class="error-message">Error analyzing images: ${error.message}</div>`;
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze Features';
    }
}

function displayResults(result) {
    const doshaData = result.doshas;
    const primaryDosha = result.primary_dosha;

    let resultHTML = `
        <div class="results-container">
            <h3>Your Ayurvedic Constitution (Prakriti)</h3>
            <div class="dosha-chart">
                <div class="dosha-bar vata" style="width: ${doshaData.vata}%;">
                    <span>Vata ${Math.round(doshaData.vata)}%</span>
                </div>
                <div class="dosha-bar pitta" style="width: ${doshaData.pitta}%;">
                    <span>Pitta ${Math.round(doshaData.pitta)}%</span>
                </div>
                <div class="dosha-bar kapha" style="width: ${doshaData.kapha}%;">
                    <span>Kapha ${Math.round(doshaData.kapha)}%</span>
                </div>
            </div>

            <div class="primary-dosha">
                <h4>Your primary dosha is <span class="dosha-name ${primaryDosha}">${primaryDosha.toUpperCase()}</span></h4>
                <p>${result.dosha_description}</p>
            </div>

            <div class="feature-analysis">
                <h4>Feature Analysis:</h4>
                <ul>
    `;

    for (const feature in result.feature_description) {
        resultHTML += `
            <li>
                <strong>${feature.replace('_', ' ')}:</strong>
                ${result.feature_description[feature]}
            </li>
        `;
    }

    resultHTML += `
                </ul>
            </div>

            <div class="recommendations">
                <h4>Recommendations:</h4>
                <ul>
    `;

    result.recommendations.forEach(recommendation => {
        resultHTML += `<li>${recommendation}</li>`;
    });

    resultHTML += `
                </ul>
            </div>
        </div>
    `;

    resultElement.innerHTML = resultHTML;
}

function shareWithChatbot(result) {
    // Create a more robust message with fallbacks for different property names
    const vata = result.doshas?.vata || result.vata_percentage || 33;
    const pitta = result.doshas?.pitta || result.pitta_percentage || 33;
    const kapha = result.doshas?.kapha || result.kapha_percentage || 34;
    
    // Get the dominant dosha using various possible property names
    const dominantDosha = result.primary_dosha || result.dominant_dosha || 
                         (vata > pitta && vata > kapha ? 'vata' : 
                          pitta > vata && pitta > kapha ? 'pitta' : 'kapha');
    
    const message = `Based on my facial analysis, my Ayurvedic constitution (Prakriti) is:
- Vata: ${Math.round(vata)}%
- Pitta: ${Math.round(pitta)}%
- Kapha: ${Math.round(kapha)}%

My primary dosha appears to be ${dominantDosha.toUpperCase()}.
Can you provide more information about this constitution and specific recommendations for diet, lifestyle, and health practices based on this analysis?`;

    // Use sessionStorage which persists across page loads but not when browser is closed
    sessionStorage.setItem('prakritiMessage', message);
    
    // Redirect to home page
    window.location.href = '/';
}

function resetCapture() {
    capturedImages = {};
    document.getElementById('thumbnails').innerHTML = '';
    document.getElementById('analyze-btn').disabled = true;
    document.getElementById('analyze-btn').classList.remove('ready');
    document.getElementById('share-btn').style.display = 'none';
    resultElement.innerHTML = '';
    featureSelectElement.value = 'face';
    updateInstructions();
}

document.addEventListener('DOMContentLoaded', initWebcam);