// Global variables
let dinoModel = null;
let imageDataUrl = null;
let cachedImageVectors = null;
let searchInProgress = false;
let currentSearchAbortController = null;
let currentThreshold = 80; // Default threshold (lowered for DINO + model differences)
let currentSearchClusters = 10; // Default number of clusters to search
const MAX_CLUSTERS = 50;

// Add pako for gzip decompression
// import pako from 'https://cdn.jsdelivr.net/npm/pako@2.1.0/dist/pako.mjs';

// Add affiliate links mapping
const affiliateLinks = {
    // Example tags
    "cat": ["https://jlist.com/shop/product/neko-musume-catgirl-pretty-h-tamapremium","https://jlist.com/shop/product/Mesuneko-Servant-----Catgirl-Servant"],
    "loli": ["https://jlist.com/shop/dohna-dohna","https://jlist.com/shop/product/Innocent-CQ-2"],
    "virgin": ["https://jlist.com/shop/product/Innocent-CQ-2","https://jlist.com/shop/product/Virgin-Matchless"],
    "Anal": ["https://peachesandscreams.co.uk/collections/anal-range"]
    //"pet": "https://amzn.to/petaccessories",
    // Add more as needed
};

// Synonyms mapping: alternate terms to canonical keys
const affiliateSynonyms = {
    "lolicon": "loli",
    "Small": "loli",
    "School": "loli",
    "Schoolgirl": "loli",
    "Defloration": "virgin",
    "Anal": "anal",
    "Ass": "anal"
    // Add more: "alt_term": "canonical_term"
};

function getAffiliateUrls(tag) {
    if (affiliateLinks[tag]) return affiliateLinks[tag];
    if (affiliateSynonyms[tag] && affiliateLinks[affiliateSynonyms[tag]]) {
        return affiliateLinks[affiliateSynonyms[tag]];
    }
    return null;
}

function clamp(val) {
    return Math.max(1, Math.min(MAX_CLUSTERS, parseInt(val) || 1));
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    loadDinoModel();
    setupEventListeners();
    initializeUI();
    preloadImageVectors(); // Preload vectors for faster search
});

// Initialize UI state
function initializeUI() {
    updateResultsInfo(0);
    hideClearButton();
    setupThresholdControl();
    setupClusterSlider();
}

// Setup threshold control with both slider and input field
function setupThresholdControl() {
    const thresholdSlider = document.getElementById('thresholdSlider');
    const thresholdInput = document.getElementById('thresholdInput');
    const thresholdValue = document.getElementById('thresholdValue');
    let debounceTimer;
    
    // Function to update threshold value
    function updateThreshold(value) {
        currentThreshold = Math.max(10, Math.min(100, parseInt(value) || 80));
        thresholdSlider.value = currentThreshold;
        thresholdInput.value = currentThreshold;
        thresholdValue.textContent = `${currentThreshold}%`;
    }
    
    // Slider event
    thresholdSlider.addEventListener('input', (e) => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            updateThreshold(e.target.value);
        }, 100);
    });
    
    // Input field event
    thresholdInput.addEventListener('input', (e) => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            updateThreshold(e.target.value);
        }, 100);
    });
    
    // Initialize with default value
    updateThreshold(80);
}

// Setup cluster slider
function setupClusterSlider() {
    const clusterSlider = document.getElementById('clusterSlider');
    const clusterInput = document.getElementById('clusterInput');
    const clusterValue = document.getElementById('clusterValue');

    clusterSlider.max = MAX_CLUSTERS;
    clusterInput.max = MAX_CLUSTERS;

    clusterSlider.addEventListener('input', (e) => {
        currentSearchClusters = clamp(e.target.value);
        clusterSlider.value = currentSearchClusters;
        clusterInput.value = currentSearchClusters;
        clusterValue.textContent = `${currentSearchClusters}`;
    });
    clusterInput.addEventListener('input', (e) => {
        currentSearchClusters = clamp(e.target.value);
        clusterSlider.value = currentSearchClusters;
        clusterInput.value = currentSearchClusters;
        clusterValue.textContent = `${currentSearchClusters}`;
    });

    currentSearchClusters = clamp(10);
}

// Preload image vectors for faster search
async function preloadImageVectors() {
    try {
        console.log('Preloading image vectors...');
        await loadImageVectors();
        console.log('Image vectors preloaded successfully');
    } catch (error) {
        console.warn('Failed to preload image vectors:', error);
    }
}

// No-op: all cluster vectors are loaded on demand in searchByClusters
async function loadImageVectors() {
    return;
}

// Load DINO model from CDN with better error handling and retry logic
async function loadDinoModel() {
    const maxRetries = 3;
    let retryCount = 0;
    
    while (retryCount < maxRetries) {
        try {
            updateStatus('Loading DINO model...', 'loading');
            console.log(`Loading DINO model (attempt ${retryCount + 1}/${maxRetries})`);
            
            // Load transformers library (v2.17.2, latest available)
            const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.6.1');
            console.log('Transformers library loaded');
            
            // Use image-feature-extraction pipeline for DINO
            dinoModel = await pipeline('image-feature-extraction', 'Xenova/dino-vits16');
            console.log('DINO model loaded with image-feature-extraction pipeline');
            
            updateStatus('DINO model loaded! Ready to find SOURCE.', 'success');
            document.getElementById('searchBtn').disabled = false;
            return;
        } catch (error) {
            retryCount++;
            console.error(`Error loading DINO model (attempt ${retryCount}):`, error);
            
            if (retryCount >= maxRetries) {
                updateStatus(`Failed to load DINO model after ${maxRetries} attempts. Please refresh the page.`, 'error');
            } else {
                updateStatus(`Loading failed, retrying... (${retryCount}/${maxRetries})`, 'error');
                await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds before retry
            }
        }
    }
}

// Setup event listeners for user interactions
function setupEventListeners() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const searchBtn = document.getElementById('searchBtn');
    const clearBtn = document.getElementById('clearBtn');

    uploadArea.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            if (file.type.startsWith('image/')) {
                handleFile(file);
            } else {
                updateStatus('Please select a valid image file', 'error');
            }
        }
    });

    searchBtn.addEventListener('click', performSearch);
    clearBtn.addEventListener('click', clearAll);
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey) {
            if (e.key === 'Enter' && !searchBtn.disabled) {
                e.preventDefault();
                performSearch();
            }
        }
    });
}

// Handle file selection with better validation and optimization
function handleFile(file) {
    // Validate file size (max 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
        updateStatus('File too large. Please select an image smaller than 10MB.', 'error');
        return;
    }

    // Validate file type more strictly
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
        updateStatus('Please select a valid image file (JPG, PNG, GIF, WebP)', 'error');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        imageDataUrl = e.target.result;
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.innerHTML = `
            <img src="${imageDataUrl}" alt="Selected image">
            <input type="file" id="fileInput" accept="image/*" style="display: none;">
        `;
        uploadArea.classList.add('has-image');
        document.getElementById('searchBtn').disabled = false;
        showClearButton();
        updateStatus('Image loaded successfully! Click "Find SOURCE" to search.', 'success');
    };
    
    reader.onerror = () => {
        updateStatus('Error reading file. Please try again.', 'error');
    };
    
    reader.readAsDataURL(file);
}

// Clear all data and reset UI
function clearAll() {
    imageDataUrl = null;
    const uploadArea = document.getElementById('uploadArea');
    uploadArea.innerHTML = `
        <div class="upload-icon">📁</div>
        <p class="upload-text">Click to select image</p>
        <p class="upload-hint">Supports: JPG, PNG, GIF, WebP</p>
        <input type="file" id="fileInput" accept="image/*" style="display: none;">
    `;
    uploadArea.classList.remove('has-image');
    document.getElementById('searchBtn').disabled = true;
    hideClearButton();
    
    // Clear results and cache
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = `
        <div class="placeholder">
            <div class="placeholder-icon">📄</div>
            <div class="placeholder-title">No search performed yet</div>
            <div class="placeholder-subtitle">Upload an image and click "Find SOURCE" to search</div>
        </div>
    `;
    updateResultsInfo(0);
    updateStatus('', '');
}

// Show/hide clear button
function showClearButton() {
    document.getElementById('clearBtn').style.display = 'flex';
}

function hideClearButton() {
    document.getElementById('clearBtn').style.display = 'none';
}

// Perform the search operation with abort capability and optimization
function performSearch() {
    if (!imageDataUrl || !dinoModel) {
        updateStatus('Please select an image first', 'error');
        return;
    }
    
    if (searchInProgress) {
        // Abort current search
        if (currentSearchAbortController) {
            currentSearchAbortController.abort();
        }
        return;
    }
    
    searchInProgress = true;
    currentSearchAbortController = new AbortController();
    
    updateStatus('Processing image with DINO...', 'loading');
    document.getElementById('searchBtn').innerHTML = `
        <span class="btn-icon">⏹️</span>
        <span class="btn-text">Cancel Search</span>
    `;
    
    // Start timing
    window._searchStartTime = performance.now();
    handleSearchResult(imageDataUrl);
}

// Assumes normalized input vectors
function computeHash(vector, hyperplanes) {
  const hash = [];
  for (let plane of hyperplanes) {
    let dot = 0;
    for (let i = 0; i < vector.length; i++) {
      dot += vector[i] * plane[i];
    }
    hash.push(dot > 0 ? '1' : '0');
  }
  return hash.join('');
}



function getBatchIndex(codeId, batchSize = 500) {
  return Math.floor(codeId / batchSize);
}

async function handleSearchResult(imageDataUrl) {
    try {
        updateStatus('Processing image with DINO...', 'loading');

        // Load the image from data URL
        const img = new Image();
        await new Promise((resolve, reject) => {
            img.onload = () => resolve();
            img.onerror = () => reject(new Error('Failed to load image'));
            img.src = imageDataUrl;
        });

        // Create canvas and resize to 256x256
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 256;
        const ctx = canvas.getContext('2d');

        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";

        ctx.drawImage(img, 0, 0, 256, 256);
        console.log('Image resized to 256x256');

        // Get image data
        const imageData = ctx.getImageData(0, 0, 256, 256);
        const pixels = imageData.data;

        // Pass image directly to DINO model (processor handles normalization)
        const result = await dinoModel(canvas.toDataURL('image/png'));
        console.log('DINO processing completed');

        const embedding = Array.from(result.data ?? []);
        if (!embedding.length) throw new Error('DINO model did not return a valid embedding');

        // Normalize embedding vector
        const normalizedEmbedding = l2Normalize(embedding);

        // Ensure vector length 384 with zero padding if needed
        const finalEmbedding = new Array(384).fill(0);
        for (let i = 0; i < Math.min(normalizedEmbedding.length, 384); i++) {
            finalEmbedding[i] = normalizedEmbedding[i];
        }
        console.log('Using original embedding with zero padding');

        updateStatus('Searching for sources...', 'loading');

        const searchStart = performance.now();
        const { results: similarities, totalCompared, clusterIds } = await searchByClusters(finalEmbedding);
        const searchElapsed = performance.now() - searchStart;

        // Get top result metadata
        let mostLikely = null;
        
        if (similarities.length > 0) {
            const topResult = similarities[0];
            const code = topResult.code;
            const batch_Id = getBatchIndex(code)
            const metadata = await fetchClusterMetaBinary(`meta/meta_${batch_Id}.bin.gz`);
            const match = metadata.find(m => m.code === topResult.code);
            if (match) {
                mostLikely = { ...match, similarity: topResult.similarity, code: match.code, page: match.page };
            }
        }
	
	
        await displayResults(similarities, mostLikely);
        updateStatus(`Search completed! Found ${similarities.length} sources (compared against ${totalCompared} images) in ${(searchElapsed/1000).toFixed(2)}s.`, 'success');
        resetSearchButton();

    } catch (error) {
        console.error('Search error:', error);
        updateStatus(`Search error: ${error.message}`, 'error');
        resetSearchButton();
    }
}


// Reset search button to normal state
function resetSearchButton() {
    searchInProgress = false;
    currentSearchAbortController = null;
    document.getElementById('searchBtn').innerHTML = `
        <span class="btn-text">Find SOURCE</span>
    `;
    document.getElementById('searchBtn').disabled = false;
}

// L2 normalization for embedding vectors (optimized)
function l2Normalize(vec) {
    const norm = Math.sqrt(vec.reduce((sum, x) => sum + x * x, 0));
    if (norm === 0) return vec.map(() => 0);
    return vec.map(x => x / norm);
}

// Validate vector format
function isValidVector(vector) {
    return Array.isArray(vector) && vector.every(x => typeof x === 'number' && !isNaN(x));
}

// Calculate similarities with simple, straightforward approach
function calculateSimilarities(queryEmbedding, imageVectors) {
    const t0 = performance.now();
    const similarities = [];
    const threshold = currentThreshold / 100;
    if (!Array.isArray(imageVectors)) return similarities;
    
    // Normalize query embedding
    const queryNorm = l2Normalize(queryEmbedding);
    const queryArr = new Float32Array(queryNorm);
    
    for (let index = 0; index < imageVectors.length; index++) {
        const item = imageVectors[index];
        if (!item || !Array.isArray(item.vector) || typeof item.vector[0] !== 'number') continue;
        
        // Normalize item vector
        const itemNorm = l2Normalize(item.vector);
        const vectorArr = new Float32Array(itemNorm);
        
        // Calculate cosine similarity (dot product of normalized vectors)
        let dotProduct = 0;
        for (let i = 0; i < queryArr.length; i += 4) {
            dotProduct += queryArr[i] * vectorArr[i]
                + (queryArr[i+1] || 0) * (vectorArr[i+1] || 0)
                + (queryArr[i+2] || 0) * (vectorArr[i+2] || 0)
                + (queryArr[i+3] || 0) * (vectorArr[i+3] || 0);
        }
        
        if (index < 5) {
            console.log(`Item ${index}: similarity = ${dotProduct.toFixed(4)}`);
        }
        
        if (dotProduct > threshold) {
            similarities.push({
                index,
                similarity: dotProduct,
                code: item.code || `Item_${index + 1}`,
                page: item.page || '01'
            });
        }
    }
    similarities.sort((a, b) => b.similarity - a.similarity);
    const t1 = performance.now();
    console.log(`[Timing] calculateSimilarities: ${(t1-t0).toFixed(2)} ms for ${imageVectors.length} vectors`);
    return similarities.slice(0, 50);
}


// Function to generate affiliate links HTML
function generateAffiliateLinks(tags, chars) {
    const items = [...tags, ...chars];
    const links = [];
    items.forEach(item => {
        const urls = getAffiliateUrls(item);
        if (urls) {
            urls.forEach((url, idx) => {
                // Extract domain for tooltip
                let domain = '';
                try { domain = new URL(url).hostname.replace('www.', ''); } catch {}
                links.push(
                    `<a href="${url}" target="_blank" class="btn btn-secondary" style="margin: 6px 8px 0 0; display:inline-block; font-size:1.08em; padding: 16px 22px; border-radius: 18px; box-shadow: 0 2px 8px #ffd70033; position:relative;"
                        title="Shop for ${item} on ${domain}">
                        <span style="font-size:1.25em; margin-right:7px; vertical-align:middle;">🛒</span>
                        <span style="font-weight:600; letter-spacing:0.01em;">Discover ${item}${urls.length > 1 ? ` #${idx+1}` : ''} Deals</span>
                        <span style="font-size:0.95em; color:#ffd700; margin-left:8px; vertical-align:middle;">→</span>
                    </a>`
                );
            });
        }
    });
    if (links.length === 0) return '';
    return `<div style="margin-top:18px;"><div style="font-weight:700; color:#b8860b; margin-bottom:15px; font-size:1.13em; letter-spacing:0.01em; text-align:center; display:block; width:100%;">🔥 Hot Affiliate Offers:</div><div style="text-align:center;">${links.join('')}</div></div>`;
}

function binaryStringToBytes(binaryStr) {
  const bytes = new Uint8Array(binaryStr.length);
  for (let i = 0; i < binaryStr.length; i++) {
    bytes[i] = binaryStr.charCodeAt(i) & 0xff; // mask to byte
  }
  return bytes;
  
  const blob = new Blob([bytes], { type: mimeType });
  return URL.createObjectURL(blob);
}


// Display search results in the UI with better formatting
async function displayResults(similarities, mostLikely) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    if (mostLikely) 
    {
        const likelyDiv = document.createElement('div');
likelyDiv.className = 'most-likely highlight-pop';

const thumbnailSrc = `data:image/webp;base64,${mostLikely.thumbnail}`;

likelyDiv.innerHTML = `
    <div class="most-likely-header">
        <span class="most-likely-crown">👑</span> Most Likely Match
    </div>
    <div class="most-likely-flex">
        <div class="most-likely-thumb-col">
            <img src="${thumbnailSrc}" alt="Thumbnail" class="most-likely-thumbnail-large">
        </div>
        <div class="most-likely-details-col">
            <div class="most-likely-detail-row"><span class="ml-label">Name:</span> <span class="ml-value">${mostLikely.name}</span></div>
            <div class="most-likely-detail-row"><span class="ml-label">Code:</span> <span class="ml-value">${mostLikely.code}</span></div>
            <div class="most-likely-detail-row">
                <span class="ml-label">Link:</span> 
                <span class="ml-value">
                    <a href="https://nhentai.to/g/${mostLikely.code}/" target="_blank" rel="noopener noreferrer">
                        www.nhentai.to/g/${mostLikely.code}
                    </a>
                </span>
            </div>
            <div class="most-likely-detail-row"><span class="ml-label">Top tags:</span> <span class="ml-value">${mostLikely.top_tags.join(', ')}</span></div>
            <div class="most-likely-detail-row"><span class="ml-label">Top Characters:</span> <span class="ml-value">${mostLikely.top_chars.join(', ')}</span></div>
            <div class="most-likely-detail-row"><span class="ml-label">Similarity:</span> <span class="ml-value">${(mostLikely.similarity * 100).toFixed(1)}%</span></div>
        </div>
    </div>
    ${generateAffiliateLinks(mostLikely.top_tags, mostLikely.top_chars)}
`;

resultsDiv.appendChild(likelyDiv);
    }

    // Hide and make expandable the list of other results
    if (similarities.length > 0) {
        const expandableContainer = document.createElement('div');
        expandableContainer.className = 'expandable-results';
        expandableContainer.style.marginTop = '16px';
        const toggleBtn = document.createElement('button');
        toggleBtn.textContent = 'Show All Results';
        toggleBtn.className = 'expand-btn';
        let expanded = false;
        const resultsContainer = document.createElement('div');
        resultsContainer.className = 'results-list';
        resultsContainer.style.display = 'none';

	similarities.forEach((item, index) => {
	  const resultItem = document.createElement('div');
	  resultItem.className = 'result-item';
	  const url = `https://nhentai.to/g/${item.code}/${item.page}`;
	  resultItem.innerHTML = `
	    <div class="result-info" style="
	      display: flex; 
	      align-items: center; 
	      justify-content: space-between; 
	      padding: 8px; 
	      max-width: 600px;">
	      
	      <div class="result-details" style="display: flex; flex-direction: column; gap: 4px;">
		<div class="result-code">Code: ${item.code}</div>
		<div class="result-page">Page: ${item.page}</div>
		<div class="similarity">Similarity: ${(item.similarity * 100).toFixed(1)}%</div>
	      </div>
	      
	      <a href="${url}" target="_blank" rel="noopener noreferrer" 
		 style="margin-left: 16px; font-size: 1.1em; font-weight: 600; color: #1a0dab; text-decoration: underline; white-space: nowrap;">
		${url}
	      </a>
	    </div>
	  `;
	  resultsContainer.appendChild(resultItem);
	});

        toggleBtn.onclick = () => {
            expanded = !expanded;
            resultsContainer.style.display = expanded ? 'block' : 'none';
            toggleBtn.textContent = expanded ? 'Hide All Results' : 'Show All Results';
        };

        expandableContainer.appendChild(toggleBtn);
        expandableContainer.appendChild(resultsContainer);
        resultsDiv.appendChild(expandableContainer);
    }

    updateResultsInfo(similarities.length);
}

// Update results info display
function updateResultsInfo(count) {
    const resultsInfo = document.getElementById('resultsInfo');
    const resultsCount = document.getElementById('resultsCount');
    
    if (count > 0) {
        resultsCount.textContent = count;
        resultsInfo.style.display = 'flex';
    } else {
        resultsInfo.style.display = 'none';
    }
}

// Update status messages in the UI
function updateStatus(message, type) {
    const statusDiv = document.getElementById('status');
    if (message) {
        statusDiv.innerText = message;
        statusDiv.className = `status ${type}`;
        statusDiv.style.display = 'block';
    } else {
        statusDiv.style.display = 'none';
    }
}

// Add dot product helper for lshSearch
function dot(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
    return sum;
}

// Helper to fetch and decompress gzip JSON, with fallback to .json if .json.gz fails
async function fetchGzippedJsonWithFallback(url) {
    try {
        // Try .json.gz first
        return await fetchGzippedJson(url + '.gz');
    } catch (e) {
        // If .json.gz fails, try .json (assume it's still gzipped)
        try {
            return await fetchGzippedJson(url);
        } catch (e2) {
            throw new Error(`Failed to fetch ${url}.gz or ${url}`);
        }
    }
}

// Web worker for parallel gzip decompression
const decompressWorkerBlob = new Blob([`
    self.onmessage = function(e) {
        importScripts('https://cdn.jsdelivr.net/npm/pako@2.1.0/dist/pako.min.js');
        const { buffer } = e.data;
        try {
            const decompressed = self.pako.ungzip(new Uint8Array(buffer), { to: 'string' });
            self.postMessage({ result: decompressed });
        } catch (err) {
            self.postMessage({ error: err.message });
        }
    };
`], { type: 'application/javascript' });
const decompressWorkerUrl = URL.createObjectURL(decompressWorkerBlob);

function decompressGzipParallel(buffer) {
    return new Promise((resolve, reject) => {
        const worker = new Worker(decompressWorkerUrl);
        worker.onmessage = function(e) {
            if (e.data.result) {
                resolve(e.data.result);
            } else {
                reject(new Error(e.data.error));
            }
            worker.terminate();
        };
        worker.postMessage({ buffer }, [buffer]);
    });
}

// Helper to fetch and decompress gzip JSON using parallel worker
async function fetchGzippedJson(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to fetch ${url}`);
    const buffer = await resp.arrayBuffer();
    const decompressed = await decompressGzipParallel(buffer);
    return JSON.parse(decompressed);
}

async function fetchClusterBinary_new(url, vectorLength = 384) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to fetch ${url}`);

    // Read compressed bytes
    const compressed = new Uint8Array(await resp.arrayBuffer());

    // Decompress gzip -> Uint8Array
    const decompressed = pako.ungzip(compressed);

    // Prepare to parse binary data
    const buffer = decompressed.buffer;
    const entrySize = 4 + 2 + 2 * vectorLength; // int32 + int16 + float16
    const count = buffer.byteLength / entrySize;
    const view = new DataView(buffer);
    const result = [];

    for (let i = 0; i < count; i++) {
        const offset = i * entrySize;
        const code = view.getInt32(offset, true);
        const page = view.getInt16(offset + 4, true);

        const vectorBytes = buffer.slice(offset + 6, offset + 6 + 2 * vectorLength);
        const f16Array = new Float16Array(vectorBytes);
        const vector = new Float32Array(f16Array);

        result.push({ code, page, vector: Array.from(vector) });
    }
    return result;
}


// Helper to fetch and parse binary cluster files
async function fetchClusterBinary(url, vectorLength = 384) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to fetch ${url}`);
    const buffer = await resp.arrayBuffer();
    const entrySize = 4 + 2 + 2 * vectorLength; // code (int32) + page (int16) + vector (float16)
    const count = buffer.byteLength / entrySize;
    const view = new DataView(buffer);
    const result = [];
    for (let i = 0; i < count; i++) {
        const offset = i * entrySize;
        const code = view.getInt32(offset, true);
        const page = view.getInt16(offset + 4, true);
        // Use float16 library to decode vector
        const vectorBytes = buffer.slice(offset + 6, offset + 6 + 2 * vectorLength);
        // Convert float16 bytes to Float32Array using @petamoriken/float16
        const f16Array = new Float16Array(vectorBytes);
        const vector = new Float32Array(f16Array);
        result.push({ code, page, vector: Array.from(vector) });
    }
    return result;
}


// Helper to fetch and parse binary centroids file
async function fetchCentroidsBinary(url, vectorLength = 384) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to fetch ${url}`);
    const buffer = await resp.arrayBuffer();
    const entrySize = 2 * vectorLength; // 512 x float16
    const count = buffer.byteLength / entrySize;
    const result = [];
    for (let i = 0; i < count; i++) {
        const offset = i * entrySize;
        // Convert float16 bytes to Float32Array using @petamoriken/float16
        const f16Array = new Float16Array(buffer.slice(offset, offset + entrySize));
        const vector = new Float32Array(f16Array);
        result.push({ id: i, vector: Array.from(vector) });
    }
    return result;
}

async function fetchClusterMetaBinary(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch ${url}`);
  const compressedBuffer = await resp.arrayBuffer();

  // Decompress gzip using pako
  const decompressed = pako.ungzip(new Uint8Array(compressedBuffer));

  // decompressed is a Uint8Array - convert to ArrayBuffer if needed
  const buffer = decompressed.buffer;

  // Now parse the decompressed buffer using your existing parser function:
  return parseClusterMetaBinary(buffer);
}


// Helper to fetch and parse binary cluster metadata (TLV format)
async function parseClusterMetaBinary(buffer) {
    const view = new DataView(buffer);
    let offset = 0;
    const result = [];
    while (offset < buffer.byteLength) {
        // code (int32)
        const code = view.getInt32(offset, true);
        offset += 4;
        // name (uint16 len + utf8)
        const nameLen = view.getUint16(offset, true);
        offset += 2;
        const name = new TextDecoder().decode(new Uint8Array(buffer, offset, nameLen));
        offset += nameLen;
        // top_tags (uint8 count, then each: uint16 len + utf8)
        const tagCount = view.getUint8(offset);
        offset += 1;
        const top_tags = [];
        for (let i = 0; i < tagCount; i++) {
            const tagLen = view.getUint16(offset, true);
            offset += 2;
            const tag = new TextDecoder().decode(new Uint8Array(buffer, offset, tagLen));
            offset += tagLen;
            top_tags.push(tag);
        }
        // top_chars (uint8 count, then each: uint16 len + utf8)
        const charCount = view.getUint8(offset);
        offset += 1;
        const top_chars = [];
        for (let i = 0; i < charCount; i++) {
            const charLen = view.getUint16(offset, true);
            offset += 2;
            const char = new TextDecoder().decode(new Uint8Array(buffer, offset, charLen));
            offset += charLen;
            top_chars.push(char);
        }
        // thumbnail (uint32 len + utf8)
        const thumbLen = view.getUint32(offset, true);
        offset += 4;
        const thumbnail = new TextDecoder().decode(new Uint8Array(buffer, offset, thumbLen));
        offset += thumbLen;
        result.push({ code, name, top_tags, top_chars, thumbnail });
    }
    return result;
}

// Main cluster-based search function
async function searchByClusters(queryEmbedding) {
    // 1. Load centroids
    const t0 = performance.now();
    const centroids = await fetchCentroidsBinary('clusters/centroids.bin'); // [{id, vector}]
    const t1 = performance.now();
    console.log(`[Timing] Centroids fetch+decompress: ${(t1-t0).toFixed(2)} ms`);

    // Dynamically update cluster slider max
    const clusterSlider = document.getElementById('clusterSlider');
    const clusterInput = document.getElementById('clusterInput');
    if (clusterSlider && clusterInput) {
        clusterSlider.max = MAX_CLUSTERS;
        clusterInput.max = MAX_CLUSTERS;
        currentSearchClusters = clamp(currentSearchClusters);
        clusterSlider.value = currentSearchClusters;
        clusterInput.value = currentSearchClusters;
        document.getElementById('clusterValue').textContent = `${currentSearchClusters}`;
    }

    // 2. Find top N closest centroids
    const centroidScores = centroids.map((c, idx) => ({
        id: c.id,
        idx,
        score: dot(queryEmbedding, c.vector)
    }));
    centroidScores.sort((a, b) => b.score - a.score);
    // Final clamp before using
    const topCentroids = centroidScores.slice(0, clamp(currentSearchClusters));
    const clusterIds = topCentroids.map(c => c.id);

    // 3. For each centroid, load its cluster JSON and search (in parallel)
    // Also count total images in each cluster
    let totalCompared = 0;
    const clusterPromises = topCentroids.map(async (centroid) => {
        const tFetch0 = performance.now();
        const clusterResp = await fetchClusterBinary_new(`clusters/clusters/cluster_${centroid.id}.bin.gz`); // [{vector, code, page}]
        const tFetch1 = performance.now();
        console.log(`[Timing] Cluster ${centroid.id} fetch+decompress: ${(tFetch1-tFetch0).toFixed(2)} ms`);
        totalCompared += clusterResp.length;
        // Search this cluster
        // Add clusterId to each result
        const tSearch0 = performance.now();
        const results = calculateSimilarities(queryEmbedding, clusterResp).map(r => ({
            ...r,
            clusterId: centroid.id
        }));
        const tSearch1 = performance.now();
        console.log(`[Timing] Cluster ${centroid.id} search: ${(tSearch1-tSearch0).toFixed(2)} ms`);
        return results;
    });
    const tClusters0 = performance.now();
    const clusterResults = await Promise.all(clusterPromises); // Array of arrays
    const tClusters1 = performance.now();
    console.log(`[Timing] All clusters fetch+decompress+search: ${(tClusters1-tClusters0).toFixed(2)} ms`);

    // 4. Combine all results and keep top 50
    const allResults = clusterResults.flat();
    allResults.sort((a, b) => b.similarity - a.similarity);
    return { results: allResults.slice(0, 50), totalCompared, clusterIds };
}
