const API_BASE_URL = 'http://localhost:8000';
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const uploadBtn = document.getElementById('uploadBtn');
const videoPreview = document.getElementById('videoPreview');
const resultContent = document.getElementById('resultContent');
const loading = document.getElementById('loading');
const errorMessage = document.getElementById('errorMessage');
const assessBtn = document.getElementById('assessBtn');
const assessLoading = document.getElementById('assessLoading');
const assessmentResult = document.getElementById('assessmentResult');

let selectedFile = null;
let currentTranscript = '';
let selectedQuestion = 'q1';
let selectedLanguage = 'en';
const langButtons = document.querySelectorAll('.lang-btn');

langButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const lang = btn.dataset.lang;
        selectedLanguage = lang;
        langButtons.forEach(b => b.classList.remove('lang-active'));
        btn.classList.add('lang-active');
    });
});

document.querySelectorAll('.question-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.question-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedQuestion = btn.dataset.qid;
    });
});

uploadArea.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    if (file && file.type.startsWith('video/')) {
        selectedFile = file;

        fileName.textContent = file.name;
        fileSize.textContent = `Ukuran: ${(file.size / 1024 / 1024).toFixed(2)} MB`;
        fileInfo.classList.add('active');
        uploadBtn.disabled = false;

        const videoURL = URL.createObjectURL(file);
        videoPreview.src = videoURL;
        videoPreview.style.display = 'block';

        errorMessage.classList.remove('active');
    } else {
        showError('Silakan pilih file video yang valid!');
    }
}

uploadBtn.addEventListener('click', async () => {
    if (selectedFile) {
        await uploadToServer(selectedFile);
    }
});

const _uploadToServerOriginal = uploadToServer;
async function uploadToServer(file) {
    loading.classList.add('active');
    uploadBtn.disabled = true;
    errorMessage.classList.remove('active');

    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('language', selectedLanguage);

        const uploadUrl = `${API_BASE_URL}/upload`;
        const response = await fetch(uploadUrl, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        displayResult(result);

        currentTranscript = result.data.asr.full_scripts;
        assessBtn.disabled = false;

    } catch (error) {
        console.error('Upload error:', error);
        showError(`Gagal memproses video: ${error.message}`);
    } finally {
        loading.classList.remove('active');
        uploadBtn.disabled = false;
    }
}

assessBtn.addEventListener('click', async () => {
    if (!currentTranscript) {
        showError('Upload video terlebih dahulu!');
        return;
    }

    await assessAnswer(selectedQuestion, currentTranscript);
});

async function assessAnswer(qid, transcript) {
    assessLoading.classList.add('active');
    assessBtn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('question_id', qid);
        formData.append('transcript', transcript);

        const response = await fetch(`${API_BASE_URL}/assess`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || 'Assessment failed');
        }

        const result = await response.json();
        displayAssessment(result.data);

    } catch (error) {
        console.error('Assessment error:', error);
        showError(`Assessment gagal: ${error.message}`);
    } finally {
        assessLoading.classList.remove('active');
        assessBtn.disabled = false;
    }
}

function displayAssessment(data) {
    let html = `
                <div style="margin-top: 20px;">
                    <div class="score-display">
                        <div class="score-number">${data.score}</div>
                        <div class="score-label">Score</div>
                    </div>

                    <div class="matched-level">
                        <div class="matched-level-label">Matched Level:</div>
                        <div class="matched-level-value">${data.matched_level}</div>
                    </div>

                    <h3 style="font-size: 15px;"> Reason</h3>
                    <div class="reason-text">
                        ${data.reason}
                    </div>
                </div>
            `;

    assessmentResult.innerHTML = html;
}

function displayResult(result) {
    const data = result.data;
    const asr = data.asr;
    const cheat = data.cheating;

    let resultHTML = '';

    resultHTML += `
                <div class="summary-box">
                    <h3 style="margin-top: 0; font-size: 16px;"> Ringkasan</h3>
                    <div class="summary-row">
                        <span style="color: #8a8a9a;">Total Speaker:</span>
                        <span style="color: #ffffff; font-weight: 600;">${asr.total_speakers} speaker</span>
                    </div>
                    <div class="summary-row">
                        <span style="color: #8a8a9a;">Durasi Total:</span>
                        <span style="color: #ffffff; font-weight: 600;">${formatDuration(asr.total_duration_seconds)}</span>
                    </div>
                    <div class="summary-row">
                        <span style="color: #8a8a9a;">Cheating:</span>
                        <span style="color: ${cheat.detected ? '#ff6b6b' : '#51cf66'}; font-weight: 600;">
                            ${cheat.detected ? 'YA (' + cheat.total_events + 'x)' : 'TIDAK'}
                        </span>
                    </div>
            `;

    if (cheat.detected && cheat.reason_counts) {
        resultHTML += `
                    <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #3a3a50;">
                        <p style="color: #8a8a9a; margin-bottom: 8px; font-weight: 600; font-size: 12px;">Jenis:</p>
                        <div>
                `;

        for (const [reason, count] of Object.entries(cheat.reason_counts)) {
            const reasonText = reason.replace(/_/g, ' ');
            resultHTML += `<span class="reason-badge">${reasonText}: ${count}x</span>`;
        }

        resultHTML += `</div></div>`;
    }

    resultHTML += `</div>`;

    if (cheat.detected) {
        resultHTML += `
                    <div class="cheating-alert">
                         Terdeteksi ${cheat.total_events} kejadian mencurigakan
                    </div>

                    <h3 style="font-size: 16px;"> Waktu Kejadian</h3>
                `;

        if (cheat.reason_details) {
            for (const [reason, details] of Object.entries(cheat.reason_details)) {
                const reasonText = reason.replace(/_/g, ' ');
                resultHTML += `<div class="reason-section">
                                <div class="reason-header">${reasonText} (${details.length}x)</div>`;

                details.forEach((detail) => {
                    resultHTML += `<span class="time-range">⏱ ${detail.time_range}</span>`;
                });

                resultHTML += `</div>`;
            }
        }
    } else {
        resultHTML += `<div class="cheating-safe"> Tidak ada perilaku mencurigakan</div>`;
    }

    resultHTML += `
                <h3 style="font-size: 16px;"> Transkrip</h3>
                <div class="transcript-box">${asr.full_scripts}</div>
            `;

    resultContent.innerHTML = resultHTML;
}

function formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.add('active');
}

window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            console.log(' Connected to server:', data);
        }
    } catch (error) {
        console.warn(' Server not reachable');
        showError('Server tidak terdeteksi di http://localhost:8000');
    }
});

const aboutBtn = document.getElementById('aboutBtn');
const aboutOverlay = document.getElementById('aboutOverlay');
const aboutClose = document.getElementById('aboutClose');

if (aboutBtn && aboutOverlay && aboutClose) {
    aboutBtn.addEventListener('click', () => {
        aboutOverlay.classList.add('active');
    });

    aboutClose.addEventListener('click', () => {
        aboutOverlay.classList.remove('active');
    });

    aboutOverlay.addEventListener('click', (e) => {
        if (e.target === aboutOverlay) {
            aboutOverlay.classList.remove('active');
        }
    });
}

const serverStatus = document.getElementById('serverStatus');
const xpBar = document.getElementById('xpBar');
const origConsoleLog = console.log;
const origConsoleWarn = console.warn;

console.log = function (...args) {
    origConsoleLog.apply(console, args);
    if (serverStatus && String(args[0]).includes('Connected to server')) {
        serverStatus.classList.add('online');
        serverStatus.classList.remove('offline');
        const label = serverStatus.querySelector('.status-label');
        if (label) label.textContent = 'Server Online';
        if (xpBar && xpBar.style.width === '0%') {
            xpBar.style.width = '15%';
        }
    }
};

console.warn = function (...args) {
    origConsoleWarn.apply(console, args);
    if (serverStatus && String(args[0]).includes('Server not reachable')) {
        serverStatus.classList.add('offline');
        serverStatus.classList.remove('online');
        const label = serverStatus.querySelector('.status-label');
        if (label) label.textContent = 'Server Offline';
    }
};