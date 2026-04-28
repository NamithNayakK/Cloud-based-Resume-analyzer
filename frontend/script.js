const API_BASE_URL = 'http://localhost:5000';
const HISTORY_KEY = 'nimbus-analysis-history-v1';
const MAX_HISTORY_ITEMS = 8;

const templates = {
  software: 'Software Engineer role requiring Python, REST APIs, Docker, cloud deployment, testing, and scalable backend architecture.',
  data: 'Data Analyst role requiring SQL, Excel, Power BI, dashboarding, statistical analysis, and stakeholder communication.',
  cloud: 'Cloud Engineer role requiring AWS or Azure, Kubernetes, IaC, CI/CD pipelines, observability, and production incident handling.',
  frontend: 'Frontend Developer role requiring JavaScript, React, UI architecture, performance optimization, accessibility, and API integration.'
};

const form = document.getElementById('analyzerForm');
const resumeInput = document.getElementById('resume');
const dropzone = document.getElementById('dropzone');
const fileMeta = document.getElementById('fileMeta');
const jobDescription = document.getElementById('jobDescription');
const jdWordCount = document.getElementById('jdWordCount');
const jdQuality = document.getElementById('jdQuality');
const templateChips = document.getElementById('templateChips');
const statusEl = document.getElementById('status');
const loadingBar = document.getElementById('loadingBar');
const resultCard = document.getElementById('resultCard');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const copyTipsBtn = document.getElementById('copyTipsBtn');
const scoreValue = document.getElementById('scoreValue');
const scoreRing = document.getElementById('scoreRing');
const scoreBand = document.getElementById('scoreBand');
const matchedKeywords = document.getElementById('matchedKeywords');
const missingKeywords = document.getElementById('missingKeywords');
const recommendations = document.getElementById('recommendations');
const historyList = document.getElementById('historyList');
const historyCount = document.getElementById('historyCount');
const bestScore = document.getElementById('bestScore');
const avgScore = document.getElementById('avgScore');
const toast = document.getElementById('toast');
const themeToggle = document.getElementById('themeToggle');

// Theme handling
function applyTheme(name) {
  if (name === 'dark') {
    document.documentElement.classList.add('dark');
    themeToggle.setAttribute('aria-pressed', 'true');
  } else {
    document.documentElement.classList.remove('dark');
    themeToggle.setAttribute('aria-pressed', 'false');
  }
  localStorage.setItem('nimbus-theme', name);
}

themeToggle?.addEventListener('click', () => {
  const current = localStorage.getItem('nimbus-theme') || 'light';
  applyTheme(current === 'dark' ? 'light' : 'dark');
});

// Initialize theme
(() => {
  const saved = localStorage.getItem('nimbus-theme');
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  applyTheme(saved || (prefersDark ? 'dark' : 'light'));
})();

function showToast(message) {
  toast.textContent = message;
  toast.classList.add('show');
  window.setTimeout(() => {
    toast.classList.remove('show');
  }, 1600);
}

function fillList(listEl, values, fallbackText) {
  listEl.innerHTML = '';
  if (!Array.isArray(values) || values.length === 0) {
    const li = document.createElement('li');
    li.textContent = fallbackText;
    listEl.appendChild(li);
    return;
  }

  values.slice(0, 12).forEach((value) => {
    const li = document.createElement('li');
    li.textContent = value;
    listEl.appendChild(li);
  });
}

function readHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function writeHistory(items) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(items));
}

function scoreToBand(score) {
  if (score >= 85) {
    return 'Excellent fit';
  }
  if (score >= 65) {
    return 'Good fit';
  }
  if (score >= 40) {
    return 'Moderate fit';
  }
  return 'Needs improvement';
}

function updateScoreUI(score) {
  const safeScore = Number.isFinite(score) ? Math.max(0, Math.min(100, score)) : 0;
  scoreValue.textContent = `${safeScore}%`;
  scoreRing.style.background = `conic-gradient(#0c7a6c ${safeScore}%, #d2d7dd ${safeScore}% 100%)`;
  scoreBand.textContent = scoreToBand(safeScore);
}

function extractScore(payload) {
  return payload?.analysis?.overallScore
    ?? payload?.analysis?.score
    ?? payload?.score
    ?? 0;
}

function updateHistoryStats() {
  const entries = readHistory();
  historyCount.textContent = String(entries.length);

  if (entries.length === 0) {
    bestScore.textContent = '0%';
    avgScore.textContent = '0%';
    return;
  }

  const scores = entries.map((item) => item.score || 0);
  const top = Math.max(...scores);
  const avg = Math.round(scores.reduce((sum, item) => sum + item, 0) / scores.length);
  bestScore.textContent = `${top}%`;
  avgScore.textContent = `${avg}%`;
}

function renderHistory() {
  const entries = readHistory();
  historyList.innerHTML = '';

  if (entries.length === 0) {
    const empty = document.createElement('p');
    empty.className = 'file-meta';
    empty.textContent = 'No saved analyses yet. Run your first analysis to build history.';
    historyList.appendChild(empty);
    return;
  }

  entries.forEach((entry) => {
    const row = document.createElement('article');
    row.className = 'history-item';

    const left = document.createElement('div');
    left.className = 'history-meta';

    const role = document.createElement('p');
    role.className = 'history-role';
    role.textContent = entry.jobTitle || 'Custom role';

    const details = document.createElement('p');
    details.textContent = `${entry.fileName} - ${new Date(entry.timestamp).toLocaleString()}`;

    left.appendChild(role);
    left.appendChild(details);

    const score = document.createElement('span');
    score.className = 'history-score';
    score.textContent = `${entry.score}%`;

    row.appendChild(left);
    row.appendChild(score);
    historyList.appendChild(row);
  });
}

function saveHistory(entry) {
  const current = readHistory();
  current.unshift(entry);
  const trimmed = current.slice(0, MAX_HISTORY_ITEMS);
  writeHistory(trimmed);
  renderHistory();
  updateHistoryStats();
}

function updateJDMicrocopy() {
  const text = jobDescription.value.trim();
  const words = text ? text.split(/\s+/).length : 0;
  jdWordCount.textContent = `${words} words`;

  if (words < 15) {
    jdQuality.textContent = 'Short JD. Include responsibilities and tools for better matching.';
  } else if (words < 40) {
    jdQuality.textContent = 'Decent JD. Add required skills and impact expectations.';
  } else {
    jdQuality.textContent = 'Great detail level. This should produce meaningful analysis.';
  }
}

function updateFileMeta() {
  const file = resumeInput.files && resumeInput.files[0];
  if (!file) {
    fileMeta.textContent = 'No file selected';
    return;
  }

  const kb = Math.max(1, Math.round(file.size / 1024));
  fileMeta.textContent = `${file.name} - ${kb} KB`;
}

function setLoading(state) {
  analyzeBtn.disabled = state;
  loadingBar.classList.toggle('hidden', !state);
}

function resetFormView() {
  form.reset();
  updateFileMeta();
  updateJDMicrocopy();
  statusEl.className = 'status';
  statusEl.textContent = '';
}

function bindDropzone() {
  ['dragenter', 'dragover'].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.add('is-active');
    });
  });

  ['dragleave', 'drop'].forEach((name) => {
    dropzone.addEventListener(name, () => {
      dropzone.classList.remove('is-active');
    });
  });

  dropzone.addEventListener('drop', (event) => {
    event.preventDefault();
    const files = event.dataTransfer?.files;
    if (!files || files.length === 0) {
      return;
    }

    resumeInput.files = files;
    updateFileMeta();
  });

  dropzone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      resumeInput.click();
    }
  });
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  statusEl.className = 'status';
  statusEl.textContent = 'Uploading and analyzing...';
  setLoading(true);

  try {
    const data = new FormData(form);
    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: 'POST',
      body: data,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload?.detail || payload?.error || 'Unable to analyze resume.');
    }

    const score = extractScore(payload);
    const analysis = payload.analysis || payload;

    updateScoreUI(score);
    fillList(matchedKeywords, analysis?.matchedKeywords, 'No matched keywords were detected.');
    fillList(missingKeywords, analysis?.missingKeywords, 'No major gaps detected.');
    fillList(recommendations, analysis?.recommendations, 'Great alignment. Add quantified achievements to improve impact.');

    resultCard.classList.remove('hidden');
    statusEl.classList.add('success');
    statusEl.textContent = 'Analysis completed successfully.';

    const historyEntry = {
      score,
      fileName: resumeInput.files?.[0]?.name || 'resume',
      jobTitle: jobDescription.value.split('.').shift().slice(0, 64) || 'Custom role',
      timestamp: Date.now(),
    };
    saveHistory(historyEntry);
  } catch (error) {
    statusEl.classList.add('error');
    statusEl.textContent = error.message;
    showToast('Analysis failed. Check API status and try again.');
  } finally {
    setLoading(false);
  }
});

resumeInput.addEventListener('change', updateFileMeta);
jobDescription.addEventListener('input', updateJDMicrocopy);

clearBtn.addEventListener('click', () => {
  resetFormView();
  showToast('Form cleared');
});

copyTipsBtn.addEventListener('click', async () => {
  const items = Array.from(recommendations.querySelectorAll('li')).map((node) => node.textContent).filter(Boolean);
  if (items.length === 0) {
    showToast('No recommendations to copy yet.');
    return;
  }

  try {
    await navigator.clipboard.writeText(items.join('\n'));
    showToast('Recommendations copied');
  } catch {
    showToast('Clipboard access blocked by browser.');
  }
});

templateChips.addEventListener('click', (event) => {
  const target = event.target;
  if (!(target instanceof HTMLButtonElement)) {
    return;
  }

  const key = target.dataset.template;
  if (!key || !templates[key]) {
    return;
  }

  jobDescription.value = templates[key];
  updateJDMicrocopy();
  showToast(`${target.textContent} template applied`);
});

bindDropzone();
updateJDMicrocopy();
updateFileMeta();
renderHistory();
updateHistoryStats();
const API_BASE_URL = 'http://localhost:5000';

const form = document.getElementById('analyzerForm');
const statusEl = document.getElementById('status');
const resultCard = document.getElementById('resultCard');
const scoreValue = document.getElementById('scoreValue');
const scoreRing = document.querySelector('.score-ring');
const matchedKeywords = document.getElementById('matchedKeywords');
const missingKeywords = document.getElementById('missingKeywords');
const recommendations = document.getElementById('recommendations');
const analyzeBtn = document.getElementById('analyzeBtn');

function fillList(listEl, values, fallbackText) {
  listEl.innerHTML = '';
  if (!values || values.length === 0) {
    const li = document.createElement('li');
    li.textContent = fallbackText;
    listEl.appendChild(li);
    return;
  }

  values.forEach((value) => {
    const li = document.createElement('li');
    li.textContent = value;
    listEl.appendChild(li);
  });
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  statusEl.className = 'status';
  statusEl.textContent = 'Uploading and analyzing...';
  analyzeBtn.disabled = true;

  try {
    const data = new FormData(form);

    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: 'POST',
      body: data,
    });

    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error || 'Unable to analyze resume.');
    }

    const score = payload.analysis?.overallScore || 0;
    scoreValue.textContent = `${score}%`;
    scoreRing.style.background = `conic-gradient(#0f766e ${score}%, #d1d5db ${score}% 100%)`;

    fillList(matchedKeywords, payload.analysis?.matchedKeywords, 'No matched keywords yet.');
    fillList(missingKeywords, payload.analysis?.missingKeywords, 'No major keyword gaps detected.');
    fillList(recommendations, payload.analysis?.recommendations, 'Resume is in good shape for this job description.');

    resultCard.classList.remove('hidden');
    statusEl.classList.add('success');
    statusEl.textContent = 'Analysis completed successfully.';
  } catch (error) {
    statusEl.classList.add('error');
    statusEl.textContent = error.message;
  } finally {
    analyzeBtn.disabled = false;
  }
});
