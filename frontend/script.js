const API_BASE_URL = 'http://localhost:5000';
const HISTORY_KEY = 'nimbus-analysis-history-v1';
const MAX_HISTORY_ITEMS = 8;

const form = document.getElementById('analyzerForm');
const resumeInput = document.getElementById('resume');
const dropzone = document.getElementById('dropzone');
const fileMeta = document.getElementById('fileMeta');
const statusEl = document.getElementById('status');
const loadingBar = document.getElementById('loadingBar');
const resultCard = document.getElementById('resultCard');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const copyTipsBtn = document.getElementById('copyTipsBtn');
const scoreValue = document.getElementById('scoreValue');
const scoreRing = document.getElementById('scoreRing');
const topRole = document.getElementById('topRole');
const scoreBand = document.getElementById('scoreBand');
const matchedKeywords = document.getElementById('matchedKeywords');
const missingKeywords = document.getElementById('missingKeywords');
const roleRecommendations = document.getElementById('roleRecommendations');
const resumeIssues = document.getElementById('resumeIssues');
const recommendations = document.getElementById('recommendations');
const historyList = document.getElementById('historyList');
const historyCount = document.getElementById('historyCount');
const bestScore = document.getElementById('bestScore');
const avgScore = document.getElementById('avgScore');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
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
    role.textContent = entry.roleName || 'Suggested role';

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

function clearHistory() {
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
  updateHistoryStats();
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

function renderRoleRecommendations(values) {
  roleRecommendations.innerHTML = '';
  if (!Array.isArray(values) || values.length === 0) {
    const li = document.createElement('li');
    li.textContent = 'No role signals detected yet.';
    roleRecommendations.appendChild(li);
    topRole.textContent = 'Awaiting analysis';
    return;
  }

  topRole.textContent = values[0]?.role || 'Suggested role found';

  values.slice(0, 3).forEach((item) => {
    const li = document.createElement('li');
    const title = document.createElement('div');
    title.className = 'role-row';
    const name = document.createElement('strong');
    name.textContent = item?.role || 'Role';
    const confidence = Number.isFinite(item?.confidence) ? Math.round(item.confidence * 100) : 0;
    const conf = document.createElement('span');
    conf.className = 'role-conf';
    conf.textContent = ` ${confidence}% fit`;
    title.appendChild(name);
    title.appendChild(conf);

    li.appendChild(title);

    // Render reason lines if present
    if (Array.isArray(item?.reason) && item.reason.length > 0) {
      const reasonsEl = document.createElement('ul');
      reasonsEl.className = 'role-reasons';
      item.reason.forEach((r) => {
        const rli = document.createElement('li');
        rli.textContent = r;
        reasonsEl.appendChild(rli);
      });
      li.appendChild(reasonsEl);
    }

    // If explicit evidence field exists, show it as a highlighted snippet
    const evidence = item?.evidence || null;
    if (evidence) {
      const ev = document.createElement('blockquote');
      ev.className = 'evidence-snippet';
      ev.textContent = evidence;
      li.appendChild(ev);
    }

    roleRecommendations.appendChild(li);
  });
}

function renderIssueList(values) {
  fillList(resumeIssues, values, 'No major mistakes detected from the extracted text.');
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  statusEl.className = 'status';
  statusEl.textContent = 'Uploading and discovering the best-fit role...';
  setLoading(true);

  try {
    const data = new FormData();
    const file = resumeInput.files?.[0];
    if (file) {
      data.append('resume', file);
    }

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
    const roleList = analysis?.roleRecommendations || payload?.roleRecommendations || [];
    const issueList = analysis?.resumeIssues || payload?.resumeIssues || [];
    const planList = analysis?.improvementPlan || payload?.improvementPlan || analysis?.recommendations || [];

    updateScoreUI(score);
    fillList(matchedKeywords, analysis?.strengthSignals || analysis?.matchedKeywords, 'No strong signals were detected.');
    fillList(missingKeywords, analysis?.skillGaps || analysis?.missingKeywords, 'No major gaps detected.');
    renderRoleRecommendations(roleList);
    renderIssueList(issueList);
    fillList(recommendations, planList, 'Great alignment. Add quantified achievements to improve impact.');

    resultCard.classList.remove('hidden');
    statusEl.classList.add('success');
    statusEl.textContent = `Analysis completed successfully. Best-fit role: ${roleList[0]?.role || analysis?.extracted?.predictedCategory || 'Unknown'}.`;
    showToast('Resume analyzed successfully');

    const historyEntry = {
      score,
      fileName: resumeInput.files?.[0]?.name || 'resume',
      roleName: roleList[0]?.role || analysis?.extracted?.predictedCategory || 'Suggested role',
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

clearBtn.addEventListener('click', () => {
  resetFormView();
  showToast('Form cleared');
});

clearHistoryBtn?.addEventListener('click', () => {
  const hasHistory = readHistory().length > 0;
  if (!hasHistory) {
    showToast('No recent analyses to clear');
    return;
  }

  const confirmed = window.confirm('Clear all recent analyses from this browser?');
  if (!confirmed) {
    return;
  }

  clearHistory();
  showToast('Recent analyses cleared');
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
bindDropzone();
updateFileMeta();
renderHistory();
updateHistoryStats();
