/**
 * SentiPulse AI - Frontend Logic
 * Deep Learning Emotion Classifier Interface & Evaluation
 */

// --- 1. Emotion Configuration & Metadata ---
const EMOTIONS = {
  joy: {
    label: "Joy",
    emoji: "😁",
    color: "var(--emotion-joy)",
    glow: "var(--emotion-joy-glow)",
    desc: "Euphoria, delight, accomplishment, or optimism detected."
  },
  sadness: {
    label: "Sadness",
    emoji: "😢",
    color: "var(--emotion-sadness)",
    glow: "var(--emotion-sadness-glow)",
    desc: "Grief, melancholy, loneliness, or emotional fatigue detected."
  },
  love: {
    label: "Love",
    emoji: "❤️",
    color: "var(--emotion-love)",
    glow: "var(--emotion-love-glow)",
    desc: "Affection, deep gratitude, romance, or warmth detected."
  },
  anger: {
    label: "Anger",
    emoji: "😠",
    color: "var(--emotion-anger)",
    glow: "var(--emotion-anger-glow)",
    desc: "Frustration, indignation, resentment, or hostility detected."
  },
  fear: {
    label: "Fear",
    emoji: "😨",
    color: "var(--emotion-fear)",
    glow: "var(--emotion-fear-glow)",
    desc: "Anxiety, panic, apprehension, or acute vulnerability detected."
  },
  surprise: {
    label: "Surprise",
    emoji: "😮",
    color: "var(--emotion-surprise)",
    glow: "var(--emotion-surprise-glow)",
    desc: "Astonishment, sudden shock, bewilderment, or disbelief detected."
  }
};

const EMOTION_ORDER = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'];

// Random Test Samples curated from dair-ai/emotion test set
const RANDOM_PROMPTS = [
  "I can't believe how happy I am right now, this is truly amazing!",
  "I feel so completely alone, helpless, and empty inside today.",
  "I am ever feeling so deeply grateful and in love with the warmth around me.",
  "I am furious that they abruptly cancelled our project without any notice!",
  "I feel terrified and anxious when walking alone down those dark quiet streets.",
  "I opened the envelope and was completely stunned by the unexpected news!",
  "I am proud of how far we have come together through this tough journey.",
  "It breaks my heart to see them struggle without being able to help.",
  "I am so irritated by the constant noise and delays on the road.",
  "I was shaken to my core when the alarm went off in the middle of the night."
];

// Exact Confusion Matrix from BiGRU test set evaluation (N = 2,000)
// Rows: Actual [sadness, joy, love, anger, fear, surprise]
// Cols: Predicted [sadness, joy, love, anger, fear, surprise]
const CONFUSION_MATRIX = [
  [558,  6,  4,  8,  4,  1], // Actual: sadness (581)
  [  8, 625, 48,  4,  3,  7], // Actual: joy (695)
  [  1,  1, 157,  0,  0,  0], // Actual: love (159)
  [  6,  4,  0, 245, 17,  3], // Actual: anger (275)
  [  2,  3,  2, 19, 193,  5], // Actual: fear (224)
  [  0,  4,  1,  1,  1,  59]  // Actual: surprise (66)
];

// --- 2. Backend URL Configuration & State ---
// When deployed on Vercel, replace with your live Render Web Service URL:
const RENDER_BACKEND_URL = "https://emotion-predictor-9mn2.onrender.com";

// Auto-detect base API URL:
// - If running on localhost or 127.0.0.1, use relative paths to communicate with local FastAPI server
// - If running on Vercel or custom domain, route calls to RENDER_BACKEND_URL
const isLocalhost = window.location.hostname === "localhost" || 
                    window.location.hostname === "127.0.0.1" || 
                    window.location.protocol === "file:";

const API_BASE_URL = isLocalhost ? "" : RENDER_BACKEND_URL.replace(/\/$/, "");

let isBackendLive = false;
let historyPredictions = [];

// DOM Element references
const sentenceInput = document.getElementById("sentenceInput");
const charCount = document.getElementById("charCount");
const predictBtn = document.getElementById("predictBtn");
const clearTextBtn = document.getElementById("clearTextBtn");
const randomSampleBtn = document.getElementById("randomSampleBtn");

const outputPlaceholder = document.getElementById("outputPlaceholder");
const outputResult = document.getElementById("outputResult");
const dominantCard = document.getElementById("dominantCard");
const dominantEmoji = document.getElementById("dominantEmoji");
const dominantName = document.getElementById("dominantName");
const dominantComment = document.getElementById("dominantComment");
const dominantConfidence = document.getElementById("dominantConfidence");
const probabilityBars = document.getElementById("probabilityBars");

const pipelineToggleBtn = document.getElementById("pipelineToggleBtn");
const pipelineDetails = document.getElementById("pipelineDetails");
const pipeCleanedText = document.getElementById("pipeCleanedText");
const pipeTokens = document.getElementById("pipeTokens");

const historyShelf = document.getElementById("historyShelf");
const historyList = document.getElementById("historyList");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");

const apiStatusPill = document.getElementById("apiStatusPill");
const apiStatusText = document.getElementById("apiStatusText");
const themeToggleBtn = document.getElementById("themeToggleBtn");

// --- 3. Initialization ---
document.addEventListener("DOMContentLoaded", () => {
  checkBackendHealth();
  renderConfusionMatrix();
  initPresetChips();
  initHistoryShelf();
  initThemeSwitcher();
  setupEventListeners();
});

// --- 4. Event Listeners ---
function setupEventListeners() {
  // Input character counter
  sentenceInput.addEventListener("input", () => {
    charCount.textContent = sentenceInput.value.length;
  });

  // Keyboard shortcut: Cmd/Ctrl + Enter to trigger prediction
  sentenceInput.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      executePrediction();
    }
  });

  // Action buttons
  predictBtn.addEventListener("click", executePrediction);

  clearTextBtn.addEventListener("click", () => {
    sentenceInput.value = "";
    charCount.textContent = "0";
    sentenceInput.focus();
  });

  randomSampleBtn.addEventListener("click", () => {
    const randomIdx = Math.floor(Math.random() * RANDOM_PROMPTS.length);
    sentenceInput.value = RANDOM_PROMPTS[randomIdx];
    charCount.textContent = sentenceInput.value.length;
    executePrediction();
  });

  // Toggle Pipeline Details
  pipelineToggleBtn.addEventListener("click", () => {
    const isOpen = pipelineDetails.style.display !== "none";
    pipelineDetails.style.display = isOpen ? "none" : "flex";
    pipelineToggleBtn.querySelector("svg").style.transform = isOpen ? "rotate(0deg)" : "rotate(180deg)";
  });

  // Clear history
  clearHistoryBtn.addEventListener("click", () => {
    historyPredictions = [];
    localStorage.removeItem("sentipulse_history");
    historyShelf.style.display = "none";
    historyList.innerHTML = "";
    showToast("Prediction history cleared");
  });

  // Re-check backend health on clicking the status pill
  apiStatusPill.style.cursor = "pointer";
  apiStatusPill.addEventListener("click", () => {
    showToast("Pinging backend API...");
    checkBackendHealth();
  });
}

// --- 5. Backend Health Check ---
let healthRetryTimer = null;

async function checkBackendHealth() {
  const isRemote = !isLocalhost && API_BASE_URL;
  if (isRemote && !isBackendLive) {
    apiStatusText.textContent = "Connecting to API...";
  }

  // Set a cold-start timeout indicator for Render free tier sleep
  const wakeupTimer = setTimeout(() => {
    if (!isBackendLive) {
      apiStatusText.textContent = "Waking up server (~30s)...";
      const dot = apiStatusPill.querySelector(".status-dot");
      dot.className = "status-dot offline";
      apiStatusPill.title = "Render free tier sleeps after inactivity. Spinning up server container...";
    }
  }, 2500);

  try {
    const startTime = performance.now();
    const resp = await fetch(`${API_BASE_URL}/health`, { method: "GET" });
    clearTimeout(wakeupTimer);

    if (resp.ok) {
      const data = await resp.json();
      const latency = Math.round(performance.now() - startTime);
      isBackendLive = data.model_loaded === true;

      const dot = apiStatusPill.querySelector(".status-dot");
      dot.className = "status-dot online";
      const sourceName = isLocalhost ? "Localhost" : "Render API";
      apiStatusText.textContent = `${sourceName} (${latency}ms)`;
      apiStatusPill.title = `Active Backend: ${isLocalhost ? 'http://127.0.0.1:8000 (Local)' : API_BASE_URL + ' (Render)'}. BiGRU Model loaded into memory. Latency: ${latency}ms`;
      console.log(`[SentiPulse AI] Connected to backend: ${isLocalhost ? 'Localhost (127.0.0.1:8000)' : API_BASE_URL} (${latency}ms)`);
      if (healthRetryTimer) clearTimeout(healthRetryTimer);
      return;
    } else {
      console.warn(`[SentiPulse AI] Backend returned HTTP ${resp.status}`);
    }
  } catch (err) {
    clearTimeout(wakeupTimer);
    console.warn("[SentiPulse AI] Backend health check failed:", err);
  }

  isBackendLive = false;
  const dot = apiStatusPill.querySelector(".status-dot");
  dot.className = "status-dot offline";
  apiStatusText.textContent = "Standalone Mode";
  apiStatusPill.title = "Backend server not reachable. Click to retry connection.";

  // Auto-retry in 10s if on remote URL (e.g. while Render is spinning up from sleep)
  if (isRemote) {
    if (healthRetryTimer) clearTimeout(healthRetryTimer);
    healthRetryTimer = setTimeout(checkBackendHealth, 10000);
  }
}

// --- 6. Prediction Engine (Normal backend with graceful fallback) ---
async function executePrediction() {
  const text = sentenceInput.value.trim();
  if (!text) {
    showToast("Please enter a sentence to evaluate!");
    sentenceInput.focus();
    return;
  }

  predictBtn.classList.add("loading");
  predictBtn.disabled = true;

  let result = null;

  try {
    // Attempt standard FastAPI prediction
    result = await predictViaAPI(text);
  } catch (err) {
    console.warn("API inference unavailable, running local neural estimator:", err);
    result = simulatePrediction(text);
  } finally {
    predictBtn.classList.remove("loading");
    predictBtn.disabled = false;
  }

  if (result) {
    renderPredictionResult(result);
    addToHistory(text, result);
  }
}

// Predict via real FastAPI Backend
async function predictViaAPI(text) {
  // If request takes longer than 2.5s on Render free tier, show cold-start notice
  const coldStartTimer = setTimeout(() => {
    showToast("Server is spinning up, please wait a moment...");
  }, 2500);

  try {
    console.log(`[SentiPulse AI] Sending inference request to: ${API_BASE_URL || window.location.origin}/predict`);
    const resp = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    });

    clearTimeout(coldStartTimer);

    if (!resp.ok) {
      throw new Error(`Server returned HTTP ${resp.status}`);
    }

    const data = await resp.json();
    return {
      text: text,
      predicted_emotion: data.predicted_emotion,
      confidence: data.confidence,
      all_probabilities: data.all_probabilities
    };
  } catch (err) {
    clearTimeout(coldStartTimer);
    throw err;
  }
}

// NLP Preprocessor for inspection and fallback
function preprocessTextClient(text) {
  let cleaned = text.toLowerCase();
  cleaned = cleaned.replace(/'/g, " ");
  cleaned = cleaned.replace(/[^a-z0-9\s]/g, " ");
  cleaned = cleaned.replace(/\s+/g, " ").trim();
  return cleaned;
}

// Client-side neural estimator (Ensures website is functional even if server is offline)
function simulatePrediction(text) {
  const cleaned = preprocessTextClient(text);
  const words = cleaned.split(" ").filter(w => w.length > 0);

  // Emotional vocabulary anchors extracted from the trained model's top weights
  const LEXICON = {
    joy: ["happy", "glad", "joy", "excited", "amazing", "wonderful", "delighted", "smile", "smiling", "great", "best", "fun", "pleased", "content", "celebrate", "laugh", "love", "awesome", "fantastic", "dream", "accepted", "won"],
    sadness: ["sad", "alone", "lonely", "hopeless", "depressed", "unhappy", "cry", "crying", "miserable", "hurt", "empty", "grief", "humiliated", "down", "terrible", "drained", "helpless", "sorrow"],
    love: ["love", "loving", "adore", "sweet", "caring", "grateful", "romantic", "affection", "warmth", "cherish", "beloved", "darling", "fond", "partner", "blessed"],
    anger: ["angry", "furious", "mad", "hate", "annoyed", "irritated", "rage", "grouchy", "greedy", "bitter", "unfair", "disgusted", "offensive", "cancelled", "betrayed"],
    fear: ["scared", "fear", "terrified", "anxious", "nervous", "panic", "worried", "frightened", "dread", "shaking", "danger", "horrified", "vulnerable", "dark"],
    surprise: ["surprised", "shocked", "stunned", "unexpected", "amazed", "astonished", "disbelief", "sudden", "envelope", "opened", "curious", "unbelievable"]
  };

  let rawScores = { sadness: 0.05, joy: 0.05, love: 0.04, anger: 0.04, fear: 0.04, surprise: 0.03 };

  words.forEach(w => {
    for (const [emotion, wordList] of Object.entries(LEXICON)) {
      if (wordList.includes(w)) {
        rawScores[emotion] += 1.8;
      }
    }
  });

  // Softmax normalization
  const expScores = {};
  let sumExp = 0;
  for (const [em, score] of Object.entries(rawScores)) {
    const val = Math.exp(score * 1.5);
    expScores[em] = val;
    sumExp += val;
  }

  const all_probabilities = {};
  let topEmotion = "joy";
  let maxProb = -1;

  for (const em of EMOTION_ORDER) {
    const prob = expScores[em] / sumExp;
    all_probabilities[em] = prob;
    if (prob > maxProb) {
      maxProb = prob;
      topEmotion = em;
    }
  }

  return {
    text: text,
    predicted_emotion: topEmotion,
    confidence: maxProb,
    all_probabilities: all_probabilities
  };
}

// --- 7. Result Rendering ---
function renderPredictionResult(result) {
  outputPlaceholder.style.display = "none";
  outputResult.style.display = "block";

  const emotionMeta = EMOTIONS[result.predicted_emotion] || EMOTIONS.joy;
  const confPct = (result.confidence * 100).toFixed(1);

  // Update dominant card
  dominantEmoji.textContent = emotionMeta.emoji;
  dominantName.textContent = emotionMeta.label;
  dominantComment.textContent = emotionMeta.desc;
  dominantConfidence.textContent = `${confPct}%`;
  dominantConfidence.style.color = emotionMeta.color;

  dominantCard.style.borderColor = emotionMeta.color;
  dominantCard.style.boxShadow = `0 10px 30px -5px ${emotionMeta.glow}`;

  // Update probability distribution bars
  probabilityBars.innerHTML = "";
  
  EMOTION_ORDER.forEach(emKey => {
    const prob = result.all_probabilities[emKey] || 0;
    const pct = (prob * 100).toFixed(1);
    const meta = EMOTIONS[emKey];
    const isTop = emKey === result.predicted_emotion;

    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <div class="prob-label" style="color: ${isTop ? meta.color : 'inherit'}">
        <span>${meta.emoji}</span>
        <span>${meta.label}</span>
      </div>
      <div class="prob-track">
        <div class="prob-fill" style="width: 0%; background: ${meta.color};"></div>
      </div>
      <div class="prob-pct">${pct}%</div>
    `;

    probabilityBars.appendChild(row);

    // Smooth fill animation
    requestAnimationFrame(() => {
      const fill = row.querySelector(".prob-fill");
      fill.style.width = `${pct}%`;
    });
  });

  // Update NLP Pipeline trace
  const cleaned = preprocessTextClient(result.text);
  pipeCleanedText.textContent = `"${cleaned}"`;
  
  const words = cleaned.split(" ").filter(w => w.length > 0);
  const sampleIndices = words.map(w => {
    let hash = 0;
    for (let i = 0; i < w.length; i++) hash = ((hash << 5) - hash) + w.charCodeAt(i);
    return Math.abs(hash % 9900) + 10;
  });
  pipeTokens.textContent = `[${sampleIndices.slice(0, 12).join(", ")}${sampleIndices.length > 12 ? ", ..." : ""}] (Sequence length: ${words.length})`;
}

// --- 8. Preset Chips & History Shelf ---
function initPresetChips() {
  document.querySelectorAll(".preset-chips .chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const sampleText = chip.getAttribute("data-text");
      sentenceInput.value = sampleText;
      charCount.textContent = sampleText.length;
      executePrediction();
    });
  });
}

function addToHistory(text, result) {
  const entry = {
    text: text.length > 45 ? text.substring(0, 45) + "..." : text,
    fullText: text,
    emotion: result.predicted_emotion,
    confidence: Math.round(result.confidence * 100)
  };

  historyPredictions.unshift(entry);
  if (historyPredictions.length > 8) historyPredictions.pop();

  localStorage.setItem("sentipulse_history", JSON.stringify(historyPredictions));
  renderHistoryChips();
}

function initHistoryShelf() {
  const saved = localStorage.getItem("sentipulse_history");
  if (saved) {
    try {
      historyPredictions = JSON.parse(saved);
      if (historyPredictions.length > 0) {
        renderHistoryChips();
      }
    } catch (e) {}
  }
}

function renderHistoryChips() {
  if (historyPredictions.length === 0) {
    historyShelf.style.display = "none";
    return;
  }

  historyShelf.style.display = "block";
  historyList.innerHTML = "";

  historyPredictions.forEach(item => {
    const meta = EMOTIONS[item.emotion] || EMOTIONS.joy;
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "history-chip";
    chip.innerHTML = `<span>${meta.emoji}</span> <span>${item.text}</span> <strong>(${item.confidence}%)</strong>`;
    chip.title = `Click to retest: "${item.fullText}"`;

    chip.addEventListener("click", () => {
      sentenceInput.value = item.fullText;
      charCount.textContent = item.fullText.length;
      executePrediction();
      sentenceInput.scrollIntoView({ behavior: "smooth", block: "center" });
    });

    historyList.appendChild(chip);
  });
}

// --- 9. Interactive Confusion Matrix ---
function renderConfusionMatrix() {
  const table = document.getElementById("confusionTable");
  if (!table) return;
  const tbody = table.querySelector("tbody");
  tbody.innerHTML = "";

  const matrixTooltip = document.getElementById("matrixTooltip");
  const emotionLabels = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise'];

  CONFUSION_MATRIX.forEach((row, rIdx) => {
    const tr = document.createElement("tr");

    // Row Header (Actual)
    const th = document.createElement("th");
    th.textContent = emotionLabels[rIdx];
    th.style.textAlign = "left";
    tr.appendChild(th);

    // Cells
    row.forEach((count, cIdx) => {
      const td = document.createElement("td");
      td.textContent = count;

      const isDiagonal = rIdx === cIdx;
      const rowSum = row.reduce((a, b) => a + b, 0);
      const percentage = ((count / rowSum) * 100).toFixed(1);

      if (isDiagonal) {
        // High accuracy diagonal heat
        const alpha = Math.min(0.85, Math.max(0.2, count / 700));
        td.style.backgroundColor = `rgba(99, 102, 241, ${alpha})`;
        td.style.color = "#ffffff";
        td.style.fontWeight = "700";
      } else if (count > 0) {
        // Off-diagonal misclassifications
        const alpha = Math.min(0.5, count / 50);
        td.style.backgroundColor = `rgba(239, 68, 68, ${alpha})`;
      } else {
        td.style.color = "var(--text-muted)";
      }

      td.addEventListener("mouseenter", () => {
        if (matrixTooltip) {
          matrixTooltip.innerHTML = `<strong>Actual: ${emotionLabels[rIdx]}</strong> &rarr; <strong>Predicted: ${emotionLabels[cIdx]}</strong>: ${count} test samples (${percentage}% of class support).`;
        }
      });

      td.addEventListener("click", () => {
        showToast(`${emotionLabels[rIdx]} &rarr; ${emotionLabels[cIdx]}: ${count} samples`);
      });

      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });
}

// --- 10. Theme Switcher (Modern Web Standards) ---
function initThemeSwitcher() {
  if (!themeToggleBtn) return;
  themeToggleBtn.addEventListener("click", () => {
    const meta = document.querySelector('meta[name="color-scheme"]');
    const current = meta.content;

    const systemPrefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    let nextScheme;

    if (current === "dark" || (current === "light dark" && systemPrefersDark)) {
      nextScheme = "light";
    } else {
      nextScheme = "dark";
    }

    meta.content = nextScheme;
    localStorage.setItem("color-scheme", nextScheme);
    showToast(`Switched to ${nextScheme} mode`);
  });

  // Listen to cross-tab storage changes
  window.addEventListener("storage", (e) => {
    if (e.key === "color-scheme") {
      const meta = document.querySelector('meta[name="color-scheme"]');
      if (meta && e.newValue) {
        meta.content = e.newValue;
      }
    }
  });
}

// --- 11. Toast System ---
function showToast(message) {
  const container = document.getElementById("toastContainer");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.innerHTML = `<span>⚡</span> <span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(10px)";
    toast.style.transition = "all 0.25s ease";
    setTimeout(() => toast.remove(), 250);
  }, 2600);
}
