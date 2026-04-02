/* ── ModelIQ Frontend App ──────────────────────────────────── */

const API = 'http://localhost:8000';
let currentFile = null;
let analysisData = null;
let predictColumns = [];

/* ── File Handling ─────────────────────────────────────────── */
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});
fileInput.addEventListener('change', e => { if (e.target.files[0]) handleFile(e.target.files[0]); });

async function handleFile(file) {
  currentFile = file;
  document.getElementById('fileName').textContent = file.name;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch(`${API}/api/columns`, { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) { alert(data.detail || 'Failed to read file.'); return; }

    renderPreview(data.columns, data.preview);

    const sel = document.getElementById('targetSelect');
    sel.innerHTML = '<option value="">— choose target —</option>';
    data.columns.forEach(col => {
      const opt = document.createElement('option');
      opt.value = col;
      opt.textContent = col + ` (${data.dtypes[col]})`;
      sel.appendChild(opt);
    });

    document.getElementById('fileInfo').classList.remove('hidden');
    document.getElementById('uploadSection').querySelector('.upload-area').classList.add('hidden');

  } catch (err) {
    alert('Could not connect to backend. Is the server running?');
  }
}

function renderPreview(columns, rows) {
  const area = document.getElementById('previewArea');
  let html = '<table class="preview-table"><thead><tr>';
  columns.forEach(c => { html += `<th>${esc(c)}</th>`; });
  html += '</tr></thead><tbody>';
  rows.forEach(row => {
    html += '<tr>';
    columns.forEach(c => { html += `<td>${esc(String(row[c] ?? ''))}</td>`; });
    html += '</tr>';
  });
  html += '</tbody></table>';
  area.innerHTML = html;
}

document.getElementById('targetSelect').addEventListener('change', function () {
  document.getElementById('analyzeBtn').disabled = !this.value;
});

function clearFile() {
  currentFile = null;
  fileInput.value = '';
  document.getElementById('fileInfo').classList.add('hidden');
  document.getElementById('uploadSection').querySelector('.upload-area').classList.remove('hidden');
  document.getElementById('analyzeBtn').disabled = true;
}

/* ── Analysis ──────────────────────────────────────────────── */
async function runAnalysis() {
  const targetCol = document.getElementById('targetSelect').value;
  if (!currentFile || !targetCol) return;

  document.getElementById('uploadSection').classList.add('hidden');
  document.getElementById('loadingSection').classList.remove('hidden');
  document.getElementById('resultsSection').classList.add('hidden');
  animatePipeline();

  const formData = new FormData();
  formData.append('file', currentFile);
  formData.append('target_col', targetCol);

  try {
    const res = await fetch(`${API}/api/analyze`, { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      document.getElementById('loadingSection').classList.add('hidden');
      document.getElementById('uploadSection').classList.remove('hidden');
      alert(data.detail || 'Analysis failed.');
      return;
    }

    analysisData = data;
    predictColumns = data.predict_columns || data.dataset_info.original_columns || [];
    renderResults(data);

  } catch (err) {
    document.getElementById('loadingSection').classList.add('hidden');
    document.getElementById('uploadSection').classList.remove('hidden');
    alert('Server error: ' + err.message);
  }
}

let pipelineInterval = null;
function animatePipeline() {
  const steps = ['step1','step2','step3','step4','step5'];
  const msgs = [
    'Detecting problem type from target column statistics...',
    'Applying preprocessing rules: imputation, encoding, scaling...',
    'Running feature intelligence: correlation pruning + mutual info...',
    'Training candidate models with 5-fold cross-validation...',
    'Running Verdict Engine: scoring rubric across 5 dimensions...',
  ];
  let i = 0;
  steps.forEach(s => { document.getElementById(s).className = 'pipeline-step'; });
  document.getElementById(steps[0]).classList.add('active');
  document.getElementById('loadingMsg').textContent = msgs[0];

  pipelineInterval = setInterval(() => {
    document.getElementById(steps[i]).className = 'pipeline-step done';
    i++;
    if (i < steps.length) {
      document.getElementById(steps[i]).classList.add('active');
      document.getElementById('loadingMsg').textContent = msgs[i];
    } else {
      clearInterval(pipelineInterval);
    }
  }, 1800);
}

/* ── Render Results ────────────────────────────────────────── */
function renderResults(data) {
  clearInterval(pipelineInterval);
  document.getElementById('loadingSection').classList.add('hidden');
  document.getElementById('resultsSection').classList.remove('hidden');

  const v = data.verdict;
  const winner = v.winner;
  const winnerData = v.verdicts[winner];

  document.getElementById('winnerName').textContent = winner || 'No eligible model';
  document.getElementById('winnerScore').textContent =
    winnerData ? `Composite Score: ${(winnerData.composite_score * 100).toFixed(2)} / 100` : '';
  document.getElementById('winnerProblem').textContent =
    `Problem type: ${data.problem_detection.problem_type.toUpperCase()} · ` +
    `${data.dataset_info.rows} rows · ${data.feature_analysis.selected_feature_count} features selected`;

  renderLeaderboard(v, data.problem_detection.problem_type);
  renderVerdict(v, data.problem_detection.problem_type);
  renderFeatures(data.feature_analysis, data.explanations);
  renderPipeline(data);
  renderPredictForm(data);
  renderLearnTab(data);
}

/* ── Leaderboard ───────────────────────────────────────────── */
function renderLeaderboard(verdict, problemType) {
  const container = document.getElementById('leaderboardContainer');
  const board = verdict.leaderboard;
  let html = '';

  board.forEach((m, idx) => {
    const isWinner = m.model_name === verdict.winner;
    const isDQ = m.status === 'disqualified';
    const score = m.composite_score || 0;
    const pct = (score * 100).toFixed(1);

    let metricsHtml = '';
    if (!isDQ && m.raw_metrics && m.raw_metrics.status !== 'failed') {
      const rm = m.raw_metrics;
      if (problemType === 'classification') {
        metricsHtml = `
          <span class="metric-chip">Accuracy <span>${(rm.test_accuracy*100).toFixed(1)}%</span></span>
          <span class="metric-chip">F1 <span>${(rm.test_f1_macro*100).toFixed(1)}%</span></span>
          <span class="metric-chip">Overfit Gap <span>${(rm.overfit_gap*100).toFixed(1)}%</span></span>
          <span class="metric-chip">Train Time <span>${rm.training_time_sec}s</span></span>
        `;
      } else {
        metricsHtml = `
          <span class="metric-chip">R² <span>${rm.test_r2.toFixed(4)}</span></span>
          <span class="metric-chip">MAE <span>${rm.test_mae.toFixed(4)}</span></span>
          <span class="metric-chip">Overfit Gap <span>${(rm.overfit_gap*100).toFixed(1)}%</span></span>
          <span class="metric-chip">Train Time <span>${rm.training_time_sec}s</span></span>
        `;
      }
    } else if (isDQ) {
      metricsHtml = `<span class="metric-chip" style="color:var(--red)">${m.disqualification_reasons?.[0] || 'Disqualified'}</span>`;
    }

    const tagClass = isWinner ? 'selected' : (isDQ ? 'disqualified' : 'rejected');
    const tagText = isWinner ? '✓ Selected' : (isDQ ? '✗ Disqualified' : 'Rejected');

    html += `
      <div class="model-card ${isWinner?'winner':''} ${isDQ?'disqualified':''}">
        <div class="model-rank ${idx===0?'first':''}">${idx+1}</div>
        <div>
          <div class="model-info-name">${esc(m.model_name)}</div>
          <div class="model-metrics">${metricsHtml}</div>
          <div class="score-bar-row">
            <div class="score-bar-track"><div class="score-bar-fill" style="width:${pct}%"></div></div>
          </div>
          <span class="verdict-tag ${tagClass}">${tagText}</span>
        </div>
        <div class="model-composite">
          <div class="composite-val">${pct}</div>
          <div class="composite-label">/ 100 pts</div>
        </div>
      </div>`;
  });

  container.innerHTML = html;
}

/* ── Verdict ───────────────────────────────────────────────── */
function renderVerdict(verdict, problemType) {
  const weights = verdict.weights_used;
  const container = document.getElementById('verdictContainer');
  const legend = document.getElementById('rubricLegend');

  const dims = Object.entries(weights).map(([k,v]) => `${k} × ${v}`).join(' + ');
  legend.innerHTML = `
    <div class="rubric-formula">Composite = ${dims}</div>
    <div class="rubric-weights">Each dimension scored 0–1 using deterministic rules. No LLM involvement.</div>`;

  let html = '';
  verdict.leaderboard.forEach(m => {
    const isDQ = m.status === 'disqualified';
    const score = (m.composite_score || 0) * 100;
    const tagClass = m.model_name === verdict.winner ? 'selected' : (isDQ ? 'disqualified' : 'rejected');

    let bodyHtml = '';
    if (!isDQ && m.dimension_scores) {
      bodyHtml += '<table class="dim-table"><thead><tr><th>Dimension</th><th>Score</th><th>Weight</th><th>Pts</th><th>Explanation</th></tr></thead><tbody>';
      Object.entries(m.dimension_scores).forEach(([dim, d]) => {
        if (d.score === null) return;
        bodyHtml += `<tr>
          <td><strong>${esc(dim)}</strong></td>
          <td class="dim-score">${((d.score||0)*100).toFixed(1)}%</td>
          <td>${d.weight}</td>
          <td class="dim-contrib">${((d.contribution||0)*100).toFixed(2)}</td>
          <td class="dim-explain">${esc(d.explanation||'')}</td>
        </tr>`;
      });
      bodyHtml += '</tbody></table>';
    }

    if (m.verdict_rationale) {
      const ratHtml = m.verdict_rationale.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
      bodyHtml += `<div class="rationale-box">${ratHtml}</div>`;
    }

    html += `
      <div class="verdict-card">
        <div class="verdict-card-header" onclick="toggleCard(this)">
          <span class="model-info-name">${esc(m.model_name)}</span>
          <span class="composite-val" style="font-size:1rem">${score.toFixed(1)}</span>
          <span class="verdict-tag ${tagClass}" style="margin:0">${m.verdict||''}</span>
          <span class="log-expand">▼</span>
        </div>
        <div class="verdict-card-body">${bodyHtml}</div>
      </div>`;
  });

  container.innerHTML = html;
}

/* ── Features ──────────────────────────────────────────────── */
function renderFeatures(featureAnalysis, explanations) {
  const container = document.getElementById('featuresContainer');
  let html = '';

  const removed = featureAnalysis.removed_features;
  const allRemoved = [...(removed.low_variance||[]), ...(removed.redundant_correlations||[])];

  html += `<div class="pruning-box">
    <div class="pruning-title">Feature Selection · ${featureAnalysis.original_feature_count} → ${featureAnalysis.selected_feature_count} features</div>
    <div class="chip-list">`;
  featureAnalysis.selected_features.forEach(f => { html += `<span class="chip selected">${esc(f)}</span>`; });
  allRemoved.forEach(f => { html += `<span class="chip removed">${esc(f)}</span>`; });
  html += '</div></div>';

  const ranking = explanations?.shap_values?.length
    ? explanations.shap_values
    : (explanations?.feature_importance || featureAnalysis.importance_ranking || []);

  if (ranking.length > 0) {
    const maxVal = Math.max(...ranking.map(r => r.mean_abs_shap || r.importance || r.mutual_info_score || 0)) || 1;
    const source = explanations?.shap_available ? 'SHAP Values (mean |ϕ|)' : 'Mutual Information Score';

    html += `<div class="section-title" style="margin-top:1.5rem">${source} <span class="badge">Higher = more predictive</span></div>`;
    html += '<div class="feature-bar-list">';
    ranking.slice(0, 15).forEach(r => {
      const val = r.mean_abs_shap || r.importance || r.mutual_info_score || 0;
      const pct = (val / maxVal) * 100;
      const signalLevel = r.signal_level || (pct > 66 ? 'high' : pct > 33 ? 'medium' : 'low');
      html += `
        <div class="feature-bar-item">
          <div class="feature-bar-header">
            <span class="feature-bar-name">#${r.rank} ${esc(r.feature)}<span class="signal-${signalLevel}">[${signalLevel}]</span></span>
            <span class="feature-bar-val">${val.toFixed(5)}</span>
          </div>
          <div class="feature-bar-track"><div class="feature-bar-fill" style="width:${pct}%"></div></div>
        </div>`;
    });
    html += '</div>';
  }

  container.innerHTML = html;
}

/* ── Pipeline ──────────────────────────────────────────────── */
function renderPipeline(data) {
  const container = document.getElementById('pipelineContainer');
  const phases = [
    { phase: 'Problem Detection', steps: [{ step: 'Problem Type Detection', details: data.problem_detection }] },
    { phase: 'Preprocessing', steps: data.preprocessing.steps || [] },
    { phase: 'Feature Intelligence', steps: data.feature_analysis.analysis_steps || [] },
  ];

  let html = '';
  phases.forEach(phase => {
    phase.steps.forEach(step => {
      const name = step.step || step.step_name || phase.phase;
      const details = { ...step };
      delete details.step;
      html += `
        <div class="log-entry">
          <div class="log-entry-header" onclick="toggleLog(this)">
            <span class="log-step-name">${esc(name)}</span>
            <span class="log-phase">${esc(phase.phase)}</span>
            <span class="log-expand">▼</span>
          </div>
          <div class="log-body"><pre>${esc(JSON.stringify(details, null, 2))}</pre></div>
        </div>`;
    });
  });

  container.innerHTML = html;
}

/* ── PREDICT ENGINE ────────────────────────────────────────── */
function renderPredictForm(data) {
  const info = document.getElementById('predictInfo');
  const grid = document.getElementById('predictFormGrid');
  const winner = data.verdict.winner;

  info.innerHTML = `
    <div class="predict-banner">
      <div>
        <div class="predict-banner-title">🎯 Using: <strong>${esc(winner)}</strong></div>
        <div class="predict-banner-sub">
          Problem: <span class="tag-pill">${data.problem_detection.problem_type.toUpperCase()}</span>
          &nbsp;·&nbsp; Target: <span class="tag-pill">${esc(data.dataset_info.target_column)}</span>
          &nbsp;·&nbsp; Fill in values below and click Run Prediction
        </div>
      </div>
    </div>`;

  const cols = predictColumns;
  if (!cols || cols.length === 0) {
    grid.innerHTML = '<p style="color:var(--text-dim);font-family:var(--mono);font-size:0.85rem">No input columns found.</p>';
    return;
  }

  let html = '';
  cols.forEach(col => {
    html += `
      <div class="predict-field">
        <label class="predict-label">${esc(col)}</label>
        <input class="predict-input" type="text" id="pred_${esc(col)}" placeholder="Enter value…" />
      </div>`;
  });
  grid.innerHTML = html;

  // Reset result
  document.getElementById('predictResult').classList.add('hidden');
  document.getElementById('predictResult').innerHTML = '';
}

async function runPrediction() {
  const btn = document.getElementById('predictBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Predicting...';

  const inputs = {};
  predictColumns.forEach(col => {
    const el = document.getElementById(`pred_${col}`);
    inputs[col] = el ? el.value : '';
  });

  try {
    const res = await fetch(`${API}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ inputs }),
    });
    const data = await res.json();

    const resultBox = document.getElementById('predictResult');
    resultBox.classList.remove('hidden');

    if (!res.ok) {
      resultBox.innerHTML = `<div class="predict-error">❌ ${esc(data.detail || 'Prediction failed')}</div>`;
    } else {
      let confHtml = '';
      if (data.confidence !== null && data.confidence !== undefined) {
        const confPct = (data.confidence * 100).toFixed(1);
        confHtml = `<div class="predict-confidence">Confidence: <strong>${confPct}%</strong></div>`;
      }

      let probaHtml = '';
      if (data.class_probabilities) {
        probaHtml = '<div class="proba-grid">';
        Object.entries(data.class_probabilities)
          .sort((a,b) => b[1]-a[1])
          .forEach(([cls, prob]) => {
            const pct = (prob * 100).toFixed(1);
            probaHtml += `
              <div class="proba-item">
                <span class="proba-label">${esc(cls)}</span>
                <div class="proba-bar-track"><div class="proba-bar-fill" style="width:${pct}%"></div></div>
                <span class="proba-val">${pct}%</span>
              </div>`;
          });
        probaHtml += '</div>';
      }

      resultBox.innerHTML = `
        <div class="predict-result-card">
          <div class="predict-result-label">Predicted ${esc(data.target_column)}</div>
          <div class="predict-result-value">${esc(String(data.prediction))}</div>
          ${confHtml}
          <div class="predict-model-used">Model: ${esc(data.model_used)}</div>
          ${probaHtml}
        </div>`;
    }
  } catch (err) {
    document.getElementById('predictResult').innerHTML = `<div class="predict-error">❌ ${esc(err.message)}</div>`;
    document.getElementById('predictResult').classList.remove('hidden');
  }

  btn.disabled = false;
  btn.textContent = '⚡ Run Prediction';
}

/* ── LEARN TAB ─────────────────────────────────────────────── */
const GLOSSARY = [
  {
    term: "Classification",
    icon: "🏷️",
    simple: "Putting things into named boxes.",
    full: "The model tries to predict which category (class) something belongs to. For example: is this email spam or not? Is this tumor benign or malignant? The answer is always one of a fixed set of labels.",
    example: "Input: flower measurements → Output: 'Setosa', 'Versicolor', or 'Virginica'",
    color: "accent"
  },
  {
    term: "Regression",
    icon: "📈",
    simple: "Predicting a number.",
    full: "The model tries to predict a continuous numerical value. Unlike classification where you pick a box, here you're estimating any number on a scale.",
    example: "Input: house features (size, location, age) → Output: predicted price like ₹45,00,000",
    color: "green"
  },
  {
    term: "Cross-Validation (CV)",
    icon: "🔄",
    simple: "Testing the model honestly by hiding parts of the data.",
    full: "Instead of training on all data and testing on the same data (which would be cheating), we split data into 5 parts (folds). Train on 4 parts, test on 1. Repeat 5 times. Average the results. This gives a trustworthy accuracy number.",
    example: "5-fold CV: data split into fold 1,2,3,4,5. Each fold gets a turn being the test set.",
    color: "accent2"
  },
  {
    term: "Overfitting",
    icon: "📚",
    simple: "The model memorized the answers instead of learning the rules.",
    full: "A model that overfits does amazingly on training data but fails on new data because it just memorized the training examples. Like a student who memorizes last year's question paper word-for-word but can't answer a new question.",
    example: "Train accuracy: 99%, Test accuracy: 60% → severe overfitting. The model memorized noise.",
    color: "red"
  },
  {
    term: "Accuracy",
    icon: "🎯",
    simple: "What % of predictions were correct.",
    full: "Out of all predictions made, how many were right. Simple and intuitive but can be misleading if one class has way more samples than others (class imbalance).",
    example: "100 predictions, 87 correct → Accuracy = 87%",
    color: "green"
  },
  {
    term: "F1 Score (Macro)",
    icon: "⚖️",
    simple: "A fairer accuracy that balances all categories.",
    full: "When you have imbalanced classes (e.g. 95% healthy, 5% sick), accuracy alone is misleading. F1 balances Precision (how many of your positives were real) and Recall (how many real positives you found). Macro-F1 averages this across all classes equally.",
    example: "If the model finds 8 out of 10 cats correctly and wrongly calls 2 dogs as cats, that's balanced by F1.",
    color: "yellow"
  },
  {
    term: "R² Score",
    icon: "📊",
    simple: "How much of the variation in the data does the model explain.",
    full: "R² (R-squared) ranges from 0 to 1 (sometimes negative for bad models). A score of 0.85 means the model explains 85% of why values vary. A score of 0 means it's no better than just guessing the average every time.",
    example: "R² = 0.90 → excellent. R² = 0.30 → model barely captures the pattern. R² < 0 → worse than baseline.",
    color: "accent"
  },
  {
    term: "MAE (Mean Absolute Error)",
    icon: "📏",
    simple: "On average, how far off were your predictions.",
    full: "MAE is the average of all absolute differences between predictions and actual values. Easier to interpret than other error metrics because it's in the same unit as your target.",
    example: "If predicting house prices, MAE = ₹2,00,000 means predictions are off by ₹2L on average.",
    color: "yellow"
  },
  {
    term: "SHAP Values",
    icon: "🔬",
    simple: "A fair way to see which features pushed the prediction up or down.",
    full: "SHAP (SHapley Additive exPlanations) comes from game theory. It fairly distributes credit for a prediction among all input features. A high positive SHAP value for a feature means it pushed the prediction higher; negative means it pushed it lower.",
    example: "For a loan rejection: 'income: -0.3 (pushed toward reject), 'credit_score: +0.5 (pushed toward approve)'",
    color: "accent2"
  },
  {
    term: "Mutual Information",
    icon: "🔗",
    simple: "How much does knowing this feature help predict the target?",
    full: "Mutual information measures the dependency between a feature and the target. A score of 0 means the feature tells you nothing about the target. Higher = more useful. It captures non-linear relationships too, unlike correlation.",
    example: "Feature 'age' MI = 0.45 (strong signal), 'zip_code' MI = 0.01 (near useless) → drop zip_code",
    color: "green"
  },
  {
    term: "Feature",
    icon: "🧩",
    simple: "One column of input data — one piece of information about each row.",
    full: "Features are the input variables your model learns from. Every column in your dataset (except the target) is a feature. Good features contain signal about the target; bad features add noise.",
    example: "Predicting salary → features: years_experience, education_level, city, role",
    color: "accent"
  },
  {
    term: "Imputation",
    icon: "🩹",
    simple: "Filling in missing values smartly.",
    full: "Real datasets often have missing values (blanks, NaN). Imputation fills them using a strategy. Median imputation replaces missing numeric values with the column's median — robust to outliers.",
    example: "Age column has 5 missing values. Median of rest = 34. Fill all blanks with 34.",
    color: "text-dim"
  },
  {
    term: "Label Encoding",
    icon: "🔢",
    simple: "Turning words into numbers so the model can understand them.",
    full: "Machine learning models work with numbers, not text. Label encoding converts each unique category string to an integer. The mapping is consistent — same word always gets same number.",
    example: "'cat'→0, 'dog'→1, 'fish'→2. So 'dog' in any row always becomes 1.",
    color: "yellow"
  },
  {
    term: "StandardScaler",
    icon: "⚖️",
    simple: "Putting all numbers on the same scale.",
    full: "Some features have values in thousands (salary) while others are 0-1 (probability). Without scaling, models like KNN or Logistic Regression get biased toward large-scale features. StandardScaler shifts every feature to have mean=0 and std=1.",
    example: "Height in cm (150-190) and Weight in kg (50-90) → both scaled to roughly -2 to +2",
    color: "accent2"
  },
  {
    term: "Random Forest",
    icon: "🌲",
    simple: "100 decision trees voting together for a smarter answer.",
    full: "A Random Forest builds many decision trees, each trained on a slightly different random subset of data and features. The final prediction is a majority vote (classification) or average (regression) of all trees. More robust and accurate than a single tree.",
    example: "100 trees each vote: 67 say 'spam', 33 say 'not spam' → prediction: spam",
    color: "green"
  },
  {
    term: "Decision Tree",
    icon: "🌳",
    simple: "A flowchart of yes/no questions that leads to a prediction.",
    full: "A decision tree splits data based on feature thresholds at each node. Highly interpretable — you can literally follow the path from root to leaf to understand a prediction. Prone to overfitting if too deep.",
    example: "Age > 30? → Yes → Income > 50k? → Yes → Predict: approved",
    color: "accent"
  },
  {
    term: "Gradient Boosting",
    icon: "🚀",
    simple: "Many weak models learning from each other's mistakes.",
    full: "Boosting builds models sequentially. Each new model focuses on correcting the errors of the previous one. The result is highly accurate but harder to interpret and slower to train than Random Forest.",
    example: "Model 1 predicts price ₹40L but actual is ₹50L. Model 2 learns that error of ₹10L and corrects it.",
    color: "accent2"
  },
  {
    term: "Composite Score",
    icon: "🏆",
    simple: "ModelIQ's overall grade for a model across multiple dimensions.",
    full: "Instead of just picking the most accurate model, ModelIQ scores each model across 5 dimensions: accuracy/R², F1/MAE, overfitting, speed, and interpretability. Each dimension is weighted and combined into one final score.",
    example: "Random Forest: accuracy 0.89×0.30 + F1 0.87×0.30 + overfit 0.95×0.25 + speed 0.80×0.05 + interp 0.60×0.10 = 0.857",
    color: "yellow"
  },
  {
    term: "Confidence (Prediction)",
    icon: "💡",
    simple: "How sure is the model about its prediction?",
    full: "For classification, most models can output the probability of each class. The confidence is the probability assigned to the predicted class. 95% confidence means the model is very sure; 51% means it barely chose one class over another.",
    example: "Predicted: 'spam' with 91% confidence → the model is quite certain this is spam.",
    color: "green"
  },
];

const ML_ROADMAP = [
  {
    step: 1,
    title: "Understand Your Data",
    icon: "🔍",
    actions: [
      "Look at the first few rows — what does each column mean?",
      "Check for missing values (blank cells) in each column",
      "Understand your target column — is it a category or a number?",
      "Plot distributions: are your numeric columns normally spread or skewed?",
    ],
    tools: "pandas df.describe(), df.info(), df.isnull().sum()"
  },
  {
    step: 2,
    title: "Clean & Preprocess",
    icon: "🧹",
    actions: [
      "Drop columns with >50% missing values (too little data to help)",
      "Fill remaining missing numbers with median (robust to outliers)",
      "Encode text/category columns as numbers (Label Encoding or One-Hot)",
      "Scale numeric features to the same range (StandardScaler)",
      "Remove duplicate rows",
    ],
    tools: "sklearn.preprocessing, sklearn.impute, pandas"
  },
  {
    step: 3,
    title: "Engineer Features",
    icon: "⚙️",
    actions: [
      "Remove features that are nearly constant (low variance = no info)",
      "Remove features that are highly correlated with each other (redundant)",
      "Score remaining features by how much they predict the target (Mutual Info)",
      "Consider creating new features from combinations (date → day of week, etc.)",
    ],
    tools: "sklearn.feature_selection, pandas"
  },
  {
    step: 4,
    title: "Select & Train Models",
    icon: "🏋️",
    actions: [
      "Start simple: Logistic Regression or Decision Tree (interpretable baselines)",
      "Try ensemble models: Random Forest, Gradient Boosting",
      "Always use cross-validation (5-fold) — never train and test on same data",
      "Compare multiple models objectively using the same metric",
    ],
    tools: "sklearn.model_selection, sklearn.ensemble"
  },
  {
    step: 5,
    title: "Evaluate Honestly",
    icon: "📊",
    actions: [
      "For classification: check Accuracy AND F1 (especially with imbalanced classes)",
      "For regression: check R² AND MAE (both in context)",
      "Check the overfit gap: if train score >> test score, you're overfitting",
      "Look at confusion matrix for classification to see where model fails",
    ],
    tools: "sklearn.metrics, seaborn heatmap"
  },
  {
    step: 6,
    title: "Explain Your Model",
    icon: "🔬",
    actions: [
      "Use SHAP to see which features drove each prediction",
      "For tree models, visualize feature importance",
      "For linear models, look at coefficients",
      "Ask: does this make real-world sense?",
    ],
    tools: "shap library, sklearn feature_importances_"
  },
  {
    step: 7,
    title: "Iterate & Improve",
    icon: "🔁",
    actions: [
      "Collect more data if accuracy is still low",
      "Try hyperparameter tuning (GridSearchCV, RandomSearchCV)",
      "Try feature engineering — sometimes a smart new column beats a complex model",
      "Consider advanced models (XGBoost, LightGBM) for harder problems",
    ],
    tools: "sklearn.model_selection.GridSearchCV, optuna"
  },
  {
    step: 8,
    title: "Deploy & Monitor",
    icon: "🚀",
    actions: [
      "Save your trained model (joblib or pickle)",
      "Build an API endpoint to serve predictions",
      "Monitor model performance over time — data can drift",
      "Retrain periodically with fresh data",
    ],
    tools: "joblib, FastAPI, Flask, mlflow"
  },
];

function renderLearnTab(data) {
  const container = document.getElementById('learnContainer');
  
  // Context-aware terms to highlight based on the analysis
  const problemType = data?.problem_detection?.problem_type;
  const winnerModel = data?.verdict?.winner;

  let html = '';

  // ── Roadmap section ────────────────────────────────────────
  html += `
    <div class="learn-section">
      <div class="learn-section-title">🗺️ ML Training Roadmap <span class="badge">8 Steps from Data to Deployment</span></div>
      <p class="learn-intro">Follow these steps to go from raw data to a trustworthy, deployed machine learning model.</p>
      <div class="roadmap-grid">`;

  ML_ROADMAP.forEach(step => {
    html += `
      <div class="roadmap-step">
        <div class="roadmap-step-header">
          <span class="roadmap-num">${step.step}</span>
          <span class="roadmap-icon">${step.icon}</span>
          <span class="roadmap-title">${esc(step.title)}</span>
        </div>
        <ul class="roadmap-actions">
          ${step.actions.map(a => `<li>${esc(a)}</li>`).join('')}
        </ul>
        <div class="roadmap-tools">🛠 ${esc(step.tools)}</div>
      </div>`;
  });

  html += `</div></div>`;

  // ── Glossary section ───────────────────────────────────────
  html += `
    <div class="learn-section">
      <div class="learn-section-title">📖 Glossary <span class="badge">Plain-English Definitions</span></div>
      <p class="learn-intro">Every term ModelIQ uses, explained as simply as possible.</p>
      <div class="glossary-grid">`;

  GLOSSARY.forEach(term => {
    const isRelevant = 
      (problemType === 'classification' && ['Classification','Accuracy','F1 Score (Macro)','Confidence (Prediction)'].includes(term.term)) ||
      (problemType === 'regression' && ['Regression','R² Score','MAE (Mean Absolute Error)'].includes(term.term)) ||
      (winnerModel && winnerModel.includes(term.term.split(' ')[0]));

    html += `
      <div class="glossary-card ${isRelevant ? 'relevant' : ''}">
        <div class="glossary-header">
          <span class="glossary-icon">${term.icon}</span>
          <span class="glossary-term">${esc(term.term)}</span>
          ${isRelevant ? '<span class="relevant-badge">Used in your analysis</span>' : ''}
        </div>
        <div class="glossary-simple">${esc(term.simple)}</div>
        <div class="glossary-full hidden" id="gloss_${term.term.replace(/[^a-z]/gi,'_')}">
          <p>${esc(term.full)}</p>
          <div class="glossary-example">💡 Example: ${esc(term.example)}</div>
        </div>
        <button class="glossary-toggle" onclick="toggleGlossary('${term.term.replace(/[^a-z]/gi,'_')}', this)">Read more ▼</button>
      </div>`;
  });

  html += `</div></div>`;

  container.innerHTML = html;
}

function toggleGlossary(id, btn) {
  const el = document.getElementById(`gloss_${id}`);
  const open = el.classList.toggle('hidden');
  btn.textContent = open ? 'Read more ▼' : 'Show less ▲';
}

/* ── PDF Export ────────────────────────────────────────────── */
async function exportPDF() {
  if (!analysisData) return;
  const btn = document.getElementById('exportBtn');
  btn.textContent = '⏳ Generating PDF…';
  btn.disabled = true;

  try {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF({ unit: 'mm', format: 'a4' });
    const data = analysisData;
    const v = data.verdict;
    const winner = v.winner;
    const winnerData = v.verdicts[winner];
    const problemType = data.problem_detection.problem_type;

    const W = 210;
    const margin = 18;
    let y = 20;

    const addText = (text, size, bold, color) => {
      doc.setFontSize(size);
      doc.setFont('helvetica', bold ? 'bold' : 'normal');
      doc.setTextColor(...(color || [30, 30, 30]));
      doc.text(String(text), margin, y);
      y += size * 0.45 + 3;
    };

    const addLine = () => {
      doc.setDrawColor(200, 200, 200);
      doc.line(margin, y, W - margin, y);
      y += 4;
    };

    const checkPage = (needed = 20) => {
      if (y + needed > 280) { doc.addPage(); y = 20; }
    };

    // ── Cover ──────────────────────────────────────────────
    doc.setFillColor(10, 12, 16);
    doc.rect(0, 0, W, 60, 'F');
    doc.setFontSize(26); doc.setFont('helvetica', 'bold');
    doc.setTextColor(0, 212, 255);
    doc.text('ModelIQ', margin, 28);
    doc.setFontSize(11); doc.setFont('helvetica', 'normal');
    doc.setTextColor(200, 210, 220);
    doc.text('Explainable AutoML Analysis Report', margin, 38);
    doc.setFontSize(9);
    doc.setTextColor(100, 120, 140);
    doc.text(`Generated: ${new Date().toLocaleString()}`, margin, 50);
    y = 72;

    // ── Dataset Summary ────────────────────────────────────
    addText('Dataset Summary', 14, true, [20, 80, 160]);
    addLine();
    addText(`Rows: ${data.dataset_info.rows}   |   Columns: ${data.dataset_info.columns}   |   Target: ${data.dataset_info.target_column}`, 10, false);
    addText(`Problem Type: ${problemType.toUpperCase()}   |   Rule: ${data.problem_detection.rule_triggered}`, 10, false);
    y += 4;

    // ── Winner ─────────────────────────────────────────────
    checkPage(40);
    addText('Selected Model', 14, true, [20, 80, 160]);
    addLine();
    addText(winner, 18, true, [0, 180, 80]);
    if (winnerData) {
      addText(`Composite Score: ${(winnerData.composite_score * 100).toFixed(2)} / 100`, 11, false);
    }
    y += 4;

    // ── Leaderboard ────────────────────────────────────────
    checkPage(50);
    addText('Model Leaderboard', 14, true, [20, 80, 160]);
    addLine();
    v.leaderboard.forEach(m => {
      checkPage(12);
      const score = ((m.composite_score || 0) * 100).toFixed(1);
      const status = m.model_name === winner ? '✓ SELECTED' : (m.status === 'disqualified' ? '✗ DQ' : 'Rejected');
      doc.setFontSize(9); doc.setFont('helvetica', 'normal'); doc.setTextColor(40, 40, 40);
      doc.text(`${m.model_name}  —  ${score} pts  —  ${status}`, margin + 2, y);
      y += 6;
    });
    y += 4;

    // ── Dimension Scores for Winner ────────────────────────
    if (winnerData && winnerData.dimension_scores) {
      checkPage(60);
      addText(`Verdict Breakdown: ${winner}`, 14, true, [20, 80, 160]);
      addLine();
      Object.entries(winnerData.dimension_scores).forEach(([dim, d]) => {
        if (d.score === null) return;
        checkPage(10);
        doc.setFontSize(9); doc.setFont('helvetica', 'normal'); doc.setTextColor(40, 40, 40);
        doc.text(`${dim}: ${((d.score||0)*100).toFixed(1)}% × ${d.weight} weight = ${((d.contribution||0)*100).toFixed(2)} pts  —  ${d.explanation || ''}`, margin + 2, y, { maxWidth: W - margin * 2 - 4 });
        y += 7;
      });
      y += 4;
    }

    // ── Feature Analysis ───────────────────────────────────
    checkPage(40);
    addText('Feature Analysis', 14, true, [20, 80, 160]);
    addLine();
    addText(`Selected: ${data.feature_analysis.selected_features.join(', ')}`, 9, false);
    const removed = data.feature_analysis.removed_features;
    const allR = [...(removed.low_variance||[]), ...(removed.redundant_correlations||[])];
    if (allR.length) addText(`Removed: ${allR.join(', ')}`, 9, false, [160, 60, 60]);
    y += 4;

    const ranking = data.explanations?.shap_values?.length
      ? data.explanations.shap_values
      : (data.explanations?.feature_importance || data.feature_analysis.importance_ranking || []);
    if (ranking.length) {
      checkPage(40);
      addText('Top Features (by importance)', 11, true);
      ranking.slice(0, 8).forEach(r => {
        checkPage(8);
        const val = r.mean_abs_shap || r.importance || r.mutual_info_score || 0;
        doc.setFontSize(9); doc.setFont('helvetica', 'normal'); doc.setTextColor(40, 40, 40);
        doc.text(`#${r.rank}  ${r.feature}  —  ${val.toFixed(5)}`, margin + 2, y);
        y += 6;
      });
    }
    y += 6;

    // ── Footer on last page ────────────────────────────────
    doc.setFontSize(8); doc.setFont('helvetica', 'italic'); doc.setTextColor(160, 160, 160);
    doc.text('ModelIQ · Deterministic AutoML · CodeWiser × VJTI Hackathon 2026', margin, 290);

    doc.save(`ModelIQ_Report_${winner.replace(/ /g,'_')}.pdf`);
  } catch (err) {
    alert('PDF export failed: ' + err.message);
  }

  btn.textContent = '⬇ Export Report PDF';
  btn.disabled = false;
}

/* ── UI Helpers ────────────────────────────────────────────── */
function showTab(name, btnEl) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById(`tab-${name}`).classList.add('active');
  if (btnEl) btnEl.classList.add('active');
  else document.getElementById(`tabBtn-${name}`)?.classList.add('active');
}

function toggleCard(header) {
  const body = header.nextElementSibling;
  const open = body.classList.toggle('open');
  header.querySelector('.log-expand').textContent = open ? '▲' : '▼';
}

function toggleLog(header) {
  const body = header.nextElementSibling;
  const open = body.classList.toggle('open');
  header.querySelector('.log-expand').textContent = open ? '▲' : '▼';
}

function resetApp() {
  currentFile = null; analysisData = null; predictColumns = [];
  fileInput.value = '';
  document.getElementById('fileInfo').classList.add('hidden');
  document.getElementById('uploadSection').querySelector('.upload-area').classList.remove('hidden');
  document.getElementById('uploadSection').classList.remove('hidden');
  document.getElementById('resultsSection').classList.add('hidden');
  document.getElementById('analyzeBtn').disabled = true;
}

function esc(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#039;');
}

/* ── Load jsPDF for PDF export ─────────────────────────────── */
(function loadJsPDF() {
  if (!window.jspdf) {
    const s = document.createElement('script');
    s.src = 'https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js';
    document.head.appendChild(s);
  }
})();