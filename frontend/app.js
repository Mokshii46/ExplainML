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
  { term: "Classification", icon: "🏷️", simple: "Putting things into named boxes.", full: "The model tries to predict which category (class) something belongs to. For example: is this email spam or not? Is this tumor benign or malignant? The answer is always one of a fixed set of labels.", example: "Input: flower measurements → Output: 'Setosa', 'Versicolor', or 'Virginica'", color: "accent" },
  { term: "Regression", icon: "📈", simple: "Predicting a number.", full: "The model tries to predict a continuous numerical value. Unlike classification where you pick a box, here you're estimating any number on a scale.", example: "Input: house features (size, location, age) → Output: predicted price like ₹45,00,000", color: "green" },
  { term: "Cross-Validation (CV)", icon: "🔄", simple: "Testing the model honestly by hiding parts of the data.", full: "Instead of training on all data and testing on the same data (which would be cheating), we split data into 5 parts (folds). Train on 4 parts, test on 1. Repeat 5 times. Average the results. This gives a trustworthy accuracy number.", example: "5-fold CV: data split into fold 1,2,3,4,5. Each fold gets a turn being the test set.", color: "accent2" },
  { term: "Overfitting", icon: "📚", simple: "The model memorized the answers instead of learning the rules.", full: "A model that overfits does amazingly on training data but fails on new data because it just memorized the training examples. Like a student who memorizes last year's question paper word-for-word but can't answer a new question.", example: "Train accuracy: 99%, Test accuracy: 60% → severe overfitting. The model memorized noise.", color: "red" },
  { term: "Accuracy", icon: "🎯", simple: "What % of predictions were correct.", full: "Out of all predictions made, how many were right. Simple and intuitive but can be misleading if one class has way more samples than others (class imbalance).", example: "100 predictions, 87 correct → Accuracy = 87%", color: "green" },
  { term: "F1 Score (Macro)", icon: "⚖️", simple: "A fairer accuracy that balances all categories.", full: "When you have imbalanced classes (e.g. 95% healthy, 5% sick), accuracy alone is misleading. F1 balances Precision and Recall. Macro-F1 averages this across all classes equally.", example: "If the model finds 8 out of 10 cats correctly and wrongly calls 2 dogs as cats, that's balanced by F1.", color: "yellow" },
  { term: "R² Score", icon: "📊", simple: "How much of the variation in the data does the model explain.", full: "R² (R-squared) ranges from 0 to 1. A score of 0.85 means the model explains 85% of why values vary. A score of 0 means it's no better than just guessing the average every time.", example: "R² = 0.90 → excellent. R² = 0.30 → model barely captures the pattern. R² < 0 → worse than baseline.", color: "accent" },
  { term: "MAE (Mean Absolute Error)", icon: "📏", simple: "On average, how far off were your predictions.", full: "MAE is the average of all absolute differences between predictions and actual values. Easier to interpret than other error metrics because it's in the same unit as your target.", example: "If predicting house prices, MAE = ₹2,00,000 means predictions are off by ₹2L on average.", color: "yellow" },
  { term: "SHAP Values", icon: "🔬", simple: "A fair way to see which features pushed the prediction up or down.", full: "SHAP (SHapley Additive exPlanations) comes from game theory. It fairly distributes credit for a prediction among all input features. A high positive SHAP value for a feature means it pushed the prediction higher; negative means it pushed it lower.", example: "For a loan rejection: 'income: -0.3 (pushed toward reject)', 'credit_score: +0.5 (pushed toward approve)'", color: "accent2" },
  { term: "Mutual Information", icon: "🔗", simple: "How much does knowing this feature help predict the target?", full: "Mutual information measures the dependency between a feature and the target. A score of 0 means the feature tells you nothing about the target. Higher = more useful. It captures non-linear relationships too, unlike correlation.", example: "Feature 'age' MI = 0.45 (strong signal), 'zip_code' MI = 0.01 (near useless) → drop zip_code", color: "green" },
  { term: "Feature", icon: "🧩", simple: "One column of input data — one piece of information about each row.", full: "Features are the input variables your model learns from. Every column in your dataset (except the target) is a feature. Good features contain signal about the target; bad features add noise.", example: "Predicting salary → features: years_experience, education_level, city, role", color: "accent" },
  { term: "Imputation", icon: "🩹", simple: "Filling in missing values smartly.", full: "Real datasets often have missing values (blanks, NaN). Imputation fills them using a strategy. Median imputation replaces missing numeric values with the column's median — robust to outliers.", example: "Age column has 5 missing values. Median of rest = 34. Fill all blanks with 34.", color: "text-dim" },
  { term: "Label Encoding", icon: "🔢", simple: "Turning words into numbers so the model can understand them.", full: "Machine learning models work with numbers, not text. Label encoding converts each unique category string to an integer. The mapping is consistent — same word always gets same number.", example: "'cat'→0, 'dog'→1, 'fish'→2. So 'dog' in any row always becomes 1.", color: "yellow" },
  { term: "StandardScaler", icon: "⚖️", simple: "Putting all numbers on the same scale.", full: "Some features have values in thousands (salary) while others are 0-1 (probability). Without scaling, models like KNN or Logistic Regression get biased toward large-scale features. StandardScaler shifts every feature to have mean=0 and std=1.", example: "Height in cm (150-190) and Weight in kg (50-90) → both scaled to roughly -2 to +2", color: "accent2" },
  { term: "Random Forest", icon: "🌲", simple: "100 decision trees voting together for a smarter answer.", full: "A Random Forest builds many decision trees, each trained on a slightly different random subset of data and features. The final prediction is a majority vote (classification) or average (regression) of all trees.", example: "100 trees each vote: 67 say 'spam', 33 say 'not spam' → prediction: spam", color: "green" },
  { term: "Decision Tree", icon: "🌳", simple: "A flowchart of yes/no questions that leads to a prediction.", full: "A decision tree splits data based on feature thresholds at each node. Highly interpretable — you can literally follow the path from root to leaf to understand a prediction. Prone to overfitting if too deep.", example: "Age > 30? → Yes → Income > 50k? → Yes → Predict: approved", color: "accent" },
  { term: "Gradient Boosting", icon: "🚀", simple: "Many weak models learning from each other's mistakes.", full: "Boosting builds models sequentially. Each new model focuses on correcting the errors of the previous one. The result is highly accurate but harder to interpret and slower to train than Random Forest.", example: "Model 1 predicts price ₹40L but actual is ₹50L. Model 2 learns that error of ₹10L and corrects it.", color: "accent2" },
  { term: "Composite Score", icon: "🏆", simple: "ModelIQ's overall grade for a model across multiple dimensions.", full: "Instead of just picking the most accurate model, ModelIQ scores each model across 5 dimensions: accuracy/R², F1/MAE, overfitting, speed, and interpretability. Each dimension is weighted and combined into one final score.", example: "Random Forest: accuracy 0.89×0.30 + F1 0.87×0.30 + overfit 0.95×0.25 + speed 0.80×0.05 + interp 0.60×0.10 = 0.857", color: "yellow" },
  { term: "Confidence (Prediction)", icon: "💡", simple: "How sure is the model about its prediction?", full: "For classification, most models can output the probability of each class. The confidence is the probability assigned to the predicted class. 95% confidence means the model is very sure; 51% means it barely chose one class over another.", example: "Predicted: 'spam' with 91% confidence → the model is quite certain this is spam.", color: "green" },
];

const ML_ROADMAP = [
  { step: 1, title: "Understand Your Data", icon: "🔍", actions: ["Look at the first few rows — what does each column mean?","Check for missing values (blank cells) in each column","Understand your target column — is it a category or a number?","Plot distributions: are your numeric columns normally spread or skewed?"], tools: "pandas df.describe(), df.info(), df.isnull().sum()" },
  { step: 2, title: "Clean & Preprocess", icon: "🧹", actions: ["Drop columns with >50% missing values (too little data to help)","Fill remaining missing numbers with median (robust to outliers)","Encode text/category columns as numbers (Label Encoding or One-Hot)","Scale numeric features to the same range (StandardScaler)","Remove duplicate rows"], tools: "sklearn.preprocessing, sklearn.impute, pandas" },
  { step: 3, title: "Engineer Features", icon: "⚙️", actions: ["Remove features that are nearly constant (low variance = no info)","Remove features that are highly correlated with each other (redundant)","Score remaining features by how much they predict the target (Mutual Info)","Consider creating new features from combinations (date → day of week, etc.)"], tools: "sklearn.feature_selection, pandas" },
  { step: 4, title: "Select & Train Models", icon: "🏋️", actions: ["Start simple: Logistic Regression or Decision Tree (interpretable baselines)","Try ensemble models: Random Forest, Gradient Boosting","Always use cross-validation (5-fold) — never train and test on same data","Compare multiple models objectively using the same metric"], tools: "sklearn.model_selection, sklearn.ensemble" },
  { step: 5, title: "Evaluate Honestly", icon: "📊", actions: ["For classification: check Accuracy AND F1 (especially with imbalanced classes)","For regression: check R² AND MAE (both in context)","Check the overfit gap: if train score >> test score, you're overfitting","Look at confusion matrix for classification to see where model fails"], tools: "sklearn.metrics, seaborn heatmap" },
  { step: 6, title: "Explain Your Model", icon: "🔬", actions: ["Use SHAP to see which features drove each prediction","For tree models, visualize feature importance","For linear models, look at coefficients","Ask: does this make real-world sense?"], tools: "shap library, sklearn feature_importances_" },
  { step: 7, title: "Iterate & Improve", icon: "🔁", actions: ["Collect more data if accuracy is still low","Try hyperparameter tuning (GridSearchCV, RandomSearchCV)","Try feature engineering — sometimes a smart new column beats a complex model","Consider advanced models (XGBoost, LightGBM) for harder problems"], tools: "sklearn.model_selection.GridSearchCV, optuna" },
  { step: 8, title: "Deploy & Monitor", icon: "🚀", actions: ["Save your trained model (joblib or pickle)","Build an API endpoint to serve predictions","Monitor model performance over time — data can drift","Retrain periodically with fresh data"], tools: "joblib, FastAPI, Flask, mlflow" },
];

function renderLearnTab(data) {
  const container = document.getElementById('learnContainer');
  const problemType = data?.problem_detection?.problem_type;
  const winnerModel = data?.verdict?.winner;
  let html = '';

  html += `<div class="learn-section"><div class="learn-section-title">🗺️ ML Training Roadmap <span class="badge">8 Steps from Data to Deployment</span></div><p class="learn-intro">Follow these steps to go from raw data to a trustworthy, deployed machine learning model.</p><div class="roadmap-grid">`;
  ML_ROADMAP.forEach(step => {
    html += `<div class="roadmap-step"><div class="roadmap-step-header"><span class="roadmap-num">${step.step}</span><span class="roadmap-icon">${step.icon}</span><span class="roadmap-title">${esc(step.title)}</span></div><ul class="roadmap-actions">${step.actions.map(a => `<li>${esc(a)}</li>`).join('')}</ul><div class="roadmap-tools">🛠 ${esc(step.tools)}</div></div>`;
  });
  html += `</div></div>`;

  html += `<div class="learn-section"><div class="learn-section-title">📖 Glossary <span class="badge">Plain-English Definitions</span></div><p class="learn-intro">Every term ModelIQ uses, explained as simply as possible.</p><div class="glossary-grid">`;
  GLOSSARY.forEach(term => {
    const isRelevant =
      (problemType === 'classification' && ['Classification','Accuracy','F1 Score (Macro)','Confidence (Prediction)'].includes(term.term)) ||
      (problemType === 'regression' && ['Regression','R² Score','MAE (Mean Absolute Error)'].includes(term.term)) ||
      (winnerModel && winnerModel.includes(term.term.split(' ')[0]));
    html += `<div class="glossary-card ${isRelevant ? 'relevant' : ''}"><div class="glossary-header"><span class="glossary-icon">${term.icon}</span><span class="glossary-term">${esc(term.term)}</span>${isRelevant ? '<span class="relevant-badge">Used in your analysis</span>' : ''}</div><div class="glossary-simple">${esc(term.simple)}</div><div class="glossary-full hidden" id="gloss_${term.term.replace(/[^a-z]/gi,'_')}"><p>${esc(term.full)}</p><div class="glossary-example">💡 Example: ${esc(term.example)}</div></div><button class="glossary-toggle" onclick="toggleGlossary('${term.term.replace(/[^a-z]/gi,'_')}', this)">Read more ▼</button></div>`;
  });
  html += `</div></div>`;
  container.innerHTML = html;
}

function toggleGlossary(id, btn) {
  const el = document.getElementById(`gloss_${id}`);
  const open = el.classList.toggle('hidden');
  btn.textContent = open ? 'Read more ▼' : 'Show less ▲';
}

/* ── PDF Export v4 — Fixed logo, fixed score bar overlap ────────────────────── */
async function exportPDF() {
  if (!analysisData) return;
  const btn = document.getElementById('exportBtn');
  btn.textContent = 'Generating PDF...';
  btn.disabled = true;

  try {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF({ unit: 'mm', format: 'a4', orientation: 'portrait' });
    const data = analysisData;
    const v = data.verdict;
    const winner = v.winner;
    const winnerData = v.verdicts[winner];
    const problemType = data.problem_detection.problem_type;
    const W = 210;
    const PL = 16;
    const PR = 16;
    const CW = W - PL - PR;
    let y = 0;

    // ── PALETTE ─────────────────────────────────────────────────────────
    const C = {
      navy:     [10,  20,  50],
      navyMid:  [18,  32,  75],
      accent:   [0,   195, 240],   // matches site cyan
      accentDk: [0,   140, 190],
      green:    [0,   200, 140],
      greenBg:  [228, 250, 240],
      red:      [210, 55,  80],
      redBg:    [253, 234, 238],
      amber:    [200, 130, 20],
      white:    [255, 255, 255],
      offwhite: [248, 249, 252],
      light:    [232, 238, 248],
      border:   [208, 217, 232],
      textDark: [18,  28,  55],
      textMid:  [70,  85,  115],
      textDim:  [128, 143, 172],
      coverBg:  [10,  16,  38],
      coverBg2: [16,  26,  58],
    };

    const sf = (c) => doc.setFillColor(c[0], c[1], c[2]);
    const sd = (c) => doc.setDrawColor(c[0], c[1], c[2]);
    const st = (c) => doc.setTextColor(c[0], c[1], c[2]);
    const sfont = (style, size) => { doc.setFont('helvetica', style); doc.setFontSize(size); };
    const fr = (rx, ry, rw, rh) => doc.rect(rx, ry, rw, rh, 'F');
    const ln = (x1, y1, x2, y2, lw=0.25) => { doc.setLineWidth(lw); doc.line(x1, y1, x2, y2); };
    const tx = (str, tx_, ty_, opts={}) => doc.text(String(str), tx_, ty_, opts);
    const sp = (str, maxW) => doc.splitTextToSize(String(str), maxW);

    // Strip ALL non-ASCII and problematic chars — prevents jsPDF garbage
    const safe = (str) => String(str)
      .replace(/[^\x20-\x7E\xC0-\xFF]/g, '')
      .replace(/\s+/g, ' ')
      .trim();

    const newPage = () => { doc.addPage(); y = 20; };
    const checkY  = (need=20) => { if (y + need > 276) newPage(); };

    // Progress bar: draws BG then fill, returns nothing. Text must be drawn AFTER.
    const pbar = (bx, by, bw, bh, pct, fillCol, bgCol=C.light) => {
      sf(bgCol); fr(bx, by, bw, bh);
      sf(fillCol); fr(bx, by, bw * Math.min(Math.max(pct, 0) / 100, 1), bh);
    };

    // Section header with left color bar + bold label
    const secHead = (label, col=C.accent) => {
      checkY(16);
      sf(col); fr(PL, y, 3, 9);
      sfont('bold', 11); st(C.navy);
      tx(label, PL + 6, y + 6.5);
      y += 14;
    };

    // Status badge pill — text on top of fill, correct z-order
    const badge = (label, bx, by, bgCol, txtCol) => {
      sfont('bold', 6.5);
      const tw = doc.getTextWidth(label);
      sf(bgCol);  fr(bx - 1, by - 3.8, tw + 6, 5.5);
      st(txtCol); tx(label, bx + 2, by);
    };

    // ── ModelIQ logo drawn as vectors ────────────────────────────────
    // Hexagon outline: 6 line segments, no fill, cyan stroke
    // Then "Model" white bold + "IQ" cyan bold
    const drawLogo = (lx, ly, hexR, fontSize) => {
      // Draw hexagon outline (flat-top orientation matching the site icon)
      const cx = lx + hexR;
      const cy = ly + hexR;
      const pts = [];
      for (let i = 0; i < 6; i++) {
        const a = (Math.PI / 180) * (60 * i - 30);
        pts.push([cx + hexR * Math.cos(a), cy + hexR * Math.sin(a)]);
      }
      sd(C.accent); doc.setLineWidth(hexR * 0.18);
      for (let i = 0; i < 6; i++) {
        const next = (i + 1) % 6;
        doc.line(pts[i][0], pts[i][1], pts[next][0], pts[next][1]);
      }
      // "Model" in white, "IQ" in cyan — side by side
      const textX = lx + hexR * 2 + hexR * 0.7;
      const textY = ly + hexR + fontSize * 0.35;
      sfont('bold', fontSize); st(C.white);
      tx('Model', textX, textY);
      st(C.accent);
      tx('IQ', textX + doc.getTextWidth('Model'), textY);
    };

    // ════════════════════════════════════════════════════════════════════
    // PAGE 1 — COVER
    // ════════════════════════════════════════════════════════════════════
    sf(C.coverBg); fr(0, 0, W, 297);

    // Top accent stripe
    sf(C.accent); fr(0, 0, W, 3);

    // Subtle side panel
    doc.setGState(new doc.GState({ opacity: 0.06 }));
    sf(C.accent); fr(118, 0, 92, 155);
    doc.setGState(new doc.GState({ opacity: 1 }));

    // ── Logo on cover (large) ─────────────────────────────────────────
    drawLogo(18, 36, 7, 18);

    // Tagline
    sfont('normal', 9.5); st([145, 165, 210]);
    tx('Explainable AutoML Analysis Report', 18, 70);

    // Divider
    sf(C.accent); fr(18, 76, 88, 0.5);

    // ── Metadata block ────────────────────────────────────────────────
    y = 90;
    const meta = [
      ['DATASET',         `${data.dataset_info.rows} rows x ${data.dataset_info.columns} columns`],
      ['TARGET COLUMN',   data.dataset_info.target_column],
      ['PROBLEM TYPE',    problemType.toUpperCase()],
      ['DETECTION RULE',  safe(data.problem_detection.rule_triggered || '')],
      ['SELECTED MODEL',  winner || 'N/A'],
      ['COMPOSITE SCORE', winnerData ? `${(winnerData.composite_score * 100).toFixed(2)} / 100` : '--'],
      ['GENERATED',       new Date().toLocaleString()],
    ];
    meta.forEach(([k, val]) => {
      sfont('normal', 6.8); st([95, 118, 162]);
      tx(k, 18, y);
      sfont('bold', 8.5); st(C.white);
      const lines = sp(safe(val), 125);
      tx(lines[0], 60, y);
      y += 9;
    });

    // ── Winner spotlight ──────────────────────────────────────────────
    y = 196;
    sf(C.navyMid); fr(16, y, W - 32, 52);
    sd(C.accent); doc.setLineWidth(0.7); doc.rect(16, y, W - 32, 52, 'S');

    // Green circle indicator
    sf(C.green); doc.circle(25, y + 11.5, 2.8, 'F');
    sfont('normal', 7); st([130, 195, 165]);
    tx('SELECTED MODEL', 30.5, y + 12.8);

    sfont('bold', 22); st(C.white);
    tx(winner || 'N/A', 18, y + 29);

    if (winnerData) {
      const score = (winnerData.composite_score * 100).toFixed(2);
      sfont('normal', 9); st([155, 178, 218]);
      tx('Composite Score: ', 18, y + 39);
      sfont('bold', 9); st(C.accent);
      tx(score + ' / 100', 18 + doc.getTextWidth('Composite Score: '), y + 39);
      pbar(18, y + 43, CW - 4, 4, parseFloat(score), C.accent, [28, 48, 98]);
    }

    // Cover footer
    sf(C.navyMid); fr(0, 282, W, 15);
    sfont('normal', 6.5); st([105, 125, 168]);
    tx('ModelIQ  |  Deterministic AutoML  |  No LLMs at Runtime  |  CodeWiser x VJTI Hackathon 2026', W / 2, 291, { align: 'center' });

    // ════════════════════════════════════════════════════════════════════
    // PAGE 2 — DATASET SUMMARY + LEADERBOARD
    // ════════════════════════════════════════════════════════════════════
    newPage();
    sf(C.navy); fr(0, 0, W, 12);
    sfont('bold', 7.5); st(C.white);
    tx('MODEL IQ  --  ANALYSIS REPORT', PL, 8);
    sfont('normal', 7); st([148, 163, 200]);
    tx(new Date().toLocaleDateString(), W - PR, 8, { align: 'right' });
    y = 22;

    // Dataset stat boxes
    secHead('DATASET SUMMARY');
    sf(C.offwhite); fr(PL, y, CW, 24);
    sd(C.border); doc.setLineWidth(0.2); doc.rect(PL, y, CW, 24, 'S');
    const dsStats = [
      ['ROWS',    String(data.dataset_info.rows)],
      ['COLUMNS', String(data.dataset_info.columns)],
      ['TARGET',  data.dataset_info.target_column],
      ['PROBLEM', problemType.toUpperCase()],
    ];
    const csW = CW / 4;
    dsStats.forEach(([lbl, val], i) => {
      const bx = PL + i * csW;
      sfont('normal', 7); st(C.textDim);
      tx(lbl, bx + csW / 2, y + 8, { align: 'center' });
      sfont('bold', 12); st(C.navy);
      tx(val, bx + csW / 2, y + 19, { align: 'center' });
      if (i < 3) { sd(C.border); ln(bx + csW, y, bx + csW, y + 24); }
    });
    y += 30;

    // Detection rule
    sf([230, 240, 255]); fr(PL, y, CW, 9);
    sfont('normal', 7.5); st(C.accentDk);
    const rLines = sp('Detection Rule: ' + safe(data.problem_detection.rule_triggered || ''), CW - 8);
    tx(rLines[0], PL + 4, y + 6.5);
    y += 15;

    // Leaderboard
    secHead('MODEL LEADERBOARD');

    // Header row
    sf(C.navy); fr(PL, y, CW, 9);
    sfont('bold', 7.5); st(C.white);
    tx('MODEL',   PL + 4,        y + 6.3);
    tx('SCORE',   PL + 80,       y + 6.3, { align: 'right' });
    if (problemType === 'classification') {
      tx('ACCURACY', PL + 107,   y + 6.3, { align: 'right' });
      tx('F1',       PL + 130,   y + 6.3, { align: 'right' });
    } else {
      tx('R2',       PL + 107,   y + 6.3, { align: 'right' });
      tx('MAE',      PL + 130,   y + 6.3, { align: 'right' });
    }
    tx('OVERFIT',  PL + 155,     y + 6.3, { align: 'right' });
    tx('STATUS',   PL + CW - 1,  y + 6.3, { align: 'right' });
    y += 9;

    v.leaderboard.forEach((m, idx) => {
      checkY(12);
      const isW  = m.model_name === winner;
      const isDQ = m.status === 'disqualified';
      const rowBg = isW ? [230, 250, 240] : (idx % 2 === 0 ? C.white : C.offwhite);
      sf(rowBg); fr(PL, y, CW, 11);
      if (isW) { sf(C.green); fr(PL, y, 3, 11); }

      const score100 = ((m.composite_score || 0) * 100).toFixed(1);

      // Score bar first (behind text)
      pbar(PL + 42, y + 3.5, 30, 4, parseFloat(score100), isW ? C.green : C.accent);

      // All text drawn AFTER bar
      sfont(isW ? 'bold' : 'normal', 8.5);
      st(isW ? C.green : (isDQ ? C.textDim : C.textDark));
      tx(m.model_name, PL + (isW ? 6 : 4), y + 7.8);

      sfont('bold', 7.5); st(isW ? C.green : C.accent);
      tx(score100, PL + 80, y + 7.8, { align: 'right' });

      if (!isDQ && m.raw_metrics && m.raw_metrics.status !== 'failed') {
        const rm = m.raw_metrics;
        sfont('normal', 7.5); st(C.textMid);
        if (problemType === 'classification') {
          tx((rm.test_accuracy * 100).toFixed(1) + '%', PL + 107, y + 7.8, { align: 'right' });
          tx((rm.test_f1_macro * 100).toFixed(1) + '%', PL + 130, y + 7.8, { align: 'right' });
        } else {
          tx(rm.test_r2.toFixed(3),  PL + 107, y + 7.8, { align: 'right' });
          tx(rm.test_mae.toFixed(4), PL + 130, y + 7.8, { align: 'right' });
        }
        const gap = rm.overfit_gap || 0;
        sfont('normal', 7.5);
        st(gap > 0.10 ? C.red : (gap > 0.05 ? C.amber : C.green));
        tx((gap * 100).toFixed(1) + '%', PL + 155, y + 7.8, { align: 'right' });
      }

      const sLabel = isW ? 'SELECTED' : (isDQ ? 'DISQUALIFIED' : 'REJECTED');
      const sBg    = isW ? C.greenBg  : (isDQ ? C.light : C.redBg);
      const sTxt   = isW ? C.green    : (isDQ ? C.textDim : C.red);
      badge(sLabel, PL + CW - doc.getTextWidth(sLabel) - 7, y + 7.8, sBg, sTxt);

      sd(C.border); ln(PL, y + 11, PL + CW, y + 11);
      y += 11;
    });
    y += 8;

    // ════════════════════════════════════════════════════════════════════
    // PAGE 3 — VERDICT ENGINE
    // ════════════════════════════════════════════════════════════════════
    newPage();
    sf(C.navy); fr(0, 0, W, 12);
    sfont('bold', 7.5); st(C.white);
    tx('MODEL IQ  --  VERDICT ENGINE BREAKDOWN', PL, 8);
    y = 22;

    secHead('SCORING RUBRIC -- ' + winner.toUpperCase(), C.green);

    const weights = v.weights_used;
    const formula = Object.entries(weights).map(([k, w]) => `${k} x ${w}`).join('  +  ');
    sf([230, 242, 255]); fr(PL, y, CW, 9);
    sfont('normal', 7); st(C.accentDk);
    const fLines = sp('Formula:  ' + safe(formula), CW - 8);
    tx(fLines[0], PL + 4, y + 6.5);
    y += 15;

    if (winnerData && winnerData.dimension_scores) {
      // Column layout — everything measured carefully
      // Row height = 13mm so bar + text both fit
      const ROW_H = 13;
      const BAR_X = PL + 44;   // bar starts here
      const BAR_W = 26;         // bar width
      const BAR_Y_OFF = 4;      // bar top offset from row top
      const BAR_H = 4;
      const TXT_Y_OFF = 9.5;    // text baseline offset from row top
      const SCR_X  = PL + 74;   // score % right-aligned
      const WGT_X  = PL + 94;   // weight right-aligned
      const PTS_X  = PL + 114;  // pts right-aligned
      const EXP_X  = PL + 118;  // explanation start
      const EXP_W  = CW - 120;  // explanation max width

      // Table header
      sf(C.navy); fr(PL, y, CW, 9);
      sfont('bold', 7.5); st(C.white);
      tx('DIMENSION',   PL + 2,  y + 6.3);
      tx('SCORE',       SCR_X,   y + 6.3, { align: 'right' });
      tx('WEIGHT',      WGT_X,   y + 6.3, { align: 'right' });
      tx('PTS',         PTS_X,   y + 6.3, { align: 'right' });
      tx('EXPLANATION', EXP_X,   y + 6.3);
      y += 9;

      let totalPts = 0;
      Object.entries(winnerData.dimension_scores).forEach(([dimName, d], idx) => {
        if (d.score === null || d.score === undefined) return;
        checkY(ROW_H + 2);
        const pts   = (d.contribution || 0) * 100;
        const score = (d.score || 0) * 100;
        totalPts += pts;

        // 1. Draw row background first
        const rowBg = idx % 2 === 0 ? C.white : C.offwhite;
        sf(rowBg); fr(PL, y, CW, ROW_H);

        // 2. Draw bar (on top of background, BELOW text)
        pbar(BAR_X, y + BAR_Y_OFF, BAR_W, BAR_H, score, C.accent);

        // 3. Draw ALL text last so nothing covers it
        sfont('bold', 8); st(C.textDark);
        tx(dimName, PL + 2, y + TXT_Y_OFF);

        sfont('bold', 7.5); st(C.accent);
        tx(score.toFixed(1) + '%', SCR_X, y + TXT_Y_OFF, { align: 'right' });

        sfont('normal', 7.5); st(C.textMid);
        tx(String(d.weight), WGT_X, y + TXT_Y_OFF, { align: 'right' });

        sfont('bold', 7.5); st(C.green);
        tx(pts.toFixed(2), PTS_X, y + TXT_Y_OFF, { align: 'right' });

        const expClean = safe(d.explanation || '');
        const expLines = sp(expClean, EXP_W);
        sfont('normal', 6.8); st(C.textDim);
        tx(expLines[0] || '', EXP_X, y + TXT_Y_OFF);

        // Row divider
        sd(C.border); ln(PL, y + ROW_H, PL + CW, y + ROW_H);
        y += ROW_H;
      });

      // Totals row
      checkY(12);
      sf(C.navy); fr(PL, y, CW, 11);
      sfont('bold', 9); st(C.white);
      tx('COMPOSITE SCORE', PL + 2, y + 8);
      sfont('bold', 10); st(C.green);
      tx(totalPts.toFixed(2) + ' / 100', PTS_X, y + 8, { align: 'right' });
      y += 17;
    }

    // Rationale box
    if (winnerData && winnerData.verdict_rationale) {
      checkY(28);
      const cleanRat = safe(
        winnerData.verdict_rationale
          .replace(/\*\*/g, '')
          .replace(/\n/g, ' ')
          .replace(/\s+/g, ' ')
      );
      const ratLines = sp(cleanRat, CW - 12);
      const ratH = Math.min(ratLines.length, 8) * 5.5 + 12;
      sf(C.green);  fr(PL, y, 3, ratH);
      sf(C.greenBg); fr(PL + 3, y, CW - 3, ratH);
      sfont('italic', 8); st(C.textMid);
      ratLines.slice(0, 8).forEach((line_, i) => {
        tx(line_, PL + 8, y + 8 + i * 5.5);
      });
      y += ratH + 8;
    }

    // ════════════════════════════════════════════════════════════════════
    // PAGE 4 — FEATURE ANALYSIS
    // ════════════════════════════════════════════════════════════════════
    newPage();
    sf(C.navy); fr(0, 0, W, 12);
    sfont('bold', 7.5); st(C.white);
    tx('MODEL IQ  --  FEATURE INTELLIGENCE', PL, 8);
    y = 22;

    secHead('FEATURE SELECTION SUMMARY', C.accent);

    const fa = data.feature_analysis;
    const removed = fa.removed_features;
    const allRemoved = [...(removed.low_variance || []), ...(removed.redundant_correlations || [])];

    // Stat boxes
    sf(C.offwhite); fr(PL, y, CW, 24);
    sd(C.border); doc.setLineWidth(0.2); doc.rect(PL, y, CW, 24, 'S');
    const fStats = [
      ['ORIGINAL', String(fa.original_feature_count), C.accent],
      ['SELECTED', String(fa.selected_feature_count), C.green],
      ['REMOVED',  String(allRemoved.length), allRemoved.length > 0 ? C.amber : C.textDim],
    ];
    const fcW = CW / 3;
    fStats.forEach(([lbl, val, col], i) => {
      const bx = PL + i * fcW;
      sfont('normal', 7); st(C.textDim);
      tx(lbl, bx + fcW / 2, y + 8, { align: 'center' });
      sfont('bold', 14); st(col);
      tx(val, bx + fcW / 2, y + 19, { align: 'center' });
      if (i < 2) { sd(C.border); ln(bx + fcW, y, bx + fcW, y + 24); }
    });
    y += 28;

    // Feature chips
    const renderChips = (feats, borderCol, bgCol, txtCol, label) => {
      if (!feats || feats.length === 0) return;
      checkY(16);
      sfont('bold', 7.5); st(txtCol);
      tx(label, PL, y); y += 5.5;
      let cx = PL; const ch = 7;
      feats.forEach(feat => {
        const tw = doc.getTextWidth(feat) + 8;
        if (cx + tw > W - PR) { cx = PL; y += ch + 3; checkY(12); }
        sf(bgCol); fr(cx, y, tw, ch);
        sd(borderCol); doc.setLineWidth(0.2); doc.rect(cx, y, tw, ch, 'S');
        sfont('normal', 6.8); st(txtCol);
        tx(feat, cx + 4, y + 5.2);
        cx += tw + 3;
      });
      y += ch + 8;
    };

    renderChips(fa.selected_features, C.green, C.greenBg, C.green, 'SELECTED FEATURES');
    renderChips(allRemoved, C.red, C.redBg, C.red, 'REMOVED FEATURES');
    y += 2;

    secHead('FEATURE IMPORTANCE RANKING', C.accent);

    const ranking = data.explanations?.shap_values?.length
      ? data.explanations.shap_values
      : (data.explanations?.feature_importance || fa.importance_ranking || []);

    const srcLabel = data.explanations?.shap_available
      ? 'Source: SHAP Values (mean absolute) -- deterministic given fixed model weights'
      : 'Source: Mutual Information Score -- model-agnostic, captures non-linear relationships';

    sf([230, 241, 255]); fr(PL, y, CW, 8);
    sfont('italic', 7); st(C.accentDk);
    tx(safe(srcLabel), PL + 4, y + 5.5);
    y += 13;

    if (ranking.length > 0) {
      const maxVal = Math.max(
        ...ranking.map(r => r.mean_abs_shap || r.importance || r.mutual_info_score || 0)
      ) || 1;

      const FT_ROW_H  = 11;
      const FT_BAR_X  = PL + 64;
      const FT_BAR_W  = CW - 104;
      const FT_SIG_X  = PL + CW - 2;

      // Table header
      sf(C.navy); fr(PL, y, CW, 8);
      sfont('bold', 7.5); st(C.white);
      tx('RK',      PL + 2,     y + 5.5);
      tx('FEATURE', PL + 14,    y + 5.5);
      tx('IMPORTANCE BAR',      FT_BAR_X, y + 5.5);
      tx('VALUE',   FT_SIG_X - 30, y + 5.5);
      tx('SIGNAL',  FT_SIG_X,   y + 5.5, { align: 'right' });
      y += 8;

      ranking.slice(0, 10).forEach((r, idx) => {
        checkY(FT_ROW_H + 2);
        const val  = r.mean_abs_shap || r.importance || r.mutual_info_score || 0;
        const pct  = (val / maxVal) * 100;
        const barCol  = pct > 66 ? C.green   : (pct > 33 ? C.accent  : C.textDim);
        const sigLabel = pct > 66 ? 'HIGH'   : (pct > 33 ? 'MED'     : 'LOW');
        const sigBg    = pct > 66 ? C.greenBg: (pct > 33 ? [226,240,255] : C.light);
        const sigTxt   = pct > 66 ? C.green  : (pct > 33 ? C.accentDk   : C.textDim);

        // 1. Background
        const rowBg = idx % 2 === 0 ? C.white : C.offwhite;
        sf(rowBg); fr(PL, y, CW, FT_ROW_H);

        // 2. Bar (behind text)
        pbar(FT_BAR_X, y + 3.5, FT_BAR_W, 4, pct, barCol);

        // 3. All text last
        sfont('bold', 7.5); st(barCol);
        tx('#' + (r.rank || idx + 1), PL + 2, y + 7.8);

        sfont('bold', 8); st(C.textDark);
        tx(r.feature || '', PL + 14, y + 7.8);

        sfont('normal', 7.5); st(barCol);
        tx(val.toFixed(5), FT_SIG_X - 30, y + 7.8);

        badge(sigLabel, FT_SIG_X - doc.getTextWidth(sigLabel) - 5, y + 7.8, sigBg, sigTxt);

        sd(C.border); ln(PL, y + FT_ROW_H, PL + CW, y + FT_ROW_H);
        y += FT_ROW_H;
      });
    }

    // ── Footer on every page ─────────────────────────────────────────
    const totalPages = doc.internal.getNumberOfPages();
    for (let pg = 1; pg <= totalPages; pg++) {
      doc.setPage(pg);
      sf(C.navy); fr(0, 287, W, 10);
      sfont('normal', 6.5); st([108, 128, 172]);
      tx('ModelIQ  |  Deterministic AutoML  |  No LLMs at Runtime  |  CodeWiser x VJTI Hackathon 2026', PL, 293);
      st([90, 110, 155]);
      tx('Page ' + pg + ' of ' + totalPages, W - PR, 293, { align: 'right' });
    }

    doc.save(`ModelIQ_Report_${(winner || 'report').replace(/ /g, '_')}.pdf`);

  } catch (err) {
    alert('PDF export failed: ' + err.message + '\n' + err.stack);
  }

  btn.textContent = 'Export Report PDF';
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