/* ── ModelIQ Frontend App ──────────────────────────────────── */

const API = 'http://localhost:8000';
let currentFile = null;
let analysisData = null;

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

    // Populate preview table
    renderPreview(data.columns, data.preview);

    // Populate target select
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

  // Show loading
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

  // Winner banner
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
}

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

function renderVerdict(verdict, problemType) {
  const weights = verdict.weights_used;
  const container = document.getElementById('verdictContainer');
  const legend = document.getElementById('rubricLegend');

  // Formula
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
      const ratHtml = m.verdict_rationale
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
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

function renderFeatures(featureAnalysis, explanations) {
  const container = document.getElementById('featuresContainer');
  let html = '';

  // Feature selection summary
  const removed = featureAnalysis.removed_features;
  const allRemoved = [...(removed.low_variance||[]), ...(removed.redundant_correlations||[])];

  html += `<div class="pruning-box">
    <div class="pruning-title">Feature Selection · ${featureAnalysis.original_feature_count} → ${featureAnalysis.selected_feature_count} features</div>
    <div class="chip-list">`;
  featureAnalysis.selected_features.forEach(f => { html += `<span class="chip selected">${esc(f)}</span>`; });
  allRemoved.forEach(f => { html += `<span class="chip removed">${esc(f)}</span>`; });
  html += '</div></div>';

  // Importance bars
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

/* ── UI Helpers ────────────────────────────────────────────── */
function showTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById(`tab-${name}`).classList.add('active');
  event.target.classList.add('active');
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
  currentFile = null; analysisData = null;
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