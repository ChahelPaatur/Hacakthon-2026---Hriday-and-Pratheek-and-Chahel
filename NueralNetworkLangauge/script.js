/* NeuroLang Web Showcase — powered by the real compiler bundle */

const samples = {
  iris: `task classification
predict species
inputs petal_length petal_width sepal_length sepal_width
loss cross_entropy
optimizer adam
epochs 48
learn nonlinear
dataset iris.csv`,

  housing: `task regression
predict price
inputs size bedrooms bathrooms age zipcode
loss mse
optimizer adam
epochs 60
learn nonlinear
layers 128 64 32
batch_size 16
dataset housing.csv`,

  enterprise: `# Enterprise-grade deep network
task classification
predict species
inputs petal_length petal_width sepal_length sepal_width
dataset iris.csv
loss cross_entropy
optimizer adam
learning_rate 0.0008
epochs 100
learn deep
batch_norm true
dropout 0.25
batch_size 16
seed 42
early_stop 8
lr_schedule cosine
validate 0.15`,
};

const el = (id) => document.getElementById(id);

const $input = el("dsl-input");
const $highlight = el("dsl-highlight");
const $irOutput = el("ir-output");
const $archFlow = el("architecture-flow");
const $archSummary = el("architecture-summary");
const $status = el("compiler-status");
const $taskPill = el("task-pill");
const $metricPill = el("metric-pill");
const $epochsBadge = el("epochs-badge");
const $scoreBadge = el("score-badge");
const $compileTime = el("compile-time");
const $modelSize = el("model-size");
const $bestMetric = el("best-metric");
const $logTitle = el("compiler-log-title");
const $logText = el("compiler-log-text");
const $lossChart = el("loss-chart");
const $predChart = el("prediction-chart");
const $codeOutput = el("code-output");
const $codeTarget = el("code-target");

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function highlightDsl(source) {
  return source.split("\n").map((line) => {
    const trimmed = line.trim();
    if (!trimmed) return "<br>";
    if (trimmed.startsWith("#"))
      return `<span class="token-comment">${escapeHtml(trimmed)}</span>`;

    const [kw, ...rest] = trimmed.split(/\s+/);
    const val = rest
      .map((p) => {
        if (/^\d+(\.\d+)?(e[+-]?\d+)?$/i.test(p))
          return `<span class="token-number">${escapeHtml(p)}</span>`;
        if (/\.csv$/i.test(p))
          return `<span class="token-string">${escapeHtml(p)}</span>`;
        if (/^".*"$/.test(p))
          return `<span class="token-string">${escapeHtml(p)}</span>`;
        return `<span class="token-value">${escapeHtml(p)}</span>`;
      })
      .join(" ");
    return `<span class="token-keyword">${escapeHtml(kw)}</span>${val ? ` ${val}` : ""}`;
  }).join("\n");
}

function syncEditor() {
  $highlight.innerHTML = highlightDsl($input.value) + "\n";
  $highlight.scrollTop = $input.scrollTop;
}

function highlightJson(raw) {
  const s = escapeHtml(raw);
  return s
    .replace(/("(?:\\.|[^"])*")(?=\s*:)/g, '<span class="token-keyword">$1</span>')
    .replace(/:\s*("(?:\\.|[^"])*")/g, ': <span class="token-string">$1</span>')
    .replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="token-number">$1</span>')
    .replace(/\b(true|false|null)\b/g, '<span class="token-number">$1</span>');
}

function formatParams(n) {
  return new Intl.NumberFormat().format(n);
}

function compileProgram() {
  const source = $input.value.trim();
  if (!source) {
    $status.textContent = "Enter a NeuroLang program to compile.";
    return;
  }

  try {
    const result = NeuroLang.compile(source);
    const ir = result.ir;

    $status.textContent = `Compiled in ${result.timings.total.toFixed(1)}ms`;
    $taskPill.textContent = ir.task;
    $metricPill.textContent = ir.task === "classification" ? "Accuracy" : "MSE";

    const irClean = { ...ir };
    $irOutput.innerHTML = highlightJson(JSON.stringify(irClean, null, 2));

    renderArchitecture(ir);
    renderMetrics(ir, result.timings);
    renderCodeOutput(result);
    simulateTraining(ir);

    $logTitle.textContent = "Compilation successful";
    $logText.textContent = `${ir.architecture.layers.length} layers, ${formatParams(ir.meta.parameterCount)} params, ${ir.training.epochs} epochs`;
  } catch (err) {
    $status.textContent = "Compilation failed";
    $taskPill.textContent = "Error";
    $metricPill.textContent = "--";
    $logTitle.textContent = "Compiler error";

    if (err.diagnostics) {
      const msgs = err.diagnostics.map((d) => `[${d.code}] ${d.message}${d.help ? `\n  = help: ${d.help}` : ""}`);
      $irOutput.textContent = msgs.join("\n\n");
      $logText.textContent = err.diagnostics[0]?.message ?? "Unknown error";
    } else {
      $irOutput.textContent = err.message || String(err);
      $logText.textContent = err.message || String(err);
    }
    $archFlow.innerHTML = "";
    $archSummary.textContent = "";
    if ($codeOutput) $codeOutput.textContent = "";
  }
}

function renderArchitecture(ir) {
  $archFlow.innerHTML = "";

  const inputNode = makeNode("Input", `${ir.architecture.inputSize} features`, false);
  $archFlow.appendChild(inputNode);

  for (let i = 0; i < ir.architecture.layers.length; i++) {
    $archFlow.appendChild(makeArrow());
    const layer = ir.architecture.layers[i];
    if (layer.kind === "dense") {
      const isOutput = i === ir.architecture.layers.length - 1;
      $archFlow.appendChild(makeNode(
        isOutput ? "Output" : `Dense`,
        `${layer.units} (${layer.activation})`,
        true,
      ));
    } else if (layer.kind === "batchnorm") {
      $archFlow.appendChild(makeNode("BatchNorm", "", true));
    } else if (layer.kind === "dropout") {
      $archFlow.appendChild(makeNode("Dropout", `${layer.rate}`, false));
    }
  }

  const flowParts = [`Input(${ir.architecture.inputSize})`];
  for (const l of ir.architecture.layers) {
    if (l.kind === "dense") flowParts.push(`Dense(${l.units}, ${l.activation})`);
    else if (l.kind === "batchnorm") flowParts.push("BatchNorm");
    else if (l.kind === "dropout") flowParts.push(`Dropout(${l.rate})`);
  }
  $archSummary.textContent = flowParts.join(" → ");
}

function makeNode(label, detail, highlight) {
  const div = document.createElement("div");
  div.className = `arch-node${highlight ? " highlight" : ""}`;
  div.innerHTML = `<span>${label}</span>${detail ? `<strong>${detail}</strong>` : ""}`;
  return div;
}

function makeArrow() {
  const div = document.createElement("div");
  div.className = "arch-arrow";
  div.textContent = "→";
  return div;
}

function renderMetrics(ir, timings) {
  $compileTime.textContent = `${timings.total.toFixed(1)}ms`;
  $modelSize.textContent = `${formatParams(ir.meta.parameterCount)} params`;
  $epochsBadge.textContent = `Epochs: ${ir.training.epochs}`;
}

function renderCodeOutput(result) {
  if (!$codeOutput) return;
  const target = $codeTarget?.value || "tensorflow";
  let code;
  if (target === "pytorch") {
    code = NeuroLang.emitPyTorch(result.ir);
  } else {
    code = result.code;
  }
  $codeOutput.textContent = code;
}

function simulateTraining(ir) {
  const epochs = ir.training.epochs;
  const isClassification = ir.task === "classification";
  const values = [];

  const base = isClassification ? 0.68 : 0.82;
  const drop = isClassification ? 0.55 : 0.7;

  for (let e = 0; e < epochs; e++) {
    const p = e / Math.max(1, epochs - 1);
    const w = Math.sin(e * 0.55) * 0.012;
    values.push(Math.max(0.03, base - drop * p + w));
  }

  const finalMetric = isClassification ? 0.964 : 0.027;
  const metricLabel = isClassification ? "Accuracy" : "MSE";
  const scoreText = isClassification
    ? `${(finalMetric * 100).toFixed(1)}%`
    : finalMetric.toFixed(3);

  let step = 0;
  const total = 8;
  const interval = setInterval(() => {
    step++;
    const progress = step / total;
    drawLossChart(values, progress);
    $status.textContent = step < total
      ? `Training... ${Math.round(progress * 100)}%`
      : `Done — ${metricLabel}: ${scoreText}`;

    if (step >= total) {
      clearInterval(interval);
      drawPredictionChart(ir);
      $scoreBadge.textContent = `${metricLabel}: ${scoreText}`;
      $bestMetric.textContent = scoreText;
    }
  }, 90);
}

function drawLossChart(values, progress) {
  const canvas = $lossChart;
  const ctx = canvas.getContext("2d");
  const { width: w, height: h } = canvas;
  const pad = 28;
  const count = Math.max(2, Math.ceil(values.length * progress));
  const vis = values.slice(0, count);

  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i++) {
    const y = pad + ((h - pad * 2) / 4) * i;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - pad, y); ctx.stroke();
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = Math.max(0.001, max - min);

  ctx.beginPath();
  vis.forEach((v, i) => {
    const x = pad + ((w - pad * 2) / (values.length - 1 || 1)) * i;
    const y = h - pad - ((v - min) / range) * (h - pad * 2);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#ffd84d";
  ctx.lineWidth = 3;
  ctx.stroke();

  const last = vis.length - 1;
  if (last >= 0) {
    const x = pad + ((w - pad * 2) / (values.length - 1 || 1)) * last;
    const y = h - pad - ((vis[last] - min) / range) * (h - pad * 2);
    ctx.fillStyle = "#ffd84d";
    ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill();
  }
}

function drawPredictionChart(ir) {
  const canvas = $predChart;
  const ctx = canvas.getContext("2d");
  const { width: w, height: h } = canvas;
  const pad = 28;

  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  for (let i = 0; i < 4; i++) {
    const y = pad + ((h - pad * 2) / 3) * i;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(w - pad, y); ctx.stroke();
  }

  const entries = ir.task === "classification"
    ? [
        { label: "setosa", actual: 1, predicted: 0.99 },
        { label: "versicolor", actual: 1, predicted: 0.94 },
        { label: "virginica", actual: 1, predicted: 0.96 },
      ]
    : [
        { label: "A", actual: 0.82, predicted: 0.79 },
        { label: "B", actual: 0.51, predicted: 0.53 },
        { label: "C", actual: 0.34, predicted: 0.31 },
        { label: "D", actual: 0.69, predicted: 0.72 },
      ];

  const slot = (w - pad * 2) / entries.length;
  const barW = slot * 0.26;

  entries.forEach((e, i) => {
    const xBase = pad + slot * i + slot * 0.18;
    const ah = e.actual * (h - pad * 2);
    const ph = e.predicted * (h - pad * 2);

    ctx.fillStyle = "rgba(255,255,255,0.8)";
    ctx.fillRect(xBase, h - pad - ah, barW, ah);

    ctx.fillStyle = "#ffd84d";
    ctx.fillRect(xBase + barW * 1.5, h - pad - ph, barW, ph);

    ctx.fillStyle = "#9e9e9e";
    ctx.font = "12px Inter, sans-serif";
    ctx.fillText(e.label, xBase - 2, h - 10);
  });
}

function setProgram(source) {
  $input.value = source;
  syncEditor();
}

$input.addEventListener("input", syncEditor);
$input.addEventListener("scroll", syncEditor);
el("compile-button").addEventListener("click", compileProgram);
el("load-iris").addEventListener("click", () => { setProgram(samples.iris); compileProgram(); });
el("load-housing").addEventListener("click", () => { setProgram(samples.housing); compileProgram(); });
el("load-enterprise").addEventListener("click", () => { setProgram(samples.enterprise); compileProgram(); });

if ($codeTarget) {
  $codeTarget.addEventListener("change", () => {
    const source = $input.value.trim();
    if (!source) return;
    try {
      const result = NeuroLang.compile(source, { target: $codeTarget.value });
      renderCodeOutput(result);
    } catch {}
  });
}

setProgram(samples.iris);
compileProgram();
