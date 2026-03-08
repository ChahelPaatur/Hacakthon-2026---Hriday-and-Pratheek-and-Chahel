#!/usr/bin/env node
import * as fs from "node:fs";
import * as path from "node:path";
import { compile, CompilationError, type CodegenTarget } from "./compiler.js";
import { emitSummary } from "./codegen.js";
import { execute } from "./runtime.js";
import { formatDiagnostics, Severity, type Diagnostic } from "./diagnostics.js";
import { startRepl } from "./repl.js";
import { inferSchema, formatSchema } from "./schema.js";
import { computeFeatureImportance, formatFeatureImportance } from "./explain.js";
import {
  getCandidates, buildCandidateIR, formatSearchResults,
  type AutoSearchResult, type ArchCandidate,
} from "./autoarch.js";
import {
  getTuneConfig, buildTrialIR, totalTrials, formatTuneResults,
  type TuneResult, type TuneTrialResult,
} from "./tune.js";
import {
  describeDataset, formatDataPreview, formatDescription,
  formatCorrelation, formatSample,
} from "./inspect.js";

const Y = "\x1b[33m";
const G = "\x1b[32m";
const R = "\x1b[31m";
const C = "\x1b[36m";
const D = "\x1b[2m";
const B = "\x1b[1m";
const X = "\x1b[0m";

function banner(): void {
  console.log(`${Y}${B}`);
  console.log(`  ╔═══════════════════════════════════════════╗`);
  console.log(`  ║       NeuroLang Compiler v4.0             ║`);
  console.log(`  ║   Declarative DSL → Neural Networks       ║`);
  console.log(`  ╚═══════════════════════════════════════════╝${X}`);
  console.log();
}

function usage(): void {
  banner();
  console.log(`${B}Usage:${X}  neurolang <file.nl> [options]`);
  console.log(`        neurolang --repl`);
  console.log(`        neurolang --infer <file.csv>`);
  console.log(`        neurolang --serve [port]`);
  console.log(`        echo "Predict ..." | neurolang --run`);
  console.log();
  console.log(`${B}Options:${X}`);
  console.log(`  --emit-code          Print the generated code`);
  console.log(`  --target <target>    Code gen target: tensorflow, pytorch, keras, jax, summary`);
  console.log(`  --run                Compile and train the model with TensorFlow.js`);
  console.log(`  --explain            Show feature importance after training (requires --run)`);
  console.log(`  --repl               Start interactive REPL mode`);
  console.log(`  --infer <csv>        Infer a program from a CSV file`);
  console.log(`  --compare <archs>    Compare architectures (e.g., --compare mlp cnn lstm resnet)`);
  console.log(`  --benchmark          Run all built-in datasets and produce results table`);
  console.log(`  --show-data          Show dataset preview`);
  console.log(`  --describe           Show dataset statistics and correlation`);
  console.log(`  --serve [port]       Start HTTP API server (default: 3000)`);
  console.log(`  --stdin              Read program from stdin (auto-detected if piped)`);
  console.log(`  --tokens             Show the token stream`);
  console.log(`  --ast                Show the AST`);
  console.log(`  --ir                 Show the intermediate representation`);
  console.log(`  --timings            Show compilation phase timings`);
  console.log(`  --help               Show this help message`);
  console.log();
  console.log(`${B}Language Keywords:${X}`);
  console.log(`  ${Y}Required${X}:  task  predict  inputs  dataset`);
  console.log(`  ${Y}Architecture${X}: architecture  loss  learn  optimizer  epochs  layers  activation  batch_norm`);
  console.log(`  ${Y}CNN/ResNet${X}: filters  kernel_size  pool_size  input_shape  pretrained  freeze_layers`);
  console.log(`  ${Y}RNN${X}:       sequence_length  embedding_dim  bidirectional`);
  console.log(`  ${Y}Training${X}:  batch_size  dropout  learning_rate  early_stop  lr_schedule  seed  tune  ensemble`);
  console.log(`  ${Y}Export${X}:    export_format (onnx, tflite, savedmodel, torchscript, coreml)`);
  console.log(`  ${Y}Preprocess${X}: normalize  split  cross_validate`);
  console.log(`  ${Y}Data${X}:      show  describe  sample  select  filter  augment`);
  console.log(`  ${Y}Output${X}:    validate  export`);
  console.log();
  console.log(`${B}Architectures:${X} mlp, cnn, lstm, gru, rnn, resnet, transformer, autoencoder`);
  console.log(`${B}Learn Modes:${X}  linear, nonlinear, deep, auto`);
  console.log(`${B}Pretrained:${X}   mobilenet, resnet50, resnet101, vgg16, vgg19, efficientnet, inception, densenet`);
  console.log(`${B}Datasets:${X}     iris, housing, titanic, wine, digits, sequences, timeseries`);
  console.log();
  console.log(`${B}Examples:${X}`);
  console.log(`  ${D}neurolang examples/iris.nl${X}`);
  console.log(`  ${D}neurolang examples/iris.nl --emit-code --target pytorch${X}`);
  console.log(`  ${D}neurolang examples/iris.nl --run --explain${X}`);
  console.log(`  ${D}neurolang --repl${X}`);
  console.log(`  ${D}neurolang --infer data.csv${X}`);
  console.log(`  ${D}neurolang examples/iris.nl --describe${X}`);
  console.log(`  ${D}echo "Predict species with a & b from iris.csv" | neurolang --run${X}`);
  console.log(`  ${D}neurolang --serve 3000${X}`);
}

interface CliFlags {
  file: string;
  emitCode: boolean;
  target: CodegenTarget;
  run: boolean;
  explain: boolean;
  repl: boolean;
  infer: string;
  benchmark: boolean;
  compare: string[];
  showData: boolean;
  describeData: boolean;
  showTokens: boolean;
  showAst: boolean;
  showIr: boolean;
  showTimings: boolean;
  stdin: boolean;
  serve: number;
}

function parseArgs(args: string[]): CliFlags | null {
  const flags: CliFlags = {
    file: "",
    emitCode: false,
    target: "tensorflow",
    run: false,
    explain: false,
    repl: false,
    infer: "",
    benchmark: false,
    compare: [],
    showData: false,
    describeData: false,
    showTokens: false,
    showAst: false,
    showIr: false,
    showTimings: false,
    stdin: false,
    serve: 0,
  };

  let i = 0;
  while (i < args.length) {
    const arg = args[i]!;
    switch (arg) {
      case "--help": case "-h":
        return null;
      case "--emit-code":
        flags.emitCode = true; break;
      case "--target":
        i++;
        flags.target = (args[i] ?? "tensorflow") as CodegenTarget; break;
      case "--run":
        flags.run = true; break;
      case "--explain":
        flags.explain = true; break;
      case "--repl":
        flags.repl = true; break;
      case "--infer":
        i++;
        flags.infer = args[i] ?? ""; break;
      case "--benchmark":
        flags.benchmark = true; break;
      case "--compare": {
        i++;
        while (i < args.length && !args[i]!.startsWith("-")) {
          flags.compare.push(args[i]!);
          i++;
        }
        i--;
        if (flags.compare.length === 0) flags.compare = ["mlp", "cnn", "lstm"];
        break;
      }
      case "--show-data": case "--data":
        flags.showData = true; break;
      case "--describe":
        flags.describeData = true; break;
      case "--stdin": case "-":
        flags.stdin = true; break;
      case "--serve":
        i++;
        flags.serve = parseInt(args[i] ?? "3000") || 3000; break;
      case "--tokens":
        flags.showTokens = true; break;
      case "--ast":
        flags.showAst = true; break;
      case "--ir":
        flags.showIr = true; break;
      case "--timings":
        flags.showTimings = true; break;
      default:
        if (arg.startsWith("-")) {
          console.error(`${R}Unknown option: ${arg}${X}`);
          return null;
        }
        flags.file = arg;
    }
    i++;
  }

  return flags;
}

async function runAutoArchSearch(
  ir: import("./ir.js").NeuralNetworkIR,
): Promise<AutoSearchResult> {
  const candidates = getCandidates();
  const probeEpochs = Math.min(10, Math.floor(ir.training.epochs * 0.1) || 5);
  const results: ArchCandidate[] = [];

  console.log(`  ${Y}Auto-architecture search: probing ${candidates.length} architectures (${probeEpochs} epochs each)${X}`);

  for (const candidate of candidates) {
    const probeIR = buildCandidateIR(ir, candidate.layers, probeEpochs);
    try {
      const probeResult = await execute(probeIR);
      results.push({
        name: candidate.name,
        layers: candidate.layers,
        probeLoss: probeResult.finalLoss,
        probeMetric: probeResult.metric.value,
      });
      console.log(`    ${D}${candidate.name.padEnd(8)} ${JSON.stringify(candidate.layers).padEnd(18)} loss=${probeResult.finalLoss.toFixed(4)}${X}`);
    } catch {
      results.push({
        name: candidate.name,
        layers: candidate.layers,
        probeLoss: Infinity,
        probeMetric: 0,
      });
    }
  }

  let bestIdx = 0;
  for (let i = 1; i < results.length; i++) {
    if (results[i]!.probeLoss < results[bestIdx]!.probeLoss) {
      bestIdx = i;
    }
  }

  return {
    candidates: results,
    bestIdx,
    bestLayers: results[bestIdx]!.layers,
  };
}

async function executeWithExplain(
  ir: import("./ir.js").NeuralNetworkIR,
): Promise<import("./explain.js").FeatureImportance[]> {
  const tf = await import("@tensorflow/tfjs");
  const { resolveDataset, parseCSV } = await import("./datasets.js");

  const ds = resolveDataset(ir.dataset);
  let features: number[][];
  let targets: number[];

  if (ds) {
    features = ds.features;
    targets = ds.targets;
    if (ir.features.length < ds.featureNames.length) {
      const colIdx = ir.features.map((f) => {
        const idx = ds.featureNames.indexOf(f);
        return idx >= 0 ? idx : ds.featureNames.findIndex((n) => n.toLowerCase() === f.toLowerCase());
      });
      if (colIdx.every((i) => i >= 0)) {
        features = features.map((row) => colIdx.map((i) => row[i]!));
      }
    }
  } else {
    const csvP = path.resolve(ir.dataset);
    const text = fs.readFileSync(csvP, "utf-8");
    const parsed = parseCSV(text);
    const featureIdx = ir.features.map((f) => parsed.headers.indexOf(f));
    const targetIdx = parsed.headers.indexOf(ir.target);
    features = parsed.rows.map((row) => featureIdx.map((i) => row[i]!));
    targets = parsed.rows.map((row) => row[targetIdx]!);
  }

  // Shuffle data (seeded for reproducibility)
  const indices = Array.from({ length: features.length }, (_, i) => i);
  let seed = (ir.preprocessing.seed ?? 42) | 0;
  for (let i = indices.length - 1; i > 0; i--) {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    const r = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    const j = Math.floor(r * (i + 1));
    [indices[i], indices[j]] = [indices[j]!, indices[i]!];
  }
  features = indices.map((i) => features[i]!);
  targets = indices.map((i) => targets[i]!);

  // Normalize features
  if (ir.preprocessing.normalize) {
    const cols = features[0]!.length;
    const means = new Array<number>(cols).fill(0);
    const stds = new Array<number>(cols).fill(0);
    for (const row of features) {
      for (let c = 0; c < cols; c++) means[c] += row[c]!;
    }
    for (let c = 0; c < cols; c++) means[c] /= features.length;
    for (const row of features) {
      for (let c = 0; c < cols; c++) stds[c] += (row[c]! - means[c]!) ** 2;
    }
    for (let c = 0; c < cols; c++) stds[c] = Math.sqrt(stds[c]! / features.length) || 1;
    features = features.map((row) => row.map((v, c) => (v - means[c]!) / stds[c]!));
  }

  const splitIdx = Math.floor(features.length * ir.preprocessing.trainTestSplit);

  const model = tf.sequential();
  let isFirst = true;
  for (const layer of ir.architecture.layers) {
    if (layer.kind === "dense") {
      if (isFirst) {
        model.add(tf.layers.dense({ inputShape: [ir.architecture.inputSize], units: layer.units, activation: layer.activation as "relu" }));
        isFirst = false;
      } else {
        model.add(tf.layers.dense({ units: layer.units, activation: layer.activation as "relu" }));
      }
    } else if (layer.kind === "batchnorm") {
      model.add(tf.layers.batchNormalization());
    } else if (layer.kind === "dropout") {
      model.add(tf.layers.dropout({ rate: layer.rate }));
    }
  }

  const lr = ir.training.optimizer.learningRate;
  const optName = ir.training.optimizer.name;
  const opt = optName === "sgd" ? tf.train.sgd(lr) : optName === "rmsprop" ? tf.train.rmsprop(lr) : tf.train.adam(lr);
  const loss: string | ((y: import("@tensorflow/tfjs").Tensor, p: import("@tensorflow/tfjs").Tensor) => import("@tensorflow/tfjs").Scalar) =
    ir.training.loss === "huberLoss"
      ? (y, p) => tf.losses.huberLoss(y, p).mean() as import("@tensorflow/tfjs").Scalar
      : ir.training.loss;

  model.compile({
    optimizer: opt,
    loss,
    metrics: ir.task === "classification" ? ["accuracy"] : ["mse"],
  });

  const xTrain = tf.tensor2d(features.slice(0, splitIdx));
  const xTest = tf.tensor2d(features.slice(splitIdx));
  let yTrain: import("@tensorflow/tfjs").Tensor;
  let yTest: import("@tensorflow/tfjs").Tensor;
  if (ir.task === "classification") {
    yTrain = tf.oneHot(tf.tensor1d(targets.slice(0, splitIdx), "int32"), ir.architecture.outputSize).cast("float32");
    yTest = tf.oneHot(tf.tensor1d(targets.slice(splitIdx), "int32"), ir.architecture.outputSize).cast("float32");
  } else {
    yTrain = tf.tensor2d(targets.slice(0, splitIdx).map((v) => [v]));
    yTest = tf.tensor2d(targets.slice(splitIdx).map((v) => [v]));
  }

  await model.fit(xTrain, yTrain, {
    epochs: ir.training.epochs,
    batchSize: ir.training.batchSize,
    validationSplit: 0.1,
    shuffle: true,
    verbose: 0,
  });

  const importance = await computeFeatureImportance(tf, model as any, xTest as any, yTest, ir);

  xTrain.dispose(); yTrain.dispose();
  xTest.dispose(); yTest.dispose();
  model.dispose();

  return importance;
}

const BENCHMARK_PROGRAMS: { name: string; program: string; task: string }[] = [
  {
    name: "Iris",
    task: "3-class",
    program: `task classification\npredict species\ninputs sepal_length sepal_width petal_length petal_width\ndataset iris.csv\nlearn nonlinear\nepochs 40`,
  },
  {
    name: "Wine",
    task: "3-class",
    program: `task classification\npredict cultivar\ninputs alcohol malic_acid ash alcalinity magnesium phenols flavanoids nonflavanoid_phenols proanthocyanins color_intensity hue od280 proline\ndataset wine.csv\nlearn deep\nepochs 40`,
  },
  {
    name: "Titanic",
    task: "binary",
    program: `task classification\npredict survived\ninputs pclass sex age sibsp parch fare embarked\ndataset titanic.csv\nloss binary_cross_entropy\nlearn nonlinear\nepochs 40`,
  },
  {
    name: "Digits",
    task: "10-class",
    program: `task classification\npredict digit\ninputs ${Array.from({ length: 64 }, (_, i) => `pixel_${i}`).join(" ")}\ndataset digits.csv\nlearn deep\nepochs 20`,
  },
  {
    name: "Housing",
    task: "regression",
    program: `task regression\npredict price\ninputs size bedrooms bathrooms age zipcode\ndataset housing.csv\nloss mse\nlearn nonlinear\nepochs 40`,
  },
];

async function runBenchmark(): Promise<void> {
  banner();
  console.log(`${B}${Y}══════════════════════════════════════════════════${X}`);
  console.log(`${B}${Y}    NeuroLang Benchmark Suite                     ${X}`);
  console.log(`${B}${Y}══════════════════════════════════════════════════${X}`);
  console.log();

  interface BenchResult {
    name: string;
    task: string;
    params: number;
    arch: string;
    metric: string;
    metricVal: number;
    loss: number;
    time: number;
    compileMs: number;
  }
  const results: BenchResult[] = [];

  for (const bench of BENCHMARK_PROGRAMS) {
    process.stdout.write(`  ${Y}Running ${bench.name}...${X}`);

    try {
      const compileStart = performance.now();
      const compiled = compile(bench.program, { target: "tensorflow" });
      const compileMs = performance.now() - compileStart;

      const layers = compiled.ir.architecture.layers
        .filter((l) => l.kind === "dense")
        .map((l) => l.kind === "dense" ? l.units : "?");
      const arch = `${compiled.ir.architecture.inputSize}→${layers.join("→")}`;

      const trainStart = performance.now();
      const trainResult = await execute(compiled.ir);
      const trainTime = performance.now() - trainStart;

      results.push({
        name: bench.name,
        task: bench.task,
        params: compiled.ir.meta.parameterCount,
        arch,
        metric: trainResult.metric.name,
        metricVal: trainResult.metric.value,
        loss: trainResult.finalLoss,
        time: trainTime / 1000,
        compileMs,
      });
      process.stdout.write(` ${G}done${X} (${(trainTime / 1000).toFixed(1)}s)\n`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      process.stdout.write(` ${R}FAILED: ${msg}${X}\n`);
    }
  }

  console.log();
  console.log(`${B}${Y}── Results ──${X}`);
  console.log();

  // Table header
  const hdr = [
    "Dataset".padEnd(10),
    "Task".padEnd(10),
    "Architecture".padEnd(20),
    "Params".padStart(8),
    "Metric".padEnd(10),
    "Score".padStart(8),
    "Loss".padStart(8),
    "Compile".padStart(8),
    "Train".padStart(8),
  ];
  console.log(`  ${D}${hdr.join("  ")}${X}`);
  console.log(`  ${"─".repeat(hdr.join("  ").length)}`);

  for (const r of results) {
    const scoreColor = r.metricVal >= 0.9 ? G : r.metricVal >= 0.7 ? Y : R;
    const row = [
      r.name.padEnd(10),
      r.task.padEnd(10),
      r.arch.padEnd(20),
      String(r.params.toLocaleString()).padStart(8),
      r.metric.padEnd(10),
      `${scoreColor}${r.metricVal.toFixed(4)}${X}`.padStart(8 + scoreColor.length + X.length),
      r.loss.toFixed(4).padStart(8),
      `${r.compileMs.toFixed(0)}ms`.padStart(8),
      `${r.time.toFixed(1)}s`.padStart(8),
    ];
    console.log(`  ${row.join("  ")}`);
  }

  console.log();
  const totalTime = results.reduce((s, r) => s + r.time, 0);
  const avgMetric = results.reduce((s, r) => s + r.metricVal, 0) / results.length;
  console.log(`  ${B}Total training time: ${totalTime.toFixed(1)}s${X}`);
  console.log(`  ${B}Average metric:      ${avgMetric.toFixed(4)}${X}`);
  console.log(`  ${B}Datasets tested:     ${results.length}/${BENCHMARK_PROGRAMS.length}${X}`);
  console.log();
}

async function runComparison(
  filePath: string,
  architectures: string[],
  target: CodegenTarget,
): Promise<void> {
  banner();
  console.log(`${B}${Y}══════════════════════════════════════════════════${X}`);
  console.log(`${B}${Y}    NeuroLang Model Comparison                    ${X}`);
  console.log(`${B}${Y}══════════════════════════════════════════════════${X}`);
  console.log();

  const resolved = path.resolve(filePath);
  if (!fs.existsSync(resolved)) {
    console.error(`${R}Error: File not found: ${resolved}${X}`);
    process.exit(1);
  }

  const source = fs.readFileSync(resolved, "utf-8");

  interface CompareResult {
    arch: string;
    metric: string;
    metricVal: number;
    loss: number;
    time: number;
    layers: number;
  }
  const results: CompareResult[] = [];

  for (const arch of architectures) {
    process.stdout.write(`  ${Y}Training with ${arch.toUpperCase()}...${X}`);

    // Inject architecture keyword into the source
    let modifiedSource = source;
    if (/\barchitecture\s+\w+/i.test(modifiedSource)) {
      modifiedSource = modifiedSource.replace(/\barchitecture\s+\w+/i, `architecture ${arch}`);
    } else {
      modifiedSource += `\narchitecture ${arch}`;
    }

    try {
      const compiled = compile(modifiedSource, { target });
      const trainStart = performance.now();
      const trainResult = await execute(compiled.ir, {
        onEpochEnd: () => {},
      });
      const trainTime = performance.now() - trainStart;

      results.push({
        arch: arch.toUpperCase(),
        metric: trainResult.metric.name,
        metricVal: trainResult.metric.value,
        loss: trainResult.finalLoss,
        time: trainTime / 1000,
        layers: compiled.ir.architecture.layers.length,
      });
      process.stdout.write(` ${G}done${X} (${(trainTime / 1000).toFixed(1)}s)\n`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      process.stdout.write(` ${R}FAILED${X}\n`);
      console.error(`    ${R}${msg}${X}`);
      results.push({ arch: arch.toUpperCase(), metric: "N/A", metricVal: 0, loss: Infinity, time: 0, layers: 0 });
    }
  }

  console.log();
  console.log(`${B}${Y}── Comparison Results ──${X}`);
  console.log();

  const hdr = [
    "Architecture".padEnd(14),
    "Layers".padStart(7),
    "Metric".padEnd(10),
    "Score".padStart(8),
    "Loss".padStart(8),
    "Time".padStart(8),
  ];
  console.log(`  ${D}${hdr.join("  ")}${X}`);
  console.log(`  ${"─".repeat(hdr.join("  ").length)}`);

  let bestIdx = 0;
  for (let i = 1; i < results.length; i++) {
    if (results[i]!.metricVal > results[bestIdx]!.metricVal) bestIdx = i;
  }

  for (let i = 0; i < results.length; i++) {
    const r = results[i]!;
    const isBest = i === bestIdx && r.metricVal > 0;
    const scoreColor = r.metricVal >= 0.9 ? G : r.metricVal >= 0.7 ? Y : R;
    const prefix = isBest ? `${G}★${X}` : " ";
    const row = [
      `${prefix}${r.arch.padEnd(13)}`,
      String(r.layers).padStart(7),
      r.metric.padEnd(10),
      `${scoreColor}${r.metricVal.toFixed(4)}${X}`.padStart(8 + scoreColor.length + X.length),
      r.loss === Infinity ? "   N/A  " : r.loss.toFixed(4).padStart(8),
      r.time === 0 ? "   N/A  " : `${r.time.toFixed(1)}s`.padStart(8),
    ];
    console.log(`  ${row.join("  ")}`);
  }

  if (results[bestIdx]!.metricVal > 0) {
    console.log();
    console.log(`  ${G}${B}Winner: ${results[bestIdx]!.arch}${X} with ${results[bestIdx]!.metric} = ${G}${results[bestIdx]!.metricVal.toFixed(4)}${X}`);
  }

  console.log();
  const totalTime = results.reduce((s, r) => s + r.time, 0);
  console.log(`  ${D}Total comparison time: ${totalTime.toFixed(1)}s${X}`);
  console.log();
}

function renderConfusionMatrix(
  predictions: { actual: number; predicted: number }[],
  numClasses: number,
): string {
  const matrix: number[][] = Array.from({ length: numClasses }, () =>
    new Array(numClasses).fill(0),
  );
  for (const p of predictions) {
    const a = Math.round(p.actual);
    const pr = Math.round(p.predicted);
    if (a >= 0 && a < numClasses && pr >= 0 && pr < numClasses) {
      matrix[a]![pr]!++;
    }
  }

  const maxVal = Math.max(...matrix.flat());
  const cellW = Math.max(3, String(maxVal).length + 1);
  const labelW = Math.max(5, String(numClasses - 1).length + 2);

  const lines: string[] = [];

  // Header
  let header = " ".repeat(labelW + 2) + `${D}Predicted${X}`;
  lines.push(header);
  let colLabels = " ".repeat(labelW + 1);
  for (let c = 0; c < numClasses; c++) {
    colLabels += String(c).padStart(cellW);
  }
  lines.push(`  ${D}${colLabels}${X}`);

  const sep = " ".repeat(labelW) + "┌" + "─".repeat(cellW * numClasses + 1) + "┐";
  lines.push(`  ${sep}`);

  for (let r = 0; r < numClasses; r++) {
    const label = r === Math.floor(numClasses / 2)
      ? `${D}Actual${X} ${String(r).padStart(labelW - 7)}`
      : `${"".padStart(labelW - String(r).length)}${r}`;
    let row = `  ${label} │`;
    for (let c = 0; c < numClasses; c++) {
      const v = matrix[r]![c]!;
      const color = r === c ? G : (v > 0 ? R : D);
      row += `${color}${String(v).padStart(cellW)}${X}`;
    }
    row += " │";
    lines.push(row);
  }

  const botSep = " ".repeat(labelW) + "└" + "─".repeat(cellW * numClasses + 1) + "┘";
  lines.push(`  ${botSep}`);

  // Per-class precision/recall
  lines.push("");
  lines.push(`  ${D}Per-class metrics:${X}`);
  for (let c = 0; c < numClasses; c++) {
    const tp = matrix[c]![c]!;
    const colSum = matrix.reduce((s, row) => s + row[c]!, 0);
    const rowSum = matrix[c]!.reduce((s, v) => s + v, 0);
    const precision = colSum > 0 ? tp / colSum : 0;
    const recall = rowSum > 0 ? tp / rowSum : 0;
    const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
    lines.push(`  Class ${c}: precision=${precision.toFixed(2)}  recall=${recall.toFixed(2)}  f1=${f1.toFixed(2)}`);
  }

  return lines.join("\n");
}

function renderLossCurve(losses: number[]): string {
  const height = 10;
  const width = Math.min(60, losses.length);
  const step = Math.max(1, Math.floor(losses.length / width));
  const sampled = Array.from({ length: width }, (_, i) => losses[Math.min(i * step, losses.length - 1)]!);

  const maxLoss = Math.max(...sampled);
  const minLoss = Math.min(...sampled);
  const range = maxLoss - minLoss || 1;

  const grid: string[][] = Array.from({ length: height }, () => new Array(width).fill(" "));

  for (let x = 0; x < width; x++) {
    const normalized = (sampled[x]! - minLoss) / range;
    const row = Math.floor((1 - normalized) * (height - 1));
    grid[row]![x] = `${Y}█${X}`;
  }

  const lines: string[] = [];
  for (let r = 0; r < height; r++) {
    const label = r === 0
      ? maxLoss.toFixed(4).padStart(10)
      : r === height - 1
        ? minLoss.toFixed(4).padStart(10)
        : "".padStart(10);
    lines.push(`  ${D}${label}${X} │${grid[r]!.join("")}│`);
  }
  lines.push(`  ${"".padStart(10)} └${"─".repeat(width)}┘`);
  lines.push(`  ${"".padStart(10)}  ${D}epoch 1${" ".repeat(Math.max(0, width - 14))}epoch ${losses.length}${X}`);

  return lines.join("\n");
}

async function main(): Promise<void> {
  let args = process.argv.slice(2);

  if (args.length === 0) {
    usage();
    process.exit(0);
  }

  // Support "neurolang run <file>" as a natural subcommand
  if (args[0] === "run" && args.length >= 2 && !args[1]!.startsWith("-")) {
    args = ["--run", ...args.slice(1)];
  }

  const flags = parseArgs(args);
  if (!flags) {
    usage();
    process.exit(1);
  }

  if (flags.repl) {
    await startRepl();
    return;
  }

  if (flags.serve) {
    const { startServer } = await import("./serve.js");
    await startServer(flags.serve);
    return;
  }

  if (flags.infer) {
    banner();
    const csvPath = path.resolve(flags.infer);
    if (!fs.existsSync(csvPath)) {
      console.error(`${R}Error: File not found: ${csvPath}${X}`);
      process.exit(1);
    }
    try {
      const schema = inferSchema(csvPath);
      console.log(formatSchema(schema));
      console.log();

      if (flags.run) {
        console.log(`${Y}Compiling and running inferred program...${X}`);
        console.log();
        const result = compile(schema.suggestedProgram, { target: flags.target });
        console.log(emitSummary(result.ir));

        const trainResult = await execute(result.ir, {
          onEpochEnd: (epoch, loss) => {
            if ((epoch + 1) % Math.max(1, Math.floor(result.ir.training.epochs / 10)) === 0 || epoch === 0) {
              process.stdout.write(`  ${D}Epoch ${String(epoch + 1).padStart(4)}/${result.ir.training.epochs}: loss=${loss.toFixed(4)}${X}\n`);
            }
          },
        });

        console.log();
        console.log(`${G}${B}Training Complete${X}`);
        console.log(`  ${trainResult.metric.name}: ${trainResult.metric.value.toFixed(4)}`);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`${R}${message}${X}`);
      process.exit(1);
    }
    return;
  }

  if (flags.benchmark) {
    await runBenchmark();
    return;
  }

  if (flags.compare.length > 0 && flags.file) {
    await runComparison(flags.file, flags.compare, flags.target);
    return;
  }

  // Stdin pipe support
  if (flags.stdin || (!flags.file && !process.stdin.isTTY)) {
    const chunks: Buffer[] = [];
    for await (const chunk of process.stdin) {
      chunks.push(chunk as Buffer);
    }
    const source = Buffer.concat(chunks).toString("utf-8").trim();
    if (!source) {
      console.error(`${R}Error: No input received from stdin${X}`);
      process.exit(1);
    }

    banner();
    console.log(`${D}Compiling from stdin...${X}`);

    const { isNaturalSyntax: isNatural, parseNatural: parseNat } = await import("./natural.js");
    if (isNatural(source)) {
      const natResult = parseNat(source);
      console.log(`${C}Natural language detected → translated to NeuroLang${X}`);
      console.log(`${D}${natResult.program.split("\n").map((l: string) => "  │ " + l).join("\n")}${X}`);
    }
    console.log();

    try {
      const result = compile(source, { target: flags.target, filename: "<stdin>" });
      console.log(emitSummary(result.ir));
      console.log(`${D}Compiled in ${result.timings.total.toFixed(1)}ms${X}`);
      console.log();
      // Data inspection in stdin mode
      const stdinShow = flags.showData || result.ir.data.show;
      const stdinDescribe = flags.describeData || result.ir.data.describe;
      const stdinSample = result.ir.data.sample;
      if (stdinShow || stdinDescribe || stdinSample) {
        const { resolveDataset: resDS } = await import("./datasets.js");
        const ds = resDS(result.ir.dataset);
        if (ds) {
          if (stdinShow) {
            const mx = typeof result.ir.data.show === "number" ? result.ir.data.show : 10;
            console.log(`${B}${Y}── Data Preview ──${X}`);
            console.log(formatDataPreview(ds, mx));
            console.log();
          }
          if (stdinDescribe) {
            const desc = describeDataset(ds);
            console.log(`${B}${Y}── Data Description ──${X}`);
            console.log(formatDescription(desc));
            console.log();
            console.log(formatCorrelation(ds));
            console.log();
          }
          if (stdinSample) {
            console.log(`${B}${Y}── Data Sample ──${X}`);
            console.log(formatSample(ds, stdinSample));
            console.log();
          }
        }
      }
      if (flags.emitCode) {
        console.log(`${B}${Y}── Generated Code (${flags.target}) ──${X}`);
        console.log();
        console.log(result.code);
      }
      if (flags.run) {
        console.log(`${B}${Y}── Executing Model ──${X}`);
        console.log();
        const trainResult = await execute(result.ir, {
          onEpochEnd: (epoch, loss) => {
            if ((epoch + 1) % Math.max(1, Math.floor(result.ir.training.epochs / 10)) === 0 || epoch === 0) {
              process.stdout.write(`  ${D}Epoch ${String(epoch + 1).padStart(4)}/${result.ir.training.epochs}: loss=${loss.toFixed(4)}${X}\n`);
            }
          },
        });
        console.log();
        console.log(`${G}${B}Training Complete${X}`);
        console.log(`  ${trainResult.metric.name}:  ${trainResult.metric.value.toFixed(4)}`);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error(`${R}${message}${X}`);
      process.exit(1);
    }
    return;
  }

  if (!flags.file) {
    console.error(`${R}Error: No input file specified${X}`);
    usage();
    process.exit(1);
  }

  const filePath = path.resolve(flags.file);
  if (!fs.existsSync(filePath)) {
    console.error(`${R}Error: File not found: ${filePath}${X}`);
    process.exit(1);
  }

  const source = fs.readFileSync(filePath, "utf-8");
  const filename = path.basename(filePath);

  banner();
  console.log(`${D}Compiling: ${filePath}${X}`);

  const { isNaturalSyntax: isNatural, parseNatural: parseNat } = await import("./natural.js");
  if (isNatural(source)) {
    const natResult = parseNat(source);
    console.log(`${C}Natural language detected → translated to NeuroLang${X}`);
    console.log(`${D}${natResult.program.split("\n").map((l: string) => "  │ " + l).join("\n")}${X}`);
  }
  console.log();

  try {
    const result = compile(source, { target: flags.target, filename });

    const warnings = result.diagnostics.filter((d) => d.severity === Severity.Warning);
    if (warnings.length > 0) {
      console.log(formatDiagnostics(warnings as Diagnostic[], source, filename));
      console.log();
    }

    console.log(emitSummary(result.ir));

    if (flags.showTimings) {
      console.log(`${B}${Y}── Phase Timings ──${X}`);
      console.log(`  Lexer:     ${result.timings.lex.toFixed(2)}ms`);
      console.log(`  Parser:    ${result.timings.parse.toFixed(2)}ms`);
      console.log(`  Analyzer:  ${result.timings.analyze.toFixed(2)}ms`);
      console.log(`  Codegen:   ${result.timings.codegen.toFixed(2)}ms`);
      console.log(`  ${B}Total:     ${result.timings.total.toFixed(2)}ms${X}`);
      console.log();
    } else {
      console.log(`${D}Compiled in ${result.timings.total.toFixed(1)}ms${X}`);
      console.log();
    }

    if (flags.showTokens) {
      console.log(`${B}${Y}── Token Stream (${result.tokens.length} tokens) ──${X}`);
      for (const tok of result.tokens) {
        const padLine = String(tok.line).padStart(3);
        const padCol = String(tok.column).padStart(2);
        console.log(`  ${D}${padLine}:${padCol}${X}  ${Y}${tok.type.padEnd(7)}${X}  ${tok.value}`);
      }
      console.log();
    }

    if (flags.showAst) {
      console.log(`${B}${Y}── AST (${result.ast.body.length} statements) ──${X}`);
      console.log(JSON.stringify(result.ast, null, 2));
      console.log();
    }

    if (flags.showIr) {
      console.log(`${B}${Y}── Intermediate Representation ──${X}`);
      console.log(JSON.stringify(result.ir, null, 2));
      console.log();
    }

    // Data inspection (from keywords or CLI flags)
    const wantShow = flags.showData || result.ir.data.show;
    const wantDescribe = flags.describeData || result.ir.data.describe;
    const wantSample = result.ir.data.sample;

    if (wantShow || wantDescribe || wantSample) {
      const { resolveDataset } = await import("./datasets.js");
      const ds = resolveDataset(result.ir.dataset);
      if (ds) {
        if (wantShow) {
          const maxRows = typeof result.ir.data.show === "number" ? result.ir.data.show : 10;
          console.log(`${B}${Y}── Data Preview ──${X}`);
          console.log(formatDataPreview(ds, maxRows));
          console.log();
        }
        if (wantDescribe) {
          const desc = describeDataset(ds);
          console.log(`${B}${Y}── Data Description ──${X}`);
          console.log(formatDescription(desc));
          console.log();
          console.log(formatCorrelation(ds));
          console.log();
        }
        if (wantSample) {
          console.log(`${B}${Y}── Data Sample ──${X}`);
          console.log(formatSample(ds, wantSample, result.ir.preprocessing.seed ?? 42));
          console.log();
        }
      } else {
        console.log(`${D}Data inspection requires a built-in dataset (iris, housing, titanic, wine, digits)${X}`);
        console.log();
      }
    }

    if (flags.emitCode) {
      console.log(`${B}${Y}── Generated Code (${flags.target}) ──${X}`);
      console.log();
      console.log(result.code);
    }

    if (flags.run) {
      // Check dataset exists before attempting training
      const ds = result.ir.dataset;
      const BUILTIN = new Set(["iris", "housing", "titanic", "wine", "digits"]);
      if (ds && !BUILTIN.has(ds.toLowerCase().replace(".csv", ""))) {
        const dsPath = path.resolve(path.dirname(filePath), ds);
        const dsPathCwd = path.resolve(ds);
        if (!fs.existsSync(dsPath) && !fs.existsSync(dsPathCwd)) {
          console.log(`${R}Dataset not found: ${ds}${X}`);
          console.log(`${D}Place the CSV file in the same directory as your .nl file, or use a built-in dataset:${X}`);
          console.log(`${D}  Built-in: iris, titanic, wine, digits, housing${X}`);
          console.log(`${D}  Or run without --run to just compile and see the architecture.${X}`);
          process.exit(1);
        }
      }

      // Check TensorFlow.js is available
      try {
        await import("@tensorflow/tfjs");
      } catch {
        console.log(`${R}TensorFlow.js is not installed.${X}`);
        console.log(`${D}Training requires TensorFlow.js. Install it in your project:${X}`);
        console.log(`${Y}  npm install @tensorflow/tfjs${X}`);
        console.log();
        console.log(`${D}To just compile and see the architecture, run without --run:${X}`);
        console.log(`${Y}  neurolang ${filePath}${X}`);
        process.exit(1);
      }

      console.log(`${B}${Y}── Executing Model ──${X}`);
      console.log();

      let finalIR = result.ir;

      // Auto-architecture search
      if (result.ir.meta.learnMode === "auto") {
        const searchResult = await runAutoArchSearch(result.ir);
        console.log(formatSearchResults(searchResult));
        console.log();
        finalIR = buildCandidateIR(result.ir, searchResult.bestLayers, result.ir.training.epochs);
      }

      // Hyperparameter tuning
      if (result.ir.training.tune) {
        console.log(`  ${Y}Hyperparameter tuning: ${totalTrials()} configurations${X}`);
        const config = getTuneConfig();
        const trials: TuneTrialResult[] = [];

        for (const arch of config.architectures) {
          for (const lr of config.learningRates) {
            for (const bs of config.batchSizes) {
              const probeEpochs = 10;
              const trialIR = buildTrialIR(result.ir, lr, bs, arch.layers, probeEpochs);
              try {
                const trialResult = await execute(trialIR);
                trials.push({
                  lr,
                  batchSize: bs,
                  archName: arch.name,
                  layers: arch.layers,
                  probeLoss: trialResult.finalLoss,
                  probeMetric: trialResult.metric.value,
                });
                process.stdout.write(`    ${D}lr=${lr} bs=${bs} arch=${arch.name} → loss=${trialResult.finalLoss.toFixed(4)}${X}\n`);
              } catch {
                trials.push({
                  lr,
                  batchSize: bs,
                  archName: arch.name,
                  layers: arch.layers,
                  probeLoss: Infinity,
                  probeMetric: 0,
                });
              }
            }
          }
        }

        let bestIdx = 0;
        for (let i = 1; i < trials.length; i++) {
          if (trials[i]!.probeLoss < trials[bestIdx]!.probeLoss) bestIdx = i;
        }

        const tuneResult: TuneResult = { trials, bestIdx, bestConfig: { lr: trials[bestIdx]!.lr, batchSize: trials[bestIdx]!.batchSize, layers: trials[bestIdx]!.layers } };
        console.log(formatTuneResults(tuneResult));
        console.log();

        const best = trials[bestIdx]!;
        finalIR = buildTrialIR(result.ir, best.lr, best.batchSize, best.layers, result.ir.training.epochs);
      }

      // Ensemble training
      const ensembleN = finalIR.training.ensemble ?? 1;
      const isEnsemble = ensembleN > 1;

      if (isEnsemble) {
        console.log(`  ${Y}Ensemble mode: training ${ensembleN} models with bagging${X}`);
        console.log();
      }

      const allEnsembleResults: import("./runtime.js").TrainingResult[] = [];
      const trainStart = performance.now();

      for (let modelIdx = 0; modelIdx < ensembleN; modelIdx++) {
        // Same data split for all ensemble members — only TF weight init differs
        const memberIR = finalIR;

        if (isEnsemble) {
          process.stdout.write(`  ${Y}Model ${modelIdx + 1}/${ensembleN}${X}\n`);
        }

        const memberResult = await execute(memberIR, {
          onEpochEnd: (epoch, loss) => {
            if ((epoch + 1) % Math.max(1, Math.floor(memberIR.training.epochs / 10)) === 0 || epoch === 0) {
              const prefix = isEnsemble ? `    ` : `  `;
              process.stdout.write(`${prefix}${D}Epoch ${String(epoch + 1).padStart(4)}/${memberIR.training.epochs}: loss=${loss.toFixed(4)}${X}\n`);
            }
          },
        });
        allEnsembleResults.push(memberResult);
      }
      const trainTime = performance.now() - trainStart;

      // Aggregate ensemble predictions if needed
      let trainResult = allEnsembleResults[0]!;
      if (isEnsemble && allEnsembleResults.length > 1) {
        const preds = trainResult.predictions.map((p, i) => {
          let finalPredicted: number;
          if (finalIR.task === "classification") {
            // True majority vote
            const votes: Record<number, number> = {};
            for (const r of allEnsembleResults) {
              const v = Math.round(r.predictions[i]?.predicted ?? 0);
              votes[v] = (votes[v] ?? 0) + 1;
            }
            finalPredicted = Number(
              Object.entries(votes).reduce((best, [cls, cnt]) =>
                cnt > best[1] ? [cls, cnt] : best, ["0", 0])[0],
            );
          } else {
            finalPredicted = allEnsembleResults.reduce((s, r) => s + (r.predictions[i]?.predicted ?? 0), 0) / allEnsembleResults.length;
          }
          return { actual: p.actual, predicted: finalPredicted };
        });

        const avgLoss = allEnsembleResults.reduce((s, r) => s + r.finalLoss, 0) / allEnsembleResults.length;
        const avgMetric = allEnsembleResults.reduce((s, r) => s + r.metric.value, 0) / allEnsembleResults.length;

        // For classification, recalculate accuracy from ensemble predictions
        let ensembleMetricVal = avgMetric;
        if (finalIR.task === "classification") {
          const correct = preds.filter((p) => Math.abs(p.actual - p.predicted) < 0.5).length;
          ensembleMetricVal = preds.length > 0 ? correct / preds.length : 0;
        }

        trainResult = {
          ...trainResult,
          predictions: preds,
          finalLoss: avgLoss,
          metric: { name: trainResult.metric.name, value: ensembleMetricVal },
          lossHistory: allEnsembleResults[0]!.lossHistory,
        };
      }

      console.log();
      console.log(`${G}${B}Training Complete${X}`);
      if (isEnsemble) {
        console.log(`  Ensemble:    ${ensembleN} models (majority vote)`);
        console.log(`  Avg Loss:    ${trainResult.finalLoss.toFixed(4)}`);
      } else {
        console.log(`  Epochs:      ${trainResult.epochs}${trainResult.earlyStopEpoch ? ` (early stopped at ${trainResult.earlyStopEpoch})` : ""}`);
        console.log(`  Final Loss:  ${trainResult.finalLoss.toFixed(4)}`);
      }
      console.log(`  ${trainResult.metric.name}:  ${trainResult.metric.value.toFixed(4)}`);
      console.log(`  Train time:  ${(trainTime / 1000).toFixed(2)}s`);
      console.log();

      console.log(`${B}Predictions (first ${trainResult.predictions.length} test samples):${X}`);
      for (const p of trainResult.predictions) {
        const isRegression = finalIR.task === "regression";
        const closeEnough = isRegression
          ? Math.abs(p.actual - p.predicted) / (Math.abs(p.actual) || 1) < 0.15
          : Math.abs(p.actual - p.predicted) < 0.5;
        const color = closeEnough ? G : R;
        if (isRegression) {
          console.log(`  actual: ${p.actual.toFixed(0).padStart(10)}  predicted: ${color}${p.predicted.toFixed(0).padStart(10)}${X}`);
        } else {
          console.log(`  actual: ${p.actual.toFixed(0)}  predicted: ${color}${p.predicted.toFixed(0)}${X}`);
        }
      }

      // Confusion matrix for classification
      if (finalIR.task === "classification" && trainResult.predictions.length > 0) {
        console.log();
        console.log(`${B}${Y}── Confusion Matrix ──${X}`);
        console.log(renderConfusionMatrix(trainResult.predictions, finalIR.architecture.outputSize));
      }

      // Cross-validation results
      if (trainResult.crossValidation) {
        const cv = trainResult.crossValidation;
        console.log(`${B}${Y}── Cross-Validation (${cv.folds}-fold) ──${X}`);
        for (let i = 0; i < cv.scores.length; i++) {
          console.log(`  Fold ${i + 1}: ${cv.metricName} = ${cv.scores[i]!.toFixed(4)}`);
        }
        console.log(`  ${B}Mean:  ${cv.mean.toFixed(4)} ± ${cv.std.toFixed(4)}${X}`);
        console.log();
      }

      // ASCII loss curve
      if (trainResult.lossHistory.length > 1) {
        console.log();
        console.log(`${B}${Y}── Loss Curve ──${X}`);
        console.log(renderLossCurve(trainResult.lossHistory));
      }

      if (finalIR.output.exportPath) {
        console.log();
        console.log(`${C}Model exported to: ${finalIR.output.exportPath}${X}`);
      }

      // Feature importance
      if (flags.explain) {
        console.log();
        console.log(`${B}${Y}── Feature Importance ──${X}`);
        console.log(`  ${D}Training a fresh model for permutation importance...${X}`);
        try {
          const explainResult = await executeWithExplain(finalIR);
          console.log(formatFeatureImportance(explainResult, finalIR));
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          console.log(`  ${R}Could not compute feature importance: ${msg}${X}`);
        }
      }
    }
  } catch (err) {
    if (err instanceof CompilationError) {
      console.log(formatDiagnostics(err.diagnostics as Diagnostic[], source, filename));
      process.exit(1);
    }
    const message = err instanceof Error ? err.message : String(err);
    console.error(`${R}${B}Error${X}`);
    console.error(`${R}${message}${X}`);
    process.exit(1);
  }
}

main();
