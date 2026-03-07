import type { ProgramNode, StatementNode, ValueNode } from "./ast.js";
import type {
  NeuralNetworkIR,
  TaskType,
  LossFunction,
  LayerSpec,
  OptimizerSpec,
  EarlyStopSpec,
  LRScheduleSpec,
  LRScheduleType,
  ArchitectureType,
  PretrainedModel,
  ExportFormat,
} from "./ir.js";
import { DiagnosticCollector, E } from "./diagnostics.js";

interface Loc { line: number; column: number; length: number }

interface RawConfig {
  task?: { value: string } & Loc;
  predict?: { value: string } & Loc;
  inputs?: { values: string[] } & Loc;
  dataset?: { value: string } & Loc;
  loss?: { value: string } & Loc;
  learn?: { value: string } & Loc;
  architecture?: { value: string } & Loc;
  optimizer?: { value: string } & Loc;
  epochs?: { value: number } & Loc;
  layers?: { values: number[] } & Loc;
  activation?: { value: string } & Loc;
  batchSize?: { value: number } & Loc;
  dropout?: { value: number } & Loc;
  normalize?: { value: boolean } & Loc;
  split?: { value: number } & Loc;
  learningRate?: { value: number } & Loc;
  batchNorm?: { value: boolean } & Loc;
  earlyStop?: { value: number } & Loc;
  lrSchedule?: { value: string } & Loc;
  seed?: { value: number } & Loc;
  validate?: { value: number } & Loc;
  exportPath?: { value: string } & Loc;
  tune?: { value: boolean } & Loc;
  crossValidate?: { value: number } & Loc;
  ensemble?: { value: number } & Loc;
  show?: { value: string } & Loc;
  describe?: { value: boolean } & Loc;
  sample?: { value: number } & Loc;
  augment?: { value: string } & Loc;
  select?: { values: string[] } & Loc;
  filter?: { value: string } & Loc;
  // CNN-specific
  filters?: { values: number[] } & Loc;
  kernelSize?: { value: number } & Loc;
  poolSize?: { value: number } & Loc;
  inputShape?: { values: number[] } & Loc;
  // RNN-specific
  sequenceLength?: { value: number } & Loc;
  embeddingDim?: { value: number } & Loc;
  bidirectional?: { value: boolean } & Loc;
  recurrentDropout?: { value: number } & Loc;
  // Transfer learning
  pretrained?: { value: string } & Loc;
  freezeLayers?: { value: number } & Loc;
  // Export
  exportFormats?: { values: string[] } & Loc;
  // Multi-output
  targets?: { values: string[] } & Loc;
}

const VALID_TASKS = new Set<string>(["classification", "regression"]);
const VALID_LEARN = new Set<string>(["linear", "nonlinear", "deep", "auto"]);
const VALID_OPTIMIZERS = new Set<string>(["adam", "sgd", "rmsprop", "adamw"]);
const VALID_ACTIVATIONS = new Set<string>(["relu", "sigmoid", "tanh", "selu", "elu", "gelu", "leaky_relu"]);
const VALID_LR_SCHEDULES = new Set<string>(["cosine", "step", "exponential", "none"]);
const VALID_ARCHITECTURES = new Set<string>(["mlp", "cnn", "rnn", "lstm", "gru", "autoencoder", "resnet", "transformer"]);
const VALID_PRETRAINED = new Set<string>(["mobilenet", "resnet50", "resnet101", "vgg16", "vgg19", "efficientnet", "inception", "densenet"]);
const VALID_EXPORT_FORMATS = new Set<string>(["onnx", "tflite", "savedmodel", "torchscript", "coreml"]);

const LOSS_MAP: Record<string, LossFunction> = {
  mse: "meanSquaredError",
  mae: "meanAbsoluteError",
  cross_entropy: "categoricalCrossentropy",
  binary_cross_entropy: "binaryCrossentropy",
  huber: "huberLoss",
};

function singleString(values: ValueNode[]): string {
  const v = values[0];
  if (!v) return "";
  return v.type === "Number" ? String(v.value) : v.value;
}

function singleNumber(values: ValueNode[]): number {
  const v = values[0];
  if (!v || v.type !== "Number") return NaN;
  return v.value;
}

function allStrings(values: ValueNode[]): string[] {
  return values.map((v) => (v.type === "Number" ? String(v.value) : v.value));
}

function allNumbers(values: ValueNode[]): number[] {
  return values.map((v) => (v.type === "Number" ? v.value : NaN));
}

function stmtSpan(stmt: StatementNode): Loc {
  const last = stmt.values[stmt.values.length - 1];
  const endCol = last ? last.location.column + (last.type === "Number" ? last.raw.length : last.value.length) : stmt.location.column + stmt.keyword.length;
  return {
    line: stmt.location.line,
    column: stmt.location.column,
    length: endCol - stmt.location.column,
  };
}

export class Analyzer {
  private collector: DiagnosticCollector;

  constructor(collector?: DiagnosticCollector) {
    this.collector = collector ?? new DiagnosticCollector();
  }

  analyze(program: ProgramNode): NeuralNetworkIR | null {
    const raw = this.extractConfig(program);

    if (!this.validateRequired(raw)) return null;
    if (!this.validateValues(raw)) return null;

    return this.buildIR(raw);
  }

  private extractConfig(program: ProgramNode): RawConfig {
    const raw: RawConfig = {};
    const seen = new Set<string>();

    for (const stmt of program.body) {
      if (seen.has(stmt.keyword)) {
        this.collector.warning(
          E.DUPLICATE_KEYWORD,
          `Duplicate keyword '${stmt.keyword}' — using latest value`,
          stmtSpan(stmt),
        );
      }
      seen.add(stmt.keyword);
      this.processStatement(stmt, raw);
    }

    return raw;
  }

  private processStatement(stmt: StatementNode, raw: RawConfig): void {
    const span = stmtSpan(stmt);

    switch (stmt.keyword) {
      case "task":
        raw.task = { value: singleString(stmt.values), ...span };
        break;
      case "predict":
        raw.predict = { value: singleString(stmt.values), ...span };
        break;
      case "inputs":
        raw.inputs = { values: allStrings(stmt.values), ...span };
        break;
      case "dataset":
        raw.dataset = { value: singleString(stmt.values), ...span };
        break;
      case "loss":
        raw.loss = { value: singleString(stmt.values), ...span };
        break;
      case "learn":
        raw.learn = { value: singleString(stmt.values), ...span };
        break;
      case "architecture":
        raw.architecture = { value: singleString(stmt.values).toLowerCase(), ...span };
        break;
      case "optimizer":
        raw.optimizer = { value: singleString(stmt.values), ...span };
        break;
      case "epochs":
        raw.epochs = { value: singleNumber(stmt.values), ...span };
        break;
      case "layers":
        raw.layers = { values: allNumbers(stmt.values), ...span };
        break;
      case "activation":
        raw.activation = { value: singleString(stmt.values), ...span };
        break;
      case "batch_size":
        raw.batchSize = { value: singleNumber(stmt.values), ...span };
        break;
      case "dropout":
        raw.dropout = { value: singleNumber(stmt.values), ...span };
        break;
      case "learning_rate":
        raw.learningRate = { value: singleNumber(stmt.values), ...span };
        break;
      case "batch_norm":
        raw.batchNorm = { value: singleString(stmt.values) !== "false", ...span };
        break;
      case "early_stop":
        raw.earlyStop = { value: singleNumber(stmt.values), ...span };
        break;
      case "lr_schedule":
        raw.lrSchedule = { value: singleString(stmt.values), ...span };
        break;
      case "seed":
        raw.seed = { value: singleNumber(stmt.values), ...span };
        break;
      case "validate":
        raw.validate = { value: singleNumber(stmt.values), ...span };
        break;
      case "export":
        raw.exportPath = { value: singleString(stmt.values), ...span };
        break;
      case "normalize": {
        const val = singleString(stmt.values);
        raw.normalize = { value: val !== "false" && val !== "0", ...span };
        break;
      }
      case "split":
        raw.split = { value: singleNumber(stmt.values), ...span };
        break;
      case "tune":
        raw.tune = { value: singleString(stmt.values) !== "false", ...span };
        break;
      case "cross_validate":
        raw.crossValidate = { value: singleNumber(stmt.values), ...span };
        break;
      case "ensemble":
        raw.ensemble = { value: singleNumber(stmt.values), ...span };
        break;
      case "show": {
        const val = singleString(stmt.values);
        raw.show = { value: val || "data", ...span };
        break;
      }
      case "describe":
        raw.describe = { value: singleString(stmt.values) !== "false", ...span };
        break;
      case "sample":
        raw.sample = { value: singleNumber(stmt.values) || 5, ...span };
        break;
      case "augment":
        raw.augment = { value: singleString(stmt.values), ...span };
        break;
      case "select":
        raw.select = { values: stmt.values.map((v) => v.type === "Number" ? String(v.value) : v.value), ...span };
        break;
      case "filter":
        raw.filter = { value: singleString(stmt.values), ...span };
        break;
      // CNN-specific
      case "filters":
        raw.filters = { values: allNumbers(stmt.values), ...span };
        break;
      case "kernel_size":
        raw.kernelSize = { value: singleNumber(stmt.values), ...span };
        break;
      case "pool_size":
        raw.poolSize = { value: singleNumber(stmt.values), ...span };
        break;
      case "input_shape":
        raw.inputShape = { values: allNumbers(stmt.values), ...span };
        break;
      // RNN-specific
      case "sequence_length":
        raw.sequenceLength = { value: singleNumber(stmt.values), ...span };
        break;
      case "embedding_dim":
        raw.embeddingDim = { value: singleNumber(stmt.values), ...span };
        break;
      case "bidirectional":
        raw.bidirectional = { value: singleString(stmt.values) !== "false", ...span };
        break;
      case "recurrent_dropout":
        raw.recurrentDropout = { value: singleNumber(stmt.values), ...span };
        break;
      // Transfer learning
      case "pretrained":
        raw.pretrained = { value: singleString(stmt.values).toLowerCase(), ...span };
        break;
      case "freeze_layers":
        raw.freezeLayers = { value: singleNumber(stmt.values), ...span };
        break;
      // Export
      case "export_format":
        raw.exportFormats = { values: allStrings(stmt.values).map(s => s.toLowerCase()), ...span };
        break;
      // Multi-output
      case "targets":
        raw.targets = { values: allStrings(stmt.values), ...span };
        break;
      // Comparison (handled at CLI level, just parse it)
      case "compare":
        break;
      default:
        this.collector.warning(
          E.UNKNOWN_KEYWORD,
          `Unknown keyword '${stmt.keyword}'`,
          span,
          `Valid keywords: task, predict, inputs, dataset, architecture, loss, learn, optimizer, epochs, layers, filters, ...`,
        );
    }
  }

  private validateRequired(raw: RawConfig): boolean {
    const required: [string, unknown][] = [
      ["task", raw.task],
      ["predict", raw.predict],
      ["inputs", raw.inputs],
      ["dataset", raw.dataset],
    ];

    let valid = true;
    for (const [name, field] of required) {
      if (!field) {
        this.collector.error(
          E.MISSING_REQUIRED,
          `Missing required keyword '${name}'`,
          { line: 1, column: 1, length: 1 },
          name === "task"
            ? "Add 'task classification' or 'task regression' at the top of your program"
            : `Add '${name} <value>' to your program`,
        );
        valid = false;
      }
    }

    return valid;
  }

  private validateValues(raw: RawConfig): boolean {
    let valid = true;

    if (raw.task && !VALID_TASKS.has(raw.task.value)) {
      this.collector.error(E.INVALID_TASK, `Invalid task '${raw.task.value}'`, raw.task, `Must be one of: ${[...VALID_TASKS].join(", ")}`);
      valid = false;
    }

    if (raw.learn && !VALID_LEARN.has(raw.learn.value)) {
      this.collector.error(E.INVALID_LEARN, `Invalid learn mode '${raw.learn.value}'`, raw.learn, `Must be one of: ${[...VALID_LEARN].join(", ")}`);
      valid = false;
    }

    if (raw.architecture && !VALID_ARCHITECTURES.has(raw.architecture.value)) {
      this.collector.error(E.INVALID_LEARN, `Invalid architecture '${raw.architecture.value}'`, raw.architecture, `Must be one of: ${[...VALID_ARCHITECTURES].join(", ")}`);
      valid = false;
    }

    if (raw.optimizer && !VALID_OPTIMIZERS.has(raw.optimizer.value)) {
      this.collector.error(E.INVALID_OPTIMIZER, `Invalid optimizer '${raw.optimizer.value}'`, raw.optimizer, `Must be one of: ${[...VALID_OPTIMIZERS].join(", ")}`);
      valid = false;
    }

    if (raw.activation && !VALID_ACTIVATIONS.has(raw.activation.value)) {
      this.collector.error(E.INVALID_ACTIVATION, `Invalid activation '${raw.activation.value}'`, raw.activation, `Must be one of: ${[...VALID_ACTIVATIONS].join(", ")}`);
      valid = false;
    }

    if (raw.loss && !LOSS_MAP[raw.loss.value]) {
      this.collector.error(E.INVALID_LOSS, `Invalid loss function '${raw.loss.value}'`, raw.loss, `Must be one of: ${Object.keys(LOSS_MAP).join(", ")}`);
      valid = false;
    }

    if (raw.lrSchedule && !VALID_LR_SCHEDULES.has(raw.lrSchedule.value)) {
      this.collector.error(E.INVALID_ACTIVATION, `Invalid learning rate schedule '${raw.lrSchedule.value}'`, raw.lrSchedule, `Must be one of: ${[...VALID_LR_SCHEDULES].join(", ")}`);
      valid = false;
    }

    if (raw.epochs && (Number.isNaN(raw.epochs.value) || raw.epochs.value <= 0)) {
      this.collector.error(E.INVALID_EPOCHS, `Epochs must be a positive integer, got '${raw.epochs.value}'`, raw.epochs, "Example: epochs 50");
      valid = false;
    }

    if (raw.layers) {
      for (const n of raw.layers.values) {
        if (Number.isNaN(n) || n <= 0) {
          this.collector.error(E.NUMERIC_EXPECTED, `Layer size must be a positive integer`, raw.layers, "Example: layers 128 64 32");
          valid = false;
          break;
        }
      }
    }

    return valid;
  }

  private resolveArchType(raw: RawConfig): ArchitectureType {
    if (raw.architecture) {
      const v = raw.architecture.value;
      if (v === "lstm" || v === "gru" || v === "rnn") return "rnn";
      if (v === "resnet") return "resnet";
      if (v === "transformer") return "transformer";
      return v as ArchitectureType;
    }
    // Infer from pretrained model
    if (raw.pretrained) {
      const p = raw.pretrained.value;
      if (p.startsWith("resnet")) return "resnet";
      if (["mobilenet", "vgg16", "vgg19", "efficientnet", "inception", "densenet"].includes(p)) return "cnn";
    }
    // Infer from keywords
    if (raw.filters || raw.kernelSize || raw.poolSize) return "cnn";
    if (raw.sequenceLength || raw.embeddingDim || raw.bidirectional || raw.recurrentDropout) return "rnn";
    return "mlp";
  }

  private buildIR(raw: RawConfig): NeuralNetworkIR {
    const task = raw.task!.value as TaskType;
    const features = raw.inputs!.values;
    const inputSize = features.length;
    const learnMode = raw.learn?.value ?? "nonlinear";
    const activation = raw.activation?.value ?? "relu";
    const dropout = raw.dropout?.value ?? 0;
    const useBatchNorm = raw.batchNorm?.value ?? false;

    const outputSize = this.inferOutputSize(task, raw);
    const archType = this.resolveArchType(raw);

    let layers: LayerSpec[];
    let inputShape: number[] | undefined;

    if (archType === "cnn") {
      const result = this.buildCNNLayers(raw, inputSize, outputSize, task, activation, dropout, useBatchNorm);
      layers = result.layers;
      inputShape = result.inputShape;
    } else if (archType === "rnn") {
      const result = this.buildRNNLayers(raw, inputSize, outputSize, task, activation, dropout);
      layers = result.layers;
      inputShape = result.inputShape;
    } else if (archType === "autoencoder") {
      layers = this.buildAutoencoderLayers(raw, inputSize, outputSize, task, activation, dropout);
    } else if (archType === "resnet") {
      const result = this.buildResNetLayers(raw, inputSize, outputSize, task, activation, dropout);
      layers = result.layers;
      inputShape = result.inputShape;
    } else if (archType === "transformer") {
      const result = this.buildTransformerLayers(raw, inputSize, outputSize, task, activation, dropout);
      layers = result.layers;
      inputShape = result.inputShape;
    } else {
      const hiddenLayers = this.generateHiddenLayers(task, learnMode, inputSize, raw.layers?.values);
      layers = this.buildLayerSpecs(hiddenLayers, outputSize, task, activation, dropout, useBatchNorm);
    }

    const hiddenForCount = (archType === "mlp")
      ? this.generateHiddenLayers(task, learnMode, inputSize, raw.layers?.values)
      : [];
    const paramCount = this.countParameters(inputSize, outputSize, hiddenForCount, useBatchNorm);
    const loss = this.resolveLoss(task, raw.loss?.value);
    const optimizer = this.resolveOptimizer(raw.optimizer?.value, raw.learningRate?.value);

    const earlyStop: EarlyStopSpec | undefined = raw.earlyStop
      ? { patience: raw.earlyStop.value || 5, minDelta: 0.0001, monitor: task === "classification" ? "val_accuracy" : "val_loss" }
      : undefined;

    const lrSchedule: LRScheduleSpec = { type: (raw.lrSchedule?.value ?? "none") as LRScheduleType };
    const validationSplit = raw.validate?.value ?? 0.1;

    const pretrained = raw.pretrained?.value as PretrainedModel | undefined;
    const freezeL = raw.freezeLayers?.value;
    const exportFormats = raw.exportFormats?.values?.filter(f => VALID_EXPORT_FORMATS.has(f)) as ExportFormat[] | undefined;
    const multiTargets = raw.targets?.values;

    return {
      task,
      target: raw.predict!.value,
      targets: multiTargets,
      features,
      dataset: raw.dataset!.value,
      architecture: {
        type: archType,
        inputSize,
        inputShape,
        outputSize,
        outputSizes: multiTargets ? multiTargets.map(() => outputSize) : undefined,
        layers,
        batchNorm: useBatchNorm,
        pretrained: pretrained && VALID_PRETRAINED.has(pretrained) ? pretrained : undefined,
        freezeLayers: freezeL,
      },
      training: {
        loss, optimizer,
        epochs: raw.epochs?.value ?? 40,
        batchSize: raw.batchSize?.value ?? 32,
        validationSplit, earlyStop, lrSchedule,
        tune: raw.tune?.value,
        ensemble: raw.ensemble?.value,
      },
      preprocessing: {
        normalize: (archType === "cnn" || archType === "resnet") ? false : (raw.normalize?.value ?? true),
        trainTestSplit: raw.split?.value ?? 0.8,
        seed: raw.seed?.value,
        crossValidation: raw.crossValidate?.value,
      },
      output: {
        exportPath: raw.exportPath?.value,
        exportFormats: exportFormats && exportFormats.length > 0 ? exportFormats : undefined,
      },
      data: {
        show: raw.show ? (raw.show.value === "data" ? true : parseInt(raw.show.value) || true) : undefined,
        describe: raw.describe?.value,
        sample: raw.sample?.value,
        augment: raw.augment?.value,
        select: raw.select?.values,
        filter: raw.filter?.value,
      },
      meta: { learnMode: archType === "mlp" ? learnMode : archType, parameterCount: paramCount },
    };
  }

  // ── CNN Architecture ────────────────────────────────────

  private buildCNNLayers(
    raw: RawConfig, inputSize: number, outputSize: number,
    task: TaskType, activation: string, dropout: number, batchNorm: boolean,
  ): { layers: LayerSpec[]; inputShape: number[] } {
    const filterList = raw.filters?.values ?? [32, 64];
    const kernelSize = raw.kernelSize?.value ?? 3;
    const poolSize = raw.poolSize?.value ?? 2;
    const inputShape = raw.inputShape?.values ?? this.inferImageShape(inputSize);

    const specs: LayerSpec[] = [];

    // Reshape flat input to image if needed
    if (!raw.inputShape) {
      specs.push({ kind: "reshape", targetShape: inputShape });
    }

    for (let i = 0; i < filterList.length; i++) {
      specs.push({
        kind: "conv2d", filters: filterList[i]!, kernelSize,
        strides: 1, padding: "same", activation,
      });
      if (batchNorm) specs.push({ kind: "batchnorm" });
      specs.push({ kind: "maxpool2d", poolSize });
      if (dropout > 0) specs.push({ kind: "dropout", rate: dropout });
    }

    specs.push({ kind: "flatten" });
    specs.push({ kind: "dense", units: 128, activation });
    if (dropout > 0) specs.push({ kind: "dropout", rate: dropout });

    const outputActivation = task === "classification" ? "softmax" : "linear";
    specs.push({ kind: "dense", units: outputSize, activation: outputActivation });

    return { layers: specs, inputShape };
  }

  private inferImageShape(inputSize: number): number[] {
    const side = Math.round(Math.sqrt(inputSize));
    if (side * side === inputSize) return [side, side, 1];
    return [inputSize, 1, 1];
  }

  // ── RNN Architecture ────────────────────────────────────

  private buildRNNLayers(
    raw: RawConfig, inputSize: number, outputSize: number,
    task: TaskType, activation: string, dropout: number,
  ): { layers: LayerSpec[]; inputShape: number[] } {
    const seqLen = raw.sequenceLength?.value ?? inputSize;
    const archName = raw.architecture?.value ?? "lstm";
    const rnnKind: "lstm" | "gru" = archName === "gru" ? "gru" : "lstm";
    const useBidir = raw.bidirectional?.value ?? false;
    const layerSizes = raw.layers?.values ?? [64];
    const inputShape = [seqLen, 1];

    const specs: LayerSpec[] = [];

    if (raw.embeddingDim) {
      specs.push({ kind: "embedding", inputDim: 10000, outputDim: raw.embeddingDim.value });
    }

    // Reshape flat features into sequence: [seqLen, features_per_step]
    const featPerStep = Math.max(1, Math.floor(inputSize / seqLen));
    if (!raw.embeddingDim) {
      specs.push({ kind: "reshape", targetShape: [seqLen, featPerStep] });
      inputShape[1] = featPerStep;
    }

    for (let i = 0; i < layerSizes.length; i++) {
      const units = layerSizes[i]!;
      const returnSequences = i < layerSizes.length - 1;
      const rnnLayer: LayerSpec = { kind: rnnKind, units, returnSequences };

      if (useBidir) {
        specs.push({ kind: "bidirectional", wrapped: rnnLayer as any });
      } else {
        specs.push(rnnLayer);
      }

      if (dropout > 0) specs.push({ kind: "dropout", rate: dropout });
    }

    specs.push({ kind: "dense", units: 32, activation });
    const outputActivation = task === "classification" ? "softmax" : "linear";
    specs.push({ kind: "dense", units: outputSize, activation: outputActivation });

    return { layers: specs, inputShape };
  }

  // ── Autoencoder Architecture ────────────────────────────

  private buildAutoencoderLayers(
    raw: RawConfig, inputSize: number, _outputSize: number,
    _task: TaskType, activation: string, dropout: number,
  ): LayerSpec[] {
    const layerSizes = raw.layers?.values ?? [64, 32];
    const specs: LayerSpec[] = [];

    // Encoder
    for (const units of layerSizes) {
      specs.push({ kind: "dense", units, activation });
      if (dropout > 0) specs.push({ kind: "dropout", rate: dropout });
    }

    // Decoder (mirror)
    for (let i = layerSizes.length - 2; i >= 0; i--) {
      specs.push({ kind: "dense", units: layerSizes[i]!, activation });
    }

    // Output: reconstruct input
    specs.push({ kind: "dense", units: inputSize, activation: "sigmoid" });

    return specs;
  }

  // ── MLP Architecture (existing) ─────────────────────────

  private inferOutputSize(task: TaskType, raw: RawConfig): number {
    if (task === "regression") return 1;
    const ds = (raw.dataset?.value ?? "").toLowerCase();
    if (ds.includes("iris")) return 3;
    if (ds.includes("wine")) return 3;
    if (ds.includes("titanic")) return 2;
    if (ds.includes("digits") || ds.includes("mnist")) return 10;
    if (ds.includes("cifar")) return 10;
    if (ds.includes("sequence")) return 2;
    return 3;
  }

  private generateHiddenLayers(
    task: TaskType, learnMode: string, inputSize: number, explicit?: number[],
  ): number[] {
    if (explicit && explicit.length > 0) return explicit;

    switch (learnMode) {
      case "linear":
        return [];
      case "deep":
        return task === "classification" ? [256, 128, 64, 32] : [128, 64, 64, 32];
      case "auto":
        return [128, 64];
      default:
        return task === "classification" ? [128, 64] : [64, 64];
    }
  }

  private buildLayerSpecs(
    hidden: number[], outputSize: number, task: TaskType,
    activation: string, dropout: number, batchNorm: boolean,
  ): LayerSpec[] {
    const specs: LayerSpec[] = [];

    for (const units of hidden) {
      specs.push({ kind: "dense", units, activation });
      if (batchNorm) specs.push({ kind: "batchnorm" });
      if (dropout > 0) specs.push({ kind: "dropout", rate: dropout });
    }

    const outputActivation = task === "classification" ? "softmax" : "linear";
    specs.push({ kind: "dense", units: outputSize, activation: outputActivation });

    return specs;
  }

  // ── ResNet Architecture ──────────────────────────────────

  private buildResNetLayers(
    raw: RawConfig, inputSize: number, outputSize: number,
    task: TaskType, activation: string, dropout: number,
  ): { layers: LayerSpec[]; inputShape: number[] } {
    const filterList = raw.filters?.values ?? [32, 64, 128];
    const kernelSize = raw.kernelSize?.value ?? 3;
    const inputShape = raw.inputShape?.values ?? this.inferImageShape(inputSize);

    const specs: LayerSpec[] = [];

    if (!raw.inputShape) {
      specs.push({ kind: "reshape", targetShape: inputShape });
    }

    // Initial conv
    specs.push({
      kind: "conv2d", filters: filterList[0] ?? 32, kernelSize,
      strides: 1, padding: "same", activation,
    });
    specs.push({ kind: "batchnorm" });

    // Residual blocks
    for (const filters of filterList) {
      specs.push({ kind: "residual", filters, kernelSize, strides: 1 });
      specs.push({ kind: "residual", filters, kernelSize, strides: 1 });
      if (dropout > 0) specs.push({ kind: "dropout", rate: dropout });
    }

    specs.push({ kind: "globalavgpool2d" });
    specs.push({ kind: "dense", units: 128, activation });
    if (dropout > 0) specs.push({ kind: "dropout", rate: dropout });

    const outputActivation = task === "classification" ? "softmax" : "linear";
    specs.push({ kind: "dense", units: outputSize, activation: outputActivation });

    return { layers: specs, inputShape };
  }

  // ── Transformer Architecture ────────────────────────────

  private buildTransformerLayers(
    raw: RawConfig, inputSize: number, outputSize: number,
    task: TaskType, activation: string, dropout: number,
  ): { layers: LayerSpec[]; inputShape: number[] } {
    const seqLen = raw.sequenceLength?.value ?? inputSize;
    const embDim = raw.embeddingDim?.value ?? 64;
    const numHeads = 4;
    const numBlocks = raw.layers?.values?.length ?? 2;
    const inputShape = [seqLen, 1];

    const specs: LayerSpec[] = [];

    const featPerStep = Math.max(1, Math.floor(inputSize / seqLen));
    specs.push({ kind: "reshape", targetShape: [seqLen, featPerStep] });
    inputShape[1] = featPerStep;

    // Project to embedding dim
    specs.push({ kind: "dense", units: embDim, activation: "linear" });

    // Transformer blocks
    for (let i = 0; i < numBlocks; i++) {
      specs.push({ kind: "multihead_attention", heads: numHeads, keyDim: Math.floor(embDim / numHeads) });
      specs.push({ kind: "layernorm" });
      if (dropout > 0) specs.push({ kind: "dropout", rate: dropout });
      specs.push({ kind: "dense", units: embDim * 2, activation });
      specs.push({ kind: "dense", units: embDim, activation: "linear" });
      specs.push({ kind: "layernorm" });
      if (dropout > 0) specs.push({ kind: "dropout", rate: dropout });
    }

    // Pool across sequence
    specs.push({ kind: "flatten" });
    specs.push({ kind: "dense", units: 128, activation });
    if (dropout > 0) specs.push({ kind: "dropout", rate: dropout });

    const outputActivation = task === "classification" ? "softmax" : "linear";
    specs.push({ kind: "dense", units: outputSize, activation: outputActivation });

    return { layers: specs, inputShape };
  }

  private countParameters(
    inputSize: number, outputSize: number, hidden: number[], batchNorm: boolean,
  ): number {
    let params = 0;
    let prev = inputSize;
    for (const units of hidden) {
      params += prev * units + units;
      if (batchNorm) params += units * 4;
      prev = units;
    }
    params += prev * outputSize + outputSize;
    return params;
  }

  private resolveLoss(task: TaskType, explicit?: string): LossFunction {
    if (explicit && LOSS_MAP[explicit]) return LOSS_MAP[explicit]!;
    return task === "classification" ? "categoricalCrossentropy" : "meanSquaredError";
  }

  private resolveOptimizer(name?: string, lr?: number): OptimizerSpec {
    const optimizerName = (name ?? "adam") as OptimizerSpec["name"];
    return { name: optimizerName, learningRate: lr ?? 0.001 };
  }
}
