import type { NeuralNetworkIR, LayerSpec } from "./ir.js";

class CodeWriter {
  private lines: string[] = [];
  private depth = 0;

  indent(): this { this.depth++; return this; }
  dedent(): this { this.depth--; return this; }

  write(line: string): this {
    this.lines.push("  ".repeat(this.depth) + line);
    return this;
  }

  blank(): this {
    this.lines.push("");
    return this;
  }

  toString(): string {
    return this.lines.join("\n") + "\n";
  }
}

// ─── TensorFlow.js Target ─────────────────────────────────────────────

export function emitTensorFlow(ir: NeuralNetworkIR): string {
  const w = new CodeWriter();

  w.write(`import * as tf from "@tensorflow/tfjs";`);
  w.write(`import * as fs from "node:fs";`);
  w.blank();
  emitHeader(w, ir, "//");
  w.blank();

  emitTfModel(w, ir);
  w.blank();
  emitTfCompile(w, ir);
  w.blank();
  emitTfDataLoader(w, ir);
  w.blank();
  emitTfTrainFn(w, ir);
  w.blank();
  w.write(`main().catch(console.error);`);

  return w.toString();
}

function emitHeader(w: CodeWriter, ir: NeuralNetworkIR, prefix: string): void {
  const archType = (ir.architecture.type ?? "mlp").toUpperCase();
  w.write(`${prefix} ────────────────────────────────────────────────────`);
  w.write(`${prefix}  NeuroLang Compiled Model`);
  w.write(`${prefix}  Task:        ${ir.task}`);
  w.write(`${prefix}  Architecture:${archType}`);
  w.write(`${prefix}  Target:      ${ir.target}`);
  w.write(`${prefix}  Features:    ${ir.features.join(", ")}`);
  w.write(`${prefix}  Dataset:     ${ir.dataset}`);
  w.write(`${prefix}  Learn mode:  ${ir.meta.learnMode}`);
  if (ir.architecture.inputShape) w.write(`${prefix}  Input shape: [${ir.architecture.inputShape.join(", ")}]`);
  if (ir.architecture.batchNorm) w.write(`${prefix}  Batch norm:  enabled`);
  if (ir.training.earlyStop) w.write(`${prefix}  Early stop:  patience=${ir.training.earlyStop.patience}`);
  if (ir.training.lrSchedule.type !== "none") w.write(`${prefix}  LR schedule: ${ir.training.lrSchedule.type}`);
  w.write(`${prefix} ────────────────────────────────────────────────────`);
}

function emitTfModel(w: CodeWriter, ir: NeuralNetworkIR): void {
  w.write(`const model = tf.sequential();`);
  w.blank();

  let isFirst = true;
  const inputShape = ir.architecture.inputShape ?? [ir.architecture.inputSize];

  for (const layer of ir.architecture.layers) {
    const firstArg = isFirst ? `inputShape: [${inputShape.join(", ")}], ` : "";

    switch (layer.kind) {
      case "dense": {
        const parts: string[] = [];
        if (isFirst) { parts.push(`inputShape: [${ir.architecture.inputSize}]`); isFirst = false; }
        parts.push(`units: ${layer.units}`, `activation: "${layer.activation}"`);
        w.write(`model.add(tf.layers.dense({ ${parts.join(", ")} }));`);
        break;
      }
      case "batchnorm":
        w.write(`model.add(tf.layers.batchNormalization());`);
        break;
      case "dropout":
        w.write(`model.add(tf.layers.dropout({ rate: ${layer.rate} }));`);
        break;
      case "conv2d":
        w.write(`model.add(tf.layers.conv2d({ ${firstArg}filters: ${layer.filters}, kernelSize: ${layer.kernelSize}, strides: ${layer.strides}, padding: "${layer.padding}", activation: "${layer.activation}" }));`);
        isFirst = false;
        break;
      case "maxpool2d":
        w.write(`model.add(tf.layers.maxPooling2d({ poolSize: [${layer.poolSize}, ${layer.poolSize}] }));`);
        break;
      case "flatten":
        w.write(`model.add(tf.layers.flatten());`);
        break;
      case "globalavgpool2d":
        w.write(`model.add(tf.layers.globalAveragePooling2d());`);
        break;
      case "reshape":
        w.write(`model.add(tf.layers.reshape({ ${firstArg}targetShape: [${layer.targetShape.join(", ")}] }));`);
        isFirst = false;
        break;
      case "lstm":
        w.write(`model.add(tf.layers.lstm({ ${firstArg}units: ${layer.units}, returnSequences: ${layer.returnSequences} }));`);
        isFirst = false;
        break;
      case "gru":
        w.write(`model.add(tf.layers.gru({ ${firstArg}units: ${layer.units}, returnSequences: ${layer.returnSequences} }));`);
        isFirst = false;
        break;
      case "embedding":
        w.write(`model.add(tf.layers.embedding({ ${firstArg}inputDim: ${layer.inputDim}, outputDim: ${layer.outputDim} }));`);
        isFirst = false;
        break;
      case "bidirectional": {
        const inner = layer.wrapped.kind === "gru"
          ? `tf.layers.gru({ units: ${layer.wrapped.units}, returnSequences: ${layer.wrapped.returnSequences} })`
          : `tf.layers.lstm({ units: ${layer.wrapped.units}, returnSequences: ${layer.wrapped.returnSequences} })`;
        w.write(`model.add(tf.layers.bidirectional({ ${firstArg}layer: ${inner} }));`);
        isFirst = false;
        break;
      }
      case "residual":
        w.write(`// Residual block: ${layer.filters} filters (skip connection handled via functional API in production)`);
        w.write(`model.add(tf.layers.conv2d({ ${firstArg}filters: ${layer.filters}, kernelSize: ${layer.kernelSize}, strides: ${layer.strides}, padding: "same", activation: "relu" }));`);
        w.write(`model.add(tf.layers.batchNormalization());`);
        w.write(`model.add(tf.layers.conv2d({ filters: ${layer.filters}, kernelSize: ${layer.kernelSize}, strides: 1, padding: "same", activation: "linear" }));`);
        w.write(`model.add(tf.layers.batchNormalization());`);
        isFirst = false;
        break;
      case "multihead_attention":
        w.write(`// Multi-head attention: ${layer.heads} heads, keyDim=${layer.keyDim}`);
        w.write(`// Note: TF.js sequential API does not support attention natively; use functional API`);
        break;
      case "layernorm":
        w.write(`// Layer normalization`);
        break;
      case "add":
        w.write(`// Skip connection (add)`);
        break;
    }
  }
}

function emitTfCompile(w: CodeWriter, ir: NeuralNetworkIR): void {
  const { optimizer, loss } = ir.training;
  const optimizerExpr = `tf.train.${optimizer.name}(${optimizer.learningRate})`;
  const metrics = ir.task === "classification" ? `["accuracy"]` : `["mse"]`;

  w.write(`model.compile({`);
  w.indent();
  w.write(`optimizer: ${optimizerExpr},`);
  w.write(`loss: "${loss}",`);
  w.write(`metrics: ${metrics},`);
  w.dedent();
  w.write(`});`);
}

function emitTfDataLoader(w: CodeWriter, ir: NeuralNetworkIR): void {
  w.write(`function loadCSV(path: string) {`);
  w.indent();
  w.write(`const text = fs.readFileSync(path, "utf-8");`);
  w.write(`const lines = text.trim().split("\\n");`);
  w.write(`const headers = lines[0].split(",").map((h: string) => h.trim());`);
  w.write(`const data = lines.slice(1).map((l: string) => l.split(",").map(Number));`);
  w.write(`return { headers, data };`);
  w.dedent();
  w.write(`}`);
  w.blank();

  w.write(`function prepareData() {`);
  w.indent();
  w.write(`const { headers, data } = loadCSV("${ir.dataset}");`);
  w.write(`const featureCols = ${JSON.stringify(ir.features)};`);
  w.write(`const targetCol = "${ir.target}";`);
  w.blank();
  w.write(`const featureIdx = featureCols.map((f: string) => headers.indexOf(f));`);
  w.write(`const targetIdx = headers.indexOf(targetCol);`);
  w.blank();
  w.write(`const X = data.map((row: number[]) => featureIdx.map((i: number) => row[i]));`);
  w.write(`const y = data.map((row: number[]) => row[targetIdx]);`);
  w.blank();

  if (ir.preprocessing.normalize) {
    w.write(`// Normalize features`);
    w.write(`const means = featureIdx.map((_: number, c: number) => X.reduce((s: number, r: number[]) => s + r[c], 0) / X.length);`);
    w.write(`const stds = featureIdx.map((_: number, c: number) => {`);
    w.indent();
    w.write(`const m = means[c];`);
    w.write(`return Math.sqrt(X.reduce((s: number, r: number[]) => s + (r[c] - m) ** 2, 0) / X.length) || 1;`);
    w.dedent();
    w.write(`});`);
    w.write(`const Xn = X.map((r: number[]) => r.map((v: number, c: number) => (v - means[c]) / stds[c]));`);
    w.blank();
  }

  const xRef = ir.preprocessing.normalize ? "Xn" : "X";
  w.write(`const split = Math.floor(${xRef}.length * ${ir.preprocessing.trainTestSplit});`);
  w.write(`const xTrain = tf.tensor2d(${xRef}.slice(0, split));`);
  w.write(`const xTest = tf.tensor2d(${xRef}.slice(split));`);
  w.blank();

  if (ir.task === "classification") {
    w.write(`const yTrain = tf.oneHot(tf.tensor1d(y.slice(0, split), "int32"), ${ir.architecture.outputSize}).cast("float32");`);
    w.write(`const yTest = tf.oneHot(tf.tensor1d(y.slice(split), "int32"), ${ir.architecture.outputSize}).cast("float32");`);
  } else {
    w.write(`const yTrain = tf.tensor2d(y.slice(0, split).map((v: number) => [v]));`);
    w.write(`const yTest = tf.tensor2d(y.slice(split).map((v: number) => [v]));`);
  }
  w.blank();
  w.write(`return { xTrain, yTrain, xTest, yTest };`);
  w.dedent();
  w.write(`}`);
}

function emitTfTrainFn(w: CodeWriter, ir: NeuralNetworkIR): void {
  w.write(`async function main() {`);
  w.indent();

  if (ir.preprocessing.seed !== undefined) {
    w.write(`tf.setBackend("cpu");`);
    w.write(`// Seed is advisory — TF.js does not guarantee deterministic results`);
  }

  w.write(`const { xTrain, yTrain, xTest, yTest } = prepareData();`);
  w.blank();
  w.write(`console.log("Architecture:");`);
  w.write(`model.summary();`);
  w.blank();

  // Callbacks
  w.write(`const callbacks: tf.CustomCallbackArgs[] = [{`);
  w.indent();
  w.write(`onEpochEnd: (epoch, logs) => {`);
  w.indent();
  w.write(`console.log(\`  Epoch \${epoch + 1}/${ir.training.epochs}: loss=\${logs?.loss?.toFixed(4)}\`);`);
  w.dedent();
  w.write(`},`);
  w.dedent();
  w.write(`}];`);
  w.blank();

  if (ir.training.earlyStop) {
    w.write(`callbacks.push(tf.callbacks.earlyStopping({`);
    w.indent();
    w.write(`monitor: "${ir.training.earlyStop.monitor}",`);
    w.write(`patience: ${ir.training.earlyStop.patience},`);
    w.write(`minDelta: ${ir.training.earlyStop.minDelta},`);
    w.dedent();
    w.write(`}));`);
    w.blank();
  }

  w.write(`await model.fit(xTrain, yTrain, {`);
  w.indent();
  w.write(`epochs: ${ir.training.epochs},`);
  w.write(`batchSize: ${ir.training.batchSize},`);
  w.write(`validationSplit: ${ir.training.validationSplit},`);
  w.write(`shuffle: true,`);
  w.write(`callbacks,`);
  w.dedent();
  w.write(`});`);
  w.blank();

  // Evaluate
  w.write(`const evalResult = model.evaluate(xTest, yTest) as tf.Scalar[];`);
  if (ir.task === "classification") {
    w.write(`console.log("\\nTest accuracy:", evalResult[1].dataSync()[0].toFixed(4));`);
  } else {
    w.write(`console.log("\\nTest MSE:", evalResult[1].dataSync()[0].toFixed(6));`);
  }

  // Export
  if (ir.output.exportPath) {
    w.blank();
    w.write(`await model.save("file://${ir.output.exportPath}");`);
    w.write(`console.log("Model exported to ${ir.output.exportPath}");`);
  }

  // Cleanup
  w.blank();
  w.write(`xTrain.dispose(); yTrain.dispose();`);
  w.write(`xTest.dispose(); yTest.dispose();`);
  w.write(`model.dispose();`);

  w.dedent();
  w.write(`}`);
}

// ─── PyTorch Target ────────────────────────────────────────────────────

export function emitPyTorch(ir: NeuralNetworkIR): string {
  const w = new CodeWriter();

  w.write(`import torch`);
  w.write(`import torch.nn as nn`);
  w.write(`import torch.optim as optim`);
  w.write(`import pandas as pd`);
  w.write(`from sklearn.model_selection import train_test_split`);
  w.write(`from sklearn.preprocessing import StandardScaler`);
  w.blank();
  emitHeader(w, ir, "#");
  w.blank();

  if (ir.preprocessing.seed !== undefined) {
    w.write(`torch.manual_seed(${ir.preprocessing.seed})`);
    w.blank();
  }

  if (ir.architecture.pretrained) {
    emitPyPretrained(w, ir);
  } else {
    emitPyModel(w, ir);
  }
  w.blank();
  emitPyDataLoader(w, ir);
  w.blank();
  emitPyTrain(w, ir);

  if (ir.output.exportFormats && ir.output.exportFormats.length > 0) {
    w.blank();
    emitPyExport(w, ir);
  }

  return w.toString();
}

const PYTORCH_ACTIVATION: Record<string, string> = {
  relu: "nn.ReLU()",
  sigmoid: "nn.Sigmoid()",
  tanh: "nn.Tanh()",
  selu: "nn.SELU()",
  elu: "nn.ELU()",
  gelu: "nn.GELU()",
  leaky_relu: "nn.LeakyReLU()",
  softmax: "nn.Softmax(dim=1)",
  linear: "",
};

function emitPyModel(w: CodeWriter, ir: NeuralNetworkIR): void {
  const archType = ir.architecture.type ?? "mlp";

  w.write(`class Model(nn.Module):`);
  w.indent();
  w.write(`def __init__(self):`);
  w.indent();
  w.write(`super().__init__()`);

  if (archType === "cnn") {
    emitPyModelCNN(w, ir);
  } else if (archType === "rnn") {
    emitPyModelRNN(w, ir);
  } else if (archType === "resnet") {
    emitPyModelResNet(w, ir);
  } else if (archType === "transformer") {
    emitPyModelTransformer(w, ir);
  } else {
    emitPyModelMLP(w, ir);
  }

  w.dedent();
  w.blank();

  w.write(`def forward(self, x):`);
  w.indent();
  if (archType === "cnn" || archType === "resnet") {
    w.write(`x = self.features(x)`);
    w.write(`x = x.view(x.size(0), -1)`);
    w.write(`return self.classifier(x)`);
  } else if (archType === "rnn") {
    w.write(`x, _ = self.rnn(x)`);
    w.write(`x = x[:, -1, :]`);
    w.write(`return self.classifier(x)`);
  } else if (archType === "transformer") {
    w.write(`return self.forward_transformer(x)`);
  } else {
    w.write(`return self.network(x)`);
  }
  w.dedent();
  w.dedent();
  w.blank();

  w.write(`model = Model()`);
  w.write(`print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")`);
}

function emitPyModelMLP(w: CodeWriter, ir: NeuralNetworkIR): void {
  const seqParts: string[] = [];
  let prevSize = ir.architecture.inputSize;

  for (const layer of ir.architecture.layers) {
    if (layer.kind === "dense") {
      seqParts.push(`nn.Linear(${prevSize}, ${layer.units})`);
      const act = PYTORCH_ACTIVATION[layer.activation] ?? "";
      if (act) seqParts.push(act);
      prevSize = layer.units;
    } else if (layer.kind === "batchnorm") {
      seqParts.push(`nn.BatchNorm1d(${prevSize})`);
    } else if (layer.kind === "dropout") {
      seqParts.push(`nn.Dropout(${layer.rate})`);
    }
  }

  w.write(`self.network = nn.Sequential(`);
  w.indent();
  for (const part of seqParts) {
    w.write(`${part},`);
  }
  w.dedent();
  w.write(`)`);
}

function emitPyModelCNN(w: CodeWriter, ir: NeuralNetworkIR): void {
  const featureParts: string[] = [];
  const classifierParts: string[] = [];
  let inChannels = 1;
  let afterFlatten = false;
  let prevDense = 0;

  for (const layer of ir.architecture.layers) {
    if (layer.kind === "reshape") continue;
    if (layer.kind === "flatten") { afterFlatten = true; continue; }

    if (!afterFlatten) {
      if (layer.kind === "conv2d") {
        featureParts.push(`nn.Conv2d(${inChannels}, ${layer.filters}, kernel_size=${layer.kernelSize}, stride=${layer.strides}, padding=${{ same: 1, valid: 0 }[layer.padding]})`);
        const act = PYTORCH_ACTIVATION[layer.activation] ?? "";
        if (act) featureParts.push(act);
        inChannels = layer.filters;
      } else if (layer.kind === "maxpool2d") {
        featureParts.push(`nn.MaxPool2d(${layer.poolSize})`);
      } else if (layer.kind === "batchnorm") {
        featureParts.push(`nn.BatchNorm2d(${inChannels})`);
      } else if (layer.kind === "dropout") {
        featureParts.push(`nn.Dropout2d(${layer.rate})`);
      }
    } else {
      if (layer.kind === "dense") {
        if (prevDense === 0) {
          classifierParts.push(`nn.LazyLinear(${layer.units})`);
        } else {
          classifierParts.push(`nn.Linear(${prevDense}, ${layer.units})`);
        }
        const act = PYTORCH_ACTIVATION[layer.activation] ?? "";
        if (act) classifierParts.push(act);
        prevDense = layer.units;
      } else if (layer.kind === "dropout") {
        classifierParts.push(`nn.Dropout(${layer.rate})`);
      }
    }
  }

  w.write(`self.features = nn.Sequential(`);
  w.indent();
  for (const p of featureParts) w.write(`${p},`);
  w.dedent();
  w.write(`)`);
  w.write(`self.classifier = nn.Sequential(`);
  w.indent();
  for (const p of classifierParts) w.write(`${p},`);
  w.dedent();
  w.write(`)`);
}

function emitPyModelRNN(w: CodeWriter, ir: NeuralNetworkIR): void {
  const inputShape = ir.architecture.inputShape ?? [ir.architecture.inputSize, 1];
  const classifierParts: string[] = [];
  let rnnOutputSize = 0;
  let prevDense = 0;

  for (const layer of ir.architecture.layers) {
    if (layer.kind === "reshape") continue;
    if (layer.kind === "lstm") {
      w.write(`self.rnn = nn.LSTM(input_size=${inputShape[1]}, hidden_size=${layer.units}, batch_first=True)`);
      rnnOutputSize = layer.units;
    } else if (layer.kind === "gru") {
      w.write(`self.rnn = nn.GRU(input_size=${inputShape[1]}, hidden_size=${layer.units}, batch_first=True)`);
      rnnOutputSize = layer.units;
    } else if (layer.kind === "bidirectional") {
      const inner = layer.wrapped;
      const rnnType = inner.kind === "gru" ? "GRU" : "LSTM";
      w.write(`self.rnn = nn.${rnnType}(input_size=${inputShape[1]}, hidden_size=${inner.units}, batch_first=True, bidirectional=True)`);
      rnnOutputSize = inner.units * 2;
    } else if (layer.kind === "dense") {
      const inSize = prevDense || rnnOutputSize;
      classifierParts.push(`nn.Linear(${inSize}, ${layer.units})`);
      const act = PYTORCH_ACTIVATION[layer.activation] ?? "";
      if (act) classifierParts.push(act);
      prevDense = layer.units;
    } else if (layer.kind === "dropout") {
      classifierParts.push(`nn.Dropout(${layer.rate})`);
    }
  }

  w.write(`self.classifier = nn.Sequential(`);
  w.indent();
  for (const p of classifierParts) w.write(`${p},`);
  w.dedent();
  w.write(`)`);
}

function emitPyModelResNet(w: CodeWriter, ir: NeuralNetworkIR): void {
  const residuals = ir.architecture.layers.filter(l => l.kind === "residual");
  const denses = ir.architecture.layers.filter(l => l.kind === "dense");

  w.blank();
  w.write(`class ResidualBlock(nn.Module):`);
  w.indent();
  w.write(`def __init__(self, in_ch, out_ch, stride=1):`);
  w.indent();
  w.write(`super().__init__()`);
  w.write(`self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1)`);
  w.write(`self.bn1 = nn.BatchNorm2d(out_ch)`);
  w.write(`self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)`);
  w.write(`self.bn2 = nn.BatchNorm2d(out_ch)`);
  w.write(`self.shortcut = nn.Sequential()`);
  w.write(`if stride != 1 or in_ch != out_ch:`);
  w.indent();
  w.write(`self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride), nn.BatchNorm2d(out_ch))`);
  w.dedent();
  w.dedent();
  w.blank();
  w.write(`def forward(self, x):`);
  w.indent();
  w.write(`out = torch.relu(self.bn1(self.conv1(x)))`);
  w.write(`out = self.bn2(self.conv2(out))`);
  w.write(`return torch.relu(out + self.shortcut(x))`);
  w.dedent();
  w.dedent();
  w.blank();

  w.write(`# Build residual blocks`);
  let prevCh = 1;
  const blocks: string[] = [];
  for (const layer of ir.architecture.layers) {
    if (layer.kind === "conv2d") {
      blocks.push(`nn.Conv2d(${prevCh}, ${layer.filters}, ${layer.kernelSize}, 1, 1), nn.BatchNorm2d(${layer.filters}), nn.ReLU()`);
      prevCh = layer.filters;
    } else if (layer.kind === "residual") {
      blocks.push(`ResidualBlock(${prevCh}, ${layer.filters})`);
      prevCh = layer.filters;
    }
  }

  w.write(`self.features = nn.Sequential(`);
  w.indent();
  for (const b of blocks) w.write(`${b},`);
  w.write(`nn.AdaptiveAvgPool2d(1),`);
  w.dedent();
  w.write(`)`);

  const classifierParts: string[] = [];
  let prev = prevCh;
  for (const layer of denses) {
    if (layer.kind === "dense") {
      classifierParts.push(`nn.Linear(${prev}, ${layer.units})`);
      const act = PYTORCH_ACTIVATION[layer.activation] ?? "";
      if (act) classifierParts.push(act);
      prev = layer.units;
    }
  }
  w.write(`self.classifier = nn.Sequential(`);
  w.indent();
  for (const p of classifierParts) w.write(`${p},`);
  w.dedent();
  w.write(`)`);
}

function emitPyModelTransformer(w: CodeWriter, ir: NeuralNetworkIR): void {
  const inputShape = ir.architecture.inputShape ?? [ir.architecture.inputSize, 1];

  w.write(`self.embed = nn.Linear(${inputShape[1]}, 64)`);
  w.write(`encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)`);
  w.write(`self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)`);

  const denses = ir.architecture.layers.filter(l => l.kind === "dense");
  const lastDenses = denses.slice(-2);
  const classifierParts: string[] = [`nn.Linear(${inputShape[0]! * 64}, ${lastDenses[0]?.kind === "dense" ? lastDenses[0].units : 128})`];
  classifierParts.push("nn.ReLU()");
  if (lastDenses.length > 1 && lastDenses[1]?.kind === "dense") {
    classifierParts.push(`nn.Linear(${lastDenses[0]?.kind === "dense" ? lastDenses[0].units : 128}, ${lastDenses[1].units})`);
  }

  w.write(`self.classifier = nn.Sequential(`);
  w.indent();
  for (const p of classifierParts) w.write(`${p},`);
  w.dedent();
  w.write(`)`);

  w.dedent();
  w.blank();
  w.write(`def forward_transformer(self, x):`);
  w.indent();
  w.write(`x = x.view(x.size(0), ${inputShape[0]}, ${inputShape[1]})`);
  w.write(`x = self.embed(x)`);
  w.write(`x = self.transformer(x)`);
  w.write(`x = x.reshape(x.size(0), -1)`);
  w.write(`return self.classifier(x)`);
}

const PRETRAINED_PYTORCH: Record<string, { import: string; model: string; inputSize: number }> = {
  mobilenet: { import: "from torchvision.models import mobilenet_v2, MobileNet_V2_Weights", model: "mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)", inputSize: 1280 },
  resnet50: { import: "from torchvision.models import resnet50, ResNet50_Weights", model: "resnet50(weights=ResNet50_Weights.DEFAULT)", inputSize: 2048 },
  resnet101: { import: "from torchvision.models import resnet101, ResNet101_Weights", model: "resnet101(weights=ResNet101_Weights.DEFAULT)", inputSize: 2048 },
  vgg16: { import: "from torchvision.models import vgg16, VGG16_Weights", model: "vgg16(weights=VGG16_Weights.DEFAULT)", inputSize: 4096 },
  vgg19: { import: "from torchvision.models import vgg19, VGG19_Weights", model: "vgg19(weights=VGG19_Weights.DEFAULT)", inputSize: 4096 },
  efficientnet: { import: "from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights", model: "efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)", inputSize: 1280 },
  inception: { import: "from torchvision.models import inception_v3, Inception_V3_Weights", model: "inception_v3(weights=Inception_V3_Weights.DEFAULT)", inputSize: 2048 },
  densenet: { import: "from torchvision.models import densenet121, DenseNet121_Weights", model: "densenet121(weights=DenseNet121_Weights.DEFAULT)", inputSize: 1024 },
};

function emitPyPretrained(w: CodeWriter, ir: NeuralNetworkIR): void {
  const pre = ir.architecture.pretrained!;
  const cfg = PRETRAINED_PYTORCH[pre];
  if (!cfg) { emitPyModel(w, ir); return; }

  w.write(cfg.import);
  w.write(`from torchvision import transforms`);
  w.blank();

  w.write(`class Model(nn.Module):`);
  w.indent();
  w.write(`def __init__(self, num_classes=${ir.architecture.outputSize}):`);
  w.indent();
  w.write(`super().__init__()`);
  w.write(`self.backbone = ${cfg.model}`);

  if (ir.architecture.freezeLayers) {
    w.write(`# Freeze first ${ir.architecture.freezeLayers} layers`);
    w.write(`for i, param in enumerate(self.backbone.parameters()):`);
    w.indent();
    w.write(`if i < ${ir.architecture.freezeLayers}:`);
    w.indent();
    w.write(`param.requires_grad = False`);
    w.dedent();
    w.dedent();
  }

  w.write(`self.backbone.fc = nn.Identity()`);
  w.write(`self.classifier = nn.Sequential(`);
  w.indent();
  w.write(`nn.Linear(${cfg.inputSize}, 256),`);
  w.write(`nn.ReLU(),`);
  w.write(`nn.Dropout(0.3),`);
  w.write(`nn.Linear(256, num_classes),`);
  w.dedent();
  w.write(`)`);
  w.dedent();
  w.blank();
  w.write(`def forward(self, x):`);
  w.indent();
  w.write(`features = self.backbone(x)`);
  w.write(`return self.classifier(features)`);
  w.dedent();
  w.dedent();
  w.blank();

  w.write(`model = Model()`);
  w.write(`trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)`);
  w.write(`total = sum(p.numel() for p in model.parameters())`);
  w.write(`print(f"Parameters: {trainable:,} trainable / {total:,} total")`);
}

function emitPyExport(w: CodeWriter, ir: NeuralNetworkIR): void {
  w.write(`# ── Model Export ──`);
  const formats = ir.output.exportFormats ?? [];

  if (formats.includes("torchscript")) {
    w.write(`scripted = torch.jit.script(model)`);
    w.write(`scripted.save("model_scripted.pt")`);
    w.write(`print("Exported: model_scripted.pt (TorchScript)")`);
  }
  if (formats.includes("onnx")) {
    w.write(`dummy = torch.randn(1, ${ir.architecture.inputSize})`);
    w.write(`torch.onnx.export(model, dummy, "model.onnx", input_names=["input"], output_names=["output"])`);
    w.write(`print("Exported: model.onnx (ONNX)")`);
  }
  if (formats.includes("savedmodel")) {
    w.write(`torch.save(model.state_dict(), "model_weights.pth")`);
    w.write(`print("Exported: model_weights.pth (PyTorch state dict)")`);
  }
  if (formats.includes("coreml")) {
    w.write(`import coremltools as ct`);
    w.write(`traced = torch.jit.trace(model, torch.randn(1, ${ir.architecture.inputSize}))`);
    w.write(`mlmodel = ct.convert(traced, inputs=[ct.TensorType(shape=(1, ${ir.architecture.inputSize}))])`);
    w.write(`mlmodel.save("model.mlpackage")`);
    w.write(`print("Exported: model.mlpackage (CoreML)")`);
  }
  if (formats.includes("tflite")) {
    w.write(`# TFLite export requires conversion via ONNX → TFLite pipeline`);
    w.write(`# pip install onnx2tf`);
    w.write(`# onnx2tf -i model.onnx -o model_tflite`);
  }
}

function emitPyDataLoader(w: CodeWriter, ir: NeuralNetworkIR): void {
  w.write(`# Load and preprocess dataset`);
  w.write(`df = pd.read_csv("${ir.dataset}")`);
  w.write(`feature_cols = ${JSON.stringify(ir.features)}`);
  w.write(`target_col = "${ir.target}"`);
  w.blank();
  w.write(`X = df[feature_cols].values.astype("float32")`);
  w.write(`y = df[target_col].values`);
  w.blank();

  if (ir.preprocessing.normalize) {
    w.write(`scaler = StandardScaler()`);
    w.write(`X = scaler.fit_transform(X)`);
    w.blank();
  }

  const seedArg = ir.preprocessing.seed !== undefined ? `, random_state=${ir.preprocessing.seed}` : "";
  w.write(`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=${(1 - ir.preprocessing.trainTestSplit).toFixed(2)}${seedArg})`);
  w.blank();
  w.write(`X_train = torch.tensor(X_train, dtype=torch.float32)`);
  w.write(`X_test = torch.tensor(X_test, dtype=torch.float32)`);

  if (ir.task === "classification") {
    w.write(`y_train = torch.tensor(y_train, dtype=torch.long)`);
    w.write(`y_test = torch.tensor(y_test, dtype=torch.long)`);
  } else {
    w.write(`y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)`);
    w.write(`y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)`);
  }
}

function emitPyTrain(w: CodeWriter, ir: NeuralNetworkIR): void {
  const { optimizer, loss } = ir.training;

  const pyOptimizer: Record<string, string> = {
    adam: `optim.Adam(model.parameters(), lr=${optimizer.learningRate})`,
    adamw: `optim.AdamW(model.parameters(), lr=${optimizer.learningRate})`,
    sgd: `optim.SGD(model.parameters(), lr=${optimizer.learningRate})`,
    rmsprop: `optim.RMSprop(model.parameters(), lr=${optimizer.learningRate})`,
  };

  const pyLoss: Record<string, string> = {
    meanSquaredError: "nn.MSELoss()",
    meanAbsoluteError: "nn.L1Loss()",
    categoricalCrossentropy: "nn.CrossEntropyLoss()",
    binaryCrossentropy: "nn.BCELoss()",
    huberLoss: "nn.HuberLoss()",
  };

  w.blank();
  w.write(`optimizer = ${pyOptimizer[optimizer.name] ?? pyOptimizer.adam}`);
  w.write(`criterion = ${pyLoss[loss] ?? "nn.MSELoss()"}`);

  if (ir.training.lrSchedule.type !== "none") {
    w.blank();
    switch (ir.training.lrSchedule.type) {
      case "cosine":
        w.write(`scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=${ir.training.epochs})`);
        break;
      case "step":
        w.write(`scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=${Math.floor(ir.training.epochs / 3)}, gamma=0.1)`);
        break;
      case "exponential":
        w.write(`scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)`);
        break;
    }
  }

  w.blank();

  if (ir.training.earlyStop) {
    w.write(`best_loss = float("inf")`);
    w.write(`patience_counter = 0`);
    w.blank();
  }

  w.write(`for epoch in range(${ir.training.epochs}):`);
  w.indent();
  w.write(`model.train()`);
  w.write(`optimizer.zero_grad()`);
  w.write(`output = model(X_train)`);
  w.write(`loss = criterion(output, y_train)`);
  w.write(`loss.backward()`);
  w.write(`optimizer.step()`);

  if (ir.training.lrSchedule.type !== "none") {
    w.write(`scheduler.step()`);
  }

  w.blank();
  w.write(`if (epoch + 1) % 10 == 0:`);
  w.indent();
  w.write(`print(f"  Epoch {epoch + 1}/${ir.training.epochs}: loss={loss.item():.4f}")`);
  w.dedent();

  if (ir.training.earlyStop) {
    w.blank();
    w.write(`if loss.item() < best_loss - ${ir.training.earlyStop.minDelta}:`);
    w.indent();
    w.write(`best_loss = loss.item()`);
    w.write(`patience_counter = 0`);
    w.dedent();
    w.write(`else:`);
    w.indent();
    w.write(`patience_counter += 1`);
    w.write(`if patience_counter >= ${ir.training.earlyStop.patience}:`);
    w.indent();
    w.write(`print(f"Early stopping at epoch {epoch + 1}")`);
    w.write(`break`);
    w.dedent();
    w.dedent();
  }

  w.dedent();
  w.blank();

  // Evaluation
  w.write(`model.eval()`);
  w.write(`with torch.no_grad():`);
  w.indent();
  w.write(`test_output = model(X_test)`);

  if (ir.task === "classification") {
    w.write(`predicted = test_output.argmax(dim=1)`);
    w.write(`accuracy = (predicted == y_test).float().mean()`);
    w.write(`print(f"\\nTest accuracy: {accuracy:.4f}")`);
  } else {
    w.write(`test_loss = criterion(test_output, y_test)`);
    w.write(`print(f"\\nTest MSE: {test_loss.item():.6f}")`);
  }
  w.dedent();

  if (ir.output.exportPath) {
    w.blank();
    w.write(`torch.save(model.state_dict(), "${ir.output.exportPath}")`);
    w.write(`print(f"Model exported to ${ir.output.exportPath}")`);
  }

  w.blank();
  w.write(`print("Training complete.")`);
}

// ─── Keras Target ──────────────────────────────────────────────────────

export function emitKeras(ir: NeuralNetworkIR): string {
  const w = new CodeWriter();

  w.write(`import numpy as np`);
  w.write(`import pandas as pd`);
  w.write(`import tensorflow as tf`);
  w.write(`from tensorflow import keras`);
  w.write(`from tensorflow.keras import layers, callbacks`);
  w.write(`from sklearn.model_selection import train_test_split`);
  w.write(`from sklearn.preprocessing import StandardScaler`);
  w.blank();
  emitHeader(w, ir, "#");
  w.blank();

  if (ir.preprocessing.seed !== undefined) {
    w.write(`tf.random.set_seed(${ir.preprocessing.seed})`);
    w.write(`np.random.seed(${ir.preprocessing.seed})`);
    w.blank();
  }

  emitKerasDataLoader(w, ir);
  w.blank();
  if (ir.architecture.pretrained) {
    emitKerasPretrained(w, ir);
  } else {
    emitKerasModel(w, ir);
  }
  w.blank();
  emitKerasCompile(w, ir);
  w.blank();
  emitKerasTrain(w, ir);

  if (ir.output.exportFormats && ir.output.exportFormats.length > 0) {
    w.blank();
    emitKerasExport(w, ir);
  }

  return w.toString();
}

function emitKerasDataLoader(w: CodeWriter, ir: NeuralNetworkIR): void {
  w.write(`df = pd.read_csv("${ir.dataset}")`);
  w.write(`feature_cols = ${JSON.stringify(ir.features)}`);
  w.write(`target_col = "${ir.target}"`);
  w.blank();
  w.write(`X = df[feature_cols].values.astype("float32")`);
  w.write(`y = df[target_col].values`);
  w.blank();

  if (ir.preprocessing.normalize) {
    w.write(`scaler = StandardScaler()`);
    w.write(`X = scaler.fit_transform(X)`);
    w.blank();
  }

  const seedArg = ir.preprocessing.seed !== undefined ? `, random_state=${ir.preprocessing.seed}` : "";
  w.write(`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=${(1 - ir.preprocessing.trainTestSplit).toFixed(2)}${seedArg})`);
  w.blank();

  if (ir.task === "classification") {
    w.write(`y_train = keras.utils.to_categorical(y_train, ${ir.architecture.outputSize})`);
    w.write(`y_test = keras.utils.to_categorical(y_test, ${ir.architecture.outputSize})`);
  }
}

const KERAS_ACTIVATION: Record<string, string> = {
  relu: "relu",
  sigmoid: "sigmoid",
  tanh: "tanh",
  selu: "selu",
  elu: "elu",
  gelu: "gelu",
  leaky_relu: "LeakyReLU",
  softmax: "softmax",
  linear: "linear",
};

function emitKerasModel(w: CodeWriter, ir: NeuralNetworkIR): void {
  const inputShape = ir.architecture.inputShape ?? [ir.architecture.inputSize];

  w.write(`model = keras.Sequential([`);
  w.indent();

  let isFirst = true;
  for (const layer of ir.architecture.layers) {
    const inputArg = isFirst ? `, input_shape=(${inputShape.join(", ")},)` : "";

    switch (layer.kind) {
      case "dense": {
        const act = KERAS_ACTIVATION[layer.activation] ?? "relu";
        w.write(`layers.Dense(${layer.units}, activation="${act}"${isFirst ? `, input_shape=(${ir.architecture.inputSize},)` : ""}),`);
        isFirst = false;
        break;
      }
      case "batchnorm":
        w.write(`layers.BatchNormalization(),`);
        break;
      case "dropout":
        w.write(`layers.Dropout(${layer.rate}),`);
        break;
      case "conv2d":
        w.write(`layers.Conv2D(${layer.filters}, ${layer.kernelSize}, strides=${layer.strides}, padding="${layer.padding}", activation="${KERAS_ACTIVATION[layer.activation] ?? "relu"}"${inputArg}),`);
        isFirst = false;
        break;
      case "maxpool2d":
        w.write(`layers.MaxPooling2D(${layer.poolSize}),`);
        break;
      case "flatten":
        w.write(`layers.Flatten(),`);
        break;
      case "globalavgpool2d":
        w.write(`layers.GlobalAveragePooling2D(),`);
        break;
      case "reshape":
        w.write(`layers.Reshape((${layer.targetShape.join(", ")},)${inputArg}),`);
        isFirst = false;
        break;
      case "lstm":
        w.write(`layers.LSTM(${layer.units}, return_sequences=${layer.returnSequences ? "True" : "False"}${inputArg}),`);
        isFirst = false;
        break;
      case "gru":
        w.write(`layers.GRU(${layer.units}, return_sequences=${layer.returnSequences ? "True" : "False"}${inputArg}),`);
        isFirst = false;
        break;
      case "embedding":
        w.write(`layers.Embedding(${layer.inputDim}, ${layer.outputDim}${inputArg}),`);
        isFirst = false;
        break;
      case "bidirectional": {
        const inner = layer.wrapped;
        const rnnType = inner.kind === "gru" ? "GRU" : "LSTM";
        w.write(`layers.Bidirectional(layers.${rnnType}(${inner.units}, return_sequences=${inner.returnSequences ? "True" : "False"})${inputArg}),`);
        isFirst = false;
        break;
      }
      case "residual":
        w.write(`# Residual block (${layer.filters} filters) — use functional API for true skip connections`);
        w.write(`layers.Conv2D(${layer.filters}, ${layer.kernelSize}, padding="same", activation="relu"${inputArg}),`);
        w.write(`layers.BatchNormalization(),`);
        w.write(`layers.Conv2D(${layer.filters}, ${layer.kernelSize}, padding="same"),`);
        w.write(`layers.BatchNormalization(),`);
        isFirst = false;
        break;
      case "multihead_attention":
        w.write(`# Multi-head attention: ${layer.heads} heads`);
        break;
      case "layernorm":
        w.write(`layers.LayerNormalization(),`);
        break;
      case "add":
        break;
    }
  }

  w.dedent();
  w.write(`])`);
  w.blank();
  w.write(`model.summary()`);
}

function emitKerasCompile(w: CodeWriter, ir: NeuralNetworkIR): void {
  const { optimizer, loss } = ir.training;

  const kerasOpt: Record<string, string> = {
    adam: `keras.optimizers.Adam(learning_rate=${optimizer.learningRate})`,
    adamw: `keras.optimizers.AdamW(learning_rate=${optimizer.learningRate})`,
    sgd: `keras.optimizers.SGD(learning_rate=${optimizer.learningRate})`,
    rmsprop: `keras.optimizers.RMSprop(learning_rate=${optimizer.learningRate})`,
  };

  const kerasLoss: Record<string, string> = {
    meanSquaredError: `"mse"`,
    meanAbsoluteError: `"mae"`,
    categoricalCrossentropy: `"categorical_crossentropy"`,
    binaryCrossentropy: `"binary_crossentropy"`,
    huberLoss: `keras.losses.Huber()`,
  };

  const metrics = ir.task === "classification" ? `["accuracy"]` : `["mse"]`;

  w.write(`model.compile(`);
  w.indent();
  w.write(`optimizer=${kerasOpt[optimizer.name] ?? kerasOpt.adam},`);
  w.write(`loss=${kerasLoss[loss] ?? `"mse"`},`);
  w.write(`metrics=${metrics},`);
  w.dedent();
  w.write(`)`);
}

function emitKerasTrain(w: CodeWriter, ir: NeuralNetworkIR): void {
  w.write(`cb = []`);

  if (ir.training.earlyStop) {
    w.write(`cb.append(callbacks.EarlyStopping(`);
    w.indent();
    w.write(`monitor="${ir.training.earlyStop.monitor}",`);
    w.write(`patience=${ir.training.earlyStop.patience},`);
    w.write(`min_delta=${ir.training.earlyStop.minDelta},`);
    w.write(`restore_best_weights=True,`);
    w.dedent();
    w.write(`))`);
  }

  if (ir.training.lrSchedule.type !== "none") {
    switch (ir.training.lrSchedule.type) {
      case "cosine":
        w.write(`cb.append(callbacks.LearningRateScheduler(`);
        w.indent();
        w.write(`lambda epoch: ${ir.training.optimizer.learningRate} * (1 + np.cos(np.pi * epoch / ${ir.training.epochs})) / 2`);
        w.dedent();
        w.write(`))`);
        break;
      case "step":
        w.write(`cb.append(callbacks.LearningRateScheduler(`);
        w.indent();
        w.write(`lambda epoch: ${ir.training.optimizer.learningRate} * (0.1 ** (epoch // ${Math.floor(ir.training.epochs / 3)}))`);
        w.dedent();
        w.write(`))`);
        break;
      case "exponential":
        w.write(`cb.append(callbacks.LearningRateScheduler(`);
        w.indent();
        w.write(`lambda epoch: ${ir.training.optimizer.learningRate} * (0.95 ** epoch)`);
        w.dedent();
        w.write(`))`);
        break;
    }
  }

  w.blank();
  w.write(`history = model.fit(`);
  w.indent();
  w.write(`X_train, y_train,`);
  w.write(`epochs=${ir.training.epochs},`);
  w.write(`batch_size=${ir.training.batchSize},`);
  w.write(`validation_split=${ir.training.validationSplit},`);
  w.write(`callbacks=cb,`);
  w.write(`verbose=1,`);
  w.dedent();
  w.write(`)`);
  w.blank();

  w.write(`results = model.evaluate(X_test, y_test, verbose=0)`);
  if (ir.task === "classification") {
    w.write(`print(f"\\nTest accuracy: {results[1]:.4f}")`);
  } else {
    w.write(`print(f"\\nTest MSE: {results[1]:.6f}")`);
  }

  if (ir.output.exportPath) {
    w.blank();
    w.write(`model.save("${ir.output.exportPath}")`);
    w.write(`print(f"Model saved to ${ir.output.exportPath}")`);
  }

  w.blank();
  w.write(`print("Training complete.")`);
}

const PRETRAINED_KERAS: Record<string, { app: string; inputSize: number }> = {
  mobilenet: { app: "keras.applications.MobileNetV2", inputSize: 1280 },
  resnet50: { app: "keras.applications.ResNet50", inputSize: 2048 },
  resnet101: { app: "keras.applications.ResNet101", inputSize: 2048 },
  vgg16: { app: "keras.applications.VGG16", inputSize: 512 },
  vgg19: { app: "keras.applications.VGG19", inputSize: 512 },
  efficientnet: { app: "keras.applications.EfficientNetB0", inputSize: 1280 },
  inception: { app: "keras.applications.InceptionV3", inputSize: 2048 },
  densenet: { app: "keras.applications.DenseNet121", inputSize: 1024 },
};

function emitKerasPretrained(w: CodeWriter, ir: NeuralNetworkIR): void {
  const pre = ir.architecture.pretrained!;
  const cfg = PRETRAINED_KERAS[pre];
  if (!cfg) { emitKerasModel(w, ir); return; }

  w.write(`base_model = ${cfg.app}(weights="imagenet", include_top=False, pooling="avg")`);
  if (ir.architecture.freezeLayers) {
    w.write(`for layer in base_model.layers[:${ir.architecture.freezeLayers}]:`);
    w.indent();
    w.write(`layer.trainable = False`);
    w.dedent();
  } else {
    w.write(`base_model.trainable = False`);
  }
  w.blank();
  w.write(`model = keras.Sequential([`);
  w.indent();
  w.write(`base_model,`);
  w.write(`layers.Dense(256, activation="relu"),`);
  w.write(`layers.Dropout(0.3),`);
  w.write(`layers.Dense(${ir.architecture.outputSize}, activation="${ir.task === "classification" ? "softmax" : "linear"}"),`);
  w.dedent();
  w.write(`])`);
  w.blank();
  w.write(`model.summary()`);
}

function emitKerasExport(w: CodeWriter, ir: NeuralNetworkIR): void {
  w.write(`# ── Model Export ──`);
  const formats = ir.output.exportFormats ?? [];

  if (formats.includes("savedmodel")) {
    w.write(`model.save("saved_model/")`);
    w.write(`print("Exported: saved_model/ (TensorFlow SavedModel)")`);
  }
  if (formats.includes("tflite")) {
    w.write(`converter = tf.lite.TFLiteConverter.from_keras_model(model)`);
    w.write(`tflite_model = converter.convert()`);
    w.write(`with open("model.tflite", "wb") as f:`);
    w.indent();
    w.write(`f.write(tflite_model)`);
    w.dedent();
    w.write(`print("Exported: model.tflite (TFLite)")`);
  }
  if (formats.includes("onnx")) {
    w.write(`import tf2onnx`);
    w.write(`import onnx`);
    w.write(`spec = (tf.TensorSpec((None, ${ir.architecture.inputSize}), tf.float32),)`);
    w.write(`model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)`);
    w.write(`onnx.save(model_proto, "model.onnx")`);
    w.write(`print("Exported: model.onnx (ONNX)")`);
  }
  if (formats.includes("coreml")) {
    w.write(`import coremltools as ct`);
    w.write(`mlmodel = ct.convert(model)`);
    w.write(`mlmodel.save("model.mlpackage")`);
    w.write(`print("Exported: model.mlpackage (CoreML)")`);
  }
}

// ─── JAX/Flax Target ─────────────────────────────────────────────────

export function emitJAX(ir: NeuralNetworkIR): string {
  const w = new CodeWriter();

  w.write(`import jax`);
  w.write(`import jax.numpy as jnp`);
  w.write(`import flax.linen as nn`);
  w.write(`import optax`);
  w.write(`from flax.training import train_state`);
  w.write(`import numpy as np`);
  w.write(`import pandas as pd`);
  w.write(`from sklearn.model_selection import train_test_split`);
  w.write(`from sklearn.preprocessing import StandardScaler`);
  w.blank();
  emitHeader(w, ir, "#");
  w.blank();

  if (ir.preprocessing.seed !== undefined) {
    w.write(`SEED = ${ir.preprocessing.seed}`);
  } else {
    w.write(`SEED = 42`);
  }
  w.write(`key = jax.random.PRNGKey(SEED)`);
  w.blank();

  emitJAXModel(w, ir);
  w.blank();
  emitJAXDataLoader(w, ir);
  w.blank();
  emitJAXTrain(w, ir);

  return w.toString();
}

const JAX_ACTIVATION: Record<string, string> = {
  relu: "nn.relu",
  sigmoid: "nn.sigmoid",
  tanh: "nn.tanh",
  selu: "nn.selu",
  elu: "nn.elu",
  gelu: "nn.gelu",
  leaky_relu: "nn.leaky_relu",
  softmax: "nn.softmax",
  linear: "lambda x: x",
};

function emitJAXModel(w: CodeWriter, ir: NeuralNetworkIR): void {
  w.write(`class Model(nn.Module):`);
  w.indent();
  w.write(`@nn.compact`);
  w.write(`def __call__(self, x, training: bool = True):`);
  w.indent();

  for (const layer of ir.architecture.layers) {
    switch (layer.kind) {
      case "dense": {
        const act = JAX_ACTIVATION[layer.activation];
        if (layer.activation === "linear" || layer.activation === "softmax") {
          w.write(`x = nn.Dense(${layer.units})(x)`);
          if (layer.activation === "softmax") w.write(`x = nn.softmax(x)`);
        } else {
          w.write(`x = ${act ?? "nn.relu"}(nn.Dense(${layer.units})(x))`);
        }
        break;
      }
      case "batchnorm":
        w.write(`x = nn.BatchNorm(use_running_average=not training)(x)`);
        break;
      case "dropout":
        w.write(`x = nn.Dropout(rate=${layer.rate}, deterministic=not training)(x)`);
        break;
      case "conv2d":
        w.write(`x = nn.Conv(features=${layer.filters}, kernel_size=(${layer.kernelSize}, ${layer.kernelSize}), strides=(${layer.strides}, ${layer.strides}), padding="${layer.padding.toUpperCase()}")(x)`);
        w.write(`x = ${JAX_ACTIVATION[layer.activation] ?? "nn.relu"}(x)`);
        break;
      case "maxpool2d":
        w.write(`x = nn.max_pool(x, window_shape=(${layer.poolSize}, ${layer.poolSize}), strides=(${layer.poolSize}, ${layer.poolSize}))`);
        break;
      case "flatten":
        w.write(`x = x.reshape((x.shape[0], -1))`);
        break;
      case "globalavgpool2d":
        w.write(`x = jnp.mean(x, axis=(1, 2))`);
        break;
      case "reshape":
        w.write(`x = x.reshape((x.shape[0], ${layer.targetShape.join(", ")}))`);
        break;
      case "lstm":
        w.write(`x = nn.RNN(nn.LSTMCell(features=${layer.units}))(x)`);
        if (!layer.returnSequences) w.write(`x = x[:, -1, :]`);
        break;
      case "gru":
        w.write(`x = nn.RNN(nn.GRUCell(features=${layer.units}))(x)`);
        if (!layer.returnSequences) w.write(`x = x[:, -1, :]`);
        break;
      case "bidirectional": {
        const inner = layer.wrapped;
        const cellType = inner.kind === "gru" ? "GRUCell" : "LSTMCell";
        w.write(`fwd = nn.RNN(nn.${cellType}(features=${inner.units}))(x)`);
        w.write(`bwd = nn.RNN(nn.${cellType}(features=${inner.units}), reverse=True)(x)`);
        w.write(`x = jnp.concatenate([fwd, bwd], axis=-1)`);
        if (!inner.returnSequences) w.write(`x = x[:, -1, :]`);
        break;
      }
      case "embedding":
        w.write(`x = nn.Embed(num_embeddings=${layer.inputDim}, features=${layer.outputDim})(x)`);
        break;
      case "residual":
        w.write(`# Residual block`);
        w.write(`residual = x`);
        w.write(`x = nn.Conv(features=${layer.filters}, kernel_size=(${layer.kernelSize}, ${layer.kernelSize}), padding="SAME")(x)`);
        w.write(`x = nn.relu(nn.BatchNorm(use_running_average=not training)(x))`);
        w.write(`x = nn.Conv(features=${layer.filters}, kernel_size=(${layer.kernelSize}, ${layer.kernelSize}), padding="SAME")(x)`);
        w.write(`x = nn.BatchNorm(use_running_average=not training)(x)`);
        w.write(`x = nn.relu(x + residual)`);
        break;
      case "multihead_attention":
        w.write(`x = nn.SelfAttention(num_heads=${layer.heads}, qkv_features=${layer.keyDim * layer.heads})(x)`);
        break;
      case "layernorm":
        w.write(`x = nn.LayerNorm()(x)`);
        break;
      case "add":
        break;
    }
  }

  w.write(`return x`);
  w.dedent();
  w.dedent();
}

function emitJAXDataLoader(w: CodeWriter, ir: NeuralNetworkIR): void {
  w.write(`df = pd.read_csv("${ir.dataset}")`);
  w.write(`feature_cols = ${JSON.stringify(ir.features)}`);
  w.write(`target_col = "${ir.target}"`);
  w.blank();
  w.write(`X = df[feature_cols].values.astype("float32")`);
  w.write(`y = df[target_col].values`);
  w.blank();

  if (ir.preprocessing.normalize) {
    w.write(`scaler = StandardScaler()`);
    w.write(`X = scaler.fit_transform(X)`);
    w.blank();
  }

  const seedArg = ir.preprocessing.seed !== undefined ? `, random_state=${ir.preprocessing.seed}` : "";
  w.write(`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=${(1 - ir.preprocessing.trainTestSplit).toFixed(2)}${seedArg})`);
  w.blank();
  w.write(`X_train = jnp.array(X_train)`);
  w.write(`X_test = jnp.array(X_test)`);

  if (ir.task === "classification") {
    w.write(`y_train = jax.nn.one_hot(jnp.array(y_train, dtype=jnp.int32), ${ir.architecture.outputSize})`);
    w.write(`y_test = jax.nn.one_hot(jnp.array(y_test, dtype=jnp.int32), ${ir.architecture.outputSize})`);
  } else {
    w.write(`y_train = jnp.array(y_train).reshape(-1, 1)`);
    w.write(`y_test = jnp.array(y_test).reshape(-1, 1)`);
  }
}

function emitJAXTrain(w: CodeWriter, ir: NeuralNetworkIR): void {
  const { optimizer } = ir.training;

  const optaxOpt: Record<string, string> = {
    adam: `optax.adam(${optimizer.learningRate})`,
    adamw: `optax.adamw(${optimizer.learningRate})`,
    sgd: `optax.sgd(${optimizer.learningRate})`,
    rmsprop: `optax.rmsprop(${optimizer.learningRate})`,
  };

  w.blank();
  w.write(`model = Model()`);
  w.write(`variables = model.init(key, jnp.ones((1, ${ir.architecture.inputSize})))`);
  w.write(`tx = ${optaxOpt[optimizer.name] ?? optaxOpt.adam}`);

  if (ir.training.lrSchedule.type !== "none") {
    w.blank();
    switch (ir.training.lrSchedule.type) {
      case "cosine":
        w.write(`schedule = optax.cosine_decay_schedule(init_value=${optimizer.learningRate}, decay_steps=${ir.training.epochs})`);
        w.write(`tx = optax.adam(schedule)`);
        break;
      case "step":
        w.write(`schedule = optax.piecewise_constant_schedule(init_value=${optimizer.learningRate}, boundaries_and_scales={${Math.floor(ir.training.epochs / 3)}: 0.1, ${Math.floor(ir.training.epochs * 2 / 3)}: 0.1})`);
        w.write(`tx = optax.adam(schedule)`);
        break;
      case "exponential":
        w.write(`schedule = optax.exponential_decay(init_value=${optimizer.learningRate}, transition_steps=1, decay_rate=0.95)`);
        w.write(`tx = optax.adam(schedule)`);
        break;
    }
  }

  w.blank();
  w.write(`state = train_state.TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)`);
  w.blank();

  if (ir.task === "classification") {
    w.write(`def loss_fn(params, x, y):`);
    w.indent();
    w.write(`logits = model.apply({"params": params}, x)`);
    w.write(`return -jnp.mean(jnp.sum(y * jnp.log(logits + 1e-8), axis=-1))`);
    w.dedent();
  } else {
    const lossExpr = ir.training.loss === "meanAbsoluteError"
      ? `jnp.mean(jnp.abs(preds - y))`
      : ir.training.loss === "huberLoss"
        ? `jnp.mean(optax.huber_loss(preds, y))`
        : `jnp.mean((preds - y) ** 2)`;
    w.write(`def loss_fn(params, x, y):`);
    w.indent();
    w.write(`preds = model.apply({"params": params}, x)`);
    w.write(`return ${lossExpr}`);
    w.dedent();
  }

  w.blank();
  w.write(`@jax.jit`);
  w.write(`def train_step(state, x, y):`);
  w.indent();
  w.write(`loss, grads = jax.value_and_grad(loss_fn)(state.params, x, y)`);
  w.write(`state = state.apply_gradients(grads=grads)`);
  w.write(`return state, loss`);
  w.dedent();

  w.blank();
  w.write(`for epoch in range(${ir.training.epochs}):`);
  w.indent();
  w.write(`state, loss = train_step(state, X_train, y_train)`);
  w.write(`if (epoch + 1) % 10 == 0:`);
  w.indent();
  w.write(`print(f"  Epoch {epoch + 1}/${ir.training.epochs}: loss={float(loss):.4f}")`);
  w.dedent();
  w.dedent();

  w.blank();
  if (ir.task === "classification") {
    w.write(`test_preds = model.apply({"params": state.params}, X_test)`);
    w.write(`accuracy = jnp.mean(jnp.argmax(test_preds, axis=-1) == jnp.argmax(y_test, axis=-1))`);
    w.write(`print(f"\\nTest accuracy: {float(accuracy):.4f}")`);
  } else {
    w.write(`test_preds = model.apply({"params": state.params}, X_test)`);
    w.write(`test_mse = jnp.mean((test_preds - y_test) ** 2)`);
    w.write(`print(f"\\nTest MSE: {float(test_mse):.6f}")`);
  }

  w.blank();
  w.write(`print("Training complete.")`);
}

// ─── Summary Target ──────────────────────────────────────────────────

export function emitSummary(ir: NeuralNetworkIR): string {
  const w = new CodeWriter();

  w.write(`╔══════════════════════════════════════════════════╗`);
  w.write(`║         NeuroLang Compilation Summary            ║`);
  w.write(`╚══════════════════════════════════════════════════╝`);
  w.blank();

  const archType = (ir.architecture.type ?? "mlp").toUpperCase();
  w.write(`Task:        ${ir.task}`);
  w.write(`Target:      ${ir.target}`);
  w.write(`Features:    ${ir.features.join(", ")} (${ir.architecture.inputSize})`);
  w.write(`Dataset:     ${ir.dataset}`);
  w.write(`Arch Type:   ${archType}`);
  w.write(`Learn Mode:  ${ir.meta.learnMode}`);
  if (ir.architecture.inputShape) {
    w.write(`Input Shape: [${ir.architecture.inputShape.join(", ")}]`);
  }
  w.blank();

  w.write(`── Architecture ──────────────────────────────────`);
  w.blank();

  const flowParts: string[] = ir.architecture.inputShape
    ? [`Input(${ir.architecture.inputShape.join("×")})`]
    : [`Input(${ir.architecture.inputSize})`];

  for (const layer of ir.architecture.layers) {
    switch (layer.kind) {
      case "dense": flowParts.push(`Dense(${layer.units}, ${layer.activation})`); break;
      case "batchnorm": flowParts.push(`BatchNorm`); break;
      case "dropout": flowParts.push(`Dropout(${layer.rate})`); break;
      case "conv2d": flowParts.push(`Conv2D(${layer.filters}, ${layer.kernelSize}×${layer.kernelSize})`); break;
      case "maxpool2d": flowParts.push(`MaxPool(${layer.poolSize})`); break;
      case "flatten": flowParts.push(`Flatten`); break;
      case "globalavgpool2d": flowParts.push(`GlobalAvgPool`); break;
      case "reshape": flowParts.push(`Reshape(${layer.targetShape.join("×")})`); break;
      case "lstm": flowParts.push(`LSTM(${layer.units}${layer.returnSequences ? ", seq" : ""})`); break;
      case "gru": flowParts.push(`GRU(${layer.units}${layer.returnSequences ? ", seq" : ""})`); break;
      case "embedding": flowParts.push(`Embed(${layer.inputDim}→${layer.outputDim})`); break;
      case "bidirectional": {
        const inner = layer.wrapped;
        const t = inner.kind === "gru" ? "GRU" : "LSTM";
        flowParts.push(`BiDir-${t}(${inner.units})`);
        break;
      }
      case "residual": flowParts.push(`ResBlock(${layer.filters})`); break;
      case "multihead_attention": flowParts.push(`MHA(${layer.heads}h)`); break;
      case "layernorm": flowParts.push(`LayerNorm`); break;
      case "add": flowParts.push(`Add`); break;
    }
  }
  w.write(`  ${flowParts.join(" → ")}`);
  w.blank();
  w.write(`  Parameters:  ~estimated`);
  if (ir.architecture.batchNorm) w.write(`  Batch Norm:  enabled`);
  w.blank();

  w.write(`── Training ──────────────────────────────────────`);
  w.blank();
  w.write(`  Loss:        ${ir.training.loss}`);
  w.write(`  Optimizer:   ${ir.training.optimizer.name} (lr=${ir.training.optimizer.learningRate})`);
  w.write(`  Epochs:      ${ir.training.epochs}`);
  w.write(`  Batch Size:  ${ir.training.batchSize}`);
  if (ir.training.earlyStop) {
    w.write(`  Early Stop:  patience=${ir.training.earlyStop.patience}, monitor=${ir.training.earlyStop.monitor}`);
  }
  if (ir.training.lrSchedule.type !== "none") {
    w.write(`  LR Schedule: ${ir.training.lrSchedule.type}`);
  }
  if (ir.training.ensemble && ir.training.ensemble > 1) {
    w.write(`  Ensemble:    ${ir.training.ensemble} models (bagging)`);
  }
  w.blank();

  w.write(`── Preprocessing ─────────────────────────────────`);
  w.blank();
  w.write(`  Normalize:   ${ir.preprocessing.normalize}`);
  w.write(`  Split:       ${(ir.preprocessing.trainTestSplit * 100).toFixed(0)}% train / ${((1 - ir.preprocessing.trainTestSplit) * 100).toFixed(0)}% test`);
  if (ir.preprocessing.crossValidation) w.write(`  Cross-Val:   ${ir.preprocessing.crossValidation}-fold`);
  if (ir.preprocessing.seed !== undefined) w.write(`  Seed:        ${ir.preprocessing.seed}`);

  if (ir.architecture.pretrained) {
    w.blank();
    w.write(`── Transfer Learning ─────────────────────────────`);
    w.blank();
    w.write(`  Pretrained:  ${ir.architecture.pretrained}`);
    if (ir.architecture.freezeLayers) w.write(`  Freeze:      first ${ir.architecture.freezeLayers} layers`);
  }

  if (ir.targets && ir.targets.length > 1) {
    w.blank();
    w.write(`── Multi-Output ──────────────────────────────────`);
    w.blank();
    w.write(`  Targets:     ${ir.targets.join(", ")}`);
  }

  if (ir.output.exportPath || (ir.output.exportFormats && ir.output.exportFormats.length > 0)) {
    w.blank();
    w.write(`── Output ────────────────────────────────────────`);
    w.blank();
    if (ir.output.exportPath) w.write(`  Export:      ${ir.output.exportPath}`);
    if (ir.output.exportFormats) w.write(`  Formats:     ${ir.output.exportFormats.join(", ").toUpperCase()}`);
  }

  return w.toString();
}
