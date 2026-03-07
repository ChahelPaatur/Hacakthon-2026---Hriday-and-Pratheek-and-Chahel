import type { NeuralNetworkIR } from "./ir.js";
import { resolveDataset, parseCSV, type Dataset } from "./datasets.js";
import * as fs from "node:fs";
import * as path from "node:path";

export interface TrainingResult {
  lossHistory: number[];
  finalLoss: number;
  metric: { name: string; value: number };
  epochs: number;
  earlyStopEpoch?: number;
  predictions: { actual: number; predicted: number }[];
  crossValidation?: {
    folds: number;
    scores: number[];
    mean: number;
    std: number;
    metricName: string;
  };
}

export interface ExecutionCallbacks {
  onEpochEnd?: (epoch: number, loss: number) => void;
  onTrainStart?: () => void;
  onTrainEnd?: () => void;
}

export async function execute(
  ir: NeuralNetworkIR,
  callbacks?: ExecutionCallbacks,
): Promise<TrainingResult> {
  const tf = await importTf();

  const dataset = loadDataset(ir);
  const prepared = prepareData(tf, ir, dataset);

  const model = buildModel(tf, ir);
  compileModel(tf, model, ir);

  callbacks?.onTrainStart?.();

  const lossHistory: number[] = [];
  let earlyStopEpoch: number | undefined;

  // Manual early stopping state
  let bestLoss = Infinity;
  let patienceCounter = 0;
  const esConfig = ir.training.earlyStop;

  const fitCallbacks: import("@tensorflow/tfjs").CustomCallbackArgs = {
    onEpochEnd: (_epoch: number, logs: Record<string, number> | undefined) => {
      const loss = logs?.loss ?? 0;
      lossHistory.push(loss);
      callbacks?.onEpochEnd?.(_epoch, loss);

      if (esConfig) {
        const monitorVal = logs?.val_loss ?? loss;
        if (monitorVal < bestLoss - esConfig.minDelta) {
          bestLoss = monitorVal;
          patienceCounter = 0;
        } else {
          patienceCounter++;
          if (patienceCounter >= esConfig.patience) {
            earlyStopEpoch = _epoch + 1;
            model.stopTraining = true;
          }
        }
      }
    },
  };

  const history = await model.fit(prepared.xTrain, prepared.yTrain, {
    epochs: ir.training.epochs,
    batchSize: ir.training.batchSize,
    validationSplit: ir.training.validationSplit,
    shuffle: true,
    callbacks: [fitCallbacks],
  });

  const trainedEpochs = (history.history.loss as number[]).length;
  if (!earlyStopEpoch && esConfig && trainedEpochs < ir.training.epochs) {
    earlyStopEpoch = trainedEpochs;
  }

  callbacks?.onTrainEnd?.();

  const metric = evaluate(tf, model, prepared.xTest, prepared.yTest, ir);
  const predictions = predict(
    tf, model, prepared.xTest, prepared.yTest, ir,
    prepared.targetMean, prepared.targetStd,
  );

  // Export model if requested
  if (ir.output.exportPath) {
    try {
      const exportDir = path.resolve(ir.output.exportPath);
      await model.save(`file://${exportDir}`);
    } catch {
      console.warn(`Warning: Could not export model to ${ir.output.exportPath}`);
    }
  }

  // Clean up all tensors
  prepared.xTrain.dispose();
  prepared.yTrain.dispose();
  prepared.xTest.dispose();
  prepared.yTest.dispose();
  model.dispose();

  let crossValidation: TrainingResult["crossValidation"];
  if (ir.preprocessing.crossValidation && ir.preprocessing.crossValidation > 1) {
    crossValidation = await runCrossValidation(
      tf, ir, ir.preprocessing.crossValidation,
    );
  }

  return {
    lossHistory,
    finalLoss: lossHistory[lossHistory.length - 1] ?? 0,
    metric,
    epochs: trainedEpochs,
    earlyStopEpoch,
    predictions,
    crossValidation,
  };
}

async function importTf(): Promise<typeof import("@tensorflow/tfjs")> {
  try {
    return await import("@tensorflow/tfjs");
  } catch {
    throw new Error(
      "TensorFlow.js is required for execution.\nInstall it with: npm install @tensorflow/tfjs",
    );
  }
}

function loadDataset(ir: NeuralNetworkIR): Dataset {
  // Try built-in datasets first
  const builtin = resolveDataset(ir.dataset);
  if (builtin) return builtin;

  // Try loading from disk
  const csvPath = path.resolve(ir.dataset);
  if (fs.existsSync(csvPath)) {
    const text = fs.readFileSync(csvPath, "utf-8");
    const { headers, rows } = parseCSV(text);

    const featureIdx = ir.features.map((f) => headers.indexOf(f));
    const targetIdx = headers.indexOf(ir.target);

    if (featureIdx.some((i) => i === -1)) {
      const missing = ir.features.filter((f) => headers.indexOf(f) === -1);
      throw new Error(`Feature columns not found in CSV: ${missing.join(", ")}`);
    }
    if (targetIdx === -1) {
      throw new Error(`Target column '${ir.target}' not found in CSV headers: ${headers.join(", ")}`);
    }

    return {
      featureNames: ir.features,
      targetName: ir.target,
      features: rows.map((row) => featureIdx.map((i) => row[i]!)),
      targets: rows.map((row) => row[targetIdx]!),
    };
  }

  throw new Error(
    `Dataset '${ir.dataset}' not found.\n` +
    `Built-in datasets: iris.csv, housing.csv, titanic.csv, wine.csv, digits.csv\n` +
    `Or provide a valid CSV file path.`,
  );
}

interface PreparedData {
  xTrain: import("@tensorflow/tfjs").Tensor2D;
  yTrain: import("@tensorflow/tfjs").Tensor;
  xTest: import("@tensorflow/tfjs").Tensor2D;
  yTest: import("@tensorflow/tfjs").Tensor;
  targetMean: number;
  targetStd: number;
}

function prepareData(
  tf: typeof import("@tensorflow/tfjs"),
  ir: NeuralNetworkIR,
  dataset: Dataset,
): PreparedData {
  // Filter to only the feature columns requested by the program
  let features: number[][];
  if (ir.features.length < dataset.featureNames.length) {
    const colIdx = ir.features.map((f) => {
      const idx = dataset.featureNames.indexOf(f);
      return idx >= 0 ? idx : dataset.featureNames.findIndex(
        (n) => n.toLowerCase() === f.toLowerCase(),
      );
    });
    const allFound = colIdx.every((i) => i >= 0);
    features = allFound
      ? dataset.features.map((row) => colIdx.map((i) => row[i]!))
      : dataset.features;
  } else {
    features = dataset.features;
  }
  let targets = dataset.targets;

  // Seeded shuffle
  const rng = seededRng(ir.preprocessing.seed ?? Date.now());
  const indices = Array.from({ length: features.length }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
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

  // Normalize regression targets
  let targetMean = 0;
  let targetStd = 1;
  if (ir.task === "regression" && ir.preprocessing.normalize) {
    targetMean = targets.reduce((s, v) => s + v, 0) / targets.length;
    let variance = 0;
    for (const v of targets) variance += (v - targetMean) ** 2;
    targetStd = Math.sqrt(variance / targets.length) || 1;
    targets = targets.map((v) => (v - targetMean) / targetStd);
  }

  // Split
  const splitIdx = Math.floor(features.length * ir.preprocessing.trainTestSplit);

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

  return { xTrain, yTrain, xTest, yTest, targetMean, targetStd };
}

function buildModel(
  tf: typeof import("@tensorflow/tfjs"),
  ir: NeuralNetworkIR,
) {
  const model = tf.sequential();
  let isFirst = true;

  const getInputShape = (): number[] => {
    if (ir.architecture.inputShape) return ir.architecture.inputShape;
    return [ir.architecture.inputSize];
  };

  for (const layer of ir.architecture.layers) {
    const firstConfig = isFirst ? { inputShape: getInputShape() } : {};

    switch (layer.kind) {
      case "dense": {
        const config = isFirst
          ? { inputShape: [ir.architecture.inputSize], units: layer.units, activation: layer.activation as "relu" }
          : { units: layer.units, activation: layer.activation as "relu" };
        model.add(tf.layers.dense(config));
        isFirst = false;
        break;
      }
      case "batchnorm":
        model.add(tf.layers.batchNormalization());
        break;
      case "dropout":
        model.add(tf.layers.dropout({ rate: layer.rate }));
        break;
      case "conv2d":
        model.add(tf.layers.conv2d({
          ...firstConfig,
          filters: layer.filters,
          kernelSize: layer.kernelSize,
          strides: layer.strides,
          padding: layer.padding,
          activation: layer.activation as "relu",
        }));
        isFirst = false;
        break;
      case "maxpool2d":
        model.add(tf.layers.maxPooling2d({ poolSize: [layer.poolSize, layer.poolSize] }));
        break;
      case "flatten":
        model.add(tf.layers.flatten());
        break;
      case "globalavgpool2d":
        model.add(tf.layers.globalAveragePooling2d({}));
        break;
      case "reshape":
        model.add(tf.layers.reshape({ ...firstConfig, targetShape: layer.targetShape }));
        isFirst = false;
        break;
      case "lstm":
        model.add(tf.layers.lstm({
          ...firstConfig,
          units: layer.units,
          returnSequences: layer.returnSequences,
        }));
        isFirst = false;
        break;
      case "gru":
        model.add(tf.layers.gru({
          ...firstConfig,
          units: layer.units,
          returnSequences: layer.returnSequences,
        }));
        isFirst = false;
        break;
      case "embedding":
        model.add(tf.layers.embedding({
          ...firstConfig,
          inputDim: layer.inputDim,
          outputDim: layer.outputDim,
        }));
        isFirst = false;
        break;
      case "bidirectional":
        model.add(tf.layers.bidirectional({
          ...firstConfig,
          layer: layer.wrapped.kind === "gru"
            ? tf.layers.gru({ units: layer.wrapped.units, returnSequences: layer.wrapped.returnSequences })
            : tf.layers.lstm({ units: layer.wrapped.units, returnSequences: layer.wrapped.returnSequences }),
        }));
        isFirst = false;
        break;
    }
  }

  return model;
}

function compileModel(
  tf: typeof import("@tensorflow/tfjs"),
  model: ReturnType<typeof tf.sequential>,
  ir: NeuralNetworkIR,
) {
  const lr = ir.training.optimizer.learningRate;
  const optimizerName = ir.training.optimizer.name;
  const optimizer =
    optimizerName === "sgd" ? tf.train.sgd(lr)
    : optimizerName === "rmsprop" ? tf.train.rmsprop(lr)
    : optimizerName === "adamw" ? tf.train.adam(lr)
    : tf.train.adam(lr);

  const loss: string | ((yTrue: import("@tensorflow/tfjs").Tensor, yPred: import("@tensorflow/tfjs").Tensor) => import("@tensorflow/tfjs").Scalar) =
    ir.training.loss === "huberLoss"
      ? (yTrue, yPred) => tf.losses.huberLoss(yTrue, yPred).mean() as import("@tensorflow/tfjs").Scalar
      : ir.training.loss;

  model.compile({
    optimizer,
    loss,
    metrics: ir.task === "classification" ? ["accuracy"] : ["mse"],
  });
}

function evaluate(
  tf: typeof import("@tensorflow/tfjs"),
  model: ReturnType<typeof tf.sequential>,
  xTest: import("@tensorflow/tfjs").Tensor2D,
  yTest: import("@tensorflow/tfjs").Tensor,
  ir: NeuralNetworkIR,
): { name: string; value: number } {
  const result = model.evaluate(xTest, yTest as import("@tensorflow/tfjs").Tensor2D) as import("@tensorflow/tfjs").Scalar[];
  const metricValue = Array.isArray(result) ? result[1]?.dataSync()[0] ?? 0 : 0;

  return ir.task === "classification"
    ? { name: "accuracy", value: metricValue }
    : { name: "mse", value: metricValue };
}

function predict(
  tf: typeof import("@tensorflow/tfjs"),
  model: ReturnType<typeof tf.sequential>,
  xTest: import("@tensorflow/tfjs").Tensor2D,
  yTest: import("@tensorflow/tfjs").Tensor,
  ir: NeuralNetworkIR,
  targetMean = 0,
  targetStd = 1,
): { actual: number; predicted: number }[] {
  const preds = (model.predict(xTest) as import("@tensorflow/tfjs").Tensor2D).arraySync() as number[][];
  const actuals = (yTest as import("@tensorflow/tfjs").Tensor2D).arraySync() as number[][] | number[];
  const results: { actual: number; predicted: number }[] = [];

  const count = Math.min(10, preds.length);

  for (let i = 0; i < count; i++) {
    if (ir.task === "classification") {
      const predClass = preds[i]!.indexOf(Math.max(...preds[i]!));
      const actualRow = actuals[i];
      const actualClass = Array.isArray(actualRow)
        ? actualRow.indexOf(Math.max(...actualRow))
        : (actualRow as number);
      results.push({ actual: actualClass, predicted: predClass });
    } else {
      const rawActual = Array.isArray(actuals[i]) ? (actuals[i] as number[])[0]! : (actuals[i] as number);
      const rawPred = preds[i]![0]!;
      results.push({
        actual: rawActual * targetStd + targetMean,
        predicted: rawPred * targetStd + targetMean,
      });
    }
  }

  return results;
}

async function runCrossValidation(
  tf: typeof import("@tensorflow/tfjs"),
  ir: NeuralNetworkIR,
  k: number,
): Promise<NonNullable<TrainingResult["crossValidation"]>> {
  const dataset = loadDataset(ir);

  let features: number[][];
  if (ir.features.length < dataset.featureNames.length) {
    const colIdx = ir.features.map((f) => {
      const idx = dataset.featureNames.indexOf(f);
      return idx >= 0 ? idx : dataset.featureNames.findIndex(
        (n) => n.toLowerCase() === f.toLowerCase(),
      );
    });
    const allFound = colIdx.every((i) => i >= 0);
    features = allFound
      ? dataset.features.map((row) => colIdx.map((i) => row[i]!))
      : dataset.features;
  } else {
    features = dataset.features;
  }
  const targets = dataset.targets;

  // Shuffle
  const rng = seededRng(ir.preprocessing.seed ?? 42);
  const indices = Array.from({ length: features.length }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [indices[i], indices[j]] = [indices[j]!, indices[i]!];
  }
  const shuffledFeatures = indices.map((i) => features[i]!);
  const shuffledTargets = indices.map((i) => targets[i]!);

  const foldSize = Math.floor(shuffledFeatures.length / k);
  const scores: number[] = [];
  const metricName = ir.task === "classification" ? "accuracy" : "mse";

  for (let fold = 0; fold < k; fold++) {
    const valStart = fold * foldSize;
    const valEnd = fold === k - 1 ? shuffledFeatures.length : valStart + foldSize;

    let trainF = [...shuffledFeatures.slice(0, valStart), ...shuffledFeatures.slice(valEnd)];
    const valF = shuffledFeatures.slice(valStart, valEnd);
    let trainT = [...shuffledTargets.slice(0, valStart), ...shuffledTargets.slice(valEnd)];
    const valT = shuffledTargets.slice(valStart, valEnd);

    // Normalize
    if (ir.preprocessing.normalize) {
      const cols = trainF[0]!.length;
      const means = new Array<number>(cols).fill(0);
      const stds = new Array<number>(cols).fill(0);
      for (const row of trainF) { for (let c = 0; c < cols; c++) means[c] += row[c]!; }
      for (let c = 0; c < cols; c++) means[c] /= trainF.length;
      for (const row of trainF) { for (let c = 0; c < cols; c++) stds[c] += (row[c]! - means[c]!) ** 2; }
      for (let c = 0; c < cols; c++) stds[c] = Math.sqrt(stds[c]! / trainF.length) || 1;
      trainF = trainF.map((row) => row.map((v, c) => (v - means[c]!) / stds[c]!));
      const normVal = valF.map((row) => row.map((v, c) => (v - means[c]!) / stds[c]!));
      valF.splice(0, valF.length, ...normVal);
    }

    const model = buildModel(tf, ir);
    compileModel(tf, model, ir);

    const xTrain = tf.tensor2d(trainF);
    const xVal = tf.tensor2d(valF);
    let yTrain: import("@tensorflow/tfjs").Tensor;
    let yVal: import("@tensorflow/tfjs").Tensor;

    if (ir.task === "classification") {
      yTrain = tf.oneHot(tf.tensor1d(trainT, "int32"), ir.architecture.outputSize).cast("float32");
      yVal = tf.oneHot(tf.tensor1d(valT, "int32"), ir.architecture.outputSize).cast("float32");
    } else {
      yTrain = tf.tensor2d(trainT.map((v) => [v]));
      yVal = tf.tensor2d(valT.map((v) => [v]));
    }

    await model.fit(xTrain, yTrain, {
      epochs: Math.min(ir.training.epochs, 30),
      batchSize: ir.training.batchSize,
      validationSplit: 0,
      shuffle: true,
      verbose: 0,
    });

    const evalResult = model.evaluate(xVal, yVal as import("@tensorflow/tfjs").Tensor2D) as import("@tensorflow/tfjs").Scalar[];
    const metricVal = Array.isArray(evalResult) ? evalResult[1]?.dataSync()[0] ?? 0 : 0;
    scores.push(metricVal);

    xTrain.dispose(); xVal.dispose();
    yTrain.dispose(); yVal.dispose();
    model.dispose();
  }

  const mean = scores.reduce((s, v) => s + v, 0) / scores.length;
  const variance = scores.reduce((s, v) => s + (v - mean) ** 2, 0) / scores.length;
  const std = Math.sqrt(variance);

  return { folds: k, scores, mean, std, metricName };
}

function seededRng(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
