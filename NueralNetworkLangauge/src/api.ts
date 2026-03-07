/**
 * NeuroLang Programmatic API
 *
 * Use this module to integrate NeuroLang into any Node.js/TypeScript project.
 *
 * @example
 * ```ts
 * import { neurolang } from 'neurolang/api';
 *
 * // One-liner: compile + train from natural language
 * const result = await neurolang("Predict species with petal_length & petal_width from iris.csv");
 *
 * // Or from keyword syntax
 * const result2 = await neurolang(`
 *   task classification
 *   predict species
 *   inputs petal_length petal_width
 *   dataset iris.csv
 *   epochs 30
 * `);
 *
 * console.log(result.accuracy);     // 0.9667
 * console.log(result.predictions);  // [{ actual: 0, predicted: 0 }, ...]
 * console.log(result.lossHistory);  // [1.08, 0.82, 0.54, ...]
 *
 * // Just compile (no training)
 * const compiled = neurolang.compile("Predict species with a & b from iris.csv");
 * console.log(compiled.ir);   // full intermediate representation
 * console.log(compiled.code); // generated TensorFlow.js code
 *
 * // Generate code for different targets
 * const pytorch = neurolang.compile("Predict species from iris.csv", { target: "pytorch" });
 * const keras   = neurolang.compile("Predict species from iris.csv", { target: "keras" });
 * const jax     = neurolang.compile("Predict species from iris.csv", { target: "jax" });
 *
 * // Inspect data
 * const info = neurolang.inspect("iris.csv");
 * console.log(info.rows);       // 150
 * console.log(info.stats);      // per-column statistics
 * console.log(info.preview);    // first 10 rows
 * ```
 */

import { compile as rawCompile, type CompilationResult, type CompileOptions, type CodegenTarget } from "./compiler.js";
import { execute, type TrainingResult } from "./runtime.js";
import { resolveDataset } from "./datasets.js";
import { describeDataset, formatDataPreview, type DataDescription } from "./inspect.js";

export interface NeurolangResult {
  accuracy?: number;
  mse?: number;
  finalLoss: number;
  lossHistory: number[];
  predictions: { actual: number; predicted: number }[];
  epochs: number;
  earlyStopEpoch?: number;
  crossValidation?: TrainingResult["crossValidation"];
  ir: CompilationResult["ir"];
  code: string;
  target: CodegenTarget;
  compileTimeMs: number;
  trainTimeMs: number;
}

export interface InspectResult {
  rows: number;
  cols: number;
  featureNames: string[];
  targetName: string;
  stats: DataDescription["stats"];
  preview: string;
}

export async function neurolang(
  source: string,
  options: CompileOptions & { run?: boolean; silent?: boolean } = {},
): Promise<NeurolangResult> {
  const compileStart = performance.now();
  const compiled = rawCompile(source, options);
  const compileTimeMs = performance.now() - compileStart;

  const trainStart = performance.now();
  const trainResult = await execute(compiled.ir);
  const trainTimeMs = performance.now() - trainStart;

  return {
    accuracy: compiled.ir.task === "classification" ? trainResult.metric.value : undefined,
    mse: compiled.ir.task === "regression" ? trainResult.metric.value : undefined,
    finalLoss: trainResult.finalLoss,
    lossHistory: trainResult.lossHistory,
    predictions: trainResult.predictions,
    epochs: trainResult.epochs,
    earlyStopEpoch: trainResult.earlyStopEpoch,
    crossValidation: trainResult.crossValidation,
    ir: compiled.ir,
    code: compiled.code,
    target: compiled.target,
    compileTimeMs,
    trainTimeMs,
  };
}

neurolang.compile = function compileOnly(
  source: string,
  options: CompileOptions = {},
): CompilationResult {
  return rawCompile(source, options);
};

neurolang.inspect = function inspect(datasetName: string): InspectResult | null {
  const ds = resolveDataset(datasetName);
  if (!ds) return null;

  const desc = describeDataset(ds);
  const preview = formatDataPreview(ds, 10);

  return {
    rows: desc.rows,
    cols: desc.cols,
    featureNames: desc.featureNames,
    targetName: desc.targetName,
    stats: desc.stats,
    preview,
  };
};

neurolang.generate = function generate(
  source: string,
  target: CodegenTarget,
): string {
  const compiled = rawCompile(source, { target });
  return compiled.code;
};

export default neurolang;
