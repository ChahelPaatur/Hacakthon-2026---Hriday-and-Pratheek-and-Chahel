/**
 * Tagged template literal for inline NeuroLang.
 *
 * Write neural-network definitions directly inside TypeScript/JavaScript
 * and interpolate variables, feature lists, or even raw data arrays.
 *
 * @example
 * ```ts
 * import { nl } from 'neurolang';
 *
 * // --- Built-in dataset ---
 * const r1 = await nl`Predict species with petal_length & petal_width from iris`;
 * console.log(r1.accuracy);   // 0.96
 *
 * // --- Your own data (array of objects) ---
 * const data = [
 *   { age: 25, income: 50000, churned: 0 },
 *   { age: 45, income: 120000, churned: 1 },
 *   // ...
 * ];
 * const r2 = await nl`Predict churned with age & income from ${data}`;
 *
 * // --- Interpolate JS variables ---
 * const target = "price";
 * const features = ["size", "bedrooms"];
 * const epochs = 50;
 * const r3 = await nl`Predict ${target} with ${features} from housing epochs ${epochs}`;
 *
 * // --- Just compile (no training) ---
 * const compiled = nl.compile`Predict species from iris`;
 * console.log(compiled.code);
 *
 * // --- Generate code for a target ---
 * const pyCode = nl.pytorch`Predict species from iris`;
 * const kerasCode = nl.keras`Predict species from iris`;
 * ```
 */

import { compile as rawCompile, type CompilationResult, type CompileOptions, type CodegenTarget } from "./compiler.js";
import { registerInlineDataset, unregisterInlineDataset, datasetFromObjects } from "./datasets.js";
import type { NeurolangResult } from "./api.js";

let inlineCounter = 0;

interface InterpolationResult {
  source: string;
  cleanups: Array<() => void>;
}

function resolveInterpolation(
  strings: TemplateStringsArray,
  values: unknown[],
): InterpolationResult {
  const cleanups: Array<() => void> = [];
  const parts: string[] = [];

  for (let i = 0; i < strings.length; i++) {
    parts.push(strings[i]!);

    if (i < values.length) {
      const val = values[i];

      if (typeof val === "string") {
        parts.push(val);
      } else if (typeof val === "number" || typeof val === "boolean") {
        parts.push(String(val));
      } else if (Array.isArray(val)) {
        if (val.length === 0) {
          parts.push("__empty__");
        } else if (typeof val[0] === "string") {
          // string[] → feature list: "a & b & c"
          parts.push((val as string[]).join(" & "));
        } else if (typeof val[0] === "object" && val[0] !== null) {
          // Record<string, number>[] → inline dataset
          const name = `__inline_${inlineCounter++}`;
          const ds = datasetFromObjects(val as Record<string, number>[]);
          registerInlineDataset(name, ds);
          cleanups.push(() => unregisterInlineDataset(name));
          parts.push(name);
        } else if (typeof val[0] === "number") {
          // number[] → space-separated (rare, but handle it)
          parts.push((val as number[]).join(" "));
        } else {
          parts.push(String(val));
        }
      } else {
        parts.push(String(val));
      }
    }
  }

  return { source: parts.join(""), cleanups };
}

function runCleanups(cleanups: Array<() => void>): void {
  for (const fn of cleanups) {
    try { fn(); } catch { /* ignore */ }
  }
}

/**
 * Tagged template literal — compile & train a neural network inline.
 *
 * Returns a Promise that resolves to a full NeurolangResult.
 */
export function nl(
  strings: TemplateStringsArray,
  ...values: unknown[]
): Promise<NeurolangResult> {
  const { source, cleanups } = resolveInterpolation(strings, values);

  const run = async (): Promise<NeurolangResult> => {
    const { execute } = await import("./runtime.js");

    const compileStart = performance.now();
    const compiled = rawCompile(source);
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
  };

  return run().finally(() => runCleanups(cleanups));
}

/**
 * Compile-only tagged template — returns the compilation result synchronously.
 */
nl.compile = function compileTag(
  strings: TemplateStringsArray,
  ...values: unknown[]
): CompilationResult {
  const { source, cleanups } = resolveInterpolation(strings, values);
  try {
    return rawCompile(source);
  } finally {
    runCleanups(cleanups);
  }
};

/**
 * Generate PyTorch code from inline NeuroLang.
 */
nl.pytorch = function pytorchTag(
  strings: TemplateStringsArray,
  ...values: unknown[]
): string {
  const { source, cleanups } = resolveInterpolation(strings, values);
  try {
    return rawCompile(source, { target: "pytorch" }).code;
  } finally {
    runCleanups(cleanups);
  }
};

/**
 * Generate Keras code from inline NeuroLang.
 */
nl.keras = function kerasTag(
  strings: TemplateStringsArray,
  ...values: unknown[]
): string {
  const { source, cleanups } = resolveInterpolation(strings, values);
  try {
    return rawCompile(source, { target: "keras" }).code;
  } finally {
    runCleanups(cleanups);
  }
};

/**
 * Generate JAX/Flax code from inline NeuroLang.
 */
nl.jax = function jaxTag(
  strings: TemplateStringsArray,
  ...values: unknown[]
): string {
  const { source, cleanups } = resolveInterpolation(strings, values);
  try {
    return rawCompile(source, { target: "jax" }).code;
  } finally {
    runCleanups(cleanups);
  }
};

/**
 * Generate code for any target from inline NeuroLang.
 */
nl.generate = function generateTag(target: CodegenTarget) {
  return function tag(
    strings: TemplateStringsArray,
    ...values: unknown[]
  ): string {
    const { source, cleanups } = resolveInterpolation(strings, values);
    try {
      return rawCompile(source, { target }).code;
    } finally {
      runCleanups(cleanups);
    }
  };
};

export { resolveInterpolation as _resolveInterpolation };
