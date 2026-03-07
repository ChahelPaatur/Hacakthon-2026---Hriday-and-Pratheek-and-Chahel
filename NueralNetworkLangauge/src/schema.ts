/**
 * Data schema detection — infers a NeuroLang program from a CSV file.
 */
import * as fs from "node:fs";
import { parseCSV } from "./datasets.js";

export interface InferredSchema {
  suggestedTask: "classification" | "regression";
  suggestedTarget: string;
  suggestedInputs: string[];
  numSamples: number;
  numFeatures: number;
  suggestedProgram: string;
}

export function inferSchema(csvPath: string): InferredSchema {
  const text = fs.readFileSync(csvPath, "utf-8");
  const { headers, rows } = parseCSV(text);

  if (headers.length < 2) {
    throw new Error("CSV must have at least 2 columns (1 target + 1 feature)");
  }

  const colStats = headers.map((name, idx) => {
    const values = rows.map((row) => row[idx]!);
    const uniqueValues = new Set(values);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const mean = values.reduce((s, v) => s + v, 0) / values.length;
    const isInteger = values.every((v) => Number.isInteger(v));
    return { name, idx, uniqueCount: uniqueValues.size, min, max, mean, isInteger };
  });

  // The target is the last column by default, unless another column looks
  // more like a class label (few unique integer values)
  let targetIdx = headers.length - 1;
  const lastCol = colStats[targetIdx]!;

  for (let i = 0; i < colStats.length - 1; i++) {
    const col = colStats[i]!;
    if (col.isInteger && col.uniqueCount <= 10 && col.uniqueCount < lastCol.uniqueCount) {
      if (!lastCol.isInteger || lastCol.uniqueCount > 20) {
        targetIdx = i;
        break;
      }
    }
  }

  const target = colStats[targetIdx]!;
  const isClassification = target.isInteger && target.uniqueCount <= 20;
  const suggestedTask = isClassification ? "classification" : "regression";

  const suggestedInputs = headers.filter((_, i) => i !== targetIdx);

  const lines: string[] = [];
  lines.push(`task ${suggestedTask}`);
  lines.push(`predict ${target.name}`);
  lines.push(`inputs ${suggestedInputs.join(" ")}`);
  lines.push(`dataset "${csvPath}"`);
  lines.push(`optimizer adam`);

  if (isClassification) {
    lines.push(`loss cross_entropy`);
  } else {
    lines.push(`loss mse`);
  }

  if (rows.length > 500) {
    lines.push(`epochs 60`);
    lines.push(`batch_size 32`);
  } else {
    lines.push(`epochs 40`);
    lines.push(`batch_size 16`);
  }

  if (suggestedInputs.length > 10) {
    lines.push(`learn deep`);
    lines.push(`dropout 0.2`);
  } else {
    lines.push(`learn nonlinear`);
  }

  return {
    suggestedTask,
    suggestedTarget: target.name,
    suggestedInputs,
    numSamples: rows.length,
    numFeatures: suggestedInputs.length,
    suggestedProgram: lines.join("\n"),
  };
}

export function formatSchema(schema: InferredSchema): string {
  const lines: string[] = [];
  lines.push(`\n  Schema Detection Results`);
  lines.push(`  ${"─".repeat(50)}`);
  lines.push(`  Samples:    ${schema.numSamples}`);
  lines.push(`  Features:   ${schema.numFeatures}`);
  lines.push(`  Task:       ${schema.suggestedTask}`);
  lines.push(`  Target:     ${schema.suggestedTarget}`);
  lines.push(`  Inputs:     ${schema.suggestedInputs.join(", ")}`);
  lines.push(`  ${"─".repeat(50)}`);
  lines.push(`\n  Suggested program:\n`);
  for (const line of schema.suggestedProgram.split("\n")) {
    lines.push(`    ${line}`);
  }
  return lines.join("\n");
}
