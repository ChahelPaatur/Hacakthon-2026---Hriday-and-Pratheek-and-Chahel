/**
 * Data inspection utilities for NeuroLang.
 * Provides show, describe, sample, and correlate functionality.
 */

import type { Dataset } from "./datasets.js";

export interface ColumnStats {
  name: string;
  count: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  median: number;
  unique: number;
  nulls: number;
}

export interface DataDescription {
  rows: number;
  cols: number;
  featureNames: string[];
  targetName: string;
  classNames?: string[];
  stats: ColumnStats[];
  targetStats: ColumnStats;
}

export function describeDataset(ds: Dataset): DataDescription {
  const stats = ds.featureNames.map((name, colIdx) => {
    const values = ds.features.map((row) => row[colIdx]!);
    return computeStats(name, values);
  });

  const targetStats = computeStats(ds.targetName, ds.targets);

  return {
    rows: ds.features.length,
    cols: ds.featureNames.length,
    featureNames: ds.featureNames,
    targetName: ds.targetName,
    classNames: ds.classNames,
    stats,
    targetStats,
  };
}

function computeStats(name: string, values: number[]): ColumnStats {
  const sorted = [...values].sort((a, b) => a - b);
  const n = values.length;
  const mean = values.reduce((s, v) => s + v, 0) / n;
  const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance);
  const min = sorted[0] ?? 0;
  const max = sorted[n - 1] ?? 0;
  const median = n % 2 === 0
    ? ((sorted[n / 2 - 1] ?? 0) + (sorted[n / 2] ?? 0)) / 2
    : sorted[Math.floor(n / 2)] ?? 0;
  const unique = new Set(values).size;

  return { name, count: n, mean, std, min, max, median, unique, nulls: 0 };
}

export function formatDataPreview(
  ds: Dataset,
  maxRows: number = 10,
): string {
  const lines: string[] = [];
  const allCols = [...ds.featureNames, ds.targetName];
  const colWidths = allCols.map((name) => Math.max(name.length, 10));

  // Header
  const header = allCols.map((name, i) => name.padStart(colWidths[i]!)).join("  ");
  lines.push(`  ${header}`);
  lines.push(`  ${"─".repeat(header.length)}`);

  // Rows
  const rowCount = Math.min(maxRows, ds.features.length);
  for (let r = 0; r < rowCount; r++) {
    const vals = [
      ...ds.features[r]!.map((v) => v.toFixed(2)),
      ds.targets[r]!.toFixed(ds.classNames ? 0 : 2),
    ];
    const row = vals.map((v, i) => v.padStart(colWidths[i]!)).join("  ");
    lines.push(`  ${row}`);
  }

  if (ds.features.length > maxRows) {
    lines.push(`  ... ${ds.features.length - maxRows} more rows`);
  }

  return lines.join("\n");
}

export function formatDescription(desc: DataDescription): string {
  const lines: string[] = [];

  lines.push(`╔══════════════════════════════════════════════════╗`);
  lines.push(`║            Dataset Description                  ║`);
  lines.push(`╚══════════════════════════════════════════════════╝`);
  lines.push(``);
  lines.push(`  Shape:      ${desc.rows} rows × ${desc.cols} features`);
  lines.push(`  Target:     ${desc.targetName}${desc.classNames ? ` (${desc.classNames.length} classes: ${desc.classNames.join(", ")})` : ""}`);
  lines.push(`  Features:   ${desc.featureNames.join(", ")}`);
  lines.push(``);

  // Stats table
  const nameW = Math.max(12, ...desc.stats.map((s) => s.name.length));
  const hdr = [
    "Feature".padEnd(nameW),
    "Count".padStart(6),
    "Mean".padStart(10),
    "Std".padStart(10),
    "Min".padStart(10),
    "Median".padStart(10),
    "Max".padStart(10),
    "Unique".padStart(7),
  ];
  lines.push(`  ${hdr.join("  ")}`);
  lines.push(`  ${"─".repeat(hdr.join("  ").length)}`);

  for (const s of [...desc.stats, desc.targetStats]) {
    const row = [
      s.name.padEnd(nameW),
      String(s.count).padStart(6),
      s.mean.toFixed(2).padStart(10),
      s.std.toFixed(2).padStart(10),
      s.min.toFixed(2).padStart(10),
      s.median.toFixed(2).padStart(10),
      s.max.toFixed(2).padStart(10),
      String(s.unique).padStart(7),
    ];
    lines.push(`  ${row.join("  ")}`);
  }

  return lines.join("\n");
}

export function formatCorrelation(ds: Dataset): string {
  const lines: string[] = [];
  const cols = ds.featureNames.length;
  const n = ds.features.length;

  // Compute means and stds
  const means: number[] = [];
  const stds: number[] = [];
  for (let c = 0; c < cols; c++) {
    const mean = ds.features.reduce((s, r) => s + r[c]!, 0) / n;
    means.push(mean);
    const variance = ds.features.reduce((s, r) => s + (r[c]! - mean) ** 2, 0) / n;
    stds.push(Math.sqrt(variance) || 1);
  }

  // Correlation matrix
  const corr: number[][] = Array.from({ length: cols }, () => new Array(cols).fill(0));
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < cols; j++) {
      let sum = 0;
      for (let r = 0; r < n; r++) {
        sum += ((ds.features[r]![i]! - means[i]!) / stds[i]!) *
               ((ds.features[r]![j]! - means[j]!) / stds[j]!);
      }
      corr[i]![j] = sum / n;
    }
  }

  // Format
  const nameW = Math.max(8, ...ds.featureNames.map((n) => n.length));
  const cellW = Math.max(6, nameW);

  lines.push(`  Feature Correlation Matrix`);
  lines.push(`  ${"".padEnd(nameW)}  ${ds.featureNames.map((n) => n.slice(0, cellW).padStart(cellW)).join(" ")}`);
  lines.push(`  ${"─".repeat(nameW + 2 + (cellW + 1) * cols)}`);

  for (let i = 0; i < cols; i++) {
    const cells = corr[i]!.map((v) => {
      const s = v.toFixed(2).padStart(cellW);
      if (Math.abs(v) > 0.7 && i !== corr.indexOf(corr[i]!)) return `\x1b[33m${s}\x1b[0m`;
      return s;
    });
    lines.push(`  ${ds.featureNames[i]!.padEnd(nameW)}  ${cells.join(" ")}`);
  }

  return lines.join("\n");
}

export function formatSample(ds: Dataset, count: number = 5, seed: number = 42): string {
  const indices: number[] = [];
  let s = seed | 0;
  while (indices.length < Math.min(count, ds.features.length)) {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    const r = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    const idx = Math.floor(r * ds.features.length);
    if (!indices.includes(idx)) indices.push(idx);
  }

  const lines: string[] = [];
  const allCols = [...ds.featureNames, ds.targetName];
  const colWidths = allCols.map((name) => Math.max(name.length, 8));

  lines.push(`  Random sample (${count} rows, seed=${seed}):`);
  lines.push(``);
  const header = allCols.map((name, i) => name.padStart(colWidths[i]!)).join("  ");
  lines.push(`  ${header}`);
  lines.push(`  ${"─".repeat(header.length)}`);

  for (const idx of indices) {
    const vals = [
      ...ds.features[idx]!.map((v) => v.toFixed(2)),
      ds.targets[idx]!.toFixed(ds.classNames ? 0 : 2),
    ];
    const row = vals.map((v, i) => v.padStart(colWidths[i]!)).join("  ");
    lines.push(`  ${row}`);
  }

  return lines.join("\n");
}
