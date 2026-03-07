/**
 * Hyperparameter tuning — grid search over learning rates, batch sizes, architectures.
 */
import type { NeuralNetworkIR } from "./ir.js";
import { buildCandidateIR } from "./autoarch.js";

export interface TuneConfig {
  learningRates: number[];
  batchSizes: number[];
  architectures: { name: string; layers: number[] }[];
}

export interface TuneTrialResult {
  lr: number;
  batchSize: number;
  archName: string;
  layers: number[];
  probeLoss: number;
  probeMetric: number;
}

export interface TuneResult {
  trials: TuneTrialResult[];
  bestIdx: number;
  bestConfig: {
    lr: number;
    batchSize: number;
    layers: number[];
  };
}

const DEFAULT_CONFIG: TuneConfig = {
  learningRates: [0.01, 0.001, 0.0001],
  batchSizes: [16, 32, 64],
  architectures: [
    { name: "small",  layers: [64, 32] },
    { name: "medium", layers: [128, 64] },
    { name: "large",  layers: [256, 128, 64] },
  ],
};

export function getTuneConfig(): TuneConfig {
  return DEFAULT_CONFIG;
}

export function buildTrialIR(
  baseIR: NeuralNetworkIR,
  lr: number,
  batchSize: number,
  layers: number[],
  probeEpochs: number,
): NeuralNetworkIR {
  const candidate = buildCandidateIR(baseIR, layers, probeEpochs);
  return {
    ...candidate,
    training: {
      ...candidate.training,
      optimizer: { ...candidate.training.optimizer, learningRate: lr },
      batchSize,
    },
  };
}

export function totalTrials(config: TuneConfig = DEFAULT_CONFIG): number {
  return config.learningRates.length * config.batchSizes.length * config.architectures.length;
}

export function formatTuneResults(result: TuneResult): string {
  const lines: string[] = [];
  lines.push(`\n  Hyperparameter Tuning Results (${result.trials.length} trials)`);
  lines.push(`  ${"─".repeat(75)}`);
  lines.push(
    `  ${"#".padEnd(4)} ${"LR".padEnd(10)} ${"Batch".padEnd(7)} ${"Arch".padEnd(10)} ${"Layers".padEnd(18)} ${"Loss".padStart(10)} ${"Metric".padStart(10)}`,
  );
  lines.push(`  ${"─".repeat(75)}`);

  const sorted = result.trials
    .map((t, i) => ({ ...t, origIdx: i }))
    .sort((a, b) => a.probeLoss - b.probeLoss);

  for (let i = 0; i < Math.min(10, sorted.length); i++) {
    const t = sorted[i]!;
    const marker = t.origIdx === result.bestIdx ? " ◀" : "";
    lines.push(
      `  ${String(i + 1).padEnd(4)} ${t.lr.toFixed(4).padEnd(10)} ${String(t.batchSize).padEnd(7)} ${t.archName.padEnd(10)} ${JSON.stringify(t.layers).padEnd(18)} ${t.probeLoss.toFixed(6).padStart(10)} ${t.probeMetric.toFixed(4).padStart(10)}${marker}`,
    );
  }

  if (sorted.length > 10) {
    lines.push(`  ... and ${sorted.length - 10} more trials`);
  }

  const best = result.trials[result.bestIdx]!;
  lines.push(`\n  Best: lr=${best.lr}, batch=${best.batchSize}, arch=${best.archName} ${JSON.stringify(best.layers)}`);

  return lines.join("\n");
}
