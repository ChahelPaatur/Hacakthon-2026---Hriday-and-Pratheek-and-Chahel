/**
 * Auto-architecture search — tries multiple network sizes and picks the best.
 */
import type { NeuralNetworkIR, LayerSpec, DenseLayerSpec } from "./ir.js";

export interface ArchCandidate {
  name: string;
  layers: number[];
  probeLoss: number;
  probeMetric: number;
}

export interface AutoSearchResult {
  candidates: ArchCandidate[];
  bestIdx: number;
  bestLayers: number[];
}

const CANDIDATES: { name: string; layers: number[] }[] = [
  { name: "tiny",   layers: [32] },
  { name: "small",  layers: [64, 32] },
  { name: "medium", layers: [128, 64] },
  { name: "large",  layers: [256, 128, 64] },
];

export function getCandidates(): typeof CANDIDATES {
  return CANDIDATES;
}

export function buildCandidateIR(
  baseIR: NeuralNetworkIR,
  hiddenLayers: number[],
  probeEpochs: number,
): NeuralNetworkIR {
  const activation = getActivation(baseIR);
  const dropout = getDropout(baseIR);
  const useBatchNorm = baseIR.architecture.batchNorm;

  const layers: LayerSpec[] = [];
  for (const units of hiddenLayers) {
    layers.push({ kind: "dense", units, activation });
    if (useBatchNorm) layers.push({ kind: "batchnorm" });
    if (dropout > 0) layers.push({ kind: "dropout", rate: dropout });
  }
  const outputActivation = baseIR.task === "classification" ? "softmax" : "linear";
  layers.push({ kind: "dense", units: baseIR.architecture.outputSize, activation: outputActivation });

  let params = 0;
  let prev = baseIR.architecture.inputSize;
  for (const units of hiddenLayers) {
    params += prev * units + units;
    if (useBatchNorm) params += units * 4;
    prev = units;
  }
  params += prev * baseIR.architecture.outputSize + baseIR.architecture.outputSize;

  return {
    ...baseIR,
    architecture: {
      ...baseIR.architecture,
      layers,
    },
    training: {
      ...baseIR.training,
      epochs: probeEpochs,
      earlyStop: undefined,
      tune: undefined,
    },
    output: { exportPath: undefined },
    meta: { ...baseIR.meta, parameterCount: params },
  };
}

function getActivation(ir: NeuralNetworkIR): string {
  const dense = ir.architecture.layers.find((l) => l.kind === "dense") as DenseLayerSpec | undefined;
  return dense?.activation ?? "relu";
}

function getDropout(ir: NeuralNetworkIR): number {
  const drop = ir.architecture.layers.find((l) => l.kind === "dropout");
  return drop && drop.kind === "dropout" ? drop.rate : 0;
}

export function formatSearchResults(result: AutoSearchResult): string {
  const lines: string[] = [];
  lines.push(`\n  Auto-Architecture Search Results`);
  lines.push(`  ${"─".repeat(55)}`);
  lines.push(`  ${"Name".padEnd(10)} ${"Layers".padEnd(20)} ${"Probe Loss".padStart(12)} ${"Metric".padStart(10)}`);
  lines.push(`  ${"─".repeat(55)}`);

  for (let i = 0; i < result.candidates.length; i++) {
    const c = result.candidates[i]!;
    const marker = i === result.bestIdx ? " ◀ best" : "";
    lines.push(
      `  ${c.name.padEnd(10)} ${JSON.stringify(c.layers).padEnd(20)} ${c.probeLoss.toFixed(6).padStart(12)} ${c.probeMetric.toFixed(4).padStart(10)}${marker}`,
    );
  }

  lines.push(`\n  Selected: ${result.candidates[result.bestIdx]!.name} → ${JSON.stringify(result.bestLayers)}`);
  return lines.join("\n");
}
