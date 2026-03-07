/**
 * Feature importance via permutation importance.
 * Shuffles each feature column independently and measures the metric drop.
 */
import type { NeuralNetworkIR } from "./ir.js";

export interface FeatureImportance {
  feature: string;
  importance: number;
  rank: number;
}

export async function computeFeatureImportance(
  tf: typeof import("@tensorflow/tfjs"),
  model: ReturnType<typeof tf.sequential>,
  xTest: import("@tensorflow/tfjs").Tensor2D,
  yTest: import("@tensorflow/tfjs").Tensor,
  ir: NeuralNetworkIR,
): Promise<FeatureImportance[]> {
  const baseResult = model.evaluate(xTest, yTest as import("@tensorflow/tfjs").Tensor2D) as import("@tensorflow/tfjs").Scalar[];
  const baseMetric = Array.isArray(baseResult) ? baseResult[1]?.dataSync()[0] ?? 0 : 0;

  const xData = xTest.arraySync() as number[][];
  const numFeatures = ir.features.length;
  const importances: { feature: string; importance: number }[] = [];

  for (let f = 0; f < numFeatures; f++) {
    const shuffled = xData.map((row) => [...row]);

    const col = shuffled.map((row) => row[f]!);
    for (let i = col.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [col[i], col[j]] = [col[j]!, col[i]!];
    }
    for (let i = 0; i < shuffled.length; i++) {
      shuffled[i]![f] = col[i]!;
    }

    const xShuffled = tf.tensor2d(shuffled);
    const shuffledResult = model.evaluate(xShuffled, yTest as import("@tensorflow/tfjs").Tensor2D) as import("@tensorflow/tfjs").Scalar[];
    const shuffledMetric = Array.isArray(shuffledResult) ? shuffledResult[1]?.dataSync()[0] ?? 0 : 0;
    xShuffled.dispose();

    const drop = ir.task === "classification"
      ? baseMetric - shuffledMetric
      : shuffledMetric - baseMetric;

    importances.push({
      feature: ir.features[f] ?? `feature_${f}`,
      importance: Math.max(0, drop),
    });
  }

  importances.sort((a, b) => b.importance - a.importance);

  return importances.map((item, idx) => ({
    ...item,
    rank: idx + 1,
  }));
}

export function formatFeatureImportance(results: FeatureImportance[], ir: NeuralNetworkIR): string {
  const lines: string[] = [];
  const metricName = ir.task === "classification" ? "accuracy" : "MSE";
  const maxNameLen = Math.max(...results.map((r) => r.feature.length), 7);

  lines.push(`\n  Feature Importance (permutation, metric: ${metricName})`);
  lines.push(`  ${"─".repeat(maxNameLen + 30)}`);
  lines.push(`  ${"Rank".padEnd(6)} ${"Feature".padEnd(maxNameLen)} ${"Importance".padStart(12)}  Bar`);
  lines.push(`  ${"─".repeat(maxNameLen + 30)}`);

  const maxImp = Math.max(...results.map((r) => r.importance), 0.001);

  for (const r of results) {
    const barLen = Math.round((r.importance / maxImp) * 20);
    const bar = "█".repeat(barLen) + "░".repeat(20 - barLen);
    lines.push(
      `  ${String(r.rank).padEnd(6)} ${r.feature.padEnd(maxNameLen)} ${r.importance.toFixed(6).padStart(12)}  ${bar}`,
    );
  }

  return lines.join("\n");
}
