/**
 * NeuroLang In-Code Integration Examples
 *
 * Run: npx tsx examples/integration.ts
 */

import { nl } from "../src/template.js";

async function main() {
  console.log("═══════════════════════════════════════════════");
  console.log("  NeuroLang — In-Code Integration Demo");
  console.log("═══════════════════════════════════════════════\n");

  // ── 1. Tagged Template — One-liner ──────────────────────────
  console.log("▸ 1. Tagged template with built-in dataset\n");

  const result = await nl`Predict species from iris epochs 20`;

  console.log(`  accuracy: ${(result.accuracy! * 100).toFixed(1)}%`);
  console.log(`  epochs:   ${result.epochs}`);
  console.log(`  loss:     ${result.finalLoss.toFixed(4)}\n`);

  // ── 2. Interpolate Variables ────────────────────────────────
  console.log("▸ 2. Interpolate JS variables\n");

  const target = "species";
  const features = ["petal_length", "petal_width"];
  const epochs = 25;

  const r2 = await nl`Predict ${target} with ${features} from iris epochs ${epochs}`;

  console.log(`  accuracy: ${(r2.accuracy! * 100).toFixed(1)}%\n`);

  // ── 3. Inline Data (your own arrays) ────────────────────────
  console.log("▸ 3. Pass your own data (no CSV needed)\n");

  const customerData = [];
  for (let i = 0; i < 200; i++) {
    const age = 18 + Math.random() * 50;
    const spend = 100 + Math.random() * 900;
    const churned = (age > 40 && spend < 400) || Math.random() > 0.7 ? 1 : 0;
    customerData.push({ age, spend, churned });
  }

  const r3 = await nl`Predict churned with age & spend from ${customerData} epochs 15`;

  console.log(`  accuracy: ${(r3.accuracy! * 100).toFixed(1)}%`);
  console.log(`  samples:  ${customerData.length}\n`);

  // ── 4. Compile-only (no training) ──────────────────────────
  console.log("▸ 4. Compile-only — get IR + code\n");

  const compiled = nl.compile`Predict species from iris epochs 30`;

  console.log(`  task:     ${compiled.ir.task}`);
  console.log(`  layers:   ${compiled.ir.architecture.layers.length}`);
  console.log(`  code:     ${compiled.code.split("\n").length} lines of TF.js\n`);

  // ── 5. Generate code for other frameworks ───────────────────
  console.log("▸ 5. Generate code for PyTorch / Keras / JAX\n");

  const pyCode = nl.pytorch`Predict species from iris`;
  const kerasCode = nl.keras`Predict species from iris`;
  const jaxCode = nl.jax`Predict species from iris`;

  console.log(`  PyTorch:  ${pyCode.split("\n").length} lines`);
  console.log(`  Keras:    ${kerasCode.split("\n").length} lines`);
  console.log(`  JAX/Flax: ${jaxCode.split("\n").length} lines\n`);

  console.log("═══════════════════════════════════════════════");
  console.log("  All integration patterns working!");
  console.log("═══════════════════════════════════════════════");
}

main().catch(console.error);
