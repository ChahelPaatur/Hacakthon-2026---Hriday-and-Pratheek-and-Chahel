import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { compile } from "../src/compiler.js";
import { resolveDataset } from "../src/datasets.js";
import { nl, _resolveInterpolation } from "../src/template.js";

describe("tagged template interpolation", () => {
  function tag(strings: TemplateStringsArray, ...values: unknown[]) {
    return _resolveInterpolation(strings, values);
  }

  it("passes through plain text", () => {
    const result = tag`Predict species from iris`;
    assert.equal(result.source, "Predict species from iris");
    assert.equal(result.cleanups.length, 0);
  });

  it("interpolates string variables", () => {
    const target = "price";
    const result = tag`Predict ${target} from housing`;
    assert.equal(result.source, "Predict price from housing");
  });

  it("interpolates number variables", () => {
    const epochs = 50;
    const result = tag`Predict species from iris epochs ${epochs}`;
    assert.equal(result.source, "Predict species from iris epochs 50");
  });

  it("interpolates string[] as feature list with &", () => {
    const features = ["size", "bedrooms", "bathrooms"];
    const result = tag`Predict price with ${features} from housing`;
    assert.equal(result.source, "Predict price with size & bedrooms & bathrooms from housing");
  });

  it("interpolates array of objects as inline dataset", () => {
    const data = [
      { x: 1, y: 2, z: 0 },
      { x: 3, y: 4, z: 1 },
    ];
    const result = tag`Predict z with x & y from ${data}`;
    assert.ok(result.source.includes("__inline_"));
    assert.equal(result.cleanups.length, 1);

    const dsName = result.source.match(/__inline_\d+/)![0];
    const ds = resolveDataset(dsName);
    assert.ok(ds);
    assert.equal(ds!.targetName, "z");
    assert.deepEqual(ds!.featureNames, ["x", "y"]);

    for (const fn of result.cleanups) fn();
    assert.equal(resolveDataset(dsName), null);
  });

  it("interpolates multiple variables at once", () => {
    const target = "species";
    const features = ["a", "b"];
    const epochs = 30;
    const result = tag`Predict ${target} with ${features} from iris epochs ${epochs}`;
    assert.equal(result.source, "Predict species with a & b from iris epochs 30");
  });

  it("compiles interpolated source correctly", () => {
    const target = "species";
    const features = ["petal_length", "petal_width"];
    const result = tag`Predict ${target} with ${features} from iris`;
    const compiled = compile(result.source);
    assert.equal(compiled.ir.task, "classification");
    assert.equal(compiled.ir.target, "species");
    assert.ok(compiled.ir.features.includes("petal_length"));
  });

  it("compiles with inline dataset", () => {
    const data = [
      { age: 20, score: 80, pass: 1 },
      { age: 30, score: 60, pass: 0 },
      { age: 25, score: 90, pass: 1 },
      { age: 40, score: 50, pass: 0 },
    ];
    const result = tag`Predict pass with age & score from ${data}`;
    const compiled = compile(result.source);
    assert.equal(compiled.ir.target, "pass");
    assert.deepEqual(compiled.ir.features, ["age", "score"]);
    for (const fn of result.cleanups) fn();
  });

  it("nl.compile produces compilation result", () => {
    const result = nl.compile`Predict species with petal_length & petal_width from iris`;
    assert.ok(result.ir);
    assert.ok(result.code);
    assert.equal(result.ir.task, "classification");
  });

  it("nl.pytorch generates Python code", () => {
    const code = nl.pytorch`Predict species from iris`;
    assert.ok(code.includes("torch"));
    assert.ok(code.includes("nn.Module"));
  });

  it("nl.keras generates Keras code", () => {
    const code = nl.keras`Predict species from iris`;
    assert.ok(code.includes("keras") || code.includes("Sequential"));
  });

  it("nl.jax generates JAX code", () => {
    const code = nl.jax`Predict species from iris`;
    assert.ok(code.includes("jax") || code.includes("flax"));
  });

  it("nl.generate(target) is a factory for code gen tags", () => {
    const gen = nl.generate("pytorch");
    const code = gen`Predict species from iris`;
    assert.ok(code.includes("torch"));
  });
});
