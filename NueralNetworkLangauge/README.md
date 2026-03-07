# NeuroLang

A declarative programming language that compiles into neural networks. Write **what** you want to predict — the compiler figures out **how**.

```
task classification
predict species
inputs petal_length petal_width sepal_length sepal_width
loss cross_entropy
optimizer adam
epochs 48
learn nonlinear
dataset iris.csv
```

This program compiles into a fully configured neural network, trains it on the Iris dataset, and achieves ~96% accuracy — in eight lines.

## Quick start

```bash
npm install
npm run compile -- examples/iris.nl --run          # train in TF.js
npm run compile -- examples/iris.nl --emit-code     # see generated code
npm run compile -- examples/iris.nl --target pytorch # emit PyTorch
```

### Web showcase (real compiler in-browser)

```bash
npm run bundle   # 48KB — bundles compiler for browser
npm run serve    # http://127.0.0.1:4173
```

## Language reference

Every program needs four required keywords. Everything else is optional.

### Required

| Keyword     | Values               | Purpose                        |
| ----------- | -------------------- | ------------------------------ |
| `task`      | `classification` / `regression` | What kind of prediction         |
| `predict`   | identifier           | Target column                  |
| `inputs`    | identifiers...       | Feature columns (space-separated) |
| `dataset`   | filename or `"path"` | Data source (`iris.csv`, `housing.csv` built-in) |

### Training

| Keyword         | Default      | Values / Example |
| --------------- | ------------ | ---------------- |
| `loss`          | auto-inferred | `mse`, `mae`, `huber`, `cross_entropy`, `binary_cross_entropy` |
| `optimizer`     | `adam`        | `adam`, `adamw`, `sgd`, `rmsprop` |
| `learning_rate` | `0.001`      | any float, e.g. `1e-5` |
| `epochs`        | `40`         | integer |
| `batch_size`    | `32`         | integer |
| `validate`      | `0.2`        | float (validation split fraction) |
| `early_stop`    | off          | patience integer, e.g. `8` |
| `lr_schedule`   | `none`       | `cosine`, `step`, `exponential` |

### Architecture

| Keyword       | Default          | Values / Example |
| ------------- | ---------------- | ---------------- |
| `learn`       | `nonlinear`      | `linear`, `nonlinear`, `deep` |
| `layers`      | auto-generated   | space-separated integers, e.g. `256 128 64` |
| `activation`  | `relu`           | `relu`, `sigmoid`, `tanh`, `selu`, `elu`, `gelu`, `leakyrelu` |
| `dropout`     | off              | float, e.g. `0.25` |
| `batch_norm`  | `false`          | `true` / `false` |

### Data & reproducibility

| Keyword     | Default | Values / Example |
| ----------- | ------- | ---------------- |
| `normalize` | `true`  | `true` / `false` |
| `split`     | `0.8`   | float (train/test split) |
| `seed`      | random  | integer for reproducibility |

### Output

| Keyword  | Default | Values / Example |
| -------- | ------- | ---------------- |
| `export` | off     | `"./model-dir"` (saves trained model to disk) |

## Compiler architecture

```
Source (.nl) → Lexer → Parser → AST → Analyzer → IR → Codegen
```

| Phase      | File              | What it does |
| ---------- | ----------------- | ------------ |
| **Lexer**  | `src/lexer.ts`    | Tokenizes source into keywords, identifiers, numbers, strings |
| **Parser** | `src/parser.ts`   | Builds AST from token stream (recursive descent) |
| **Analyzer** | `src/analyzer.ts` | Validates, infers defaults, builds typed IR |
| **Codegen**  | `src/codegen.ts`  | Emits TensorFlow.js or PyTorch code |
| **Runtime**  | `src/runtime.ts`  | Trains model using TF.js with early stopping, seeded RNG |
| **Diagnostics** | `src/diagnostics.ts` | Rust-style error messages with source context |

## Code generation targets

**TensorFlow.js** (default) — generates a complete runnable TypeScript program with:
- `tf.sequential` model construction
- Automatic data loading, normalization, and train/test split
- Early stopping, learning rate schedules, model export

**PyTorch** — generates a complete Python program with:
- `nn.Module` class definition
- `pandas` + `sklearn` data pipeline
- `torch.optim.lr_scheduler` support
- Manual early stopping, model checkpointing

## Built-in datasets

- **iris** (150 samples, 4 features, 3 classes) — classification
- **housing** (120 samples, 5 features) — regression

The compiler resolves `iris.csv` / `housing.csv` to embedded data. For custom CSVs, the runtime loads from disk.

## CLI options

```
npx tsx src/cli.ts <file.nl> [options]

Options:
  --run           Train the model after compiling
  --emit-code     Print generated source code
  --target <t>    Code generation target: tensorflow | pytorch | summary
  --tokens        Show lexer output
  --ast           Show parser output
  --ir            Show intermediate representation
  --timings       Show compilation phase timings
  --verbose       Enable all debug output
```

## Tests

```bash
npm test              # runs 133 tests across 6 suites
npm run typecheck     # zero-error TypeScript strict mode
```

## Examples

| File                        | What it demonstrates |
| --------------------------- | -------------------- |
| `examples/iris.nl`          | Standard classification |
| `examples/housing.nl`       | Regression with custom layers |
| `examples/minimal.nl`       | Bare minimum (4 keywords) |
| `examples/deep.nl`          | Deep mode + dropout |
| `examples/linear.nl`        | Linear regression (no hidden layers) |
| `examples/robust.nl`        | Huber loss + AdamW + batch norm |
| `examples/quick-classify.nl`| RMSProp + step LR schedule |
| `examples/enterprise.nl`    | Every feature at once |

## Project structure

```
src/
  tokens.ts       Token types and keyword set
  ast.ts          AST node types
  ir.ts           Intermediate representation types
  diagnostics.ts  Error/warning system
  lexer.ts        Tokenizer
  parser.ts       Recursive descent parser
  analyzer.ts     Semantic analysis + IR builder
  codegen.ts      TF.js + PyTorch code generators
  datasets.ts     Built-in datasets + CSV loader
  runtime.ts      TF.js training engine
  compiler.ts     Pipeline orchestrator
  cli.ts          Command-line interface
  browser.ts      Browser bundle entry point
  index.ts        Public API exports

tests/
  compiler.test.ts   133 unit + integration tests

examples/
  *.nl               Sample programs

index.html           Web showcase (uses real compiler)
styles.css           Black/white/yellow theme
script.js            Showcase logic
neurolang.bundle.js  Browser bundle (48KB)
```
