import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { Lexer } from "../src/lexer.js";
import { Parser } from "../src/parser.js";
import { Analyzer } from "../src/analyzer.js";
import { compile, CompilationError } from "../src/compiler.js";
import { emitTensorFlow, emitPyTorch, emitKeras, emitJAX, emitSummary } from "../src/codegen.js";
import { resolveDataset, loadIris, loadHousing, loadTitanic, loadWine, loadDigits, loadSequences, loadTimeseries, parseCSV } from "../src/datasets.js";
import { DiagnosticCollector } from "../src/diagnostics.js";
import { TokenType } from "../src/tokens.js";
import { getCandidates, buildCandidateIR, formatSearchResults } from "../src/autoarch.js";
import { getTuneConfig, buildTrialIR, totalTrials, formatTuneResults } from "../src/tune.js";
import { formatFeatureImportance } from "../src/explain.js";
import { parseNatural, isNaturalSyntax } from "../src/natural.js";
import { describeDataset, formatDataPreview, formatDescription, formatCorrelation, formatSample } from "../src/inspect.js";
import { registerInlineDataset, unregisterInlineDataset, datasetFromObjects } from "../src/datasets.js";

// ─── Helpers ────────────────────────────────────────────────────────

function lex(source: string) { return new Lexer(source).tokenize(); }
function parse(source: string) { return new Parser(lex(source)).parse(); }
function analyze(source: string) { return new Analyzer().analyze(parse(source)); }

const MINIMAL = `task regression\npredict y\ninputs x\ndataset d.csv`;
const IRIS = `task classification\npredict species\ninputs a b c d\nloss cross_entropy\noptimizer adam\nepochs 48\nlearn nonlinear\ndataset iris.csv`;
const HOUSING = `task regression\npredict price\ninputs size beds baths age zip\nloss mse\nepochs 60\nlayers 128 64 32\ndataset housing.csv`;

// ═══════════════════════════════════════════════════════════════════
//  LEXER
// ═══════════════════════════════════════════════════════════════════

describe("Lexer", () => {
  it("tokenizes minimal program", () => {
    const tokens = lex(MINIMAL);
    const kws = tokens.filter((t) => t.type === TokenType.KEYWORD).map((t) => t.value);
    assert.deepEqual(kws, ["task", "predict", "inputs", "dataset"]);
  });

  it("produces EOF as last token", () => {
    const tokens = lex(MINIMAL);
    assert.equal(tokens[tokens.length - 1]!.type, TokenType.EOF);
  });

  it("handles empty input", () => {
    const tokens = lex("");
    assert.equal(tokens.length, 1);
    assert.equal(tokens[0]!.type, TokenType.EOF);
  });

  it("handles whitespace-only input", () => {
    const tokens = lex("   \t\t   ");
    assert.equal(tokens.length, 1);
    assert.equal(tokens[0]!.type, TokenType.EOF);
  });

  it("handles comment-only input", () => {
    const tokens = lex("# just a comment");
    assert.equal(tokens.length, 1);
    assert.equal(tokens[0]!.type, TokenType.EOF);
  });

  it("recognizes all keywords including tune", () => {
    const keywords = [
      "task", "predict", "inputs", "dataset", "loss", "learn", "optimizer",
      "epochs", "layers", "activation", "batch_norm", "batch_size", "dropout",
      "learning_rate", "early_stop", "lr_schedule", "seed", "normalize",
      "split", "validate", "export", "tune",
    ];
    for (const kw of keywords) {
      const tokens = lex(`${kw} value`);
      assert.ok(
        tokens.some((t) => t.type === TokenType.KEYWORD && t.value === kw),
        `'${kw}' should be recognized as KEYWORD`,
      );
    }
  });

  it("distinguishes identifiers from keywords", () => {
    const tokens = lex("task regression");
    assert.equal(tokens[0]!.type, TokenType.KEYWORD);
    assert.equal(tokens[1]!.type, TokenType.IDENT);
  });

  it("parses integers", () => {
    const tokens = lex("epochs 100");
    const num = tokens.find((t) => t.type === TokenType.NUMBER)!;
    assert.equal(num.value, "100");
    assert.equal(parseFloat(num.value), 100);
  });

  it("parses floats", () => {
    const tokens = lex("dropout 0.25");
    const num = tokens.find((t) => t.type === TokenType.NUMBER)!;
    assert.equal(num.value, "0.25");
  });

  it("parses scientific notation (1e-5)", () => {
    const tokens = lex("learning_rate 1e-5");
    const num = tokens.find((t) => t.type === TokenType.NUMBER)!;
    assert.equal(num.value, "1e-5");
    assert.ok(Math.abs(parseFloat(num.value) - 0.00001) < 1e-10);
  });

  it("parses scientific notation (2.5E3)", () => {
    const tokens = lex("epochs 2.5E3");
    const num = tokens.find((t) => t.type === TokenType.NUMBER)!;
    assert.equal(parseFloat(num.value), 2500);
  });

  it("parses double-quoted strings", () => {
    const tokens = lex(`dataset "path/to/data.csv"`);
    const str = tokens.find((t) => t.type === TokenType.STRING)!;
    assert.equal(str.value, "path/to/data.csv");
  });

  it("parses single-quoted strings", () => {
    const tokens = lex(`dataset 'file.csv'`);
    const str = tokens.find((t) => t.type === TokenType.STRING)!;
    assert.equal(str.value, "file.csv");
  });

  it("handles escape sequences in strings", () => {
    const tokens = lex(`dataset "path\\nwith\\ttabs"`);
    const str = tokens.find((t) => t.type === TokenType.STRING)!;
    assert.equal(str.value, "path\nwith\ttabs");
  });

  it("skips # comments", () => {
    const tokens = lex("# this is a comment\ntask regression");
    assert.ok(!tokens.some((t) => t.value.includes("comment")));
    assert.ok(tokens.some((t) => t.value === "task"));
  });

  it("skips inline comments", () => {
    const tokens = lex("task regression # inline comment");
    assert.equal(tokens.filter((t) => t.type === TokenType.KEYWORD).length, 1);
    assert.equal(tokens.filter((t) => t.type === TokenType.IDENT).length, 1);
  });

  it("collapses consecutive newlines", () => {
    const tokens = lex("task regression\n\n\n\n\npredict price");
    assert.equal(tokens.filter((t) => t.type === TokenType.NEWLINE).length, 1);
  });

  it("strips leading/trailing newlines", () => {
    const tokens = lex("\n\ntask regression\n\n");
    assert.notEqual(tokens[0]!.type, TokenType.NEWLINE);
  });

  it("tracks line numbers correctly", () => {
    const tokens = lex("task regression\npredict price\ninputs x y");
    const predict = tokens.find((t) => t.value === "predict")!;
    const inputs = tokens.find((t) => t.value === "inputs")!;
    assert.equal(predict.line, 2);
    assert.equal(inputs.line, 3);
  });

  it("tracks column numbers correctly", () => {
    const tokens = lex("task regression");
    assert.equal(tokens[0]!.column, 1);
    assert.equal(tokens[1]!.column, 6);
  });

  it("measures token length", () => {
    const tokens = lex("task classification");
    assert.equal(tokens[0]!.length, 4);
    assert.equal(tokens[1]!.length, 14);
  });

  it("handles dotted identifiers (file.csv)", () => {
    const tokens = lex("dataset iris.csv");
    const ident = tokens.find((t) => t.type === TokenType.IDENT)!;
    assert.equal(ident.value, "iris.csv");
  });

  it("reports unexpected characters via diagnostics", () => {
    const collector = new DiagnosticCollector();
    lex.call(null, "");
    new Lexer("task @bad", collector).tokenize();
    assert.ok(collector.hasErrors);
    assert.equal(collector.errors[0]!.code, "NL0001");
  });

  it("reports unterminated strings", () => {
    const collector = new DiagnosticCollector();
    new Lexer(`dataset "unterminated`, collector).tokenize();
    assert.ok(collector.hasErrors);
    assert.equal(collector.errors[0]!.code, "NL0002");
  });
});

// ═══════════════════════════════════════════════════════════════════
//  PARSER
// ═══════════════════════════════════════════════════════════════════

describe("Parser", () => {
  it("parses minimal program", () => {
    const ast = parse(MINIMAL);
    assert.equal(ast.type, "Program");
    assert.equal(ast.body.length, 4);
  });

  it("preserves keyword names in statements", () => {
    const ast = parse(IRIS);
    const kws = ast.body.map((s) => s.keyword);
    assert.ok(kws.includes("task"));
    assert.ok(kws.includes("predict"));
    assert.ok(kws.includes("inputs"));
    assert.ok(kws.includes("dataset"));
    assert.ok(kws.includes("loss"));
    assert.ok(kws.includes("optimizer"));
    assert.ok(kws.includes("epochs"));
    assert.ok(kws.includes("learn"));
  });

  it("parses multiple values per statement", () => {
    const ast = parse("inputs x y z w v");
    assert.equal(ast.body[0]!.values.length, 5);
  });

  it("parses identifier values", () => {
    const ast = parse("task regression");
    assert.equal(ast.body[0]!.values[0]!.type, "Identifier");
    assert.equal(ast.body[0]!.values[0]!.value, "regression");
  });

  it("parses number values", () => {
    const ast = parse("epochs 50");
    const val = ast.body[0]!.values[0]!;
    assert.equal(val.type, "Number");
    if (val.type === "Number") assert.equal(val.value, 50);
  });

  it("parses string values", () => {
    const ast = parse(`dataset "my file.csv"`);
    const val = ast.body[0]!.values[0]!;
    assert.equal(val.type, "String");
    if (val.type === "String") assert.equal(val.value, "my file.csv");
  });

  it("parses mixed value types on one line", () => {
    const ast = parse("layers 128 64 32");
    assert.equal(ast.body[0]!.values.length, 3);
    for (const v of ast.body[0]!.values) {
      assert.equal(v.type, "Number");
    }
  });

  it("allows keywords used as value names", () => {
    const ast = parse("predict loss");
    assert.equal(ast.body[0]!.values[0]!.type, "Identifier");
    assert.equal(ast.body[0]!.values[0]!.value, "loss");
  });

  it("sets source location on statements", () => {
    const ast = parse("task regression\npredict price");
    assert.equal(ast.body[0]!.location.line, 1);
    assert.equal(ast.body[1]!.location.line, 2);
  });

  it("recovers from errors and continues", () => {
    const collector = new DiagnosticCollector();
    const tokens = new Lexer("123 bad\ntask regression", collector).tokenize();
    const ast = new Parser(tokens, collector).parse();
    assert.ok(ast.body.some((s) => s.keyword === "task"));
  });

  it("reports errors for non-keyword start", () => {
    const collector = new DiagnosticCollector();
    const tokens = new Lexer("notakeyword value", collector).tokenize();
    new Parser(tokens, collector).parse();
    assert.ok(collector.hasErrors);
  });

  it("reports empty keyword (no values)", () => {
    const collector = new DiagnosticCollector();
    const tokens = new Lexer("task\npredict species", collector).tokenize();
    new Parser(tokens, collector).parse();
    assert.ok(collector.hasErrors);
  });

  it("handles empty program", () => {
    const ast = parse("");
    assert.equal(ast.body.length, 0);
  });

  it("handles all-comment program", () => {
    const ast = parse("# comment 1\n# comment 2\n# comment 3");
    assert.equal(ast.body.length, 0);
  });
});

// ═══════════════════════════════════════════════════════════════════
//  ANALYZER
// ═══════════════════════════════════════════════════════════════════

describe("Analyzer", () => {
  it("builds IR for classification", () => {
    const ir = analyze(IRIS)!;
    assert.equal(ir.task, "classification");
    assert.equal(ir.target, "species");
    assert.equal(ir.features.length, 4);
    assert.equal(ir.architecture.outputSize, 3);
    assert.equal(ir.training.loss, "categoricalCrossentropy");
    assert.equal(ir.training.optimizer.name, "adam");
    assert.equal(ir.training.epochs, 48);
    assert.equal(ir.meta.learnMode, "nonlinear");
  });

  it("builds IR for regression", () => {
    const ir = analyze(HOUSING)!;
    assert.equal(ir.task, "regression");
    assert.equal(ir.architecture.outputSize, 1);
    assert.equal(ir.training.loss, "meanSquaredError");
  });

  it("infers default loss for classification", () => {
    const ir = analyze(`task classification\npredict y\ninputs x\ndataset d.csv`)!;
    assert.equal(ir.training.loss, "categoricalCrossentropy");
  });

  it("infers default loss for regression", () => {
    const ir = analyze(MINIMAL)!;
    assert.equal(ir.training.loss, "meanSquaredError");
  });

  it("infers default optimizer (adam)", () => {
    const ir = analyze(MINIMAL)!;
    assert.equal(ir.training.optimizer.name, "adam");
  });

  it("infers default epochs (40)", () => {
    const ir = analyze(MINIMAL)!;
    assert.equal(ir.training.epochs, 40);
  });

  it("infers default batch size (32)", () => {
    const ir = analyze(MINIMAL)!;
    assert.equal(ir.training.batchSize, 32);
  });

  it("infers default learn mode (nonlinear)", () => {
    const ir = analyze(MINIMAL)!;
    assert.equal(ir.meta.learnMode, "nonlinear");
  });

  it("infers default normalize (true)", () => {
    const ir = analyze(MINIMAL)!;
    assert.equal(ir.preprocessing.normalize, true);
  });

  it("infers default split (0.8)", () => {
    const ir = analyze(MINIMAL)!;
    assert.equal(ir.preprocessing.trainTestSplit, 0.8);
  });

  // --- explicit keywords ---

  it("handles explicit loss mse", () => {
    const ir = analyze(`${MINIMAL}\nloss mse`)!;
    assert.equal(ir.training.loss, "meanSquaredError");
  });

  it("handles explicit loss mae", () => {
    const ir = analyze(`${MINIMAL}\nloss mae`)!;
    assert.equal(ir.training.loss, "meanAbsoluteError");
  });

  it("handles explicit loss huber", () => {
    const ir = analyze(`${MINIMAL}\nloss huber`)!;
    assert.equal(ir.training.loss, "huberLoss");
  });

  it("handles optimizer sgd", () => {
    const ir = analyze(`${MINIMAL}\noptimizer sgd`)!;
    assert.equal(ir.training.optimizer.name, "sgd");
  });

  it("handles optimizer rmsprop", () => {
    const ir = analyze(`${MINIMAL}\noptimizer rmsprop`)!;
    assert.equal(ir.training.optimizer.name, "rmsprop");
  });

  it("handles optimizer adamw", () => {
    const ir = analyze(`${MINIMAL}\noptimizer adamw`)!;
    assert.equal(ir.training.optimizer.name, "adamw");
  });

  it("handles custom learning_rate", () => {
    const ir = analyze(`${MINIMAL}\nlearning_rate 0.01`)!;
    assert.equal(ir.training.optimizer.learningRate, 0.01);
  });

  it("handles custom epochs", () => {
    const ir = analyze(`${MINIMAL}\nepochs 200`)!;
    assert.equal(ir.training.epochs, 200);
  });

  it("handles custom batch_size", () => {
    const ir = analyze(`${MINIMAL}\nbatch_size 64`)!;
    assert.equal(ir.training.batchSize, 64);
  });

  it("handles custom layers", () => {
    const ir = analyze(`${MINIMAL}\nlayers 256 128 64`)!;
    const denses = ir.architecture.layers.filter((l) => l.kind === "dense");
    assert.equal(denses.length, 4); // 3 hidden + output
  });

  it("handles dropout", () => {
    const ir = analyze(`${MINIMAL}\ndropout 0.3`)!;
    assert.ok(ir.architecture.layers.some((l) => l.kind === "dropout"));
  });

  it("handles batch_norm true", () => {
    const ir = analyze(`${MINIMAL}\nbatch_norm true`)!;
    assert.equal(ir.architecture.batchNorm, true);
    assert.ok(ir.architecture.layers.some((l) => l.kind === "batchnorm"));
  });

  it("handles batch_norm false", () => {
    const ir = analyze(`${MINIMAL}\nbatch_norm false`)!;
    assert.equal(ir.architecture.batchNorm, false);
  });

  it("handles early_stop", () => {
    const ir = analyze(`${MINIMAL}\nearly_stop 10`)!;
    assert.ok(ir.training.earlyStop);
    assert.equal(ir.training.earlyStop!.patience, 10);
  });

  it("handles lr_schedule cosine", () => {
    const ir = analyze(`${MINIMAL}\nlr_schedule cosine`)!;
    assert.equal(ir.training.lrSchedule.type, "cosine");
  });

  it("handles lr_schedule step", () => {
    const ir = analyze(`${MINIMAL}\nlr_schedule step`)!;
    assert.equal(ir.training.lrSchedule.type, "step");
  });

  it("handles seed", () => {
    const ir = analyze(`${MINIMAL}\nseed 42`)!;
    assert.equal(ir.preprocessing.seed, 42);
  });

  it("handles normalize false", () => {
    const ir = analyze(`${MINIMAL}\nnormalize false`)!;
    assert.equal(ir.preprocessing.normalize, false);
  });

  it("handles split", () => {
    const ir = analyze(`${MINIMAL}\nsplit 0.7`)!;
    assert.equal(ir.preprocessing.trainTestSplit, 0.7);
  });

  it("handles validate", () => {
    const ir = analyze(`${MINIMAL}\nvalidate 0.2`)!;
    assert.equal(ir.training.validationSplit, 0.2);
  });

  it("handles export path", () => {
    const ir = analyze(`${MINIMAL}\nexport "model-dir"`)!;
    assert.equal(ir.output.exportPath, "model-dir");
  });

  it("handles activation relu", () => {
    const ir = analyze(`${MINIMAL}\nactivation relu`)!;
    const dense = ir.architecture.layers.find((l) => l.kind === "dense" && (l as any).activation !== "linear")! as any;
    assert.equal(dense.activation, "relu");
  });

  // --- learn modes ---

  it("learn linear produces no hidden layers", () => {
    const ir = analyze(`${MINIMAL}\nlearn linear`)!;
    const denses = ir.architecture.layers.filter((l) => l.kind === "dense");
    assert.equal(denses.length, 1); // output only
  });

  it("learn nonlinear produces 2 hidden layers for regression", () => {
    const ir = analyze(`${MINIMAL}\nlearn nonlinear`)!;
    const denses = ir.architecture.layers.filter((l) => l.kind === "dense");
    assert.equal(denses.length, 3); // 2 hidden + output
  });

  it("learn deep produces 4+ hidden layers", () => {
    const ir = analyze(`task classification\npredict y\ninputs x1 x2\ndataset d.csv\nlearn deep`)!;
    const denses = ir.architecture.layers.filter((l) => l.kind === "dense");
    assert.ok(denses.length >= 5); // 4 hidden + output
  });

  it("learn auto is accepted", () => {
    const ir = analyze(`${MINIMAL}\nlearn auto`)!;
    assert.equal(ir.meta.learnMode, "auto");
  });

  it("learn auto produces hidden layers", () => {
    const ir = analyze(`${MINIMAL}\nlearn auto`)!;
    const denses = ir.architecture.layers.filter((l) => l.kind === "dense");
    assert.ok(denses.length >= 2);
  });

  it("explicit layers override learn mode", () => {
    const ir = analyze(`${MINIMAL}\nlearn deep\nlayers 32 16`)!;
    const denses = ir.architecture.layers.filter((l) => l.kind === "dense");
    assert.equal(denses.length, 3); // 2 specified + output
    assert.equal((denses[0] as any).units, 32);
  });

  // --- tune keyword ---

  it("handles tune true", () => {
    const ir = analyze(`${MINIMAL}\ntune true`)!;
    assert.equal(ir.training.tune, true);
  });

  it("handles tune false", () => {
    const ir = analyze(`${MINIMAL}\ntune false`)!;
    assert.equal(ir.training.tune, false);
  });

  it("tune defaults to undefined when absent", () => {
    const ir = analyze(MINIMAL)!;
    assert.equal(ir.training.tune, undefined);
  });

  // --- parameter counting ---

  it("counts parameters correctly for simple model", () => {
    const ir = analyze(`task regression\npredict y\ninputs x1 x2\ndataset d.csv\nlayers 4`)!;
    assert.equal(ir.meta.parameterCount, 17);
  });

  // --- output size inference ---

  it("infers 3 classes for iris", () => {
    const ir = analyze(`task classification\npredict species\ninputs a b c d\ndataset iris.csv`)!;
    assert.equal(ir.architecture.outputSize, 3);
  });

  it("infers 2 classes for titanic", () => {
    const ir = analyze(`task classification\npredict survived\ninputs a b c\ndataset titanic.csv`)!;
    assert.equal(ir.architecture.outputSize, 2);
  });

  it("infers 3 classes for wine", () => {
    const ir = analyze(`task classification\npredict cultivar\ninputs a b c\ndataset wine.csv`)!;
    assert.equal(ir.architecture.outputSize, 3);
  });

  it("infers 10 classes for digits", () => {
    const ir = analyze(`task classification\npredict digit\ninputs a b\ndataset digits.csv`)!;
    assert.equal(ir.architecture.outputSize, 10);
  });

  it("infers 10 classes for mnist", () => {
    const ir = analyze(`task classification\npredict digit\ninputs a b\ndataset mnist.csv`)!;
    assert.equal(ir.architecture.outputSize, 10);
  });

  it("infers 1 output for regression", () => {
    const ir = analyze(MINIMAL)!;
    assert.equal(ir.architecture.outputSize, 1);
  });

  // --- error cases ---

  it("returns null for missing task", () => {
    const c = new DiagnosticCollector();
    const ir = new Analyzer(c).analyze(parse("predict y\ninputs x\ndataset d.csv"));
    assert.equal(ir, null);
    assert.ok(c.hasErrors);
  });

  it("returns null for missing predict", () => {
    const c = new DiagnosticCollector();
    const ir = new Analyzer(c).analyze(parse("task regression\ninputs x\ndataset d.csv"));
    assert.equal(ir, null);
  });

  it("returns null for missing inputs", () => {
    const c = new DiagnosticCollector();
    const ir = new Analyzer(c).analyze(parse("task regression\npredict y\ndataset d.csv"));
    assert.equal(ir, null);
  });

  it("returns null for missing dataset", () => {
    const c = new DiagnosticCollector();
    const ir = new Analyzer(c).analyze(parse("task regression\npredict y\ninputs x"));
    assert.equal(ir, null);
  });

  it("rejects invalid task type", () => {
    const c = new DiagnosticCollector();
    new Analyzer(c).analyze(parse("task unknown\npredict y\ninputs x\ndataset d.csv"));
    assert.ok(c.errors.some((e) => e.code === "NL0011"));
  });

  it("rejects invalid optimizer", () => {
    const c = new DiagnosticCollector();
    new Analyzer(c).analyze(parse(`${MINIMAL}\noptimizer unknown`));
    assert.ok(c.errors.some((e) => e.code === "NL0013"));
  });

  it("rejects invalid loss", () => {
    const c = new DiagnosticCollector();
    new Analyzer(c).analyze(parse(`${MINIMAL}\nloss unknown`));
    assert.ok(c.errors.some((e) => e.code === "NL0015"));
  });

  it("rejects invalid activation", () => {
    const c = new DiagnosticCollector();
    new Analyzer(c).analyze(parse(`${MINIMAL}\nactivation unknown`));
    assert.ok(c.errors.some((e) => e.code === "NL0014"));
  });

  it("rejects invalid learn mode", () => {
    const c = new DiagnosticCollector();
    new Analyzer(c).analyze(parse(`${MINIMAL}\nlearn unknown`));
    assert.ok(c.errors.some((e) => e.code === "NL0012"));
  });

  it("warns on duplicate keywords", () => {
    const c = new DiagnosticCollector();
    new Analyzer(c).analyze(parse(`${MINIMAL}\nepochs 10\nepochs 20`));
    assert.ok(c.warnings.length > 0);
  });

  // --- architecture types ---

  it("architecture cnn creates conv2d layers", () => {
    const ir = analyze(`task classification\npredict digit\ninputs ${Array.from({length: 64}, (_, i) => `p${i}`).join(" ")}\ndataset digits\narchitecture cnn`)!;
    assert.equal(ir.architecture.type, "cnn");
    assert.ok(ir.architecture.layers.some((l) => l.kind === "conv2d"));
    assert.ok(ir.architecture.layers.some((l) => l.kind === "maxpool2d"));
    assert.ok(ir.architecture.layers.some((l) => l.kind === "flatten"));
    assert.ok(ir.architecture.inputShape);
  });

  it("architecture lstm creates LSTM layers", () => {
    const ir = analyze(`task classification\npredict cls\ninputs ${Array.from({length: 10}, (_, i) => `t${i}`).join(" ")}\ndataset sequences\narchitecture lstm`)!;
    assert.equal(ir.architecture.type, "rnn");
    assert.ok(ir.architecture.layers.some((l) => l.kind === "lstm"));
    assert.ok(ir.architecture.inputShape);
  });

  it("architecture gru creates GRU layers", () => {
    const ir = analyze(`task classification\npredict cls\ninputs ${Array.from({length: 10}, (_, i) => `t${i}`).join(" ")}\ndataset sequences\narchitecture gru`)!;
    assert.equal(ir.architecture.type, "rnn");
    assert.ok(ir.architecture.layers.some((l) => l.kind === "gru"));
  });

  it("architecture autoencoder creates encoder-decoder pattern", () => {
    const ir = analyze(`task regression\npredict y\ninputs a b c d e f g h\ndataset d.csv\narchitecture autoencoder`)!;
    assert.equal(ir.architecture.type, "autoencoder");
    const denses = ir.architecture.layers.filter((l) => l.kind === "dense");
    assert.ok(denses.length >= 4);
  });

  it("infers cnn from filters keyword", () => {
    const ir = analyze(`task classification\npredict digit\ninputs ${Array.from({length: 64}, (_, i) => `p${i}`).join(" ")}\ndataset digits\nfilters 16 32`)!;
    assert.equal(ir.architecture.type, "cnn");
  });

  it("infers rnn from sequence_length keyword", () => {
    const ir = analyze(`task classification\npredict cls\ninputs ${Array.from({length: 10}, (_, i) => `t${i}`).join(" ")}\ndataset sequences\nsequence_length 10`)!;
    assert.equal(ir.architecture.type, "rnn");
  });

  it("bidirectional wraps rnn layer", () => {
    const ir = analyze(`task classification\npredict cls\ninputs ${Array.from({length: 10}, (_, i) => `t${i}`).join(" ")}\ndataset sequences\narchitecture lstm\nbidirectional true`)!;
    assert.ok(ir.architecture.layers.some((l) => l.kind === "bidirectional"));
  });

  it("custom filters for cnn", () => {
    const ir = analyze(`task classification\npredict digit\ninputs ${Array.from({length: 64}, (_, i) => `p${i}`).join(" ")}\ndataset digits\narchitecture cnn\nfilters 16 32 64`)!;
    const convs = ir.architecture.layers.filter((l) => l.kind === "conv2d");
    assert.equal(convs.length, 3);
  });

  // --- ResNet architecture ---

  it("architecture resnet creates residual blocks", () => {
    const src = `task classification\npredict digit\ninputs ${Array.from({length: 64}, (_, i) => `p${i}`).join(" ")}\ndataset digits\narchitecture resnet`;
    const ir = analyze(src)!;
    assert.equal(ir.architecture.type, "resnet");
    assert.ok(ir.architecture.layers.some((l) => l.kind === "residual"));
    assert.ok(ir.architecture.layers.some((l) => l.kind === "globalavgpool2d"));
  });

  // --- Transformer architecture ---

  it("architecture transformer creates attention layers", () => {
    const src = `task classification\npredict cls\ninputs ${Array.from({length: 10}, (_, i) => `t${i}`).join(" ")}\ndataset sequences\narchitecture transformer`;
    const ir = analyze(src)!;
    assert.equal(ir.architecture.type, "transformer");
    assert.ok(ir.architecture.layers.some((l) => l.kind === "multihead_attention"));
    assert.ok(ir.architecture.layers.some((l) => l.kind === "layernorm"));
  });

  // --- Transfer learning ---

  it("pretrained mobilenet sets pretrained field", () => {
    const src = `task classification\npredict cls\ninputs ${Array.from({length: 64}, (_, i) => `p${i}`).join(" ")}\ndataset digits\npretrained mobilenet`;
    const ir = analyze(src)!;
    assert.equal(ir.architecture.pretrained, "mobilenet");
  });

  it("pretrained resnet50 sets arch type to resnet", () => {
    const src = `task classification\npredict cls\ninputs ${Array.from({length: 64}, (_, i) => `p${i}`).join(" ")}\ndataset digits\npretrained resnet50`;
    const ir = analyze(src)!;
    assert.equal(ir.architecture.pretrained, "resnet50");
    assert.equal(ir.architecture.type, "resnet");
  });

  it("freeze_layers sets freeze count", () => {
    const src = `task classification\npredict cls\ninputs ${Array.from({length: 64}, (_, i) => `p${i}`).join(" ")}\ndataset digits\npretrained mobilenet\nfreeze_layers 10`;
    const ir = analyze(src)!;
    assert.equal(ir.architecture.freezeLayers, 10);
  });

  // --- Export formats ---

  it("export_format onnx sets export format", () => {
    const src = `task classification\npredict species\ninputs a b c d\ndataset iris\nexport_format onnx tflite`;
    const ir = analyze(src)!;
    assert.ok(ir.output.exportFormats);
    assert.ok(ir.output.exportFormats!.includes("onnx"));
    assert.ok(ir.output.exportFormats!.includes("tflite"));
  });

  // --- Multi-output ---

  it("targets keyword sets multiple targets", () => {
    const src = `task regression\npredict price\ninputs a b c\ndataset d.csv\ntargets price category`;
    const ir = analyze(src)!;
    assert.ok(ir.targets);
    assert.deepEqual(ir.targets, ["price", "category"]);
  });
});

// ═══════════════════════════════════════════════════════════════════
//  CODEGEN — TensorFlow.js
// ═══════════════════════════════════════════════════════════════════

describe("Codegen — TensorFlow.js", () => {
  it("output contains tf.sequential", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitTensorFlow(ir).includes("tf.sequential"));
  });

  it("output contains correct activations", () => {
    const ir = analyze(IRIS)!;
    const code = emitTensorFlow(ir);
    assert.ok(code.includes('"relu"'));
    assert.ok(code.includes('"softmax"'));
  });

  it("output contains loss function", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitTensorFlow(ir).includes("categoricalCrossentropy"));
  });

  it("regression uses linear output", () => {
    const ir = analyze(HOUSING)!;
    const code = emitTensorFlow(ir);
    assert.ok(code.includes('"linear"'));
    assert.ok(code.includes("meanSquaredError"));
  });

  it("includes inputShape on first layer only", () => {
    const ir = analyze(IRIS)!;
    const code = emitTensorFlow(ir);
    const matches = code.match(/inputShape/g);
    assert.equal(matches?.length, 1);
  });

  it("includes batchNormalization when enabled", () => {
    const ir = analyze(`${MINIMAL}\nbatch_norm true`)!;
    assert.ok(emitTensorFlow(ir).includes("batchNormalization"));
  });

  it("includes dropout when enabled", () => {
    const ir = analyze(`${MINIMAL}\ndropout 0.3`)!;
    assert.ok(emitTensorFlow(ir).includes("dropout"));
    assert.ok(emitTensorFlow(ir).includes("0.3"));
  });

  it("includes early stopping when enabled", () => {
    const ir = analyze(`${MINIMAL}\nearly_stop 5`)!;
    assert.ok(emitTensorFlow(ir).includes("earlyStopping"));
  });

  it("includes model export when set", () => {
    const ir = analyze(`${MINIMAL}\nexport "model-out"`)!;
    assert.ok(emitTensorFlow(ir).includes("model.save"));
  });

  it("includes CSV data loader", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitTensorFlow(ir).includes("loadCSV"));
    assert.ok(emitTensorFlow(ir).includes("prepareData"));
  });
});

// ═══════════════════════════════════════════════════════════════════
//  CODEGEN — PyTorch
// ═══════════════════════════════════════════════════════════════════

describe("Codegen — PyTorch", () => {
  it("output contains nn.Module", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitPyTorch(ir).includes("nn.Module"));
  });

  it("output contains nn.Linear", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitPyTorch(ir).includes("nn.Linear"));
  });

  it("classification uses CrossEntropyLoss", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitPyTorch(ir).includes("CrossEntropyLoss"));
  });

  it("regression uses MSELoss", () => {
    const ir = analyze(HOUSING)!;
    assert.ok(emitPyTorch(ir).includes("MSELoss"));
  });

  it("includes BatchNorm1d when enabled", () => {
    const ir = analyze(`${MINIMAL}\nbatch_norm true`)!;
    assert.ok(emitPyTorch(ir).includes("BatchNorm1d"));
  });

  it("includes lr_schedule cosine", () => {
    const ir = analyze(`${MINIMAL}\nlr_schedule cosine`)!;
    assert.ok(emitPyTorch(ir).includes("CosineAnnealingLR"));
  });

  it("includes lr_schedule step", () => {
    const ir = analyze(`${MINIMAL}\nlr_schedule step`)!;
    assert.ok(emitPyTorch(ir).includes("StepLR"));
  });

  it("includes lr_schedule exponential", () => {
    const ir = analyze(`${MINIMAL}\nlr_schedule exponential`)!;
    assert.ok(emitPyTorch(ir).includes("ExponentialLR"));
  });

  it("includes seed when set", () => {
    const ir = analyze(`${MINIMAL}\nseed 42`)!;
    assert.ok(emitPyTorch(ir).includes("manual_seed(42)"));
  });

  it("includes model export when set", () => {
    const ir = analyze(`${MINIMAL}\nexport "model.pt"`)!;
    assert.ok(emitPyTorch(ir).includes("torch.save"));
  });

  it("includes early stopping when set", () => {
    const ir = analyze(`${MINIMAL}\nearly_stop 5`)!;
    const code = emitPyTorch(ir);
    assert.ok(code.includes("best_loss"));
    assert.ok(code.includes("patience_counter"));
  });

  it("includes pandas data loader", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitPyTorch(ir).includes("pd.read_csv"));
  });
});

// ═══════════════════════════════════════════════════════════════════
//  CODEGEN — Keras
// ═══════════════════════════════════════════════════════════════════

describe("Codegen — Keras", () => {
  it("output contains keras.Sequential", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitKeras(ir).includes("keras.Sequential"));
  });

  it("output contains layers.Dense", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitKeras(ir).includes("layers.Dense"));
  });

  it("classification uses categorical_crossentropy", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitKeras(ir).includes("categorical_crossentropy"));
  });

  it("regression uses mse", () => {
    const ir = analyze(HOUSING)!;
    assert.ok(emitKeras(ir).includes('"mse"'));
  });

  it("includes BatchNormalization when enabled", () => {
    const ir = analyze(`${MINIMAL}\nbatch_norm true`)!;
    assert.ok(emitKeras(ir).includes("BatchNormalization"));
  });

  it("includes Dropout when enabled", () => {
    const ir = analyze(`${MINIMAL}\ndropout 0.3`)!;
    assert.ok(emitKeras(ir).includes("Dropout(0.3)"));
  });

  it("includes EarlyStopping callback when enabled", () => {
    const ir = analyze(`${MINIMAL}\nearly_stop 5`)!;
    assert.ok(emitKeras(ir).includes("EarlyStopping"));
  });

  it("includes model.save when export set", () => {
    const ir = analyze(`${MINIMAL}\nexport "my-model"`)!;
    assert.ok(emitKeras(ir).includes('model.save("my-model")'));
  });

  it("includes to_categorical for classification", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitKeras(ir).includes("to_categorical"));
  });

  it("includes model.summary()", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitKeras(ir).includes("model.summary()"));
  });

  it("includes model.fit", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitKeras(ir).includes("model.fit"));
  });

  it("sets seed when specified", () => {
    const ir = analyze(`${MINIMAL}\nseed 42`)!;
    assert.ok(emitKeras(ir).includes("tf.random.set_seed(42)"));
  });

  it("includes lr schedule cosine", () => {
    const ir = analyze(`${MINIMAL}\nlr_schedule cosine`)!;
    assert.ok(emitKeras(ir).includes("LearningRateScheduler"));
    assert.ok(emitKeras(ir).includes("cos"));
  });

  it("includes input_shape on first layer", () => {
    const ir = analyze(IRIS)!;
    const code = emitKeras(ir);
    const matches = code.match(/input_shape/g);
    assert.equal(matches?.length, 1);
  });

  it("uses keras.optimizers", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitKeras(ir).includes("keras.optimizers.Adam"));
  });

  it("supports AdamW optimizer", () => {
    const ir = analyze(`${MINIMAL}\noptimizer adamw`)!;
    assert.ok(emitKeras(ir).includes("keras.optimizers.AdamW"));
  });
});

// ═══════════════════════════════════════════════════════════════════
//  CODEGEN — JAX/Flax
// ═══════════════════════════════════════════════════════════════════

describe("Codegen — JAX/Flax", () => {
  it("output contains flax.linen imports", () => {
    const ir = analyze(IRIS)!;
    const code = emitJAX(ir);
    assert.ok(code.includes("import flax.linen as nn"));
    assert.ok(code.includes("import jax"));
  });

  it("output contains nn.Module class", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitJAX(ir).includes("class Model(nn.Module)"));
  });

  it("output contains @nn.compact", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitJAX(ir).includes("@nn.compact"));
  });

  it("output contains nn.Dense", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitJAX(ir).includes("nn.Dense"));
  });

  it("output uses optax optimizers", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitJAX(ir).includes("optax.adam"));
  });

  it("includes jax.jit training step", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitJAX(ir).includes("@jax.jit"));
    assert.ok(emitJAX(ir).includes("train_step"));
  });

  it("classification uses one_hot", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitJAX(ir).includes("jax.nn.one_hot"));
  });

  it("regression uses jnp.array reshape", () => {
    const ir = analyze(HOUSING)!;
    assert.ok(emitJAX(ir).includes("reshape(-1, 1)"));
  });

  it("includes loss_fn", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitJAX(ir).includes("def loss_fn"));
  });

  it("includes train_state", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitJAX(ir).includes("train_state.TrainState.create"));
  });

  it("includes model.init", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitJAX(ir).includes("model.init(key"));
  });

  it("sets SEED when specified", () => {
    const ir = analyze(`${MINIMAL}\nseed 42`)!;
    assert.ok(emitJAX(ir).includes("SEED = 42"));
  });

  it("supports adamw optimizer via optax", () => {
    const ir = analyze(`${MINIMAL}\noptimizer adamw`)!;
    assert.ok(emitJAX(ir).includes("optax.adamw"));
  });

  it("supports lr_schedule cosine via optax", () => {
    const ir = analyze(`${MINIMAL}\nlr_schedule cosine`)!;
    assert.ok(emitJAX(ir).includes("optax.cosine_decay_schedule"));
  });

  it("includes BatchNorm when enabled", () => {
    const ir = analyze(`${MINIMAL}\nbatch_norm true`)!;
    assert.ok(emitJAX(ir).includes("nn.BatchNorm"));
  });

  it("includes Dropout when enabled", () => {
    const ir = analyze(`${MINIMAL}\ndropout 0.3`)!;
    assert.ok(emitJAX(ir).includes("nn.Dropout(rate=0.3"));
  });
});

// ═══════════════════════════════════════════════════════════════════
//  CODEGEN — Summary
// ═══════════════════════════════════════════════════════════════════

describe("Codegen — Summary", () => {
  it("summary output contains architecture flow", () => {
    const ir = analyze(IRIS)!;
    const s = emitSummary(ir);
    assert.ok(s.includes("Input(4)"));
    assert.ok(s.includes("Dense(128, relu)"));
    assert.ok(s.includes("Dense(3, softmax)"));
  });

  it("summary output shows parameter count", () => {
    const ir = analyze(IRIS)!;
    assert.ok(emitSummary(ir).includes("Parameters:"));
  });

  it("summary shows CNN architecture flow", () => {
    const src = `task classification\npredict digit\ninputs ${Array.from({length: 64}, (_, i) => `p${i}`).join(" ")}\ndataset digits\narchitecture cnn`;
    const ir = analyze(src)!;
    const s = emitSummary(ir);
    assert.ok(s.includes("CNN"));
    assert.ok(s.includes("Conv2D"));
    assert.ok(s.includes("Flatten"));
  });

  it("summary shows LSTM architecture flow", () => {
    const src = `task classification\npredict cls\ninputs ${Array.from({length: 10}, (_, i) => `t${i}`).join(" ")}\ndataset sequences\narchitecture lstm`;
    const ir = analyze(src)!;
    const s = emitSummary(ir);
    assert.ok(s.includes("RNN"));
    assert.ok(s.includes("LSTM"));
  });
});

// ═══════════════════════════════════════════════════════════════════
//  CODEGEN — Architecture-Specific
// ═══════════════════════════════════════════════════════════════════

describe("Codegen — CNN Architecture", () => {
  const CNN_SRC = `task classification\npredict digit\ninputs ${Array.from({length: 64}, (_, i) => `p${i}`).join(" ")}\ndataset digits\narchitecture cnn`;

  it("TensorFlow emits conv2d layers", () => {
    const ir = analyze(CNN_SRC)!;
    const code = emitTensorFlow(ir);
    assert.ok(code.includes("tf.layers.conv2d"));
    assert.ok(code.includes("tf.layers.maxPooling2d"));
    assert.ok(code.includes("tf.layers.flatten"));
  });

  it("PyTorch emits Conv2d layers", () => {
    const ir = analyze(CNN_SRC)!;
    const code = emitPyTorch(ir);
    assert.ok(code.includes("nn.Conv2d"));
    assert.ok(code.includes("nn.MaxPool2d"));
  });

  it("Keras emits Conv2D layers", () => {
    const ir = analyze(CNN_SRC)!;
    const code = emitKeras(ir);
    assert.ok(code.includes("layers.Conv2D"));
    assert.ok(code.includes("layers.MaxPooling2D"));
    assert.ok(code.includes("layers.Flatten"));
  });

  it("JAX emits Conv layers", () => {
    const ir = analyze(CNN_SRC)!;
    const code = emitJAX(ir);
    assert.ok(code.includes("nn.Conv"));
    assert.ok(code.includes("nn.max_pool"));
  });
});

describe("Codegen — RNN Architecture", () => {
  const LSTM_SRC = `task classification\npredict cls\ninputs ${Array.from({length: 10}, (_, i) => `t${i}`).join(" ")}\ndataset sequences\narchitecture lstm`;

  it("TensorFlow emits lstm layers", () => {
    const ir = analyze(LSTM_SRC)!;
    const code = emitTensorFlow(ir);
    assert.ok(code.includes("tf.layers.lstm") || code.includes("tf.layers.reshape"));
  });

  it("PyTorch emits LSTM layers", () => {
    const ir = analyze(LSTM_SRC)!;
    const code = emitPyTorch(ir);
    assert.ok(code.includes("nn.LSTM"));
  });

  it("Keras emits LSTM layers", () => {
    const ir = analyze(LSTM_SRC)!;
    const code = emitKeras(ir);
    assert.ok(code.includes("layers.LSTM"));
  });

  it("JAX emits LSTMCell", () => {
    const ir = analyze(LSTM_SRC)!;
    const code = emitJAX(ir);
    assert.ok(code.includes("LSTMCell"));
  });

  it("GRU variant works", () => {
    const src = `task classification\npredict cls\ninputs ${Array.from({length: 10}, (_, i) => `t${i}`).join(" ")}\ndataset sequences\narchitecture gru`;
    const ir = analyze(src)!;
    assert.ok(emitPyTorch(ir).includes("nn.GRU"));
    assert.ok(emitKeras(ir).includes("layers.GRU"));
  });
});

describe("Codegen — ResNet Architecture", () => {
  const RESNET_SRC = `task classification\npredict digit\ninputs ${Array.from({length: 64}, (_, i) => `p${i}`).join(" ")}\ndataset digits\narchitecture resnet`;

  it("PyTorch emits ResidualBlock class", () => {
    const ir = analyze(RESNET_SRC)!;
    const code = emitPyTorch(ir);
    assert.ok(code.includes("ResidualBlock"));
    assert.ok(code.includes("shortcut"));
  });

  it("Keras emits residual layers", () => {
    const ir = analyze(RESNET_SRC)!;
    const code = emitKeras(ir);
    assert.ok(code.includes("Residual block"));
  });

  it("JAX emits residual skip connection", () => {
    const ir = analyze(RESNET_SRC)!;
    const code = emitJAX(ir);
    assert.ok(code.includes("residual = x") || code.includes("Residual"));
  });
});

describe("Codegen — Transformer Architecture", () => {
  const TRANS_SRC = `task classification\npredict cls\ninputs ${Array.from({length: 10}, (_, i) => `t${i}`).join(" ")}\ndataset sequences\narchitecture transformer`;

  it("PyTorch emits TransformerEncoder", () => {
    const ir = analyze(TRANS_SRC)!;
    const code = emitPyTorch(ir);
    assert.ok(code.includes("TransformerEncoder"));
  });

  it("Keras emits LayerNormalization", () => {
    const ir = analyze(TRANS_SRC)!;
    const code = emitKeras(ir);
    assert.ok(code.includes("LayerNormalization"));
  });
});

describe("Codegen — Transfer Learning", () => {
  it("PyTorch pretrained emits torchvision model", () => {
    const src = `task classification\npredict cls\ninputs ${Array.from({length: 64}, (_, i) => `p${i}`).join(" ")}\ndataset digits\npretrained mobilenet`;
    const ir = analyze(src)!;
    const code = emitPyTorch(ir);
    assert.ok(code.includes("mobilenet_v2"));
    assert.ok(code.includes("MobileNet_V2_Weights"));
  });

  it("Keras pretrained emits keras.applications", () => {
    const src = `task classification\npredict cls\ninputs ${Array.from({length: 64}, (_, i) => `p${i}`).join(" ")}\ndataset digits\npretrained mobilenet`;
    const ir = analyze(src)!;
    const code = emitKeras(ir);
    assert.ok(code.includes("keras.applications.MobileNetV2"));
    assert.ok(code.includes("imagenet"));
  });

  it("freeze_layers generates freeze code", () => {
    const src = `task classification\npredict cls\ninputs ${Array.from({length: 64}, (_, i) => `p${i}`).join(" ")}\ndataset digits\npretrained resnet50\nfreeze_layers 20`;
    const ir = analyze(src)!;
    const pyCode = emitPyTorch(ir);
    assert.ok(pyCode.includes("requires_grad = False"));
    assert.ok(pyCode.includes("20"));
  });
});

describe("Codegen — Model Export", () => {
  it("PyTorch export generates ONNX code", () => {
    const src = `task classification\npredict species\ninputs a b c d\ndataset iris\nexport_format onnx torchscript`;
    const ir = analyze(src)!;
    const code = emitPyTorch(ir);
    assert.ok(code.includes("torch.onnx.export"));
    assert.ok(code.includes("torch.jit.script"));
  });

  it("Keras export generates TFLite code", () => {
    const src = `task classification\npredict species\ninputs a b c d\ndataset iris\nexport_format tflite savedmodel`;
    const ir = analyze(src)!;
    const code = emitKeras(ir);
    assert.ok(code.includes("tf.lite.TFLiteConverter"));
    assert.ok(code.includes("model.save"));
  });
});

// ═══════════════════════════════════════════════════════════════════
//  DATASETS
// ═══════════════════════════════════════════════════════════════════

describe("Datasets", () => {
  it("loads iris dataset (150 samples)", () => {
    const ds = loadIris();
    assert.equal(ds.features.length, 150);
    assert.equal(ds.targets.length, 150);
    assert.equal(ds.featureNames.length, 4);
    assert.deepEqual(ds.classNames, ["setosa", "versicolor", "virginica"]);
  });

  it("iris features have 4 columns each", () => {
    const ds = loadIris();
    for (const row of ds.features) {
      assert.equal(row.length, 4);
    }
  });

  it("iris targets are 0, 1, or 2", () => {
    const ds = loadIris();
    for (const t of ds.targets) {
      assert.ok(t === 0 || t === 1 || t === 2);
    }
  });

  it("loads housing dataset (120 samples)", () => {
    const ds = loadHousing();
    assert.equal(ds.features.length, 120);
    assert.equal(ds.targets.length, 120);
    assert.equal(ds.featureNames.length, 5);
  });

  // --- New datasets ---

  it("loads titanic dataset (891 samples)", () => {
    const ds = loadTitanic();
    assert.equal(ds.features.length, 891);
    assert.equal(ds.targets.length, 891);
    assert.equal(ds.featureNames.length, 7);
    assert.deepEqual(ds.classNames, ["died", "survived"]);
  });

  it("titanic features have 7 columns each", () => {
    const ds = loadTitanic();
    for (const row of ds.features) {
      assert.equal(row.length, 7);
    }
  });

  it("titanic targets are 0 or 1", () => {
    const ds = loadTitanic();
    for (const t of ds.targets) {
      assert.ok(t === 0 || t === 1);
    }
  });

  it("titanic has expected feature names", () => {
    const ds = loadTitanic();
    assert.deepEqual(ds.featureNames, ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]);
  });

  it("loads wine dataset (178 samples)", () => {
    const ds = loadWine();
    assert.equal(ds.features.length, 178);
    assert.equal(ds.targets.length, 178);
    assert.equal(ds.featureNames.length, 13);
    assert.deepEqual(ds.classNames, ["cultivar_1", "cultivar_2", "cultivar_3"]);
  });

  it("wine features have 13 columns each", () => {
    const ds = loadWine();
    for (const row of ds.features) {
      assert.equal(row.length, 13);
    }
  });

  it("wine targets are 0, 1, or 2", () => {
    const ds = loadWine();
    for (const t of ds.targets) {
      assert.ok(t === 0 || t === 1 || t === 2);
    }
  });

  it("loads digits dataset (1000 samples)", () => {
    const ds = loadDigits();
    assert.equal(ds.features.length, 1000);
    assert.equal(ds.targets.length, 1000);
    assert.equal(ds.featureNames.length, 64);
    assert.equal(ds.classNames!.length, 10);
  });

  it("digits features have 64 columns each", () => {
    const ds = loadDigits();
    for (const row of ds.features) {
      assert.equal(row.length, 64);
    }
  });

  it("digits targets are 0-9", () => {
    const ds = loadDigits();
    for (const t of ds.targets) {
      assert.ok(t >= 0 && t <= 9);
    }
  });

  it("digits pixel values are 0-16", () => {
    const ds = loadDigits();
    for (const row of ds.features) {
      for (const v of row) {
        assert.ok(v >= 0 && v <= 16);
      }
    }
  });

  // --- resolveDataset ---

  it("resolves 'iris.csv' to built-in", () => {
    assert.ok(resolveDataset("iris.csv"));
    assert.ok(resolveDataset("iris"));
    assert.ok(resolveDataset("IRIS.CSV"));
  });

  it("resolves 'housing.csv' to built-in", () => {
    assert.ok(resolveDataset("housing.csv"));
    assert.ok(resolveDataset("housing"));
  });

  it("resolves 'titanic.csv' to built-in", () => {
    assert.ok(resolveDataset("titanic.csv"));
    assert.ok(resolveDataset("titanic"));
    assert.ok(resolveDataset("TITANIC.CSV"));
  });

  it("resolves 'wine.csv' to built-in", () => {
    assert.ok(resolveDataset("wine.csv"));
    assert.ok(resolveDataset("wine"));
  });

  it("resolves 'digits.csv' to built-in", () => {
    assert.ok(resolveDataset("digits.csv"));
    assert.ok(resolveDataset("digits"));
  });

  it("loads sequences dataset (200 samples)", () => {
    const ds = loadSequences();
    assert.equal(ds.features.length, 200);
    assert.equal(ds.targets.length, 200);
    assert.equal(ds.featureNames.length, 10);
    assert.equal(ds.classNames!.length, 2);
  });

  it("resolves 'sequences' to built-in", () => {
    assert.ok(resolveDataset("sequences.csv"));
    assert.ok(resolveDataset("sequences"));
  });

  it("loads timeseries dataset (200 samples)", () => {
    const ds = loadTimeseries();
    assert.equal(ds.features.length, 200);
    assert.equal(ds.targets.length, 200);
    assert.equal(ds.featureNames.length, 8);
  });

  it("resolves 'timeseries' to built-in", () => {
    assert.ok(resolveDataset("timeseries.csv"));
    assert.ok(resolveDataset("timeseries"));
  });

  it("returns null for unknown dataset", () => {
    assert.equal(resolveDataset("unknown.csv"), null);
  });

  it("parseCSV parses simple CSV", () => {
    const { headers, rows } = parseCSV("a,b,c\n1,2,3\n4,5,6");
    assert.deepEqual(headers, ["a", "b", "c"]);
    assert.equal(rows.length, 2);
    assert.deepEqual(rows[0], [1, 2, 3]);
  });

  it("parseCSV throws on non-numeric data", () => {
    assert.throws(() => parseCSV("a,b\nhello,world"));
  });

  it("parseCSV throws on empty input", () => {
    assert.throws(() => parseCSV(""));
  });
});

// ═══════════════════════════════════════════════════════════════════
//  AUTO-ARCHITECTURE
// ═══════════════════════════════════════════════════════════════════

describe("Auto-Architecture Search", () => {
  it("getCandidates returns 4 candidates", () => {
    const candidates = getCandidates();
    assert.equal(candidates.length, 4);
  });

  it("candidates have expected names", () => {
    const names = getCandidates().map((c) => c.name);
    assert.deepEqual(names, ["tiny", "small", "medium", "large"]);
  });

  it("candidates have increasing complexity", () => {
    const candidates = getCandidates();
    const paramCounts = candidates.map((c) => c.layers.reduce((s, l) => s + l, 0));
    for (let i = 1; i < paramCounts.length; i++) {
      assert.ok(paramCounts[i]! > paramCounts[i - 1]!);
    }
  });

  it("buildCandidateIR creates valid IR with candidate layers", () => {
    const ir = analyze(IRIS)!;
    const candidate = buildCandidateIR(ir, [64, 32], 5);
    assert.equal(candidate.training.epochs, 5);
    assert.equal(candidate.training.earlyStop, undefined);
    assert.equal(candidate.output.exportPath, undefined);
    const denses = candidate.architecture.layers.filter((l) => l.kind === "dense");
    assert.equal(denses.length, 3); // 64, 32, output
  });

  it("buildCandidateIR preserves task and dataset", () => {
    const ir = analyze(IRIS)!;
    const candidate = buildCandidateIR(ir, [128], 10);
    assert.equal(candidate.task, "classification");
    assert.equal(candidate.dataset, "iris.csv");
  });

  it("formatSearchResults produces output", () => {
    const result = {
      candidates: [
        { name: "tiny", layers: [32], probeLoss: 0.5, probeMetric: 0.8 },
        { name: "small", layers: [64, 32], probeLoss: 0.3, probeMetric: 0.9 },
      ],
      bestIdx: 1,
      bestLayers: [64, 32],
    };
    const output = formatSearchResults(result);
    assert.ok(output.includes("tiny"));
    assert.ok(output.includes("small"));
    assert.ok(output.includes("best"));
  });
});

// ═══════════════════════════════════════════════════════════════════
//  HYPERPARAMETER TUNING
// ═══════════════════════════════════════════════════════════════════

describe("Hyperparameter Tuning", () => {
  it("getTuneConfig returns config with expected shape", () => {
    const config = getTuneConfig();
    assert.ok(config.learningRates.length >= 3);
    assert.ok(config.batchSizes.length >= 3);
    assert.ok(config.architectures.length >= 3);
  });

  it("totalTrials returns correct count", () => {
    const config = getTuneConfig();
    const expected = config.learningRates.length * config.batchSizes.length * config.architectures.length;
    assert.equal(totalTrials(config), expected);
  });

  it("totalTrials returns 27 by default", () => {
    assert.equal(totalTrials(), 27);
  });

  it("buildTrialIR sets learning rate", () => {
    const ir = analyze(IRIS)!;
    const trial = buildTrialIR(ir, 0.01, 64, [128, 64], 10);
    assert.equal(trial.training.optimizer.learningRate, 0.01);
  });

  it("buildTrialIR sets batch size", () => {
    const ir = analyze(IRIS)!;
    const trial = buildTrialIR(ir, 0.001, 64, [128, 64], 10);
    assert.equal(trial.training.batchSize, 64);
  });

  it("buildTrialIR sets probe epochs", () => {
    const ir = analyze(IRIS)!;
    const trial = buildTrialIR(ir, 0.001, 32, [64], 10);
    assert.equal(trial.training.epochs, 10);
  });

  it("formatTuneResults produces output", () => {
    const result = {
      trials: [
        { lr: 0.01, batchSize: 32, archName: "small", layers: [64, 32], probeLoss: 0.3, probeMetric: 0.9 },
        { lr: 0.001, batchSize: 32, archName: "medium", layers: [128, 64], probeLoss: 0.2, probeMetric: 0.95 },
      ],
      bestIdx: 1,
      bestConfig: { lr: 0.001, batchSize: 32, layers: [128, 64] },
    };
    const output = formatTuneResults(result);
    assert.ok(output.includes("Tuning Results"));
    assert.ok(output.includes("Best"));
  });
});

// ═══════════════════════════════════════════════════════════════════
//  FEATURE IMPORTANCE
// ═══════════════════════════════════════════════════════════════════

describe("Feature Importance", () => {
  it("formatFeatureImportance produces table", () => {
    const results = [
      { feature: "petal_length", importance: 0.15, rank: 1 },
      { feature: "petal_width", importance: 0.10, rank: 2 },
      { feature: "sepal_length", importance: 0.03, rank: 3 },
      { feature: "sepal_width", importance: 0.01, rank: 4 },
    ];
    const ir = analyze(IRIS)!;
    const output = formatFeatureImportance(results, ir);
    assert.ok(output.includes("Feature Importance"));
    assert.ok(output.includes("petal_length"));
    assert.ok(output.includes("petal_width"));
    assert.ok(output.includes("sepal_length"));
    assert.ok(output.includes("Rank"));
  });

  it("formatFeatureImportance includes bar chart", () => {
    const results = [
      { feature: "a", importance: 0.5, rank: 1 },
      { feature: "b", importance: 0.2, rank: 2 },
    ];
    const ir = analyze(MINIMAL)!;
    const output = formatFeatureImportance(results, ir);
    assert.ok(output.includes("█"));
  });
});

// ═══════════════════════════════════════════════════════════════════
//  FULL PIPELINE
// ═══════════════════════════════════════════════════════════════════

describe("compile() pipeline", () => {
  it("compiles iris program end-to-end", () => {
    const r = compile(IRIS);
    assert.equal(r.ir.task, "classification");
    assert.ok(r.code.length > 0);
    assert.ok(r.timings.total >= 0);
    assert.equal(r.target, "tensorflow");
  });

  it("compiles housing program end-to-end", () => {
    const r = compile(HOUSING);
    assert.equal(r.ir.task, "regression");
    assert.ok(r.code.includes("meanSquaredError"));
  });

  it("compiles to pytorch target", () => {
    const r = compile(IRIS, { target: "pytorch" });
    assert.equal(r.target, "pytorch");
    assert.ok(r.code.includes("nn.Module"));
  });

  it("compiles to keras target", () => {
    const r = compile(IRIS, { target: "keras" });
    assert.equal(r.target, "keras");
    assert.ok(r.code.includes("keras.Sequential"));
  });

  it("compiles to jax target", () => {
    const r = compile(IRIS, { target: "jax" });
    assert.equal(r.target, "jax");
    assert.ok(r.code.includes("flax.linen"));
  });

  it("compiles to summary target", () => {
    const r = compile(IRIS, { target: "summary" });
    assert.equal(r.target, "summary");
    assert.ok(r.code.includes("NeuroLang Compilation Summary"));
  });

  it("includes all intermediate outputs", () => {
    const r = compile(IRIS);
    assert.ok(r.tokens.length > 0);
    assert.ok(r.ast.body.length > 0);
    assert.ok(r.ir.features.length > 0);
    assert.ok(r.code.length > 0);
  });

  it("includes phase timings", () => {
    const r = compile(IRIS);
    assert.ok(r.timings.lex >= 0);
    assert.ok(r.timings.parse >= 0);
    assert.ok(r.timings.analyze >= 0);
    assert.ok(r.timings.codegen >= 0);
    assert.ok(r.timings.total >= 0);
  });

  it("throws CompilationError for missing required", () => {
    assert.throws(
      () => compile("task classification\ndataset d.csv"),
      (err: unknown) => err instanceof CompilationError,
    );
  });

  it("throws CompilationError for invalid task", () => {
    assert.throws(
      () => compile("task invalid\npredict y\ninputs x\ndataset d.csv"),
      (err: unknown) => err instanceof CompilationError,
    );
  });

  it("CompilationError contains diagnostics", () => {
    try {
      compile("task classification\ndataset d.csv");
      assert.fail("should have thrown");
    } catch (err) {
      assert.ok(err instanceof CompilationError);
      assert.ok(err.diagnostics.length > 0);
      assert.ok(err.source.length > 0);
    }
  });

  it("includes warnings for duplicate keywords", () => {
    const r = compile(`${MINIMAL}\nepochs 10\nepochs 20`);
    assert.ok(r.diagnostics.length > 0);
  });

  it("compiles enterprise program with all features", () => {
    const enterprise = `task classification
predict species
inputs a b c d
dataset iris.csv
loss cross_entropy
optimizer adam
learning_rate 0.0008
epochs 100
learn deep
batch_norm true
dropout 0.25
batch_size 16
seed 42
early_stop 8
lr_schedule cosine
validate 0.15
export "model-out"`;
    const r = compile(enterprise);
    assert.equal(r.ir.task, "classification");
    assert.ok(r.ir.architecture.batchNorm);
    assert.ok(r.ir.training.earlyStop);
    assert.equal(r.ir.training.lrSchedule.type, "cosine");
    assert.equal(r.ir.preprocessing.seed, 42);
    assert.equal(r.ir.output.exportPath, "model-out");
  });

  it("compiles titanic classification", () => {
    const titanic = `task classification
predict survived
inputs pclass sex age sibsp parch fare embarked
dataset titanic.csv
loss binary_cross_entropy`;
    const r = compile(titanic);
    assert.equal(r.ir.task, "classification");
    assert.equal(r.ir.architecture.outputSize, 2);
    assert.equal(r.ir.training.loss, "binaryCrossentropy");
  });

  it("compiles wine classification", () => {
    const wine = `task classification
predict cultivar
inputs alcohol malic_acid ash
dataset wine.csv`;
    const r = compile(wine);
    assert.equal(r.ir.architecture.outputSize, 3);
  });

  it("compiles digits classification", () => {
    const digits = `task classification
predict digit
inputs pixel_0 pixel_1
dataset digits.csv`;
    const r = compile(digits);
    assert.equal(r.ir.architecture.outputSize, 10);
  });

  it("compiles auto learn mode", () => {
    const auto = `task classification
predict species
inputs a b c d
dataset iris.csv
learn auto`;
    const r = compile(auto);
    assert.equal(r.ir.meta.learnMode, "auto");
  });

  it("compiles tune mode", () => {
    const tuned = `task classification
predict species
inputs a b c d
dataset iris.csv
tune true`;
    const r = compile(tuned);
    assert.equal(r.ir.training.tune, true);
  });

  it("compiles to keras with all features", () => {
    const program = `task regression
predict price
inputs size beds
dataset housing.csv
optimizer adam
epochs 30
early_stop 5
lr_schedule cosine
export "model"
seed 42`;
    const r = compile(program, { target: "keras" });
    const code = r.code;
    assert.ok(code.includes("keras.Sequential"));
    assert.ok(code.includes("EarlyStopping"));
    assert.ok(code.includes("LearningRateScheduler"));
    assert.ok(code.includes('model.save("model")'));
    assert.ok(code.includes("tf.random.set_seed(42)"));
  });

  it("compiles to jax with all features", () => {
    const program = `task classification
predict species
inputs a b c d
dataset iris.csv
optimizer adamw
batch_norm true
dropout 0.2
seed 42`;
    const r = compile(program, { target: "jax" });
    const code = r.code;
    assert.ok(code.includes("nn.Module"));
    assert.ok(code.includes("@nn.compact"));
    assert.ok(code.includes("optax.adamw"));
    assert.ok(code.includes("nn.BatchNorm"));
    assert.ok(code.includes("nn.Dropout"));
    assert.ok(code.includes("SEED = 42"));
  });

  it("recognises cross_validate keyword", () => {
    const program = `task classification
predict species
inputs a b c d
dataset iris.csv
cross_validate 5`;
    const r = compile(program);
    assert.equal(r.ir.preprocessing.crossValidation, 5);
  });

  it("cross_validate appears in summary", () => {
    const program = `task classification
predict species
inputs a b c d
dataset iris.csv
cross_validate 10`;
    const r = compile(program, { target: "summary" });
    assert.ok(r.code.includes("10-fold"));
  });

  it("cross_validate defaults to undefined", () => {
    const program = `task classification
predict species
inputs a b c d
dataset iris.csv`;
    const r = compile(program);
    assert.equal(r.ir.preprocessing.crossValidation, undefined);
  });

  it("recognises ensemble keyword", () => {
    const program = `task classification
predict species
inputs a b c d
dataset iris.csv
ensemble 3`;
    const r = compile(program);
    assert.equal(r.ir.training.ensemble, 3);
  });

  it("ensemble appears in summary", () => {
    const program = `task classification
predict species
inputs a b c d
dataset iris.csv
ensemble 5`;
    const r = compile(program, { target: "summary" });
    assert.ok(r.code.includes("5 models"));
    assert.ok(r.code.includes("bagging"));
  });

  it("ensemble defaults to undefined", () => {
    const program = `task classification
predict species
inputs a b c d
dataset iris.csv`;
    const r = compile(program);
    assert.equal(r.ir.training.ensemble, undefined);
  });
});

// ─── Natural Language Parser ───────────────────────────────────────────

describe("isNaturalSyntax()", () => {
  it("detects Predict as natural", () => {
    assert.ok(isNaturalSyntax("Predict species with a & b from iris.csv"));
  });
  it("detects Classify as natural", () => {
    assert.ok(isNaturalSyntax("Classify survived with age & fare from titanic.csv"));
  });
  it("detects Estimate as natural", () => {
    assert.ok(isNaturalSyntax("Estimate price from size & beds in housing.csv"));
  });
  it("does not detect keyword syntax as natural", () => {
    assert.ok(!isNaturalSyntax("task classification\npredict species"));
  });
  it("does not detect empty string", () => {
    assert.ok(!isNaturalSyntax(""));
  });
});

describe("parseNatural()", () => {
  it("parses basic Predict with from", () => {
    const r = parseNatural("Predict species with a & b from iris.csv");
    assert.ok(r.detected);
    assert.ok(r.program.includes("predict species"));
    assert.ok(r.program.includes("inputs a b"));
    assert.ok(r.program.includes("dataset iris.csv"));
    assert.ok(r.program.includes("task classification"));
  });

  it("parses Estimate as regression", () => {
    const r = parseNatural("Estimate price from size & bedrooms in housing.csv");
    assert.ok(r.detected);
    assert.ok(r.program.includes("task regression"));
    assert.ok(r.program.includes("predict price"));
    assert.ok(r.program.includes("inputs size bedrooms"));
    assert.ok(r.program.includes("dataset housing.csv"));
  });

  it("parses deeply modifier", () => {
    const r = parseNatural("Predict x with a & b from iris.csv deeply");
    assert.ok(r.program.includes("learn deep"));
  });

  it("parses epoch count", () => {
    const r = parseNatural("Predict x with a & b from iris.csv for 100 epochs");
    assert.ok(r.program.includes("epochs 100"));
  });

  it("parses ensemble modifier", () => {
    const r = parseNatural("Predict x with a & b from iris.csv using 5 models");
    assert.ok(r.program.includes("ensemble 5"));
  });

  it("parses batch normalization", () => {
    const r = parseNatural("Predict x with a & b from iris.csv with batch normalization");
    assert.ok(r.program.includes("batch_norm true"));
  });

  it("parses combined modifiers", () => {
    const r = parseNatural("Classify x with a & b from iris.csv deeply for 50 epochs using 3 models");
    assert.ok(r.program.includes("learn deep"));
    assert.ok(r.program.includes("epochs 50"));
    assert.ok(r.program.includes("ensemble 3"));
  });

  it("infers classification from Classify verb", () => {
    const r = parseNatural("Classify target with a & b from data.csv");
    assert.ok(r.program.includes("task classification"));
  });

  it("infers regression from price target", () => {
    const r = parseNatural("Predict price with a & b from data.csv");
    assert.ok(r.program.includes("task regression"));
  });

  it("passes keyword syntax through unchanged", () => {
    const kw = "task classification\npredict species";
    const r = parseNatural(kw);
    assert.ok(!r.detected);
    assert.equal(r.program, kw);
  });

  it("detects CNN architecture from natural language", () => {
    const r = parseNatural("Classify digit from digits using CNN");
    assert.ok(r.detected);
    assert.ok(r.program.includes("architecture cnn"));
  });

  it("detects LSTM architecture from natural language", () => {
    const r = parseNatural("Predict frequency_class from sequences using LSTM");
    assert.ok(r.detected);
    assert.ok(r.program.includes("architecture lstm"));
  });

  it("detects GRU architecture from natural language", () => {
    const r = parseNatural("Classify cls from sequences with a GRU");
    assert.ok(r.detected);
    assert.ok(r.program.includes("architecture gru"));
  });

  it("detects autoencoder architecture from natural language", () => {
    const r = parseNatural("Detect anomalies from data.csv using autoencoder");
    assert.ok(r.detected);
    assert.ok(r.program.includes("architecture autoencoder"));
  });

  it("detects bidirectional from natural language", () => {
    const r = parseNatural("Classify sentiment from reviews using bidirectional");
    assert.ok(r.detected);
    assert.ok(r.program.includes("bidirectional true"));
  });

  it("detects resnet from natural language", () => {
    const r = parseNatural("Classify digit from digits using resnet");
    assert.ok(r.detected);
    assert.ok(r.program.includes("architecture resnet"));
  });

  it("detects transformer from natural language", () => {
    const r = parseNatural("Classify cls from sequences using transformer");
    assert.ok(r.detected);
    assert.ok(r.program.includes("architecture transformer"));
  });

  it("detects pretrained mobilenet from natural language", () => {
    const r = parseNatural("Classify digit from digits using pretrained mobilenet");
    assert.ok(r.detected);
    assert.ok(r.program.includes("pretrained mobilenet"));
  });

  it("detects export to onnx from natural language", () => {
    const r = parseNatural("Predict species from iris and export onnx");
    assert.ok(r.detected);
    assert.ok(r.program.includes("export_format onnx"));
  });
});

describe("natural syntax → compile()", () => {
  it("compiles natural Predict sentence", () => {
    const r = compile("Predict species with sepal_length & sepal_width & petal_length & petal_width from iris.csv");
    assert.equal(r.ir.task, "classification");
    assert.equal(r.ir.target, "species");
    assert.deepEqual(r.ir.features, ["sepal_length", "sepal_width", "petal_length", "petal_width"]);
    assert.equal(r.ir.dataset, "iris.csv");
  });

  it("compiles natural Classify sentence with modifiers", () => {
    const r = compile("Classify survived with pclass & sex & age from titanic.csv deeply for 50 epochs");
    assert.equal(r.ir.task, "classification");
    assert.equal(r.ir.target, "survived");
    assert.equal(r.ir.meta.learnMode, "deep");
    assert.equal(r.ir.training.epochs, 50);
  });

  it("compiles natural Estimate sentence as regression", () => {
    const r = compile("Estimate price from size & bedrooms in housing.csv");
    assert.equal(r.ir.task, "regression");
    assert.equal(r.ir.target, "price");
    assert.deepEqual(r.ir.features, ["size", "bedrooms"]);
  });

  it("compiles natural ensemble sentence", () => {
    const r = compile("Predict species with a & b & c & d from iris.csv using 3 models");
    assert.equal(r.ir.training.ensemble, 3);
  });

  it("generates code from natural syntax", () => {
    const r = compile("Predict species with a & b from iris.csv", { target: "pytorch" });
    assert.ok(r.code.includes("nn.Linear"));
    assert.ok(r.code.includes("nn.Sequential"));
  });

  it("generates keras from natural syntax", () => {
    const r = compile("Estimate price from size & beds in housing.csv", { target: "keras" });
    assert.ok(r.code.includes("keras.Sequential"));
    assert.ok(r.code.includes("mse"));
  });
});

// ─── Data Inspection ───────────────────────────────────────────────

describe("data inspection", () => {
  it("describeDataset returns correct shape", () => {
    const ds = loadIris();
    const desc = describeDataset(ds);
    assert.equal(desc.rows, 150);
    assert.equal(desc.cols, 4);
    assert.deepEqual(desc.featureNames, ["sepal_length", "sepal_width", "petal_length", "petal_width"]);
    assert.equal(desc.targetName, "species");
  });

  it("describeDataset computes stats", () => {
    const ds = loadIris();
    const desc = describeDataset(ds);
    assert.equal(desc.stats.length, 4);
    const sl = desc.stats[0]!;
    assert.equal(sl.name, "sepal_length");
    assert.equal(sl.count, 150);
    assert.ok(sl.mean > 5 && sl.mean < 6);
    assert.ok(sl.min >= 4);
    assert.ok(sl.max <= 8);
  });

  it("formatDataPreview includes header and rows", () => {
    const ds = loadIris();
    const preview = formatDataPreview(ds, 5);
    assert.ok(preview.includes("sepal_length"));
    assert.ok(preview.includes("species"));
    assert.ok(preview.includes("145 more rows"));
  });

  it("formatDescription includes shape info", () => {
    const ds = loadIris();
    const desc = describeDataset(ds);
    const text = formatDescription(desc);
    assert.ok(text.includes("150 rows"));
    assert.ok(text.includes("4 features"));
  });

  it("formatCorrelation includes feature names", () => {
    const ds = loadIris();
    const text = formatCorrelation(ds);
    assert.ok(text.includes("sepal_length"));
    assert.ok(text.includes("petal_width"));
    assert.ok(text.includes("1.00"));
  });

  it("formatSample returns requested rows", () => {
    const ds = loadIris();
    const text = formatSample(ds, 3, 42);
    assert.ok(text.includes("3 rows"));
    const lines = text.split("\n").filter((l) => l.trim() && !l.includes("─") && !l.includes("sample") && !l.includes("sepal"));
    assert.equal(lines.length, 3);
  });
});

describe("show/describe/sample keywords", () => {
  it("parses show keyword", () => {
    const r = compile("task classification\npredict x\ninputs a b\ndataset iris.csv\nshow data");
    assert.equal(r.ir.data.show, true);
  });

  it("parses describe keyword", () => {
    const r = compile("task classification\npredict x\ninputs a b\ndataset iris.csv\ndescribe true");
    assert.equal(r.ir.data.describe, true);
  });

  it("parses sample keyword", () => {
    const r = compile("task classification\npredict x\ninputs a b\ndataset iris.csv\nsample 5");
    assert.equal(r.ir.data.sample, 5);
  });

  it("data commands default to undefined", () => {
    const r = compile("task classification\npredict x\ninputs a b\ndataset iris.csv");
    assert.equal(r.ir.data.show, undefined);
    assert.equal(r.ir.data.describe, undefined);
    assert.equal(r.ir.data.sample, undefined);
  });

  it("natural syntax and describe works", () => {
    const r = compile("Predict species with a & b from iris.csv and describe");
    assert.equal(r.ir.data.describe, true);
  });
});

// ═══════════════════════════════════════════════════════════════════
//  INLINE DATASET REGISTRY
// ═══════════════════════════════════════════════════════════════════

describe("inline dataset registry", () => {
  it("datasetFromObjects creates a valid dataset", () => {
    const rows = [
      { age: 25, income: 50000, churned: 0 },
      { age: 45, income: 120000, churned: 1 },
      { age: 35, income: 80000, churned: 0 },
    ];
    const ds = datasetFromObjects(rows);
    assert.equal(ds.targetName, "churned");
    assert.deepEqual(ds.featureNames, ["age", "income"]);
    assert.equal(ds.features.length, 3);
    assert.equal(ds.targets.length, 3);
    assert.deepEqual(ds.features[0], [25, 50000]);
    assert.equal(ds.targets[1], 1);
  });

  it("datasetFromObjects accepts explicit target column", () => {
    const rows = [
      { x: 1, y: 2, z: 3 },
      { x: 4, y: 5, z: 6 },
    ];
    const ds = datasetFromObjects(rows, "y");
    assert.equal(ds.targetName, "y");
    assert.deepEqual(ds.featureNames, ["x", "z"]);
    assert.deepEqual(ds.features[0], [1, 3]);
    assert.equal(ds.targets[0], 2);
  });

  it("datasetFromObjects throws on empty array", () => {
    assert.throws(() => datasetFromObjects([]), /at least one row/);
  });

  it("datasetFromObjects throws on single-column data", () => {
    assert.throws(() => datasetFromObjects([{ x: 1 }]), /at least 2 columns/);
  });

  it("register/unregister makes dataset available to resolveDataset", () => {
    const ds = datasetFromObjects([
      { a: 1, b: 2, c: 0 },
      { a: 3, b: 4, c: 1 },
    ]);
    registerInlineDataset("__test_ds", ds);
    const resolved = resolveDataset("__test_ds");
    assert.ok(resolved);
    assert.equal(resolved!.targetName, "c");

    unregisterInlineDataset("__test_ds");
    const after = resolveDataset("__test_ds");
    assert.equal(after, null);
  });

  it("inline dataset compiles when registered", () => {
    const ds = datasetFromObjects([
      { age: 25, income: 50000, churned: 0 },
      { age: 45, income: 120000, churned: 1 },
      { age: 35, income: 80000, churned: 0 },
      { age: 50, income: 90000, churned: 1 },
    ]);
    registerInlineDataset("__test_inline", ds);
    try {
      const result = compile(
        "task classification\npredict churned\ninputs age income\ndataset __test_inline",
      );
      assert.equal(result.ir.task, "classification");
      assert.equal(result.ir.target, "churned");
      assert.equal(result.ir.dataset, "__test_inline");
    } finally {
      unregisterInlineDataset("__test_inline");
    }
  });
});

// Template tests are in tests/template.test.ts
