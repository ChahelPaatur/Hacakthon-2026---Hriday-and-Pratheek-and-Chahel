/**
 * Browser entry point for NeuroLang.
 * Bundles the compiler (no TF.js runtime) for in-browser use.
 */
export { compile, CompilationError } from "./compiler.js";
export { emitTensorFlow, emitPyTorch, emitKeras, emitJAX, emitSummary } from "./codegen.js";
export { Lexer } from "./lexer.js";
export { Parser } from "./parser.js";
export { Analyzer } from "./analyzer.js";
export { DiagnosticCollector, formatDiagnostic, formatDiagnostics } from "./diagnostics.js";
export { parseNatural, isNaturalSyntax } from "./natural.js";
export { datasetFromObjects } from "./datasets.js";

export type { CompilationResult, CompileOptions, CodegenTarget, PhaseTimings } from "./compiler.js";
export type { NeuralNetworkIR, LayerSpec } from "./ir.js";
export type { Token } from "./tokens.js";
export type { ProgramNode } from "./ast.js";
export type { Diagnostic } from "./diagnostics.js";
