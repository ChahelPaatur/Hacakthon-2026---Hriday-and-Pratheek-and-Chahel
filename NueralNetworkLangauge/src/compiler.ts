import { Lexer } from "./lexer.js";
import { Parser } from "./parser.js";
import { Analyzer } from "./analyzer.js";
import { emitTensorFlow, emitPyTorch, emitKeras, emitJAX, emitSummary } from "./codegen.js";
import { DiagnosticCollector, type Diagnostic } from "./diagnostics.js";
import { isNaturalSyntax, parseNatural } from "./natural.js";
import type { Token } from "./tokens.js";
import type { ProgramNode } from "./ast.js";
import type { NeuralNetworkIR } from "./ir.js";

export type CodegenTarget = "tensorflow" | "pytorch" | "keras" | "jax" | "summary";

export interface PhaseTimings {
  lex: number;
  parse: number;
  analyze: number;
  codegen: number;
  total: number;
}

export interface CompilationResult {
  tokens: Token[];
  ast: ProgramNode;
  ir: NeuralNetworkIR;
  code: string;
  target: CodegenTarget;
  diagnostics: readonly Diagnostic[];
  timings: PhaseTimings;
}

export interface CompileOptions {
  target?: CodegenTarget;
  filename?: string;
}

/**
 * Full compilation pipeline: Source → Tokens → AST → IR → Generated Code.
 *
 * Each phase feeds the shared DiagnosticCollector so all warnings/errors
 * are presented together with source context.
 */
export function compile(source: string, options: CompileOptions = {}): CompilationResult {
  const target = options.target ?? "tensorflow";
  const collector = new DiagnosticCollector();
  const totalStart = performance.now();

  // Phase 0: Natural language detection & translation
  let effectiveSource = source;
  if (isNaturalSyntax(source)) {
    const result = parseNatural(source);
    effectiveSource = result.program;
  }

  // Phase 1: Lexical analysis
  const lexStart = performance.now();
  const lexer = new Lexer(effectiveSource, collector);
  const tokens = lexer.tokenize();
  const lexTime = performance.now() - lexStart;

  if (collector.hasErrors) {
    throw new CompilationError(collector.diagnostics, source, options.filename);
  }

  // Phase 2: Parsing (recursive descent → AST)
  const parseStart = performance.now();
  const parser = new Parser(tokens, collector);
  const ast = parser.parse();
  const parseTime = performance.now() - parseStart;

  if (collector.hasErrors) {
    throw new CompilationError(collector.diagnostics, source, options.filename);
  }

  // Phase 3: Semantic analysis (AST → validated config → IR)
  const analyzeStart = performance.now();
  const analyzer = new Analyzer(collector);
  const ir = analyzer.analyze(ast);
  const analyzeTime = performance.now() - analyzeStart;

  if (!ir || collector.hasErrors) {
    throw new CompilationError(collector.diagnostics, source, options.filename);
  }

  // Phase 4: Code generation (IR → target code)
  const codegenStart = performance.now();
  const code = generateCode(ir, target);
  const codegenTime = performance.now() - codegenStart;

  const totalTime = performance.now() - totalStart;

  return {
    tokens,
    ast,
    ir,
    code,
    target,
    diagnostics: collector.diagnostics,
    timings: {
      lex: lexTime,
      parse: parseTime,
      analyze: analyzeTime,
      codegen: codegenTime,
      total: totalTime,
    },
  };
}

function generateCode(ir: NeuralNetworkIR, target: CodegenTarget): string {
  switch (target) {
    case "tensorflow": return emitTensorFlow(ir);
    case "pytorch": return emitPyTorch(ir);
    case "keras": return emitKeras(ir);
    case "jax": return emitJAX(ir);
    case "summary": return emitSummary(ir);
  }
}

export class CompilationError extends Error {
  constructor(
    public diagnostics: readonly Diagnostic[],
    public source: string,
    public filename?: string,
  ) {
    const msgs = diagnostics
      .filter((d) => d.severity === "error")
      .map((d) => d.message);
    super(`Compilation failed with ${msgs.length} error(s):\n${msgs.join("\n")}`);
    this.name = "CompilationError";
  }
}
