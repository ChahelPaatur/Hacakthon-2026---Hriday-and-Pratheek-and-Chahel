/**
 * Structured diagnostic system for NeuroLang.
 * Inspired by Rust's error model: every diagnostic has a code, severity,
 * source span, and optional fix suggestion.
 */

export enum Severity {
  Error = "error",
  Warning = "warning",
  Info = "info",
}

export interface SourceSpan {
  line: number;
  column: number;
  length: number;
}

export interface Diagnostic {
  code: string;
  severity: Severity;
  message: string;
  span: SourceSpan;
  help?: string;
}

// ─── Error Codes ────────────────────────────────────────────────────

export const E = {
  UNEXPECTED_CHAR:    "NL0001",
  UNTERMINATED_STR:   "NL0002",
  EXPECTED_KEYWORD:   "NL0003",
  EXPECTED_VALUE:     "NL0004",
  MISSING_REQUIRED:   "NL0010",
  INVALID_TASK:       "NL0011",
  INVALID_LEARN:      "NL0012",
  INVALID_OPTIMIZER:  "NL0013",
  INVALID_ACTIVATION: "NL0014",
  INVALID_LOSS:       "NL0015",
  INVALID_EPOCHS:     "NL0016",
  DUPLICATE_KEYWORD:  "NL0020",
  UNKNOWN_KEYWORD:    "NL0021",
  NUMERIC_EXPECTED:   "NL0022",
  DATASET_NOT_FOUND:  "NL0030",
  TF_NOT_INSTALLED:   "NL0031",
} as const;

// ─── Diagnostic Builder ─────────────────────────────────────────────

export function error(code: string, message: string, span: SourceSpan, help?: string): Diagnostic {
  return { code, severity: Severity.Error, message, span, help };
}

export function warning(code: string, message: string, span: SourceSpan, help?: string): Diagnostic {
  return { code, severity: Severity.Warning, message, span, help };
}

export function info(code: string, message: string, span: SourceSpan, help?: string): Diagnostic {
  return { code, severity: Severity.Info, message, span, help };
}

// ─── Renderer ───────────────────────────────────────────────────────

const RED = "\x1b[31m";
const YELLOW = "\x1b[33m";
const CYAN = "\x1b[36m";
const BLUE = "\x1b[34m";
const BOLD = "\x1b[1m";
const DIM = "\x1b[2m";
const RESET = "\x1b[0m";

function severityColor(s: Severity): string {
  switch (s) {
    case Severity.Error: return RED;
    case Severity.Warning: return YELLOW;
    case Severity.Info: return CYAN;
  }
}

export function formatDiagnostic(d: Diagnostic, source: string, filename = "<input>"): string {
  const lines = source.split("\n");
  const color = severityColor(d.severity);
  const label = d.severity.toUpperCase();
  const out: string[] = [];

  out.push(`${color}${BOLD}${label.toLowerCase()}[${d.code}]${RESET}${BOLD}: ${d.message}${RESET}`);
  out.push(`  ${BLUE}-->${RESET} ${filename}:${d.span.line}:${d.span.column}`);

  const lineIdx = d.span.line - 1;
  const gutterWidth = String(d.span.line + 1).length + 1;
  const pad = (n: number) => String(n).padStart(gutterWidth);

  out.push(`${DIM}${" ".repeat(gutterWidth)} |${RESET}`);

  // Show up to 1 line of context before
  if (lineIdx > 0 && lines[lineIdx - 1] !== undefined) {
    out.push(`${DIM}${pad(d.span.line - 1)} |${RESET} ${lines[lineIdx - 1]}`);
  }

  // The offending line
  if (lines[lineIdx] !== undefined) {
    out.push(`${color}${pad(d.span.line)} |${RESET} ${lines[lineIdx]}`);

    // Underline caret
    const caretPad = " ".repeat(d.span.column - 1);
    const caret = "^".repeat(Math.max(1, d.span.length));
    out.push(`${DIM}${" ".repeat(gutterWidth)} |${RESET} ${caretPad}${color}${BOLD}${caret}${RESET}`);
  }

  // Show 1 line of context after
  if (lines[lineIdx + 1] !== undefined) {
    out.push(`${DIM}${pad(d.span.line + 1)} |${RESET} ${lines[lineIdx + 1]}`);
  }

  out.push(`${DIM}${" ".repeat(gutterWidth)} |${RESET}`);

  if (d.help) {
    out.push(`  ${BLUE}= help${RESET}: ${d.help}`);
  }

  return out.join("\n");
}

export function formatDiagnostics(diagnostics: Diagnostic[], source: string, filename?: string): string {
  return diagnostics.map((d) => formatDiagnostic(d, source, filename)).join("\n\n");
}

// ─── Collector ──────────────────────────────────────────────────────

export class DiagnosticCollector {
  private items: Diagnostic[] = [];

  add(d: Diagnostic): void {
    this.items.push(d);
  }

  error(code: string, message: string, span: SourceSpan, help?: string): void {
    this.items.push(error(code, message, span, help));
  }

  warning(code: string, message: string, span: SourceSpan, help?: string): void {
    this.items.push(warning(code, message, span, help));
  }

  get diagnostics(): readonly Diagnostic[] {
    return this.items;
  }

  get errors(): Diagnostic[] {
    return this.items.filter((d) => d.severity === Severity.Error);
  }

  get warnings(): Diagnostic[] {
    return this.items.filter((d) => d.severity === Severity.Warning);
  }

  get hasErrors(): boolean {
    return this.items.some((d) => d.severity === Severity.Error);
  }

  clear(): void {
    this.items = [];
  }
}
