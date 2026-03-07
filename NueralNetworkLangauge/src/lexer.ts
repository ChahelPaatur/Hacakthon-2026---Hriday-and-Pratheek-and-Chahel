import { Token, TokenType, isKeyword } from "./tokens.js";
import { DiagnosticCollector, E } from "./diagnostics.js";

export class Lexer {
  private source: string;
  private pos = 0;
  private line = 1;
  private column = 1;
  private tokens: Token[] = [];
  private collector: DiagnosticCollector;

  constructor(source: string, collector?: DiagnosticCollector) {
    this.source = source;
    this.collector = collector ?? new DiagnosticCollector();
  }

  tokenize(): Token[] {
    while (!this.isAtEnd()) {
      this.skipSpaces();
      if (this.isAtEnd()) break;

      const ch = this.peek();

      if (ch === "#") {
        this.skipComment();
        continue;
      }

      if (ch === "\n") {
        this.emitToken(TokenType.NEWLINE, "\\n", 1);
        this.advance();
        this.line++;
        this.column = 1;
        continue;
      }

      if (ch === "\r") {
        this.advance();
        if (this.peek() === "\n") this.advance();
        this.emitToken(TokenType.NEWLINE, "\\n", 1);
        this.line++;
        this.column = 1;
        continue;
      }

      if (ch === '"' || ch === "'") {
        this.readString(ch);
        continue;
      }

      if (this.isDigit(ch) || (ch === "." && this.isDigit(this.peekAt(1)))) {
        this.readNumber();
        continue;
      }

      if (this.isIdentStart(ch)) {
        this.readWord();
        continue;
      }

      this.collector.error(
        E.UNEXPECTED_CHAR,
        `Unexpected character '${ch}'`,
        { line: this.line, column: this.column, length: 1 },
        `Remove or replace '${ch}'`,
      );
      this.advance();
    }

    this.collapseNewlines();
    this.tokens.push({
      type: TokenType.EOF,
      value: "",
      line: this.line,
      column: this.column,
      length: 0,
    });
    return this.tokens;
  }

  private emitToken(type: TokenType, value: string, length: number): void {
    this.tokens.push({ type, value, line: this.line, column: this.column, length });
  }

  private peek(): string {
    return this.source[this.pos] ?? "\0";
  }

  private peekAt(offset: number): string {
    return this.source[this.pos + offset] ?? "\0";
  }

  private advance(): string {
    const ch = this.source[this.pos] ?? "\0";
    this.pos++;
    this.column++;
    return ch;
  }

  private isAtEnd(): boolean {
    return this.pos >= this.source.length;
  }

  private isDigit(ch: string): boolean {
    return ch >= "0" && ch <= "9";
  }

  private isIdentStart(ch: string): boolean {
    return (ch >= "a" && ch <= "z") || (ch >= "A" && ch <= "Z") || ch === "_";
  }

  private isIdentChar(ch: string): boolean {
    return this.isIdentStart(ch) || this.isDigit(ch) || ch === "." || ch === "/" || ch === "-";
  }

  private skipSpaces(): void {
    while (!this.isAtEnd()) {
      const ch = this.peek();
      if (ch === " " || ch === "\t") {
        this.advance();
      } else {
        break;
      }
    }
  }

  private skipComment(): void {
    while (!this.isAtEnd() && this.peek() !== "\n") {
      this.advance();
    }
  }

  private readString(quote: string): void {
    const startCol = this.column;
    this.advance(); // consume opening quote
    let value = "";

    while (!this.isAtEnd() && this.peek() !== quote && this.peek() !== "\n") {
      if (this.peek() === "\\") {
        this.advance();
        const esc = this.advance();
        switch (esc) {
          case "n": value += "\n"; break;
          case "t": value += "\t"; break;
          case "\\": value += "\\"; break;
          case '"': value += '"'; break;
          case "'": value += "'"; break;
          default: value += esc;
        }
      } else {
        value += this.advance();
      }
    }

    if (this.isAtEnd() || this.peek() === "\n") {
      this.collector.error(
        E.UNTERMINATED_STR,
        "Unterminated string literal",
        { line: this.line, column: startCol, length: value.length + 1 },
        `Add a closing ${quote} to end the string`,
      );
    } else {
      this.advance(); // consume closing quote
    }

    const length = value.length + 2;
    this.tokens.push({
      type: TokenType.STRING,
      value,
      line: this.line,
      column: startCol,
      length,
    });
  }

  /** Reads integers, floats, and scientific notation (1e-5, 2.5e3). */
  private readNumber(): void {
    const startCol = this.column;
    let raw = "";

    while (!this.isAtEnd() && this.isDigit(this.peek())) {
      raw += this.advance();
    }

    // Decimal part
    if (!this.isAtEnd() && this.peek() === "." && this.isDigit(this.peekAt(1))) {
      raw += this.advance(); // '.'
      while (!this.isAtEnd() && this.isDigit(this.peek())) {
        raw += this.advance();
      }
    }

    // Scientific notation: e/E followed by optional +/- and digits
    if (!this.isAtEnd() && (this.peek() === "e" || this.peek() === "E")) {
      raw += this.advance(); // 'e' or 'E'
      if (!this.isAtEnd() && (this.peek() === "+" || this.peek() === "-")) {
        raw += this.advance();
      }
      while (!this.isAtEnd() && this.isDigit(this.peek())) {
        raw += this.advance();
      }
    }

    this.tokens.push({
      type: TokenType.NUMBER,
      value: raw,
      line: this.line,
      column: startCol,
      length: raw.length,
    });
  }

  private readWord(): void {
    const startCol = this.column;
    let word = "";

    while (!this.isAtEnd() && this.isIdentChar(this.peek())) {
      word += this.advance();
    }

    const type = isKeyword(word) ? TokenType.KEYWORD : TokenType.IDENT;
    this.tokens.push({
      type,
      value: word,
      line: this.line,
      column: startCol,
      length: word.length,
    });
  }

  private collapseNewlines(): void {
    const collapsed: Token[] = [];
    let prevWasNewline = false;
    for (const tok of this.tokens) {
      if (tok.type === TokenType.NEWLINE) {
        if (!prevWasNewline) collapsed.push(tok);
        prevWasNewline = true;
      } else {
        collapsed.push(tok);
        prevWasNewline = false;
      }
    }
    while (collapsed.length && collapsed[0]!.type === TokenType.NEWLINE) collapsed.shift();
    while (collapsed.length && collapsed[collapsed.length - 1]!.type === TokenType.NEWLINE) collapsed.pop();
    this.tokens = collapsed;
  }
}
