import { Token, TokenType } from "./tokens.js";
import type { ProgramNode, StatementNode, ValueNode } from "./ast.js";
import { DiagnosticCollector, E } from "./diagnostics.js";

export class Parser {
  private tokens: Token[];
  private pos = 0;
  private collector: DiagnosticCollector;

  constructor(tokens: Token[], collector?: DiagnosticCollector) {
    this.tokens = tokens;
    this.collector = collector ?? new DiagnosticCollector();
  }

  parse(): ProgramNode {
    const body: StatementNode[] = [];

    while (!this.isAtEnd()) {
      this.skipNewlines();
      if (this.isAtEnd()) break;

      const stmt = this.parseStatement();
      if (stmt) body.push(stmt);
    }

    return { type: "Program", body };
  }

  private parseStatement(): StatementNode | null {
    const tok = this.peek();

    if (tok.type !== TokenType.KEYWORD) {
      this.collector.error(
        E.EXPECTED_KEYWORD,
        `Expected a keyword (task, predict, inputs, ...), got '${tok.value}'`,
        { line: tok.line, column: tok.column, length: tok.length },
        "Each line must start with a NeuroLang keyword",
      );
      this.recoverToNextLine();
      return null;
    }

    const kwToken = this.advance();
    const values: ValueNode[] = [];

    while (
      !this.isAtEnd() &&
      !this.check(TokenType.NEWLINE) &&
      !this.check(TokenType.EOF)
    ) {
      const val = this.parseValue();
      if (val) values.push(val);
    }

    if (values.length === 0) {
      this.collector.error(
        E.EXPECTED_VALUE,
        `Keyword '${kwToken.value}' requires at least one value`,
        { line: kwToken.line, column: kwToken.column, length: kwToken.length },
        `Example: ${kwToken.value} <value>`,
      );
      this.recoverToNextLine();
      return null;
    }

    this.skipNewlines();

    return {
      type: "Statement",
      keyword: kwToken.value,
      values,
      location: { line: kwToken.line, column: kwToken.column },
    };
  }

  private parseValue(): ValueNode | null {
    const tok = this.peek();

    if (tok.type === TokenType.NUMBER) {
      this.advance();
      return {
        type: "Number",
        value: parseFloat(tok.value),
        raw: tok.value,
        location: { line: tok.line, column: tok.column },
      };
    }

    if (tok.type === TokenType.IDENT) {
      this.advance();
      return {
        type: "Identifier",
        value: tok.value,
        location: { line: tok.line, column: tok.column },
      };
    }

    if (tok.type === TokenType.STRING) {
      this.advance();
      return {
        type: "String",
        value: tok.value,
        location: { line: tok.line, column: tok.column },
      };
    }

    // Allow keywords used as values (e.g. column named "loss")
    if (tok.type === TokenType.KEYWORD) {
      this.advance();
      return {
        type: "Identifier",
        value: tok.value,
        location: { line: tok.line, column: tok.column },
      };
    }

    this.collector.error(
      E.EXPECTED_VALUE,
      `Unexpected token '${tok.value}', expected a value`,
      { line: tok.line, column: tok.column, length: tok.length },
    );
    this.advance();
    return null;
  }

  private peek(): Token {
    return this.tokens[this.pos] ?? {
      type: TokenType.EOF,
      value: "",
      line: this.tokens.length > 0 ? this.tokens[this.tokens.length - 1]!.line : 1,
      column: 1,
      length: 0,
    };
  }

  private advance(): Token {
    const tok = this.peek();
    this.pos++;
    return tok;
  }

  private check(type: TokenType): boolean {
    return this.peek().type === type;
  }

  private isAtEnd(): boolean {
    return this.peek().type === TokenType.EOF;
  }

  private skipNewlines(): void {
    while (this.check(TokenType.NEWLINE)) {
      this.advance();
    }
  }

  private recoverToNextLine(): void {
    while (!this.isAtEnd() && !this.check(TokenType.NEWLINE)) {
      this.advance();
    }
    this.skipNewlines();
  }
}
