export interface SourceLocation {
  line: number;
  column: number;
}

export interface ProgramNode {
  type: "Program";
  body: StatementNode[];
}

export interface StatementNode {
  type: "Statement";
  keyword: string;
  values: ValueNode[];
  location: SourceLocation;
}

export interface IdentifierValue {
  type: "Identifier";
  value: string;
  location: SourceLocation;
}

export interface NumberValue {
  type: "Number";
  value: number;
  raw: string;
  location: SourceLocation;
}

export interface StringValue {
  type: "String";
  value: string;
  location: SourceLocation;
}

export type ValueNode = IdentifierValue | NumberValue | StringValue;
