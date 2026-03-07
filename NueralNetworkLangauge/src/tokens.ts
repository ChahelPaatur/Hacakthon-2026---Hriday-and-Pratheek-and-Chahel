export enum TokenType {
  KEYWORD = "KEYWORD",
  IDENT = "IDENT",
  NUMBER = "NUMBER",
  STRING = "STRING",
  NEWLINE = "NEWLINE",
  EOF = "EOF",
}

export interface Token {
  type: TokenType;
  value: string;
  line: number;
  column: number;
  length: number;
}

export const KEYWORDS = new Set([
  // Required
  "task",
  "predict",
  "inputs",
  "dataset",
  // Architecture
  "architecture",
  "loss",
  "learn",
  "optimizer",
  "epochs",
  "layers",
  "activation",
  "batch_norm",
  "filters",
  "kernel_size",
  "pool_size",
  "input_shape",
  "sequence_length",
  "embedding_dim",
  "bidirectional",
  "recurrent_dropout",
  // Training
  "batch_size",
  "dropout",
  "learning_rate",
  "early_stop",
  "lr_schedule",
  "seed",
  // Preprocessing
  "normalize",
  "split",
  // Validation & output
  "validate",
  "export",
  // Transfer learning
  "pretrained",
  "freeze_layers",
  // Export
  "export_format",
  // Multi-output
  "targets",
  // Comparison
  "compare",
  // Advanced
  "tune",
  "cross_validate",
  "ensemble",
  // Data inspection
  "show",
  "describe",
  "sample",
  // Data manipulation
  "augment",
  "select",
  "filter",
]);

export function isKeyword(word: string): boolean {
  return KEYWORDS.has(word);
}
