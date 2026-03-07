/**
 * Natural language front-end for NeuroLang.
 *
 * Translates English-like sentences into standard NeuroLang keyword syntax,
 * which then flows through the existing compiler pipeline unchanged.
 *
 * Supported patterns:
 *   Predict species with petal_length & petal_width from iris.csv
 *   Classify survived using age, fare, pclass from titanic.csv deeply
 *   Estimate price from size & bedrooms in housing.csv for 100 epochs
 */

import { resolveDataset } from "./datasets.js";

export interface NaturalParseResult {
  program: string;
  detected: boolean;
}

const VERB_PATTERN =
  /^(predict|classify|estimate|forecast|determine|learn|find|compute|calculate|model|detect)/i;

const FEATURE_SEPS = /\s*(?:&|,|\band\b)\s*/;

const DATASET_INTRO = /\b(?:from|in|on|using|with\s+data(?:set)?)\s+/i;

const MODIFIER_PATTERNS: {
  pattern: RegExp;
  emit: (match: RegExpMatchArray) => string;
}[] = [
  { pattern: /\b(?:deeply|deep\s+learning|deep\s+network)\b/i, emit: () => "learn deep" },
  { pattern: /\b(?:linearly|linear\s+model|linear\s+regression)\b/i, emit: () => "learn linear" },
  { pattern: /\bauto(?:matically|[\s-]?search)?\b/i, emit: () => "learn auto" },
  {
    pattern: /\b(?:using|with)\s+(\d+)\s+models?\b/i,
    emit: (m) => `ensemble ${m[1]}`,
  },
  {
    pattern: /\b(?:ensemble|bag(?:ging)?)\s+(?:of\s+)?(\d+)\b/i,
    emit: (m) => `ensemble ${m[1]}`,
  },
  {
    pattern: /\bfor\s+(\d+)\s+(?:epochs?|rounds?|iterations?|steps?)\b/i,
    emit: (m) => `epochs ${m[1]}`,
  },
  {
    pattern: /\b(?:batch[\s_]?size|batches?\s+of)\s+(\d+)\b/i,
    emit: (m) => `batch_size ${m[1]}`,
  },
  {
    pattern: /\b(?:learning[\s_]?rate|lr|learn(?:ing)?\s+at)\s+([\d.]+)\b/i,
    emit: (m) => `learning_rate ${m[1]}`,
  },
  {
    pattern: /\b(?:with\s+)?dropout\s+([\d.]+)\b/i,
    emit: (m) => `dropout ${m[1]}`,
  },
  {
    pattern: /\bnormalize[d]?\b/i,
    emit: () => "normalize true",
  },
  {
    pattern: /\b(?:cross[\s_]?validate?|cv)\s+(\d+)[\s-]?fold/i,
    emit: (m) => `cross_validate ${m[1]}`,
  },
  {
    pattern: /\b(\d+)[\s-]?fold\s+(?:cross[\s_]?valid)/i,
    emit: (m) => `cross_validate ${m[1]}`,
  },
  {
    pattern: /\b(?:and\s+)?tun(?:e|ing|ed)\b/i,
    emit: () => "tune true",
  },
  {
    pattern: /\b(?:and\s+)?explain(?:ability)?\b/i,
    emit: () => "",  // handled as CLI flag, not language keyword
  },
  {
    pattern: /\bseed\s+(\d+)\b/i,
    emit: (m) => `seed ${m[1]}`,
  },
  {
    pattern: /\b(?:export|save)\s+(?:to\s+)?["']([^"']+)["']/i,
    emit: (m) => `export "${m[1]}"`,
  },
  {
    pattern: /\bfreeze[\s_]?layers?\s+(\d+)\b/i,
    emit: (m) => `freeze_layers ${m[1]}`,
  },
  {
    pattern: /\bexport[\s_]?format\s+(\w+(?:\s+\w+)*)/i,
    emit: (m) => `export_format ${m[1]}`,
  },
  {
    pattern: /\b(?:with\s+)?early[\s_]?stop(?:ping)?\s+(\d+)\b/i,
    emit: (m) => `early_stop ${m[1]}`,
  },
  {
    pattern: /\b(?:with\s+)?batch[\s_]?norm(?:alization)?\b/i,
    emit: () => "batch_norm true",
  },
  {
    pattern: /\b(?:and\s+)?show(?:\s+data)?\b/i,
    emit: () => "show data",
  },
  {
    pattern: /\b(?:and\s+)?describe\b/i,
    emit: () => "describe true",
  },
  {
    pattern: /\b(?:and\s+)?sample\s+(\d+)\b/i,
    emit: (m) => `sample ${m[1]}`,
  },
  {
    pattern: /\b(?:optimizer|optimise|optimize)\s+(\w+)\b/i,
    emit: (m) => `optimizer ${m[1]!.toLowerCase()}`,
  },
  {
    pattern: /\bloss\s+(\w+)\b/i,
    emit: (m) => `loss ${m[1]!.toLowerCase()}`,
  },
];

export function isNaturalSyntax(source: string): boolean {
  const firstLine = source.trim().split(/\n/)[0]?.trim() ?? "";
  return VERB_PATTERN.test(firstLine);
}

export function parseNatural(source: string): NaturalParseResult {
  const trimmed = source.trim();
  if (!isNaturalSyntax(trimmed)) {
    return { program: trimmed, detected: false };
  }

  const lines = trimmed.split(/\n/).map((l) => l.trim()).filter(Boolean);
  const programLines: string[] = [];
  let taskOverride: "classification" | "regression" | null = null;

  for (const line of lines) {
    const parsed = parseSentence(line);
    if (parsed) {
      if (parsed.taskHint) taskOverride = parsed.taskHint;
      programLines.push(...parsed.lines);
    } else {
      // Pass through keyword-syntax lines unchanged
      programLines.push(line);
    }
  }

  // Infer task if not explicitly set
  if (!programLines.some((l) => l.startsWith("task "))) {
    if (taskOverride) {
      programLines.unshift(`task ${taskOverride}`);
    }
  }

  return { program: programLines.join("\n"), detected: true };
}

interface ParsedSentence {
  lines: string[];
  taskHint: "classification" | "regression" | null;
}

const ARCH_PATTERNS: { pattern: RegExp; emit: string }[] = [
  { pattern: /\b(?:using\s+|with\s+)?(?:a\s+)?cnn\b/i, emit: "architecture cnn" },
  { pattern: /\b(?:using\s+|with\s+)?(?:a\s+)?convolutional\b/i, emit: "architecture cnn" },
  { pattern: /\b(?:using\s+|with\s+)?(?:an?\s+)?lstm\b/i, emit: "architecture lstm" },
  { pattern: /\b(?:using\s+|with\s+)?(?:an?\s+)?gru\b/i, emit: "architecture gru" },
  { pattern: /\b(?:using\s+|with\s+)?(?:an?\s+)?rnn\b/i, emit: "architecture rnn" },
  { pattern: /\b(?:using\s+|with\s+)?(?:an?\s+)?recurrent\b/i, emit: "architecture rnn" },
  { pattern: /\b(?:using\s+|with\s+)?(?:an?\s+)?autoencoder\b/i, emit: "architecture autoencoder" },
  { pattern: /\b(?:using\s+|with\s+)?(?:an?\s+)?bi(?:directional|dir)\b/i, emit: "architecture lstm\nbidirectional true" },
  { pattern: /\b(?:using\s+|with\s+)?(?:a\s+)?resnet\b/i, emit: "architecture resnet" },
  { pattern: /\b(?:using\s+|with\s+)?(?:a\s+)?transformer\b/i, emit: "architecture transformer" },
  { pattern: /\b(?:using\s+|with\s+)?(?:a\s+)?residual\b/i, emit: "architecture resnet" },
  // Pretrained models
  { pattern: /\b(?:using\s+|with\s+)?pretrained\s+mobilenet\b/i, emit: "pretrained mobilenet" },
  { pattern: /\b(?:using\s+|with\s+)?pretrained\s+resnet50\b/i, emit: "pretrained resnet50" },
  { pattern: /\b(?:using\s+|with\s+)?pretrained\s+resnet101\b/i, emit: "pretrained resnet101" },
  { pattern: /\b(?:using\s+|with\s+)?pretrained\s+vgg16\b/i, emit: "pretrained vgg16" },
  { pattern: /\b(?:using\s+|with\s+)?pretrained\s+vgg19\b/i, emit: "pretrained vgg19" },
  { pattern: /\b(?:using\s+|with\s+)?pretrained\s+efficientnet\b/i, emit: "pretrained efficientnet" },
  { pattern: /\b(?:using\s+|with\s+)?pretrained\s+inception\b/i, emit: "pretrained inception" },
  { pattern: /\b(?:using\s+|with\s+)?pretrained\s+densenet\b/i, emit: "pretrained densenet" },
  // Export
  { pattern: /\b(?:and\s+)?export\s+(?:to\s+)?onnx\b/i, emit: "export_format onnx" },
  { pattern: /\b(?:and\s+)?export\s+(?:to\s+)?tflite\b/i, emit: "export_format tflite" },
  { pattern: /\b(?:and\s+)?export\s+(?:to\s+)?coreml\b/i, emit: "export_format coreml" },
];

function looksLikeDataset(word: string): boolean {
  if (word.includes(".")) return true;
  const known = ["iris", "housing", "titanic", "wine", "digits", "mnist", "cifar", "sentiment", "sequences"];
  return known.includes(word.toLowerCase());
}

function parseSentence(sentence: string): ParsedSentence | null {
  const verbMatch = sentence.match(VERB_PATTERN);
  if (!verbMatch) return null;

  const verb = verbMatch[1]!.toLowerCase();
  const afterVerb = sentence.slice(verbMatch[0].length).trim();

  let taskHint: "classification" | "regression" | null = null;
  if (["classify", "detect", "determine"].includes(verb)) {
    taskHint = "classification";
  } else if (["estimate", "forecast"].includes(verb)) {
    taskHint = "regression";
  }

  // Strategy: scan for all preposition+word groups, figure out which one
  // introduces the dataset (word looks like a filename or known dataset name).
  // The feature list is everything between the target and the dataset intro.

  // Find all occurrences of dataset-introducing prepositions
  const dsIntros = [
    ...afterVerb.matchAll(/\b(?:from|in|on|using|with\s+data(?:set)?)\s+([\w./-]+)/gi),
  ];

  // Pick the dataset: prefer the one whose captured word looks like a dataset
  let dataset: string | null = null;
  let dsMatchIndex = -1;
  let dsMatchEnd = -1;

  // Try last match first (in "X from features in dataset", "in dataset" is last)
  for (let i = dsIntros.length - 1; i >= 0; i--) {
    const m = dsIntros[i]!;
    if (looksLikeDataset(m[1]!)) {
      dataset = m[1]!;
      dsMatchIndex = m.index!;
      dsMatchEnd = m.index! + m[0].length;
      break;
    }
  }

  // If no match looks like a dataset, take the last one
  if (!dataset && dsIntros.length > 0) {
    const last = dsIntros[dsIntros.length - 1]!;
    dataset = last[1]!;
    dsMatchIndex = last.index!;
    dsMatchEnd = last.index! + last[0].length;
  }

  // Split: before dataset intro = target + features, after = modifiers
  const beforeDS = dsMatchIndex >= 0 ? afterVerb.slice(0, dsMatchIndex).trim() : afterVerb;
  const afterDS = dsMatchEnd >= 0 ? afterVerb.slice(dsMatchEnd).trim() : "";

  // Extract target and features from beforeDS
  // Pattern: <target> (with|using|from|based on|given) <features>
  const tfSplit = beforeDS.match(
    /^(\w+)\s+(?:with|using|from|based\s+on|given)\s+(.+)$/i,
  );

  let target: string;
  let features: string[];

  if (tfSplit) {
    target = tfSplit[1]!;
    const featureStr = tfSplit[2]!.trim();
    features = featureStr.split(FEATURE_SEPS).map((f) => f.trim()).filter(Boolean);
  } else {
    const simpleTarget = beforeDS.match(/^(\w+)/);
    if (!simpleTarget) return null;
    target = simpleTarget[1]!;
    features = [];
  }

  // Auto-fill features from known dataset when none specified
  if (features.length === 0 && dataset) {
    const ds = resolveDataset(dataset);
    if (ds) {
      features = ds.featureNames;
    }
  }

  const lines: string[] = [];
  lines.push(`predict ${target}`);
  if (features.length > 0) {
    lines.push(`inputs ${features.join(" ")}`);
  }
  if (dataset) {
    lines.push(`dataset ${dataset}`);
  }

  // Infer task from target name first (more specific)
  if (!taskHint) {
    const tLower = target.toLowerCase();
    if (["price", "cost", "salary", "amount", "revenue", "value", "weight", "height", "score", "income", "age", "rating", "temperature", "profit", "loss"].includes(tLower)) {
      taskHint = "regression";
    }
  }

  // Then try dataset name
  if (!taskHint && dataset) {
    const dsLower = dataset.toLowerCase();
    if (dsLower.includes("housing") || dsLower.includes("price") || dsLower.includes("regress")) {
      taskHint = "regression";
    } else {
      taskHint = "classification";
    }
  }

  if (!taskHint) {
    taskHint = "classification";
  }

  // Extract architecture patterns (CNN, LSTM, GRU, autoencoder, etc.)
  for (const arch of ARCH_PATTERNS) {
    if (arch.pattern.test(sentence)) {
      lines.push(...arch.emit.split("\n"));
      break;
    }
  }

  // Extract modifiers from the full sentence (they can appear anywhere)
  const modifierText = `${afterVerb} ${afterDS}`;
  for (const mod of MODIFIER_PATTERNS) {
    const m = modifierText.match(mod.pattern);
    if (m) {
      const emitted = mod.emit(m);
      if (emitted) lines.push(emitted);
    }
  }

  return { lines, taskHint };
}
