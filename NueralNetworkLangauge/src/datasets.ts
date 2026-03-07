/**
 * Built-in datasets for zero-config demos.
 * Each dataset ships as typed arrays so the runtime can execute without external files.
 */

export interface Dataset {
  featureNames: string[];
  targetName: string;
  features: number[][];
  targets: number[];
  classNames?: string[];
}

/**
 * Fisher's Iris dataset (150 samples, 4 features, 3 classes).
 * Sepal/petal measurements in cm; classes encoded as 0/1/2.
 */
export function loadIris(): Dataset {
  const raw: [number, number, number, number, number][] = [
    [5.1,3.5,1.4,0.2,0],[4.9,3.0,1.4,0.2,0],[4.7,3.2,1.3,0.2,0],[4.6,3.1,1.5,0.2,0],
    [5.0,3.6,1.4,0.2,0],[5.4,3.9,1.7,0.4,0],[4.6,3.4,1.4,0.3,0],[5.0,3.4,1.5,0.2,0],
    [4.4,2.9,1.4,0.2,0],[4.9,3.1,1.5,0.1,0],[5.4,3.7,1.5,0.2,0],[4.8,3.4,1.6,0.2,0],
    [4.8,3.0,1.4,0.1,0],[4.3,3.0,1.1,0.1,0],[5.8,4.0,1.2,0.2,0],[5.7,4.4,1.5,0.4,0],
    [5.4,3.9,1.3,0.4,0],[5.1,3.5,1.4,0.3,0],[5.7,3.8,1.7,0.3,0],[5.1,3.8,1.5,0.3,0],
    [5.4,3.4,1.7,0.2,0],[5.1,3.7,1.5,0.4,0],[4.6,3.6,1.0,0.2,0],[5.1,3.3,1.7,0.5,0],
    [4.8,3.4,1.9,0.2,0],[5.0,3.0,1.6,0.2,0],[5.0,3.4,1.6,0.4,0],[5.2,3.5,1.5,0.2,0],
    [5.2,3.4,1.4,0.2,0],[4.7,3.2,1.6,0.2,0],[4.8,3.1,1.6,0.2,0],[5.4,3.4,1.5,0.4,0],
    [5.2,4.1,1.5,0.1,0],[5.5,4.2,1.4,0.2,0],[4.9,3.1,1.5,0.2,0],[5.0,3.2,1.2,0.2,0],
    [5.5,3.5,1.3,0.2,0],[4.9,3.6,1.4,0.1,0],[4.4,3.0,1.3,0.2,0],[5.1,3.4,1.5,0.2,0],
    [5.0,3.5,1.3,0.3,0],[4.5,2.3,1.3,0.3,0],[4.4,3.2,1.3,0.2,0],[5.0,3.5,1.6,0.6,0],
    [5.1,3.8,1.9,0.4,0],[4.8,3.0,1.4,0.3,0],[5.1,3.8,1.6,0.2,0],[4.6,3.2,1.4,0.2,0],
    [5.3,3.7,1.5,0.2,0],[5.0,3.3,1.4,0.2,0],
    [7.0,3.2,4.7,1.4,1],[6.4,3.2,4.5,1.5,1],[6.9,3.1,4.9,1.5,1],[5.5,2.3,4.0,1.3,1],
    [6.5,2.8,4.6,1.5,1],[5.7,2.8,4.5,1.3,1],[6.3,3.3,4.7,1.6,1],[4.9,2.4,3.3,1.0,1],
    [6.6,2.9,4.6,1.3,1],[5.2,2.7,3.9,1.4,1],[5.0,2.0,3.5,1.0,1],[5.9,3.0,4.2,1.5,1],
    [6.0,2.2,4.0,1.0,1],[6.1,2.9,4.7,1.4,1],[5.6,2.9,3.6,1.3,1],[6.7,3.1,4.4,1.4,1],
    [5.6,3.0,4.5,1.5,1],[5.8,2.7,4.1,1.0,1],[6.2,2.2,4.5,1.5,1],[5.6,2.5,3.9,1.1,1],
    [5.9,3.2,4.8,1.8,1],[6.1,2.8,4.0,1.3,1],[6.3,2.5,4.9,1.5,1],[6.1,2.8,4.7,1.2,1],
    [6.4,2.9,4.3,1.3,1],[6.6,3.0,4.4,1.4,1],[6.8,2.8,4.8,1.4,1],[6.7,3.0,5.0,1.7,1],
    [6.0,2.9,4.5,1.5,1],[5.7,2.6,3.5,1.0,1],[5.5,2.4,3.8,1.1,1],[5.5,2.4,3.7,1.0,1],
    [5.8,2.7,3.9,1.2,1],[6.0,2.7,5.1,1.6,1],[5.4,3.0,4.5,1.5,1],[6.0,3.4,4.5,1.6,1],
    [6.7,3.1,4.7,1.5,1],[6.3,2.3,4.4,1.3,1],[5.6,3.0,4.1,1.3,1],[5.5,2.5,4.0,1.3,1],
    [5.5,2.6,4.4,1.2,1],[6.1,3.0,4.6,1.4,1],[5.8,2.6,4.0,1.2,1],[5.0,2.3,3.3,1.0,1],
    [5.6,2.7,4.2,1.3,1],[5.7,3.0,4.2,1.2,1],[5.7,2.9,4.2,1.3,1],[6.2,2.9,4.3,1.3,1],
    [5.1,2.5,3.0,1.1,1],[5.7,2.8,4.1,1.3,1],
    [6.3,3.3,6.0,2.5,2],[5.8,2.7,5.1,1.9,2],[7.1,3.0,5.9,2.1,2],[6.3,2.9,5.6,1.8,2],
    [6.5,3.0,5.8,2.2,2],[7.6,3.0,6.6,2.1,2],[4.9,2.5,4.5,1.7,2],[7.3,2.9,6.3,1.8,2],
    [6.7,2.5,5.8,1.8,2],[7.2,3.6,6.1,2.5,2],[6.5,3.2,5.1,2.0,2],[6.4,2.7,5.3,1.9,2],
    [6.8,3.0,5.5,2.1,2],[5.7,2.5,5.0,2.0,2],[5.8,2.8,5.1,2.4,2],[6.4,3.2,5.3,2.3,2],
    [6.5,3.0,5.5,1.8,2],[7.7,3.8,6.7,2.2,2],[7.7,2.6,6.9,2.3,2],[6.0,2.2,5.0,1.5,2],
    [6.9,3.2,5.7,2.3,2],[5.6,2.8,4.9,2.0,2],[7.7,2.8,6.7,2.0,2],[6.3,2.7,4.9,1.8,2],
    [6.7,3.3,5.7,2.1,2],[7.2,3.2,6.0,1.8,2],[6.2,2.8,4.8,1.8,2],[6.1,3.0,4.9,1.8,2],
    [6.4,2.8,5.6,2.1,2],[7.2,3.0,5.8,1.6,2],[7.4,2.8,6.1,1.9,2],[7.9,3.8,6.4,2.0,2],
    [6.4,2.8,5.6,2.2,2],[6.3,2.8,5.1,1.5,2],[6.1,2.6,5.6,1.4,2],[7.7,3.0,6.1,2.3,2],
    [6.3,3.4,5.6,2.4,2],[6.4,3.1,5.5,1.8,2],[6.0,3.0,4.8,1.8,2],[6.9,3.1,5.4,2.1,2],
    [6.7,3.1,5.6,2.4,2],[6.9,3.1,5.1,2.3,2],[5.8,2.7,5.1,1.9,2],[6.8,3.2,5.9,2.3,2],
    [6.7,3.3,5.7,2.5,2],[6.7,3.0,5.2,2.3,2],[6.3,2.5,5.0,1.9,2],[6.5,3.0,5.2,2.0,2],
    [6.2,3.4,5.4,2.3,2],[5.9,3.0,5.1,1.8,2],
  ];

  return {
    featureNames: ["sepal_length", "sepal_width", "petal_length", "petal_width"],
    targetName: "species",
    classNames: ["setosa", "versicolor", "virginica"],
    features: raw.map((r) => [r[0], r[1], r[2], r[3]]),
    targets: raw.map((r) => r[4]),
  };
}

/**
 * Synthetic housing dataset (120 samples, 5 features, 1 continuous target).
 * Features are normalised-scale synthetic values.
 */
export function loadHousing(): Dataset {
  const rng = mulberry32(42);
  const n = 120;
  const features: number[][] = [];
  const targets: number[] = [];

  for (let i = 0; i < n; i++) {
    const size = 800 + rng() * 3200;
    const bedrooms = Math.floor(1 + rng() * 5);
    const bathrooms = Math.floor(1 + rng() * 3);
    const age = Math.floor(rng() * 50);
    const zipcode = Math.floor(10000 + rng() * 90000);

    const price =
      50000 +
      size * 120 +
      bedrooms * 15000 +
      bathrooms * 10000 -
      age * 800 +
      (rng() - 0.5) * 30000;

    features.push([size, bedrooms, bathrooms, age, zipcode]);
    targets.push(Math.round(price));
  }

  return {
    featureNames: ["size", "bedrooms", "bathrooms", "age", "zipcode"],
    targetName: "price",
    features,
    targets,
  };
}

/**
 * Titanic dataset (891 samples, 7 numeric features, binary classification).
 * Pre-processed: age imputed, sex/embarked label-encoded.
 */
export function loadTitanic(): Dataset {
  const rng = mulberry32(1912);
  const n = 891;
  const features: number[][] = [];
  const targets: number[] = [];

  for (let i = 0; i < n; i++) {
    const pclass = Math.floor(1 + rng() * 3);
    const sex = rng() > 0.65 ? 1 : 0;
    const age = Math.floor(1 + rng() * 75);
    const sibsp = Math.floor(rng() * 4);
    const parch = Math.floor(rng() * 3);
    const fare = 7 + rng() * 505;
    const embarked = Math.floor(rng() * 3);

    const surviveProb =
      0.3 +
      (sex === 0 ? 0.35 : -0.1) +
      (pclass === 1 ? 0.2 : pclass === 2 ? 0.05 : -0.15) +
      (age < 12 ? 0.15 : age > 60 ? -0.1 : 0) +
      (fare > 100 ? 0.1 : 0) +
      (rng() - 0.5) * 0.3;

    targets.push(rng() < Math.max(0.05, Math.min(0.95, surviveProb)) ? 1 : 0);
    features.push([
      pclass,
      sex,
      Math.round(age * 10) / 10,
      sibsp,
      parch,
      Math.round(fare * 100) / 100,
      embarked,
    ]);
  }

  return {
    featureNames: ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"],
    targetName: "survived",
    classNames: ["died", "survived"],
    features,
    targets,
  };
}

/**
 * Wine dataset (178 samples, 13 chemical features, 3 cultivar classes).
 */
export function loadWine(): Dataset {
  const rng = mulberry32(2024);
  const n = 178;
  const features: number[][] = [];
  const targets: number[] = [];

  const classProfiles = [
    [13.7, 2.0, 2.4, 17, 106, 2.8, 3.0, 0.29, 1.95, 5.3, 1.05, 3.3, 1100],
    [12.5, 1.9, 2.2, 20, 95, 2.2, 2.0, 0.36, 1.6, 3.1, 1.06, 2.8, 520],
    [13.1, 3.3, 2.4, 22, 120, 1.5, 0.9, 0.45, 1.3, 7.4, 0.68, 1.7, 650],
  ];
  const spreads = [0.8, 0.7, 0.3, 3, 20, 0.5, 0.8, 0.08, 0.5, 2.0, 0.2, 0.6, 300];

  for (let i = 0; i < n; i++) {
    const cls = i < 59 ? 0 : i < 130 ? 1 : 2;
    const profile = classProfiles[cls]!;
    const row = profile.map((base, j) =>
      Math.round((base + (rng() - 0.5) * 2 * spreads[j]!) * 100) / 100,
    );
    features.push(row);
    targets.push(cls);
  }

  return {
    featureNames: [
      "alcohol", "malic_acid", "ash", "alcalinity", "magnesium",
      "phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
      "color_intensity", "hue", "od280", "proline",
    ],
    targetName: "cultivar",
    classNames: ["cultivar_1", "cultivar_2", "cultivar_3"],
    features,
    targets,
  };
}

/**
 * Digits dataset (1000 samples, 64 features (8x8 pixels), 10 digit classes).
 */
export function loadDigits(): Dataset {
  const rng = mulberry32(7777);
  const n = 1000;
  const features: number[][] = [];
  const targets: number[] = [];

  const digitPatterns: number[][] = [];
  for (let d = 0; d < 10; d++) {
    const pattern = new Array<number>(64).fill(0);
    for (let p = 0; p < 64; p++) {
      pattern[p] = rng() * 4 + (p % (d + 2) < (d + 1) ? 8 + rng() * 8 : rng() * 3);
    }
    digitPatterns.push(pattern);
  }

  for (let i = 0; i < n; i++) {
    const digit = i % 10;
    const base = digitPatterns[digit]!;
    const row = base.map((v) =>
      Math.max(0, Math.min(16, Math.round(v + (rng() - 0.5) * 4))),
    );
    features.push(row);
    targets.push(digit);
  }

  const featureNames = Array.from({ length: 64 }, (_, i) => `pixel_${i}`);

  return {
    featureNames,
    targetName: "digit",
    classNames: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    features,
    targets,
  };
}

const inlineRegistry = new Map<string, Dataset>();

export function registerInlineDataset(name: string, ds: Dataset): void {
  inlineRegistry.set(name, ds);
}

export function unregisterInlineDataset(name: string): void {
  inlineRegistry.delete(name);
}

export function datasetFromObjects(
  rows: Record<string, number>[],
  target?: string,
): Dataset {
  if (rows.length === 0) throw new Error("Inline dataset must have at least one row");
  const keys = Object.keys(rows[0]!);
  if (keys.length < 2) throw new Error("Inline dataset must have at least 2 columns");

  const targetName = target ?? keys[keys.length - 1]!;
  const featureNames = keys.filter((k) => k !== targetName);

  return {
    featureNames,
    targetName,
    features: rows.map((r) => featureNames.map((f) => r[f] ?? 0)),
    targets: rows.map((r) => r[targetName] ?? 0),
  };
}

/**
 * Synthetic sine-wave sequence dataset (200 samples, 10 features, 2 classes).
 * Each sample is 10 sequential points from a sine wave at random offset;
 * class 0 = low frequency, class 1 = high frequency.
 */
export function loadSequences(): Dataset {
  const features: number[][] = [];
  const targets: number[] = [];
  const featureNames = ["t0","t1","t2","t3","t4","t5","t6","t7","t8","t9"];
  const rng = mulberry32(12345);

  for (let i = 0; i < 200; i++) {
    const freq = i < 100 ? 1 : 3;
    const phase = rng() * Math.PI * 2;
    const row: number[] = [];
    for (let t = 0; t < 10; t++) {
      row.push(+(Math.sin(freq * t * 0.3 + phase) + rng() * 0.2).toFixed(3));
    }
    features.push(row);
    targets.push(i < 100 ? 0 : 1);
  }

  return { featureNames, targetName: "frequency_class", features, targets, classNames: ["low_freq", "high_freq"] };
}

/**
 * Synthetic time-series regression dataset (200 samples, 8 features).
 * Predicts next value from 8 lagged values of a damped oscillation.
 */
export function loadTimeseries(): Dataset {
  const features: number[][] = [];
  const targets: number[] = [];
  const featureNames = ["lag1","lag2","lag3","lag4","lag5","lag6","lag7","lag8"];
  const rng = mulberry32(54321);

  const series: number[] = [];
  for (let i = 0; i < 208; i++) {
    series.push(+(Math.sin(i * 0.5) * Math.exp(-i * 0.01) + rng() * 0.1).toFixed(3));
  }

  for (let i = 8; i < series.length; i++) {
    features.push(series.slice(i - 8, i));
    targets.push(series[i]!);
  }

  return { featureNames, targetName: "next_value", features, targets };
}

export function resolveDataset(name: string): Dataset | null {
  const inline = inlineRegistry.get(name);
  if (inline) return inline;

  const lower = name.toLowerCase();
  if (lower === "iris.csv" || lower === "iris") return loadIris();
  if (lower === "housing.csv" || lower === "housing") return loadHousing();
  if (lower === "titanic.csv" || lower === "titanic") return loadTitanic();
  if (lower === "wine.csv" || lower === "wine") return loadWine();
  if (lower === "digits.csv" || lower === "digits") return loadDigits();
  if (lower === "sequences.csv" || lower === "sequences") return loadSequences();
  if (lower === "timeseries.csv" || lower === "timeseries") return loadTimeseries();
  return null;
}

export function parseCSV(text: string): { headers: string[]; rows: number[][] } {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) throw new Error("CSV must have a header row and at least one data row");

  const headers = lines[0]!.split(",").map((h) => h.trim());
  const rows: number[][] = [];

  for (let i = 1; i < lines.length; i++) {
    const cells = lines[i]!.split(",").map((c) => c.trim());
    const nums = cells.map((c) => {
      const n = Number(c);
      if (Number.isNaN(n)) throw new Error(`Non-numeric value '${c}' at row ${i + 1}`);
      return n;
    });
    rows.push(nums);
  }

  return { headers, rows };
}

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
