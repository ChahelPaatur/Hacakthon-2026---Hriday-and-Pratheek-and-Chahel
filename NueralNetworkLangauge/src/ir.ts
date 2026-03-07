export type TaskType = "classification" | "regression";
export type ArchitectureType = "mlp" | "cnn" | "rnn" | "autoencoder" | "resnet" | "transformer";

export type PretrainedModel =
  | "mobilenet" | "resnet50" | "resnet101" | "vgg16" | "vgg19"
  | "efficientnet" | "inception" | "densenet" | "none";

export type ExportFormat = "onnx" | "tflite" | "savedmodel" | "torchscript" | "coreml";

export interface DenseLayerSpec {
  kind: "dense";
  units: number;
  activation: string;
}

export interface DropoutLayerSpec {
  kind: "dropout";
  rate: number;
}

export interface BatchNormLayerSpec {
  kind: "batchnorm";
}

export interface Conv2DLayerSpec {
  kind: "conv2d";
  filters: number;
  kernelSize: number;
  strides: number;
  padding: "same" | "valid";
  activation: string;
}

export interface MaxPool2DLayerSpec {
  kind: "maxpool2d";
  poolSize: number;
}

export interface FlattenLayerSpec {
  kind: "flatten";
}

export interface GlobalAvgPool2DLayerSpec {
  kind: "globalavgpool2d";
}

export interface ReshapeLayerSpec {
  kind: "reshape";
  targetShape: number[];
}

export interface LSTMLayerSpec {
  kind: "lstm";
  units: number;
  returnSequences: boolean;
}

export interface GRULayerSpec {
  kind: "gru";
  units: number;
  returnSequences: boolean;
}

export interface EmbeddingLayerSpec {
  kind: "embedding";
  inputDim: number;
  outputDim: number;
}

export interface BidirectionalLayerSpec {
  kind: "bidirectional";
  wrapped: LSTMLayerSpec | GRULayerSpec;
}

export interface ResidualBlockLayerSpec {
  kind: "residual";
  filters: number;
  kernelSize: number;
  strides: number;
}

export interface MultiHeadAttentionLayerSpec {
  kind: "multihead_attention";
  heads: number;
  keyDim: number;
}

export interface AddLayerSpec {
  kind: "add";
}

export interface LayerNormLayerSpec {
  kind: "layernorm";
}

export type LayerSpec =
  | DenseLayerSpec
  | DropoutLayerSpec
  | BatchNormLayerSpec
  | Conv2DLayerSpec
  | MaxPool2DLayerSpec
  | FlattenLayerSpec
  | GlobalAvgPool2DLayerSpec
  | ReshapeLayerSpec
  | LSTMLayerSpec
  | GRULayerSpec
  | EmbeddingLayerSpec
  | BidirectionalLayerSpec
  | ResidualBlockLayerSpec
  | MultiHeadAttentionLayerSpec
  | AddLayerSpec
  | LayerNormLayerSpec;

export type LossFunction =
  | "meanSquaredError"
  | "meanAbsoluteError"
  | "categoricalCrossentropy"
  | "binaryCrossentropy"
  | "huberLoss";

export type LRScheduleType = "cosine" | "step" | "exponential" | "none";

export interface OptimizerSpec {
  name: "adam" | "sgd" | "rmsprop" | "adamw";
  learningRate: number;
}

export interface EarlyStopSpec {
  patience: number;
  minDelta: number;
  monitor: string;
}

export interface LRScheduleSpec {
  type: LRScheduleType;
  decayRate?: number;
  decaySteps?: number;
}

export interface NeuralNetworkIR {
  task: TaskType;
  target: string;
  targets?: string[];
  features: string[];
  dataset: string;

  architecture: {
    type: ArchitectureType;
    inputSize: number;
    inputShape?: number[];
    outputSize: number;
    outputSizes?: number[];
    layers: LayerSpec[];
    batchNorm: boolean;
    pretrained?: PretrainedModel;
    freezeLayers?: number;
  };

  training: {
    loss: LossFunction;
    optimizer: OptimizerSpec;
    epochs: number;
    batchSize: number;
    validationSplit: number;
    earlyStop?: EarlyStopSpec;
    lrSchedule: LRScheduleSpec;
    tune?: boolean;
    ensemble?: number;
  };

  preprocessing: {
    normalize: boolean;
    trainTestSplit: number;
    seed?: number;
    crossValidation?: number;
  };

  output: {
    exportPath?: string;
    exportFormats?: ExportFormat[];
  };

  data: {
    show?: boolean | number;
    describe?: boolean;
    sample?: number;
    augment?: string;
    select?: string[];
    filter?: string;
  };

  meta: {
    learnMode: string;
    parameterCount: number;
  };
}
