# NeuroLang Language Reference

> **NeuroLang** is a declarative programming language that compiles into neural networks.
> Write what you want to predict — the compiler designs, builds, and trains the model for you.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Syntax Modes](#2-syntax-modes)
3. [Core Keywords](#3-core-keywords)
4. [Architecture Keywords](#4-architecture-keywords)
5. [Training Keywords](#5-training-keywords)
6. [Preprocessing Keywords](#6-preprocessing-keywords)
7. [Data Inspection Keywords](#7-data-inspection-keywords)
8. [Advanced Keywords](#8-advanced-keywords)
9. [Built-in Datasets](#9-built-in-datasets)
10. [CLI Reference](#10-cli-reference)
11. [Code Generation Targets](#11-code-generation-targets)
12. [Natural Language Syntax](#12-natural-language-syntax)
13. [Examples](#13-examples)
14. [Integration](#14-integration)
15. [Error Reference](#15-error-reference)

---

## 1. Quick Start

Create a file with the `.nl` extension and run it:

```
task classification
predict species
inputs petal_length petal_width sepal_length sepal_width
dataset iris.csv
```

```bash
neurolang myfile.nl
```

Output:
```
Architecture: Input(4) → Dense(128, relu) → Dense(64, relu) → Softmax(3)
```

Or write it in plain English:

```
Predict species with petal_length and petal_width from iris.csv
```

Same result. Both syntaxes compile to identical neural networks.

---

## 2. Syntax Modes

NeuroLang supports two syntaxes. You can mix them freely in the same file.

### Keyword Syntax

Structured, explicit, and precise. Best for production use.

```
task regression
predict house_price
inputs size bedrooms location
dataset housing.csv
optimizer adam
epochs 100
learning_rate 0.001
```

### Natural Language Syntax

Plain English sentences. The compiler detects and translates them automatically.

```
Predict house_price with size, bedrooms, and location from housing.csv for 100 epochs
```

If the first line starts with a verb (`predict`, `classify`, `estimate`, `forecast`, `detect`, `learn`, `find`, `compute`, `calculate`, `model`, `determine`), the natural language parser activates automatically.

---

## 3. Core Keywords

These four keywords are the minimum required for any NeuroLang program.

### `task`

Declares the machine learning task type.

```
task classification    # multiple categories (softmax output)
task regression        # continuous value (linear output)
```

**Auto-inferred** when using natural language syntax based on the target column name and dataset.

---

### `predict`

The column name to predict (the target/label).

```
predict species
predict house_price
predict survived
```

---

### `inputs`

Space-separated list of feature column names.

```
inputs petal_length petal_width sepal_length sepal_width
inputs size bedrooms location age
```

**Auto-inferred** from known datasets when not specified in natural language mode.

---

### `dataset`

The CSV file to load. Can be a built-in dataset name or a path to a file.

```
dataset iris.csv
dataset housing.csv
dataset /path/to/mydata.csv
dataset ../data/customers.csv
```

**Built-in datasets** (no file needed): `iris.csv`, `housing.csv`, `titanic.csv`, `wine.csv`, `digits.csv`

---

## 4. Architecture Keywords

### `architecture`

Explicitly set the neural network type. If omitted, the compiler infers from context.

```
architecture mlp          # Multi-layer perceptron (default)
architecture cnn          # Convolutional neural network
architecture lstm         # Long short-term memory (sequence data)
architecture gru          # Gated recurrent unit
architecture rnn          # Basic recurrent network
architecture resnet       # Residual network with skip connections
architecture transformer  # Attention-based transformer encoder
architecture autoencoder  # Encoder-decoder for unsupervised learning
```

---

### `learn`

Controls the learning mode / network depth.

```
learn linear      # Single linear layer, no activation
learn nonlinear   # 2-layer MLP with ReLU (default for classification/regression)
learn deep        # 4-layer deep network
learn auto        # Auto-architecture search — tries 4 sizes and picks the best
```

---

### `layers`

Explicitly define hidden layer sizes as a space-separated list.

```
layers 256 128 64      # Three hidden layers
layers 64              # One hidden layer
layers 512 256 128 64  # Four hidden layers
```

---

### `activation`

Activation function for hidden layers.

```
activation relu      # Rectified Linear Unit (default)
activation tanh      # Hyperbolic tangent
activation sigmoid   # Sigmoid (0–1 output)
activation selu      # Scaled ELU (self-normalizing)
activation elu       # Exponential Linear Unit
activation gelu      # Gaussian Error Linear Unit
activation leakyrelu # Leaky ReLU
```

---

### `loss`

Loss function for training.

```
loss mse              # Mean Squared Error (regression)
loss mae              # Mean Absolute Error (regression)
loss cross_entropy    # Categorical Cross-Entropy (multi-class)
loss binary_cross_entropy  # Binary Cross-Entropy (binary classification)
loss huber            # Huber loss (robust regression)
```

**Auto-selected** based on `task` if not specified.

---

### `batch_norm`

Enable Batch Normalization layers after each hidden layer.

```
batch_norm true
```

---

### CNN-specific keywords

```
filters 32            # Number of convolution filters
kernel_size 3         # Size of the convolution kernel
pool_size 2           # Pooling window size
input_shape 28 28 1   # Image dimensions (height width channels)
```

---

### RNN-specific keywords

```
sequence_length 50    # Length of input sequences
embedding_dim 64      # Embedding vector size
bidirectional true    # Use bidirectional LSTM/GRU
recurrent_dropout 0.2 # Dropout inside recurrent layers
```

---

### `pretrained`

Use a pretrained model as the backbone (transfer learning).

```
pretrained mobilenet
pretrained resnet50
pretrained resnet101
pretrained vgg16
pretrained vgg19
pretrained efficientnet
pretrained inception
pretrained densenet
```

When a pretrained model is set, the compiler generates code using the framework's pretrained weights and attaches a custom classification head.

---

### `freeze_layers`

Number of pretrained layers to freeze during training.

```
pretrained resnet50
freeze_layers 100
```

---

### `targets`

Define multiple output targets for multi-task learning.

```
targets price quality confidence
```

---

## 5. Training Keywords

### `optimizer`

```
optimizer adam     # Adaptive Moment Estimation (default)
optimizer sgd      # Stochastic Gradient Descent
optimizer rmsprop  # RMSProp
optimizer adamw    # Adam with weight decay
```

---

### `epochs`

Number of full passes through the training data.

```
epochs 50
epochs 200
```

**Default**: auto-selected based on dataset size (typically 40–100).

---

### `learning_rate`

Step size for gradient updates.

```
learning_rate 0.001   # Default
learning_rate 0.0001
learning_rate 0.01
```

---

### `batch_size`

Number of samples per gradient update.

```
batch_size 32    # Default
batch_size 64
batch_size 16
```

---

### `dropout`

Dropout rate (fraction of neurons to randomly zero during training).

```
dropout 0.3    # 30% dropout
dropout 0.5
```

---

### `early_stop`

Stop training if validation loss doesn't improve for N epochs.

```
early_stop 10    # Stop if no improvement for 10 epochs
early_stop 5
```

---

### `lr_schedule`

Learning rate schedule.

```
lr_schedule cosine       # Cosine annealing
lr_schedule step         # Step decay
lr_schedule exponential  # Exponential decay
```

---

### `seed`

Random seed for reproducibility.

```
seed 42
seed 0
```

---

### `tune`

Activate hyperparameter tuning. Searches over learning rates `[0.01, 0.001, 0.0001]`, batch sizes `[16, 32, 64]`, and architectures (small/medium/large) using short probe runs.

```
tune true
```

---

### `ensemble`

Train N models with different random seeds and average their predictions (bagging).

```
ensemble 3
ensemble 5
```

---

### `cross_validate`

K-fold cross-validation.

```
cross_validate 5    # 5-fold CV
cross_validate 10
```

---

## 6. Preprocessing Keywords

### `normalize`

Normalize features to zero mean, unit variance.

```
normalize true    # Default: true for MLP/RNN, false for CNN
normalize false
```

---

### `split`

Train/test split fraction (value is the training portion).

```
split 0.8    # 80% train, 20% test (default)
split 0.9    # 90% train, 10% test
split 0.7
```

---

### `validate`

Enable validation during training (uses a portion of the training set).

```
validate true
```

---

### `export`

Save the trained model to a file.

```
export "mymodel"
export "/path/to/model.json"
```

---

### `export_format`

Export the trained model in additional formats.

```
export_format onnx          # ONNX format (cross-framework)
export_format tflite        # TensorFlow Lite (mobile/edge)
export_format savedmodel    # TensorFlow SavedModel
export_format torchscript   # PyTorch TorchScript
export_format coreml        # Apple Core ML (iOS/macOS)
```

---

## 7. Data Inspection Keywords

### `show`

Print a preview of the dataset.

```
show data
```

---

### `describe`

Print summary statistics (mean, std, min, max, quartiles) for each column.

```
describe true
```

---

### `sample`

Print N random rows from the dataset.

```
sample 10
sample 5
```

---

## 8. Advanced Keywords

### `select`

Select specific columns from the dataset.

```
select age fare pclass survived
```

---

### `filter`

Filter rows (basic expression).

```
filter age > 18
```

---

### `augment`

Enable data augmentation (for image datasets).

```
augment true
```

---

## 9. Built-in Datasets

No CSV file needed — these are embedded in the compiler.

| Name | Samples | Features | Task | Target |
|---|---|---|---|---|
| `iris.csv` | 150 | 4 | Classification (3 classes) | `species` |
| `housing.csv` | 506 | 13 | Regression | `price` |
| `titanic.csv` | 891 | 7 | Classification (2 classes) | `survived` |
| `wine.csv` | 178 | 13 | Classification (3 classes) | `quality` |
| `digits.csv` | 1,797 | 64 | Classification (10 classes) | `digit` |

### Feature names

**iris.csv**: `sepal_length sepal_width petal_length petal_width`

**housing.csv**: `crim zn indus chas nox rm age dis rad tax ptratio b lstat`

**titanic.csv**: `pclass age fare sibsp parch sex embarked`

**wine.csv**: `alcohol malic_acid ash alcalinity_of_ash magnesium total_phenols flavanoids nonflavanoid_phenols proanthocyanins color_intensity hue od280_od315_of_diluted_wines proline`

**digits.csv**: `pixel_0` through `pixel_63` (64 pixel values from 8×8 images)

---

## 10. CLI Reference

```bash
# Compile a file (show architecture summary)
neurolang myfile.nl

# Compile and train
neurolang myfile.nl --run

# Show generated code
neurolang myfile.nl --emit-code

# Choose code generation target
neurolang myfile.nl --emit-code --target pytorch
neurolang myfile.nl --emit-code --target keras
neurolang myfile.nl --emit-code --target jax
neurolang myfile.nl --emit-code --target tensorflow
neurolang myfile.nl --emit-code --target summary

# Compare architectures side by side
neurolang myfile.nl --compare mlp lstm resnet
neurolang myfile.nl --compare mlp cnn gru transformer

# Infer a program from a raw CSV file
neurolang --infer mydata.csv

# Show dataset statistics
neurolang myfile.nl --show-data
neurolang myfile.nl --describe

# Feature importance (after training)
neurolang myfile.nl --run --explain

# Interactive REPL
neurolang --repl

# Start HTTP API server
neurolang --serve
neurolang --serve 8080

# Debug internals
neurolang myfile.nl --tokens    # Show token stream
neurolang myfile.nl --ast       # Show Abstract Syntax Tree
neurolang myfile.nl --ir        # Show Intermediate Representation
neurolang myfile.nl --timings   # Show compilation phase timings

# Pipe from stdin
echo "Predict species with petal_length from iris.csv" | neurolang --run

# Benchmark all built-in datasets
neurolang --benchmark
```

---

## 11. Code Generation Targets

NeuroLang can output ready-to-run code for four ML frameworks.

### TensorFlow.js (default)

```bash
neurolang myfile.nl --emit-code --target tensorflow
```

Generates JavaScript/TypeScript using `@tensorflow/tfjs`. Runnable in Node.js or the browser.

### PyTorch

```bash
neurolang myfile.nl --emit-code --target pytorch
```

Generates Python using `torch` and `torch.nn`. Includes a training loop, data loading, and model evaluation.

### Keras

```bash
neurolang myfile.nl --emit-code --target keras
```

Generates Python using `tensorflow.keras`. Sequential model with `model.compile()` and `model.fit()`.

### JAX / Flax

```bash
neurolang myfile.nl --emit-code --target jax
```

Generates Python using `jax`, `flax.linen`, and `optax`. Includes a `jax.jit`-compiled training loop.

---

## 12. Natural Language Syntax

Any sentence starting with a recognized verb is parsed as natural language.

### Recognized verbs

`predict`, `classify`, `estimate`, `forecast`, `determine`, `learn`, `find`, `compute`, `calculate`, `model`, `detect`

### Target and features

```
Predict <target> with <feature1>, <feature2>, and <feature3> from <dataset>
Classify <target> using <features> in <dataset>
Estimate <target> based on <features> from <dataset>
```

### Architecture modifiers

```
... using a CNN
... using an LSTM
... using a GRU
... using a transformer
... using a ResNet
... using an autoencoder
... using a bidirectional LSTM
... with pretrained mobilenet
... with pretrained resnet50
```

### Training modifiers

```
... for 100 epochs
... for 50 rounds
... with learning rate 0.001
... with dropout 0.3
... with batch size 64
... with early stopping 10
... with batch normalization
... with seed 42
... with 5-fold cross validation
... and tuning
... deeply                      (deep network)
... linearly                    (linear model)
... automatically               (auto-architecture search)
... using 3 models              (ensemble of 3)
... with optimizer sgd
... normalized
```

### Export modifiers

```
... and export to onnx
... and export to tflite
... and export to coreml
```

### Multi-line natural language

```
Predict survival with age fare pclass using titanic.csv
For 60 epochs with dropout 0.3 and early stopping 10
```

---

## 13. Examples

### Minimal — one line

```
Predict species with petal_length and petal_width from iris.csv
```

### Standard classification

```
task classification
predict species
inputs petal_length petal_width sepal_length sepal_width
dataset iris.csv
optimizer adam
epochs 60
learning_rate 0.001
dropout 0.3
early_stop 10
```

### Standard regression

```
task regression
predict price
inputs crim zn indus nox rm age dis rad tax ptratio b lstat
dataset housing.csv
loss mse
optimizer adam
epochs 100
learning_rate 0.001
normalize true
split 0.8
```

### Deep network with tuning

```
task classification
predict survived
inputs pclass age fare sibsp parch sex embarked
dataset titanic.csv
learn deep
batch_norm true
dropout 0.4
tune true
early_stop 15
seed 42
```

### CNN for image data

```
task classification
predict digit
inputs pixel_0
dataset digits.csv
architecture cnn
input_shape 8 8 1
filters 32
kernel_size 3
pool_size 2
epochs 50
```

### LSTM for sequences

```
task classification
predict label
inputs feature
dataset sequences.csv
architecture lstm
sequence_length 50
embedding_dim 64
bidirectional true
dropout 0.2
epochs 30
```

### ResNet with residual connections

```
task classification
predict digit
inputs pixel_0
dataset digits.csv
architecture resnet
batch_norm true
dropout 0.3
epochs 80
```

### Transfer learning

```
task classification
predict label
inputs image
dataset myimages.csv
pretrained mobilenet
freeze_layers 80
epochs 20
learning_rate 0.0001
```

### Multi-output

```
task regression
predict price
inputs size bedrooms location age
dataset housing.csv
targets price quality condition
epochs 80
```

### Ensemble

```
task classification
predict species
inputs petal_length petal_width sepal_length sepal_width
dataset iris.csv
ensemble 5
epochs 50
```

### Cross-validation

```
task classification
predict quality
inputs alcohol malic_acid ash magnesium total_phenols flavanoids proline
dataset wine.csv
cross_validate 5
epochs 60
```

### Export to ONNX and CoreML

```
task classification
predict species
inputs petal_length petal_width sepal_length sepal_width
dataset iris.csv
export_format onnx
export_format coreml
export "iris_model"
```

### Compare architectures

```bash
neurolang iris.nl --compare mlp cnn lstm resnet transformer
```

### Auto-architecture search

```
task classification
predict species
inputs petal_length petal_width sepal_length sepal_width
dataset iris.csv
learn auto
```

### Schema detection from raw CSV

```bash
neurolang --infer mydata.csv
```

The compiler reads the CSV, infers the task, target, and input features, and prints a complete NeuroLang program.

---

## 14. Integration

### TypeScript / JavaScript — Tagged Template Literal

```typescript
import { nl } from "neurolang/template";

const result = await nl`
  task classification
  predict species
  inputs petal_length petal_width
  dataset iris.csv
`;

console.log(result.architecture);
console.log(result.generatedCode);
```

### JavaScript — Programmatic API

```javascript
import { compile, generateCode } from "neurolang";

const ir = compile(`
  task regression
  predict price
  inputs size bedrooms
  dataset housing.csv
`);

const pytorchCode = generateCode(ir, "pytorch");
console.log(pytorchCode);
```

### Express Middleware

```typescript
import express from "express";
import { neuroLangMiddleware } from "neurolang/middleware";

const app = express();
app.use("/ml", neuroLangMiddleware());

// POST /ml/compile    { source: "..." }
// POST /ml/generate  { source: "...", target: "pytorch" }
// POST /ml/train     { source: "..." }
```

### HTTP API Server

```bash
neurolang --serve 3000
```

```bash
# Compile
curl -X POST http://localhost:3000/compile \
  -H "Content-Type: application/json" \
  -d '{"source": "Predict species with petal_length from iris.csv"}'

# Generate code
curl -X POST http://localhost:3000/generate \
  -H "Content-Type: application/json" \
  -d '{"source": "...", "target": "pytorch"}'
```

### Python SDK

```python
from neurolang import NeuroLang

nl = NeuroLang()
result = nl.compile("""
    task classification
    predict species
    inputs petal_length petal_width
    dataset iris.csv
""")
print(result["architecture"])
pytorch_code = nl.generate(result["source"], target="pytorch")
```

### Java SDK

```java
import com.neurolang.NeuroLang;

NeuroLang nl = new NeuroLang();
Map<String, Object> result = nl.compile(
    "task classification\n" +
    "predict species\n" +
    "inputs petal_length petal_width\n" +
    "dataset iris.csv"
);
System.out.println(result.get("architecture"));
```

### REPL — Interactive Mode

```bash
neurolang --repl
```

Commands inside the REPL:

| Command | Action |
|---|---|
| `:run` | Compile and train the current program |
| `:reset` | Clear the buffer |
| `:show` | Print current source and IR |
| `:target pytorch` | Switch code generation target |
| `:help` | List all commands |
| `:quit` | Exit |

---

## 15. Error Reference

NeuroLang produces structured error messages with line numbers and suggestions.

### Missing required keyword

```
error  Missing required keyword: 'predict'
  Hint: Add 'predict <column_name>' to specify the target
```

### Unknown architecture

```
error  Unknown architecture: 'bert'
  Hint: Valid architectures are: mlp, cnn, lstm, gru, rnn, resnet, transformer, autoencoder
```

### Unknown task

```
error  Unknown task: 'generation'
  Hint: Valid tasks are: classification, regression
```

### Invalid value

```
error  Invalid value for 'dropout': '1.5'
  Hint: Dropout must be between 0.0 and 1.0
```

### Dataset not found

```
error  Dataset 'mydata.csv' not found
  Hint: Provide an absolute path or use a built-in dataset (iris.csv, housing.csv, titanic.csv, wine.csv, digits.csv)
```

---

*NeuroLang v4.0 — Turn machine learning into programming, not math.*
