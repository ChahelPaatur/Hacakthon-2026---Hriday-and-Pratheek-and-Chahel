# NeuroLang for VS Code

Language support for **NeuroLang** — the declarative programming language that compiles into neural networks.

## Features

- **Syntax Highlighting** — Full colorization for keywords, values, datasets, and natural language syntax
- **Autocomplete** — IntelliSense for all keywords, architectures, optimizers, pretrained models, and export formats
- **Hover Documentation** — Hover over any keyword to see docs, valid values, and examples
- **Snippets** — Quick templates for classification, regression, CNN, LSTM, ResNet, Transformer, and more
- **Run Commands** — Compile, train, and generate code directly from the editor
- **Architecture Comparison** — Compare multiple architectures with one click

## Quick Start

1. Create a `.nl` file
2. Start typing or use a snippet (e.g., type `classify` and press Tab)
3. Press `Cmd+Shift+R` (Mac) or `Ctrl+Shift+R` to compile and train

## Example

```neurolang
Predict species from iris with sepal_length & petal_width
```

Or with full keyword syntax:

```neurolang
task classification
predict species
inputs sepal_length sepal_width petal_length petal_width
dataset iris
architecture resnet
pretrained mobilenet
learn deep
epochs 30
export_format onnx tflite
```

## Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| NeuroLang: Compile | `Cmd+Shift+B` | Compile the current file |
| NeuroLang: Compile & Train | `Cmd+Shift+R` | Compile and train with TensorFlow.js |
| NeuroLang: Generate Code | — | Generate PyTorch/Keras/JAX/TF.js code |
| NeuroLang: Compare Architectures | — | Compare MLP vs CNN vs LSTM etc. |
| NeuroLang: Run Benchmark | — | Run the built-in benchmark suite |

## Snippets

| Prefix | Description |
|--------|-------------|
| `classify` | Classification model template |
| `regress` | Regression model template |
| `cnn` | CNN architecture |
| `lstm` | LSTM architecture |
| `resnet` | ResNet with skip connections |
| `transformer` | Transformer/attention model |
| `transfer` | Transfer learning with pretrained model |
| `fullmodel` | Full model with all options |
| `nlclassify` | Natural language classification |
| `export` | Model export format |

## Requirements

- Node.js 18+
- `neurolang` package installed (`npm install -g neurolang`)
