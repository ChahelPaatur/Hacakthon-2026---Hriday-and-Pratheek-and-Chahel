import * as vscode from "vscode";
import * as path from "path";

let outputChannel: vscode.OutputChannel;

export function activate(context: vscode.ExtensionContext) {
  outputChannel = vscode.window.createOutputChannel("NeuroLang");

  context.subscriptions.push(
    vscode.commands.registerCommand("neurolang.compile", () => runNeurolang([])),
    vscode.commands.registerCommand("neurolang.run", () => runNeurolang(["--run"])),
    vscode.commands.registerCommand("neurolang.emitCode", () => emitCode()),
    vscode.commands.registerCommand("neurolang.showIR", () => runNeurolang(["--ir"])),
    vscode.commands.registerCommand("neurolang.benchmark", () => runInTerminal(["--benchmark"])),
    vscode.commands.registerCommand("neurolang.compare", () => compareArchitectures()),
  );

  if (vscode.workspace.getConfiguration("neurolang").get("autoCompileOnSave")) {
    context.subscriptions.push(
      vscode.workspace.onDidSaveTextDocument((doc) => {
        if (doc.languageId === "neurolang") {
          runNeurolang([]);
        }
      }),
    );
  }

  const hoverProvider = vscode.languages.registerHoverProvider("neurolang", {
    provideHover(document, position) {
      const range = document.getWordRangeAtPosition(position);
      if (!range) return undefined;
      const word = document.getText(range).toLowerCase();
      const doc = KEYWORD_DOCS[word];
      if (!doc) return undefined;
      const md = new vscode.MarkdownString();
      md.appendMarkdown(`**${word}** — ${doc.description}\n\n`);
      if (doc.values) md.appendMarkdown(`Values: \`${doc.values}\`\n\n`);
      if (doc.example) md.appendCodeblock(doc.example, "neurolang");
      return new vscode.Hover(md);
    },
  });
  context.subscriptions.push(hoverProvider);

  const completionProvider = vscode.languages.registerCompletionItemProvider(
    "neurolang",
    {
      provideCompletionItems(document, position) {
        const linePrefix = document.lineAt(position).text.substring(0, position.character).trim();
        const items: vscode.CompletionItem[] = [];

        if (linePrefix === "" || /^[a-z_]*$/i.test(linePrefix)) {
          for (const [keyword, doc] of Object.entries(KEYWORD_DOCS)) {
            const item = new vscode.CompletionItem(keyword, vscode.CompletionItemKind.Keyword);
            item.detail = doc.description;
            if (doc.insertText) item.insertText = new vscode.SnippetString(doc.insertText);
            items.push(item);
          }
        }

        if (/^architecture\s*/i.test(linePrefix)) {
          for (const arch of ["mlp", "cnn", "lstm", "gru", "rnn", "resnet", "transformer", "autoencoder"]) {
            items.push(new vscode.CompletionItem(arch, vscode.CompletionItemKind.EnumMember));
          }
        }

        if (/^pretrained\s*/i.test(linePrefix)) {
          for (const model of ["mobilenet", "resnet50", "resnet101", "vgg16", "vgg19", "efficientnet", "inception", "densenet"]) {
            items.push(new vscode.CompletionItem(model, vscode.CompletionItemKind.EnumMember));
          }
        }

        if (/^export_format\s*/i.test(linePrefix)) {
          for (const fmt of ["onnx", "tflite", "savedmodel", "torchscript", "coreml"]) {
            items.push(new vscode.CompletionItem(fmt, vscode.CompletionItemKind.EnumMember));
          }
        }

        if (/^optimizer\s*/i.test(linePrefix)) {
          for (const opt of ["adam", "sgd", "rmsprop", "adamw"]) {
            items.push(new vscode.CompletionItem(opt, vscode.CompletionItemKind.EnumMember));
          }
        }

        if (/^task\s*/i.test(linePrefix)) {
          items.push(new vscode.CompletionItem("classification", vscode.CompletionItemKind.EnumMember));
          items.push(new vscode.CompletionItem("regression", vscode.CompletionItemKind.EnumMember));
        }

        if (/^learn\s*/i.test(linePrefix)) {
          for (const mode of ["linear", "nonlinear", "deep", "auto"]) {
            items.push(new vscode.CompletionItem(mode, vscode.CompletionItemKind.EnumMember));
          }
        }

        if (/^dataset\s*/i.test(linePrefix)) {
          for (const ds of ["iris", "housing", "titanic", "wine", "digits", "sequences", "timeseries"]) {
            items.push(new vscode.CompletionItem(ds, vscode.CompletionItemKind.File));
          }
        }

        return items;
      },
    },
    " ",
  );
  context.subscriptions.push(completionProvider);
}

function getFilePath(): string | undefined {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showWarningMessage("No active NeuroLang file");
    return undefined;
  }
  if (editor.document.languageId !== "neurolang") {
    vscode.window.showWarningMessage("Active file is not a .nl file");
    return undefined;
  }
  return editor.document.fileName;
}

function runNeurolang(extraArgs: string[]) {
  const filePath = getFilePath();
  if (!filePath) return;

  const terminal = vscode.window.createTerminal({ name: "NeuroLang" });
  const args = [JSON.stringify(filePath), ...extraArgs].join(" ");
  terminal.sendText(`npx neurolang ${args}`);
  terminal.show();
}

function runInTerminal(args: string[]) {
  const terminal = vscode.window.createTerminal({ name: "NeuroLang" });
  terminal.sendText(`npx neurolang ${args.join(" ")}`);
  terminal.show();
}

async function emitCode() {
  const filePath = getFilePath();
  if (!filePath) return;

  const target = vscode.workspace.getConfiguration("neurolang").get("codegenTarget", "tensorflow");

  const targets = ["tensorflow", "pytorch", "keras", "jax", "summary"];
  const selected = await vscode.window.showQuickPick(targets, {
    placeHolder: `Select codegen target (current: ${target})`,
  });

  if (!selected) return;

  const terminal = vscode.window.createTerminal({ name: "NeuroLang Code" });
  terminal.sendText(`npx neurolang ${JSON.stringify(filePath)} --emit-code --target ${selected}`);
  terminal.show();
}

async function compareArchitectures() {
  const filePath = getFilePath();
  if (!filePath) return;

  const archs = await vscode.window.showQuickPick(
    ["mlp", "cnn", "lstm", "gru", "resnet", "transformer", "autoencoder"],
    { placeHolder: "Select architectures to compare", canPickMany: true },
  );

  if (!archs || archs.length < 2) {
    vscode.window.showWarningMessage("Select at least 2 architectures to compare");
    return;
  }

  const terminal = vscode.window.createTerminal({ name: "NeuroLang Compare" });
  terminal.sendText(`npx neurolang ${JSON.stringify(filePath)} --compare ${archs.join(" ")}`);
  terminal.show();
}

interface KeywordDoc {
  description: string;
  values?: string;
  example?: string;
  insertText?: string;
}

const KEYWORD_DOCS: Record<string, KeywordDoc> = {
  task: {
    description: "The type of ML task",
    values: "classification, regression",
    example: "task classification",
    insertText: "task ${1|classification,regression|}",
  },
  predict: {
    description: "The target variable to predict",
    example: "predict species",
    insertText: "predict ${1:target}",
  },
  inputs: {
    description: "Input feature names",
    example: "inputs sepal_length sepal_width petal_length petal_width",
    insertText: "inputs ${1:feature1} ${2:feature2}",
  },
  dataset: {
    description: "Dataset source (built-in name or CSV path)",
    values: "iris, housing, titanic, wine, digits, sequences, timeseries, or path.csv",
    example: "dataset iris",
    insertText: "dataset ${1|iris,housing,titanic,wine,digits,sequences,timeseries|}",
  },
  architecture: {
    description: "Neural network architecture type",
    values: "mlp, cnn, lstm, gru, rnn, resnet, transformer, autoencoder",
    example: "architecture cnn",
    insertText: "architecture ${1|mlp,cnn,lstm,gru,resnet,transformer,autoencoder|}",
  },
  pretrained: {
    description: "Use a pretrained model backbone for transfer learning",
    values: "mobilenet, resnet50, resnet101, vgg16, vgg19, efficientnet, inception, densenet",
    example: "pretrained mobilenet",
    insertText: "pretrained ${1|mobilenet,resnet50,resnet101,vgg16,vgg19,efficientnet|}",
  },
  freeze_layers: {
    description: "Number of pretrained layers to freeze during fine-tuning",
    example: "freeze_layers 15",
    insertText: "freeze_layers ${1:15}",
  },
  export_format: {
    description: "Model export format(s)",
    values: "onnx, tflite, savedmodel, torchscript, coreml",
    example: "export_format onnx tflite",
    insertText: "export_format ${1|onnx,tflite,savedmodel,torchscript,coreml|}",
  },
  targets: {
    description: "Multiple output targets for multi-task learning",
    example: "targets price category",
    insertText: "targets ${1:target1} ${2:target2}",
  },
  learn: {
    description: "Model complexity mode",
    values: "linear, nonlinear, deep, auto",
    example: "learn deep",
    insertText: "learn ${1|nonlinear,linear,deep,auto|}",
  },
  epochs: {
    description: "Number of training epochs",
    example: "epochs 40",
    insertText: "epochs ${1:40}",
  },
  optimizer: {
    description: "Optimization algorithm",
    values: "adam, sgd, rmsprop, adamw",
    example: "optimizer adam",
    insertText: "optimizer ${1|adam,sgd,rmsprop,adamw|}",
  },
  learning_rate: {
    description: "Optimizer learning rate",
    example: "learning_rate 0.001",
    insertText: "learning_rate ${1:0.001}",
  },
  batch_size: {
    description: "Training batch size",
    example: "batch_size 32",
    insertText: "batch_size ${1:32}",
  },
  dropout: {
    description: "Dropout regularization rate (0-1)",
    example: "dropout 0.2",
    insertText: "dropout ${1:0.2}",
  },
  normalize: {
    description: "Enable feature normalization",
    values: "true, false",
    example: "normalize true",
    insertText: "normalize true",
  },
  split: {
    description: "Train/test split ratio",
    example: "split 0.8",
    insertText: "split ${1:0.8}",
  },
  early_stop: {
    description: "Stop training when loss plateaus (patience in epochs)",
    example: "early_stop 5",
    insertText: "early_stop ${1:5}",
  },
  ensemble: {
    description: "Train N models and combine predictions",
    example: "ensemble 5",
    insertText: "ensemble ${1:5}",
  },
  tune: {
    description: "Enable hyperparameter auto-tuning",
    example: "tune true",
    insertText: "tune true",
  },
  filters: {
    description: "CNN convolutional filter counts per layer",
    example: "filters 32 64 128",
    insertText: "filters ${1:32} ${2:64}",
  },
  show: {
    description: "Preview the dataset",
    example: "show 10",
    insertText: "show ${1:10}",
  },
  describe: {
    description: "Show dataset statistics",
    example: "describe true",
    insertText: "describe true",
  },
};

export function deactivate() {}
