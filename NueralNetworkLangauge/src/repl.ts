/**
 * NeuroLang REPL — interactive playground for building neural networks.
 */
import * as readline from "node:readline";
import { compile, CompilationError, type CodegenTarget } from "./compiler.js";
import { emitSummary } from "./codegen.js";
import { execute } from "./runtime.js";

const Y = "\x1b[33m";
const G = "\x1b[32m";
const R = "\x1b[31m";
const C = "\x1b[36m";
const D = "\x1b[2m";
const B = "\x1b[1m";
const X = "\x1b[0m";

export async function startRepl(): Promise<void> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: `${Y}neurolang>${X} `,
  });

  const buffer: string[] = [];
  let target: CodegenTarget = "tensorflow";

  console.log(`${Y}${B}`);
  console.log(`  ╔═══════════════════════════════════════════╗`);
  console.log(`  ║       NeuroLang REPL v2.0                 ║`);
  console.log(`  ║   Interactive Neural Network Builder      ║`);
  console.log(`  ╚═══════════════════════════════════════════╝${X}`);
  console.log();
  console.log(`  Type NeuroLang statements line by line.`);
  console.log(`  ${D}Commands: :help :run :show :code :reset :target :quit${X}`);
  console.log();

  rl.prompt();

  rl.on("line", async (rawLine: string) => {
    const line = rawLine.trim();

    if (line.startsWith(":")) {
      await handleCommand(line, buffer, target, (t) => { target = t; });
      rl.prompt();
      return;
    }

    if (line.length === 0) {
      rl.prompt();
      return;
    }

    buffer.push(rawLine);
    const source = buffer.join("\n");

    try {
      const result = compile(source, { target });
      const ir = result.ir;
      const arch = ir.architecture.layers
        .filter((l) => l.kind === "dense")
        .map((l) => l.kind === "dense" ? `Dense(${l.units})` : "")
        .join(" → ");
      console.log(`  ${G}✓${X} ${ir.task} | Input(${ir.architecture.inputSize}) → ${arch} | ${ir.meta.parameterCount.toLocaleString()} params`);
    } catch (err) {
      if (err instanceof CompilationError) {
        const firstError = err.diagnostics.find((d) => d.severity === "error");
        if (firstError) {
          console.log(`  ${R}✗ ${firstError.message}${X}`);
          if (firstError.help) console.log(`    ${D}= help: ${firstError.help}${X}`);
        }
      }
    }

    rl.prompt();
  });

  rl.on("close", () => {
    console.log(`\n${D}Goodbye.${X}`);
    process.exit(0);
  });
}

async function handleCommand(
  cmd: string,
  buffer: string[],
  target: CodegenTarget,
  setTarget: (t: CodegenTarget) => void,
): Promise<void> {
  const parts = cmd.split(/\s+/);
  const command = parts[0]!.toLowerCase();

  switch (command) {
    case ":help":
      console.log(`\n  ${B}REPL Commands${X}`);
      console.log(`  ${Y}:run${X}          Compile and train the current program`);
      console.log(`  ${Y}:show${X}         Show the compilation summary`);
      console.log(`  ${Y}:code${X}         Show the generated code`);
      console.log(`  ${Y}:reset${X}        Clear the program buffer`);
      console.log(`  ${Y}:target${X} <t>   Switch target (tensorflow, pytorch, keras, jax)`);
      console.log(`  ${Y}:source${X}       Show the current source`);
      console.log(`  ${Y}:quit${X}         Exit the REPL`);
      console.log();
      break;

    case ":reset":
      buffer.length = 0;
      console.log(`  ${D}Buffer cleared.${X}`);
      break;

    case ":source":
      if (buffer.length === 0) {
        console.log(`  ${D}(empty program)${X}`);
      } else {
        console.log();
        for (let i = 0; i < buffer.length; i++) {
          console.log(`  ${D}${String(i + 1).padStart(3)}${X} ${buffer[i]}`);
        }
        console.log();
      }
      break;

    case ":show": {
      if (buffer.length === 0) {
        console.log(`  ${R}No program to compile.${X}`);
        break;
      }
      try {
        const result = compile(buffer.join("\n"), { target });
        console.log();
        console.log(emitSummary(result.ir));
      } catch (err) {
        printError(err);
      }
      break;
    }

    case ":code": {
      if (buffer.length === 0) {
        console.log(`  ${R}No program to compile.${X}`);
        break;
      }
      try {
        const result = compile(buffer.join("\n"), { target });
        console.log();
        console.log(result.code);
      } catch (err) {
        printError(err);
      }
      break;
    }

    case ":target": {
      const validTargets = ["tensorflow", "pytorch", "keras", "jax", "summary"];
      const t = parts[1]?.toLowerCase();
      if (t && validTargets.includes(t)) {
        setTarget(t as CodegenTarget);
        console.log(`  ${G}Target set to: ${t}${X}`);
      } else {
        console.log(`  ${R}Valid targets: ${validTargets.join(", ")}${X}`);
      }
      break;
    }

    case ":run": {
      if (buffer.length === 0) {
        console.log(`  ${R}No program to run.${X}`);
        break;
      }
      try {
        const result = compile(buffer.join("\n"), { target });
        console.log(`\n  ${Y}${B}Training...${X}`);
        const trainResult = await execute(result.ir, {
          onEpochEnd: (epoch, loss) => {
            const total = result.ir.training.epochs;
            if ((epoch + 1) % Math.max(1, Math.floor(total / 10)) === 0 || epoch === 0) {
              process.stdout.write(`  ${D}Epoch ${String(epoch + 1).padStart(4)}/${total}: loss=${loss.toFixed(4)}${X}\n`);
            }
          },
        });
        console.log(`\n  ${G}${B}Done${X} — ${trainResult.metric.name}: ${trainResult.metric.value.toFixed(4)}, epochs: ${trainResult.epochs}`);
        console.log();
      } catch (err) {
        printError(err);
      }
      break;
    }

    case ":quit":
    case ":exit":
    case ":q":
      console.log(`${D}Goodbye.${X}`);
      process.exit(0);

    default:
      console.log(`  ${R}Unknown command: ${command}${X} — type :help for commands`);
  }
}

function printError(err: unknown): void {
  if (err instanceof CompilationError) {
    for (const d of err.diagnostics) {
      if (d.severity === "error") {
        console.log(`  ${R}[${d.code}] ${d.message}${X}`);
        if (d.help) console.log(`    ${D}= help: ${d.help}${X}`);
      }
    }
  } else {
    const message = err instanceof Error ? err.message : String(err);
    console.log(`  ${R}Error: ${message}${X}`);
  }
}
