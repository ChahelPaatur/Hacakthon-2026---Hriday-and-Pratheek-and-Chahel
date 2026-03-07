#!/usr/bin/env node
/**
 * Build NeuroLang distribution artifacts:
 * 1. Standalone CJS bundle (runs with any Node.js installation)
 * 2. SEA blob for native binary creation (used by CI/CD)
 * 3. npm-publishable package
 */
import { execSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
const distDir = path.join(root, "dist-bin");

const Y = "\x1b[33m";
const G = "\x1b[32m";
const B = "\x1b[1m";
const D = "\x1b[2m";
const X = "\x1b[0m";

function run(cmd) {
  console.log(`  ${D}$ ${cmd}${X}`);
  return execSync(cmd, { cwd: root, stdio: "pipe" }).toString().trim();
}

const platform = process.platform === "darwin" ? "macOS" : process.platform === "win32" ? "Windows" : "Linux";
const arch = process.arch;

console.log(`\n${B}${Y}╔══════════════════════════════════════════════╗${X}`);
console.log(`${B}${Y}║   NeuroLang — Build Distribution             ║${X}`);
console.log(`${B}${Y}╚══════════════════════════════════════════════╝${X}\n`);
console.log(`  Platform:  ${G}${platform} ${arch}${X}\n`);

// Clean
if (fs.existsSync(distDir)) fs.rmSync(distDir, { recursive: true });
fs.mkdirSync(distDir, { recursive: true });

// Step 1: Bundle with esbuild
console.log(`${B}[1/3] Bundling with esbuild...${X}`);
const result = run(`npx esbuild src/cli.ts --bundle --platform=node --target=node18 --outfile=dist-bin/neurolang.cjs --format=cjs --external:@tensorflow/tfjs`);
console.log(`  ${G}✓${X} dist-bin/neurolang.cjs`);

const bundlePath = path.join(distDir, "neurolang.cjs");
fs.chmodSync(bundlePath, 0o755);

// Step 2: Create launcher scripts
console.log(`\n${B}[2/3] Creating launcher scripts...${X}`);

// Unix launcher
const unixLauncher = `#!/usr/bin/env bash
# NeuroLang - Declarative Neural Network Language
# https://github.com/ChahelPaatur/Hacakthon-2026---Hriday-and-Pratheek-and-Chahel
DIR="$(cd "$(dirname "$0")" && pwd)"
exec node "$DIR/neurolang.cjs" "$@"
`;
fs.writeFileSync(path.join(distDir, "neurolang"), unixLauncher);
fs.chmodSync(path.join(distDir, "neurolang"), 0o755);

// Windows launcher
const winLauncher = `@echo off
REM NeuroLang - Declarative Neural Network Language
node "%~dp0neurolang.cjs" %*
`;
fs.writeFileSync(path.join(distDir, "neurolang.cmd"), winLauncher);

// PowerShell launcher
const psLauncher = `#!/usr/bin/env pwsh
# NeuroLang - Declarative Neural Network Language
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
node "$ScriptDir/neurolang.cjs" @args
`;
fs.writeFileSync(path.join(distDir, "neurolang.ps1"), psLauncher);

console.log(`  ${G}✓${X} neurolang       (macOS/Linux)`);
console.log(`  ${G}✓${X} neurolang.cmd   (Windows CMD)`);
console.log(`  ${G}✓${X} neurolang.ps1   (PowerShell)`);

// Step 3: Create SEA config for CI native builds
console.log(`\n${B}[3/3] Preparing SEA config for CI...${X}`);
const seaConfig = {
  main: "dist-bin/neurolang.cjs",
  output: "dist-bin/sea-prep.blob",
  disableExperimentalSEAWarning: true,
  useCodeCache: true,
};
fs.writeFileSync(path.join(distDir, "sea-config.json"), JSON.stringify(seaConfig, null, 2));
console.log(`  ${G}✓${X} sea-config.json (for GitHub Actions native builds)`);

// Summary
const bundleStat = fs.statSync(bundlePath);
const sizeMB = (bundleStat.size / (1024 * 1024)).toFixed(1);

console.log(`\n${B}${G}Build complete!${X}`);
console.log(`\n  Bundle size: ${Y}${sizeMB} MB${X}`);
console.log(`  Output dir:  ${Y}dist-bin/${X}`);

console.log(`\n${B}Usage:${X}`);
console.log(`  ${Y}macOS/Linux:${X}  ./dist-bin/neurolang examples/iris.nl --run`);
console.log(`  ${Y}Windows:${X}      dist-bin\\neurolang.cmd examples/iris.nl --run`);
console.log(`  ${Y}Direct:${X}       node dist-bin/neurolang.cjs examples/iris.nl --run`);

console.log(`\n${B}Publish to npm:${X}`);
console.log(`  ${Y}npm publish${X}   → users install with ${Y}npm install -g neurolang${X}`);

console.log(`\n${B}Native binaries:${X}`);
console.log(`  Push a git tag (${Y}git tag v1.0.0 && git push --tags${X})`);
console.log(`  → GitHub Actions builds .exe / macOS / Linux binaries automatically`);
console.log();
