/**
 * NeuroLang HTTP API Server
 *
 * Exposes the compiler and runtime as a REST API.
 * Start with: neurolang --serve [port]
 *
 * Endpoints:
 *   POST /compile    { source, target? }       → { ir, code, timings }
 *   POST /run        { source, target? }       → { accuracy, loss, predictions, ... }
 *   GET  /inspect/:dataset                     → { rows, cols, stats, preview }
 *   GET  /datasets                             → ["iris", "housing", ...]
 *   GET  /health                               → { status: "ok" }
 */

import * as http from "node:http";
import { compile, type CodegenTarget } from "./compiler.js";
import { execute } from "./runtime.js";
import { resolveDataset, registerInlineDataset, unregisterInlineDataset, datasetFromObjects } from "./datasets.js";
import { describeDataset, formatDataPreview } from "./inspect.js";

const CORS_HEADERS: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

function json(res: http.ServerResponse, data: unknown, status = 200): void {
  res.writeHead(status, { "Content-Type": "application/json", ...CORS_HEADERS });
  res.end(JSON.stringify(data));
}

function error(res: http.ServerResponse, message: string, status = 400): void {
  json(res, { error: message }, status);
}

async function readBody(req: http.IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);
  return Buffer.concat(chunks).toString("utf-8");
}

export async function startServer(port: number = 3000): Promise<void> {
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url ?? "/", `http://localhost:${port}`);
    const method = req.method ?? "GET";

    if (method === "OPTIONS") {
      res.writeHead(204, CORS_HEADERS);
      res.end();
      return;
    }

    try {
      // GET /health
      if (url.pathname === "/health" && method === "GET") {
        json(res, { status: "ok", version: "3.0" });
        return;
      }

      // GET /datasets
      if (url.pathname === "/datasets" && method === "GET") {
        json(res, {
          datasets: ["iris", "housing", "titanic", "wine", "digits"],
        });
        return;
      }

      // GET /inspect/:dataset
      if (url.pathname.startsWith("/inspect/") && method === "GET") {
        const dsName = url.pathname.slice("/inspect/".length);
        const ds = resolveDataset(dsName);
        if (!ds) {
          error(res, `Unknown dataset: ${dsName}`, 404);
          return;
        }
        const desc = describeDataset(ds);
        const preview = formatDataPreview(ds, 10);
        json(res, { ...desc, preview });
        return;
      }

      // POST /compile
      if (url.pathname === "/compile" && method === "POST") {
        const body = JSON.parse(await readBody(req));
        const source = body.source as string;
        const target = (body.target ?? "tensorflow") as CodegenTarget;

        if (!source) {
          error(res, "Missing 'source' field");
          return;
        }

        const result = compile(source, { target });
        json(res, {
          ir: result.ir,
          code: result.code,
          target: result.target,
          timings: result.timings,
          diagnostics: result.diagnostics,
        });
        return;
      }

      // POST /run
      if (url.pathname === "/run" && method === "POST") {
        const body = JSON.parse(await readBody(req));
        let source = body.source as string;
        const target = (body.target ?? "tensorflow") as CodegenTarget;

        if (!source) {
          error(res, "Missing 'source' field");
          return;
        }

        let inlineName: string | undefined;
        if (body.data && Array.isArray(body.data)) {
          inlineName = `__srv_${Date.now()}`;
          const ds = datasetFromObjects(body.data as Record<string, number>[], body.targetColumn);
          registerInlineDataset(inlineName, ds);

          if (/\bfrom\s+\S+/i.test(source)) {
            source = source.replace(/\bfrom\s+\S+/i, `from ${inlineName}`);
          } else if (/\bdataset\s+\S+/i.test(source)) {
            source = source.replace(/\bdataset\s+\S+/i, `dataset ${inlineName}`);
          } else {
            source += `\ndataset ${inlineName}`;
          }
        }

        try {
          const compiled = compile(source, { target });
          const trainResult = await execute(compiled.ir);

          json(res, {
            task: compiled.ir.task,
            accuracy: compiled.ir.task === "classification" ? trainResult.metric.value : undefined,
            mse: compiled.ir.task === "regression" ? trainResult.metric.value : undefined,
            metric: trainResult.metric,
            finalLoss: trainResult.finalLoss,
            lossHistory: trainResult.lossHistory,
            epochs: trainResult.epochs,
            earlyStopEpoch: trainResult.earlyStopEpoch,
            predictions: trainResult.predictions,
            crossValidation: trainResult.crossValidation,
            code: compiled.code,
            timings: compiled.timings,
          });
        } finally {
          if (inlineName) unregisterInlineDataset(inlineName);
        }
        return;
      }

      // POST /generate
      if (url.pathname === "/generate" && method === "POST") {
        const body = JSON.parse(await readBody(req));
        const source = body.source as string;
        const targets = (body.targets ?? ["tensorflow"]) as CodegenTarget[];

        if (!source) {
          error(res, "Missing 'source' field");
          return;
        }

        const results: Record<string, string> = {};
        for (const t of targets) {
          results[t] = compile(source, { target: t }).code;
        }

        json(res, { code: results });
        return;
      }

      error(res, "Not found", 404);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      error(res, msg, 500);
    }
  });

  server.listen(port, () => {
    console.log(`\x1b[33m\x1b[1m`);
    console.log(`  ╔═══════════════════════════════════════════╗`);
    console.log(`  ║       NeuroLang API Server v3.0           ║`);
    console.log(`  ║   http://localhost:${port}                   ║`);
    console.log(`  ╚═══════════════════════════════════════════╝\x1b[0m`);
    console.log();
    console.log(`  Endpoints:`);
    console.log(`    POST /compile       Compile NeuroLang source`);
    console.log(`    POST /run           Compile + train model`);
    console.log(`    POST /generate      Generate code for targets`);
    console.log(`    GET  /inspect/:ds   Inspect a dataset`);
    console.log(`    GET  /datasets      List available datasets`);
    console.log(`    GET  /health        Health check`);
    console.log();
    console.log(`  \x1b[2mExample:`);
    console.log(`    curl -X POST http://localhost:${port}/compile \\`);
    console.log(`      -H "Content-Type: application/json" \\`);
    console.log(`      -d '{"source": "Predict species with a & b from iris.csv", "target": "pytorch"}'\x1b[0m`);
    console.log();
  });
}
