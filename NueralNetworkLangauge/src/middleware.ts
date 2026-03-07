/**
 * Framework middleware for integrating NeuroLang into web servers.
 *
 * Works with Express, Koa (via adapter), Fastify (via adapter), or any
 * framework that uses the `(req, res, next)` pattern.
 *
 * @example
 * ```ts
 * import express from 'express';
 * import { neurolangMiddleware } from 'neurolang';
 *
 * const app = express();
 * app.use(express.json());
 *
 * // Mount NeuroLang at /ml
 * app.use('/ml', neurolangMiddleware());
 *
 * // Now clients can:
 * //   POST /ml/compile   { source: "Predict species from iris" }
 * //   POST /ml/run       { source: "Predict species from iris" }
 * //   POST /ml/generate  { source: "...", target: "pytorch" }
 * //   GET  /ml/datasets
 * //   GET  /ml/health
 *
 * app.listen(3000);
 * ```
 */

import { compile as rawCompile, type CodegenTarget } from "./compiler.js";
import { execute } from "./runtime.js";
import { resolveDataset, registerInlineDataset, unregisterInlineDataset, datasetFromObjects } from "./datasets.js";
import { describeDataset, formatDataPreview } from "./inspect.js";
import type { IncomingMessage, ServerResponse } from "node:http";

export interface MiddlewareOptions {
  prefix?: string;
  enableTraining?: boolean;
  maxSourceLength?: number;
  corsOrigin?: string;
}

type NextFn = (err?: unknown) => void;

export function neurolangMiddleware(options: MiddlewareOptions = {}) {
  const {
    enableTraining = true,
    maxSourceLength = 10_000,
    corsOrigin = "*",
  } = options;

  return async function handler(
    req: IncomingMessage & { body?: unknown; path?: string; url?: string },
    res: ServerResponse,
    next?: NextFn,
  ): Promise<void> {
    const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);
    const pathname = req.path ?? url.pathname;
    const method = (req.method ?? "GET").toUpperCase();

    if (corsOrigin) {
      res.setHeader("Access-Control-Allow-Origin", corsOrigin);
      res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
      res.setHeader("Access-Control-Allow-Headers", "Content-Type");
      if (method === "OPTIONS") { res.writeHead(204); res.end(); return; }
    }

    const json = (data: unknown, status = 200) => {
      res.writeHead(status, { "Content-Type": "application/json" });
      res.end(JSON.stringify(data));
    };

    const readBody = (): Promise<string> => {
      if (req.body) return Promise.resolve(typeof req.body === "string" ? req.body : JSON.stringify(req.body));
      return new Promise((resolve, reject) => {
        const chunks: Buffer[] = [];
        req.on("data", (c: Buffer) => chunks.push(c));
        req.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
        req.on("error", reject);
      });
    };

    try {
      if (pathname === "/health" && method === "GET") {
        json({ status: "ok", engine: "neurolang", version: "1.0.0" });
        return;
      }

      if (pathname === "/datasets" && method === "GET") {
        const names = ["iris", "housing", "titanic", "wine", "digits"];
        const list = names.map((n) => {
          const ds = resolveDataset(n);
          return ds ? { name: n, rows: ds.features.length, features: ds.featureNames.length, target: ds.targetName } : null;
        }).filter(Boolean);
        json({ datasets: list });
        return;
      }

      if (pathname === "/compile" && method === "POST") {
        const body = JSON.parse(await readBody());
        const source = body.source as string;
        if (!source || source.length > maxSourceLength) {
          json({ error: "source is required and must be <= " + maxSourceLength + " chars" }, 400);
          return;
        }
        const target = (body.target ?? "tensorflow") as CodegenTarget;
        const result = rawCompile(source, { target });
        json({ ir: result.ir, code: result.code, target: result.target });
        return;
      }

      if (pathname === "/run" && method === "POST") {
        if (!enableTraining) { json({ error: "Training disabled on this endpoint" }, 403); return; }
        const body = JSON.parse(await readBody());
        const source = body.source as string;
        if (!source || source.length > maxSourceLength) {
          json({ error: "source is required and must be <= " + maxSourceLength + " chars" }, 400);
          return;
        }

        let inlineName: string | undefined;
        if (body.data && Array.isArray(body.data)) {
          inlineName = `__mw_${Date.now()}`;
          const ds = datasetFromObjects(body.data, body.targetColumn);
          registerInlineDataset(inlineName, ds);
        }

        try {
          const effectiveSource = inlineName
            ? source.replace(/from\s+\S+/i, `from ${inlineName}`)
            : source;
          const compiled = rawCompile(effectiveSource);
          const trainResult = await execute(compiled.ir);
          json({
            accuracy: compiled.ir.task === "classification" ? trainResult.metric.value : undefined,
            mse: compiled.ir.task === "regression" ? trainResult.metric.value : undefined,
            finalLoss: trainResult.finalLoss,
            epochs: trainResult.epochs,
            predictions: trainResult.predictions,
            code: compiled.code,
          });
        } finally {
          if (inlineName) unregisterInlineDataset(inlineName);
        }
        return;
      }

      if (pathname === "/generate" && method === "POST") {
        const body = JSON.parse(await readBody());
        const source = body.source as string;
        const target = (body.target ?? "pytorch") as CodegenTarget;
        if (!source) { json({ error: "source is required" }, 400); return; }
        const compiled = rawCompile(source, { target });
        json({ code: compiled.code, target: compiled.target });
        return;
      }

      if (pathname === "/inspect" && method === "POST") {
        const body = JSON.parse(await readBody());
        const name = body.dataset as string;
        if (!name) { json({ error: "dataset name is required" }, 400); return; }

        let ds = resolveDataset(name);
        if (!ds && body.data && Array.isArray(body.data)) {
          ds = datasetFromObjects(body.data, body.targetColumn);
        }
        if (!ds) { json({ error: `Dataset '${name}' not found` }, 404); return; }

        const desc = describeDataset(ds);
        const preview = formatDataPreview(ds, body.rows ?? 10);
        json({ ...desc, preview });
        return;
      }

      if (next) { next(); } else { json({ error: "Not found" }, 404); }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      json({ error: message }, 500);
    }
  };
}
