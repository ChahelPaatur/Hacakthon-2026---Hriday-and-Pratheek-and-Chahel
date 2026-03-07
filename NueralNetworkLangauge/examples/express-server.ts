/**
 * NeuroLang Express Middleware Example
 *
 * Shows how to add ML endpoints to any Express app.
 *
 * Usage:
 *   npm install express
 *   npx tsx examples/express-server.ts
 *
 * Then:
 *   curl -X POST http://localhost:4000/ml/compile \
 *     -H "Content-Type: application/json" \
 *     -d '{"source": "Predict species from iris"}'
 *
 *   curl -X POST http://localhost:4000/ml/run \
 *     -H "Content-Type: application/json" \
 *     -d '{"source": "Predict species from iris epochs 10"}'
 *
 *   curl -X POST http://localhost:4000/ml/run \
 *     -H "Content-Type: application/json" \
 *     -d '{
 *       "source": "Predict churned with age & spend",
 *       "data": [
 *         {"age": 25, "spend": 500, "churned": 0},
 *         {"age": 55, "spend": 200, "churned": 1}
 *       ]
 *     }'
 */

import { neurolangMiddleware } from "../src/middleware.js";
import { createServer } from "node:http";

const middleware = neurolangMiddleware();

const server = createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", `http://${req.headers.host}`);

  if (url.pathname.startsWith("/ml")) {
    (req as any).path = url.pathname.replace(/^\/ml/, "");
    await middleware(req, res);
    return;
  }

  res.writeHead(200, { "Content-Type": "text/html" });
  res.end(`
    <h1>NeuroLang API</h1>
    <p>POST /ml/compile — compile NeuroLang source</p>
    <p>POST /ml/run — compile & train</p>
    <p>POST /ml/generate — generate PyTorch/Keras/JAX code</p>
    <p>GET /ml/datasets — list built-in datasets</p>
    <p>GET /ml/health — health check</p>
  `);
});

const PORT = 4000;
server.listen(PORT, () => {
  console.log(`NeuroLang API running at http://localhost:${PORT}`);
  console.log(`Try: curl http://localhost:${PORT}/ml/health`);
});
