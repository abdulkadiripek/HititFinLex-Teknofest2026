import { resolve } from "node:path";
import { startProdServer } from "vinext/server/prod-server";
import { projectRoot } from "./runtime-env.mjs";

const host = process.env.HOST?.trim() || "0.0.0.0";
const port = Number.parseInt(process.env.PORT ?? "3000", 10);

if (!Number.isInteger(port) || port < 1 || port > 65_535) {
  throw new Error(`Invalid PORT: ${process.env.PORT}`);
}

await startProdServer({
  host,
  port,
  outDir: resolve(projectRoot, "dist"),
});
