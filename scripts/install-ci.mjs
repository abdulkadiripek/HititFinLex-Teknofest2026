import { createHash } from "node:crypto";
import { open, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import {
  createRuntimeEnvironment,
  npmInvocation,
  packageInvocation,
  parseDuration,
  projectRoot,
  runBounded,
} from "./runtime-env.mjs";

const environment = await createRuntimeEnvironment({
  NPM_CONFIG_MAXSOCKETS: "1",
  NPM_CONFIG_FETCH_RETRIES: "0",
  NPM_CONFIG_FETCH_TIMEOUT: "30000",
});
const runtimeRoot = environment.SITES_RUNTIME_ROOT;
const lockPath = join(runtimeRoot, "install.lock");
let lock;

async function acquireLock() {
  try {
    lock = await open(lockPath, "wx");
  } catch (error) {
    if (error.code !== "EEXIST") throw error;
    const owner = (await readFile(lockPath, "utf8").catch(() => "unknown")).trim();
    throw new Error(`Another dependency install holds ${lockPath} (owner ${owner})`);
  }
  await lock.writeFile(`${process.pid}\n`, "utf8");
}

async function releaseLock() {
  await lock?.close();
  await rm(lockPath, { force: true });
}

await acquireLock();
try {
  const lockfilePath = join(projectRoot, "package-lock.json");
  const lockfileBytes = await readFile(lockfilePath);
  const lockfile = JSON.parse(lockfileBytes.toString("utf8"));
  const vinext = lockfile.packages?.["node_modules/vinext"];
  if (!vinext?.resolved || !vinext?.integrity) {
    throw new Error("package-lock.json lacks an integrity-pinned vinext package");
  }
  const lockfileSha256 = createHash("sha256").update(lockfileBytes).digest("hex");

  console.log(`[install] package-lock sha256 ${lockfileSha256}`);
  const npm = npmInvocation(["ci", "--cache", environment.NPM_CONFIG_CACHE]);
  await runBounded(npm.command, npm.args, {
    env: environment,
    timeoutMs: parseDuration(process.env.SITES_INSTALL_TIMEOUT, 8 * 60_000),
  });

  await packageInvocation("vinext");
  await writeFile(
    join(projectRoot, "node_modules", ".sites-install.json"),
    `${JSON.stringify(
      {
        lockfile_sha256: lockfileSha256,
        node: process.version,
        platform: `${process.platform}-${process.arch}`,
      },
      null,
      2,
    )}\n`,
    "utf8",
  );
  console.log("[install] npm ci passed and vinext is available");
} finally {
  await releaseLock();
}
