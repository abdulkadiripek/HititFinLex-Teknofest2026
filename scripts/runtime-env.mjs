import { spawn } from "node:child_process";
import { mkdir, readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

export async function createRuntimeEnvironment(overrides = {}) {
  const runtimeRoot = resolve(
    process.env.SITES_RUNTIME_ROOT ?? join(projectRoot, ".sites-runtime"),
  );
  const npmCache = join(runtimeRoot, "npm-cache");
  const xdgConfig = join(runtimeRoot, "xdg-config");
  const wranglerRoot = join(runtimeRoot, "wrangler");

  await Promise.all([
    mkdir(npmCache, { recursive: true }),
    mkdir(xdgConfig, { recursive: true }),
    mkdir(join(wranglerRoot, "logs"), { recursive: true }),
  ]);

  const environment = {
    ...process.env,
    SITES_ENV_READY: "1",
    SITES_PROJECT_ROOT: projectRoot,
    SITES_RUNTIME_ROOT: runtimeRoot,
    XDG_CONFIG_HOME: xdgConfig,
    NPM_CONFIG_CACHE: npmCache,
    NPM_CONFIG_AUDIT: "false",
    NPM_CONFIG_FUND: "false",
    NPM_CONFIG_UPDATE_NOTIFIER: "false",
    WRANGLER_WRITE_LOGS: "false",
    WRANGLER_LOG_PATH: join(wranglerRoot, "logs"),
    MINIFLARE_REGISTRY_PATH: join(wranglerRoot, "registry"),
    ...overrides,
  };

  for (const name of [
    "npm_config_proxy",
    "npm_config_http_proxy",
    "npm_config_https_proxy",
    "NPM_CONFIG_PROXY",
    "NPM_CONFIG_HTTP_PROXY",
    "NPM_CONFIG_HTTPS_PROXY",
  ]) {
    delete environment[name];
  }

  return environment;
}

export function parseDuration(value, fallbackMilliseconds) {
  if (!value) return fallbackMilliseconds;
  const match = /^(\d+)(ms|s|m)?$/i.exec(value.trim());
  if (!match) {
    throw new Error(`Invalid duration: ${value}`);
  }
  const amount = Number(match[1]);
  const multiplier = { ms: 1, s: 1_000, m: 60_000 }[
    (match[2] ?? "ms").toLowerCase()
  ];
  return amount * multiplier;
}

export async function packageInvocation(name) {
  const packageNames = {
    tsc: "typescript",
  };
  const packageName = packageNames[name] ?? name;
  const packageRoot = join(projectRoot, "node_modules", packageName);
  const metadata = JSON.parse(
    await readFile(join(packageRoot, "package.json"), "utf8"),
  );
  const relativeBinary =
    typeof metadata.bin === "string"
      ? metadata.bin
      : metadata.bin?.[name] ?? Object.values(metadata.bin ?? {})[0];
  if (!relativeBinary) {
    throw new Error(`Package ${packageName} does not expose a binary for ${name}`);
  }
  return {
    command: process.execPath,
    args: [resolve(packageRoot, relativeBinary)],
  };
}

export function npmInvocation(args) {
  const npmCli = process.env.npm_execpath;
  if (!npmCli) {
    throw new Error("npm_execpath is missing; invoke this command through npm run");
  }
  return { command: process.execPath, args: [npmCli, ...args] };
}

export async function runBounded(
  command,
  args,
  { cwd = projectRoot, env = process.env, timeoutMs = 180_000 } = {},
) {
  const child = spawn(command, args, {
    cwd,
    env,
    shell: false,
    stdio: "inherit",
    windowsHide: true,
  });

  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    child.kill("SIGTERM");
    setTimeout(() => child.kill("SIGKILL"), 10_000).unref();
  }, timeoutMs);

  const result = await new Promise((resolveResult, reject) => {
    child.once("error", reject);
    child.once("exit", (code, signal) => resolveResult({ code, signal }));
  });
  clearTimeout(timer);

  if (timedOut) {
    throw new Error(`Command timed out after ${timeoutMs} ms: ${command}`);
  }
  if (result.code !== 0) {
    throw new Error(
      `Command failed with ${result.code ?? result.signal}: ${command}`,
    );
  }
}
