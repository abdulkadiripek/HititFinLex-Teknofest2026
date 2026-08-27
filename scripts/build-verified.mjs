import {
  createRuntimeEnvironment,
  packageInvocation,
  parseDuration,
  runBounded,
} from "./runtime-env.mjs";

let vinext;
try {
  vinext = await packageInvocation("vinext");
} catch {
  throw new Error("vinext is unavailable; run npm run install:ci first");
}

console.log("[build] running bounded vinext build");
await runBounded(vinext.command, [...vinext.args, "build"], {
  env: await createRuntimeEnvironment(),
  timeoutMs: parseDuration(process.env.SITES_BUILD_TIMEOUT, 3 * 60_000),
});
