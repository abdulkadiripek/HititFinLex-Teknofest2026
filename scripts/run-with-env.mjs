import {
  createRuntimeEnvironment,
  packageInvocation,
  parseDuration,
  runBounded,
} from "./runtime-env.mjs";

const input = process.argv.slice(2);
if (input[0] === "--") input.shift();
if (input.length === 0) {
  throw new Error("usage: node scripts/run-with-env.mjs command [args...]");
}

const [requestedCommand, ...args] = input;
let invocation = { command: requestedCommand, args: [] };
if (!requestedCommand.includes("/") && !requestedCommand.includes("\\")) {
  try {
    invocation = await packageInvocation(requestedCommand);
  } catch {
    // Commands not installed in node_modules are resolved from PATH.
  }
}

await runBounded(invocation.command, [...invocation.args, ...args], {
  env: await createRuntimeEnvironment(),
  timeoutMs: parseDuration(process.env.SITES_COMMAND_TIMEOUT, 5 * 60_000),
});
