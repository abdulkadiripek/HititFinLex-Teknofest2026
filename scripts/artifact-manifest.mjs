import { createHash } from "node:crypto";
import { lstat, readFile, readdir, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";

function sha256(buffer) {
  return createHash("sha256").update(buffer).digest("hex");
}

function portable(path) {
  return path.split(sep).join("/");
}

async function collectFiles(root, current = root) {
  const stat = await lstat(current);
  if (stat.isSymbolicLink()) {
    throw new Error(`Symbolic links are not accepted in release artifacts: ${current}`);
  }
  if (stat.isFile()) {
    const bytes = await readFile(current);
    return [
      {
        path: portable(relative(root, current) || "."),
        bytes: stat.size,
        sha256: sha256(bytes),
      },
    ];
  }
  if (!stat.isDirectory()) {
    throw new Error(`Unsupported artifact entry: ${current}`);
  }

  const entries = await readdir(current, { withFileTypes: true });
  entries.sort((left, right) => left.name.localeCompare(right.name, "en"));
  const files = [];
  for (const entry of entries) {
    files.push(...(await collectFiles(root, resolve(current, entry.name))));
  }
  return files;
}

function packageDigest(files) {
  const digest = createHash("sha256");
  for (const file of files) {
    digest.update(`${file.path}\0${file.bytes}\0${file.sha256}\n`, "utf8");
  }
  return digest.digest("hex");
}

async function inspectArtifact(id, version, inputPath, manifestDirectory) {
  const absolutePath = resolve(inputPath);
  const files = await collectFiles(absolutePath);
  return {
    id,
    version,
    path: portable(relative(manifestDirectory, absolutePath)),
    bytes: files.reduce((total, file) => total + file.bytes, 0),
    sha256: packageDigest(files),
    files,
  };
}

function parseArtifactSpec(spec) {
  const equals = spec.indexOf("=");
  const at = spec.indexOf("@");
  if (at < 1 || equals < at + 2) {
    throw new Error(`Expected id@version=path, got: ${spec}`);
  }
  return {
    id: spec.slice(0, at),
    version: spec.slice(at + 1, equals),
    path: spec.slice(equals + 1),
  };
}

function optionValue(args, name) {
  const index = args.indexOf(name);
  if (index < 0 || index === args.length - 1) {
    throw new Error(`Missing ${name}`);
  }
  const value = args[index + 1];
  args.splice(index, 2);
  return value;
}

async function createManifest(args) {
  const release = optionValue(args, "--release");
  const output = resolve(optionValue(args, "--output"));
  if (args.length === 0) throw new Error("At least one artifact is required");
  const manifestDirectory = dirname(output);
  const artifacts = [];
  for (const value of args) {
    const spec = parseArtifactSpec(value);
    artifacts.push(
      await inspectArtifact(spec.id, spec.version, spec.path, manifestDirectory),
    );
  }
  artifacts.sort((left, right) => left.id.localeCompare(right.id, "en"));
  const manifest = {
    schema_version: 1,
    release,
    generated_at: new Date().toISOString(),
    hash_algorithm: "sha256",
    artifacts,
  };
  await writeFile(output, `${JSON.stringify(manifest, null, 2)}\n`, {
    encoding: "utf8",
    flag: "wx",
  });
  console.log(`created ${output} (${artifacts.length} artifacts)`);
}

async function verifyManifest(input) {
  const manifestPath = resolve(input);
  const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  if (manifest.schema_version !== 1 || manifest.hash_algorithm !== "sha256") {
    throw new Error("Unsupported artifact manifest format");
  }
  const manifestDirectory = dirname(manifestPath);
  for (const expected of manifest.artifacts ?? []) {
    const inputPath = isAbsolute(expected.path)
      ? expected.path
      : resolve(manifestDirectory, expected.path);
    const actual = await inspectArtifact(
      expected.id,
      expected.version,
      inputPath,
      manifestDirectory,
    );
    if (
      actual.sha256 !== expected.sha256 ||
      actual.bytes !== expected.bytes ||
      JSON.stringify(actual.files) !== JSON.stringify(expected.files)
    ) {
      throw new Error(`Checksum verification failed: ${expected.id}`);
    }
    console.log(`verified ${expected.id}@${expected.version} ${expected.sha256}`);
  }
}

const [command, ...args] = process.argv.slice(2);
if (command === "create") {
  await createManifest(args);
} else if (command === "verify" && args.length === 1) {
  await verifyManifest(args[0]);
} else {
  throw new Error(
    "usage:\n" +
      "  npm run artifacts -- create --release <version> --output <manifest.json> id@version=path [...]\n" +
      "  npm run artifacts -- verify <manifest.json>",
  );
}
