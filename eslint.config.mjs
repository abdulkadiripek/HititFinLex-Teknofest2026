import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Generated output and non-frontend packages:
    "node_modules/**",
    ".next/**",
    "out/**",
    "dist/**",
    ".sites-runtime/**",
    ".wrangler/**",
    "backend/**",
    "dataset/**",
    "db/**",
    "drizzle/**",
    "examples/**",
    "worker/**",
    "next-env.d.ts",
  ]),
]);

export default eslintConfig;
