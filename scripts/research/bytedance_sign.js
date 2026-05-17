#!/usr/bin/env node

const fs = require("fs");
const vm = require("vm");

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i];
    const value = argv[i + 1];
    if (!key.startsWith("--")) {
      continue;
    }
    if (value === undefined || value.startsWith("--")) {
      args[key.slice(2)] = "true";
      i -= 1;
      continue;
    }
    args[key.slice(2)] = value;
  }
  return args;
}

function usage() {
  console.error(
    [
      "Usage:",
      "  node scripts/research/bytedance_sign.js \\",
      "    --chunk /tmp/bytedance_chunks_2350.894ccf9a.js \\",
      "    --url '/api/v1/search/job/posts?keyword=&limit=10&offset=0&portal_type=2' \\",
      "    [--body '{}'] [--href 'https://jobs.bytedance.com/experienced/position']",
    ].join("\n"),
  );
}

function loadSigner(chunkPath, href, userAgent) {
  const code = fs.readFileSync(chunkPath, "utf8");
  const modules = {};
  const sandbox = {
    navigator: { userAgent },
    location: { href },
    window: {
      webpackChunkportal_: [],
      navigator: { userAgent },
      location: { href },
    },
    self: {},
    globalThis: {},
    console,
  };

  sandbox.window.webpackChunkportal_.push = (arr) => {
    Object.assign(modules, arr[1] || {});
  };

  vm.createContext(sandbox);
  vm.runInContext(code, sandbox, { timeout: 10000 });

  const signerModule = modules[57195];
  if (typeof signerModule !== "function") {
    throw new Error(`module 57195 not found in ${chunkPath}`);
  }

  const mod = { exports: {} };
  signerModule(mod, mod.exports);
  if (typeof mod.exports.sign !== "function") {
    throw new Error(`sign export not found in ${chunkPath}`);
  }
  return mod.exports.sign;
}

function main() {
  const args = parseArgs(process.argv);
  const chunk = args.chunk;
  const url = args.url;
  const bodyText = args.body || "{}";
  const href = args.href || "https://jobs.bytedance.com/experienced/position";
  const userAgent =
    args.ua ||
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36";

  if (!chunk || !url) {
    usage();
    process.exit(2);
  }

  let body;
  try {
    body = JSON.parse(bodyText);
  } catch (error) {
    console.error(`invalid --body JSON: ${error.message}`);
    process.exit(2);
  }

  const sign = loadSigner(chunk, href, userAgent);
  const signature = sign({ body, url });
  process.stdout.write(`${signature}\n`);
}

main();
