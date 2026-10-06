import { spawnSync } from "node:child_process"
import { platform } from "node:os"

const command = platform() === "win32" ? "npx.cmd" : "npx"
const mode = process.argv[2] || "emergency"
const spec = mode === "rag" ? "frontend/tests/e2e/chat-real-rag.spec.js" : "frontend/tests/e2e/chat-real.spec.js"
const envFlag = mode === "rag" ? { RUN_REAL_RAG_E2E: "1" } : { RUN_REAL_E2E: "1" }
const result = spawnSync(command, ["playwright", "test", spec, "--project=chromium"], {
  stdio: "inherit",
  shell: true,
  env: { ...process.env, ...envFlag },
})

if (result.error) {
  console.error(result.error.message)
}
process.exit(result.status ?? 1)
