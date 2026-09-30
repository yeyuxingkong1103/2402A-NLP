import { requestJson } from "./http.js"

function userPath(userId, suffix) {
  return `/users/${encodeURIComponent(userId)}${suffix}`
}

async function listMemories(userId) { return requestJson(userPath(userId, "/memories")) }
async function updateMemoryPreferences(userId, enabled) { return requestJson(userPath(userId, "/memory-preferences"), { method: "PATCH", body: JSON.stringify({ enabled }) }) }
async function createExport(userId) {
  const challenge = await requestJson(userPath(userId, "/export/challenge"), { method: "POST" })
  return requestJson(userPath(userId, "/export"), { method: "POST", body: JSON.stringify({ second_factor_token: challenge.second_factor_token }) })
}
async function downloadExport(userId, jobId) {
  return requestJson(userPath(userId, `/export/${encodeURIComponent(jobId)}/download`))
}
async function claimExportPassword(userId, jobId) {
  const challenge = await requestJson(userPath(userId, "/export/challenge"), { method: "POST" })
  return requestJson(userPath(userId, `/export/${encodeURIComponent(jobId)}/password`), { method: "POST", body: JSON.stringify({ second_factor_token: challenge.second_factor_token }) })
}
async function deleteAccount(userId) { return requestJson(userPath(userId, ""), { method: "DELETE", body: JSON.stringify({ confirmed: true }) }) }

export { listMemories, updateMemoryPreferences, createExport, downloadExport, claimExportPassword, deleteAccount }
