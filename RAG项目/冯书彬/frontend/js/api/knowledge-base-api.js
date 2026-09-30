import { requestJson } from "./http.js"

async function uploadMaterial(payload) { return requestJson("/knowledge-bases/uploads", { method: "POST", body: JSON.stringify(payload) }) }
async function listMaterials() { return requestJson("/knowledge-bases/materials") }
async function reviewMaterial(materialId, decision, reason) { return requestJson(`/knowledge-bases/materials/${materialId}/review`, { method: "POST", body: JSON.stringify({ decision, reason }) }) }
async function publishMaterial(materialId) { return requestJson(`/knowledge-bases/materials/${materialId}/publish`, { method: "POST" }) }

export { listMaterials, publishMaterial, reviewMaterial, uploadMaterial }
