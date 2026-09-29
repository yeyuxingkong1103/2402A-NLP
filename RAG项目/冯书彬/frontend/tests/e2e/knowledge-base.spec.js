import { expect, test } from "@playwright/test"

const reviewerToken = "eyJhbGciOiJub25lIn0.eyJyb2xlcyI6WyJjb250ZW50X3Jldmlld2VyIl19.signature"
const adminToken = "eyJhbGciOiJub25lIn0.eyJyb2xlcyI6WyJzdXBlcl9hZG1pbiJdfQ.signature"

test("content reviewer reviews material but cannot publish it", async ({ page }) => {
  let materialStatus = "pending_review"
  await page.addInitScript((token) => {
    localStorage.setItem("access_token", token)
    localStorage.setItem("roles", JSON.stringify(["content_reviewer"]))
  }, reviewerToken)
  await page.route("**/api/v1/knowledge-bases/materials", async (route) => {
    expect(route.request().headers().authorization).toBe(`Bearer ${reviewerToken}`)
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify([{ id: "mat-1", title: "婚姻家庭法律材料", status: materialStatus }]) })
  })
  await page.route("**/api/v1/knowledge-bases/materials/mat-1/review", async (route) => {
    const request = route.request()
    expect(request.method()).toBe("POST")
    expect(request.headers().authorization).toBe(`Bearer ${reviewerToken}`)
    expect(await request.postDataJSON()).toMatchObject({ decision: "approved", reason: "本地上传材料已核验" })
    materialStatus = "reviewed"
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ id: "mat-1", status: materialStatus }) })
  })
  await page.goto("/frontend/pages/knowledge-bases.html")
  await page.click('[data-testid="review-material-button"]')
  await expect(page.locator('[data-testid="material-status"]')).toContainText("已审核")
  await expect(page.locator('[data-testid="publish-material-button"]')).toHaveAttribute("hidden", "")
})

test("super admin publishes reviewed material", async ({ page }) => {
  let materialStatus = "reviewed"
  await page.addInitScript((token) => {
    localStorage.setItem("access_token", token)
    localStorage.setItem("roles", JSON.stringify(["super_admin"]))
  }, adminToken)
  await page.route("**/api/v1/knowledge-bases/materials", async (route) => {
    expect(route.request().headers().authorization).toBe(`Bearer ${adminToken}`)
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify([{ id: "mat-2", title: "继承法律材料", status: materialStatus }]) })
  })
  await page.route("**/api/v1/knowledge-bases/materials/mat-2/publish", async (route) => {
    const request = route.request()
    expect(request.method()).toBe("POST")
    expect(request.headers().authorization).toBe(`Bearer ${adminToken}`)
    materialStatus = "published"
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ id: "mat-2", status: materialStatus }) })
  })
  await page.goto("/frontend/pages/knowledge-bases.html")
  await page.click('[data-testid="publish-material-button"]')
  await expect(page.locator('[data-testid="material-status"]')).toContainText("已发布")
})
