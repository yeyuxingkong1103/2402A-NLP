import { expect, test } from "@playwright/test"

function authorizeProfile(page) {
  return page.addInitScript(() => {
    if (!window.location.pathname.endsWith("/profile.html")) return
    localStorage.setItem("access_token", "profile-token")
    localStorage.setItem("user_id", "user-1")
  })
}

test("profile manages memories and downloads encrypted export", async ({ page }) => {
  await authorizeProfile(page)
  await page.route("**/api/v1/users/user-1/memories", async (route) => {
    expect(route.request().headers().authorization).toBe("Bearer profile-token")
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ memories: [{ id: "mem-1", summary: "已保存的脱敏案件事实" }] }),
    })
  })
  await page.route("**/api/v1/users/user-1/memory-preferences", async (route) => {
    const request = route.request()
    expect(request.method()).toBe("PATCH")
    expect(request.headers().authorization).toBe("Bearer profile-token")
    expect(await request.postDataJSON()).toMatchObject({ enabled: false })
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ enabled: false }) })
  })
  await page.route("**/api/v1/users/user-1/export/challenge", async (route) => {
    const request = route.request()
    expect(request.method()).toBe("POST")
    expect(request.headers().authorization).toBe("Bearer profile-token")
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ second_factor_token: "second-factor-token" }) })
  })
  await page.route("**/api/v1/users/user-1/export", async (route) => {
    const request = route.request()
    expect(request.method()).toBe("POST")
    expect(request.headers().authorization).toBe("Bearer profile-token")
    expect(await request.postDataJSON()).toMatchObject({ second_factor_token: "second-factor-token" })
    await route.fulfill({ status: 201, contentType: "application/json", body: JSON.stringify({ id: "export-job-1" }) })
  })
  await page.route("**/api/v1/users/user-1/export/export-job-1/download", async (route) => {
    const request = route.request()
    expect(request.method()).toBe("GET")
    expect(request.headers().authorization).toBe("Bearer profile-token")
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ encrypted_zip: { ciphertext: "encrypted-data" }, password_delivery: "claim_required" }),
    })
  })
  await page.route("**/api/v1/users/user-1/export/export-job-1/password", async (route) => {
    const request = route.request()
    expect(request.method()).toBe("POST")
    expect(request.headers().authorization).toBe("Bearer profile-token")
    expect(await request.postDataJSON()).toMatchObject({ second_factor_token: "second-factor-token" })
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ zip_password: "临时口令-123" }) })
  })

  await page.goto("/frontend/pages/profile.html")
  await expect(page.locator("#memory-list")).toContainText("已保存的脱敏案件事实")
  await page.uncheck("#memory-toggle")
  await expect(page.locator("#profile-status")).toContainText("已关闭新增记忆提取")
  await page.click('[data-testid="export-button"]')
  await expect(page.locator("#profile-status")).toContainText("导出已下载，解密口令：临时口令-123")
})

test("profile deletes account and redirects to login", async ({ page }) => {
  await authorizeProfile(page)
  await page.route("**/api/v1/users/user-1/memories", async (route) => {
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ memories: [] }) })
  })
  await page.route("**/api/v1/users/user-1", async (route) => {
    const request = route.request()
    expect(request.method()).toBe("DELETE")
    expect(request.headers().authorization).toBe("Bearer profile-token")
    expect(await request.postDataJSON()).toMatchObject({ confirmed: true })
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ deleted: true }) })
  })

  page.once("dialog", async (dialog) => {
    expect(dialog.message()).toBe("确定注销账号并删除个人数据吗？")
    await dialog.accept()
  })
  await page.goto("/frontend/pages/profile.html")
  await page.click('[data-testid="delete-account-button"]')
  await expect(page).toHaveURL(/login\.html/)
  await expect.poll(() => page.evaluate(() => localStorage.getItem("access_token"))).toBeNull()
})
