import { expect, test } from "@playwright/test"

test("user logs in with mocked phone code flow", async ({ page }) => {
  await page.route("**/api/v1/auth/code", async (route) => {
    const request = route.request()
    expect(request.method()).toBe("POST")
    expect(await request.postDataJSON()).toMatchObject({ phone: "13900000001", client_id: "browser" })
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ sent: false, expires_in_seconds: 300 }) })
  })
  await page.route("**/api/v1/auth/login", async (route) => {
    const request = route.request()
    expect(request.headers().authorization).toBeUndefined()
    expect(await request.postDataJSON()).toMatchObject({ phone: "13900000001", code: "123456", client_id: "browser" })
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ access_token: "access-token", refresh_token: "refresh-token", user_id: "user-1", token_type: "bearer" }),
    })
  })

  await page.goto("/frontend/pages/login.html")
  await page.fill('[data-testid="phone-input"]', "13900000001")
  await page.click('[data-testid="request-code-button"]')
  await page.fill('[data-testid="code-input"]', "123456")
  await page.check("#privacy-checkbox")
  await page.check("#agreement-checkbox")
  await page.check("#adult-checkbox")
  await page.click('[data-testid="login-button"]')

  await expect(page).toHaveURL(/chat\.html/)
  await expect.poll(() => page.evaluate(() => localStorage.getItem("user_id"))).toBe("user-1")
})
