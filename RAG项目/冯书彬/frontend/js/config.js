function defaultApiBaseUrl() {
  if (window.APP_API_BASE_URL) return window.APP_API_BASE_URL
  if (window.location.pathname.startsWith("/frontend/")) return "/api/v1"
  if (window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost") {
    return `${window.location.protocol}//${window.location.hostname}:8010/api/v1`
  }
  return "/api/v1"
}

const AppConfig = {
  apiBaseUrl: defaultApiBaseUrl(),
  devUserId: "guest-user",
  requestTimeout: 30000,
  streamTimeout: 120000
}

export default AppConfig
