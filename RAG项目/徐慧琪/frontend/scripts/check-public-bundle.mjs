#!/usr/bin/env node
// 产物闸门：公众入口的构建产物里不许出现律师侧专用端点（设计 §三 定调 1/3）。
// 为什么要扫产物而不是只扫源码：结构性隔离的最终事实在打包结果里——源码 import
// 分析看不见 tree-shaking 的擦除效果，也看不见某次改动把律师代码拖进公众 chunk。
// 反向断言（律师包不含公众端点）故意不做：律师页复用 public.ts 的 nav/article
// 是允许的（Task 6 审查裁定），钉反方向会误红。
//
// marker 用**端点后缀字面量**而不是 `/api/v1/qa` 全串：产物由 `API_PREFIX + path`
// 运行时拼接，完整串根本不存在（2026-10-03 实测）；且匹配必须带边界，否则
// `/public/qa` 会把 `/qa` 带响。两向夹具见 tests/check-public-bundle.spec.ts。
import { readFileSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

export const MARKERS = ['/qa', '/search', '/cases/search', '/admin/audit/export']
// 正对照：律师包必须含 /qa。没有它，「公众包干净」可能只是 API 客户端整体没打进
// 产物（本项目「不可能失败的测试」的典型形态：扫描面为空而恒绿）
export const POSITIVE_CONTROL = '/qa'

// 边界判定：marker 相邻字符不属于 [A-Za-z0-9_/] 才算一次端点字面量命中。
// `/public/qa` 的 `/qa` 前是 c、`/cases/search` 的 `/search` 前是 s，都被挡掉；
// 产物里真实的 `` `/qa` `` 两侧是反引号，命中。
const WORD_CHAR = /[A-Za-z0-9_/]/

export function findForbiddenMarkers(text, markers = MARKERS) {
  return markers.filter((marker) => hasBoundedOccurrence(text, marker))
}

function hasBoundedOccurrence(text, marker) {
  let from = 0
  for (;;) {
    const i = text.indexOf(marker, from)
    if (i === -1) return false
    const before = i === 0 ? '' : text[i - 1]
    const after = i + marker.length >= text.length ? '' : text[i + marker.length]
    if (!WORD_CHAR.test(before) && !WORD_CHAR.test(after)) return true
    from = i + 1
  }
}

// html 里的模块入口：<script src> 与 <link rel="modulepreload" href> 都算公众包的
// 组成部分（Vite 给静态 import 的 chunk 发的 preload 提示）；seen 去重，重复无碍
export function extractHtmlRefs(html) {
  const refs = []
  for (const tag of html.match(/<script\b[^>]*>/gi) ?? []) {
    const m = tag.match(/\bsrc\s*=\s*["']([^"']+)["']/i)
    if (m) refs.push(m[1])
  }
  for (const tag of html.match(/<link\b[^>]*>/gi) ?? []) {
    if (!/\brel\s*=\s*["']modulepreload["']/i.test(tag)) continue
    const m = tag.match(/\bhref\s*=\s*["']([^"']+)["']/i)
    if (m) refs.push(m[1])
  }
  return refs
}

// js 里的相对模块引用：静态 `import"./x.js"` / `from"./x.js"` 与动态
// `import("./x.js")` 都被这一条抓（打包产物的模块说明符必以 ./ 或 ../ 开头）。
// 形似的普通字符串字面量会被一并跟随——方向安全：多读一个文件只会让公众包更严
export function extractJsRefs(js) {
  return [...js.matchAll(/["'`](\.\.?\/[^"'`]+\.js)["'`]/g)].map((m) => m[1])
}

function resolveRef(fromFile, ref, rootDir) {
  const clean = ref.replace(/[?#].*$/, '') // 去掉可能的查询串/锚点
  return clean.startsWith('/')
    ? resolve(rootDir, clean.replace(/^\/+/, '')) // 产物 html 用站点根路径 /assets/…
    : resolve(dirname(fromFile), clean)
}

// 入口 → 传递闭包：入口 js 只是壳，公众包的实际代码在共享 chunk 里（两入口都
// import NavTree-*.js，2026-10-03 实测）。不跟随 import 会让「律师代码混进共享
// chunk」假绿（裁决②），所以这里 BFS 递归读 file + text。
/**
 * @param {string} entryFile
 * @param {{ rootDir?: string, readFile?: (path: string) => string }} [options]
 * @returns {{ file: string, text: string }[]}
 */
export function collectModuleTexts(entryFile, options = {}) {
  const rootDir = options.rootDir ?? dirname(entryFile)
  const read = options.readFile ?? ((p) => readFileSync(p, 'utf8'))
  const seen = new Set()
  const out = []
  const queue = [resolve(entryFile)]
  for (let i = 0; i < queue.length; i++) {
    const file = queue[i]
    if (seen.has(file)) continue
    seen.add(file)
    let text
    try {
      text = read(file)
    } catch (e) {
      if (out.length === 0) throw new Error(`读不到入口文件 ${file}（${e.message}）`)
      continue // 被引用的文件不存在（字符串字面量误判/可选资源）：跳过，不当违规
    }
    out.push({ file, text })
    const refs = /\.html?$/i.test(file) ? extractHtmlRefs(text) : extractJsRefs(text)
    for (const ref of refs) {
      const p = resolveRef(file, ref, rootDir)
      if (!seen.has(p)) queue.push(p)
    }
  }
  return out
}

export function scanBundle(htmlFile) {
  const files = collectModuleTexts(htmlFile, { rootDir: dirname(htmlFile) })
  const hits = []
  for (const f of files) {
    for (const marker of findForbiddenMarkers(f.text)) hits.push({ file: f.file, marker })
  }
  return { files, hits }
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]
if (isMain) {
  const root = fileURLToPath(new URL('..', import.meta.url))
  const dist = join(root, 'dist')
  let publicScan
  let lawyerScan
  try {
    publicScan = scanBundle(join(dist, 'index.html'))
    lawyerScan = scanBundle(join(dist, 'lawyer.html'))
  } catch (e) {
    console.log(`产物闸门失败：${e.message}`)
    console.log('（先跑 npm run build 生成 dist/ 再跑本闸门）')
    process.exit(1)
  }
  const problems = []
  // 律师包正对照（裁决③）：缺 /qa 说明 API 客户端根本没打进去，公众包的「干净」
  // 不具说服力。只查 /qa 一个——律师包含 /search 等是预期，不钉更多
  const lawyerText = lawyerScan.files.map((f) => f.text).join('\n')
  if (findForbiddenMarkers(lawyerText, [POSITIVE_CONTROL]).length === 0) {
    problems.push(`律师包缺少正对照 ${POSITIVE_CONTROL}：API 客户端可能整体没打进产物，`
      + '此时「公众包干净」不代表隔离成立（假绿防呆）')
  }
  for (const hit of publicScan.hits) {
    problems.push(`公众包出现律师侧专用端点串：${hit.marker}`
      + `（${relative(root, hit.file).replace(/\\/g, '/')}）`)
  }
  console.log(`产物闸门：公众包 ${publicScan.files.length} 个模块、律师包 ${lawyerScan.files.length} 个模块`)
  if (problems.length) {
    for (const p of problems) console.log(`- ${p}`)
    console.log(`产物闸门失败：${problems.length} 处`)
    process.exit(1)
  }
  console.log(`检查通过：公众包不含律师侧端点串（${MARKERS.join('、')}）；`
    + `律师包正对照 ${POSITIVE_CONTROL} 在场`)
}
