#!/usr/bin/env node
// 文件规模闸门：非空行 ≤ 300。与后端 check_style 的精神一致——超限就拆文件，
// 不靠压缩空行规避（所以判据数的是非空行）。导出函数供测试 import；
// 直接执行时才扫仓库（import 时应无副作用）。
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'

export const MAX_NON_EMPTY_LINES = 300
export const SCAN_DIRS = ['src', 'tests']

export function nonEmptyLines(text) {
  return text.split(/\r?\n/).filter((l) => l.trim() !== '').length
}

// 生成物豁免：openapi-typescript 的产物 `src/core/api-types.gen.ts`（体量由后端契约
// 决定、当前 945 行）拆不动也不该手改——「新鲜度」由 `npm run gen:api` 每次再生保证，
// 「禁止手改」由审查与再生对照守。豁免**只认这一个精确路径**，不认 `*.gen.ts` 后缀：
// 后缀制下任何人把大文件改名成 `x.gen.ts` 就整体躲开 300 行闸门；同名文件放在别的
// 目录同样是手写代码，也不豁免。isGenerated 收到的路径相对被扫描根（Windows 的
// 反斜杠先归一成 `/`）。
export const GENERATED_WHITELIST = ['src/core/api-types.gen.ts']

export function isGenerated(file) {
  return GENERATED_WHITELIST.includes(file.replace(/\\/g, '/'))
}

export function collectFiles(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name)
    if (statSync(p).isDirectory()) collectFiles(p, out)
    else if (/\.(ts|vue|mjs)$/.test(name)) out.push(p)
  }
  return out
}

export function findViolations(root, dirs = SCAN_DIRS, max = MAX_NON_EMPTY_LINES) {
  const bad = []
  for (const d of dirs) {
    let files
    try { files = collectFiles(join(root, d)) } catch { continue } // 目录未建时跳过
    for (const f of files) {
      if (isGenerated(relative(root, f))) continue // 生成物豁免，理由见 isGenerated 的注释
      const n = nonEmptyLines(readFileSync(f, 'utf8'))
      if (n > max) bad.push({ file: relative(root, f), lines: n })
    }
  }
  return bad
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]
if (isMain) {
  const root = fileURLToPath(new URL('..', import.meta.url))
  const bad = findViolations(root)
  for (const b of bad) console.log(`${b.file}: 非空行 ${b.lines}（上限 ${MAX_NON_EMPTY_LINES}）`)
  console.log(bad.length ? `超限 ${bad.length} 处` : '检查通过：0 处超限')
  process.exit(bad.length ? 1 : 0)
}
