// 产物闸门自身的判据：公众包里出现律师侧专用端点串即红线（设计 §三 定调 1/3），
// 但 marker 必须按**真实产物形态**匹配——产物里是端点后缀字面量（反引号/引号定界，
// `/api/v1` 前缀由运行时拼接），不是完整的 `/api/v1/qa`；且必须**带边界**：
// 没有边界时 `/public/qa` 会把 `/qa` 带响、`/cases/search` 会把 `/search` 带响，
// 闸门当场变误报机器（控制者 2026-10-03 实测产物后的裁决①，机械偏离计划字面）。
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { collectModuleTexts, findForbiddenMarkers } from '../scripts/check-public-bundle.mjs'

// 期望值自备一份，不从被测脚本 import MARKERS：脚本把 marker 改错时，这份 fixture
// 要能红给出来（测试不该拿被测对象的常量当期望）
const MARKERS = ['/qa', '/search', '/cases/search', '/admin/audit/export']

describe('check-public-bundle：律师侧端点串扫描', () => {
  it('命中真实产物的字面量形态（反引号定界的 /qa）', () => {
    // 计划 given 的 `const p="/api/v1/qa"` 在产物里不存在（前缀运行时拼接），
    // 照抄会让正对照假红——按 2026-10-03 产物实测的真实形态改写
    const text = '...return b(`/qa`,{method:`POST`,body:{question:e},...})'
    expect(findForbiddenMarkers(text, MARKERS)).toEqual(['/qa'])
  })

  it('共享端点不许误伤：/public/qa 是公众端点，不是 /qa', () => {
    const text = '...return Il(`/public/qa`,{method:`POST`,body:{question:e},...})'
    expect(findForbiddenMarkers(text, MARKERS)).toEqual([])
  })

  it('后缀子串必须有边界：/cases/search 不许把 /search 带响', () => {
    const text = 'return b(`/cases/search`,{method:`POST`,auth:!0,...})'
    expect(findForbiddenMarkers(text, MARKERS)).toEqual(['/cases/search'])
  })

  it('四个 marker 同时在场的文本全部报出（律师包正对照的形态）', () => {
    const text = 'b(`/qa`,{});b(`/search`,{});b(`/cases/search`,{});b(`/admin/audit/export`,{})'
    expect(findForbiddenMarkers(text, MARKERS)).toEqual(MARKERS)
  })
})

describe('check-public-bundle：入口 → 共享 chunk 的传递闭包', () => {
  // 入口 js 只是壳：公众包的实际代码在共享 chunk 里（2026-10-03 实测：两入口都
  // import"./NavTree-....js"）。只扫入口文件时，「律师代码混进共享 chunk」是绿的——
  // 闸门必须在传递闭包上判（裁决②）。夹具走两层 import，证明闭包真的递归。
  it('沿静态 import 递归到二层，且闭包内的 marker 能被扫出', () => {
    const root = mkdtempSync(join(tmpdir(), 'bundle-'))
    mkdirSync(join(root, 'assets'))
    writeFileSync(join(root, 'index.html'),
      '<script type="module" crossorigin src="/assets/entry.js"></script>')
    writeFileSync(join(root, 'assets', 'entry.js'), 'import"./chunk-a.js";')
    writeFileSync(join(root, 'assets', 'chunk-a.js'), 'import"./chunk-b.js";export const a=1')
    writeFileSync(join(root, 'assets', 'chunk-b.js'), 'const p=`/qa`')
    const files = collectModuleTexts(join(root, 'index.html'), { rootDir: root })
    expect(files.map((f) => basename(f.file))).toEqual([
      'index.html', 'entry.js', 'chunk-a.js', 'chunk-b.js',
    ])
    expect(findForbiddenMarkers(files.map((f) => f.text).join('\n'), MARKERS)).toEqual(['/qa'])
  })
})
