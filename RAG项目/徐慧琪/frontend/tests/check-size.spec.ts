// 规模闸门本身要有判据：300 与 301 的边界必须精确（差一位就等于没有闸门），
// 且要覆盖「空行不计入」——否则一个塞满空行的文件会绕过限制。
import { describe, expect, it } from 'vitest'
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { findViolations, isGenerated, nonEmptyLines } from '../scripts/check-size.mjs'

function fixture(lines: number): string {
  return Array.from({ length: lines }, (_, i) => `const v${i} = ${i}`).join('\n')
}

describe('check-size', () => {
  it('空行不计入非空行数', () => {
    expect(nonEmptyLines('a\n\n\nb\n   \nc')).toBe(3)
  })

  it('300 行通过、301 行违规（边界精确）', () => {
    const root = mkdtempSync(join(tmpdir(), 'size-'))
    mkdirSync(join(root, 'src'))
    writeFileSync(join(root, 'src', 'ok.ts'), fixture(300))
    expect(findViolations(root)).toEqual([])
    writeFileSync(join(root, 'src', 'bad.ts'), fixture(301))
    expect(findViolations(root)).toEqual([{ file: join('src', 'bad.ts'), lines: 301 }])
  })

  it('递归进子目录，且 .vue/.mjs 也在扫描范围内', () => {
    // 上一条用平铺的 .ts 夹具，对「递归」与「扩展名过滤」两条行为不敏感：真实工程里
    // src/ 下的文件几乎都在子目录、且多数是 .vue——扫描器退化成「只扫顶层 .ts」时，
    // check:size 会以「0 处超限」静默失效。所以这里两条行为各钉一个夹具。
    const root = mkdtempSync(join(tmpdir(), 'size-'))
    mkdirSync(join(root, 'src', 'nest'), { recursive: true })
    writeFileSync(join(root, 'src', 'nest', 'bad.vue'), fixture(301)) // 子目录 + .vue：要递归且认 .vue
    writeFileSync(join(root, 'src', 'bad.mjs'), fixture(301)) // 顶层 + .mjs：要过滤含 .mjs
    // readdirSync 的顺序不保证，排序后再断言，避免判据被文件系统顺序绑架
    const found = findViolations(root).sort((a, b) => (a.file < b.file ? -1 : a.file > b.file ? 1 : 0))
    expect(found).toEqual([
      { file: join('src', 'bad.mjs'), lines: 301 },
      { file: join('src', 'nest', 'bad.vue'), lines: 301 },
    ])
  })

  it('生成物豁免：白名单文件超限不违规，同名变体仍违规', () => {
    // api-types.gen.ts 是 openapi-typescript 产物（体量由后端契约决定，当前 945 行），
    // 拆不动也不该手改；但豁免必须**只认 src/core/api-types.gen.ts 这一个精确路径**：
    // 后缀制下任何人把大文件改名成 x.gen.ts 就整体躲开 300 行闸门。夹具两向都钉——
    // 豁免生效（白名单文件不报）与豁免不越界（同名变体、手写文件照报），缺一条
    // 都测不出「豁免范围错了」
    const root = mkdtempSync(join(tmpdir(), 'size-'))
    mkdirSync(join(root, 'src', 'core'), { recursive: true })
    writeFileSync(join(root, 'src', 'core', 'api-types.gen.ts'), fixture(301))
    writeFileSync(join(root, 'src', 'other.gen.ts'), fixture(301)) // 后缀相同、路径不同
    writeFileSync(join(root, 'src', 'api-types.gen.ts'), fixture(301)) // 同名、目录不对
    writeFileSync(join(root, 'src', 'core', 'handwritten.ts'), fixture(301)) // 手写 .ts
    const found = findViolations(root).sort((a, b) => (a.file < b.file ? -1 : a.file > b.file ? 1 : 0))
    expect(found).toEqual([
      { file: join('src', 'api-types.gen.ts'), lines: 301 },
      { file: join('src', 'core', 'handwritten.ts'), lines: 301 },
      { file: join('src', 'other.gen.ts'), lines: 301 },
    ])
  })

  it('isGenerated 只认白名单精确路径（不认后缀，反斜杠先归一）', () => {
    expect(isGenerated('src/core/api-types.gen.ts')).toBe(true)
    expect(isGenerated('src\\core\\api-types.gen.ts')).toBe(true) // Windows 相对路径
    expect(isGenerated('src/other.gen.ts')).toBe(false) // 后缀相同、路径不同
    expect(isGenerated('src/api-types.gen.ts')).toBe(false) // 同名、目录不对
    expect(isGenerated('tests/core/api-types.gen.ts')).toBe(false) // 同名、扫描根不对
    expect(isGenerated('src/core/handwritten.ts')).toBe(false)
    // 名字里带 gen 但不是 .gen.ts 的（目录名、其它扩展名）不豁免
    expect(isGenerated('src/gen/handwritten.ts')).toBe(false)
    expect(isGenerated('src/core/notes.gen.md')).toBe(false)
  })
})
