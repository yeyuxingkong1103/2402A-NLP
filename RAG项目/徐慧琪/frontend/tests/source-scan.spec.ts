// 两条红线用「扫源码」承重：v-html 一旦出现，渲染方式即 XSS 面；公众入口
// 一旦出现本地存储，AC-19 的「不采集」就在前端破功。扫描比约定可靠。
import { readdirSync, readFileSync, statSync } from 'node:fs'
// 不用「new URL('../src', import.meta.url)」解析路径：本仓 vitest 5 + jsdom 下
// 该表达式的结果随模块上下文漂——有的上下文里全局 URL 是 jsdom 的 whatwg-url，
// 把 file: 基准解析成 http://localhost:3000/src；有的上下文里绑定的 URL 类对
// 盘符基准解析出非 file: 协议，fileURLToPath 直接抛「The URL must be of scheme
// file」（实测，两种红都不是判据本身要抓的东西）。fileURLToPath 吃字符串则只
// 经过 Node 自己的解析，dirname + resolve 拼出 src——行为不随上下文漂
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'

const SRC = resolve(dirname(fileURLToPath(import.meta.url)), '../src')

function filesUnder(dir: string): string[] {
  return readdirSync(dir).flatMap((n) => {
    const p = join(dir, n)
    return statSync(p).isDirectory() ? filesUnder(p) : [p]
  })
}

describe('源码红线扫描', () => {
  it('扫描范围含子目录：递归是三条断言的共同地基（失效则它们绿着失效）', () => {
    // 三条扫描全靠 filesUnder 递归：src/public/pages 等红线所在目录都在子目录里，
    // 只扫一层时违规文件根本进不了扫描面，三条断言仍全绿——绿的不是「没有违规」，
    // 是「没扫到」。这条钉住递归本身，让那三条的绿有含义；路径写法与 filesUnder
    // 内部一致（join 拼接），与平台无关
    expect(filesUnder(SRC)).toContain(join(SRC, 'public', 'pages', 'AskPage.vue'))
  })

  it('全仓无 v-html', () => {
    // 判据是**属性用法**（v-html=）而不是裸字串：裸字串会把注释里对这条红线的
    // 说明（AnswerCard.vue：「正文只走文本插值（不 v-html）」）也判红——那条
    // 注释是红线的解释文本，不是 XSS 面；扫描要红在真正的渲染方式上
    // （控制者裁决 B：schema 给定写法在本仓注定恒红，改为属性用法正则）
    const hits = filesUnder(SRC).filter((f) => /\bv-html\s*=/.test(readFileSync(f, 'utf8')))
    expect(hits).toEqual([])
  })

  it('公众入口无任何本地存储与 cookie 写入', () => {
    const patterns = [/localStorage/, /sessionStorage/, /document\.cookie/]
    const hits = filesUnder(join(SRC, 'public')).filter((f) =>
      patterns.some((re) => re.test(readFileSync(f, 'utf8'))))
    expect(hits).toEqual([])
  })

  it('公众入口不 import useAuth（本地存储的唯一所在地在律师入口）', () => {
    const hits = filesUnder(join(SRC, 'public')).filter((f) =>
      readFileSync(f, 'utf8').includes('useAuth'))
    expect(hits).toEqual([])
  })
})
