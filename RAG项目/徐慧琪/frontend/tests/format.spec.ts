// 复制文本是律师写进文书的原料（FR-7.4）——格式就是这条需求的验收对象，
// 所以断言到字面量，而不是「包含条号」这类松判据。
import { describe, expect, it, vi } from 'vitest'
import { articleLabel, articleNo, buildCopyText, citationUrl, formatCitation } from '@/core/format'

describe('引用格式化与复制文本（FR-7.4）', () => {
  it('条号 + 引文（无款/项）', () => {
    expect(formatCitation({ article: '584', paragraph: null, item: null, quote: 'q1' }))
      .toBe('《中华人民共和国民法典》第584条：“q1”')
  })
  it('款/项存在时附上', () => {
    expect(formatCitation({ article: '509', paragraph: '2', item: '3', quote: 'q2' }))
      .toBe('《中华人民共和国民法典》第509条（第2款）（第3项）：“q2”')
  })
  it('复制文本 = 解读 + 空行 + 依据列表', () => {
    expect(buildCopyText('解读正文', [
      { article: '584', paragraph: null, item: null, quote: 'q1' },
      { article: '509', paragraph: null, item: null, quote: 'q2' },
    ])).toBe('解读正文\n\n《中华人民共和国民法典》第584条：“q1”\n《中华人民共和国民法典》第509条：“q2”')
  })
  it('引用跳转 URL', () => {
    expect(citationUrl({ article: '584', paragraph: null, item: null, quote: 'q' }))
      .toBe('/law/minfadian/articles/584')
  })
})

// 人工验收首曝缺陷的回归组：真实 citation.article 是中文全称（「第五百七十九条」），
// 旧实现按阿拉伯数字假设——展示二次包条、URL 拼成中文路径（后端实测 404）。
// 三形态规则以 tools/ask.py 的 _citation_label 为先例，中文数字边界照
// backend/tests/test_cn_num.py 选例。断言到字面量，含「不含坏形状」的反向判据。
describe('条号三形态：中文全称 / 半截 / 裸数字', () => {
  const citation = (article: string | number) =>
    ({ article, paragraph: null, item: null, quote: 'q' })

  it('全称原样保留，不二次包条', () => {
    const text = formatCitation(citation('第五百八十四条'))
    expect(text).toContain('第五百八十四条：“')
    expect(text).not.toContain('第第五百八十四条条')
  })
  it('半截「第584」补尾字，不补第二个「第」', () => {
    const text = formatCitation(citation('第584'))
    expect(text).toContain('第584条：“')
    expect(text).not.toContain('第第584条')
  })
  it('articleLabel 三形态归一', () => {
    expect(articleLabel('第五百八十四条')).toBe('第五百八十四条')
    expect(articleLabel('第584')).toBe('第584条')
    expect(articleLabel('584')).toBe('第584条')
    expect(articleLabel(584)).toBe('第584条')
  })
  it('citationUrl 一律用阿拉伯数（后端只认阿拉伯）', () => {
    expect(citationUrl(citation('第五百七十九条'))).toBe('/law/minfadian/articles/579')
    expect(citationUrl(citation('第584'))).toBe('/law/minfadian/articles/584')
    expect(citationUrl(citation(584))).toBe('/law/minfadian/articles/584')
  })
  it('中文数字边界（照 backend/tests/test_cn_num.py 选例）', () => {
    expect(articleNo('第十条')).toBe(10)
    expect(articleNo('第十五条')).toBe(15)
    expect(articleNo('第二十条')).toBe(20)
    expect(articleNo('第一百零五条')).toBe(105)
    expect(articleNo('第一千零四十条')).toBe(1040)
    expect(articleNo('第两百条')).toBe(200)
  })
  it('解析不出条号时不生成坏 URL：告警 + 落到导航页', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    expect(articleNo('附则')).toBeNaN()
    expect(citationUrl(citation('附则'))).toBe('/nav')
    expect(warn).toHaveBeenCalledTimes(1)
    warn.mockRestore()
  })
})
