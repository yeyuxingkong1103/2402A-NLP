// 树的层级数由数据决定（本库有 2/3/4 三种深度）——写死三层的转树会静默丢掉
// 四层节点，所以 fixture 用四层并断言最深叶子仍在。
import { describe, expect, it } from 'vitest'
import { toTreeData } from '@/core/nav'

const NODES = [{
  title: '第一编 总则', level: 0, path: '第一编 总则',
  articles: [{ article_no: 1, article_no_cn: '第一条' }],
  children: [{
    title: '第一章 基本规定', level: 1, path: '第一编 总则 > 第一章 基本规定',
    articles: [{ article_no: 10, article_no_cn: null }],
    children: [{
      title: '第一节 示例', level: 2, path: '第一编 总则 > 第一章 基本规定 > 第一节 示例',
      articles: [],
      // 再深一层（第 4 级 NavNode）：fixture 必须真的比三层深，否则「写死
      // 递归层数」的回归不会丢任何节点、也不会变红——这正是它要防的失效
      children: [{
        title: '第一目 示例', level: 3, path: '第一编 总则 > 第一章 基本规定 > 第一节 示例 > 第一目 示例',
        articles: [{ article_no: 100, article_no_cn: '第一百条' }],
        children: [],
      }],
    }],
  }],
}]

describe('toTreeData', () => {
  it('四层结构原样保留；文章叶 key 稳定、中文条号缺失时退回数字', () => {
    const tree = toTreeData(NODES as never)
    const lvl0 = tree[0]
    expect(lvl0.label).toBe('第一编 总则')
    expect(lvl0.children[0].key).toBe('a-1')            // 本节点的文章叶在前
    const lvl1 = lvl0.children[1]                        // 子结构在后（顺序是约定）
    expect(lvl1.label).toBe('第一章 基本规定')
    expect(lvl1.children[0].key).toBe('a-10')
    expect(lvl1.children[0].label).toBe('10')            // 中文条号缺失 → 数字
    expect(lvl1.children[1].label).toBe('第一节 示例')   // 第三层不丢
    // 目录节点 key 是 path 而非 title（el-tree 的 node-key）：同名目录跨编重复，
    // title 作 key 会撞车——四个层级的 key 逐个钉住
    expect(lvl0.key).toBe('第一编 总则')
    expect(lvl1.key).toBe('第一编 总则 > 第一章 基本规定')
    // 第四层不丢：写死递归层数的转树到这里会静默少一层，必须在断言上变红
    const lvl2 = lvl1.children[1]
    expect(lvl2.label).toBe('第一节 示例')
    expect(lvl2.key).toBe('第一编 总则 > 第一章 基本规定 > 第一节 示例')
    expect(lvl2.children).toHaveLength(1)
    const lvl3 = lvl2.children[0]
    expect(lvl3.label).toBe('第一目 示例')
    expect(lvl3.key).toBe('第一编 总则 > 第一章 基本规定 > 第一节 示例 > 第一目 示例')
    expect(lvl3.children[0].key).toBe('a-100')            // 最深那层的文章叶
    expect(lvl3.children[0].label).toBe('第一百条')
  })

  it('目录节点 key 用 path：同名「第一章」在不同编下不撞 key', () => {
    // 同一条约定（key=path）的第二向：真实语料里「第一章」跨编重名，title 作
    // node-key 会让 el-tree 的节点状态串到另一章——这条用例钉「同名不撞」
    const tree = toTreeData([
      {
        title: '第一编 总则', level: 0, path: '第一编 总则',
        articles: [],
        children: [{
          title: '第一章 基本规定', level: 1, path: '第一编 总则 > 第一章 基本规定',
          articles: [], children: [],
        }],
      },
      {
        title: '第二编 物权', level: 0, path: '第二编 物权',
        articles: [],
        children: [{
          title: '第一章 基本规定', level: 1, path: '第二编 物权 > 第一章 基本规定',
          articles: [], children: [],
        }],
      },
    ] as never)
    const first = tree[0].children[0]
    const second = tree[1].children[0]
    expect(first.label).toBe('第一章 基本规定')
    expect(second.label).toBe('第一章 基本规定')                  // 标题同名
    expect(first.key).toBe('第一编 总则 > 第一章 基本规定')        // key 取 path
    expect(second.key).toBe('第二编 物权 > 第一章 基本规定')
    expect(first.key).not.toBe(second.key)                        // 同名不同 path：不撞
  })
})
