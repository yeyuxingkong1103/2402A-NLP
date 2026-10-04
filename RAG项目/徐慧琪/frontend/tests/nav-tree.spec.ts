// 导航树是「点条号 → 看原文」的入口：只有文章叶算选择，目录节点点击不该
// emit（否则页面会拿着一个目录去请求条文详情）；过滤按 label 缩窄，且清空
// 过滤必须恢复全树（filter-node-method 对空串返回 false 会把树整体藏掉）。
import { describe, expect, it } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import NavTree from '@/components/NavTree.vue'
import type { NavNode } from '@/core/api/public'

// 两层足够：叶在目录下、目录有自己的文章叶（与 nav.spec 的四层互补——
// 这里要的是交互，不是层级数）
const NODES: NavNode[] = [{
  title: '第一编 总则', level: 0, path: '第一编 总则',
  articles: [{ article_no: 1, article_no_cn: '第一条' }],
  children: [{
    title: '第一章 基本规定', level: 1, path: '第一编 总则 > 第一章 基本规定',
    articles: [{ article_no: 10, article_no_cn: null }],
    children: [],
  }],
}]

/** 按自己的 label 精确找内容行：祖先的 content 只含自己的 label（子节点在兄弟 div 里） */
function contentOf(w: ReturnType<typeof mount>, label: string) {
  const hit = w.findAll('.el-tree-node__content').find((c) => c.text() === label)
  if (!hit) throw new Error(`未找到节点：${label}`)
  return hit
}

/** 节点是否可见：el-tree 过滤用 v-show 在节点根 div 上切 display */
function visible(w: ReturnType<typeof mount>, label: string) {
  const el = contentOf(w, label).element.closest('.el-tree-node') as HTMLElement | null
  return el?.style.display !== 'none'
}

async function expand(w: ReturnType<typeof mount>, label: string) {
  await contentOf(w, label).find('.el-tree-node__expand-icon').trigger('click')
  await flushPromises()
}

describe('NavTree', () => {
  it('点击文章叶 emit select(10)；目录节点点击不 emit', async () => {
    const w = mount(NavTree, { props: { nodes: NODES } })
    await expand(w, '第一编 总则')
    await expand(w, '第一章 基本规定')
    // 目录节点不算选择（展开只是浏览结构）
    await contentOf(w, '第一章 基本规定').trigger('click')
    expect(w.emitted('select')).toBeUndefined()
    await contentOf(w, '10').trigger('click')
    expect(w.emitted('select')).toEqual([[10]])
  })

  it('过滤按 label 缩窄；清空过滤恢复全树', async () => {
    const w = mount(NavTree, { props: { nodes: NODES } })
    await w.get('input').setValue('基本')
    await flushPromises()
    expect(visible(w, '第一章 基本规定')).toBe(true) // 命中本节点
    expect(visible(w, '第一编 总则')).toBe(true)     // 自身不命中但后代命中：路径保留
    expect(visible(w, '第一条')).toBe(false)          // 不命中的叶被藏掉
    await w.get('input').setValue('')
    await flushPromises()
    expect(visible(w, '第一条')).toBe(true) // 空串 = 不过滤，不是全藏
  })
})
