// 导航树整形：后端 NavNode（递归结构，层级数由数据决定）→ el-tree 可吃的
// 扁平化节点。纯函数、不碰 DOM 与网络——key 与排序是跨任务的接口约定，
// 由 nav.spec.ts 钉死（Task 5/6 的页面只消费，不重排）。
import type { NavNode } from '@/core/api/public'

export interface TreeNode {
  key: string
  label: string
  articleNo?: number
  children: TreeNode[]
}

// 目录节点 key 用 path 而不是 title：同名的「第一章」在不同编下重复出现，
// title 作 key 会让 el-tree 的 node-key 撞车（高亮/过滤元数据串到另一章）；
// path 是后端给的「本级唯一落点」，天然唯一。
//
// children 顺序 = 本节点自己的文章叶在前、子节点在后（接口约定，测试钉死）。
// 语义上这是「本级直管的条文」先于「下级结构」——总则编直管第 1~9 条，
// 若子节点在前，用户展开「第一编」会先看到各章空壳、看不到总则条文本身。
export function toTreeData(nodes: NavNode[]): TreeNode[] {
  return nodes.map((node) => ({
    key: node.path,
    label: node.title,
    children: [
      ...node.articles.map((article): TreeNode => ({
        key: `a-${article.article_no}`,
        // 中文条号缺失（后端允许 null）时退回阿拉伯数字，不渲染空标题
        label: article.article_no_cn ?? String(article.article_no),
        // 叶子的 articleNo 是 NavTree 点击 emit select(articleNo) 的唯一来源
        articleNo: article.article_no,
        children: [],
      })),
      ...toTreeData(node.children),
    ],
  }))
}
