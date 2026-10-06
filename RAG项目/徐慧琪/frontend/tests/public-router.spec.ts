// 公众入口的路由表就是 FR-8.7「只读」的结构性保证：字面量快照让任何新增路由
// 都必须过一次评审——写操作入口（上传/编辑/删除/导出）在这一层出现即红。
import { describe, expect, it } from 'vitest'
import { routes } from '@/public/router'

describe('公众入口路由表（只读红线）', () => {
  it('恰为三条只读路由', () => {
    expect(routes.map((r) => [r.path, r.name])).toEqual([
      ['/', 'ask'],
      ['/nav', 'nav'],
      ['/law/:lawId/articles/:no', 'article'],
    ])
  })
})
