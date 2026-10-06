<template>
  <section class="nav-tree">
    <el-input
      v-model="filterText"
      class="nav-tree__filter"
      data-test="filter"
      placeholder="输入条号或标题过滤"
      clearable
    />
    <el-tree
      ref="treeRef"
      class="nav-tree__tree"
      :data="treeData"
      :props="{ label: 'label', children: 'children' }"
      node-key="key"
      :filter-node-method="filterNode"
      @node-click="onNodeClick"
    />
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { toTreeData } from '@/core/nav'
import type { TreeNode } from '@/core/nav'
// import type：编译期擦除，组件不产生指向 api/public.ts 的运行时模块边（裁决 1）
import type { NavNode } from '@/core/api/public'

const props = defineProps<{ nodes: NavNode[] }>()
const emit = defineEmits<{ select: [articleNo: number] }>()

// 转树与排序约定全在 core/nav.ts（本节点文章叶在前、子目录在后），组件只消费
const treeData = computed(() => toTreeData(props.nodes))

const filterText = ref('')
// el-tree 的过滤是命令式 API（实例方法 filter(value)）。这里只声明用到的
// 最小结构类型，组件无需 import element-plus 的类型/运行时入口。
const treeRef = ref<{ filter: (value: string) => void } | null>(null)
watch(filterText, (value) => treeRef.value?.filter(value))

// 空串必须显式全过：给了 filter-node-method 后 el-tree 对每个节点都问它，
// 返回 false 会把整棵树藏掉（清空输入框 = 恢复全树，不是清空树）
function filterNode(value: string, data: TreeNode) {
  if (!value) return true
  return data.label.includes(value)
}

// 只有文章叶带 articleNo；目录节点的点击不 emit（展开/收起由 el-tree 自己管）
function onNodeClick(data: TreeNode) {
  if (data.articleNo != null) emit('select', data.articleNo)
}
</script>
