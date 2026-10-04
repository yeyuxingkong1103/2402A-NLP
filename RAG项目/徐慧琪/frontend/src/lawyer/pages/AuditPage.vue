<template>
  <section class="audit-page">
    <h2 class="audit-page__title">审计导出</h2>
    <!-- 只有 partner 渲染导出入口（设计 §四）：普通律师打开这一页看到说明而不是
         一个会被后端拒绝的按钮；判据在测试里对两个角色各钉一向 -->
    <template v-if="isPartner">
      <p class="audit-page__hint">导出最近 24 小时的审计记录（最多 1000 行）。</p>
      <el-button type="primary" data-test="export" :loading="loading" @click="doExport">
        导出
      </el-button>
      <p v-if="error" class="audit-page__error" data-test="error">{{ error }}</p>
      <p v-if="exported !== null" class="audit-page__done" data-test="done">
        已导出 {{ exported }} 条记录
      </p>
    </template>
    <p v-else class="audit-page__denied" data-test="denied">
      当前角色没有审计导出权限（仅限 partner）。
    </p>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ApiError } from '@/core/api/request'
import { exportAudit } from '@/core/api/lawyer'
import { session } from '@/core/useAuth'
import { handleAuthExpired } from '@/lawyer/router'

const router = useRouter()
const isPartner = computed(() => session.role === 'partner')
const loading = ref(false)
const error = ref('')
const exported = ref<number | null>(null)

async function doExport() {
  if (loading.value) return
  error.value = ''
  loading.value = true
  try {
    const rows = await exportAudit()
    // Blob 下载：文件名是固定契约（审查与验收按它找文件），导出内容就是后端
    // 返回的数组原样，前端不重排不筛列
    const blob = new Blob([JSON.stringify(rows)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'audit-export.json'
    a.click()
    URL.revokeObjectURL(url)
    exported.value = rows.length
  } catch (e) {
    if (!(e instanceof ApiError)) throw e
    if (handleAuthExpired(e, router)) return
    error.value = e.message
  } finally {
    loading.value = false
  }
}
</script>
