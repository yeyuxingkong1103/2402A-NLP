<!-- 第 9 步：资料上传页面，上传 PDF 并查看解析结果 -->
<template>
  <!-- 页面容器 -->
  <div class="page">
    <el-card>
      <!-- 卡片标题 -->
      <template #header>上传 PDF 资料</template>
      <!-- 上传控件：多选、只收 pdf、隐藏自带文件列表（自己维护队列） -->
      <el-upload drag multiple accept=".pdf" :show-file-list="false"
                 :http-request="doUpload">
        <!-- 图标 -->
        <el-icon class="up-icon"><UploadFilled /></el-icon>
        <!-- 提示文字 -->
        <div class="up-text">把 PDF 拖到这里，或<em>点击选择文件</em>（支持多选排队）</div>
        <!-- 补充说明 -->
        <div class="up-hint">本接口只解析不入库；入库请到「知识库」页点重建</div>
      </el-upload>

      <!-- 上传队列 -->
      <el-table v-if="queue.length" :data="queue" class="gap">
        <!-- 文件名 -->
        <el-table-column prop="file_name" label="文件名" min-width="180" />
        <!-- 状态：排队中 / 上传中 / 成功 / 失败 -->
        <el-table-column prop="status" label="状态" width="110">
          <template #default="{ row }">
            <!-- 按状态给不同颜色的标签 -->
            <el-tag :type="tagType(row.status)" size="small">{{ row.status }}</el-tag>
          </template>
        </el-table-column>
        <!-- 进度条 -->
        <el-table-column label="进度" width="180">
          <template #default="{ row }">
            <el-progress :percentage="row.percent" :status="row.status === '失败' ? 'exception' : undefined" />
          </template>
        </el-table-column>
        <!-- 页数 -->
        <el-table-column prop="pages" label="页数" width="80" />
        <!-- 切分块数 -->
        <el-table-column prop="chunks" label="块数" width="80" />
        <!-- 失败原因 -->
        <el-table-column prop="error" label="说明" min-width="160" />
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { reactive } from 'vue'                  // 导入响应式工具
import { ElMessage } from 'element-plus'        // 导入消息提示
import { uploadFile } from '../api'             // 导入上传接口

// 上传队列：每行对应一个文件，先入队占位再随进度更新
const queue = reactive([])

// 状态到标签颜色的映射
function tagType(status) {
  if (status === '成功') return 'success'        // 成功绿色
  if (status === '失败') return 'danger'         // 失败红色
  if (status === '上传中') return 'primary'      // 上传中蓝色
  return 'info'                                  // 排队中灰色
}

// el-upload 的自定义上传：接管每个文件的请求，自己做进度与状态
async function doUpload(options) {
  const file = options.file                                   // 当前文件
  // 入队占位，后面的进度回调直接改这一行
  const row = reactive({ file_name: file.name, status: '上传中', percent: 0,
                         pages: 0, chunks: 0, error: '' })
  queue.push(row)
  try {
    // 调后端上传接口，onProgress 回传百分比
    const data = await uploadFile(file, (p) => { row.percent = p })
    row.percent = 100                                         // 收尾置满
    row.status = '成功'                                       // 状态改为成功
    row.pages = data.pages                                    // 解析页数
    row.chunks = data.chunks                                  // 切分块数
    ElMessage.success(`${file.name} 解析完成`)                  // 提示
    options.onSuccess(data)                                   // 通知 el-upload 组件
  } catch (err) {
    row.status = '失败'                                       // 状态改为失败
    row.error = err?.response?.data?.detail || err.message || '上传失败'   // 记录原因
    options.onError(err)                                      // 通知 el-upload 组件
  }
}
</script>

<style scoped>
/* 页面容器 */
.page {
  max-width: 900px;         /* 限制宽度 */
  margin: 0 auto;           /* 水平居中 */
}
/* 上传区图标 */
.up-icon {
  font-size: 48px;          /* 放大图标 */
  color: #c0c4cc;           /* 浅灰 */
}
/* 上传区提示文字 */
.up-text {
  margin-top: 8px;          /* 与图标留白 */
  color: #606266;           /* 正文色 */
}
/* 提示文字中的强调部分 */
.up-text em {
  color: #409eff;           /* 主题蓝 */
  font-style: normal;       /* 不用斜体 */
}
/* 补充说明小字 */
.up-hint {
  margin-top: 4px;          /* 上间距 */
  font-size: 12px;          /* 小字 */
  color: #909399;           /* 次要色 */
}
/* 队列表格与上传区留白 */
.gap {
  margin-top: 16px;         /* 上间距 */
}
</style>
