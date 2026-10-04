<template>
  <footer class="site-footer">
    <!-- FR-8.4 的页脚常驻位：免责不依赖某次回答的 disclaimer 字段，
         每个页面都带着这句话（文案断言在 site-footer.spec.ts） -->
    <p class="site-footer__disclaimer">
      本系统输出仅为一般性法律信息，不构成正式法律意见，具体案件请咨询律师。
    </p>
    <!-- 必须走 RouterLink（与 CitationList 同一笔账，终审 I1）：普通锚点 href
         '/nav' 是整页导航，把律师用户送出律师壳进入公众导航页；RouterLink 在
         当前入口的路由表内解析（两入口都有 nav 路由），点击留在本入口内 -->
    <RouterLink class="site-footer__nav" to="/nav">法条浏览</RouterLink>
  </footer>
</template>
