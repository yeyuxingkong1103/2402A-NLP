"""MinerU layout 模型配置兼容层。

历史背景：在 transformers 4.44.2（无原生 hgnet_v2）下，本项目曾用 ResNet 子类
伪造 `hgnet_v2` 配置，导致 layout 模型权重加载不匹配、检测全空。

当前状态：依赖栈已升级到 transformers 5.x（原生支持 `hgnet_v2` backbone）与
mineru 3.4.5，`hgnet_v2` / `pp_doclayout_v2` 均由 transformers 原生注册，
本模块不再需要任何注册。保留此文件仅为兼容历史导入；**请勿在此注册
ResNet 版 HGNetV2Config**，否则会覆盖原生实现并重新破坏 layout 检测。
"""
