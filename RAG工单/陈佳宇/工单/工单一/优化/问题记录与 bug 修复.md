# Bug记录与修复
1. Bug：Milvus检索时报错 `struct.error: required argument is not a float`
原因：sentence-transformers返回numpy.float32，Milvus不识别numpy类型。
解决方案：增加to_python_float_list函数，循环转为原生Python float列表。

2. Bug：网页Favicon.ico 404警告
原因：页面没有图标文件。
解决方案：忽略警告，不影响业务功能。

# 其他小问题
- 重启项目重复建表：Milvus集合每次启动自动删除重建，保证向量和文档同步。

