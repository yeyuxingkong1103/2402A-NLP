"""第一步【在 mineru 环境运行，零参数】：MinerU 云端 API 批量解析 PDF 为 Markdown。

直接运行即可（扫描 config.mineru_pdf_dir 下全部 PDF，输出到 config.mineru_out_dir）:
    D:\\conda\\envs\\mineru\\python.exe parse_pdf.py

所有参数在 config.py 的 "MinerU 云端文档解析" 段修改（目录、档位等）；
API 密钥从环境变量 MINERU_API_KEY 读取（申请：https://mineru.net/apiManage/token）。

数据会上传到 mineru.net 云端解析，仅限公开/非敏感文档；免费额度 2000 页/天。
第二步【切换到 langchain 环境】入库:
    python main.py ingest --dir docs_mineru --domain medical
"""  # 模块级说明文档（用法、参数来源、注意事项）
import re  # 标准库：正则表达式，用于 Markdown 清洗
import sys  # 标准库：系统相关，用于缺少 Key 时退出程序
import time  # 标准库：计时（统计单文件解析耗时）
from pathlib import Path  # 标准库：面向对象的路径处理

import config  # 项目统一配置（mineru 环境也可导入：仅依赖 os/pathlib）
from mineru import MinerUApiParser  # 云端解析客户端（内部完成上传→轮询→下载）


def clean_markdown(text: str) -> str:  # 清洗 Markdown：剔除图片(base64/文件引用)、HTML表格转行文本、压空行
    text = re.sub(r"!\[[^\]]*\]\(data:image/[^)]*\)", "", text)  # 剔 base64 内嵌图（单行可达几十万字符）
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # 剔普通图片文件引用（如 ![x](images/a.png)）

    def table_to_lines(m: re.Match) -> str:  # 内部回调：把一整段 HTML 表格 -> "单元格 | 单元格"逐行文本
        lines = []  # 存放转换后的每一行文本
        for row in re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(0), flags=re.S):  # 遍历表格内的每个 <tr> 行
            cells = [re.sub(r"<[^>]+>", "", c).strip()
                     for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.S)]  # 取该行所有 <td>/<th> 内容并去标签、去首尾空白
            if any(cells):  # 至少有一个非空单元格才保留该行（跳过纯空行）
                lines.append(" | ".join(cells))  # 单元格用 " | " 连接成一行
        return "\n".join(lines)  # 所有行用换行符拼接后返回

    text = re.sub(r"<table>.*?</table>", table_to_lines, text, flags=re.S)  # 用上面回调替换每个整表（re.S 让 . 匹配换行）
    return re.sub(r"\n{3,}", "\n\n", text)  # 连续 3 个以上换行压成 1 个空行，最后返回


def main():  # 主流程：扫描 PDF -> 云端解析 -> 清洗 -> 保存为 Markdown
    if not config.mineru_api_key:  # 云端 API 必须有 Key，缺失则直接报错退出
        print("未检测到 MINERU_API_KEY 环境变量。请先在 https://mineru.net/apiManage/token 申请并设置：")  # 提示申请地址
        print('  PowerShell: $env:MINERU_API_KEY = "sk_你的key"   （或 setx 永久生效）')  # 提示设置环境变量命令
        sys.exit(1)  # 无 Key 终止程序（返回码 1）

    pdf_dir = config.mineru_pdf_dir  # 待解析 PDF 所在目录（来自 config.py）
    pdfs = sorted(pdf_dir.rglob("*.pdf")) if pdf_dir.is_dir() else []  # 递归扫描目录下所有 PDF 并排序；目录不存在则为空列表
    if not pdfs:  # 没扫到任何 PDF
        print(f"{pdf_dir} 下未找到 PDF 文件（请检查 config.py 的 mineru_pdf_dir）")  # 提示检查配置目录
        sys.exit(1)  # 无输入文件，终止程序

    out_dir = Path(config.mineru_out_dir)  # 输出目录（Markdown 保存位置）
    out_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在（parents 递归创建，已存在不报错）

    skipped = []  # 已有同名产物、本次跳过的文件（Path）
    pending = []  # 需要上传解析的文件（Path）
    for pdf in pdfs:  # 先分类：已有产物跳过，避免重复上传浪费免费额度
        if (out_dir / f"{pdf.stem}.md").exists():  # 输出目录中已存在同名 .md 产物
            skipped.append(pdf)  # 加入跳过列表
        else:  # 尚无产物
            pending.append(pdf)  # 加入待解析列表

    print(f"扫描到 PDF {len(pdfs)} 个（来源目录: {pdf_dir}，档位: {config.mineru_tier}）")  # 打印扫描统计信息
    for pdf in skipped:  # 逐个说明被跳过的文件
        print(f"[跳过] {pdf.name}：已存在同名 Markdown（需重新解析请先删除 {config.mineru_out_dir} 下对应文件）")  # 提示跳过原因与重解析方法
    if not pending:  # 全部文件都已有产物
        print("没有需要解析的文件，全部已存在。")  # 提示无需解析
        return  # 直接结束主流程

    client = MinerUApiParser(  # 创建云端解析客户端（全部参数来自 config.py）
        api_url=config.mineru_api_url,  # 云端 API 地址
        api_key=config.mineru_api_key,  # 云端 API 密钥（环境变量读取）
        tier=config.mineru_tier,  # 解析档位（精度/速度权衡）
        include_images=config.mineru_include_images,  # 是否在产物中保留图片
    )

    ok = []  # 解析成功的文件名列表
    failed = []  # 解析失败列表：(文件名, 错误信息)
    for pdf in pending:  # 逐个云端解析并保存；单个失败不影响后续文件
        print(f"[上传解析中] {pdf.name}（云端耗时取决于页数与排队）...")  # 提示开始上传与解析
        started = time.perf_counter()  # 单文件计时起点（含上传+排队+下载）
        try:  # 捕获单文件异常，保证循环继续
            result = client.parse(str(pdf))  # 内部完成 上传→轮询→下载，同步返回 ParseResult
            raw = result.markdown()  # 从解析结果中取出原始 Markdown 文本
            text = clean_markdown(raw)  # 入库前清洗（去图片、表格转文本、压空行）
            dest = out_dir / f"{pdf.stem}.md"  # 目标输出文件路径（与原 PDF 同名）
            dest.write_text(text, encoding="utf-8")  # 以 UTF-8 写入磁盘
            ok.append(pdf.name)  # 记录成功文件名
            print(f"[完成] {pdf.name} -> {dest}  ({len(raw)} -> {len(text)} 字符，耗时 {time.perf_counter() - started:.1f}s)")  # 打印完成信息（清洗前后字符数、耗时）
        except Exception as exc:  # 记录失败并继续下一个文件
            failed.append((pdf.name, f"{type(exc).__name__}: {exc}"))  # 保存文件名与错误描述
            print(f"[失败] {pdf.name}: {type(exc).__name__}: {exc}（耗时 {time.perf_counter() - started:.1f}s）")  # 打印失败原因与耗时

    print("-" * 60)  # 打印汇总分隔线
    print(f"解析结束：成功 {len(ok)} 个，失败 {len(failed)} 个，跳过 {len(skipped)} 个")  # 打印总体统计
    for name in ok:  # 遍历成功清单
        print(f"  成功: {name}")  # 逐条打印成功文件名
    for name, err in failed:  # 遍历失败清单
        print(f"  失败: {name} -> {err}")  # 逐条打印失败文件名与错误原因


if __name__ == "__main__":  # 仅当本文件被直接运行时（被 import 时不执行）
    main()  # 调用主流程
