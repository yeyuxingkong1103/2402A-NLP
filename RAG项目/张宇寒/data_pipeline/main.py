"""公共法律知识库的离线处理入口。

可以把这个文件理解成离线数据管道的“总开关”：

1. 读取命令行参数，判断要处理全部公共资料还是只预览一份资料；
2. 调用 :mod:`data_pipeline.index` 完成读取、解析、清洗等工作；
3. 默认只把处理结果保存成 JSON，避免误改数据库；
4. 只有显式传入 ``--append`` 或 ``--rebuild`` 时，才连接 Milvus 并写入向量。

这个文件不负责具体解析 PDF、分块或生成向量，它只负责安排调用顺序。
"""

# 允许类型注解引用尚未定义的类型，减少类型之间相互导入的问题。
from __future__ import annotations

# argparse 用来读取启动命令中的 --data-dir、--append 等参数。
import argparse
# sys 用来临时补充 Python 的模块查找路径。
import sys
# Path 用统一、跨平台的方式表示文件和文件夹路径。
from pathlib import Path

# 当前文件位于 data_pipeline 目录；parents[1] 是项目根目录。
# 把项目根目录放到模块查找路径开头后，直接运行本文件时也能导入 backend。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# get_settings 从 .env 和默认值中读取 Milvus、模型等项目配置。
from backend.app.config import get_settings
# ModelGateway 是模型统一入口，入库时通过它调用 BGE-M3 嵌入模型。
from backend.app.models import ModelGateway
# 下面五个函数分别负责：追加入库、构建处理结果、单文件预览、重建入库和入库前检查。
from data_pipeline.index import (
    append_public_collections,
    build_public_records,
    process_public_file,
    rebuild_public_collections,
    validate_public_input,
)
# file_sha256 计算文件指纹，用来给单文件预览目录生成不会轻易重复的名称。
from data_pipeline.load import file_sha256
# SOURCE_FILES 保存“集合名称 -> 原始文件名”的对应关系。
from data_pipeline.parse import SOURCE_FILES


def ensure_milvus_database(settings) -> None:
    """确保配置中指定的 Milvus 数据库已经存在。

    参数 ``settings`` 是项目配置对象。函数没有返回值；数据库不存在时创建，
    已经存在时什么也不改。把 pymilvus 放在函数内导入，可以让只做本地预览的
    命令不必提前连接或初始化 Milvus。
    """

    # MilvusClient 是 pymilvus 提供的数据库客户端。
    from pymilvus import MilvusClient

    # 先连接 Milvus 服务；这里还没有指定 db_name，因为目标数据库可能尚不存在。
    client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token)
    # 读取服务中已有的数据库名称。
    databases = client.list_databases()
    # 只有目标数据库不存在时才创建，避免每次运行都重复创建。
    if settings.milvus_database not in databases:
        client.create_database(settings.milvus_database)
    # 操作结束后主动关闭连接，释放网络资源。
    client.close()


def main() -> int:
    """执行一次公共知识库离线处理任务，并返回进程退出码。

    返回 ``0`` 表示任务正常完成。主要有三种用法：

    - 不加写库参数：处理并保存 JSON，Milvus 不变；
    - ``--append``：保留旧数据并追加本批数据；
    - ``--rebuild``：用本批完整数据重建公共集合。
    """

    # 创建命令行参数解析器，并把说明文字显示在 --help 中。
    parser = argparse.ArgumentParser(description="处理公共法律资料；默认仅保存文件，不连接数据库")
    # 原始公共资料所在目录；未指定时读取项目下的 data/public。
    parser.add_argument("--data-dir", default="data/public", help="公共法律原始数据目录")
    # 处理后 JSON 的保存目录；不传时使用 data/processed。
    parser.add_argument("--output-dir", help="处理结果输出目录；单份资料默认单独保存到预览目录")
    # --file 用于只验证和预览一份新资料，不直接入库。
    parser.add_argument("--file", type=Path, help="只处理这一份新增资料")
    # 单文件预览时必须说明资料属于哪个公共集合；自动派生的 civil_elements 不能手选。
    parser.add_argument("--collection", choices=[name for name in SOURCE_FILES if name != "civil_elements"], help="单份资料所属分类")
    # PDF 读取策略：本地读取、OCR 后必要时多模态补读，或直接用多模态读取。
    parser.add_argument("--pdf-method", choices=["auto", "auto-vision", "vision"], default="auto",
                        help="auto只在本地读取；auto-vision先OCR、失败再多模态补读；vision直接多模态（后两者会上传页面）")
    # append 和 rebuild 会改变数据库，二者不能同时出现，所以放进互斥参数组。
    mode = parser.add_mutually_exclusive_group()
    # 追加模式不删除旧集合，适合确认不会重复的新批次资料。
    mode.add_argument("--append", action="store_true", help="将处理结果追加到数据库，不删除旧集合；不要重复追加同一批资料")
    # 重建模式会替换旧集合，适合完整数据重新建库。
    mode.add_argument("--rebuild", action="store_true", help="重建公共集合，会替换原有数据，谨慎使用")
    # 真正解析用户在终端中输入的参数，结果保存在 args 对象里。
    args = parser.parse_args()

    # bool(args.file) 与 bool(args.collection) 必须同时为真或同时为假。
    # 也就是说，指定单文件时必须同时指定它的集合，反过来也一样。
    if bool(args.file) != bool(args.collection):
        parser.error("--file和--collection需要一起使用。")
    # 单文件模式只用于预览，禁止与会修改整库的 append/rebuild 一起使用。
    if args.file and (args.append or args.rebuild):
        parser.error("单份资料只供预览；入库需要完整资料共同校验编号和关联关系。")
    # 多模态读取可能上传页面，因此必须让用户用 --file 明确指定目标文件。
    if not args.file and args.pdf_method != "auto":
        parser.error("多模态读取请通过--file明确选择一份PDF或图片。")

    # 如果没有指定输出目录，就把处理结果写到 data/processed。
    output = Path(args.output_dir or "data/processed")
    # 单文件且未手动指定输出目录时，为它建立独立预览目录，防止覆盖完整知识库结果。
    if args.file and not args.output_dir:
        # 文件名前八位哈希可以区分“同名但内容不同”的资料。
        output = output / "previews" / (args.file.stem + "-" + file_sha256(args.file)[:8])

    # 有 --file 就走单文件预览路线；否则处理 data_dir 中的全部公共资料。
    if args.file:
        result = process_public_file(args.file, collection=args.collection, output_dir=output, pdf_method=args.pdf_method)
    else:
        result = build_public_records(Path(args.data_dir), output)

    # 输出每个集合处理后的记录数量，flush=True 让长任务的日志立即显示出来。
    print("处理完成：", {name: len(rows) for name, rows in result["collections"].items()}, flush=True)

    # 没有 append/rebuild 就在此结束；这是默认且安全的“只生成文件”模式。
    if not (args.append or args.rebuild):
        print(f"结果保存在：{output.resolve()}；数据库未修改。", flush=True)
        return 0

    # 入库前再次检查全部字段、主键、正文和长度，避免坏数据影响数据库。
    validate_public_input(result["collections"])
    # 读取模型地址、Milvus 地址、向量维度等配置。
    settings = get_settings()
    # 确保配置指定的 Milvus 数据库存在。
    ensure_milvus_database(settings)
    # 创建模型网关；后续入库函数通过它把正文转换成 BGE-M3 向量。
    model = ModelGateway(settings)
    # 根据命令参数选择“重建”或“追加”函数；这里只选择函数，还没有执行。
    write = rebuild_public_collections if args.rebuild else append_public_collections
    # 执行入库，并打印每个集合最终的记录数量。
    print(write(settings=settings, collections=result["collections"], model=model, embedding_dimension=settings.embedding_dim), flush=True)
    # 返回 0 告诉操作系统：本次离线任务成功结束。
    return 0


# 只有直接运行 python data_pipeline/main.py 时才调用 main；被别的文件导入时不会自动执行。
if __name__ == "__main__":
    # SystemExit 会把 main 返回的数字作为命令行退出码交给操作系统。
    raise SystemExit(main())
