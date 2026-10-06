import argparse
import asyncio
import sys
from pathlib import Path

# 获取当前脚本所在路径，也就是 legal-rag/scripts/import_sample_laws.py。
CURRENT_FILE = Path(__file__).resolve()
# 获取项目根目录，也就是 legal-rag 目录。
PROJECT_ROOT = CURRENT_FILE.parents[1]
# 把项目根目录加入 Python 模块搜索路径，保证能导入 backend、ingestion 等目录。
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.database import SessionLocal, create_tables
from backend.app.core.security import hash_password
from backend.app.models.entities import Document, IngestTask, KnowledgeBase, User
from ingestion.ingest_pipeline import IngestService


# 创建命令行参数解析器，让你可以控制每次导入多少个 PDF。
def parse_args() -> argparse.Namespace:
    # argparse 是 Python 自带的命令行参数工具。
    parser = argparse.ArgumentParser(description="分批导入 data/raw_laws 里的示例法律 PDF")
    # batch-size 表示本次最多处理几个还没有入库的 PDF。
    parser.add_argument("--batch-size", type=int, default=3, help="本次最多导入几个未入库 PDF，默认 3 个")
    # start 表示从排序后的第几个 PDF 开始看，适合你想跳过前面的文件。
    parser.add_argument("--start", type=int, default=0, help="从排序后的第几个 PDF 开始扫描，默认 0")
    # limit 表示最多扫描多少个 PDF，和 batch-size 不同：limit 是扫描范围，batch-size 是真正导入数量。
    parser.add_argument("--limit", type=int, default=0, help="最多扫描多少个 PDF，0 表示不限制")
    # reset-failed 表示如果之前某个文档失败了，可以重新创建任务再导入。
    parser.add_argument("--reset-failed", action="store_true", help="允许重新导入之前失败的文档")
    # 返回解析后的参数对象。
    return parser.parse_args()


# 获取或创建默认演示用户。
def get_or_create_admin_user(db: SessionLocal) -> User:
    # 先查找是否已经存在 admin 用户。
    user = db.query(User).filter(User.username == "admin").first()
    # 如果 admin 用户已存在，直接返回。
    if user:
        return user
    # 如果不存在，就创建默认演示用户。
    user = User(username="admin", password_hash=hash_password("123456"))
    # 把用户对象加入数据库会话。
    db.add(user)
    # 提交事务，让用户真正写入 MySQL。
    db.commit()
    # 刷新对象，拿到数据库生成的 id。
    db.refresh(user)
    # 返回新用户。
    return user


# 获取或创建默认公共法律示例知识库。
def get_or_create_sample_kb(db: SessionLocal, user: User) -> KnowledgeBase:
    # 查询 admin 用户是否已经有“公共法律示例库”。
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.user_id == user.id, KnowledgeBase.name == "公共法律示例库").first()
    # 如果知识库已存在，直接返回。
    if kb:
        return kb
    # 如果不存在，就创建一个示例知识库。
    kb = KnowledgeBase(user_id=user.id, name="公共法律示例库", description="从 data/raw_laws 分批导入的示例法律 PDF")
    # 把知识库对象加入数据库会话。
    db.add(kb)
    # 提交事务，让知识库写入 MySQL。
    db.commit()
    # 刷新对象，拿到知识库 id。
    db.refresh(kb)
    # 返回知识库。
    return kb


# 判断某个 PDF 是否已经成功入库。
def should_skip_pdf(db: SessionLocal, user_id: int, kb_id: int, pdf_name: str, reset_failed: bool) -> bool:
    # 查询当前文件是否已经作为文档记录存在。
    document = db.query(Document).filter(Document.user_id == user_id, Document.knowledge_base_id == kb_id, Document.filename == pdf_name).first()
    # 如果文档不存在，说明需要导入。
    if not document:
        return False
    # 只有已经完成并且至少有一个分块的文档才跳过。
    if document.status == "done" and document.chunk_count > 0:
        return True
    # 失败文档默认跳过，用户传入 --reset-failed 时才重试。
    if document.status == "failed" and not reset_failed:
        return True
    # done 但分块数量为 0、uploaded、pending、processing 都允许重新处理。
    return False


# 为一个 PDF 创建文档记录和导入任务。
def create_document_task(db: SessionLocal, user_id: int, kb_id: int, pdf_path: Path, reset_failed: bool) -> IngestTask:
    # 查询是否已经存在同名文档。
    document = db.query(Document).filter(Document.user_id == user_id, Document.knowledge_base_id == kb_id, Document.filename == pdf_path.name).first()
    # 如果不存在文档，就创建新的文档记录。
    if not document:
        document = Document(user_id=user_id, knowledge_base_id=kb_id, filename=pdf_path.name, file_path=str(pdf_path))
        # 加入数据库会话。
        db.add(document)
        # 提交事务。
        db.commit()
        # 刷新对象，拿到文档 id。
        db.refresh(document)
    # 如果文档之前失败，并且允许重试，就把状态改回 uploaded。
    elif reset_failed and document.status == "failed":
        document.status = "uploaded"
        # 清空旧错误信息。
        document.error_message = ""
        # 提交文档状态修改。
        db.commit()
    # 为这份文档创建一个新的入库任务。
    task = IngestTask(document_id=document.id, user_id=user_id, knowledge_base_id=kb_id)
    # 把任务加入数据库会话。
    db.add(task)
    # 提交事务。
    db.commit()
    # 刷新任务，拿到任务 id。
    db.refresh(task)
    # 返回任务对象。
    return task


# 打印某一次导入任务的最终结果。
def print_task_result(db: SessionLocal, task_id: int) -> bool:
    # 回滚当前会话可能打开的旧事务，让下一次查询读取数据库最新提交内容。
    db.rollback()
    # 清除 SQLAlchemy 当前会话中可能缓存的旧对象状态。
    db.expire_all()
    # 重新查询任务，拿到入库流水线写回的最新状态。
    task = db.get(IngestTask, task_id)
    # 如果任务不存在，说明数据库记录异常。
    if not task:
        print(f"❌ 任务不存在：task_id={task_id}")
        return False
    # 重新查询文档，拿到文档状态和分块数量。
    document = db.get(Document, task.document_id)
    # 如果文档不存在，说明任务关联异常。
    if not document:
        print(f"❌ 文档不存在：document_id={task.document_id}")
        return False
    # 任务和文档都完成，并且确实生成了分块，才算解析成功。
    if task.status == "done" and document.status == "done" and document.chunk_count > 0:
        print(f"✅ 解析成功：{document.filename}，分块数量：{document.chunk_count}，任务ID：{task.id}")
        return True
    # done 但没有分块时，给出明确原因。
    if task.status == "done" and document.status == "done" and document.chunk_count == 0:
        print(f"❌ 解析失败：{document.filename}，任务完成但没有生成有效分块，请检查 MinerU 或 Qwen-VL 配置")
        return False
    # 如果没有成功，就优先展示任务错误，再展示文档错误。
    error_message = task.error_message or document.error_message or "未返回具体错误"
    # 打印失败信息，方便你立刻知道是哪一个 PDF 出问题。
    print(f"❌ 解析失败：{document.filename}，状态：{task.status}，步骤：{task.current_step}，错误：{error_message}")
    return False


# 主流程：按批次导入 PDF。
async def main() -> None:
    # 读取命令行参数。
    args = parse_args()
    # 自动创建 MySQL 表。
    create_tables()
    # 创建数据库会话。
    db = SessionLocal()
    try:
        # 获取或创建默认用户。
        user = get_or_create_admin_user(db)
        # 获取或创建默认知识库。
        kb = get_or_create_sample_kb(db, user)
        # PDF 目录固定为 data/raw_laws。
        pdf_dir = Path("data/raw_laws")
        # 按文件名排序，保证每次分批顺序稳定。
        all_pdfs = sorted(pdf_dir.glob("*.pdf"), key=lambda path: path.name)
        # 根据 start 参数跳过前面的文件。
        selected_pdfs = all_pdfs[args.start :]
        # 如果设置了 limit，就只扫描限定数量。
        if args.limit > 0:
            selected_pdfs = selected_pdfs[: args.limit]
        # 记录本批真正导入了多少个文件。
        imported_count = 0
        # 记录本批解析成功了多少个文件。
        success_count = 0
        # 记录本批解析失败了多少个文件。
        failed_count = 0
        # 记录扫描过多少个文件。
        scanned_count = 0
        # 遍历本次扫描范围内的 PDF。
        for pdf_path in selected_pdfs:
            # 扫描数量加一。
            scanned_count += 1
            # 如果已经达到本批导入数量，就停止。
            if imported_count >= args.batch_size:
                break
            # 如果这个 PDF 已经导入过，就跳过。
            if should_skip_pdf(db, user.id, kb.id, pdf_path.name, args.reset_failed):
                print(f"跳过已存在：{pdf_path.name}")
                continue
            # 为当前 PDF 创建文档和任务。
            task = create_document_task(db, user.id, kb.id, pdf_path, args.reset_failed)
            # 打印当前导入进度。
            print(f"[{imported_count + 1}/{args.batch_size}] 开始导入：{pdf_path.name}")
            # 执行完整入库流水线。
            await IngestService().run(task.id)
            # 入库结束后重新查询数据库，并打印当前 PDF 是否解析成功。
            is_success = print_task_result(db, task.id)
            # 成功数量加一。
            if is_success:
                success_count += 1
            # 失败数量加一。
            else:
                failed_count += 1
            # 本批处理数量加一。注意：失败状态会记录在任务里，脚本继续跑下一个。
            imported_count += 1
        # 打印本批汇总信息。
        print(f"本批扫描 {scanned_count} 个 PDF，实际处理 {imported_count} 个 PDF，成功 {success_count} 个，失败 {failed_count} 个。")
        # 打印默认账号信息。
        print("默认账号：admin，默认密码：123456")
        # 提醒下一批怎么执行。
        print("下一批可继续执行：python scripts/import_sample_laws.py --batch-size 3")
    finally:
        # 无论成功失败，都关闭数据库连接。
        db.close()


# Python 直接运行这个文件时，从这里进入主流程。
if __name__ == "__main__":
    # asyncio.run 用来运行异步 main 函数。
    asyncio.run(main())
