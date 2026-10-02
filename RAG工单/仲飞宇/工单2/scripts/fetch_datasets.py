#!/usr/bin/env python
"""从开源数据集拉取语料，转换成可入库的 Markdown（律师 / 心理咨询师两个角色）。

数据来源
========
两个角色的语料都取自公开数据集，且都经 hf-mirror.com 镜像下载——本机 github.com 与
huggingface.co 实测均不通（超时），只有 hf-mirror 可达（约 10~60 KB/s，且大文件会中途断流）。

  律师：《中国法律语料》Duyu/Chinese_Law
        单文件 Markdown（20.8MB），按「# 法律名」分节收录了宪法、法律、法规与司法解释，
        章 / 节 / 法条结构完整。本脚本先按一级标题切成单部法律，再挑一批常用法进语料库。
        法律、法规原文依《著作权法》第五条不受著作权保护，故无授权风险。
  心理咨询师：CPsyCounD —— 中科院 CAS-SIAT《CPsyCoun》的中文多轮心理咨询对话集
        （3134 段，9 主题 × 7 流派，CC-BY-SA-4.0，论文 arXiv:2405.16433）。
        原仓 CAS-SIAT-XinHai/CPsyCoun 在 HF 上是 gated（无授权下不了），这里用的是社区
        转载 baicuya/CPsyCoun，其内容同为 CPsyCounD.json。
        + 《精神卫生法》全文，取自上面同一份法律语料——它直接对应心理咨询的业务边界
          （保密例外、非自愿住院等），补上纯对话语料缺的「知识型条文」。

为什么必须断点续传
==================
hf-mirror 实测会在大文件上下到一半断流（同一 URL 时快时慢），所以「重跑脚本能接着下」
不是保险而是主路径：先 HEAD 拿 content-length，本地已下够的直接跳过；没下够的按 256KB
分段 Range 续传，每段带重试。整批约 29MB，冷启动几分钟到十几分钟。

产物
====
  data/raw/                   原始下载文件（保留，便于复核；已在 .gitignore 里）
  data/raw/laws_split/        上面那份大 Markdown 按法律名切成的单部法律（中间产物）
  data/corpus/<role>/*.md     转换后的语料，scripts/seed.py 直接入库这个目录
  data/corpus/manifest.json   每个语料的来源 / 许可 / 字节数 / sha256

用法
====
  python scripts/fetch_datasets.py            # 下载并转换（已就绪且校验通过的跳过）
  python scripts/fetch_datasets.py --list     # 只打印清单与状态，不联网、不写盘
  python scripts/fetch_datasets.py --force    # 忽略已有产物，重新下载转换
  python scripts/fetch_datasets.py --only psychologist   # 只处理某一个角色
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.core.logging_config import get_logger, setup_logging  # noqa: E402

log = get_logger("fetch")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"
SPLIT_DIR = RAW_DIR / "laws_split"

# 镜像地址：写成常量而不是读环境变量，是因为这个值在本机是实测结论（见模块开头），
# 不是部署时可调的旋钮；真要换源，改这一行比找 .env 更快。
MIRROR = "https://hf-mirror.com"
TIMEOUT = 30
RETRIES = 4
CHUNK = 256 * 1024  # 单次 Range 大小。太大（>=32MB）容易在断流时白下，太小则请求次数暴涨

# 法律语料：单文件 Markdown，按一级标题切成单部法律
LAW_CORPUS = {
    "repo": "Duyu/Chinese_Law",
    "file": "Chinese_law.md",
    "license": "未声明（法律、法规原文依《著作权法》第五条不受著作权保护）",
}

# 心理咨询对话：单 JSON，字段 [{instruction, input, output, history}]
#
# 首选原仓（权威、可引论文），社区转载作兜底：原仓的 CPsyCounR（咨询案例报告）是 gated，
# 但 CPsyCounD 这份是可下的（实测 8,364,968 字节，与转载版逐字节一致），曾一度以为整个仓库
# 都下不了而绕道转载版，这里两个源都留着——任一源抽风时不必改代码。
CPsyCoun = {
    "repos": ["CAS-SIAT-XinHai/CPsyCoun", "baicuya/CPsyCoun"],
    "file": "CPsyCounD.json",
    "license": "CC-BY-SA-4.0",
    "note": "CPsyCounD，中科院 CAS-SIAT，arXiv:2405.16433（CPsyCounR 为 gated，D 版可下）",
}

# 民法典在这份语料里是按编拆开的 8 个一级标题，入库前合并成一篇——documents 表里出现
# 8 条同属一部法典的记录，既不好解释也会让「刑法/民法典」这种按法典名的检索变散。
# 顺序按法典原序，不从输入文件里推断。
CIVIL_CODE_PARTS = [
    "中华人民共和国民法典-总则",
    "中华人民共和国民法典-物权编",
    "中华人民共和国民法典-合同编",
    "中华人民共和国民法典-人格权编",
    "中华人民共和国民法典-婚姻家庭编",
    "中华人民共和国民法典-继承编",
    "中华人民共和国民法典-侵权责任编",
    "中华人民共和国民法典-附则",
]

# 进语料库的法律清单：法律全名 -> 语料文件名。
# 按「普法咨询里最常被问到的」挑，不是把 793 部全灌进去——全量约 20MB、几万个 chunk，
# 一个演示用的知识库没必要这么大，而且检索质量会被稀释。
# 顺序即入库顺序，写在一起是为了让「这个角色的知识库覆盖了哪些法」一眼可见。
LAWYER_LAWS: dict[str, str] = {
    "中华人民共和国民法典": "civil_code.md",  # 由 CIVIL_CODE_PARTS 合并而来
    "中华人民共和国刑法": "criminal_law.md",
    "中华人民共和国宪法": "constitution.md",
    "中华人民共和国劳动法": "labor_law.md",
    "中华人民共和国劳动合同法": "labor_contract_law.md",
    "中华人民共和国劳动争议调解仲裁法": "labor_dispute_arbitration_law.md",
    "中华人民共和国消费者权益保护法": "consumer_rights_law.md",
    "中华人民共和国道路交通安全法": "road_traffic_safety_law.md",
    "中华人民共和国治安管理处罚法": "public_security_administration_law.md",
    "中华人民共和国行政处罚法": "administrative_penalty_law.md",
    "中华人民共和国个人信息保护法": "personal_information_protection_law.md",
    "中华人民共和国律师法": "lawyers_law.md",
    "中华人民共和国法律援助法": "legal_aid_law.md",
    "中华人民共和国民事诉讼法": "civil_procedure_law.md",
}

# 心理咨询师侧的知识型语料：从同一份法律语料里取，故单独列一张表
PSYCHOLOGIST_LAWS: dict[str, str] = {
    "中华人民共和国精神卫生法": "mental_health_law.md",
}

# 对话语料切成几个文件：一个文件一个 source，便于按篇目定位与重灌。
# 切 8 份（每份约 390 段）是折中——单文件太大时一篇文档的 chunk 会挤占检索池，
# 太碎又会让 documents 表里全是同质条目。
DIALOGUE_FILES = 8

# 单部法律少于这个字符数就认为文件有问题（切分错了或原文残缺），不入库。
# 定这个阈值是因为实测踩过：Dusker/chinese-laws-pretrain 的「婚姻家庭编」只有 4128 字符、
# 从第五章收养开始，前四章整段缺失——残文件进库比缺文件更糟，它会以「有资料」的姿态
# 让模型给出片面结论。这里的法律语料按一级标题切，理论上不会残缺，阈值是兜底。
MIN_LAW_CHARS = 2000


def _head_size(url: str) -> int:
    """HEAD 拿 content-length；拿不到（服务端不给）返回 -1，调用方按「不知道」处理。"""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "curl/8"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return int(resp.headers.get("Content-Length") or -1)
    except Exception as exc:  # noqa: BLE001 —— HEAD 失败不该让整个流程挂掉，GET 还能试
        log.warning("HEAD 失败（%s），改用 GET 探测: %s", url, exc)
        return -1


def _download(url: str, dest: Path, *, force: bool = False) -> Path:
    """下载到 dest，支持断点续传。已下满则直接返回。

    hf-mirror 会在中途断流，所以「续传」是主路径：每次从本地已有字节数往后 Range 拉
    一块，写完 flush+fsync 再拉下一块——中途断了也不丢已下部分，重跑接着来。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = _head_size(url)
    have = dest.stat().st_size if dest.exists() and not force else 0

    if force and dest.exists():
        dest.unlink()
        have = 0
    if total > 0 and have == total:
        log.info("已下载完成，跳过：%s（%d 字节）", dest.name, total)
        return dest
    if total > 0 and have > total:  # 上一次断在重定向/错误页上，落了个比正文还大的文件
        log.warning("%s 本地比远端还大（%d > %d），重新下载", dest.name, have, total)
        dest.unlink()
        have = 0

    log.info("开始下载 %s（已有 %d / 共 %s 字节）", dest.name, have, total if total > 0 else "?")
    started = time.time()
    with open(dest, "ab") as fh:
        while total <= 0 or have < total:
            end = have + CHUNK - 1
            headers = {"User-Agent": "curl/8", "Range": f"bytes={have}-{end}"}
            ok = False
            for attempt in range(1, RETRIES + 1):
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                        block = resp.read()
                    if not block:
                        # 服务端无视 Range 返回了空体：说明已经到底，结束
                        log.info("远端返回空块，下载结束")
                        ok = True
                        break
                    if have > 0 and len(block) > total - have:
                        block = block[: total - have]  # 服务端忽略 Range 返回了整份，截掉已下的
                    fh.write(block)
                    fh.flush()
                    have += len(block)
                    ok = True
                    break
                except Exception as exc:  # noqa: BLE001 —— 断流是常态，重试就是正常路径
                    wait = 2**attempt
                    log.warning("第 %d/%d 次失败（%s），%ds 后重试", attempt, RETRIES, exc, wait)
                    time.sleep(wait)
            if not ok:
                raise RuntimeError(f"下载失败：{url}（已下 {have} 字节，可重跑续传）")
            if total > 0 and have >= total:
                break
            if total <= 0 and len(block) < CHUNK:
                break  # 服务端不支持 HEAD 时，靠「块不满」判断到底
    took = time.time() - started
    log.info("下载完成 %s：%d 字节，用时 %.0fs（%.1f KB/s）", dest.name, have, took, have / 1024 / max(took, 1e-6))
    return dest


def _download_any(urls: list[str], dest: Path, *, force: bool = False) -> Path:
    """按顺序试多个源，第一个成功的为准。

    换源时**先删掉半截文件**：不同源的同名文件不保证字节一致，拿 A 源的半截去续 B 源，
    会拼出一个两边都对不上的文件——那比重新下更糟，因为大小可能恰好「看起来对」。
    """
    last: Exception | None = None
    for i, url in enumerate(urls, 1):
        try:
            return _download(url, dest, force=force)
        except Exception as exc:  # noqa: BLE001 —— 换源是设计内的降级路径
            last = exc
            log.warning("源 %d/%d 下载失败（%s），换下一个源", i, len(urls), exc)
            if dest.exists():
                dest.unlink()
    raise RuntimeError(f"{len(urls)} 个源全部下载失败，最后一个错误：{last}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def split_laws(md_path: Path) -> dict[str, str]:
    """把「一部法一个大 Markdown」切成单部法律，返回 {法律全名: 正文}（同时落盘到 SPLIT_DIR）。

    按一级标题（`# `）切——那份语料里 `#` 只用于法律名，`##`/`###` 是章/节/条，
    所以一级标题就是法律边界。标题去掉首尾空白，保留《》与年份等原文写法。

    这里**不做长度过滤**：过滤是「挑哪几部进语料库」时的事（见 `_pick`）。切分阶段丢掉短文件
    会误伤《民法典-附则》这类天生就短、但属于完整法典一部分的篇目。
    """
    bodies: dict[str, str] = {}
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    text = md_path.read_text(encoding="utf-8", errors="replace")

    # 用 re.split 的捕获组切：奇数位是标题，偶数位是正文
    parts = re.split(r"^# (.+)$", text, flags=re.M)
    # parts[0] 是第一个标题之前的页首内容（若有），一般只有版权/说明，丢弃
    for i in range(1, len(parts) - 1, 2):
        name = parts[i].strip()
        if not name:
            continue
        # 同名法律会重复出现（如各年修正案并列），后来的覆盖前面的：以文件里靠后的版本为准
        bodies[name] = f"# {name}\n{parts[i + 1]}"

    for name, content in bodies.items():
        # 直接拿法律名当文件名在 Windows 上有全角括号/引号的风险，先换掉非法字符
        safe = re.sub(r"[\\/:*?\"<>|]", "_", name)
        dest = SPLIT_DIR / f"{safe}.md"
        if not dest.exists() or dest.read_text(encoding="utf-8") != content:
            dest.write_text(content, encoding="utf-8")
    log.info("法律语料切分完成：%d 部（按一级标题切）", len(bodies))
    return bodies


def _pick(laws: dict[str, str], name: str, *, min_chars: int = MIN_LAW_CHARS) -> str | None:
    """从切好的法律里取一部；不存在或短得可疑就返回 None（由调用方汇总告警）。

    长度阈值是兜底：正常单部法律都远超它，真踩到只可能是切分错位或原文残缺。
    残文件进库比缺文件更糟——它会以「有资料」的姿态让模型给出片面结论。
    """
    content = laws.get(name)
    if content is None:
        return None
    if len(content) < min_chars:
        log.warning("%s 只有 %d 字符（阈值 %d），疑似残缺，不入库", name, len(content), min_chars)
        return None
    return content


def _write(dest: Path, content: str, *, force: bool) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if force or not dest.exists() or dest.read_text(encoding="utf-8") != content:
        dest.write_text(content, encoding="utf-8")
    return dest


def build_civil_code(laws: dict[str, str], out_dir: Path, *, force: bool = False) -> Path | None:
    """把按编拆开的民法典 8 篇合成一篇 civil_code.md。

    合并后每编前面补一个 `## 编名` 标题，让「编」在检索到的 chunk 里仍然可见——
    否则合成一篇之后，模型看到某条法条时不知道它出自哪一编。
    """
    sections: list[str] = []
    for part in CIVIL_CODE_PARTS:
        content = laws.get(part)  # 附则天生短，这里不套长度阈值
        if content is None:
            log.warning("民法典缺 %s，合并后的法典不完整", part)
            continue
        # 去掉原文件的一级标题（它是「民法典-总则」这种带后缀的名字），换成干净的编名
        body = re.sub(r"^# .+\n", "", content, count=1).strip()
        # 正文里的「## 第一章」是章，得比「## 编」低一级，整体下沉一档，
        # 否则合并后编与章同级，读起来像并列关系
        body = re.sub(r"^(#{2,4}) ", r"#\1 ", body, flags=re.M)
        sections.append(f"## {part.replace('中华人民共和国民法典-', '')}\n\n{body}")
    if not sections:
        log.warning("民法典各编一编都没找到，跳过")
        return None
    merged = "# 中华人民共和国民法典\n\n" + "\n\n".join(sections) + "\n"
    log.info("民法典合并完成：%d 编，%d 字符", len(sections), len(merged))
    return _write(out_dir / "civil_code.md", merged, force=force)


def build_lawyer(laws: dict[str, str], *, force: bool = False) -> list[Path]:
    """挑出常用法进律师语料库，返回写出的文件列表。"""
    out_dir = CORPUS_DIR / "lawyer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []

    civil = build_civil_code(laws, out_dir, force=force)
    if civil:
        out.append(civil)

    missing: list[str] = []
    for name, slug in LAWYER_LAWS.items():
        if slug == "civil_code.md":
            continue  # 上面已由 build_civil_code 生成
        content = _pick(laws, name)
        if content is None:
            missing.append(name)
            continue
        out.append(_write(out_dir / slug, content, force=force))
        log.info("律师语料 <- %s（%d 字符）", name, len(content))
    if missing:
        # 数据集里没有（或残缺）的法要显式报出来，不能静默少几个——「知识库覆盖哪几部法」
        # 是演示时要能说清的事
        log.warning("以下 %d 部法没有取到，已跳过：%s", len(missing), "、".join(missing))
    return out


def _dialogue_to_md(record: dict) -> str:
    """一条咨询记录 -> Markdown 片段（多轮对话按顺序还原）。"""
    lines: list[str] = []
    for turn in record.get("history") or []:
        if not isinstance(turn, (list, tuple)) or len(turn) < 2:
            continue
        lines.append(f"**来访者**：{turn[0]}")
        lines.append(f"**咨询师**：{turn[1]}")
        lines.append("")
    # 最后一轮单独放在 instruction/input/output 里（用 .get 兜底：字段缺失时不要崩）
    last_in = (record.get("instruction") or "") + (record.get("input") or "")
    if last_in.strip():
        lines.append(f"**来访者**：{last_in.strip()}")
    if (record.get("output") or "").strip():
        lines.append(f"**咨询师**：{record['output'].strip()}")
    return "\n".join(lines).strip()


def build_psychologist(laws: dict[str, str], cpsycoun_path: Path, *, force: bool = False) -> list[Path]:
    """心理咨询师语料 = CPsyCounD 对话（切若干文件）+ 精神卫生法。"""
    out_dir = CORPUS_DIR / "psychologist"
    out_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []

    records = json.loads(cpsycoun_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise RuntimeError(f"CPsyCounD 结构不是列表：{type(records)}")
    log.info("CPsyCounD 载入 %d 段咨询对话", len(records))

    # 对话按序等分进 N 个文件。语料本身没有主题字段（只有 instruction/input/output/history），
    # 所以按下标切而不是按主题切——不要在注释外假装它们分了类。
    per = (len(records) + DIALOGUE_FILES - 1) // DIALOGUE_FILES
    for idx in range(DIALOGUE_FILES):
        batch = records[idx * per : (idx + 1) * per]
        if not batch:
            continue
        chunks = [
            f"## 咨询对话 {idx * per + i + 1}\n\n{_dialogue_to_md(rec)}"
            for i, rec in enumerate(batch)
            if _dialogue_to_md(rec)
        ]
        title = f"心理咨询对话集（第 {idx + 1}/{DIALOGUE_FILES} 辑，CPsyCounD）"
        body = f"# {title}\n\n" + "\n\n".join(chunks) + "\n"
        dest = _write(out_dir / f"cpsyccoun_dialogues_{idx + 1:02d}.md", body, force=force)
        out.append(dest)
        log.info("心理语料 <- CPsyCounD 第 %d 辑：%d 段对话，%d 字符", idx + 1, len(chunks), len(body))

    for name, slug in PSYCHOLOGIST_LAWS.items():
        content = _pick(laws, name)
        if content is None:
            log.warning("法律语料里没有 %s，跳过（心理侧就只剩对话语料了）", name)
            continue
        out.append(_write(out_dir / slug, content, force=force))
        log.info("心理语料 <- %s（%d 字符）", name, len(content))
    return out


def write_manifest(files_by_role: dict[str, list[Path]]) -> Path:
    """写来源/许可/大小/校验和清单。

    这个文件是语料**可追溯**的凭据：报告里要写清数据从哪来、什么许可、多少字节，
    而 data/corpus 本身在 .gitignore 里（生成物、体积大），留一份 manifest 进仓库
    才能让别人不必下载就知道知识库里是什么。
    """
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": {
            "lawyer": LAW_CORPUS,
            "psychologist_dialogues": CPsyCoun,
            "psychologist_law": LAW_CORPUS,
        },
        "roles": {},
    }
    for role, files in files_by_role.items():
        manifest["roles"][role] = [
            {
                "path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "bytes": p.stat().st_size,
                "sha256": _sha256(p)[:16],
            }
            for p in sorted(files)
        ]
    dest = CORPUS_DIR / "manifest.json"
    dest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def main() -> None:
    setup_logging(settings)
    ap = argparse.ArgumentParser(description="下载开源数据集并转换成入库语料")
    ap.add_argument("--force", action="store_true", help="忽略已有产物，重新下载并转换")
    ap.add_argument("--list", action="store_true", help="只打印清单与状态，不联网、不写盘")
    ap.add_argument("--only", choices=["lawyer", "psychologist"], help="只处理指定角色")
    args = ap.parse_args()

    if args.list:
        print("进语料库的法律：")
        for name, slug in LAWYER_LAWS.items():
            p = CORPUS_DIR / "lawyer" / slug
            print(f"  [{'√' if p.exists() else '×'}] lawyer/{slug:<40} {name}")
        for name, slug in PSYCHOLOGIST_LAWS.items():
            p = CORPUS_DIR / "psychologist" / slug
            print(f"  [{'√' if p.exists() else '×'}] psychologist/{slug:<32} {name}")
        p = CORPUS_DIR / "psychologist" / "cpsyccoun_dialogues_01.md"
        print(f"  [{'√' if p.exists() else '×'}] psychologist/cpsyccoun_dialogues_*.md  "
              f"CPsyCounD 对话（共 {DIALOGUE_FILES} 辑）")
        return

    # 两个角色都要用法律语料，所以它永远要先下
    law_md = _download(
        f"{MIRROR}/datasets/{LAW_CORPUS['repo']}/resolve/main/{LAW_CORPUS['file']}",
        RAW_DIR / LAW_CORPUS["file"],
        force=args.force,
    )
    laws = split_laws(law_md)

    files_by_role: dict[str, list[Path]] = {}
    if args.only in (None, "lawyer"):
        files_by_role["lawyer"] = build_lawyer(laws, force=args.force)

    if args.only in (None, "psychologist"):
        cpsycoun = _download_any(
            [f"{MIRROR}/datasets/{repo}/resolve/main/{CPsyCoun['file']}" for repo in CPsyCoun["repos"]],
            RAW_DIR / CPsyCoun["file"],
            force=args.force,
        )
        files_by_role["psychologist"] = build_psychologist(laws, cpsycoun, force=args.force)

    dest = write_manifest(files_by_role)
    print(f"\n语料清单：{dest}")
    for role, files in files_by_role.items():
        total = sum(p.stat().st_size for p in files)
        print(f"  {role:<14} {len(files):>2} 个文件，{total / 1024 / 1024:.2f} MB")
    print("\n下一步：python scripts/seed.py --reset  （跑之前先停掉 web 服务，Milvus Lite 独占）")


if __name__ == "__main__":
    main()
