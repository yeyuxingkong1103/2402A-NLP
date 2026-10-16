# -*- coding: utf-8 -*-
"""工单6 演示视频自动生成 —— 用**真实系统截图**拼装带中文讲解的 mp4。

工单编号：人工智能NLP-RAG-混合检索任务

为什么要自动生成而不是只写录屏脚本
------------------------------------
工单把「演示视频」列为交付物。人工录屏当然更好（`部署/脚本/录屏步骤.md` 就是给人照着录的），
但人工录制依赖录制软件、麦克风与讲解者，**不保证交付时一定存在**。
本脚本给出一条**确定性兜底**：把系统**真跑出来的截图**按镜头顺序拼成 mp4，
每个镜头配「标题 + 可照读旁白」，并烧进画面 —— 于是「演示视频」这件交付物**必然存在且可复现**，
且画面内容 100% 来自真实运行结果（不是示意图、不是手绘）。

诚实边界（必须知道）
--------------------
* 本脚本产出的是**图文讲解片**（截图 + 旁白字幕），**不是**屏幕操作录像，
  **没有配音音轨**（本机未安装可用的 TTS 引擎，实测 `ffmpeg` 与 Python 侧均无 TTS）。
* 它**不能替代** `录屏步骤.md` 描述的人工录制：人工录制才有鼠标操作、页面滚动、实时响应。
* 两件交付物定位不同：本片保证「有」且「内容真实」；人工录制保证「动」且「有讲解声」。

依赖
----
* Pillow（已装）—— 合成「截图 + 字幕」画面；
* ffmpeg —— 本机 PATH 里没有，脚本会自动探测已知位置
  （`C:\\Program Files\\EVCapture\\ffmpeg.exe` 等，实测可用，含 libx264）；
* 中文字体 `C:\\Windows\\Fonts\\msyh.ttc` / `msyhbd.ttc`（实测存在）。

用法
----
.. code-block:: powershell

    pwsh -NoProfile -File run_py.ps1 部署/脚本/生成演示视频.py
    # 输出：部署/演示视频/工单6_混合检索演示.mp4 与 画面帧/ 下的分镜图
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

WORK_ORDER = "人工智能NLP-RAG-混合检索任务"

WIDTH, HEIGHT = 1600, 900
CAPTION_H = 118                      # 底部字幕条高度
CONTENT_H = HEIGHT - CAPTION_H       # 内容区高度
MARGIN = 18
BG = (244, 246, 251)
BAR_BG = (26, 29, 36)
TITLE_COLOR = (255, 255, 255)
NARRATION_COLOR = (206, 214, 228)
ACCENT = (88, 166, 255)
CARD_BG = (17, 24, 39)

SHOT_DIR = _REPO / "测试" / "测试截图"
OUT_DIR = _REPO / "部署" / "演示视频"
FRAME_DIR = OUT_DIR / "画面帧"

FONT_BOLD_CANDIDATES = [
    r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\Deng.ttf",
]
FONT_REG_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\Deng.ttf", r"C:\Windows\Fonts\simhei.ttf",
]
FFMPEG_CANDIDATES = [
    r"C:\Program Files\EVCapture\ffmpeg.exe",
    r"C:\ProgramData\GamePPPublic\PCBenchmark\ffmpeg.exe",
    r"C:\ffmpeg\bin\ffmpeg.exe",
]


@dataclass(slots=True)
class Scene:
    """一个镜头：一张真实截图 + 标题 + 可照读旁白 + 时长。"""

    title: str
    narration: str
    image: str = ""                   # 相对 测试/测试截图/ 的文件名；空 = 纯文字卡
    seconds: float = 5.5
    kind: str = "shot"                # shot | title | end
    bullets: list[str] = field(default_factory=list)


#: 分镜表（与 部署/脚本/录屏步骤.md 的镜头一一对应；画面全部来自实跑截图）
SCENES: tuple[Scene, ...] = (
    Scene(kind="title", title="工单6 · 基于 RAG 的混合检索优化系统",
          narration="人工智能NLP-RAG-混合检索任务 —— 向量检索 / 全文检索 / 混合检索 三模式可配置",
          bullets=["数据：武汉兴图新科（科创板）＋ 武汉力源信息（创业板）两份招股说明书",
                   "索引：2676 个块，bge-m3(1024) 与 m3e-base(768) 两套向量索引 + 自实现 BM25 倒排",
                   "全部画面与数字均来自真实运行结果"], seconds=7.0),

    Scene(title="① 检索策略面板：三模式可切换（混合检索）",
          narration="界面右侧是工单6 的核心：检索模式下拉框、嵌入模型、融合算法、"
                    "向量/全文权重滑块、重排模型、Top-K、召回 K、分数阈值、全文检索字段与布尔运算符。",
          image="混合检索_发行股数.png",
          bullets=["混合检索｜融合=weighted_average｜向量 0.5 / 全文 0.5",
                   "答案：发行股数 1,670 万股，占发行后总股本 25.04%（引用第 24 页）"], seconds=7.0),

    Scene(title="② 向量检索（嵌入召回 + 重排）",
          narration="技术：本地 Ollama 的 bge-m3 模型把问句编成 1024 维向量，"
                    "在 FAISS 索引上做余弦相似度召回 Top-20，再用规则特征重排取 Top-5。",
          image="向量检索_发行股数.png",
          bullets=["bge-m3（1024 维，多语）→ FAISS 余弦召回 K=20 → 规则重排取 Top-5",
                   "该题向量路召回 20 条，检索耗时 108.9 ms"], seconds=6.5),

    Scene(title="③ 全文检索（倒排索引 + 布尔/短语/模糊）",
          narration="技术：jieba 分词的倒排索引，BM25 打分。支持布尔查询（AND/OR/NOT）、"
                    "短语匹配（引号）、模糊匹配（词~）、以及标题/正文/关键词/摘要多字段限定。",
          image="全文检索_发行股数.png",
          bullets=["字段可选：正文×1 / 标题×1.35 / 关键词×1.2 / 摘要×0.9；默认运算符 OR",
                   "该题全文路检索耗时仅 11.8 ms，比向量路快一个数量级"], seconds=6.5),

    Scene(title="④ 混合检索 + 权重调整",
          narration="两路并行检索后按融合算法合并。这里把向量权重从 0.5 调到 0.8、全文权重降到 0.2，"
                    "界面会实时回显生效权重。",
          image="混合检索_权重调整.png",
          bullets=["权重滑块：向量 0.80 / 全文 0.20（两者联动，和恒为 1）",
                   "融合算法另有 RRF 倒数排名融合、投票机制（Borda 计票）可选"], seconds=6.5),

    Scene(title="⑤ 三模式对比：同一问题跑三种策略",
          narration="点「三模式对比」，同一问题依次用向量、全文、混合检索各跑一次并列表对照。"
                    "看这道题：纯向量检索答错了，全文与混合检索答对了。",
          image="三模式对比_同一问题.png",
          bullets=["vector：3,507.52 / 1,343.32 / 5,500.69 / 404.20 万元 —— 错（引第 125 页表格）",
                   "fulltext 与 hybrid：6,464.51 / 14,414.16 / 18,780.67 / 4,627.14 万元 —— 对（引第 129 页）",
                   "这就是「混合检索提升检索精度」的直接证据"], seconds=9.0),

    Scene(title="⑥ 准确率与召回率汇总（10 题 × 3 策略 = 30 次）",
          narration="汇总结果：混合检索与全文检索准确率 100%，纯向量检索 70%；"
                    "混合检索比纯向量高 30 个百分点，修正了第 33、34、260 题，且零退化。",
          image="准确率汇总.png",
          bullets=["混合 vs 向量：准确率 +30.00 个百分点，召回率 +30.00 个百分点",
                   "召回率权威层 90% —— 剔除 1 道「证据结构不可达」题后为 100%（已穷举 2676 块证明）"], seconds=9.5),

    Scene(title="⑦ 响应时间测试（目标 ≤3 秒）",
          narration="端到端 P50 909 毫秒、P95 2392 毫秒，30 次运行中超过 3000 毫秒的有 0 次。"
                    "检索层本身只要 36 毫秒中位数。",
          image="响应时间测试.png",
          bullets=["端到端 P50 / P95 / max = 908.72 / 2392.43 / 2418.87 ms，超线 0 轮",
                   "检索层 P50 / P95 = 36.15 / 65.67 ms（全文路 P95 仅 13.74 ms）"], seconds=8.5),

    Scene(title="⑧ 多轮对话回归：指代消解仍然生效",
          narration="工单5 的多轮能力未退化。「他」和「这个公司」被正确消解为武汉兴图新科，"
                    "并继承上一轮的问题槽位。",
          image="多轮对话_指代消解.png",
          bullets=["「他参与的哪个工程…」→ 武汉兴图新科电子股份有限公司参与的哪个工程…（策略 pronoun）",
                   "「这个公司的法定代表人是谁？」→ 同上消解 —— 答案：程家明"], seconds=7.5),

    Scene(title="⑨ 多轮对话回归：三种检索策略逐轮对照",
          narration="同一段五轮序列在三种检索策略下各跑一遍：改写策略三种策略都是 5/5，"
                    "答案向量路 4/5、全文与混合 5/5。",
          image="多轮对话回归_三策略.png",
          bullets=["改写策略：none / pronoun / pronoun / ellipsis_entity_switch / none —— 全对",
                   "答案正确：vector 4/5、fulltext 5/5、hybrid 5/5"], seconds=8.0),

    Scene(title="⑩ 异常输入测试（14 用例 × 3 策略 = 42 次）",
          narration="空输入、纯标点、乱码、emoji、600 字超长文本、SQL 注入、脚本注入、"
                    "提示词注入等全部不崩溃且有非空回应，通过率 100%。",
          image="异常输入测试.png",
          bullets=["通过 42/42 = 100%；判据：不崩溃 + 有非空回应（拒答也算通过）",
                   "服务端对非法检索策略参数返回 400 并给出中文原因，不静默降级"], seconds=8.0),

    Scene(kind="end", title="交付小结",
          narration="工单6 完成：三模式可配置、多嵌入模型、四种策略切换途径、10 张测试截图与可复现评测。",
          bullets=["代码：研发/app/core/{retrieval_strategy, fusion, fulltext_query, embedding_models}.py",
                   "评测：测试/混合检索/ 三个脚本；单元测试 255 passed",
                   "文档：设计 4 份、部署说明、演示分镜、README 与测试报告",
                   "本片由 部署/脚本/生成演示视频.py 自动生成，画面全部来自真实运行截图"], seconds=8.0),
)


# ---------------------------------------------------------------------------
# 字体与绘图工具
# ---------------------------------------------------------------------------
def _pick_font(candidates: Sequence[str]) -> str:
    for path in candidates:
        if Path(path).is_file():
            return path
    raise SystemExit("[FATAL] 找不到可用的中文字体（尝试过：" + "、".join(candidates) + "）")


def _font(path: str, size: int) -> Any:
    try:
        return ImageFont.truetype(path, size)
    except OSError:                                    # pragma: no cover
        return ImageFont.load_default()


def _wrap(draw: Any, text: str, font: Any, max_width: int) -> list[str]:
    """按像素宽度折行（中英文混排均按字符逐个累加，够用且不依赖分词）。"""
    lines: list[str] = []
    current = ""
    for ch in str(text):
        trial = current + ch
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = ch
    if current:
        lines.append(current)
    return lines


def _fit_contain(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    """等比缩放到「完整放进 box」并居中贴到白底上（适用于宽扁的界面截图）。"""
    scale = min(box_w / img.width, box_h / img.height)
    size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    canvas = Image.new("RGB", (box_w, box_h), (255, 255, 255))
    resized = img.resize(size, Image.LANCZOS)
    canvas.paste(resized, ((box_w - size[0]) // 2, (box_h - size[1]) // 2))
    return canvas


def _window_pages(img: Image.Image, box_w: int, box_h: int) -> list[Image.Image]:
    """把**很高的**报告页按 box_h 切成一页页（每页都铺满宽度），便于在视频里读清表格。"""
    scale = box_w / img.width
    resized = img.resize((box_w, max(1, int(img.height * scale))), Image.LANCZOS)
    pages: list[Image.Image] = []
    top = 0
    while top < resized.height:
        page = Image.new("RGB", (box_w, box_h), (255, 255, 255))
        piece = resized.crop((0, top, box_w, min(top + box_h, resized.height)))
        page.paste(piece, (0, 0))
        pages.append(page)
        top += box_h
    return pages or [Image.new("RGB", (box_w, box_h), (255, 255, 255))]


def _draw_caption(draw: Any, scene: Scene, *, page: int = 0, pages: int = 1,
                  fonts: dict[str, Any]) -> None:
    """底部字幕条：标题 + 旁白（自动折行）+ 可选的右侧页码。"""
    top = CONTENT_H
    draw.rectangle([0, top, WIDTH, HEIGHT], fill=BAR_BG)
    draw.rectangle([0, top, WIDTH, top + 4], fill=ACCENT)
    draw.text((MARGIN + 6, top + 12), scene.title, font=fonts["title"], fill=TITLE_COLOR)
    lines = _wrap(draw, scene.narration, fonts["narration"], WIDTH - 2 * MARGIN - 150)
    for i, line in enumerate(lines[:2]):
        draw.text((MARGIN + 6, top + 48 + i * 27), line, font=fonts["narration"],
                  fill=NARRATION_COLOR)
    if pages > 1:
        draw.text((WIDTH - MARGIN - 120, top + 12), f"第 {page + 1}/{pages} 页",
                  font=fonts["small"], fill=ACCENT)


def _bullet_lines(draw: Any, items: Sequence[str], font: Any, max_width: int) -> list[str]:
    """把要点逐条折行（首行加「·」前缀，续行缩进对齐）。"""
    out: list[str] = []
    for item in items:
        wrapped = _wrap(draw, "· " + str(item), font, max_width)
        for i, line in enumerate(wrapped):
            out.append(line if i == 0 else "  " + line)
    return out


def render_scene_frames(scene: Scene, index: int, *, fonts: dict[str, Any]) -> list[Path]:
    """把一个镜头渲染成 1..N 张 1600×900 的帧图，返回文件路径列表。"""
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    pages: list[Image.Image] = []

    if scene.kind in ("title", "end"):
        canvas = Image.new("RGB", (WIDTH, HEIGHT), CARD_BG)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, WIDTH, 8], fill=ACCENT)
        draw.rectangle([0, HEIGHT - 8, WIDTH, HEIGHT], fill=ACCENT)
        title_lines = _wrap(draw, scene.title, fonts["card"], WIDTH - 200)
        y = 210
        for line in title_lines:
            draw.text(((WIDTH - draw.textlength(line, font=fonts["card"])) / 2, y),
                      line, font=fonts["card"], fill=TITLE_COLOR)
            y += 78
        y += 26
        for line in _wrap(draw, scene.narration, fonts["card_sub"], WIDTH - 240):
            draw.text(((WIDTH - draw.textlength(line, font=fonts["card_sub"])) / 2, y),
                      line, font=fonts["card_sub"], fill=ACCENT)
            y += 42
        y += 40
        for line in _bullet_lines(draw, scene.bullets, fonts["bullet"], WIDTH - 320):
            draw.text((160, y), line, font=fonts["bullet"], fill=NARRATION_COLOR)
            y += 38
        pages = [canvas]
    else:
        src = SHOT_DIR / scene.image
        if not src.is_file():
            raise SystemExit(f"[FATAL] 镜头 {index} 的截图不存在：{src}")
        with Image.open(src) as raw:
            img = raw.convert("RGB")
            box_w, box_h = WIDTH - 2 * MARGIN, CONTENT_H - 2 * MARGIN
            # 宽扁的界面截图 ⇒ 整幅放进内容区；很高的报告页 ⇒ 切成多页读清表格
            if img.height / img.width <= 0.75:
                pages = [_fit_contain(img, box_w, box_h)]
            else:
                pages = _window_pages(img, box_w, box_h)
        composed: list[Image.Image] = []
        for page_img in pages:
            canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
            canvas.paste(page_img, (MARGIN, MARGIN))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle([MARGIN - 1, MARGIN - 1, WIDTH - MARGIN, CONTENT_H - MARGIN],
                           outline=(206, 212, 224), width=1)
            _draw_caption(draw, scene, page=len(composed), pages=len(pages), fonts=fonts)
            composed.append(canvas)
        pages = composed

    paths: list[Path] = []
    for i, page_img in enumerate(pages):
        path = FRAME_DIR / f"s{index:02d}_p{i}.png"
        page_img.save(path)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------
def find_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for path in FFMPEG_CANDIDATES:
        if Path(path).is_file():
            return path
    raise SystemExit(
        "[FATAL] 找不到 ffmpeg.exe。已知位置：" + "、".join(FFMPEG_CANDIDATES)
        + "；请用 --ffmpeg 指定，或只跑 --frames-only 生成分镜图（不合成视频）。")


def encode(entries: Sequence[tuple[Path, float]], out_path: Path, *, ffmpeg: str,
           logger: Any = None) -> None:
    """用 concat demuxer 把「帧图 + 每帧时长」合成 mp4（H.264 / yuv420p，兼容播放器）。"""
    list_path = out_path.parent / "_concat.txt"
    lines: list[str] = []
    for path, seconds in entries:
        lines.append(f"file '{path.as_posix()}'")
        lines.append(f"duration {max(0.5, float(seconds)):.3f}")
    lines.append(f"file '{entries[-1][0].as_posix()}'")     # concat 结尾需重复最后一帧
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-vf", f"fps=25,scale={WIDTH}:{HEIGHT}:flags=lanczos,format=yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "22",
        "-movflags", "+faststart", str(out_path),
    ]
    if logger is not None:
        logger.log_event("demo_video.encode", cmd=" ".join(cmd[:8]) + " …", out=str(out_path))
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise SystemExit(f"[FATAL] ffmpeg 编码失败（exit={proc.returncode}）：\n{proc.stderr[-1500:]}")
    list_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"[{WORK_ORDER}] 用真实截图生成演示视频（mp4）")
    parser.add_argument("--out", default=None, help="输出 mp4 路径")
    parser.add_argument("--ffmpeg", default=None, help="ffmpeg.exe 路径（默认自动探测）")
    parser.add_argument("--frames-only", action="store_true", help="只渲染分镜图，不合成视频")
    parser.add_argument("--seconds", type=float, default=None, help="统一覆盖每个镜头的时长")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bold, regular = _pick_font(FONT_BOLD_CANDIDATES), _pick_font(FONT_REG_CANDIDATES)
    fonts = {
        "title": _font(bold, 23),
        "narration": _font(regular, 19),
        "small": _font(regular, 15),
        "card": _font(bold, 46),
        "card_sub": _font(regular, 25),
        "bullet": _font(regular, 20),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    for old in FRAME_DIR.glob("*.png"):
        old.unlink()

    print(f"[{WORK_ORDER}] 渲染分镜画面（共 {len(SCENES)} 个镜头）…", flush=True)
    entries: list[tuple[Path, float]] = []
    for idx, scene in enumerate(SCENES):
        frames = render_scene_frames(scene, idx, fonts=fonts)
        per = (args.seconds if args.seconds else scene.seconds) / len(frames)
        for frame in frames:
            entries.append((frame, per))
        print(f"  镜头 {idx:02d} {scene.title[:34]:36} → {len(frames)} 页 × {per:.1f}s",
              flush=True)

    total = sum(sec for _p, sec in entries)
    print(f"[画面] {len(entries)} 帧，总时长约 {total/60:.1f} 分钟", flush=True)
    if args.frames_only:
        print(f"[完成] 只渲染画面（--frames-only）；位置 {FRAME_DIR}")
        return 0

    out_path = Path(args.out) if args.out else (OUT_DIR / "工单6_混合检索演示.mp4")
    ffmpeg = args.ffmpeg or find_ffmpeg()
    print(f"[编码] ffmpeg = {ffmpeg}", flush=True)
    encode(entries, out_path, ffmpeg=ffmpeg)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[完成] 视频：{out_path}（{size_mb:.1f} MB，约 {total/60:.1f} 分钟）", flush=True)
    print(f"[说明] 本片为「截图 + 旁白字幕」讲解片，**无音轨**；"
          f"人工录屏分镜见 部署/脚本/录屏步骤.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
