# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统
模块：语音识别（ASR）—— Windows 系统自带离线识别器

工单"产出物 / 一、系统功能 / 1"要求问答界面**支持文字和语音输入**。

【为什么用 SAPI 而不是 Whisper 类模型】
  硬约束禁止下载任何模型；本机实测无 whisper / faster-whisper / funasr / vosk，
  HF 与 ModelScope 缓存里也没有任何 ASR 权重。唯一可行的离线路径是
  **操作系统自带**的 System.Speech（SAPI 5.4）识别器 —— 它在 Windows 里
  随系统安装，不需要下载，也不需要额外 pip 依赖。

【本机实测准确率】（zh-CN 识别器 MS-2052-80-DESK，TTS 合成语音回环测试）
    注册资本是多少                      -> 注册资本是多少            ✅
    法定代表人是谁                      -> 法定代表人是谁            ✅
    军用领域的收入是多少                -> 军用领域的收入是多少      ✅
    公司参与制定了哪个技术标准          -> 公司参与制定了哪个技术标准 ✅
    募集资金多少用于补充流动资金        -> 募集资金多少用于补充流动资金 ✅
    武汉兴图新科电子股份有限公司注册资本是多少
                                        -> 武汉信徒新科电子股份有限公司注册资本是多少
                                           （"兴图"→"信徒" 同音错字，由 hotwords 纠正）
  即：**日常问句识别准确，专有名词偶有同音错字**，故配套
  `correct_domain_terms()` 做领域词近音纠正，且界面强制"回显确认"后再提问。

【踩坑记录】
  1. `Recognize()` 在到达音频末尾时**不返回 null**，而是把最后一段结果
     无限重复返回（实测第 531 类长句会把 CPU 打满且永不退出）。
     对策：以"结果文本与上一条完全相同"作为 EOF 信号，另加墙钟与次数上限。
  2. `RecognizeAsync()` + 事件回调在 PowerShell 5.1 里会卡死（事件处理器
     与主管道争用），故改用同步 `Recognize()` + 上述 EOF 守卫。
  3. PowerShell 5.1 按 **GBK** 读取 .ps1 文件，脚本里的中文会乱码并导致
     语法错误 —— 脚本必须以 **UTF-8 BOM** 写入；识别结果则写入
     **临时文件**而不是 stdout，彻底绕开控制台编码问题。

限制：本机只安装了 zh-CN 识别器，**英文语音输入不可用**；英文文字问答不受影响。
"""

from __future__ import annotations

from src import bootstrap  # noqa: F401  —— 必须最先导入

import audioop
import os
import subprocess
import tempfile
import threading
import wave

from src import config

# ---------------------------------------------------------------------------
# PowerShell 识别脚本（以 UTF-8 BOM 写入临时文件后执行）
# ---------------------------------------------------------------------------
_PS_SCRIPT = r'''
Add-Type -AssemblyName System.Speech
$wav = $args[0]
$out = $args[1]
$maxResults = [int]$args[2]
Remove-Item $out -ErrorAction SilentlyContinue

try {
    $r = New-Object System.Speech.Recognition.SpeechRecognitionEngine
    $r.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar))
    $r.SetInputToWaveFile($wav)
} catch {
    "ERROR:$($_.Exception.Message)" | Out-File $out -Encoding utf8
    exit 1
}

$parts = @()
$prev = $null
$n = 0
# EOF 守卫：Recognize() 到文件末尾不会返回 null，而是重复返回最后一段；
# 因此"结果与上一条完全相同"即判定到达末尾。
while ($n -lt $maxResults) {
    try { $res = $r.Recognize() } catch { break }
    if ($res -eq $null) { break }
    $text = $res.Text
    $n++
    if ($prev -ne $null -and $text -eq $prev) { break }
    $prev = $text
    $parts += $text
}
$r.Dispose()
($parts -join "") | Out-File $out -Encoding utf8
exit 0
'''

_available_cache: dict | None = None
_lock = threading.Lock()

# SAPI 要求的输入格式
_TARGET_RATE = 16000
_TARGET_WIDTH = 2      # 16 bit
_TARGET_CHANNELS = 1


# ---------------------------------------------------------------------------
# 可用性探测
# ---------------------------------------------------------------------------
def _recognizers() -> list[dict]:
    """列出本机已安装的 SAPI 识别器。失败返回空列表（不抛异常）。"""
    ps = ("Add-Type -AssemblyName System.Speech; "
          "[System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers() | "
          "ForEach-Object { $_.Id + '|' + $_.Culture.Name }")
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30)
    except Exception:
        return []
    out = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if "|" in line:
            rid, _, culture = line.partition("|")
            out.append({"id": rid.strip(), "culture": culture.strip()})
    return out


def available(force: bool = False) -> dict:
    """返回语音输入可用性：{ok, recognizers, cultures, reason}。

    不可用时界面应**隐藏**语音入口，而不是抛错 —— 该能力依赖操作系统组件。
    """
    global _available_cache
    if _available_cache is not None and not force:
        return _available_cache

    if os.name != "nt":
        _available_cache = {"ok": False, "recognizers": [], "cultures": [],
                            "reason": "语音输入依赖 Windows 自带的语音识别引擎"}
        return _available_cache

    recs = _recognizers()
    if not recs:
        _available_cache = {
            "ok": False, "recognizers": [], "cultures": [],
            "reason": "本机未安装 Windows 语音识别组件（控制面板→语音识别 可安装）"}
        return _available_cache

    _available_cache = {
        "ok": True, "recognizers": recs,
        "cultures": [r["culture"] for r in recs],
        "reason": "",
    }
    return _available_cache


# ---------------------------------------------------------------------------
# 音频格式转换（stdlib wave + audioop，不引入新依赖）
# ---------------------------------------------------------------------------
def to_sapi_wav(raw: bytes) -> bytes:
    """把任意 WAV 字节流转成 SAPI 需要的 16kHz / 16bit / 单声道 PCM。

    浏览器录制的采样率通常 44.1k 或 48k，直接喂给识别器会得到乱码，
    因此这里统一重采样。仅支持 PCM；其它编码抛出可读异常。
    """
    with wave.open(_BytesReader(raw), "rb") as wf:
        nchannels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        comptype = wf.getcomptype()

    if comptype != "NONE":
        raise ValueError(f"不支持的 WAV 编码：{comptype}（需要未压缩 PCM）")

    # 位深 -> 16 bit
    if width != _TARGET_WIDTH:
        frames = audioop.lin2lin(frames, width, _TARGET_WIDTH)
        width = _TARGET_WIDTH
    # 立体声 -> 单声道
    if nchannels != _TARGET_CHANNELS:
        frames = audioop.tomono(frames, width, 0.5, 0.5)
        nchannels = _TARGET_CHANNELS
    # 重采样
    if rate != _TARGET_RATE:
        frames, _ = audioop.ratecv(frames, width, nchannels, rate, _TARGET_RATE, None)
        rate = _TARGET_RATE

    buf = _BytesWriter()
    with wave.open(buf, "wb") as out:
        out.setnchannels(nchannels)
        out.setsampwidth(width)
        out.setframerate(rate)
        out.writeframes(frames)
    return buf.getvalue()


class _BytesReader:
    """给 wave.open 用的内存字节流（wave 需要 read/seek/tell）。"""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            chunk = self._data[self._pos:]
            self._pos = len(self._data)
            return chunk
        chunk = self._data[self._pos:self._pos + n]
        self._pos += len(chunk)
        return chunk

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        else:
            self._pos = len(self._data) + pos
        return self._pos

    def tell(self) -> int:
        return self._pos


class _BytesWriter:
    """给 wave.open 用的内存字节流（write 模式）。"""

    def __init__(self):
        self._chunks: list[bytes] = []

    def write(self, data: bytes) -> int:
        self._chunks.append(bytes(data))
        return len(data)

    def getvalue(self) -> bytes:
        return b"".join(self._chunks)

    def close(self) -> None:
        pass

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 领域词纠正
# ---------------------------------------------------------------------------
def correct_domain_terms(text: str) -> tuple[str, list[dict]]:
    """把识别结果里与热词"等长、仅个别字不同"的片段纠正为热词。

    通用听写模型不认识"兴图新科"这类专名，实测会输出同音字"信徒新科"。
    这里按 **等长窗口 + 高字符重合度** 做保守纠正：长度必须完全一致，
    且最多允许 1~2 个字不同，避免误伤正常文本。

    返回 (纠正后文本, 纠正记录)。
    """
    if not text:
        return text, []

    fixes: list[dict] = []
    result = text
    for term in config.ASR_HOTWORDS:
        L = len(term)
        if L < 3 or len(result) < L:
            continue
        max_diff = 2 if L >= 8 else 1
        i = 0
        while i <= len(result) - L:
            window = result[i:i + L]
            if window == term:
                i += L
                continue
            same = sum(1 for a, b in zip(window, term) if a == b)
            if L - same <= max_diff:
                result = result[:i] + term + result[i + L:]
                fixes.append({"from": window, "to": term, "pos": i})
                i += L
            else:
                i += 1
    return result, fixes


# ---------------------------------------------------------------------------
# 识别主入口
# ---------------------------------------------------------------------------
def transcribe(wav_bytes: bytes, correct: bool = True) -> tuple[str, list[dict]]:
    """识别一段 WAV 音频，返回 (文本, 领域词纠正记录)。

    失败时抛 RuntimeError（附可读原因），由界面捕获后提示用户。
    """
    if not wav_bytes:
        raise RuntimeError("没有收到音频数据")

    av = available()
    if not av["ok"]:
        raise RuntimeError(av["reason"])

    try:
        wav = to_sapi_wav(wav_bytes)
    except Exception as exc:
        raise RuntimeError(f"音频格式转换失败：{exc}") from exc

    tmpdir = tempfile.mkdtemp(prefix="asr_")
    ps_path = os.path.join(tmpdir, "recog.ps1")
    wav_path = os.path.join(tmpdir, "in.wav")
    out_path = os.path.join(tmpdir, "out.txt")
    try:
        # PowerShell 5.1 按 GBK 读 .ps1，中文必须靠 BOM 才不乱码
        with open(ps_path, "w", encoding="utf-8-sig") as fh:
            fh.write(_PS_SCRIPT)
        with open(wav_path, "wb") as fh:
            fh.write(wav)

        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive",
                 "-ExecutionPolicy", "Bypass", "-File", ps_path,
                 wav_path, out_path, str(config.ASR_MAX_RESULTS)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=config.ASR_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"语音识别超时（>{config.ASR_TIMEOUT_S}s），请缩短录音后重试") from exc

        text = ""
        if os.path.isfile(out_path):
            with open(out_path, encoding="utf-8-sig", errors="replace") as fh:
                text = fh.read().strip()

        if text.startswith("ERROR:"):
            raise RuntimeError(f"语音识别引擎报错：{text[6:].strip()}")
        if not text and proc.returncode != 0:
            err = (proc.stderr or "").strip().splitlines()
            raise RuntimeError(f"语音识别失败：{err[-1] if err else '未知错误'}")
    finally:
        for p in (ps_path, wav_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass

    if correct:
        text, fixes = correct_domain_terms(text)
    else:
        fixes = []
    return text, fixes


def info() -> dict:
    """供界面/文档展示的 ASR 配置摘要。"""
    av = available()
    return {
        "enabled": config.ASR_ENABLED,
        "engine": "Windows System.Speech (SAPI 5.4) —— 系统自带，零下载",
        "available": av["ok"],
        "cultures": av["cultures"],
        "recognizers": [r["id"] for r in av["recognizers"]],
        "reason": av["reason"],
        "hotwords": len(config.ASR_HOTWORDS),
        "timeout_s": config.ASR_TIMEOUT_S,
    }


if __name__ == "__main__":  # 自检：python -m src.asr
    import json
    print(json.dumps(info(), ensure_ascii=False, indent=1))
