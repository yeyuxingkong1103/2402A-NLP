# 工单编号：人工智能NLP-RAG-基于GraphRAG实现金融问答
"""
模块：语音输入（工单产出物要求「问答界面支持文字和语音输入」）

选型：浏览器原生 **Web Speech API**（`webkitSpeechRecognition`）
  · 零安装、零模型下载 —— 与本工程「零下载」约束一致；
  · 识别在浏览器/系统侧完成，中英双语通过 `lang` 切换。

⚠️ 降级说明（工单硬性要求0 Step 4：用户拒绝下载的模型必须标注降级方案）：
  本机**没有 Whisper 等本地 ASR 权重**，用户已确认「零下载」策略，
  因此语音识别依赖 Web Speech API。该 API：
    · Chrome/Edge 需要联网（识别在云端完成）；
    · 完全离线环境或 Firefox 中不可用。
  不可用时界面**明确提示并降级为文字输入**。此降级方案不满足工单
  「语音输入」在离线场景下的字面要求，特此标注。
"""
from __future__ import annotations

SPEECH_DEGRADATION_NOTE = (
    "⚠️ 语音输入使用浏览器原生 Web Speech API（零安装、未下载任何 ASR 模型）。"
    "该方法依赖浏览器联网与系统语音服务；在完全离线环境或不支持的浏览器中会不可用，"
    "此时请使用文字输入。此为本工程的**降级方案**，"
    "若需完全离线的语音识别，需额外下载 Whisper 等 ASR 模型（本工程未下载）。"
)


def browser_support_note() -> str:
    return ("语音输入支持情况：Chrome / Edge 通常可用（需联网）；"
            "Firefox 不支持；离线环境不可用。")


def speech_component_html(lang: str = "zh-CN",
                          button_label: str = "🎤 语音输入") -> str:
    """返回可嵌入 Streamlit components.html 的自包含语音输入组件。

    识别结果通过 `window.parent.postMessage` 发送，宿主页面可用
    `components.html(..., height=...)` + 查询参数方式读取；
    同时组件内也直接显示识别文本，保证任何宿主下都不丢结果。
    """
    lang_js = "zh-CN" if not lang.lower().startswith("en") else "en-US"
    zone = "中文" if lang_js == "zh-CN" else "English"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
  body {{ margin:0; font-family:"Microsoft YaHei",system-ui,sans-serif; }}
  #btn {{ padding:8px 16px; font-size:14px; border:1px solid #d0d5dd;
          border-radius:8px; background:#fff; cursor:pointer; }}
  #btn.rec {{ background:#e5484d; color:#fff; border-color:#e5484d; }}
  #out {{ margin-top:6px; font-size:13px; color:#333; min-height:18px; }}
  #warn {{ font-size:12px; color:#b54708; }}
</style></head>
<body>
<button id="btn">{button_label}</button>
<div id="out"></div>
<div id="warn"></div>
<script>
(function() {{
  var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  var btn = document.getElementById('btn');
  var out = document.getElementById('out');
  var warn = document.getElementById('warn');

  if (!SR) {{
    warn.textContent = '当前浏览器不支持语音识别（Web Speech API），'
                     + '请改用文字输入。';
    btn.disabled = true;
    return;
  }}
  warn.textContent = '语音识别语言：{zone}（需浏览器联网；离线环境不可用）';

  var recognition = new SR();
  recognition.lang = '{lang_js}';
  recognition.continuous = false;
  recognition.interimResults = true;

  var finalText = '';
  recognition.onresult = function(e) {{
    var interim = '';
    for (var i = e.resultIndex; i < e.results.length; i++) {{
      var t = e.results[i][0].transcript;
      if (e.results[i].isFinal) {{ finalText += t; }} else {{ interim += t; }}
    }}
    out.textContent = finalText || interim;
    if (finalText) {{
      try {{
        window.parent.postMessage(
          {{type: 'wo08-speech', text: finalText,
            params: new URLSearchParams({{q: finalText}}).toString()}}, '*');
      }} catch (err) {{ /* 忽略跨域失败，页面内已显示文本 */ }}
    }}
  }};
  recognition.onerror = function(e) {{
    warn.textContent = '语音识别出错：' + e.error + '（可改用文字输入）';
    btn.classList.remove('rec');
  }};
  recognition.onend = function() {{ btn.classList.remove('rec'); }};

  btn.onclick = function() {{
    finalText = '';
    out.textContent = '';
    try {{
      recognition.start();
      btn.classList.add('rec');
    }} catch (err) {{
      warn.textContent = '无法启动语音识别：' + err;
    }}
  }};
}})();
</script>
</body></html>"""
