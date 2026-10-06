"""VLM 冒烟测试：渲染 PDF 一页，调用千问视觉模型识别。"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PDF = Path(r"C:\Users\tirito\Downloads\GB_T_44653-2024_六氟化硫气体现场循环再利用导则.pdf")


def main() -> None:
    from backend.app.config import AppSettings
    from backend.app.vision import OpenAIVisionClient, render_pdf_pages

    settings = AppSettings()
    print(f"VLM: enabled={settings.vlm_enabled} model={settings.vlm_model} base={settings.vlm_base_url}")

    client = OpenAIVisionClient(
        settings.vlm_base_url,
        settings.vlm_api_key,
        settings.vlm_model,
        settings.vlm_timeout,
    )

    # 渲染第 9 页（含表格）验证
    with tempfile.TemporaryDirectory(prefix="vlm-smoke-") as temp_dir:
        rendered = render_pdf_pages(PDF, {9}, Path(temp_dir))
        image_path = rendered[9]
        print(f"渲染第 9 页: {image_path.name} ({image_path.stat().st_size} bytes)")
        markdown = client.describe_image(image_path, "请把这张文档图片转换成 Markdown，保留表格结构。")
        print("\n=== 千问返回 ===")
        print(markdown[:500] if markdown else "(空返回)")

        if markdown:
            print("\n✅ VLM 调用成功")
        else:
            print("\n❌ VLM 返回为空，请检查模型名/权限/额度")


if __name__ == "__main__":
    main()
