"""为 mineru 3.4.5 应用 transformers 5.x 兼容补丁。

背景：mineru 3.4.4/3.4.5 的 PP-DocLayoutV2 模型存在 3 处与 transformers 5.x 不兼容的
问题，导致 layout 模型要么加载失败、要么检测全空（进而回退 pypdf，入库劣质数据）。

本脚本幂等地给 `mineru/model/layout/pp_doclayoutv2.py` 打上这 3 处补丁。
运行：`python scripts/patch_mineru_tf5.py`（用实际运行环境里的 python）

已知问题与补丁：
1. PPDocLayoutV2Config：transformers 5.x 严格校验在 super().__init__() 期间调用
   to_dict()，而 reading_order_config 在 super() 之后才赋值 → AttributeError。
   修复：先赋值再 super()。
2. PPDocLayoutV2ForObjectDetection：transformers 5.x 把 class_embed/bbox_embed 从
   self 移到了 self.model.decoder，替换 self.model 前取不到 → AttributeError。
   修复：从原 model.decoder 捕获后赋给新 model。
3. PPDocLayoutV2LayoutModel：transformers 5.x + torch>=2.13 在 from_pretrained 期间
   破坏两个 persistent=False buffer（读出未初始化内存）→ 检测全空 / 索引越界。
   修复：加载后从 config 重置这两个 buffer。
"""

import sys
from pathlib import Path

MARKERS = {
    "config": "# NOTE: transformers>=5.x 在 super().__init__() 期间做严格配置校验",
    "object_detection": "# NOTE: transformers 4.x 把 class_embed/bbox_embed 挂在 self 上",
    "layout_model": "# NOTE: transformers>=5.x + torch>=2.13 在 from_pretrained 期间会破坏",
}

# (原代码片段, 补丁代码片段, marker)
PATCHES = [
    (
        """        if isinstance(reading_order_config, PPDocLayoutV2ReadingOrderConfig):
            reading_order = reading_order_config
        else:
            reading_order = PPDocLayoutV2ReadingOrderConfig(**(reading_order_config or {}))

        super().__init__(""",
        """        if isinstance(reading_order_config, PPDocLayoutV2ReadingOrderConfig):
            reading_order = reading_order_config
        else:
            reading_order = PPDocLayoutV2ReadingOrderConfig(**(reading_order_config or {}))

        # NOTE: transformers>=5.x 在 super().__init__() 期间做严格配置校验，会调用 to_dict()，
        # 而 to_dict() 需要 reading_order_config，因此必须在此处先赋值。
        self.reading_order_config = reading_order

        super().__init__(""",
        "config",
    ),
    (
        """    def __init__(self, config: PPDocLayoutV2Config):
        super().__init__(config)
        self.model = PPDocLayoutV2Model(config)
        self.model.decoder.class_embed = self.class_embed
        self.model.decoder.bbox_embed = self.bbox_embed
        self.reading_order = PPDocLayoutV2ReadingOrder(config.reading_order_config)""",
        """    def __init__(self, config: PPDocLayoutV2Config):
        super().__init__(config)
        # NOTE: transformers 4.x 把 class_embed/bbox_embed 挂在 self 上，5.x 挂在
        # self.model.decoder 上。替换 self.model 前先取到原解码器 head，避免 5.x 下 AttributeError。
        class_embed = getattr(self, "class_embed", None)
        if class_embed is None:
            class_embed = self.model.decoder.class_embed
        bbox_embed = getattr(self, "bbox_embed", None)
        if bbox_embed is None:
            bbox_embed = self.model.decoder.bbox_embed
        self.model = PPDocLayoutV2Model(config)
        self.model.decoder.class_embed = class_embed
        self.model.decoder.bbox_embed = bbox_embed
        self.reading_order = PPDocLayoutV2ReadingOrder(config.reading_order_config)""",
        "object_detection",
    ),
    (
        """        self.config = PPDocLayoutV2Config.from_pretrained(self.model_dir)
        self.model = PPDocLayoutV2ForObjectDetection.from_pretrained(self.model_dir, config=self.config)
        self.model.to(self.device)""",
        """        self.config = PPDocLayoutV2Config.from_pretrained(self.model_dir)
        self.model = PPDocLayoutV2ForObjectDetection.from_pretrained(self.model_dir, config=self.config)
        # NOTE: transformers>=5.x + torch>=2.13 在 from_pretrained 期间会破坏这两个 persistent=False
        # buffer（读出未初始化内存）。必须从 config 重置，否则检测全空或索引越界。
        self.model._class_order_tensor.copy_(torch.tensor(self.config.class_order, dtype=torch.long))
        self.model._class_thresholds_tensor.copy_(
            torch.tensor(self.config.class_thresholds, dtype=torch.float32)
        )
        self.model.to(self.device)""",
        "layout_model",
    ),
]


def find_mineru_layout_file() -> Path:
    import mineru

    package_dir = Path(mineru.__file__).resolve().parent
    return package_dir / "model" / "layout" / "pp_doclayoutv2.py"


def main() -> None:
    target = find_mineru_layout_file()
    print(f"target: {target}")
    if not target.exists():
        print("!! mineru pp_doclayoutv2.py not found")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")
    changed = False
    for old, new, marker in PATCHES:
        if MARKERS[marker] in content:
            print(f"[skip] {marker}: already patched")
            continue
        if old not in content:
            print(f"[!!]  {marker}: original block not found (mineru version may differ)")
            continue
        content = content.replace(old, new, 1)
        changed = True
        print(f"[ok]  {marker}: patched")

    if not changed:
        print("全部补丁已就位，无需修改。")
        return

    target.write_text(content, encoding="utf-8")
    print(f"已写入 {target}")


if __name__ == "__main__":
    main()
