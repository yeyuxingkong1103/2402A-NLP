"""MinerU layout 配置与 transformers 5.x 原生 hgnet_v2 的兼容性测试。

依赖栈已升级：transformers 5.x 原生注册 `hgnet_v2` backbone，mineru 3.4.5
不再依赖本项目里的 ResNet shim。这些测试验证原生链路仍然可用。
"""

import backend.app.mineru  # noqa: F401  确保应用模块可导入
from transformers import AutoBackbone, AutoConfig
from transformers.models.hgnet_v2.configuration_hgnet_v2 import HGNetV2Config as NativeHGNetV2Config
from transformers.models.hgnet_v2.modeling_hgnet_v2 import HGNetV2Backbone
from transformers.models.resnet.configuration_resnet import ResNetConfig

MODEL_DIR = r"D:\Claude Code代码存放处\MinerU\models\models\OpenDataLab--PDF-Extract-Kit-1.0\snapshots\master\models\Layout\PP-DocLayoutV2"


def test_hgnet_v2_config_uses_native_transformers_impl():
    config = AutoConfig.for_model(
        "hgnet_v2",
        arch="L",
        return_idx=[1, 2, 3],
        freeze_stem_only=True,
        freeze_at=0,
        freeze_norm=True,
        lr_mult_list=[0, 0.05, 0.05, 0.05, 0.05],
        out_features=["stage2", "stage3", "stage4"],
    )

    assert config.model_type == "hgnet_v2"
    assert isinstance(config, NativeHGNetV2Config)
    # 关键：必须是原生实现，不能退回 ResNet shim
    assert not isinstance(config, ResNetConfig)
    assert config.stage_names == ["stem", "stage1", "stage2", "stage3", "stage4"]
    assert config.out_features == ["stage2", "stage3", "stage4"]
    assert config.out_indices == [2, 3, 4]


def test_hgnet_v2_backbone_uses_native_transformers_impl():
    config = AutoConfig.for_model("hgnet_v2", out_features=["stage2", "stage3", "stage4"])

    backbone = AutoBackbone.from_config(config)

    assert isinstance(backbone, HGNetV2Backbone)
    assert backbone.config_class is NativeHGNetV2Config
    assert backbone.config.stage_names == ["stem", "stage1", "stage2", "stage3", "stage4"]
    assert backbone.config.out_features == ["stage2", "stage3", "stage4"]


def test_pp_doclayout_v2_config_loads_from_local_model_dir():
    # 应用实际通过 mineru 的 PPDocLayoutV2Config.from_pretrained 加载；原生 AutoConfig 也已注册该类型
    config = AutoConfig.from_pretrained(MODEL_DIR)

    assert config.model_type == "pp_doclayout_v2"


def test_mineru_pp_doclayout_config_loads_directly():
    # mineru 内部实际使用的加载路径
    from mineru.model.layout.pp_doclayoutv2 import PPDocLayoutV2Config

    config = PPDocLayoutV2Config.from_pretrained(MODEL_DIR)

    assert config.model_type == "pp_doclayout_v2"
    assert config.reading_order_config is not None
