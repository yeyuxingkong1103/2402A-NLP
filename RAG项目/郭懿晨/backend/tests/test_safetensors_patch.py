import backend.app.mineru  # noqa: F401


def test_safetensors_metadata_patch_handles_none_metadata():
    from safetensors import safe_open

    model_path = r"D:\Claude Code代码存放处\MinerU\models\models\OpenDataLab--PDF-Extract-Kit-1.0\snapshots\master\models\Layout\PP-DocLayoutV2\model.safetensors"
    with safe_open(model_path, framework="pt") as file_handle:
        metadata = file_handle.metadata()

    assert metadata is None or isinstance(metadata, dict)
