from huggingface_hub import snapshot_download

model_id = "all-MiniLM-L6-v2"
print("开始下载模型 all-MiniLM-L6-v2 ...")
snapshot_download(
    repo_id=model_id,
    local_dir_use_symlinks=False,
    resume_download=True,
    force_download=False
)
print("✅ 模型下载完成！")
