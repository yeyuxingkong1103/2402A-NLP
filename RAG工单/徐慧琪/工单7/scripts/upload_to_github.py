# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-功能测试及评估
"""
工单07 · 上传到课程仓库（2402A-NLP / rag工单 / 徐慧琪 / 工单7）

为什么要单独一个脚本
--------------------
本工程的本地 git 仓库是**独立初始化**的，与课程仓库**没有共同历史**，
因此不能直接 `git push` 上去。正确做法是：克隆课程仓库 →
把本工程内容放进目标目录 → 提交 → 推送。

脚本只**新增**目标目录下的文件，不删除、不修改仓库里其他同学的任何内容。
默认 **dry-run**：只克隆 + 拷贝 + 打印将要提交的清单，不动远端；
加 `--push` 才真正提交并推送。

用法
----
    # 1) 先演练，确认清单无误（不会推送）
    python scripts/upload_to_github.py --repo <仓库地址>

    # 2) 确认后真正推送
    python scripts/upload_to_github.py --repo <仓库地址> --push

仓库地址示例
------------
    https://github.com/<用户名>/2402A-NLP.git
    https://gitee.com/<用户名>/2402A-NLP.git

凭据
----
HTTPS 方式下 git 会自己弹出凭据输入（或使用已配置的 credential helper）。
本脚本**不接触、不保存任何凭据**。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 仓库内的目标路径：2402A-NLP / rag工单 / 徐慧琪 / 工单7
TARGET_SUBPATH = ["rag工单", "徐慧琪", "工单7"]


def run(cmd: list[str], cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """执行命令并回显（凭据交互交给 git 自己处理，不捕获 stdout）。"""
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=cwd, check=check)


def export_tracked(dest_zip: str) -> int:
    """用 git archive 导出**已提交且未被 .gitignore 排除**的文件。

    这是关键：直接拷贝目录会把 1.1GB 模型、97MB 向量库等一起搬过去；
    git archive 只导出 HEAD 里跟踪的文件，与"将上传的内容"完全一致。
    """
    print("      从本地仓库导出已跟踪文件（等价于将要上传的内容）")
    res = subprocess.run(["git", "archive", "HEAD", "-o", dest_zip],
                         cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"git archive 失败：{res.stderr.strip()}")
    with zipfile.ZipFile(dest_zip) as z:
        n = len([x for x in z.namelist() if not x.endswith("/")])
    print(f"      导出 {n} 个文件 → {os.path.basename(dest_zip)}")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="上传工单07 到课程仓库")
    ap.add_argument("--repo", required=True, help="仓库地址（https://... 或 git@...）")
    ap.add_argument("--push", action="store_true", help="真正提交并推送（默认只演练）")
    ap.add_argument("--branch", default="", help="目标分支（默认用仓库当前分支）")
    ap.add_argument("--workdir", default="", help="克隆到哪个目录（默认临时目录）")
    ap.add_argument("--message", default="", help="提交信息（默认自动生成）")
    args = ap.parse_args()

    if os.path.exists(os.path.join(ROOT, ".git")) is False:
        raise SystemExit("本工程不是 git 仓库，请先在工程根目录执行 git init && git commit")

    workdir = args.workdir or tempfile.mkdtemp(prefix="ccf_upload_")
    clone_dir = os.path.join(workdir, "repo")

    try:
        print(f"[0/5] 工作目录：{workdir}")
        print(f"[1/5] 克隆仓库（只读，尚未做任何修改）")
        run(["git", "clone", args.repo, clone_dir])
        if args.branch:
            run(["git", "checkout", args.branch], cwd=clone_dir)

        print("[2/5] 导出本地已提交内容")
        zip_path = os.path.join(workdir, "upload.zip")
        n_files = export_tracked(zip_path)

        print(f"[3/5] 解包到 {'/'.join(TARGET_SUBPATH)}/")
        dest = os.path.join(clone_dir, *TARGET_SUBPATH)
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)

        print("[4/5] 变更预览（只新增，不删除）")
        # 注意：pathspec 只能给**顶层**那一级（"rag工单"），
        # 给多级子路径（"徐慧琪"）会因该路径尚不存在而报
        # `pathspec ... did not match any files`（实测踩坑）。
        rel_top = TARGET_SUBPATH[0]
        run(["git", "add", "--", rel_top], cwd=clone_dir)
        # encoding 必须显式指定 utf-8：Windows 默认按 GBK 解码 git 输出，
        # 遇到中文文件名会抛 UnicodeDecodeError 并使 stdout 变成 None（实测踩坑）。
        st = subprocess.run(["git", "-c", "core.quotePath=false",
                             "status", "--short", "--", rel_top],
                            cwd=clone_dir, capture_output=True,
                            encoding="utf-8", errors="replace")
        lines = [l for l in (st.stdout or "").splitlines() if l.strip()]
        print(f"      将提交 {len(lines)} 个文件，其中：")
        added = sum(1 for l in lines if l.startswith("A"))
        other = [l for l in lines if not l.startswith("A")]
        print(f"        新增 A : {added}")
        if other:
            print(f"        其它状态（**请人工确认是否正常**）: {len(other)}")
            for l in other[:10]:
                print(f"          {l}")
        if added != n_files and not other:
            print(f"      [提示] 导出 {n_files} 个但新增 {added} 个，请核对")

        if not args.push:
            print("\n[演练结束] 未做任何提交/推送。")
            print(f"  克隆保留在：{clone_dir}")
            print(f"  确认无误后重跑并加 --push 即可真正上传。")
            return 0

        print("[5/5] 提交并推送")
        msg = args.message or "工单07：功能测试及评估（人工智能NLP-RAG-功能测试及评估）"
        run(["git", "commit", "-m", msg], cwd=clone_dir)
        run(["git", "push"], cwd=clone_dir)
        print("\n[完成] 已推送。")
        return 0

    except subprocess.CalledProcessError as exc:
        print(f"\n[失败] 命令返回 {exc.returncode}：{' '.join(exc.cmd)}", file=sys.stderr)
        print("      常见原因：仓库地址错误 / 无写权限 / 凭据未配置。", file=sys.stderr)
        print(f"      克隆目录保留在 {clone_dir}，可手动排查。", file=sys.stderr)
        return 1
    finally:
        if not args.push and os.path.isdir(workdir):
            pass          # 演练模式保留现场，便于人工核对


if __name__ == "__main__":
    sys.exit(main())
