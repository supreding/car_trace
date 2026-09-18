# -*- coding: utf-8 -*-
"""打包部署压缩包：排除大体积/可重建内容，data/ 只放空目录占位。

用法：在项目根目录执行
    python make_deploy_package.py
输出：D:\\WorkSpace\\车辆轨迹系统部署包_<日期>.zip
可重复执行；库或代码更新后重新打包即可。
"""
from __future__ import annotations

import zipfile
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT.parent

# 根目录白名单（只带脚本与文档，不带开发产物）
ROOT_FILES = [
    "启动查询服务.bat",
    "启动识别流水线.bat",
    "start_server.ps1",
    "start_pipeline.ps1",
    "run_reid_pipeline.ps1",
    "requirements-server.txt",
    "requirements-reid.txt",
    "部署说明.txt",
    "PaddleDetection_FastReID_车辆识别16步实施方案.md",
]

# app/ 内排除项：缓存、运行目录、数据库备份
APP_EXCLUDE = {"__pycache__", ".mpl", ".reid-home"}
APP_EXCLUDE_SUFFIX = (".bak-20260907",)

# 空目录占位：data/ 及现有园区名（照片单独拷贝进来）
EMPTY_DIRS = ["data/", "data/山高荃银丰麦田库/", "data/农易行临泉辉煌库/"]

# 整目录携带（排除 .git）
TREE_DIRS = ["models", "third_party"]


def zip_dir(zf: zipfile.ZipFile, src: Path, arc_prefix: str) -> tuple[int, int]:
    """把目录写入 zip，返回 (文件数, 字节数)。跳过 .git 与 app 排除项。"""
    count = total = 0
    for path in sorted(src.rglob("*")):
        relative = path.relative_to(src)
        if any(part in APP_EXCLUDE or part == ".git" for part in relative.parts):
            continue
        if path.is_dir():
            continue
        if arc_prefix == "app" and str(relative).endswith(APP_EXCLUDE_SUFFIX):
            continue
        with path.open("rb") as f:
            info = zipfile.ZipInfo(
                f"{arc_prefix}/{relative.as_posix()}",
                date_time=datetime.fromtimestamp(path.stat().st_mtime).timetuple()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, f.read(), compresslevel=1)
        count += 1
        total += path.stat().st_size
        if count % 500 == 0:
            print(f"  {arc_prefix}: 已打包 {count} 个文件...", flush=True)
    return count, total


def main() -> None:
    out_path = OUT_DIR / f"车辆轨迹系统部署包_{date.today():%Y%m%d}.zip"
    files = bytes_total = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for name in ROOT_FILES:
            path = ROOT / name
            if not path.is_file():
                print(f"[跳过] 根文件不存在：{name}")
                continue
            zf.write(path, name)
            files += 1
            bytes_total += path.stat().st_size

        count, size = zip_dir(zf, ROOT / "app", "app")
        files += count
        bytes_total += size
        print(f"app/：{count} 个文件，{size / 1e9:.2f} GB")

        for name in EMPTY_DIRS:
            info = zipfile.ZipInfo(name)
            info.external_attr = 0o40755 << 16 | 0x10  # 目录属性
            zf.writestr(info, "")

        for name in TREE_DIRS:
            count, size = zip_dir(zf, ROOT / name, name)
            files += count
            bytes_total += size
            print(f"{name}/：{count} 个文件，{size / 1e9:.2f} GB")

    print()
    print(f"打包完成：{out_path}")
    print(f"文件数：{files}，原始体积：{bytes_total / 1e9:.2f} GB，"
          f"压缩包：{out_path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
