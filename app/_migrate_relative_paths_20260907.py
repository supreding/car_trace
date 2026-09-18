# -*- coding: utf-8 -*-
"""一次性迁移：库内原图路径改为项目根相对路径（正斜杠），裁剪路径统一正斜杠。

背景：captures.filepath 原先存开发机绝对路径，照片移动到 data/<园区>/ 后
（以及未来部署到其他机器时）会全部失效。迁移后 filepath 形如
    data/<园区>/2026年8月份抓拍照片/<日期>/<订单>/<文件>
配合 build_db.resolve_stored_path / server.resolve_capture_path 在任何机器
上都能按项目根解析。执行前自动备份，可幂等重跑。
"""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
STAMP = "20260907"

MIGRATIONS = [
    # (库文件, 旧绝对路径前缀, 新相对路径前缀, park 值(None=不回填))
    ("granary.db",
     r"D:\WorkSpace\car_trace\2026年8月份抓拍照片",
     "data/山高荃银丰麦田库/2026年8月份抓拍照片",
     "山高荃银丰麦田库"),
    ("nongyixing.db",
     r"D:\WorkSpace\car_trace\data\农易行临泉辉煌库\2026年8月份抓拍照片",
     "data/农易行临泉辉煌库/2026年8月份抓拍照片",
     None),
]


def migrate(db_name: str, old_prefix: str, new_prefix: str, park: str | None) -> None:
    db_path = APP_DIR / db_name
    backup = APP_DIR / f"{db_name}.bak-{STAMP}"
    if not backup.exists():
        shutil.copy2(db_path, backup)
        print(f"[{db_name}] 已备份 -> {backup.name}")

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id, filepath FROM captures").fetchall()
    updates, skipped, problems = [], 0, []
    for capture_id, filepath in rows:
        filepath = str(filepath or "")
        if filepath.replace("\\", "/").startswith(new_prefix):
            skipped += 1  # 已是目标形式（幂等重跑）
            continue
        if not filepath.startswith(old_prefix):
            problems.append((capture_id, "前缀不匹配", filepath))
            continue
        relative = new_prefix + "/" + filepath[len(old_prefix) + 1:].replace("\\", "/")
        if not (PROJECT_ROOT / relative).is_file():
            problems.append((capture_id, "目标文件不存在", relative))
            continue
        updates.append((relative, capture_id))
    conn.executemany("UPDATE captures SET filepath=? WHERE id=?", updates)
    print(f"[{db_name}] filepath 更新 {len(updates)} 条，跳过(已是相对) {skipped} 条，异常 {len(problems)} 条")
    for capture_id, reason, path in problems[:5]:
        print(f"  异常样例 id={capture_id} ({reason}): {path}")

    if park:
        n1 = conn.execute(
            "UPDATE captures SET park=? WHERE coalesce(park,'')=''", (park,)).rowcount
        n2 = conn.execute(
            "UPDATE order_attributes SET park=? WHERE coalesce(park,'')=''", (park,)).rowcount
        print(f"[{db_name}] park 回填：captures {n1} 条，order_attributes {n2} 条")

    # 裁剪路径统一正斜杠（原为 reid_crops\xxx.jpg 形式，相对 app 目录）。
    for table, column in (("captures", "vehicle_crop_path"), ("vehicle_instances", "crop_path")):
        n = conn.execute(
            f"UPDATE {table} SET {column}=replace({column}, '\\', '/') "
            f"WHERE {column} LIKE '%\\%'"
        ).rowcount
        print(f"[{db_name}] {table}.{column} 反斜杠转正斜杠：{n} 条")
    conn.commit()

    bad = sum(1 for (p,) in conn.execute("SELECT filepath FROM captures")
              if not (PROJECT_ROOT / str(p or "")).is_file()
              and not (APP_DIR / str(p or "")).is_file())
    print(f"[{db_name}] 迁移后全表校验：原图缺失 {bad} 条")
    conn.close()


if __name__ == "__main__":
    for db_name, old_prefix, new_prefix, park in MIGRATIONS:
        migrate(db_name, old_prefix, new_prefix, park)
