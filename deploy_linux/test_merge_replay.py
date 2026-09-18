# -*- coding: utf-8 -*-
"""重放功能受控测试：同一数据复制两份，无 pair 做基线 vs 有 pair，组数差必须为 1，
且两组成员并到同一 vehicle_id、第二轮幂等。"""
import shutil
import sqlite3
import subprocess

SRC = "/opt/car-trace/app/granary.db"
DB_A = "/tmp/replay_base.db"
DB_B = "/tmp/replay_pair.db"
PY = "/opt/car-trace/.venv-reid/bin/python"


def run_associate(db: str) -> str:
    r = subprocess.run([PY, "/opt/car-trace/app/build_db.py", "--db", db,
                        "--associate-only"], capture_output=True, text=True)
    return r.stdout + r.stderr


def group_count(db: str) -> int:
    conn = sqlite3.connect(db)
    n = conn.execute(
        "SELECT COUNT(DISTINCT vehicle_id) FROM captures WHERE vehicle_id IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    return n


shutil.copy(SRC, DB_A)
shutil.copy(SRC, DB_B)

conn = sqlite3.connect(DB_B)
rows = conn.execute(
    """SELECT vehicle_id, MIN(id) FROM captures
       WHERE vehicle_id IS NOT NULL
       GROUP BY vehicle_id ORDER BY vehicle_id LIMIT 2""").fetchall()
(v1, c1), (v2, c2) = rows
conn.execute(
    """INSERT INTO manual_vehicle_merge_pairs
       (capture_id, anchor_capture_id, park, created_at)
       SELECT ?, ?, coalesce(park,''), datetime('now') FROM captures WHERE id=?""",
    (c1, c2, c2))
conn.commit()
conn.close()
print(f"[准备] vehicle {v1} + vehicle {v2} 人工合并（代表抓拍 {c1} -> {c2}）")

out_a = run_associate(DB_A)
G0 = group_count(DB_A)
print(f"[基线] 无 pair 组数: {G0}")

out_b = run_associate(DB_B)
G1 = group_count(DB_B)
line = [ln for ln in out_b.splitlines() if "人工合并重放" in ln]
print(f"[带pair] 组数: {G1} | {line[0] if line else '(无重放输出!!)'}")
assert G1 == G0 - 1, f"组数差 {G0 - G1} != 1"

conn = sqlite3.connect(DB_B)
v_c1 = conn.execute("SELECT vehicle_id FROM captures WHERE id=?", (c1,)).fetchone()[0]
v_c2 = conn.execute("SELECT vehicle_id FROM captures WHERE id=?", (c2,)).fetchone()[0]
conn.close()
print(f"[带pair] 抓拍 {c1} -> vehicle {v_c1}，抓拍 {c2} -> vehicle {v_c2}（应相等）")
assert v_c1 == v_c2, "两组没有并到同一个 vehicle_id"

out_b2 = run_associate(DB_B)
G2 = group_count(DB_B)
line2 = [ln for ln in out_b2.splitlines() if "人工合并重放" in ln]
# 第二轮重聚类会再次拆开不同车牌的组，pair 再次重放是预期行为；
# 验收标准是最终状态稳定：组数不变、两抓拍仍同组。
conn = sqlite3.connect(DB_B)
v_c1b = conn.execute("SELECT vehicle_id FROM captures WHERE id=?", (c1,)).fetchone()[0]
v_c2b = conn.execute("SELECT vehicle_id FROM captures WHERE id=?", (c2,)).fetchone()[0]
conn.close()
print(f"[幂等] 第二轮组数: {G2} | {line2[0] if line2 else '(无重放输出)'} | "
      f"两抓拍同组: {v_c1b == v_c2b}")
assert G2 == G1, "第二轮组数变化"
assert v_c1b == v_c2b, "第二轮两抓拍不同组"

print("== 重放测试全部通过 ==")
