# -*- coding: utf-8 -*-
"""补搬 nongyixing 的 manual_vehicle_moves（capture_id 映射：new = 4908 + old）。"""
import sqlite3

OFFSET = 4908  # granary 原 max(id)，与合并脚本一致
conn = sqlite3.connect("/tmp/merge.db")
conn.execute("ATTACH ? AS ny", ("/opt/car-trace/app/nongyixing.db",))
rows = conn.execute("SELECT capture_id, target_plate, scope, moved_at FROM ny.manual_vehicle_moves").fetchall()
for old_id, plate, scope, moved_at in rows:
    new_id = OFFSET + old_id
    exists = conn.execute("SELECT COUNT(*) FROM captures WHERE id=?", (new_id,)).fetchone()[0]
    if not exists:
        print(f"跳过（映射后无此抓拍）: old={old_id}")
        continue
    conn.execute(
        """INSERT INTO manual_vehicle_moves (capture_id, target_plate, scope, moved_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(capture_id) DO UPDATE SET
             target_plate=excluded.target_plate, scope=excluded.scope,
             moved_at=excluded.moved_at""",
        (new_id, plate, scope, moved_at))
    print(f"搬运: capture {old_id} -> {new_id}, target_plate={plate}, scope={scope}")
conn.commit()
n = conn.execute("SELECT COUNT(*) FROM manual_vehicle_moves").fetchone()[0]
print(f"合并库 manual_vehicle_moves 总数: {n}")
conn.close()
