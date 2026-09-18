# -*- coding: utf-8 -*-
"""把 nongyixing.db（农易行）全量合并进 granary.db 副本（单库多园区）。

- captures / vehicle_instances / order_attributes / manual_vehicle_merges
  全部搬运，OCR、车牌、embedding、裁剪图路径等计算结果原样保留，不重跑 GPU；
- capture_id 重新自增分配并建立新旧映射，vehicle_instances 跟随重映射；
- vehicle_id 加偏移量（两园区各自从 1 编号，直接合并会撞号）；
- 裁剪图路径 "reid_crops_nongyixing/..." 相对 app/ 目录，合并后仍然有效；
- 标注表（人工真值）不做搬运（nongyixing 仅 1 条 pair，可忽略）。
在副本上操作，验证通过后再由外部替换正式库。
"""
import sqlite3
import sys

DST = "/tmp/merge.db"
SRC = "/opt/car-trace/app/nongyixing.db"

conn = sqlite3.connect(DST)
conn.execute("ATTACH ? AS ny", (SRC,))

gcols = [r[1] for r in conn.execute("PRAGMA table_info(captures)")]
ncols = [r[1] for r in conn.execute("PRAGMA ny.table_info(captures)")]
common = [c for c in gcols if c in ncols and c != "id"]
missing_in_src = [c for c in gcols if c not in ncols and c != "id"]
if missing_in_src:
    print("警告：目标库存在而源库没有的列（填 NULL）：", missing_in_src)

vid_offset = conn.execute(
    "SELECT coalesce(MAX(vehicle_id),0) FROM captures").fetchone()[0]
print(f"vehicle_id 偏移量：{vid_offset}")

# ---- 1. captures：逐行插入，建立 old_id -> new_id 映射，vehicle_id 加偏移 ----
conn.execute("CREATE TEMP TABLE idmap(old_id INTEGER PRIMARY KEY, new_id INTEGER)")
sel = ", ".join(common)
placeholders = ",".join("?" for _ in common)
rows = conn.execute(f"SELECT id, {sel} FROM ny.captures ORDER BY id").fetchall()
ins = conn.execute(f"PRAGMA table_info(captures)")  # 仅确认表存在
vid_idx = common.index("vehicle_id")
inserted = 0
for old_id, *values in rows:
    values = list(values)
    if values[vid_idx] is not None:
        values[vid_idx] = values[vid_idx] + vid_offset
    cur = conn.execute(
        f"INSERT INTO captures ({sel}) VALUES ({placeholders})", values)
    conn.execute("INSERT INTO idmap VALUES (?, ?)", (old_id, cur.lastrowid))
    inserted += 1
print(f"captures 合并：{inserted} 行")

# ---- 2. vehicle_instances：capture_id 按映射，vehicle_id 加偏移 ----
vi_cols = [r[1] for r in conn.execute("PRAGMA table_info(vehicle_instances)")]
vi_common = [c for c in vi_cols if c in
             [r[1] for r in conn.execute("PRAGMA ny.table_info(vehicle_instances)")]
             and c != "id"]
vi_sel = ", ".join(vi_common)
vi_rows = conn.execute(
    f"""SELECT {vi_sel} FROM ny.vehicle_instances vi
        JOIN idmap ON idmap.old_id = vi.capture_id""").fetchall()
cap_idx = vi_common.index("capture_id")
vidx2 = vi_common.index("vehicle_id") if "vehicle_id" in vi_common else -1
ph = ",".join("?" for _ in vi_common)
count_vi = 0
for values in vi_rows:
    values = list(values)
    values[cap_idx] = conn.execute(
        "SELECT new_id FROM idmap WHERE old_id=?", (values[cap_idx],)).fetchone()[0]
    if vidx2 >= 0 and values[vidx2] is not None:
        values[vidx2] += vid_offset
    conn.execute(
        f"INSERT INTO vehicle_instances ({vi_sel}) VALUES ({ph})", values)
    count_vi += 1
print(f"vehicle_instances 合并：{count_vi} 行")

# ---- 3. order_attributes（含 park，直接插） ----
oa_cols = [r[1] for r in conn.execute("PRAGMA table_info(order_attributes)")]
oa_common = [c for c in oa_cols if c in
             [r[1] for r in conn.execute("PRAGMA ny.table_info(order_attributes)")]]
oa_sel = ", ".join(oa_common)
oa_rows = conn.execute(f"SELECT {oa_sel} FROM ny.order_attributes").fetchall()
ph = ",".join("?" for _ in oa_common)
conn.executemany(
    f"INSERT OR REPLACE INTO order_attributes ({oa_sel}) VALUES ({ph})", oa_rows)
print(f"order_attributes 合并：{len(oa_rows)} 行")

# ---- 4. 人工合并决定（如有，vehicle_id 加偏移） ----
n_merges = conn.execute("SELECT COUNT(*) FROM ny.manual_vehicle_merges").fetchone()[0]
if n_merges:
    conn.execute(
        """INSERT INTO manual_vehicle_merges
           SELECT source_vehicle_id + ?, target_vehicle_id + ?, created_at
           FROM ny.manual_vehicle_merges""", (vid_offset, vid_offset))
print(f"manual_vehicle_merges：{n_merges} 行")
n_moves = conn.execute("SELECT COUNT(*) FROM ny.manual_vehicle_moves").fetchone()[0]
print(f"manual_vehicle_moves（源库）：{n_moves} 行" +
      ("，如有需要请人工复核" if n_moves else ""))

conn.commit()

# ---- 5. 验证 ----
print("---- 合并后验证 ----")
cap_total = conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0]
per_park = conn.execute(
    "SELECT park, COUNT(*) FROM captures GROUP BY park").fetchall()
vi_total = conn.execute("SELECT COUNT(*) FROM vehicle_instances").fetchone()[0]
cross = conn.execute(
    """SELECT COUNT(*) FROM (SELECT vehicle_id FROM captures
       WHERE vehicle_id IS NOT NULL GROUP BY vehicle_id
       HAVING COUNT(DISTINCT park) > 1)""").fetchone()[0]
emb = conn.execute(
    "SELECT COUNT(*) FROM captures WHERE reid_embedding IS NOT NULL").fetchone()[0]
ny_order_hit = conn.execute(
    """SELECT COUNT(*) FROM captures WHERE order_id=
       (SELECT order_id FROM ny.captures LIMIT 1)""").fetchone()[0]
conn.close()
print(f"captures 总数：{cap_total}，分布：{per_park}")
print(f"vehicle_instances：{vi_total} | 跨园区撞号 vehicle_id：{cross} | 有特征：{emb}")
print(f"农易行订单号在合并库中命中：{ny_order_hit}（应>0）")
if cap_total != 8302 or vi_total != 8126 or cross != 0 or ny_order_hit == 0:
    print("!! 验证未通过")
    sys.exit(1)
print("== 合并验证通过 ==")
