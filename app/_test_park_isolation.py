# -*- coding: utf-8 -*-
"""多园区隔离验证：
1) 两园区同车牌聚类隔离——同牌车在两个园区必须得到相互独立的 vehicle_id；
2) 同园区内同牌车仍正常跨订单合并；
3) nongyixing.db 副本：park 回填迁移 + 重扫不重复 + 重聚类冒烟。
"""
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_db


def make_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    build_db.ensure_schema(conn)
    return conn


def insert_capture(conn, park, order_id, date, plate, filename, hour=10):
    # 真实流水线中车辆帧都有 reid_embedding；组特征查询要求其非空，
    # 这里用全零 4 维向量代替（不影响按车牌合并的逻辑路径）。
    fake_emb = b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"  # 4x float32
    conn.execute(
        """INSERT INTO captures (filepath, park, date, order_id, stage, filename,
                                 capture_status, plate_final, lpr_conf, quality_score,
                                 reid_embedding)
           VALUES (?, ?, ?, ?, '入磅', ?, 'vehicle', ?, 0.98, 80.0, ?)""",
        (f"D:\\fake\\data\\{park}\\2026年8月份抓拍照片\\8月10日\\{order_id}\\入磅_x.jpg",
         park, date, order_id, filename, plate, fake_emb),
    )


def test_cross_park_same_plate():
    conn = make_db(Path(tempfile.mkdtemp()) / "t1.db")
    # 园区 A：两个订单同牌 皖K12345（应并成一辆车）
    insert_capture(conn, "农易行临泉辉煌库", "109002608170001", "2026-08-17", "皖K12345", "a1.jpg")
    insert_capture(conn, "农易行临泉辉煌库", "109002608170001", "2026-08-17", "皖K12345", "a2.jpg")
    insert_capture(conn, "农易行临泉辉煌库", "109002608180001", "2026-08-18", "皖K12345", "a3.jpg")
    insert_capture(conn, "农易行临泉辉煌库", "109002608180001", "2026-08-18", "皖K12345", "a4.jpg")
    # 园区 B：两个订单，完全相同的车牌（必须独立成车，绝不跨园区合并）
    insert_capture(conn, "老库", "68002608090001", "2026-08-09", "皖K12345", "b1.jpg")
    insert_capture(conn, "老库", "68002608090001", "2026-08-09", "皖K12345", "b2.jpg")
    insert_capture(conn, "老库", "68002608100001", "2026-08-10", "皖K12345", "b3.jpg")
    insert_capture(conn, "老库", "68002608100001", "2026-08-10", "皖K12345", "b4.jpg")
    conn.commit()

    build_db.associate_orders(conn)

    rows = conn.execute(
        "SELECT DISTINCT coalesce(park,''), vehicle_id FROM captures WHERE vehicle_id IS NOT NULL"
    ).fetchall()
    by_park = {}
    for park, vid in rows:
        by_park.setdefault(park, set()).add(vid)
    print("各园区 vehicle_id 集合：", by_park)

    assert set(by_park) == {"农易行临泉辉煌库", "老库"}, f"园区缺失: {by_park}"
    vids_a, vids_b = by_park["农易行临泉辉煌库"], by_park["老库"]
    assert vids_a.isdisjoint(vids_b), f"跨园区车辆被合并！A={vids_a} B={vids_b}"
    assert len(vids_a) == 1, f"园区 A 同牌车应并成 1 辆，实际 {len(vids_a)} 辆"
    assert len(vids_b) == 1, f"园区 B 同牌车应并成 1 辆，实际 {len(vids_b)} 辆"

    # 相近车牌挂靠也不能跨园区：园区 B 增加一个 1 字符差异的读数
    insert_capture(conn, "老库", "68002608110001", "2026-08-11", "皖K12346", "b5.jpg")
    insert_capture(conn, "老库", "68002608110001", "2026-08-11", "皖K12346", "b6.jpg")
    insert_capture(conn, "农易行临泉辉煌库", "109002608190001", "2026-08-19", "皖K12346", "a5.jpg")
    insert_capture(conn, "农易行临泉辉煌库", "109002608190001", "2026-08-19", "皖K12346", "a6.jpg")
    conn.commit()
    build_db.associate_orders(conn)
    rows = conn.execute(
        "SELECT DISTINCT coalesce(park,''), vehicle_id FROM captures WHERE vehicle_id IS NOT NULL"
    ).fetchall()
    by_park = {}
    for park, vid in rows:
        by_park.setdefault(park, set()).add(vid)
    print("相近车牌加入后各园区 vehicle_id 集合：", by_park)
    assert by_park["农易行临泉辉煌库"].isdisjoint(by_park["老库"]), "相近车牌挂靠跨园区合并！"
    conn.close()
    print("[PASS] 跨园区同牌/近牌车辆保持隔离\n")


def test_nongyixing_migration():
    src = Path(__file__).resolve().parent / "nongyixing.db"
    tmp = Path(tempfile.mkdtemp()) / "nongyixing_copy.db"
    shutil.copy(src, tmp)
    conn = sqlite3.connect(tmp)
    before = conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0]
    before_groups = conn.execute(
        "SELECT COUNT(DISTINCT vehicle_id) FROM captures").fetchone()[0]

    build_db.ensure_schema(conn)  # 建列 + 一次性回填
    backfilled = conn.execute(
        "SELECT coalesce(park,''), COUNT(*) FROM captures GROUP BY 1").fetchall()
    print("回填后园区分布：", backfilled)
    assert len(backfilled) == 1 and backfilled[0][0] == "农易行临泉辉煌库", backfilled

    # 重扫同一目录：key=(park, order_id, filename) 应全部命中更新，不重复插入
    data_root = Path("D:/WorkSpace/car_trace/data")
    summary = build_db.sync_files(conn, data_root)
    print("重扫结果：", summary)
    after = conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0]
    assert after == before, f"重扫后记录数变化：{before} -> {after}"
    # 再次 ensure_schema 不应重复回填/报错
    build_db.ensure_schema(conn)
    assert conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0] == after

    # 重聚类冒烟：组数应与升级前同量级（单园区库行为不变）
    build_db.associate_orders(conn)
    after_groups = conn.execute(
        "SELECT COUNT(DISTINCT vehicle_id) FROM captures").fetchone()[0]
    linked = conn.execute(
        "SELECT COUNT(*) FROM captures WHERE vehicle_id IS NOT NULL").fetchone()[0]
    print(f"聚类：升级前 {before_groups} 组 -> 升级后 {after_groups} 组，关联 {linked} 张")
    assert after_groups == before_groups, "单园区库重聚类后组数不应变化"

    # 订单属性表迁移为 (park, order_id) 主键
    cols = {r[1] for r in conn.execute("PRAGMA table_info(order_attributes)")}
    assert "park" in cols, f"order_attributes 缺 park 列：{cols}"
    n = conn.execute("SELECT COUNT(*) FROM order_attributes WHERE coalesce(park,'')!=''").fetchone()[0]
    print(f"order_attributes 迁移完成，带园区记录 {n} 条")
    conn.close()
    print("[PASS] nongyixing.db 迁移/重扫/重聚类冒烟\n")
    return tmp


if __name__ == "__main__":
    test_cross_park_same_plate()
    db_copy = test_nongyixing_migration()
    print("nongyixing.db 副本（已迁移）位置：", db_copy)
    print("ALL TESTS PASSED")
