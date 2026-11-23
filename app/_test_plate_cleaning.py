# -*- coding: utf-8 -*-
"""逐帧车牌清洗验证（associate_orders 第6步）：
1) 双重印证（图像模型与 OSD 独立读出同一牌）不清——同框第二辆车的真实车牌；
2) 组内重复出现（同一字符串 >=2 帧一致）不清——一致的少数派读数是真实车辆；
3) 单帧孤立噪声读数照常清洗（原始防幻觉目的），原文留 plate_observed 可审计。
"""
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_db

MONTH = "2026年11月份抓拍照片"
DAY = "11月20日"
ORDER = "109002611220001"
MAIN_PLATE = "皖KS3265"
SECOND_PLATE = "皖KB5A12"
NOISE_PLATE = "皖KZ9999"


def make_db(tmp: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp / "t.db")
    build_db.ensure_schema(conn)
    return conn


def add_capture(conn, *, seq: int, plate: str, source: str) -> int:
    cur = conn.execute(
        """INSERT INTO captures
           (park, order_id, date, stage, filename, filepath, capture_time,
            plate_final, recognition_source, capture_status, vehicle_crop_path)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'vehicle', ?)""",
        ("测试园区", ORDER, "2026-11-20", "抓拍",
         f"抓拍2026年11月20日10：0{seq}:00_{1000 + seq}.jpg",
         f"data/测试园区/{MONTH}/{DAY}/未关联图片/抓拍{seq}.jpg",
         f"2026-11-20 10:0{seq}:00",
         plate, source,
         f"data/测试园区/{MONTH}/{DAY}/未关联图片/crop{seq}.jpg"),
    )
    return cur.lastrowid


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    conn = make_db(tmp)
    # 主车 6 帧 + 同框第二辆车（双重印证 2 帧 + 模型单独读出但组内重复 2 帧）
    # + 单帧孤立噪声 1 帧；另预置 1 帧"旧规则已清洗"的双重印证帧（验证恢复）
    for seq in (1, 2, 3, 10, 11, 12):
        add_capture(conn, seq=seq, plate=MAIN_PLATE, source="车牌模型")
    for seq in (4, 5):
        add_capture(conn, seq=seq, plate=SECOND_PLATE, source="OSD+车牌模型一致")
    for seq in (6, 7):
        add_capture(conn, seq=seq, plate=SECOND_PLATE, source="车牌模型")
    add_capture(conn, seq=8, plate=NOISE_PLATE, source="车牌模型")
    add_capture(conn, seq=9, plate=SECOND_PLATE, source="OSD+车牌模型一致")
    conn.execute(
        """UPDATE captures SET plate_final='', plate_observed=?,
                  plate_match_reason='单帧车牌与车辆主牌差异过大，判为误读'
           WHERE filename LIKE '%1009.jpg'""",
        (SECOND_PLATE,))
    conn.commit()

    report = build_db.associate_orders(conn)
    conn.commit()

    def plate_of(seq: int) -> str:
        return conn.execute(
            "SELECT plate_final FROM captures WHERE filename LIKE ?",
            (f"%{1000 + seq}.jpg",)).fetchone()[0]

    # 主车读数保留
    assert plate_of(1) == MAIN_PLATE and plate_of(3) == MAIN_PLATE
    # 双重印证：不清
    assert plate_of(4) == SECOND_PLATE and plate_of(5) == SECOND_PLATE, \
        "图像模型与OSD互相印证的真实车牌被误清"
    # 组内重复出现：不清
    assert plate_of(6) == SECOND_PLATE and plate_of(7) == SECOND_PLATE, \
        "组内多帧一致的少数派读数被误清"
    # 历史被清洗的双重印证帧：恢复
    assert plate_of(9) == SECOND_PLATE, "旧规则误清的真实车牌未被恢复"
    assert report["frame_misread_restored"] == 1, report
    # 单帧孤立噪声：清洗，原文留 plate_observed
    row = conn.execute(
        """SELECT plate_final, plate_observed, plate_match_reason
           FROM captures WHERE filename LIKE '%1008.jpg'""").fetchone()
    assert row[0] == "" and row[1] == NOISE_PLATE, row
    assert "误读" in (row[2] or ""), row
    assert report["frame_misread_cleaned"] == 1, report
    # 车牌一致性拆分：双重印证/组内重复的第二辆车帧拆出主车轨迹（保留车牌）
    for seq in (4, 5, 6, 7, 9):
        vid, reason = conn.execute(
            """SELECT vehicle_id, plate_match_reason FROM captures
               WHERE filename LIKE ?""", (f"%{1000 + seq}.jpg",)).fetchone()
        assert vid is None, f"#{1000 + seq} 未拆出主车轨迹"
        assert "车牌与车辆主牌不符" in (reason or ""), (seq, reason)
    assert report["plate_split"] == 5, report
    # 主车帧与噪声帧（无牌）留在组内
    for seq in (1, 2, 3, 8):
        vid = conn.execute(
            "SELECT vehicle_id FROM captures WHERE filename LIKE ?",
            (f"%{1000 + seq}.jpg",)).fetchone()[0]
        assert vid is not None, f"#{1000 + seq} 不应被拆出"
    conn.close()
    print("[PASS] 车牌清洗与车牌拆分（真实第二辆车拆出待人工，噪声照常清洗）")


if __name__ == "__main__":
    main()
