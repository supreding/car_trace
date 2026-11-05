# -*- coding: utf-8 -*-
"""共用大门跨园区去重验证：
1) 新版命名（末尾唯一id一致）直接判重，id 不同绝不合并（字节相同也不并）；
2) 旧命名（无id）必须字节一致（MD5）才判重——同秒不同车字节不同不合并，
   文件不在盘无法确认跳过，截断uuid撞号同理；
3) 仲裁：真实订单 > 车牌锚定 > 行程时间线 > ... > 编号（强制选边），
   全是未关联图片保留两边记待人工；
4) 落败副本移入 重复抓拍隔离-<原园区>，全盘重扫幂等不复活，--undo 可撤销；
5) 隔离园区被 assign_gate / associate_orders 排除。
"""
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_db

MONTH = "2026年11月份抓拍照片"
DAY = "11月5日"
GATE_NO_UID = "出磅2026年11月5日08：15：59.jpg"
ORDER_A = "109002611050001"
ORDER_B = "109002611050002"


def make_db(tmp: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp / "t.db")
    build_db.ensure_schema(conn)
    return conn


def make_jpg(path: Path, color=(120, 140, 160)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color).save(path, "JPEG")


def copy_jpg(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def gate_pair(tmp: Path, *, uid: str = "", identical: bool = True):
    """两个园区同秒的出磅照片：返回 (root, 园区A路径, 园区B路径)。"""
    root = tmp / "data"
    name = f"出磅2026年11月5日08：15：59{uid}.jpg"
    fa = root / "园区A" / MONTH / DAY / ORDER_A / name
    fb = root / "园区B" / MONTH / DAY / ORDER_B / name
    make_jpg(fa, color=(120, 140, 160))
    if identical:
        copy_jpg(fa, fb)
    else:
        make_jpg(fb, color=(10, 200, 30))
    return root, fa, fb


def cid_of(conn, park: str, filename: str) -> int:
    row = conn.execute("SELECT id FROM captures WHERE park=? AND filename=?",
                       (park, filename)).fetchone()
    assert row, f"缺少抓拍行：{park}/{filename}"
    return row[0]


def set_capture(conn, cid: int, *, order_id=None, capture_time=None,
                plate=None, emb=None) -> None:
    fields, params = [], []
    if order_id is not None:
        fields.append("order_id=?")
        params.append(order_id)
    if capture_time is not None:
        fields.append("capture_time=?")
        params.append(capture_time)
    if plate is not None:
        fields.append("plate_final=?")
        params.append(plate)
    if emb is not None:
        fields.append("reid_embedding=?")
        params.append(np.asarray(emb, dtype=np.float32).tobytes())
    if fields:
        conn.execute(f"UPDATE captures SET {', '.join(fields)} WHERE id=?",
                     (*params, cid))


def test_a_class_uid_rules():
    """唯一id规则：id 全等直接判重（不做内容校验）；id 不同绝不合并。"""
    tmp = Path(tempfile.mkdtemp())
    conn = make_db(tmp)
    # id 相同但字节不同（模拟拷贝被转码）：仍须判重——上游保证 id 即事件
    root, fa, fb = gate_pair(tmp, uid="_2346200", identical=False)
    build_db.sync_files(conn, root)
    cid_a = cid_of(conn, "园区A", fa.name)
    cid_b = cid_of(conn, "园区B", fb.name)
    report = build_db.dedup_gate_captures(conn, apply_changes=True, root=root)
    assert report["dup_groups"] == 1 and report["moved"] == 1, report
    log = conn.execute(
        "SELECT capture_id, winner_id, confirm FROM gate_dedup_log").fetchall()
    assert log == [(cid_b, cid_a, "filename_id")], log
    assert not fb.is_file()
    assert (root / "重复抓拍隔离-园区B" / MONTH / DAY / ORDER_B / fb.name).is_file()
    conn.close()

    # id 不同（同秒两辆车）但字节相同：id 优先于内容，绝不合并
    tmp2 = Path(tempfile.mkdtemp())
    conn2 = make_db(tmp2)
    root2 = tmp2 / "data"
    name1 = "出磅2026年11月5日08：15：59_2346200.jpg"
    name2 = "出磅2026年11月5日08：15：59_2346201.jpg"
    fa2 = root2 / "园区A" / MONTH / DAY / ORDER_A / name1
    fb2 = root2 / "园区B" / MONTH / DAY / ORDER_B / name2
    make_jpg(fa2)
    copy_jpg(fa2, fb2)
    build_db.sync_files(conn2, root2)
    report = build_db.dedup_gate_captures(conn2, apply_changes=True, root=root2)
    assert report["dup_groups"] == 0 and report["moved"] == 0, report
    assert fb2.is_file() and fa2.is_file()
    assert conn2.execute("SELECT COUNT(*) FROM gate_dedup_log").fetchone()[0] == 0
    conn2.close()
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(tmp2, ignore_errors=True)
    print("[PASS] A类唯一id规则（id等判重/id异不并）")


def test_b_class_requires_identical_bytes():
    """旧命名（无id）：字节一致才判重；字节不同（同秒不同车）不合并。"""
    tmp = Path(tempfile.mkdtemp())
    conn = make_db(tmp)
    root, fa, fb = gate_pair(tmp, identical=False)
    build_db.sync_files(conn, root)
    cid_a = cid_of(conn, "园区A", fa.name)
    cid_b = cid_of(conn, "园区B", fb.name)
    # 空值安全：时间/车牌/特征全 NULL 也必须正常完成仲裁前的一切步骤
    report = build_db.dedup_gate_captures(conn, apply_changes=True, root=root)
    assert report["moved"] == 0 and report["skipped_md5_mismatch"] == 1, report
    assert conn.execute("SELECT park FROM captures WHERE id=?", (cid_b,)).fetchone()[0] == "园区B"
    assert conn.execute("SELECT COUNT(*) FROM gate_dedup_log").fetchone()[0] == 0
    assert fb.is_file() and fa.is_file()

    # 字节一致：判重；证据全平（无成员无牌）按编号强制选边（小 id 胜）
    tmp2 = Path(tempfile.mkdtemp())
    conn2 = make_db(tmp2)
    root2, fa2, fb2 = gate_pair(tmp2, identical=True)
    build_db.sync_files(conn2, root2)
    cid_a2 = cid_of(conn2, "园区A", fa2.name)
    cid_b2 = cid_of(conn2, "园区B", fb2.name)
    report = build_db.dedup_gate_captures(conn2, apply_changes=True, root=root2)
    assert report["dup_groups"] == 1 and report["moved"] == 1, report
    log = conn2.execute(
        "SELECT capture_id, winner_id, confirm FROM gate_dedup_log").fetchall()
    winner = min(cid_a2, cid_b2)
    loser = max(cid_a2, cid_b2)
    assert log == [(loser, winner, "content_md5")], log
    conn.close()
    conn2.close()
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(tmp2, ignore_errors=True)
    print("[PASS] B类内容确认（字节同判重/字节异不并/空值安全）")


def test_b_class_missing_file_skipped():
    """旧命名任一文件不在盘：无法确认，整组跳过，不合并不报错。"""
    tmp = Path(tempfile.mkdtemp())
    conn = make_db(tmp)
    root, fa, fb = gate_pair(tmp, identical=True)
    build_db.sync_files(conn, root)
    fb.unlink()  # 模拟归档已清理的园区（库记录还在）
    cid_b = cid_of(conn, "园区B", fb.name)
    report = build_db.dedup_gate_captures(conn, apply_changes=True, root=root)
    assert report["moved"] == 0 and report["skipped_no_file"] == 1, report
    assert conn.execute("SELECT park FROM captures WHERE id=?", (cid_b,)).fetchone()[0] == "园区B"
    assert conn.execute("SELECT COUNT(*) FROM gate_dedup_log").fetchone()[0] == 0
    conn.close()
    shutil.rmtree(tmp, ignore_errors=True)
    print("[PASS] B类文件不在盘跳过")


def test_uuid_collision():
    """老命名 <阶段>_<截断uuid>：uuid 撞号字节不同不并；字节相同判重。"""
    tmp = Path(tempfile.mkdtemp())
    conn = make_db(tmp)
    root = tmp / "data"
    # 撞号 + 字节不同
    name = "入磅_467f1888-f8a.jpg"
    fa = root / "园区A" / MONTH / DAY / ORDER_A / name
    fb = root / "园区B" / MONTH / DAY / ORDER_B / name
    make_jpg(fa, color=(1, 2, 3))
    make_jpg(fb, color=(4, 5, 6))
    build_db.sync_files(conn, root)
    report = build_db.dedup_gate_captures(conn, apply_changes=True, root=root)
    assert report["moved"] == 0 and report["skipped_md5_mismatch"] == 1, report
    assert fb.is_file() and fa.is_file()
    conn.close()

    # uuid 同 + 字节同
    tmp2 = Path(tempfile.mkdtemp())
    conn2 = make_db(tmp2)
    root2, fa2, fb2 = None, None, None
    root2 = tmp2 / "data"
    name2 = "入磅_aabbccdd-01.jpg"
    fa2 = root2 / "园区A" / MONTH / DAY / ORDER_A / name2
    fb2 = root2 / "园区B" / MONTH / DAY / ORDER_B / name2
    make_jpg(fa2)
    copy_jpg(fa2, fb2)
    build_db.sync_files(conn2, root2)
    report = build_db.dedup_gate_captures(conn2, apply_changes=True, root=root2)
    assert report["dup_groups"] == 1 and report["moved"] == 1, report
    conn.close()
    conn2.close()
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(tmp2, ignore_errors=True)
    print("[PASS] 老uuid命名（撞号不并/字节同判重）")


def test_cross_park_timeline_wins_and_rescan_idempotent():
    """核心场景：车牌锚定打平时，行程时间线定胜负；重扫幂等；可撤销。"""
    tmp = Path(tempfile.mkdtemp())
    conn = make_db(tmp)
    root, fa, fb = gate_pair(tmp, identical=True)
    # A 订单：入磅 07:00（同牌）-> 出磅 08:15 行程顺
    make_jpg(root / "园区A" / MONTH / DAY / ORDER_A / "入磅2026年11月5日07：00：00.jpg",
             color=(50, 60, 70))
    # B 订单：入磅在 09:30（出磅之后）-> 这张 08:15 的出磅行程逆
    make_jpg(root / "园区B" / MONTH / DAY / ORDER_B / "入磅2026年11月5日09：30：00.jpg",
             color=(80, 90, 100))
    build_db.sync_files(conn, root)
    before = conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0]
    plate = "皖K12345"
    cid_a = cid_of(conn, "园区A", fa.name)
    cid_b = cid_of(conn, "园区B", fb.name)
    set_capture(conn, cid_of(conn, "园区A", "入磅2026年11月5日07：00：00.jpg"),
                order_id=ORDER_A, capture_time="2026-11-05 07:00:00", plate=plate)
    set_capture(conn, cid_of(conn, "园区B", "入磅2026年11月5日09：30：00.jpg"),
                order_id=ORDER_B, capture_time="2026-11-05 09:30:00", plate=plate)
    set_capture(conn, cid_a, order_id=ORDER_A,
                capture_time="2026-11-05 08:15:59", plate=plate)
    set_capture(conn, cid_b, order_id=ORDER_B,
                capture_time="2026-11-05 08:15:59", plate=plate)
    conn.commit()

    # 试运行：只报告，不动库不动文件
    report = build_db.dedup_gate_captures(conn, apply_changes=False, root=root)
    assert report["duplicates"] == 1 and report["moved"] == 0, report
    assert conn.execute("SELECT park FROM captures WHERE id=?", (cid_b,)).fetchone()[0] == "园区B"
    assert fb.is_file()

    # apply：赢家留 A，落败副本进隔离园区
    report = build_db.dedup_gate_captures(conn, apply_changes=True, root=root)
    assert report["moved"] == 1 and report["failed"] == 0, report
    assert conn.execute("SELECT park, order_id FROM captures WHERE id=?",
                        (cid_a,)).fetchone() == ("园区A", ORDER_A)
    row = conn.execute("SELECT park, order_id, filepath FROM captures WHERE id=?",
                       (cid_b,)).fetchone()
    assert row[0] == "重复抓拍隔离-园区B" and row[1] == ORDER_B, row
    assert not fb.is_file()
    assert (root / "重复抓拍隔离-园区B" / MONTH / DAY / ORDER_B / fb.name).is_file()
    log = conn.execute(
        "SELECT capture_id, winner_id, from_park, to_park FROM gate_dedup_log"
    ).fetchall()
    assert log == [(cid_b, cid_a, "园区B", "重复抓拍隔离-园区B")], log

    # 全盘重扫幂等：隔离文件按新键命中同一行，不复活不增行
    build_db.sync_files(conn, root)
    assert conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0] == before
    row = conn.execute("SELECT park, order_id, filename FROM captures WHERE id=?",
                       (cid_b,)).fetchone()
    assert row[0] == "重复抓拍隔离-园区B" and row[1] == ORDER_B and row[2] == fb.name
    # 再跑去重：赢家独自成组，无新增动作
    report = build_db.dedup_gate_captures(conn, apply_changes=True, root=root)
    assert report["moved"] == 0 and report["dup_groups"] == 0, report

    # 撤销：文件移回原园区目录、字段还原、日志清空
    assert build_db.undo_gate_dedup(conn) == 1
    assert fb.is_file() and not (root / "重复抓拍隔离-园区B" / MONTH / DAY
                                 / ORDER_B / fb.name).exists()
    row = conn.execute("SELECT park, order_id FROM captures WHERE id=?",
                       (cid_b,)).fetchone()
    assert row == ("园区B", ORDER_B), row
    assert conn.execute("SELECT COUNT(*) FROM gate_dedup_log").fetchone()[0] == 0
    conn.close()
    shutil.rmtree(tmp, ignore_errors=True)
    print("[PASS] 跨园区时间线仲裁 + 重扫幂等 + 撤销")


def test_all_unassociated_needs_review():
    """两份都在未关联图片：无正确订单可归属，保留两边记待人工。"""
    tmp = Path(tempfile.mkdtemp())
    conn = make_db(tmp)
    root = tmp / "data"
    fa = root / "园区A" / MONTH / DAY / "未关联图片" / GATE_NO_UID
    fb = root / "园区B" / MONTH / DAY / "未关联图片" / GATE_NO_UID
    make_jpg(fa)
    copy_jpg(fa, fb)
    build_db.sync_files(conn, root)
    report = build_db.dedup_gate_captures(conn, apply_changes=True, root=root)
    assert report["needs_review_groups"] == 1 and report["moved"] == 0, report
    assert fb.is_file() and fa.is_file()
    assert conn.execute("SELECT COUNT(*) FROM gate_dedup_log").fetchone()[0] == 0
    conn.close()
    shutil.rmtree(tmp, ignore_errors=True)
    print("[PASS] 全未关联组保留两边待人工")


def test_quarantine_excluded_from_assign_and_associate():
    """隔离园区被 assign_gate / associate_orders 排除：不重排、不聚车。"""
    tmp = Path(tempfile.mkdtemp())
    conn = make_db(tmp)
    root, fa, fb = gate_pair(tmp, uid="_2346200", identical=True)
    build_db.sync_files(conn, root)
    cid_a = cid_of(conn, "园区A", fa.name)
    cid_b = cid_of(conn, "园区B", fb.name)
    set_capture(conn, cid_a, order_id=ORDER_A, capture_time="2026-11-05 08:15:59",
                plate="皖K12345", emb=[1, 0, 0, 0])
    set_capture(conn, cid_b, order_id=ORDER_B, capture_time="2026-11-05 08:15:59",
                plate="皖K12345", emb=[1, 0, 0, 0])
    conn.commit()
    report = build_db.dedup_gate_captures(conn, apply_changes=True, root=root)
    assert report["moved"] == 1, report

    # assign_gate 不得触碰隔离园区（赢家自身无锚保留现归属，也不产生日志）
    build_db.assign_gate_orders(conn, apply_changes=True)
    assert conn.execute("SELECT COUNT(*) FROM loose_assignments").fetchone()[0] == 0
    assert conn.execute("SELECT park FROM captures WHERE id=?",
                        (cid_b,)).fetchone()[0] == "重复抓拍隔离-园区B"

    # 车辆重聚：赢家随订单成组，隔离区行 vehicle_id 保持 NULL
    build_db.associate_orders(conn)
    winner_vid = conn.execute("SELECT vehicle_id FROM captures WHERE id=?",
                              (cid_a,)).fetchone()[0]
    loser_vid = conn.execute("SELECT vehicle_id FROM captures WHERE id=?",
                             (cid_b,)).fetchone()[0]
    assert winner_vid is not None and loser_vid is None, (winner_vid, loser_vid)
    conn.close()
    shutil.rmtree(tmp, ignore_errors=True)
    print("[PASS] 隔离园区被归类/重聚排除")


if __name__ == "__main__":
    test_a_class_uid_rules()
    test_b_class_requires_identical_bytes()
    test_b_class_missing_file_skipped()
    test_uuid_collision()
    test_cross_park_timeline_wins_and_rescan_idempotent()
    test_all_unassociated_needs_review()
    test_quarantine_excluded_from_assign_and_associate()
    print("ALL TESTS PASSED")
