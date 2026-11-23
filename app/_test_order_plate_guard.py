# -*- coding: utf-8 -*-
"""订单归类与聚车车牌一致性防串单回归测试。

覆盖实测农易行临泉辉煌农事服务中心（test_nov.db）暴露的全部五类根因：
1) 黑洞订单防线：未关联抓拍自带明确车牌时，绝不退化走无牌外观通道塞进无牌/无共识订单；
2) 防串单回溯：历史被清洗清空 plate_final 但高置信 lpr_plate/osd_plate 存在的抓拍，
   归类不得盲视车牌冲突；且 associate_orders 绝不清洗未关联散图与高置信(>=0.95)真实车牌；
3) 订单共识归簇：省份不同/1字符差异（如 皖K5713A+皖B5713A）归入同一主簇，
   同时存在两辆不同车（如 4帧A+2帧B）时订单组不作跨车队桥接；
4) 大门同秒连拍与时序防线：assign_gate_orders 不得拆散同秒多机位过磅连拍、
   不得把无牌车尾帧跨牌挪入异牌订单、不得产生出磅早于目标入磅的时序倒挂；
5) 正常无牌图回归：真正无车牌的未关联货车仍按外观+时间正常归类。
"""
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_db


def _emb(angle: float) -> bytes:
    v = np.array([np.cos(angle), np.sin(angle), 0.0, 0.0], dtype=np.float32)
    return v.tobytes()


def _make_db(tmp: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp / "guard.db")
    build_db.ensure_schema(conn)
    return conn


def _add_cap(conn: sqlite3.Connection, tmp: Path, *,
             order_id: str, stage: str, ts: str, filename: str,
             plate_final: str = "", lpr_plate: str | None = None,
             lpr_conf: float = 0.0, osd_plate: str = "",
             plate_observed: str | None = None,
             source: str = "车牌模型",
             reason: str = "",
             vtype: str = "truck", angle: float = 0.0) -> int:
    img = tmp / order_id / filename
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"fake-jpg")
    cur = conn.execute(
        """INSERT INTO captures
           (park, order_id, date, stage, filename, filepath, capture_time,
            plate_final, lpr_plate, lpr_conf, osd_plate, plate_observed,
            recognition_source, plate_match_reason,
            vehicle_type, reid_embedding, capture_status, vehicle_crop_path)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'vehicle', ?)""",
        ("测试园区", order_id, ts[:10], stage, filename, str(img), ts,
         plate_final, lpr_plate, lpr_conf, osd_plate, plate_observed,
         source, reason, vtype, _emb(angle), str(img)),
    )
    return cur.lastrowid


def test_plated_loose_not_sucked_into_no_plate_order() -> None:
    tmp = Path(tempfile.mkdtemp())
    conn = _make_db(tmp)
    # 无牌订单 O_EMPTY（如 109012611040002）与有牌订单 O_A（皖K5713A）
    _add_cap(conn, tmp, order_id="O_EMPTY", stage="入磅",
             ts="2026-11-04 10:43:42", filename="in1.jpg", angle=0.0)
    _add_cap(conn, tmp, order_id="O_EMPTY", stage="出磅",
             ts="2026-11-04 11:14:23", filename="out1.jpg", angle=0.0)
    _add_cap(conn, tmp, order_id="O_A", stage="入磅",
             ts="2026-11-04 10:50:00", filename="a_in.jpg",
             plate_final="皖K5713A", lpr_plate="皖K5713A", lpr_conf=0.99, angle=0.5)
    # 过路带牌车（与 O_EMPTY 外观完全一致 angle=0.0，但有明确车牌 皖KX3609）
    c_plated = _add_cap(conn, tmp, order_id="未关联图片", stage="抓拍",
                        ts="2026-11-04 10:45:00", filename="pass1.jpg",
                        plate_final="皖KX3609", lpr_plate="皖KX3609", lpr_conf=0.99,
                        angle=0.0)
    # 真正无牌的未关联货车（外观与 O_EMPTY 一致 -> 仍应正常归入 O_EMPTY）
    c_unplated = _add_cap(conn, tmp, order_id="未关联图片", stage="抓拍",
                          ts="2026-11-04 10:46:00", filename="unplated.jpg",
                          angle=0.0)
    # 历史被误清 plate_final 但高置信 lpr_plate=皖KM9V08 的散图 -> 不得塞进 O_A 或 O_EMPTY
    c_wiped = _add_cap(conn, tmp, order_id="未关联图片", stage="抓拍",
                       ts="2026-11-04 10:47:00", filename="wiped.jpg",
                       plate_final="", lpr_plate="皖KM9V08", lpr_conf=0.985,
                       plate_observed="皖KM9V08", angle=0.0)
    # 车牌匹配 O_A 的散图 -> 仍应走车牌锚定归入 O_A
    c_match = _add_cap(conn, tmp, order_id="未关联图片", stage="抓拍",
                       ts="2026-11-04 10:55:00", filename="match_a.jpg",
                       plate_final="皖K5713A", lpr_plate="皖K5713A", lpr_conf=0.99,
                       angle=0.4)
    conn.commit()

    build_db.assign_loose_orders(conn, apply_changes=True)
    orders = {cid: conn.execute("SELECT order_id FROM captures WHERE id=?", (cid,)).fetchone()[0]
              for cid in (c_plated, c_unplated, c_wiped, c_match)}
    assert orders[c_plated] == "未关联图片", f"带牌过路车不应被无牌订单吸走: {orders}"
    assert orders[c_wiped] == "未关联图片", f"高置信被清洗车牌不应绕过否决: {orders}"
    assert orders[c_unplated] == "O_EMPTY", f"真正无牌货车应正常归入 O_EMPTY: {orders}"
    assert orders[c_match] == "O_A", f"同牌抓拍应走车牌锚定归入 O_A: {orders}"
    conn.close()
    print("[PASS] 黑洞订单与防串单车牌回溯防线")


def test_order_consensus_no_cross_fleet_bridge_and_restore() -> None:
    tmp = Path(tempfile.mkdtemp())
    conn = _make_db(tmp)
    # 订单 O_MIX 同时拍到主车 皖KM9V08(4帧) 与背景车 皖KS3265(2帧)
    for i in range(4):
        _add_cap(conn, tmp, order_id="O_MIX", stage="入磅",
                 ts=f"2026-11-06 14:55:4{i}", filename=f"m{i}.jpg",
                 plate_final="皖KM9V08", lpr_plate="皖KM9V08", lpr_conf=0.99)
    for i in range(2):
        _add_cap(conn, tmp, order_id="O_MIX", stage="出磅",
                 ts=f"2026-11-06 15:32:5{i}", filename=f"b{i}.jpg",
                 plate_final="皖KS3265", lpr_plate="皖KS3265", lpr_conf=0.999)
    # 另一订单 O_KS 为纯 皖KS3265
    for i in range(3):
        _add_cap(conn, tmp, order_id="O_KS", stage="入磅",
                 ts=f"2026-11-06 16:00:0{i}", filename=f"k{i}.jpg",
                 plate_final="皖KS3265", lpr_plate="皖KS3265", lpr_conf=0.99)
    # 历史被误清洗的未关联散图（即使 reason 已被撤销覆盖为空）
    c_loose = _add_cap(conn, tmp, order_id="未关联图片", stage="抓拍",
                       ts="2026-11-07 15:14:29", filename="loose_km.jpg",
                       plate_final="", lpr_plate="皖KM9V08", lpr_conf=0.985,
                       plate_observed="皖KM9V08", reason="")
    conn.commit()

    build_db.associate_orders(conn)
    vid_mix = conn.execute(
        "SELECT vehicle_id FROM captures WHERE order_id='O_MIX' AND plate_final='皖KM9V08' LIMIT 1"
    ).fetchone()[0]
    vid_ks = conn.execute(
        "SELECT vehicle_id FROM captures WHERE order_id='O_KS' LIMIT 1"
    ).fetchone()[0]
    assert vid_mix is not None and vid_ks is not None and vid_mix != vid_ks, \
        f"混合帧订单不应把 皖KM9V08 与 皖KS3265 桥接进同一 vehicle_id ({vid_mix} vs {vid_ks})"
    # 未关联散图的真实车牌必须恢复且不被清洗
    pf_loose = conn.execute("SELECT plate_final FROM captures WHERE id=?", (c_loose,)).fetchone()[0]
    assert pf_loose == "皖KM9V08", f"未关联散图真实车牌未恢复: {pf_loose!r}"
    # O_MIX 中高置信第二辆车 皖KS3265 应保留车牌并从 O_MIX 主车轨迹拆出
    ks_rows = conn.execute(
        "SELECT plate_final, vehicle_id FROM captures WHERE order_id='O_MIX' AND lpr_plate='皖KS3265'"
    ).fetchall()
    assert all(r[0] == "皖KS3265" and r[1] is None for r in ks_rows), ks_rows
    conn.close()
    print("[PASS] 订单唯一主牌防桥接 + 未关联散图/高置信车牌保护与恢复")


def test_gate_burst_protection_and_timeline_guard() -> None:
    tmp = Path(tempfile.mkdtemp())
    conn = _make_db(tmp)
    # O1 (皖K5713A): 06:51 入磅 ~ 07:25 出磅
    _add_cap(conn, tmp, order_id="O1", stage="入磅", ts="2026-11-14 06:51:35",
             filename="o1_in1.jpg", plate_final="皖K5713A", lpr_plate="皖K5713A",
             lpr_conf=0.99, angle=0.0)
    c_o1_out_unplated = _add_cap(
        conn, tmp, order_id="O1", stage="出磅", ts="2026-11-14 07:25:19",
        filename="o1_out1.jpg", angle=0.0)
    _add_cap(conn, tmp, order_id="O1", stage="出磅", ts="2026-11-14 07:25:19",
             filename="o1_out2.jpg", plate_final="皖K5713A", lpr_plate="皖K5713A",
             lpr_conf=0.99, angle=0.5)
    # O2 (皖KR8210): 09:11 入磅 ~ 09:39 出磅（外观与 c_o1_out_unplated 相同）
    c_o2_in_unplated = _add_cap(
        conn, tmp, order_id="O2", stage="入磅", ts="2026-11-14 09:11:19",
        filename="o2_in1.jpg", angle=0.0)
    _add_cap(conn, tmp, order_id="O2", stage="入磅", ts="2026-11-14 09:11:19",
             filename="o2_in2.jpg", plate_final="皖KR8210", lpr_plate="皖KR8210",
             lpr_conf=0.99, angle=0.1)
    # O3 (鲁WYL887) 出磅 15:06:07 拍到排队车 皖KS3265，而 O4 (皖KS3265) 15:12:19 才入磅
    _add_cap(conn, tmp, order_id="O3", stage="入磅", ts="2026-11-14 14:49:55",
             filename="o3_in.jpg", plate_final="鲁WYL887", lpr_plate="鲁WYL887",
             lpr_conf=0.99, angle=1.0)
    _add_cap(conn, tmp, order_id="O3", stage="出磅", ts="2026-11-14 15:06:07",
             filename="o3_out1.jpg", plate_final="鲁WYL887", lpr_plate="鲁WYL887",
             lpr_conf=0.99, angle=1.0)
    c_o3_out_bg = _add_cap(
        conn, tmp, order_id="O3", stage="出磅", ts="2026-11-14 15:06:07",
        filename="o3_out2.jpg", plate_final="皖KS3265", lpr_plate="皖KS3265",
        lpr_conf=0.999, angle=1.0)
    _add_cap(conn, tmp, order_id="O4", stage="入磅", ts="2026-11-14 15:12:19",
             filename="o4_in.jpg", plate_final="皖KS3265", lpr_plate="皖KS3265",
             lpr_conf=0.99, angle=1.0)
    _add_cap(conn, tmp, order_id="O4", stage="出磅", ts="2026-11-14 15:25:48",
             filename="o4_out.jpg", plate_final="皖KS3265", lpr_plate="皖KS3265",
             lpr_conf=0.99, angle=1.0)
    conn.commit()

    build_db.assign_gate_orders(conn, apply_changes=True)
    assert conn.execute("SELECT order_id FROM captures WHERE id=?",
                        (c_o1_out_unplated,)).fetchone()[0] == "O1", "同秒出磅连拍被拆散"
    assert conn.execute("SELECT order_id FROM captures WHERE id=?",
                        (c_o2_in_unplated,)).fetchone()[0] == "O2", "同秒入磅连拍被拆散"
    assert conn.execute("SELECT order_id FROM captures WHERE id=?",
                        (c_o3_out_bg,)).fetchone()[0] == "O3", "早于目标入磅的出磅帧不应移入产生时序倒挂"
    conn.close()
    print("[PASS] 大门同秒连拍保护 + 跨牌否决 + 行程时序倒挂防线")


def test_black_footer_detection_and_ocr_large_cache() -> None:
    # 1920x1080 明亮画面 + 184px 纯黑底部 OSD 条（含稀疏白字笔画模拟）
    img = np.full((1264, 1920, 3), 180, dtype=np.uint8)
    img[1080:, :] = 0
    img[1100:1130, 50:1500:4] = 255
    img[1150:1180, 50:1600:4] = 255
    img[1200:1230, 50:1200:4] = 255
    y0 = build_db._find_black_footer_y(img)
    assert 1075 <= y0 <= 1085, f"纯黑 OSD 条边界检测异常: y0={y0}"
    # 无纯黑底条时返回 -1
    plain = np.full((1080, 1920, 3), 140, dtype=np.uint8)
    assert build_db._find_black_footer_y(plain) == -1
    # get_ocr(large=True) 连续调用不得退回 _ocr（旧代码 if large and _ocr_large is None 的 bug）
    orig_small, orig_large = build_db._ocr, build_db._ocr_large
    try:
        sentinel_small, sentinel_large = object(), object()
        build_db._ocr, build_db._ocr_large = sentinel_small, sentinel_large
        assert build_db.get_ocr(large=True) is sentinel_large
        assert build_db.get_ocr(large=False) is sentinel_small
    finally:
        build_db._ocr, build_db._ocr_large = orig_small, orig_large
    print("[PASS] 纯黑底部 OSD 边界定位 + get_ocr(large=True) 缓存返回")


def test_fuzzy_plate_matching_rules() -> None:
    # 1) 忽略首个省份汉字 + 兼容首字缺失/未识别
    assert build_db._plate_edit_distance("皖K5713A", "赣K5713A") == 0.0
    assert build_db._plate_edit_distance("皖K5713A", "K5713A") == 0.0
    assert build_db.normalize_plate("K5713A", allow_partial=True) == "K5713A"
    assert build_db.normalize_plate("皖K5713", allow_partial=True) == "皖K5713"
    # 噪声型号不得误当残缺车牌
    for noise in ("DMC01", "DMR02", "J6P", "CA4250", "FAW123"):
        assert build_db.normalize_plate(noise, allow_partial=True) == "", f"噪声被误收: {noise}"

    # 2) 易混淆字符：O<->0、I<->1 成本 0.0；形近对 0.5（两处形近=1.0 仍匹配）
    assert build_db._plate_edit_distance("皖K5713A", "皖K57I3A") == 0.0
    assert build_db._plate_edit_distance("皖KM9U08", "皖KM9UO8") == 0.0
    assert build_db._plate_edit_distance("皖KM9U08", "皖KM9V08") == 0.5
    assert build_db._plate_edit_distance("皖K90063", "皖K9D063") == 0.5
    assert build_db._plate_edit_distance("皖KB162S", "皖K81625") == 1.0

    # 3) 少/多 1 个字符容错（成本 1.0 <= 1.0）
    assert build_db._plate_edit_distance("皖K5713", "皖K5713A") == 1.0
    assert build_db._plate_edit_distance("鲁WL887", "鲁WYL887") == 1.0
    assert build_db._plate_edit_distance("皖KU0P888", "皖KUP888") == 1.0

    # 4) 真正不同车牌必须明确拒绝（> 1.0）
    assert build_db._plate_edit_distance("皖KR5750", "皖K5713A") > 1.0
    assert build_db._plate_edit_distance("皖K11AP5", "皖K13AK5") > 1.0

    # 5) Web 搜索 plate_matches_query（完整模糊、去省、形近、缺位、短片段、单省份）
    assert build_db.plate_matches_query("皖K5713A", "K5713A")
    assert build_db.plate_matches_query("皖K5713A", "鲁K57I3A")
    assert build_db.plate_matches_query("皖K5713A", "皖K5713")
    assert build_db.plate_matches_query("皖KM9U08", "KM9V08")
    assert build_db.plate_matches_query("皖KM9U08", "M9VO")
    assert build_db.plate_matches_query("皖K5713A", "皖")
    assert not build_db.plate_matches_query("皖KR5750", "皖K5713A")
    print("[PASS] 去省份 + 形近混淆 + 缺/多1位车牌模糊匹配规则")


def test_dual_source_lpr_osd_fusion() -> None:
    tmp = Path(tempfile.mkdtemp())
    conn = _make_db(tmp)
    # 两个订单：O_YU (皖KYU963) 和 O_57 (皖K5713A)
    _add_cap(conn, tmp, order_id="O_YU", stage="入磅", ts="2026-11-09 10:00:00",
             filename="yu_in.jpg", plate_final="皖KYU963", lpr_plate="皖KYU963",
             lpr_conf=0.99, angle=0.0)
    _add_cap(conn, tmp, order_id="O_57", stage="入磅", ts="2026-11-09 10:00:00",
             filename="a57_in.jpg", plate_final="皖K5713A", lpr_plate="皖K5713A",
             lpr_conf=0.99, angle=1.0)
    # 场景 A：OSD 误读为 辽A10963，但图像 LPR 高置信 0.974 识别出 皖KYU963 -> 必须走图像车牌归入 O_YU
    c_lpr_win = _add_cap(conn, tmp, order_id="未关联图片", stage="抓拍",
                         ts="2026-11-09 10:15:00", filename="lpr_win.jpg",
                         plate_final="皖KYU963", lpr_plate="皖KYU963", lpr_conf=0.974,
                         osd_plate="辽A10963", angle=0.1)
    # 场景 B：图像 LPR 低置信 0.66 误读 皖A10021，但 OSD 正确读出 皖K5713A -> 应采信 OSD 归入 O_57
    c_osd_win = _add_cap(conn, tmp, order_id="未关联图片", stage="抓拍",
                         ts="2026-11-09 10:16:00", filename="osd_win.jpg",
                         plate_final="皖K5713A", lpr_plate="皖A10021", lpr_conf=0.66,
                         osd_plate="皖K5713A", angle=0.9)
    # 场景 C：图像 LPR 高置信 0.96 读出异牌 皖KR5750，但 OSD 误拍成 皖K5713A -> 高置信 LPR 否决，不得误入 O_57
    c_veto = _add_cap(conn, tmp, order_id="未关联图片", stage="抓拍",
                      ts="2026-11-09 10:17:00", filename="lpr_veto.jpg",
                      plate_final="皖KR5750", lpr_plate="皖KR5750", lpr_conf=0.96,
                      osd_plate="皖K5713A", angle=1.0)
    # 场景 D：无首字+少1位模糊车牌（K5713，仅有图像 LPR 0.88）-> 仍应匹配归入 O_57
    c_fuzzy = _add_cap(conn, tmp, order_id="未关联图片", stage="抓拍",
                       ts="2026-11-09 10:18:00", filename="fuzzy_lpr.jpg",
                       plate_final="K5713", lpr_plate="K5713", lpr_conf=0.88,
                       osd_plate="", angle=1.0)
    # 场景 E：残缺5位车身贴字噪声（PLA70，置信 0.96）不得否决完整 OSD 车牌（皖K5713A）
    c_partial_noise = _add_cap(conn, tmp, order_id="未关联图片", stage="抓拍",
                               ts="2026-11-09 10:19:00", filename="partial_noise.jpg",
                               plate_final="皖K5713A", lpr_plate="PLA70", lpr_conf=0.96,
                               osd_plate="皖K5713A", angle=1.0)
    conn.commit()

    build_db.assign_loose_orders(conn, apply_changes=True)
    orders = {cid: conn.execute("SELECT order_id FROM captures WHERE id=?", (cid,)).fetchone()[0]
              for cid in (c_lpr_win, c_osd_win, c_veto, c_fuzzy, c_partial_noise)}
    assert orders[c_lpr_win] == "O_YU", f"高置信 LPR 应克服误判 OSD 归入 O_YU: {orders}"
    assert orders[c_osd_win] == "O_57", f"低置信 LPR 应让位正确 OSD 归入 O_57: {orders}"
    assert orders[c_veto] == "未关联图片", f"高置信异牌 LPR 必须否决误判 OSD: {orders}"
    assert orders[c_fuzzy] == "O_57", f"无首字+少1位车牌应归入 O_57: {orders}"
    assert orders[c_partial_noise] == "O_57", f"残缺5位 LPR 噪声不得否决完整 OSD: {orders}"
    conn.close()
    print("[PASS] 图像 LPR 与底部 OSD 双源互证/否决 + 模糊车牌归类")


def test_trim_osd_footer_in_lpr() -> None:
    import reid_pipeline

    # 裁剪图底部带 140px 纯黑 OSD 条
    crop = np.full((600, 800, 3), 160, dtype=np.uint8)
    crop[460:, :] = 0
    crop[490:515, 20:700:4] = 255
    trimmed = reid_pipeline.PlateLPR._trim_osd_footer(crop)
    assert 450 <= trimmed.shape[0] <= 470, f"黑底 OSD 条未正确裁除: {trimmed.shape}"
    # 普通无黑条裁剪图保持原高度
    normal = np.full((600, 800, 3), 120, dtype=np.uint8)
    assert reid_pipeline.PlateLPR._trim_osd_footer(normal).shape[0] == 600
    print("[PASS] PlateLPR 车辆裁剪图底部黑条 OSD 裁除")


def test_choose_main_and_plate_promote_crop_sync() -> None:
    import json
    import reid_pipeline

    # 1) 真实 #462 场景 (2560x1440 出磅)：边缘黑轿车 0.93 vs 地磅大货车 0.55 -> 必须选大货车
    dets_462 = [
        {"score": 0.93, "box": [1873, 737, 2323, 1093]},  # 边缘轿车
        {"score": 0.55, "box": [898, 248, 1361, 860]},    # 地磅中心货车
    ]
    attrs_462 = [{"type": "sedan", "color": "black"}, {"type": "truck", "color": "gray"}]
    chosen_462 = reid_pipeline.PaddleVehicleCropper.choose_main(
        dets_462, 2560, 1440, attrs=attrs_462, stage="出磅"
    )
    assert chosen_462 is dets_462[1], f"#462 应选地磅大货车而非边缘轿车: {chosen_462}"

    # 2) 真实 #1917 场景 (2560x1440 入磅)：远景小黑轿车 0.95 vs 大红货车 0.50 -> 必须选大红货车
    dets_1917 = [
        {"score": 0.95, "box": [1264, 489, 1581, 704]},
        {"score": 0.50, "box": [1439, 236, 2295, 640]},
    ]
    attrs_1917 = [{"type": "sedan", "color": "black"}, {"type": "truck", "color": "red"}]
    chosen_1917 = reid_pipeline.PaddleVehicleCropper.choose_main(
        dets_1917, 2560, 1440, attrs=attrs_1917, stage="入磅"
    )
    assert chosen_1917 is dets_1917[1], f"#1917 应选大货车而非小轿车: {chosen_1917}"

    # 3) 订单车牌自动主车提升 + 安全对调两张裁剪图（绝不覆盖成两张相同缩略图）
    tmp = Path(tempfile.mkdtemp())
    db_path = tmp / "promote.db"
    conn = sqlite3.connect(db_path)
    build_db.ensure_schema(conn)
    c_main = tmp / "crops" / "00000001.jpg"
    c_inst = tmp / "crops" / "1_01.jpg"
    c_main.parent.mkdir(parents=True, exist_ok=True)
    c_main.write_bytes(b"OLD-SEDAN-PIXELS")
    c_inst.write_bytes(b"REAL-TRUCK-PIXELS")
    # 同单已有 2 帧确认主车为 皖K5713A / truck
    for idx in (2, 3):
        _add_cap(conn, tmp, order_id="O_57", stage="入磅", ts=f"2026-11-09 10:00:0{idx}",
                 filename=f"ref{idx}.jpg", plate_final="皖K5713A", lpr_plate="皖K5713A",
                 lpr_conf=0.99, vtype="truck")
    # 抓拍 #1 主车误选成黑轿车 皖K70110，车1 才是大货车 皖H5713A
    cid1 = _add_cap(conn, tmp, order_id="O_57", stage="出磅", ts="2026-11-09 10:20:00",
                    filename="p1.jpg", plate_final="皖K70110", lpr_plate="皖K70110",
                    lpr_conf=0.93, vtype="sedan")
    conn.execute("UPDATE captures SET vehicle_crop_path=?, vehicle_box=? WHERE id=?",
                 ("crops/00000001.jpg", json.dumps([1873, 737, 2323, 1093]), cid1))
    conn.execute(
        """INSERT INTO vehicle_instances
           (capture_id, instance_index, crop_path, vehicle_box, vehicle_det_score,
            vehicle_color, vehicle_type, lpr_plate, lpr_conf)
           VALUES (?, 0, 'crops/00000001.jpg', ?, 0.93, 'black', 'sedan', '皖K70110', 0.93)""",
        (cid1, json.dumps([1873, 737, 2323, 1093])),
    )
    conn.execute(
        """INSERT INTO vehicle_instances
           (capture_id, instance_index, crop_path, vehicle_box, vehicle_det_score,
            vehicle_color, vehicle_type, lpr_plate, lpr_conf)
           VALUES (?, 1, 'crops/1_01.jpg', ?, 0.55, 'gray', 'truck', '皖H5713A', 0.92)""",
        (cid1, json.dumps([898, 248, 1361, 860])),
    )
    conn.commit()
    conn.close()

    res = reid_pipeline.promote_by_order_plate(db_path)
    assert res["promoted"] == 1, f"应自动切换主车为命中订单车牌的实例: {res}"
    # 磁盘两张图必须对调而非覆盖成相同文件
    assert c_main.read_bytes() == b"REAL-TRUCK-PIXELS"
    assert c_inst.read_bytes() == b"OLD-SEDAN-PIXELS"
    conn = sqlite3.connect(db_path)
    cap_row = conn.execute(
        "SELECT vehicle_box, vehicle_type, lpr_plate, plate_final FROM captures WHERE id=?", (cid1,)
    ).fetchone()
    inst0 = conn.execute(
        "SELECT crop_path, vehicle_box, vehicle_type, lpr_plate FROM vehicle_instances WHERE capture_id=? AND instance_index=0",
        (cid1,),
    ).fetchone()
    inst1 = conn.execute(
        "SELECT crop_path, vehicle_box, vehicle_type, lpr_plate FROM vehicle_instances WHERE capture_id=? AND instance_index=1",
        (cid1,),
    ).fetchone()
    conn.close()
    assert cap_row == (json.dumps([898, 248, 1361, 860]), "truck", "皖H5713A", "皖H5713A"), cap_row
    assert inst0 == ("crops/00000001.jpg", json.dumps([898, 248, 1361, 860]), "truck", "皖H5713A"), inst0
    assert inst1 == ("crops/1_01.jpg", json.dumps([1873, 737, 2323, 1093]), "sedan", "皖K70110"), inst1
    print("[PASS] 主车几何+车型选择优化 + 订单车牌自动切主车 + 裁剪图安全对调同步")


def main() -> None:
    test_plated_loose_not_sucked_into_no_plate_order()
    test_order_consensus_no_cross_fleet_bridge_and_restore()
    test_gate_burst_protection_and_timeline_guard()
    test_black_footer_detection_and_ocr_large_cache()
    test_fuzzy_plate_matching_rules()
    test_dual_source_lpr_osd_fusion()
    test_trim_osd_footer_in_lpr()
    test_choose_main_and_plate_promote_crop_sync()
    print("ALL ORDER PLATE GUARD TESTS PASSED")


if __name__ == "__main__":
    main()
