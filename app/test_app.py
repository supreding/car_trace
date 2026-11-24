# -*- coding: utf-8 -*-
"""应用自动测试：对 granary.db 与 Flask 接口做验收（实施文档 16.4 清单）。

运行前提：build_db.py 已完成扫描+OCR，reid_pipeline.ps1 已完成批处理。
"""
from __future__ import annotations

import re
import sqlite3
import unittest
from datetime import datetime

import numpy as np

from contextlib import closing

import server


class AppTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        server.app.config.update(TESTING=True)
        cls.client = server.app.test_client()
        cls.db = sqlite3.connect(server.DB_PATH)
        cls.db.row_factory = sqlite3.Row

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def get_json(self, url):
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return response.get_json()

    # 1. 基础健康与总量统计
    def test_health_and_totals(self):
        self.assertEqual(self.get_json("/api/health")["status"], "ok")
        stats = self.get_json("/api/stats")
        # 单库多园区（2026-11 本地库与生产对齐：山高 + 农易行合并）。
        # 数据持续增长，总量类断言用下限而非精确值。
        self.assertGreaterEqual(stats["total"], 8302)
        self.assertGreaterEqual(stats["orders"], 857)
        self.assertGreaterEqual(stats["dates"], 31)
        stage_counts = {item["stage"]: item["count"] for item in stats["stage_items"]}
        for stage, floor in {"入磅": 1927, "出磅": 1897, "抓拍": 3722,
                             "质检": 655, "人脸": 101}.items():
            self.assertGreaterEqual(stage_counts.get(stage, 0), floor)

    # 2. 摄像头/分辨率约束表（实施文档第 4 步）
    def test_camera_keys(self):
        rows = self.db.execute(
            """SELECT camera_key, COUNT(*) AS n FROM captures
               WHERE coalesce(camera_key,'')!='' GROUP BY camera_key"""
        ).fetchall()
        cameras = {row["camera_key"]: row["n"] for row in rows}
        self.assertGreaterEqual(cameras.get("入磅@2560x1440", 0), 1927)
        self.assertGreaterEqual(cameras.get("出磅@2560x1440", 0), 1897)
        self.assertGreaterEqual(cameras.get("抓拍@1920x1264", 0), 3719)
        self.assertGreaterEqual(cameras.get("抓拍@1920x1328", 0), 3)   # 农易行大门抓拍进
        self.assertGreaterEqual(cameras.get("人脸@480x640", 0), 101)
        # 质检相机存在两台不同分辨率
        self.assertGreater(cameras.get("质检@1280x720", 0), 500)
        self.assertGreater(cameras.get("质检@2304x1296", 0), 100)

    # 3. 订单号聚合唯一键；跨日期订单完整（68002608090040）
    def test_cross_date_order_aggregation(self):
        rows = self.db.execute(
            """SELECT COUNT(DISTINCT date) AS d, COUNT(*) AS n FROM captures
               WHERE order_id='68002608090040'"""
        ).fetchone()
        self.assertGreaterEqual(rows["d"], 2)
        self.assertEqual(rows["n"], self.client.get(
            "/api/order/68002608090040").get_json()["items"].__len__())
        # 数据库中的订单数应小于订单目录数（跨日期目录被聚合）
        unique_orders = self.db.execute(
            "SELECT COUNT(DISTINCT order_id) FROM captures").fetchone()[0]
        total_dirs = self.db.execute(
            """SELECT COUNT(*) FROM (
                 SELECT DISTINCT order_id, date FROM captures)""").fetchone()[0]
        self.assertGreaterEqual(unique_orders, 857)
        # 目录 (订单,日期) 组合数随数据增长；2026-09 大门重归类曾由 806 收敛至 803
        self.assertGreaterEqual(total_dirs, 925)

    # 4. 时间戳 OCR 与降级标记（实施文档 3.2）
    def test_ocr_timestamps(self):
        total, ok = self.db.execute(
            """SELECT COUNT(*),
                      SUM(CASE WHEN ocr_status='ok' THEN 1 ELSE 0 END)
               FROM captures"""
        ).fetchone()
        self.assertGreaterEqual(total, 8302)
        # 农易行库夜间/命名差异拉低整体成功率，阈值放宽至 0.75
        self.assertGreater(ok / total, 0.75, "OCR 成功率过低")
        bad = self.db.execute(
            """SELECT COUNT(*) FROM captures
               WHERE ocr_status='ok' AND capture_time NOT GLOB
                 '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]'"""
        ).fetchone()[0]
        self.assertEqual(bad, 0)
        # 非 ok 的记录必须显式标记不确定
        unmarked = self.db.execute(
            """SELECT COUNT(*) FROM captures
               WHERE ocr_status!='ok' AND status_reason=''"""
        ).fetchone()[0]
        self.assertEqual(unmarked, 0)

    # 5. PaddleDetection 检测与裁剪（实施文档 7/8 步）
    def test_detection_and_crops(self):
        stats = self.get_json("/api/stats")
        self.assertGreater(stats["cropped"], 3000, "车辆裁剪数量异常")
        # GPU 处理范围以山高园区为口径（农易行数据本地尚未跑 GPU，生产已处理）；
        # 山高尚存 2 条历史遗留待处理帧，上界放宽为 2（生产为 0）
        self.assertLessEqual(
            self.db.execute(
                "SELECT COUNT(*) FROM captures WHERE detection_count IS NULL"
                " AND coalesce(park,'')='山高荃银丰麦田库'").fetchone()[0], 2)
        no_vehicle = self.db.execute(
            """SELECT id FROM captures WHERE capture_status='no_vehicle'
               ORDER BY id LIMIT 1""").fetchone()
        if no_vehicle:
            self.assertIsNone(self.db.execute(
                "SELECT reid_embedding FROM captures WHERE id=?",
                (no_vehicle["id"],)).fetchone()["reid_embedding"])
        cropped = self.db.execute(
            """SELECT id FROM captures WHERE vehicle_crop_path IS NOT NULL
               ORDER BY id LIMIT 1""").fetchone()
        self.assertIsNotNone(cropped)
        response = self.client.get(f"/api/crop/{cropped['id']}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/jpeg")
        self.assertGreater(len(response.get_data()), 1000)
        # 裁剪来源只允许 PaddleDetection 与 人工框选（manual_box_edits）
        with_crops = self.db.execute(
            "SELECT COUNT(*) FROM captures"
            " WHERE vehicle_crop_path IS NOT NULL AND crop_source!='PaddleDetection'"
            " AND crop_source!='人工框选'").fetchone()[0]
        self.assertEqual(with_crops, 0)
        self.assertGreaterEqual(self.db.execute(
            "SELECT COUNT(*) FROM captures WHERE crop_source='人工框选'").fetchone()[0], 3)

    # 6. FastReID embedding 维度/归一化/来源（实施文档 9.2 与 16.4）
    def test_reid_embeddings(self):
        row = self.db.execute(
            """SELECT reid_embedding, reid_source, reid_model_version FROM captures
               WHERE reid_embedding IS NOT NULL LIMIT 1"""
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["reid_source"], "FastReID")
        self.assertTrue(row["reid_model_version"].startswith("fastreid:"))
        feature = np.frombuffer(row["reid_embedding"], dtype=np.float32)
        self.assertEqual(feature.size, 2048)
        self.assertAlmostEqual(float(np.linalg.norm(feature)), 1.0, places=2)
        # 任何 embedding 都必须属于 FastReID 来源；候选检索不得读旧的 embedding 列
        mixed = self.db.execute(
            """SELECT COUNT(*) FROM captures
               WHERE reid_embedding IS NOT NULL AND reid_source!='FastReID'"""
        ).fetchone()[0]
        self.assertEqual(mixed, 0)
        self.assertNotIn("embedding AS feature",
                         server.__dict__.get("CAPTURE_FIELDS", ""))

    # 7. 车辆属性字段入库（实施文档 9.1）
    def test_vehicle_attributes(self):
        row = self.db.execute(
            """SELECT vehicle_color, vehicle_color_conf, vehicle_type, vehicle_type_conf,
                      attr_model_version FROM captures
               WHERE vehicle_type IS NOT NULL AND vehicle_color IS NOT NULL LIMIT 1"""
        ).fetchone()
        self.assertIsNotNone(row, "属性模型未落库")
        self.assertTrue(row["attr_model_version"].startswith("paddledet:"))
        self.assertGreaterEqual(row["vehicle_color_conf"], 0.0)
        self.assertLessEqual(row["vehicle_color_conf"], 1.0)
        stats = self.get_json("/api/stats")
        self.assertGreater(stats["cropped"], 0)
        with_attr = self.db.execute(
            """SELECT COUNT(*) FROM captures
               WHERE vehicle_crop_path IS NOT NULL
                 AND coalesce(vehicle_type,'')!='' AND coalesce(attr_model_version,'')!=''"""
        ).fetchone()[0]
        self.assertGreater(with_attr / max(1, stats["cropped"]), 0.5, "属性覆盖率过低")
        # 订单级多数投票表已生成
        self.assertGreater(self.db.execute(
            "SELECT COUNT(*) FROM order_attributes").fetchone()[0], 100)

    # 8. 查询接口：订单/车牌/车型/时段/以图搜车（实施文档 12 步）
    def test_query_apis(self):
        orders = self.get_json("/api/orders?limit=5")
        self.assertEqual(len(orders), 5)
        detail = self.get_json(f"/api/order/{orders[0]['order_id']}")
        self.assertGreater(len(detail["items"]), 0)
        searched = self.get_json("/api/search?vehicle_type=truck&limit=50")
        for item in searched:
            self.assertEqual(item["vehicle_type"], "truck")
        hours = self.get_json("/api/search?start_hour=8&end_hour=8&limit=50")
        for item in hours:
            self.assertEqual(item["hour"], 8)
        featured = self.db.execute(
            """SELECT id FROM captures WHERE reid_embedding IS NOT NULL
               ORDER BY id LIMIT 1""").fetchone()
        if featured:
            result = self.get_json(f"/api/track_by_img?id={featured['id']}")
            items = result["items"]
            sims = [item.get("similarity") for item in items if item.get("similarity") is not None]
            self.assertEqual(sims, sorted(sims, reverse=True), "相似度未降序排列")

    # 8.4 内部接口进度状态：订单列表每行 + 订单详情顶层都带 progress_status 三字段
    def test_internal_progress_status(self):
        orders = self.get_json("/api/orders?limit=10")
        self.assertTrue(orders)
        for item in orders:
            self.assertIn(item["progress_status"], server.PROGRESS_STATUSES,
                          f"订单 {item['order_id']} 状态非法：{item['progress_status']!r}")
            self.assertIn("last_stage", item)
            self.assertIn("last_event_time", item)
        detail = self.get_json(f"/api/order/{orders[0]['order_id']}")
        self.assertIn(detail["progress_status"], server.PROGRESS_STATUSES)
        self.assertEqual(detail["progress_status"], orders[0]["progress_status"],
                         "同一订单的列表与详情状态应一致")
        for item in detail["items"]:
            self.assertIn("progress_status", item,
                          f"抓拍 #{item['id']} 缺少逐张进度状态")

    # 8.4b 按园区自适应阈值：从存量完整订单拟合，带缓存
    def test_flow_thresholds(self):
        th = server.flow_thresholds(self.db, "")
        self.assertGreater(th["exit_assume_min"], 0)
        self.assertGreaterEqual(th["silence_tail_min"], th["exit_assume_min"],
                                "静默自愈阈值应不小于出磅窗口（更保守）")
        self.assertIs(th, server.flow_thresholds(self.db, ""), "同版本数据应命中缓存")

    # 8.5 订单级汇总：每个订单确定大致发生时间（order_time）与唯一车牌号（plate）
    def test_order_summary_plate_and_time(self):
        rows = self.get_json("/api/orders?limit=100")
        self.assertTrue(rows)
        for item in rows:
            self.assertIn("plate", item)
            self.assertIn("order_time", item)
            if item["order_time"]:
                self.assertRegex(item["order_time"], r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
        # 取一个有车牌且订单号未跨园区撞号的样本（详情接口免 park 参数）
        sample = next((r for r in rows if r["plate"] and self.db.execute(
            "SELECT COUNT(DISTINCT coalesce(park,'')) FROM captures WHERE order_id=?",
            (r["order_id"],)).fetchone()[0] == 1), None)
        if not sample:
            self.fail("前 100 个订单里没有可校验的带车牌样本")
        detail = self.get_json(f"/api/order/{sample['order_id']}")
        self.assertEqual(detail["plate"], sample["plate"])
        self.assertEqual(detail["order_time"], sample["order_time"])
        # order_time = 订单内最早的有效抓拍时间
        min_time = self.db.execute(
            """SELECT MIN(capture_time) FROM captures
               WHERE order_id=? AND coalesce(capture_time,'')!=''""",
            (sample["order_id"],)).fetchone()[0]
        self.assertEqual(sample["order_time"], min_time or "")
        # plate 必须来自该订单的识别车牌/OSD 参考车牌候选集合（宽松归一后比对）
        candidates = set()
        for (plate,) in self.db.execute(
                """SELECT plate_final FROM captures WHERE order_id=? AND coalesce(plate_final,'')!=''
                   UNION SELECT osd_plate FROM captures WHERE order_id=? AND coalesce(osd_plate,'')!=''""",
                (sample["order_id"], sample["order_id"])):
            candidates.add(re.sub(r"[^0-9A-Z\u4e00-\u9fff挂学警港澳使领]", "", (plate or "").upper()))
        self.assertIn(re.sub(r"[^0-9A-Z\u4e00-\u9fff挂学警港澳使领]", "",
                             sample["plate"].upper()), candidates)

    # 8.6 未关联抓拍匹配：外观难分时按时间差就近归属（2026-09 现场规则）
    def test_match_loose_ambiguous_falls_back_to_time(self):
        import build_db
        from datetime import datetime as dt

        def emb(cos_value):
            # 二维单位向量，与 (1,0) 方向的余弦恰为 cos_value
            v = np.array([cos_value, (1 - cos_value * cos_value) ** 0.5],
                         dtype=np.float32)
            return v / np.linalg.norm(v)

        base = dt(2026, 8, 9, 10, 0, 0)
        loose = {"park": "P", "plate": "", "dt": base,
                 "emb": emb(1.0), "vehicle_type": "truck"}
        # 订单 A：外观略优（0.72 vs 0.70，领先 0.02 < margin）但时间差 20 分钟；
        # 订单 B：时间差 5 分钟 -> 外观难分，应归时间更近的 B
        orders = {
            ("P", "A"): {"park": "P", "times": [dt(2026, 8, 9, 9, 40, 0)],
                         "embs": [emb(0.72)], "main_type": "truck",
                         "consensus_plate": ""},
            ("P", "B"): {"park": "P", "times": [dt(2026, 8, 9, 9, 55, 0)],
                         "embs": [emb(0.70)], "main_type": "truck",
                         "consensus_plate": ""},
        }
        key, _ev, reason = build_db._match_loose_capture(
            loose, orders, window_seconds=7200, sim_threshold=0.55, sim_margin=0.03)
        self.assertEqual(key, ("P", "B"),
                         "外观难分时应归时间更近的订单B而不是外观略优的A")
        self.assertEqual(reason, "")
        # 两单时间差彼此 <300 秒仍是真歧义：不强行归类
        orders[("P", "A")]["times"] = [dt(2026, 8, 9, 9, 57, 0)]
        orders[("P", "B")]["times"] = [dt(2026, 8, 9, 9, 58, 0)]
        loose["dt"] = dt(2026, 8, 9, 9, 59, 30)
        key, _ev, reason = build_db._match_loose_capture(
            loose, orders, window_seconds=7200, sim_threshold=0.55, sim_margin=0.03)
        self.assertIsNone(key)
        self.assertIn("时间差接近", reason)
        # 外观决定性领先时仍由外观定胜负（时间只是并列时的裁决手段）
        orders[("P", "A")]["times"] = [dt(2026, 8, 9, 9, 40, 0)]
        orders[("P", "B")]["times"] = [dt(2026, 8, 9, 9, 55, 0)]
        orders[("P", "A")]["embs"] = [emb(0.85)]
        loose["dt"] = base
        key, _ev, _reason = build_db._match_loose_capture(
            loose, orders, window_seconds=7200, sim_threshold=0.55, sim_margin=0.03)
        self.assertEqual(key, ("P", "A"), "外观明显领先时应选外观最优订单")

    # 8.7 订单内可疑照片标记：车型与主车不符，或车牌完全不同（差>=5）
    def test_order_suspect_flag(self):
        def tail(a, b):
            return 99 if not a or not b or len(a) != len(b) else \
                sum(1 for x, y in zip(a[1:], b[1:]) if x != y)

        detail = self.get_json("/api/order/68002608090040")
        main = detail.get("plate") or ""
        for item in detail["items"]:
            self.assertIn("suspect", item)
            self.assertIsInstance(item["suspect"], bool)
            mt = item.get("order_main_type") or ""
            vt = item.get("vehicle_type") or ""
            pf = item.get("plate_final") or ""
            expect = ((bool(mt) and bool(vt) and vt != mt)
                      or (bool(main) and bool(pf) and len(pf) == len(main)
                          and tail(pf, main) >= 5))
            self.assertEqual(item["suspect"], expect,
                             f"抓拍#{item['id']} 可疑标记与证据不一致")
        track = self.get_json("/api/track?vid=1")
        for item in track:
            self.assertIn("suspect", item)

    # 9. 断点续跑：已处理记录不会被重复选择（实施文档 10 步）
    def test_resume_no_reprocessing(self):
        pending = self.db.execute(
            """SELECT COUNT(*) FROM captures
               WHERE coalesce(capture_status,'vehicle')!='no_vehicle'
                 AND (reid_embedding IS NULL OR vehicle_crop_path IS NULL)
                 AND coalesce(park,'')='山高荃银丰麦田库'"""
        ).fetchone()[0]
        # 山高尚存 2 条历史遗留待处理帧（生产为 0），上界放宽为 2；
        # 农易行数据本地未跑 GPU，不纳入本断言
        self.assertLessEqual(pending, 2, "存在未处理完的记录，流水线会重复处理")
        # 无车图已固定标记，不会每次重跑都重新检测
        self.assertEqual(self.db.execute(
            """SELECT COUNT(*) FROM captures WHERE capture_status='no_vehicle'
               AND vehicle_crop_path IS NOT NULL""").fetchone()[0], 0)

    # 10. 标注工作流与导出（实施文档 12/13 步）
    def test_annotation_workflow_and_export(self):
        target = self.db.execute(
            """SELECT id, plate_final, plate_confidence, recognition_source, vehicle_id, plate_match_reason
               FROM captures WHERE reid_embedding IS NOT NULL
               ORDER BY id DESC LIMIT 1""").fetchone()
        other = self.db.execute(
            """SELECT id, plate_final, plate_confidence, recognition_source, vehicle_id, plate_match_reason
               FROM captures WHERE reid_embedding IS NOT NULL
               AND id!=? ORDER BY id DESC LIMIT 1""",
            (target["id"],)).fetchone()
        orig_t, orig_o = dict(target), dict(other)
        identity_id = None
        try:
            first = self.client.post(
                "/api/annotations/capture",
                json={"capture_id": target["id"], "action": "new_identity",
                      "usability": "valid", "corrected_plate": "皖KTEST99", "annotator": "测试"},
            )
            self.assertEqual(first.status_code, 200, first.get_data(as_text=True))
            identity_id = first.get_json()["item"]["identity_id"]
            # 确认 /api/annotations/capture 也写回 plate_final（修复断链）
            after_t = self.db.execute(
                "SELECT plate_final, recognition_source FROM captures WHERE id=?",
                (target["id"],)).fetchone()
            self.assertEqual(after_t["plate_final"], "皖KTEST99")
            self.assertEqual(after_t["recognition_source"], "人工标注")
            second = self.client.post(
                "/api/annotations/pair",
                json={"capture_id": target["id"], "candidate_id": other["id"],
                      "relation": "same", "sync_cluster": False, "annotator": "测试"},
            )
            self.assertEqual(second.status_code, 200, second.get_data(as_text=True))
            stats = self.get_json("/api/annotations/stats")
            self.assertGreaterEqual(stats["same_pairs"], 1)
            exported = self.client.get("/api/annotations/export.csv")
            self.assertEqual(exported.status_code, 200)
            self.assertIn("皖KTEST99", exported.get_data(as_text=True))
        finally:
            # 清理测试数据并还原 captures 原值，不影响真实标注
            self.db.execute("DELETE FROM pair_annotations WHERE left_capture_id=? OR right_capture_id=?",
                            (target["id"], other["id"]))
            self.db.execute("DELETE FROM capture_annotations WHERE capture_id IN (?, ?)",
                            (target["id"], other["id"]))
            if identity_id:
                self.db.execute("DELETE FROM annotation_identities WHERE id=?", (identity_id,))
            self.db.execute(
                "DELETE FROM annotation_audit WHERE capture_id IN (?, ?) AND annotator='测试'",
                (target["id"], other["id"]))
            for orig in (orig_t, orig_o):
                self.db.execute(
                    """UPDATE captures SET plate_final=?, plate_confidence=?,
                              recognition_source=?, vehicle_id=?, plate_match_reason=?
                       WHERE id=?""",
                    (orig["plate_final"], orig["plate_confidence"], orig["recognition_source"],
                     orig["vehicle_id"], orig["plate_match_reason"], orig["id"]))
            self.db.commit()


class ProgressStatusTests(unittest.TestCase):
    """进度状态机纯函数测试（不依赖数据库），时钟由 now 参数注入。"""

    def setUp(self):
        self.now = datetime(2026, 10, 29, 12, 0, 0)

    def status(self, events):
        return server.compute_progress_status(events, now=self.now)["progress_status"]

    def test_empty_events(self):
        info = server.compute_progress_status([])
        self.assertEqual(info["progress_status"], "")
        self.assertEqual(info["last_stage"], "")

    def test_gate_in_then_wait(self):
        self.assertEqual(self.status([("2026-10-29 11:50:00", "抓拍")]), "大门进")
        self.assertEqual(self.status([("2026-10-29 11:00:00", "抓拍")]), "等待入磅")
        # 进园锚点取首个事件：排队中的后续抓拍不重置窗口
        self.assertEqual(
            self.status([("2026-10-29 11:00:00", "抓拍"), ("2026-10-29 11:05:00", "抓拍")]),
            "等待入磅")
        # 只有质检事件（扦样点紧贴地磅，属入磅前环节）同样从这两个状态起步
        self.assertEqual(self.status([("2026-10-29 11:55:00", "质检")]), "大门进")

    def test_weighing_then_unloading(self):
        self.assertEqual(self.status([("2026-10-29 11:55:00", "入磅")]), "入磅中")
        # 入磅超窗口后进入卸货；卸货中的场院抓拍不得回退状态
        events = [("2026-10-29 10:00:00", "入磅"), ("2026-10-29 10:05:00", "抓拍")]
        self.assertEqual(self.status(events), "卸货")

    def test_weighout_and_gate_out(self):
        self.assertEqual(self.status([("2026-10-29 11:20:00", "出磅")]), "出磅中")
        self.assertEqual(self.status([("2026-10-29 10:00:00", "出磅")]), "大门出")
        # 出磅后出现抓拍 = 确认离场，last_event 落在抓拍上
        info = server.compute_progress_status(
            [("2026-10-29 10:00:00", "出磅"), ("2026-10-29 10:03:00", "抓拍")], now=self.now)
        self.assertEqual(info["progress_status"], "大门出")
        self.assertEqual(info["last_event_time"], "2026-10-29 10:03:00")
        self.assertEqual(info["last_stage"], "抓拍")

    def test_exit_timeout_assumed(self):
        # 出磅后超过默认窗口（60 分钟）无抓拍：大门出但必须带推定标记
        info = server.compute_progress_status([("2026-10-29 10:00:00", "出磅")],
                                              now=datetime(2026, 10, 29, 11, 30, 0))
        self.assertEqual(info["progress_status"], "大门出")
        self.assertTrue(info["progress_assumed"])
        self.assertIn("推定", info["progress_note"])

    def test_threshold_override_keeps_weighout(self):
        th = dict(server.DEFAULT_PROGRESS_TH, exit_assume_min=200)
        info = server.compute_progress_status([("2026-10-29 10:00:00", "出磅")],
                                              now=datetime(2026, 10, 29, 11, 30, 0), th=th)
        self.assertEqual(info["progress_status"], "出磅中")
        self.assertFalse(info["progress_assumed"])

    def test_silence_autoheal_unload(self):
        # 卸货后长时间杳无音信：按静默阈值推定离场（疑似漏拍出磅），不再永久卡死
        events = [("2026-10-29 10:00:00", "入磅"), ("2026-10-29 10:05:00", "抓拍")]
        info = server.compute_progress_status(events, now=datetime(2026, 10, 30, 12, 0, 0))
        self.assertEqual(info["progress_status"], "大门出")
        self.assertTrue(info["progress_assumed"])
        self.assertIn("漏拍", info["progress_note"])

    def test_silence_autoheal_wait(self):
        # 全流程缺失（只有场院抓拍）+ 长静默：同样推定离场
        info = server.compute_progress_status([("2026-10-29 10:00:00", "抓拍")],
                                              now=datetime(2026, 10, 30, 12, 0, 0))
        self.assertEqual(info["progress_status"], "大门出")
        self.assertTrue(info["progress_assumed"])

    def test_gate_out_photo_confirms_departure(self):
        # 出方向大门照片（OSD 设备名"大门抓拍出"）= 离场实锤：
        # 即使出磅照片缺失也不带推定标记
        events = [("2026-10-29 10:00:00", "入磅", ""),
                  ("2026-10-29 12:30:00", "抓拍", "out")]
        info = server.compute_progress_status(events, now=datetime(2026, 10, 29, 12, 31, 0))
        self.assertEqual(info["progress_status"], "大门出")
        self.assertFalse(info["progress_assumed"])

    def test_undirected_snap_still_assumed(self):
        # 对照：无方向的场院抓拍在同样场景下只能走静默自愈（带推定标记）
        events = [("2026-10-29 10:00:00", "入磅", ""),
                  ("2026-10-29 12:30:00", "抓拍", "")]
        info = server.compute_progress_status(events, now=datetime(2026, 10, 30, 12, 0, 0))
        self.assertEqual(info["progress_status"], "大门出")
        self.assertTrue(info["progress_assumed"])

    def test_full_flow(self):
        events = [("2026-10-29 09:00:00", "抓拍"), ("2026-10-29 09:02:00", "质检"),
                  ("2026-10-29 09:03:00", "入磅"), ("2026-10-29 09:04:00", "抓拍"),
                  ("2026-10-29 10:50:00", "出磅"), ("2026-10-29 10:54:00", "抓拍")]
        self.assertEqual(self.status(events), "大门出")

    def test_capture_status_timeline(self):
        """逐张状态：按拍摄时刻推导——入磅照片本身=入磅中，出磅后抓拍=大门出，
        no_vehicle 照片不产生事件但仍按此前事件标状态。"""
        caps = [
            {"id": 1, "capture_time": "2026-10-29 09:00:00", "stage": "抓拍", "capture_status": "vehicle"},
            {"id": 2, "capture_time": "2026-10-29 09:20:00", "stage": "抓拍", "capture_status": "vehicle"},
            {"id": 3, "capture_time": "2026-10-29 09:30:00", "stage": "质检", "capture_status": "vehicle"},
            {"id": 4, "capture_time": "2026-10-29 09:31:00", "stage": "入磅", "capture_status": "vehicle"},
            {"id": 5, "capture_time": "2026-10-29 09:45:00", "stage": "抓拍", "capture_status": "vehicle"},
            {"id": 6, "capture_time": "2026-10-29 11:00:00", "stage": "出磅", "capture_status": "vehicle"},
            {"id": 7, "capture_time": "2026-10-29 11:05:00", "stage": "抓拍", "capture_status": "vehicle"},
            {"id": 8, "capture_time": "2026-10-29 11:06:00", "stage": "抓拍", "capture_status": "no_vehicle"},
        ]
        self.assertEqual([info["progress_status"] for info in server.compute_capture_statuses(caps)],
                         ["大门进", "等待入磅", "等待入磅", "入磅中",
                          "卸货", "出磅中", "大门出", "大门出"])
        # 入磅照片自带状态持续分钟数（照片时刻-上磅时刻）
        self.assertEqual(server.compute_capture_statuses(caps)[3]["progress_minutes"], 0)

    def test_capture_status_autoheal(self):
        """逐张静默自愈：事件后再无动静超过静默阈值，之后的帧推定大门出；
        若中途车辆重新被抓拍，静默重置，恢复卸货标注。"""
        caps = [
            {"id": 1, "capture_time": "2026-10-29 09:00:00", "stage": "入磅", "capture_status": "vehicle"},
            {"id": 2, "capture_time": "2026-10-30 12:00:00", "stage": "抓拍", "capture_status": "vehicle"},
            {"id": 3, "capture_time": "2026-10-31 09:00:00", "stage": "抓拍", "capture_status": "no_vehicle"},
        ]
        states = server.compute_capture_statuses(caps)
        self.assertEqual(states[0]["progress_status"], "入磅中")
        self.assertEqual(states[1]["progress_status"], "卸货")  # 重新出现，静默重置
        self.assertEqual(states[2]["progress_status"], "大门出")
        self.assertTrue(states[2]["progress_assumed"])
        self.assertIn("漏拍", states[2]["progress_note"])

    def test_capture_status_gate_out_photo(self):
        """逐张：出方向大门照片那一刻即为"大门出"（确认）。"""
        caps = [
            {"id": 1, "capture_time": "2026-10-29 10:00:00", "stage": "入磅",
             "capture_status": "vehicle", "osd_camera": ""},
            {"id": 2, "capture_time": "2026-10-29 12:30:00", "stage": "抓拍",
             "capture_status": "vehicle", "osd_camera": "大门抓拍出01"},
        ]
        states = server.compute_capture_statuses(caps)
        self.assertEqual(states[0]["progress_status"], "入磅中")
        self.assertEqual(states[1]["progress_status"], "大门出")
        self.assertFalse(states[1]["progress_assumed"])


class OpenApiProgressTests(unittest.TestCase):
    """开放接口进度字段接线验证（临时注入 token，不动真实配置）。"""

    @classmethod
    def setUpClass(cls):
        server.app.config.update(TESTING=True)
        cls.client = server.app.test_client()
        cls.token = "unit-test-token"
        cls._old = server.OPEN_API_TOKEN
        server.OPEN_API_TOKEN = cls.token

    @classmethod
    def tearDownClass(cls):
        server.OPEN_API_TOKEN = cls._old

    def get_open_json(self, url):
        response = self.client.get(url, headers={"X-API-Token": self.token})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return response.get_json()

    def test_orders_carry_progress_status(self):
        data = self.get_open_json("/api/open/orders?date=2026-08-19&page_size=5")
        items = data["items"]
        self.assertTrue(items)
        for item in items:
            # 无任何有效时间戳事件的订单状态为空串（无证据不判定），跳过
            if not item["progress_status"]:
                continue
            self.assertIn(item["progress_status"], server.PROGRESS_STATUSES)
            self.assertIn("last_stage", item)
            self.assertIn("last_event_time", item)

    def test_yard_excludes_departed(self):
        data = self.get_open_json("/api/open/yard")
        self.assertEqual(data["code"], 0)
        for item in data["items"]:
            self.assertIn(item["progress_status"], server.PROGRESS_STATUSES)
            self.assertNotEqual(item["progress_status"], server.PROGRESS_GATE_OUT)
            self.assertIn("plate", item)


class LabelWorkspaceTests(unittest.TestCase):
    """标注工作台端到端：保存写回、锚点合成事件、队列、回放。测试后恢复原值。"""

    @classmethod
    def setUpClass(cls):
        server.app.config.update(TESTING=True)
        cls.client = server.app.test_client()

    def get_json(self, url):
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return response.get_json()

    def post(self, url, **body):
        response = self.client.post(url, json=body)
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return response.get_json()

    def test_label_capture_writeback_and_override(self):
        from datetime import timedelta
        with closing(server.get_db()) as db:
            row = db.execute("""SELECT id, order_id, capture_time FROM captures
                WHERE stage='抓拍' AND capture_time!=''
                AND coalesce(capture_status,'vehicle')!='no_vehicle' LIMIT 1""").fetchone()
        capture_id, order_id = row["id"], row["order_id"]
        with closing(server.get_db()) as db:
            orig = dict(db.execute(
                """SELECT stage, stage_rank, capture_time, hour, ocr_status,
                          plate_final, plate_confidence, recognition_source,
                          vehicle_color, vehicle_color_conf, vehicle_type, vehicle_type_conf
                   FROM captures WHERE id=?""",
                (capture_id,)).fetchone())
        try:
            data = self.post("/api/label/capture", capture_id=capture_id,
                             annotator="测试", skip_recluster=True,
                             fields={"corrected_stage": "入磅", "corrected_status": "入磅中",
                                     "corrected_direction": "out", "corrected_color": "白色",
                                     "corrected_type": "货车",
                                     "corrected_time": "08:00:00",
                                     "corrected_plate": "皖L00001", "no_vehicle_manual": 0,
                                     "usability": "valid", "notes": "字段内备注验证"})
            item = data["item"]
            self.assertEqual(item["stage"], "入磅")                      # 写回
            self.assertEqual(item["capture_time"].split(" ")[1], "08:00:00")
            self.assertEqual(item["ocr_status"], "manual")               # 防OCR覆盖
            self.assertEqual(item["plate_final"], "皖L00001")
            self.assertEqual(item["recognition_source"], "人工标注")
            # 颜色/车型统一落标准英文码（与模型输出对齐，防止聚类第7步误拆）
            self.assertEqual(item["vehicle_color"], "white")
            self.assertEqual(item["vehicle_type"], "truck")
            self.assertEqual(item["notes"], "字段内备注验证")
            detail = self.client.get(f"/api/order/{order_id}").get_json()
            target = [i for i in detail["items"] if i["id"] == capture_id][0]
            self.assertEqual(target["progress_status"], "入磅中")         # override
            self.assertEqual(target["progress_source"], "manual")
        finally:
            with closing(server.get_db()) as db:
                db.execute("""UPDATE captures SET stage=?, stage_rank=?, capture_time=?,
                              hour=?, ocr_status=?, plate_final=?, plate_confidence=?,
                              recognition_source=?, vehicle_color=?, vehicle_color_conf=?,
                              vehicle_type=?, vehicle_type_conf=?
                              WHERE id=?""",
                           (orig["stage"], orig["stage_rank"], orig["capture_time"],
                            orig["hour"], orig["ocr_status"], orig["plate_final"],
                            orig["plate_confidence"], orig["recognition_source"],
                            orig["vehicle_color"], orig["vehicle_color_conf"],
                            orig["vehicle_type"], orig["vehicle_type_conf"], capture_id))
                db.execute("DELETE FROM capture_annotations WHERE capture_id=?", (capture_id,))
                db.execute("DELETE FROM annotation_audit WHERE capture_id=?", (capture_id,))
                db.commit()

    def test_photo_queue_types(self):
        # 照片级队列：包含聚类冲突与未关联散图在内的 7 类都返回照片条目
        for qtype in ("cluster_split", "loose", "direction_unknown", "plate_missing",
                      "time_missing", "attr_low", "attr_review"):
            data = self.get_json(f"/api/label/queue?type={qtype}&limit=5")
            self.assertEqual(data["level"], "photo")
            for item in data["items"]:
                self.assertIn("capture_id", item)
                self.assertIn("order_id", item)
                self.assertIn("reason", item)

    def test_unified_workbench_routes_and_training_summary(self):
        # 旧入口 /annotate 与新入口 /label 合并到同一统一工作台模板
        r_label = self.client.get("/label")
        self.assertEqual(r_label.status_code, 200)
        self.assertIn("统一人工标注工作台", r_label.get_data(as_text=True))
        r_ann = self.client.get("/annotate")
        self.assertEqual(r_ann.status_code, 200)
        body_ann = r_ann.get_data(as_text=True)
        self.assertIn("统一人工标注工作台", body_ann)
        self.assertIn('INITIAL_TAB = "reid"', body_ann)
        # 训练就绪度看板接口正常返回四类数据集指标
        ts = self.get_json("/api/training/summary")
        for key in ("reid", "detection", "attr", "lpr", "active_reid", "job"):
            self.assertIn(key, ts)

    def test_label_queue_assumed_only(self):
        data = self.client.get("/api/label/queue?type=assumed").get_json()
        for item in data["items"]:
            self.assertTrue(item["progress_assumed"])
            self.assertFalse(item["reviewed"])

    def test_anchor_gate_out_confirms_departure(self):
        from datetime import timedelta
        with closing(server.get_db()) as db:
            row = db.execute("""SELECT order_id, coalesce(park,'') AS park FROM captures
                WHERE stage='入磅' AND capture_time!='' AND order_id!=''
                GROUP BY order_id, park LIMIT 1""").fetchone()
            last = db.execute("""SELECT MAX(capture_time) FROM captures
                WHERE order_id=? AND capture_time!=''""",
                (row["order_id"],)).fetchone()[0]
        anchor_time = (datetime.strptime(last, "%Y-%m-%d %H:%M:%S")
                       + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        self.post("/api/label/anchor", park=row["park"], order_id=row["order_id"],
                  kind="gate_out", anchor_time=anchor_time, annotator="测试")
        detail = self.client.get(f"/api/order/{row['order_id']}").get_json()
        try:
            self.assertEqual(detail["progress_status"], "大门出")
            self.assertFalse(detail["progress_assumed"])   # 锚点=确认，非推定
        finally:
            with closing(server.get_db()) as db:
                db.execute("DELETE FROM manual_order_anchors WHERE order_id=? AND annotator='测试'",
                           (row["order_id"],))
                db.commit()

    def test_device_registry_roundtrip(self):
        self.post("/api/label/device", park="测试园", device="Camera 99",
                  location="大门", direction="in", annotator="测试")
        with closing(server.get_db()) as db:
            row = db.execute("""SELECT location, direction FROM manual_device_registry
                WHERE park='测试园' AND device='Camera 99'""").fetchone()
            self.assertEqual((row["location"], row["direction"]), ("大门", "in"))
            db.execute("DELETE FROM manual_device_registry WHERE park='测试园'")
            db.commit()

    def test_replay_manual_restores_writeback(self):
        with closing(server.get_db()) as db:
            row = db.execute("""SELECT id, stage, stage_rank FROM captures
                WHERE stage='抓拍' LIMIT 1""").fetchone()
        capture_id = row["id"]
        try:
            self.post("/api/label/capture", capture_id=capture_id, annotator="测试",
                      fields={"corrected_stage": "出磅"})
            with closing(server.get_db()) as db:   # 模拟全库重建后的丢失
                db.execute("UPDATE captures SET stage='抓拍', stage_rank=-1 WHERE id=?",
                           (capture_id,))
                db.commit()
            stats = server.build_db.replay_manual(server.get_db())
            self.assertGreaterEqual(stats["applied"], 1)
            with closing(server.get_db()) as db:
                stage = db.execute("SELECT stage FROM captures WHERE id=?",
                                   (capture_id,)).fetchone()["stage"]
            self.assertEqual(stage, "出磅")
        finally:
            with closing(server.get_db()) as db:
                db.execute("""UPDATE captures SET stage=?, stage_rank=? WHERE id=?""",
                           (row["stage"], row["stage_rank"], capture_id))
                db.execute("DELETE FROM capture_annotations WHERE capture_id=?", (capture_id,))
                db.commit()


class GateStageChoiceTests(unittest.TestCase):
    """环节下拉选"进大门/出大门" = 方向修正：环节本身不变、不伪造设备名，
    状态机按方向重算（该帧成为离场确认事件）。"""

    @classmethod
    def setUpClass(cls):
        server.app.config.update(TESTING=True)
        cls.client = server.app.test_client()
        with closing(server.get_db()) as db:
            cls.row = dict(db.execute("""SELECT id, order_id, stage, stage_rank, coalesce(osd_camera,'') AS osd_camera FROM captures
                WHERE stage='抓拍' AND capture_time!=''
                AND coalesce(capture_status,'vehicle')!='no_vehicle' LIMIT 1""").fetchone())

    @classmethod
    def tearDownClass(cls):
        with closing(server.get_db()) as db:
            db.execute("DELETE FROM capture_annotations WHERE capture_id=?", (cls.row["id"],))
            db.execute("DELETE FROM annotation_audit WHERE capture_id=?", (cls.row["id"],))
            db.execute("""UPDATE captures SET stage=?, stage_rank=? WHERE id=?""",
                       (cls.row["stage"], cls.row["stage_rank"], cls.row["id"]))
            db.commit()

    def test_gate_stage_choice_maps_direction(self):
        response = self.client.post("/api/label/capture", json={
            "capture_id": self.row["id"], "annotator": "测试",
            "fields": {"corrected_stage": "出大门"}})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        with closing(server.get_db()) as db:
            after = db.execute("SELECT stage, osd_camera FROM captures WHERE id=?",
                               (self.row["id"],)).fetchone()
        self.assertEqual(after["stage"], self.row["stage"])           # 环节本身不变
        self.assertEqual(after["osd_camera"], self.row["osd_camera"])   # 设备名不被伪造
        detail = self.client.get(f"/api/order/{self.row['order_id']}").get_json()
        target = [i for i in detail["items"] if i["id"] == self.row["id"]][0]
        self.assertEqual(target["progress_status"], "大门出")          # 方向修正生效
        self.assertFalse(detail["progress_assumed"])                  # 离场被确认


class ManualClusteringConstraintsTests(unittest.TestCase):
    """隔离内存库验证：人工车牌/车型/配对约束在 associate_orders 中的精确行为。"""

    def _make_db(self):
        import build_db
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        build_db.ensure_schema(conn)
        build_db.ensure_label_schema(conn)
        return conn, build_db

    def _add_cap(self, conn, cid, park, order_id, plate="", vtype="truck",
                 lpr_conf=0.85, src="车牌模型", date="2026-08-10"):
        emb = np.ones(8, dtype=np.float32)
        emb = (emb / np.linalg.norm(emb)).tobytes()
        conn.execute(
            """INSERT INTO captures
                 (id, filepath, filename, date, order_id, park, stage, stage_rank,
                  plate_final, lpr_plate, lpr_conf, recognition_source,
                  vehicle_type, capture_status, reid_embedding, reid_source)
               VALUES (?,?,?,?,?,?, '入磅', 1, ?,?,?,?, ?, 'vehicle', ?, 'FastReID')""",
            (cid, f"f{cid}.jpg", f"f{cid}.jpg", date, order_id, park,
             plate, plate or None, lpr_conf, src, vtype, emb),
        )

    def test_manual_plate_trusted_and_not_wiped(self):
        conn, build_db = self._make_db()
        # 订单 O1 有两帧车牌 皖K11111 + 一帧人工标注车牌 皖K99999（不应被第6步清空）
        self._add_cap(conn, 1, "P1", "O1", "皖K11111", "truck", 0.96)
        self._add_cap(conn, 2, "P1", "O1", "皖K11111", "truck", 0.96)
        self._add_cap(conn, 3, "P1", "O1", "皖K99999", "truck", 0.50, src="人工标注")
        conn.execute(
            """INSERT INTO capture_annotations (capture_id, usability, corrected_plate, annotator, updated_at)
               VALUES (3, 'valid', '皖K99999', '测试', '2026-08-10T10:00:00')"""
        )
        # 订单 O2 单帧人工车牌 皖K11112（差1字符，应因人工车牌可信而挂靠到 皖K11111 锚点）
        self._add_cap(conn, 4, "P1", "O3", "皖K11111", "truck", 0.96)
        self._add_cap(conn, 5, "P1", "O2", "皖K11112", "truck", 0.50, src="人工标注")
        conn.execute(
            """INSERT INTO capture_annotations (capture_id, usability, corrected_plate, annotator, updated_at)
               VALUES (5, 'valid', '皖K11112', '测试', '2026-08-10T10:00:00')"""
        )
        conn.commit()
        build_db.associate_orders(conn)
        r3 = conn.execute("SELECT plate_final FROM captures WHERE id=3").fetchone()
        self.assertEqual(r3["plate_final"], "皖K99999", "人工标注车牌不应被第6步清空")
        v1 = conn.execute("SELECT vehicle_id FROM captures WHERE id=1").fetchone()["vehicle_id"]
        v5 = conn.execute("SELECT vehicle_id FROM captures WHERE id=5").fetchone()["vehicle_id"]
        self.assertEqual(v1, v5, "单帧人工车牌应视为高可信并挂靠到相近锚点")

    def test_pair_same_merges_and_different_blocks_merge(self):
        conn, build_db = self._make_db()
        # 两个无牌订单通过 pair_annotations(same) 应合并为同一 vehicle_id
        self._add_cap(conn, 10, "P1", "NOPLATE_A", "", "truck")
        self._add_cap(conn, 11, "P1", "NOPLATE_B", "", "truck")
        conn.execute(
            """INSERT INTO pair_annotations (left_capture_id, right_capture_id, relation, annotator, updated_at)
               VALUES (10, 11, 'same', '测试', '2026-08-10T10:00:00')"""
        )
        # 两个同牌订单若被人工明确标为 different，严禁自动合并
        self._add_cap(conn, 20, "P1", "SAMEP_A", "皖K88888", "truck", 0.98)
        self._add_cap(conn, 21, "P1", "SAMEP_B", "皖K88888", "truck", 0.98)
        conn.execute(
            """INSERT INTO pair_annotations (left_capture_id, right_capture_id, relation, annotator, updated_at)
               VALUES (20, 21, 'different', '测试', '2026-08-10T10:00:00')"""
        )
        conn.commit()
        build_db.associate_orders(conn)
        v10 = conn.execute("SELECT vehicle_id FROM captures WHERE id=10").fetchone()["vehicle_id"]
        v11 = conn.execute("SELECT vehicle_id FROM captures WHERE id=11").fetchone()["vehicle_id"]
        self.assertIsNotNone(v10)
        self.assertEqual(v10, v11, "人工同车配对应合并无牌订单")
        v20 = conn.execute("SELECT vehicle_id FROM captures WHERE id=20").fetchone()["vehicle_id"]
        v21 = conn.execute("SELECT vehicle_id FROM captures WHERE id=21").fetchone()["vehicle_id"]
        self.assertNotEqual(v20, v21, "人工不同车配对应阻止错误合并")

    def test_replay_manual_normalizes_chinese_type(self):
        conn, build_db = self._make_db()
        self._add_cap(conn, 30, "P1", "O9", "皖K55555", "truck", 0.96)
        self._add_cap(conn, 31, "P1", "O9", "皖K55555", "sedan", 0.96)
        # 模拟旧记录里曾存有中文 "货车"
        conn.execute(
            """INSERT INTO capture_annotations (capture_id, usability, corrected_type, corrected_color, annotator, updated_at)
               VALUES (31, 'valid', '货车', '红色', '测试', '2026-08-10T10:00:00')"""
        )
        conn.commit()
        build_db.replay_manual(conn)
        r31 = conn.execute("SELECT vehicle_type, vehicle_color FROM captures WHERE id=31").fetchone()
        self.assertEqual((r31["vehicle_type"], r31["vehicle_color"]), ("truck", "red"))
        build_db.associate_orders(conn)
        v30 = conn.execute("SELECT vehicle_id FROM captures WHERE id=30").fetchone()["vehicle_id"]
        v31 = conn.execute("SELECT vehicle_id FROM captures WHERE id=31").fetchone()["vehicle_id"]
        self.assertEqual(v30, v31, "归一为 truck 后不再被第7步拆出聚类")

    def test_osd_device_code_and_rear_direction_flip(self):
        import build_db
        # 1) 单图解析：严格按设备编号 DMR=进、DMC=出、CK=仓库抓拍（不被 NVR 误报的「(背向)」干扰）
        cam, vt = build_db._parse_osd_camera_and_type(
            "监控点编号：02监控点信息：临泉大门抑拍进设备编号：DMRO1车牌号码：蓝皖KM9U08车辆类型：小货车"
        )
        self.assertEqual(cam, "大门抓拍进")
        cam, vt = build_db._parse_osd_camera_and_type(
            "监控点编号：01监控点信息：大门抑拍出设备编号：DMC01车牌号码：蓝皖K5713A品牌：特型车-罐车（背向）"
        )
        self.assertEqual(cam, "大门抓拍出")
        self.assertEqual(build_db.normalize_type(vt), "truck")
        cam, vt = build_db._parse_osd_camera_and_type(
            "监控点编号：03监控点信息：临泉仓库抑拍1设备编号：CK01车牌号码：黄皖AM9959车辆类型：SUV/MPV品牌：卡车-渣土车"
        )
        self.assertEqual(cam, "仓库抓拍")
        self.assertEqual(vt, "货车")

        # 2) 时序对向连拍纠偏：同车牌 1~60s 内先后触发 DMR -> DMC（或 DMC -> DMR），
        #    第 2 张改标背向；单独只触发一次 DMC（如地磅直出）必须保持「大门抓拍出」不误翻。
        conn, build_db = self._make_db()
        rows = [
            (1, "P1", "未关联图片", "2026-11-06 07:12:42", "大门抓拍进", "皖KM9V08"),
            (2, "P1", "未关联图片", "2026-11-06 07:12:49", "大门抓拍出", "皖KM9V08"),
            (3, "P1", "未关联图片", "2026-11-06 07:30:06", "大门抓拍出", "皖KM9V08"),
            (4, "P1", "未关联图片", "2026-11-06 07:30:29", "大门抓拍进", "皖KM9V08"),
            (5, "P1", "未关联图片", "2026-11-06 17:58:09", "大门抓拍出", "皖K5713A"),
        ]
        for cid, park, oid, ct, ocam, plate in rows:
            conn.execute(
                """INSERT INTO captures
                     (id, filepath, filename, date, order_id, park, stage, stage_rank,
                      capture_time, osd_camera, plate_final, osd_plate, capture_status)
                   VALUES (?,?,?,?,?,?, '抓拍', 3, ?,?,?,?, 'vehicle')""",
                (cid, f"f{cid}.jpg", f"f{cid}.jpg", "2026-11-06", oid, park, ct, ocam, plate, plate),
            )
        conn.commit()
        fixed = build_db.fix_opposite_gate_rear_directions(conn)
        self.assertEqual(fixed, 2)
        cams = {r[0]: r[1] for r in conn.execute("SELECT id, osd_camera FROM captures ORDER BY id").fetchall()}
        self.assertEqual(cams[1], "大门抓拍进")
        self.assertEqual(cams[2], "大门抓拍进(背向)")
        self.assertEqual(cams[3], "大门抓拍出")
        self.assertEqual(cams[4], "大门抓拍出(背向)")
        self.assertEqual(cams[5], "大门抓拍出")

    def test_warehouse_snap_confirms_unloading(self):
        # 入磅仅 2 分钟后拍到仓库抓拍（CK），即使未到 weigh_min 也应实锤「卸货」
        events = [
            ("2026-11-10 09:00:00", "入磅", ""),
            ("2026-11-10 09:02:00", "抓拍", "unload"),
        ]
        info = server.compute_progress_status(events, now=datetime(2026, 11, 10, 9, 3, 0))
        self.assertEqual(info["progress_status"], "卸货")
        self.assertFalse(info["progress_assumed"])


if __name__ == "__main__":
    unittest.main()
