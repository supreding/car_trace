# -*- coding: utf-8 -*-
"""应用自动测试：对 granary.db 与 Flask 接口做验收（实施文档 16.4 清单）。

运行前提：build_db.py 已完成扫描+OCR，reid_pipeline.ps1 已完成批处理。
"""
from __future__ import annotations

import re
import sqlite3
import unittest

import numpy as np

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
        self.assertEqual(stats["total"], 4908)
        self.assertEqual(stats["orders"], 749)
        self.assertGreaterEqual(stats["dates"], 19)
        stage_counts = {item["stage"]: item["count"] for item in stats["stage_items"]}
        self.assertEqual(stage_counts, {"入磅": 1503, "出磅": 1486,
                                        "抓拍": 1372, "质检": 547})

    # 2. 摄像头/分辨率约束表（实施文档第 4 步）
    def test_camera_keys(self):
        rows = self.db.execute(
            """SELECT camera_key, COUNT(*) AS n FROM captures
               WHERE coalesce(camera_key,'')!='' GROUP BY camera_key"""
        ).fetchall()
        cameras = {row["camera_key"]: row["n"] for row in rows}
        self.assertEqual(cameras.get("入磅@2560x1440"), 1503)
        self.assertEqual(cameras.get("出磅@2560x1440"), 1486)
        self.assertEqual(cameras.get("抓拍@1920x1264"), 1372)
        # 质检相机存在两台不同分辨率
        self.assertGreater(cameras.get("质检@1280x720", 0), 500)
        self.assertGreater(cameras.get("质检@2304x1296", 0), 0)

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
        self.assertEqual(unique_orders, 749)
        # 2026-09-23 大门照片重归类（--assign-gate）后，若干 (订单,日期)
        # 目录组合合并/消失，目录数由 806 变为 803
        self.assertEqual(total_dirs, 803)

    # 4. 时间戳 OCR 与降级标记（实施文档 3.2）
    def test_ocr_timestamps(self):
        total, ok = self.db.execute(
            """SELECT COUNT(*),
                      SUM(CASE WHEN ocr_status='ok' THEN 1 ELSE 0 END)
               FROM captures"""
        ).fetchone()
        self.assertEqual(total, 4908)
        self.assertGreater(ok / total, 0.85, "OCR 成功率过低")
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
        self.assertEqual(
            self.db.execute(
                "SELECT COUNT(*) FROM captures WHERE detection_count IS NULL").fetchone()[0], 0)
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
        with_crops = self.db.execute(
            """SELECT COUNT(*) FROM captures
               WHERE vehicle_crop_path IS NOT NULL AND crop_source!='PaddleDetection'"""
        ).fetchone()[0]
        self.assertEqual(with_crops, 0)

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

    # 8.7 订单内可疑照片标记：车牌与主车车牌差>=2 的行 plate_mismatch=True
    def test_order_plate_mismatch_flag(self):
        def tail(a, b):
            return 99 if not a or not b or len(a) != len(b) else \
                sum(1 for x, y in zip(a[1:], b[1:]) if x != y)

        detail = self.get_json("/api/order/68002608090040")
        main = detail.get("plate") or ""
        for item in detail["items"]:
            self.assertIn("plate_mismatch", item)
            self.assertIsInstance(item["plate_mismatch"], bool)
            pf, osd = item.get("plate_final") or "", item.get("osd_plate") or ""
            expect = (tail(pf, main) >= 2) if pf else (bool(osd) and tail(osd, main) >= 2)
            self.assertEqual(item["plate_mismatch"], expect,
                             f"抓拍#{item['id']} 可疑标记与车牌证据不一致")
        track = self.get_json("/api/track?vid=1")
        for item in track:
            self.assertIn("plate_mismatch", item)

    # 9. 断点续跑：已处理记录不会被重复选择（实施文档 10 步）
    def test_resume_no_reprocessing(self):
        pending = self.db.execute(
            """SELECT COUNT(*) FROM captures
               WHERE coalesce(capture_status,'vehicle')!='no_vehicle'
                 AND (reid_embedding IS NULL OR vehicle_crop_path IS NULL)"""
        ).fetchone()[0]
        self.assertEqual(pending, 0, "存在未处理完的记录，流水线会重复处理")
        # 无车图已固定标记，不会每次重跑都重新检测
        self.assertEqual(self.db.execute(
            """SELECT COUNT(*) FROM captures WHERE capture_status='no_vehicle'
               AND vehicle_crop_path IS NOT NULL""").fetchone()[0], 0)

    # 10. 标注工作流与导出（实施文档 12/13 步）
    def test_annotation_workflow_and_export(self):
        target = self.db.execute(
            """SELECT id FROM captures WHERE reid_embedding IS NOT NULL
               ORDER BY id DESC LIMIT 1""").fetchone()
        other = self.db.execute(
            """SELECT id FROM captures WHERE reid_embedding IS NOT NULL
               AND id!=? ORDER BY id DESC LIMIT 1""",
            (target["id"],)).fetchone()
        import json as _json

        first = self.client.post(
            "/api/annotations/capture",
            json={"capture_id": target["id"], "action": "new_identity",
                  "usability": "valid", "corrected_plate": "皖KTEST99", "annotator": "测试"},
        )
        self.assertEqual(first.status_code, 200, first.get_data(as_text=True))
        identity_id = first.get_json()["item"]["identity_id"]
        second = self.client.post(
            "/api/annotations/pair",
            json={"capture_id": target["id"], "candidate_id": other["id"],
                  "relation": "same", "annotator": "测试"},
        )
        self.assertEqual(second.status_code, 200, second.get_data(as_text=True))
        stats = self.get_json("/api/annotations/stats")
        self.assertGreaterEqual(stats["same_pairs"], 1)
        exported = self.client.get("/api/annotations/export.csv")
        self.assertEqual(exported.status_code, 200)
        self.assertIn("皖KTEST99", exported.get_data(as_text=True))
        # 清理测试数据，不影响真实标注
        self.db.execute("DELETE FROM pair_annotations WHERE left_capture_id=? OR right_capture_id=?",
                        (target["id"], other["id"]))
        self.db.execute("DELETE FROM capture_annotations WHERE capture_id IN (?, ?)",
                        (target["id"], other["id"]))
        self.db.execute("DELETE FROM annotation_identities WHERE id=?", (identity_id,))
        self.db.execute(
            "DELETE FROM annotation_audit WHERE capture_id IN (?, ?) AND annotator='测试'",
            (target["id"], other["id"]))
        self.db.commit()


if __name__ == "__main__":
    unittest.main()
