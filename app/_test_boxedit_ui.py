# -*- coding: utf-8 -*-
"""临时脚本：构建常驻的框选编辑器 UI 测试环境（合成数据，不碰真实库）。

用法：python _test_boxedit_ui.py  ->  在 127.0.0.1:5057 起服务，打印临时目录。
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))


def make_image(path: Path, width: int, height: int, color):
    img = Image.new("RGB", (width, height), color)
    for x in range(0, width, 60):
        for y in range(0, height, 60):
            img.putpixel((x, y), (255, 255, 255))
    img.save(path, "JPEG", quality=90)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="boxedit_ui_"))
    db_path = tmp / "granary.db"
    os.environ["GRANARY_DB"] = str(db_path)
    os.environ["FLASK_DEBUG"] = "1"
    import server
    import build_db

    orig = tmp / "orig.jpg"
    make_image(orig, 1920, 1080, (90, 110, 160))
    crop_main = tmp / "crop_main.jpg"
    crop_inst = tmp / "crop_inst.jpg"
    crop_inst2 = tmp / "crop_inst2.jpg"
    make_image(crop_main, 460, 360, (200, 60, 30))
    make_image(crop_inst, 420, 380, (30, 180, 60))
    make_image(crop_inst2, 300, 340, (60, 60, 220))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    build_db.ensure_schema(conn)
    emb = lambda seed: np.random.default_rng(seed).random(8).astype(np.float32).tobytes()
    for n, (oid, plate) in enumerate([("UI1001", "皖K12345"), ("UI1002", "皖K67890")]):
        cid = 100 + n * 10
        conn.execute(
            """INSERT INTO captures (id, date, park, order_id, stage, filename, filepath,
                     capture_time, hour, capture_status, vehicle_crop_path, vehicle_box,
                     crop_source, detection_count, vehicle_det_score, plate_final,
                     vehicle_color, vehicle_color_conf, vehicle_type, vehicle_type_conf,
                     reid_embedding, reid_source, quality_score)
               VALUES (?, '2026-08-10', '测试园区', ?, '出磅', ?, ?, ?, 10, 'vehicle',
                       ?, ?, 'PaddleDetection', 3, 0.93, ?, 'blue', 0.8, 'truck', 0.7,
                       ?, 'FastReID', 0.9)""",
            (cid, oid, f"o{cid}.jpg", str(orig), f"2026-08-10 10:0{n}:00",
             str(crop_main), f"[200,{150 + n * 60},660,{510 + n * 60}]", plate,
             emb(cid * 10)))
        conn.execute(
            "UPDATE captures SET vehicle_id=? WHERE id=?",
            (9000 + n, cid))
        boxes = [(0, f"[200,{150 + n * 60},660,{510 + n * 60}]", 0.93, "blue", "truck", plate),
                 (1, "[1100,120,1600,620]", 0.77, "red", "truck", None),
                 (2, "[400,700,700,1000]", 0.61, "black", "suv", None)]
        for idx, box, score, color, vtype, lpr in boxes:
            crop = crop_main if idx == 0 else (crop_inst if idx == 1 else crop_inst2)
            conn.execute(
                """INSERT INTO vehicle_instances (capture_id, instance_index, crop_path,
                         vehicle_box, vehicle_det_score, vehicle_color, vehicle_color_conf,
                         vehicle_type, vehicle_type_conf, lpr_plate, lpr_conf,
                         lpr_attempted_at, reid_embedding, reid_source)
                   VALUES (?, ?, ?, ?, ?, ?, 0.8, ?, 0.7, ?, ?, '2026-08-10T10:00:00', ?, 'FastReID')""",
                (cid, idx, str(crop), box, score, color, vtype, lpr,
                 0.9 if lpr else None, emb(cid * 10 + idx)))
    conn.commit()
    conn.close()
    print(f"UI-TEST-ENV: {tmp}")
    print(f"UI-TEST-DB: {db_path}")
    server.app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5057")))


if __name__ == "__main__":
    main()
