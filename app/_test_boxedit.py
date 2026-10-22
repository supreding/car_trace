# -*- coding: utf-8 -*-
"""临时测试：人工框选修正 API（/api/boxedit/*）与流水线跳过逻辑。

在临时目录构造最小库（合成原图/裁剪图），不触碰真实 granary.db 与
reid_crops。运行：.venv-reid/Scripts/python.exe _test_boxedit.py
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


def make_image(path: Path, width: int = 1280, height: int = 720, color=(120, 40, 200)):
    img = Image.new("RGB", (width, height), color)
    for x in range(0, width, 40):
        for y in range(0, height, 40):
            img.putpixel((x, y), (255, 255, 255))
    img.save(path, "JPEG", quality=92)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="boxedit_test_"))
    db_path = tmp / "granary.db"
    os.environ["GRANARY_DB"] = str(db_path)
    sys.path.insert(0, str(APP_DIR))
    import server  # noqa: E402  （须在 GRANARY_DB 设置后导入）

    # ---- 造数据：合成原图 + 两张裁剪图 ----
    orig1 = tmp / "orig_main.jpg"
    orig2 = tmp / "orig_second.jpg"
    make_image(orig1)
    make_image(orig2, 960, 540, (30, 90, 200))
    crop_main = tmp / "crop_main.jpg"
    crop_inst = tmp / "crop_inst.jpg"
    make_image(crop_main, 400, 300, (200, 30, 30))
    make_image(crop_inst, 500, 350, (30, 200, 30))

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    build_db = sys.modules.get("build_db") or __import__("build_db")
    build_db.ensure_schema(conn)
    emb = np.random.rand(8).astype(np.float32)
    emb2 = np.random.rand(8).astype(np.float32)
    conn.execute(
        """INSERT INTO captures (id, date, park, order_id, stage, filename, filepath,
                                 capture_status, vehicle_crop_path, vehicle_box,
                                 crop_source, detection_count, vehicle_det_score,
                                 plate_final, reid_embedding)
           VALUES (1, '2026-08-10', '测试园区', 'A1', '出磅', 'o1.jpg', ?, 'vehicle',
                   ?, ?, 'PaddleDetection', 2, 0.91, '皖K00001', ?)""",
        (str(orig1), str(crop_main), "[100,100,500,400]", emb.tobytes()))
    conn.execute(
        """INSERT INTO captures (id, date, park, order_id, stage, filename, filepath,
                                 capture_status)
           VALUES (2, '2026-08-10', '测试园区', 'A1', '出磅', 'o2.jpg', ?, 'no_vehicle')""",
        (str(orig2),))
    conn.execute(
        "UPDATE captures SET capture_status='no_vehicle', status_reason='PaddleDetection 未检测到车辆' WHERE id=2")
    conn.executemany(
        """INSERT INTO vehicle_instances (capture_id, instance_index, crop_path,
                                          vehicle_box, vehicle_det_score, vehicle_color,
                                          vehicle_color_conf, vehicle_type, vehicle_type_conf,
                                          lpr_plate, lpr_conf, lpr_attempted_at,
                                          reid_embedding, reid_source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(1, 0, str(crop_main), "[100,100,500,400]", 0.91, "blue", 0.8, "truck", 0.7,
          "皖K00001", 0.9, "2026-08-10T10:00:00", emb.tobytes(), "FastReID"),
         (1, 1, str(crop_inst), "[600,100,1100,500]", 0.77, "red", 0.75, "truck", 0.66,
          "皖K12345", 0.88, "2026-08-10T10:00:01", emb2.tobytes(), "FastReID")])
    conn.commit()
    conn.close()

    client = server.app.test_client()
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = ""):
        print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    def post(url, payload):
        return client.post(url, json=payload)

    # ---- 1. 编辑器上下文 ----
    r = client.get("/api/boxedit/1")
    check("boxedit GET 200", r.status_code == 200, r.get_data(as_text=True))
    ctx = r.get_json()
    check("boxedit GET 尺寸", ctx["image_size"] == [1280, 720], str(ctx.get("image_size")))
    check("boxedit GET 当前框", ctx["capture"]["box"] == [100, 100, 500, 400])
    check("boxedit GET 实例2条", len(ctx["instances"]) == 2, str(len(ctx["instances"])))
    check("boxedit GET 实例框", ctx["instances"][1]["box"] == [600, 100, 1100, 500])
    check("boxedit GET 不暴露filepath", "filepath" not in ctx["capture"])
    check("boxedit GET 404", client.get("/api/boxedit/999").status_code == 404)

    # ---- 2. 保存人工框 ----
    new_box = [200, 150, 700, 500]
    r = post("/api/boxedit/save", {"capture_id": 1, "box": new_box})
    check("save 200", r.status_code == 200, r.get_data(as_text=True))
    body = r.get_json()
    check("save 返回框", body["box"] == new_box)
    check("save 裁剪文件已写", crop_main.is_file())
    with Image.open(crop_main) as im:
        check("save 裁剪尺寸=框", im.size == (500, 350), str(im.size))
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM captures WHERE id=1").fetchone()
    check("save crop_source", row["crop_source"] == "人工框选", row["crop_source"])
    check("save vehicle_box", row["vehicle_box"] == "[200, 150, 700, 500]", row["vehicle_box"])
    check("save plate 清空", row["plate_final"] is None and row["lpr_attempted_at"] is None)
    check("save embedding 清空", row["reid_embedding"] is None)
    inst0 = conn.execute(
        "SELECT * FROM vehicle_instances WHERE capture_id=1 AND instance_index=0").fetchone()
    check("save 实例0同步框", inst0["vehicle_box"] == "[200, 150, 700, 500]")
    check("save 实例0同步路径", inst0["crop_path"] == str(crop_main))
    edit = conn.execute("SELECT * FROM manual_box_edits WHERE capture_id=1").fetchone()
    check("save manual_box_edits 记录", edit is not None and edit["source"] == "manual_box")
    conn.close()

    # ---- 3. 保存的边界与校验 ----
    check("save 太小 400", post("/api/boxedit/save", {"capture_id": 1, "box": [0, 0, 10, 10]}).status_code == 400)
    r = post("/api/boxedit/save", {"capture_id": 1, "box": [-50, -50, 99999, 99999]})
    check("save 越界收拢 200", r.status_code == 200)
    check("save 越界收拢结果", r.get_json()["box"] == [0, 0, 1280, 720])
    check("save 非法box 400", post("/api/boxedit/save", {"capture_id": 1, "box": "bad"}).status_code == 400)
    check("save 缺ID 400", post("/api/boxedit/save", {"box": [1, 2, 3, 4]}).status_code == 400)

    # ---- 4. no_vehicle 帧手动建裁剪（默认路径分支，APP_DIR 指向临时目录） ----
    old_app_dir = server.APP_DIR
    server.APP_DIR = tmp
    try:
        r = post("/api/boxedit/save", {"capture_id": 2, "box": [100, 100, 600, 400]})
        check("save no_vehicle 200", r.status_code == 200, r.get_data(as_text=True))
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM captures WHERE id=2").fetchone()
        check("save no_vehicle 复活", row["capture_status"] == "vehicle")
        check("save no_vehicle 默认路径", row["vehicle_crop_path"] == "reid_crops/00000002.jpg",
              row["vehicle_crop_path"])
        check("save no_vehicle 文件", (tmp / "reid_crops/00000002.jpg").is_file())
        check("save no_vehicle 实例0补建", conn.execute(
            "SELECT 1 FROM vehicle_instances WHERE capture_id=2 AND instance_index=0").fetchone() is not None)
        conn.close()
    finally:
        server.APP_DIR = old_app_dir

    # ---- 5. 同帧切换主车（实例裁剪文件在 -> 直接复制） ----
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    inst1_before = dict(conn.execute(
        "SELECT * FROM vehicle_instances WHERE capture_id=1 AND instance_index=1").fetchone())
    conn.close()
    r = post("/api/boxedit/promote", {"capture_id": 1, "instance_id": inst1_before["id"]})
    check("promote 200", r.status_code == 200, r.get_data(as_text=True))
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM captures WHERE id=1").fetchone()
    check("promote 换框", row["vehicle_box"] == "[600, 100, 1100, 500]", row["vehicle_box"])
    check("promote 换来源", row["crop_source"] == "人工切换")
    check("promote 换属性", row["vehicle_color"] == "red" and row["vehicle_type"] == "truck")
    check("promote 复用embedding", row["reid_embedding"] == emb2.tobytes())
    check("promote plate清空待补算", row["plate_final"] is None and row["lpr_attempted_at"] is None)
    check("promote 归属重算", row["vehicle_id"] is not None, str(row["vehicle_id"]))
    new0 = dict(conn.execute(
        "SELECT * FROM vehicle_instances WHERE capture_id=1 AND instance_index=0").fetchone())
    new1 = dict(conn.execute(
        "SELECT * FROM vehicle_instances WHERE capture_id=1 AND instance_index=1").fetchone())
    check("promote 行对调0", new0["vehicle_box"] == "[600, 100, 1100, 500]"
          and new0["reid_embedding"] == emb2.tobytes() and new0["vehicle_color"] == "red")
    # 前面的 save 步骤已按断点续跑规则清空主车 embedding（等 embed 阶段补算），
    # 对调忠实搬移当前值：降级行拿到 NULL 是预期行为。
    check("promote 行对调1", new1["vehicle_box"] == "[0, 0, 1280, 720]"
          and new1["reid_embedding"] is None and new1["vehicle_color"] == "blue")
    edit = conn.execute("SELECT * FROM manual_box_edits WHERE capture_id=1").fetchone()
    check("promote manual 记录", edit is not None and edit["source"] == "manual_promote")
    conn.close()
    with Image.open(crop_main) as im:
        check("promote 裁剪图=实例裁剪", im.size == (500, 350), str(im.size))
    check("promote 主车实例 400", post("/api/boxedit/promote", {
        "capture_id": 1, "instance_id": new0["id"]}).status_code == 404)

    # ---- 6. 切换回退：实例文件缺失 -> 按框从原图重裁 ----
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        """INSERT INTO captures (id, date, park, order_id, filepath, capture_status,
                                 vehicle_crop_path, vehicle_box, crop_source, detection_count)
           VALUES (3, '2026-08-10', '测试园区', 'A2', ?, 'vehicle', ?, '[100,100,400,350]',
                   'PaddleDetection', 2)""", (str(orig2), str(tmp / "crop3.jpg")))
    missing_crop = tmp / "crop_missing.jpg"
    conn.execute(
        """INSERT INTO vehicle_instances (capture_id, instance_index, crop_path, vehicle_box,
                                          vehicle_det_score, reid_embedding)
           VALUES (3, 0, ?, '[100,100,400,350]', 0.9, ?)""", (str(missing_crop), emb.tobytes()))
    conn.execute(
        """INSERT INTO vehicle_instances (capture_id, instance_index, crop_path, vehicle_box,
                                          vehicle_det_score, vehicle_color, vehicle_type)
           VALUES (3, 1, ?, '[500,60,900,460]', 0.7, 'black', 'suv')""", (str(missing_crop),))
    conn.commit()
    inst_row = conn.execute(
        "SELECT id FROM vehicle_instances WHERE capture_id=3 AND instance_index=1").fetchone()
    conn.close()
    r = post("/api/boxedit/promote", {"capture_id": 3, "instance_id": inst_row["id"]})
    check("promote 回退重裁 200", r.status_code == 200, r.get_data(as_text=True))
    with Image.open(tmp / "crop3.jpg") as im:
        check("promote 回退裁剪尺寸", im.size == (400, 400), str(im.size))
    conn = sqlite3.connect(db_path)
    check("promote 回退框", conn.execute(
        "SELECT vehicle_box FROM captures WHERE id=3").fetchone()[0] == "[500, 60, 900, 460]")
    conn.close()

    # ---- 7. 撤销人工标记 ----
    r = post("/api/boxedit/reset", {"capture_id": 1})
    check("reset 200", r.status_code == 200 and r.get_json()["removed"] == 1)
    conn = sqlite3.connect(db_path)
    check("reset 已删除", conn.execute(
        "SELECT COUNT(*) FROM manual_box_edits WHERE capture_id=1").fetchone()[0] == 0)
    conn.close()

    # ---- 8. 流水线跳过：detect_pass(force) / correct_pass 不碰人工框选帧 ----
    pipe_db = tmp / "pipe.db"
    conn = sqlite3.connect(pipe_db)
    conn.row_factory = sqlite3.Row
    build_db.ensure_schema(conn)
    pipe_crop = tmp / "pipe_crop.jpg"
    make_image(pipe_crop, 300, 200)
    conn.execute(
        """INSERT INTO captures (id, date, park, order_id, filepath, capture_status,
                                 vehicle_crop_path, vehicle_box, crop_source)
           VALUES (10, '2026-08-10', '测试园区', 'B1', ?, 'vehicle', ?,
                   '[10,10,310,210]', 'PaddleDetection')""", (str(orig1), str(pipe_crop)))
    conn.execute("INSERT INTO manual_box_edits (capture_id, box, source, edited_at) "
                 "VALUES (10, '[10,10,310,210]', 'manual_box', 'now')")
    conn.commit()
    conn.close()
    det = __import__("reid_pipeline")
    res = det.detect_pass(pipe_db, tmp / "crops", None, None, force=True)
    check("detect force 跳过人工框选", res["selected"] == 0, str(res))
    check("detect force 未删裁剪", pipe_crop.is_file())
    res = det.correct_pass(pipe_db, None, None)
    check("correct 跳过人工框选", res["selected"] == 0, str(res))

    print()
    if failures:
        print(f"失败 {len(failures)} 项：{'、'.join(failures)}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
