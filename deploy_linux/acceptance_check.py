# -*- coding: utf-8 -*-
"""部署验收：两库统计 + 裁剪图/特征覆盖 + 缺文件检查。在项目根目录运行。"""
import os
import sqlite3

os.chdir("/opt/car-trace")

for db in ("app/granary.db", "app/nongyixing.db"):
    conn = sqlite3.connect(db)
    cap = conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0]
    vi = conn.execute("SELECT COUNT(*) FROM vehicle_instances").fetchone()[0]
    emb = conn.execute("SELECT COUNT(*) FROM captures WHERE reid_embedding IS NOT NULL").fetchone()[0]
    crop = conn.execute(
        "SELECT COUNT(*) FROM captures WHERE coalesce(vehicle_crop_path,'')!=''").fetchone()[0]
    noveh = conn.execute(
        "SELECT COUNT(*) FROM captures WHERE capture_status='no_vehicle'").fetchone()[0]
    missing = sum(1 for (p,) in conn.execute(
        "SELECT vehicle_crop_path FROM captures WHERE coalesce(vehicle_crop_path,'')!=''")
        if not os.path.isfile(os.path.join("app", p)))
    orders = conn.execute("SELECT COUNT(DISTINCT order_id) FROM captures").fetchone()[0]
    plates = conn.execute(
        "SELECT COUNT(*) FROM captures WHERE coalesce(plate_final,'')!=''").fetchone()[0]
    conn.close()
    print(f"{db}: 抓拍{cap} | 订单{orders} | 车辆实例{vi} | 有特征{emb} | "
          f"有裁剪{crop}(缺文件{missing}) | 无车帧{noveh} | 有车牌{plates}")
