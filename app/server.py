# -*- coding: utf-8 -*-
"""园区车辆轨迹查询 Flask 后端。

查询维度（实施文档第 12 步）：车牌、车型、时段、订单号；订单号在园区内是
聚合唯一键（不同园区可能使用同一套系统撞号），跨日期文件夹的订单会聚合成
一条完整轨迹。向量检索只读 reid_embedding（FastReID 特征），候选重排用
属性/时间过滤，不读取任何旧分类模型特征。

园区隔离：captures.park 记录抓拍所属园区（来自 data/<园区>/ 目录结构）。
车辆身份（vehicle_id）聚类、以图搜车、人工移动/合并都在园区内进行，绝不
跨园区；查询接口普遍支持可选 park 参数（缺省=全部园区，订单轨迹在订单号
跨园区撞号时要求指定园区）。

对外开放接口见文件末尾 /api/open/ 段落：只读 + token 鉴权，
token 由环境变量 OPEN_API_TOKEN 配置（未配置则整组停用）。
"""
from __future__ import annotations

import io
import csv
import hmac
import os
import re
import sqlite3
import sys
from contextlib import closing
from datetime import datetime
from functools import wraps
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_file
from PIL import Image, ImageOps

import build_db
from build_db import ensure_schema

APP_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("GRANARY_DB", APP_DIR / "granary.db"))
PLATE_CLEAN_RE = re.compile(r"[^\u4e00-\u9fffA-Z0-9挂学警港澳使领]")
app = Flask(__name__)
app.json.ensure_ascii = False
# 模板改动即时生效（Jinja 默认按进程缓存模板，改模板不重启看不到）。
app.config["TEMPLATES_AUTO_RELOAD"] = True


def normalize_plate(value: str) -> str:
    """宽松清洗：仅去除非法字符并大写，保留片段（搜索/标注用）。

    不要换成 build_db 的严格版：那里要求完整省份+字母+序号才能通过，
    会把“KDV691”这类车牌片段整体清空，导致搜索永远为空。
    """
    return PLATE_CLEAN_RE.sub("", (value or "").upper())[:10]


def resolve_capture_path(stored_path: str) -> Path:
    """同时支持开发机绝对路径与便携包内相对路径。"""
    path = Path(stored_path)
    if path.is_absolute():
        return path
    candidates = [(APP_DIR / path).resolve(), (APP_DIR.parent / path).resolve()]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def get_db() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    ensure_schema(connection)
    return connection


def positive_int(value: str, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def park_param() -> str:
    """园区过滤参数：空串=全部园区。"""
    return request.args.get("park", "").strip()


def park_where(value: str, column: str = "park") -> tuple[str, list]:
    """园区过滤 SQL 条件；value 为空表示不过滤（全部园区）。"""
    if not value:
        return "", []
    return f"coalesce({column},'')=?", [value]


def blob_to_feature(blob: bytes) -> np.ndarray:
    feature = np.frombuffer(blob, dtype=np.float32)
    return feature / (np.linalg.norm(feature) + 1e-9)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size:
        return -1.0
    return float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-9))


CAPTURE_FIELDS = """
    id, filename, date, coalesce(park,'') AS park, order_id, stage, camera_key, resolution,
    capture_time, hour, ocr_status,
    osd_plate, osd_color, osd_type,
    plate_final, plate_observed, plate_confidence, recognition_source,
    vehicle_color, vehicle_color_conf, vehicle_type, vehicle_type_conf,
    attr_model_version,
    vehicle_crop_path, vehicle_box, crop_source, detection_count, vehicle_det_score,
    (SELECT COUNT(*) FROM vehicle_instances WHERE capture_id=captures.id
     AND instance_index>0) AS extra_vehicles,
    reid_source, reid_model_version,
    quality_score, vehicle_id, capture_status, status_reason
"""

# 轨迹时间排序：未检测到车辆的图片沉底；有 OCR 时间按时间正序，OCR 失败的
# 记录排在其所在日期末尾，再按阶段顺序（入磅->质检->出磅，抓拍按 id）。
TRACK_ORDER = ("ORDER BY CASE WHEN capture_status='no_vehicle' THEN 1 ELSE 0 END ASC, "
               "coalesce(NULLIF(capture_time,''), date || ' 00:00:00') ASC, "
               "stage_rank ASC, id ASC")


@app.after_request
def disable_api_cache(response):
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/annotate")
def annotate():
    return render_template("annotate.html")


@app.route("/api/health")
def health():
    """用于部署自检；不暴露本机绝对路径。"""
    try:
        with closing(get_db()) as db:
            db.execute("SELECT 1 FROM captures LIMIT 1").fetchone()
        database = "ok"
    except sqlite3.Error as exc:
        database = f"error: {exc}"
    return jsonify({"status": "ok" if database == "ok" else "degraded",
                    "database": database, "python": sys.version.split()[0]})


@app.route("/api/parks")
def parks():
    """园区清单及各园区的抓拍/订单/车辆数量，前端园区下拉与统计用。

    park 为空字符串表示未标注园区（老库数据，如项目根目录的 2026年8月份抓拍照片）。
    """
    with closing(get_db()) as db:
        rows = db.execute(
            """SELECT coalesce(park,'') AS park, COUNT(*) AS captures,
                      COUNT(DISTINCT CASE WHEN coalesce(order_id,'')!='' THEN
                           coalesce(park,'') || '|' || order_id END) AS orders,
                      COUNT(DISTINCT vehicle_id) AS vehicles,
                      MIN(date) AS first_date, MAX(date) AS last_date
               FROM captures GROUP BY coalesce(park,'') ORDER BY park"""
        ).fetchall()
    return jsonify([dict(row) for row in rows])


@app.route("/api/dates")
def dates():
    park = park_param()
    where_sql, park_params = park_where(park)
    with closing(get_db()) as db:
        rows = db.execute(
            f"""SELECT date, COUNT(*) AS count FROM captures
                WHERE date!='' {'AND ' + where_sql if where_sql else ''}
                GROUP BY date ORDER BY date""",
            park_params,
        ).fetchall()
    return jsonify([dict(row) for row in rows])


# 颜色/车型的人工可读映射（前端下拉用；接受中英文输入）
COLOR_CODES = {"yellow": "黄色", "orange": "橙色", "green": "绿色", "gray": "灰色",
               "red": "红色", "blue": "蓝色", "white": "白色", "golden": "金色",
               "brown": "棕色", "black": "黑色"}
TYPE_CODES = {"sedan": "轿车", "suv": "SUV", "van": "厢式车", "hatchback": "两厢车",
              "mpv": "MPV", "pickup": "皮卡", "bus": "客车", "truck": "货车",
              "estate": "旅行车"}


@app.route("/api/color_type_codes")
def color_type_codes():
    return jsonify({"colors": COLOR_CODES, "types": TYPE_CODES})


@app.route("/api/capture_update", methods=["POST"])
def capture_update():
    """人工复核编辑：修改单个截取车辆的车牌/颜色/车型/时间/订单归属。

    - 颜色/车型接受中文或英文代码（如"红色"或"red"），人工确认后置信度
      记 1.0、来源记"人工复核"；
    - 订单号变更即转移归属（车辆分组随订单重算）；
    - reviewed=true 或填写备注视为完成人工复核，清除"待人工复核"标记。
    """
    payload = request.get_json(silent=True) or {}
    capture_id = positive_int(str(payload.get("id", "")))
    if not capture_id:
        return jsonify({"error": "缺少有效图片ID"}), 400
    with closing(get_db()) as db:
        row = db.execute("SELECT id, order_id, capture_time, hour FROM captures WHERE id=?",
                         (capture_id,)).fetchone()
        if not row:
            return jsonify({"error": "抓拍不存在"}), 404

        def norm_color(v):
            v = (v or "").strip()
            if not v:
                return None
            v_low = v.lower()
            if v_low in COLOR_CODES:
                return v_low
            for code, cn in COLOR_CODES.items():
                if v == cn:
                    return code
            return None

        def norm_type(v):
            v = (v or "").strip()
            if not v:
                return None
            v_low = v.lower()
            if v_low in TYPE_CODES:
                return v_low
            for code, cn in TYPE_CODES.items():
                if v == cn:
                    return code
            return None

        sets, params, touched = [], [], []

        if "plate" in payload:
            plate = normalize_plate(str(payload.get("plate", "")))
            sets.append("plate_final=?, plate_observed=CASE WHEN plate_final!='' "
                        "THEN plate_final ELSE plate_observed END, plate_confidence=?, "
                        "recognition_source=?")
            params += [plate, 1.0 if plate else None, "人工复核" if plate else ""]
            touched.append("车牌")
        if "color" in payload:
            color = norm_color(payload.get("color"))
            sets.append("vehicle_color=?, vehicle_color_conf=?")
            params += [color, 1.0 if color else None]
            touched.append("颜色")
        if "type" in payload:
            vtype = norm_type(payload.get("type"))
            sets.append("vehicle_type=?, vehicle_type_conf=?")
            params += [vtype, 1.0 if vtype else None]
            touched.append("车型")
        if "capture_time" in payload:
            ct = str(payload.get("capture_time", "")).strip()
            m = re.match(r"^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})$", ct)
            if ct and not m:
                return jsonify({"error": "时间格式应为 YYYY-MM-DD HH:MM:SS"}), 400
            sets.append("capture_time=?, hour=?")
            params += [ct if ct else None, int(m.group(4)) if m else -1]
            touched.append("时间")
        if "order_id" in payload:
            new_order = str(payload.get("order_id", "")).strip()
            if new_order and new_order != row["order_id"]:
                sets.append("order_id=?, vehicle_id=NULL")
                params += [new_order]
                touched.append("订单")
        note = str(payload.get("note", "")).strip()[:200]
        reviewed = bool(payload.get("reviewed"))
        if reviewed or note:
            base = "人工复核确认"
            if note:
                base += "：" + note
            sets.append("status_reason=?")
            params.append(base)
            touched.append("复核标记")

        if not sets:
            return jsonify({"error": "没有需要修改的字段"}), 400
        params.append(capture_id)
        db.execute(f"UPDATE captures SET {', '.join(sets)} WHERE id=?", params)
        db.commit()
        updated = db.execute(
            """SELECT id, plate_final, vehicle_color, vehicle_color_conf, vehicle_type,
                      vehicle_type_conf, capture_time, hour, order_id, status_reason
               FROM captures WHERE id=?""",
            (capture_id,),
        ).fetchone()
    return jsonify({"ok": True, "touched": touched, "item": dict(updated)})


@app.route("/api/capture/<int:capture_id>")
def capture_single(capture_id: int):
    with closing(get_db()) as db:
        row = db.execute(
            f"SELECT {CAPTURE_FIELDS} FROM captures WHERE id=?", (capture_id,)
        ).fetchone()
    if not row:
        return jsonify({"error": "抓拍不存在"}), 404
    return jsonify(dict(row))


@app.route("/api/vehicle_move", methods=["POST"])
def vehicle_move():
    """人工移动：把抓拍（或整个订单）从当前车辆聚合中拆出为新车辆，
    或移动到目标车牌所在的车辆聚合。人工决定持久化存储，重跑聚类后自动重新应用。

    payload: {capture_id, target_plate(null=新开独立聚合), scope('capture'|'order')}
    """
    payload = request.get_json(silent=True) or {}
    capture_id = positive_int(str(payload.get("capture_id", "")))
    if not capture_id:
        return jsonify({"error": "缺少有效图片ID"}), 400
    scope = payload.get("scope", "capture")
    if scope not in ("capture", "order"):
        return jsonify({"error": "scope 无效"}), 400
    target_raw = str(payload.get("target_plate", "") or "").strip()
    target_plate = normalize_plate(target_raw)
    target_key = target_plate[1:] if len(target_plate) >= 2 else ""

    with closing(get_db()) as db:
        row = db.execute(
            "SELECT id, order_id, coalesce(park,'') AS park FROM captures WHERE id=?",
            (capture_id,),
        ).fetchone()
        if not row:
            return jsonify({"error": "抓拍不存在"}), 404
        order_id, capture_park = row["order_id"], row["park"]

        if target_key:
            # 园区隔离：目标车牌只在抓拍所属园区内查找，移动不跨园区。
            hit = db.execute(
                """SELECT vehicle_id FROM captures WHERE vehicle_id IS NOT NULL
                   AND coalesce(plate_final,'')!='' AND coalesce(park,'')=?
                   AND substr(plate_final,2)=?
                   LIMIT 1""",
                (capture_park, target_key),
            ).fetchone()
            if not hit:
                return jsonify({"error": f"未找到车牌含「{target_plate}」的车辆聚合"}), 404
            new_vid = hit["vehicle_id"]
        else:
            new_vid = None  # 拆出为独立车辆

        now = datetime.now().isoformat(timespec="seconds")
        db.execute(
            """CREATE TABLE IF NOT EXISTS manual_vehicle_moves (
                 capture_id INTEGER PRIMARY KEY,
                 target_plate TEXT,
                 scope TEXT NOT NULL DEFAULT 'capture',
                 moved_at TEXT NOT NULL)"""
        )
        if scope == "order" and order_id:
            ids = [r[0] for r in db.execute(
                """SELECT id FROM captures WHERE order_id=? AND coalesce(park,'')=?
                   AND coalesce(capture_status,'vehicle')!='no_vehicle'""",
                (order_id, capture_park),
            ).fetchall()]
        else:
            ids = [capture_id]
        for cid in ids:
            db.execute(
                """INSERT INTO manual_vehicle_moves (capture_id, target_plate, scope, moved_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(capture_id) DO UPDATE SET
                     target_plate=excluded.target_plate,
                     scope=excluded.scope, moved_at=excluded.moved_at""",
                (cid, target_key or None, scope, now),
            )
            db.execute("UPDATE captures SET vehicle_id=? WHERE id=?", (new_vid, cid))
        db.commit()
    return jsonify({"ok": True, "moved": len(ids),
                    "target": target_plate or "新开独立聚合",
                    "scope": scope})


@app.route("/api/vehicle_move_delete", methods=["POST"])
def vehicle_move_delete():
    """撤销某抓拍的人工移动决定（恢复自动聚类归属）。"""
    payload = request.get_json(silent=True) or {}
    capture_id = positive_int(str(payload.get("capture_id", "")))
    if not capture_id:
        return jsonify({"error": "缺少有效图片ID"}), 400
    with closing(get_db()) as db:
        db.execute("DELETE FROM manual_vehicle_moves WHERE capture_id=?", (capture_id,))
        db.commit()
        build_db.associate_orders(db)
    return jsonify({"ok": True})


@app.route("/api/instances/<int:capture_id>")
def instances_of_capture(capture_id: int):
    """一张抓拍图里同帧检测到的其他车辆（每车一条记录）。"""
    with closing(get_db()) as db:
        rows = db.execute(
            """SELECT id, capture_id, instance_index, crop_path, vehicle_box,
                      vehicle_det_score, vehicle_color, vehicle_color_conf,
                      vehicle_type, vehicle_type_conf, lpr_plate, lpr_conf,
                      vehicle_id
               FROM vehicle_instances WHERE capture_id=? AND instance_index>0
               ORDER BY vehicle_det_score DESC""",
            (capture_id,),
        ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["crop_url"] = f"/api/instance_crop/{row['id']}"
        out.append(item)
    return jsonify(out)


@app.route("/api/instance_crop/<int:instance_id>")
def instance_crop(instance_id: int):
    with closing(get_db()) as db:
        row = db.execute(
            "SELECT crop_path FROM vehicle_instances WHERE id=?", (instance_id,)
        ).fetchone()
    if not row or not row["crop_path"]:
        return jsonify({"error": "实例裁剪不存在"}), 404
    path = resolve_capture_path(row["crop_path"])
    if not path.is_file():
        return jsonify({"error": "实例裁剪文件不存在"}), 404
    return send_file(path, conditional=True, max_age=3600)


@app.route("/api/stats")
def stats():
    park = park_param()
    where_sql, park_params = park_where(park)
    park_filter = f" WHERE {where_sql}" if where_sql else ""
    with closing(get_db()) as db:
        totals = db.execute(
            f"""SELECT COUNT(*) AS total,
                      COUNT(DISTINCT date) AS dates,
                      COUNT(DISTINCT CASE WHEN coalesce(order_id,'')!='' THEN
                           coalesce(park,'') || '|' || order_id END) AS orders,
                      COUNT(DISTINCT camera_key) AS cameras,
                      COUNT(DISTINCT vehicle_id) AS vehicles,
                      SUM(CASE WHEN vehicle_id IS NOT NULL THEN 1 ELSE 0 END) AS linked,
                      SUM(CASE WHEN reid_embedding IS NOT NULL THEN 1 ELSE 0 END) AS processed,
                      SUM(CASE WHEN coalesce(vehicle_crop_path,'')!='' THEN 1 ELSE 0 END) AS cropped,
                      SUM(CASE WHEN coalesce(plate_final,'')!='' THEN 1 ELSE 0 END) AS with_plate,
                      SUM(CASE WHEN capture_status='no_vehicle' THEN 1 ELSE 0 END) AS no_vehicle,
                      SUM(CASE WHEN detection_count>1 THEN 1 ELSE 0 END) AS multi_vehicle,
                      SUM(CASE WHEN coalesce(ocr_status,'')='fallback' THEN 1 ELSE 0 END) AS ocr_fallback,
                      SUM(CASE WHEN coalesce(ocr_status,'')='conflict' THEN 1 ELSE 0 END) AS ocr_conflict
               FROM captures{park_filter}""",
            park_params,
        ).fetchone()
        stage_rows = db.execute(
            f"""SELECT stage, COUNT(*) AS count FROM captures
                WHERE coalesce(stage,'')!=''{(' AND ' + where_sql) if where_sql else ''}
                GROUP BY stage ORDER BY count DESC""",
            park_params,
        ).fetchall()
        camera_rows = db.execute(
            f"""SELECT camera_key, COUNT(*) AS count FROM captures
                WHERE coalesce(camera_key,'')!=''{(' AND ' + where_sql) if where_sql else ''}
                GROUP BY camera_key ORDER BY camera_key""",
            park_params,
        ).fetchall()
        park_rows = db.execute(
            """SELECT coalesce(park,'') AS park, COUNT(*) AS count FROM captures
               GROUP BY coalesce(park,'') ORDER BY park"""
        ).fetchall()
    result = dict(totals)
    for key in ("linked", "processed", "cropped", "with_plate", "no_vehicle",
                "multi_vehicle", "ocr_fallback", "ocr_conflict"):
        result[key] = result[key] or 0
    result["stage_items"] = [dict(row) for row in stage_rows]
    result["camera_items"] = [dict(row) for row in camera_rows]
    result["park_items"] = [dict(row) for row in park_rows]
    return jsonify(result)


@app.route("/api/stages")
def stages():
    park = park_param()
    where_sql, park_params = park_where(park)
    with closing(get_db()) as db:
        rows = db.execute(
            f"""SELECT stage, COUNT(*) AS count FROM captures
                WHERE coalesce(stage,'')!=''{(' AND ' + where_sql) if where_sql else ''}
                GROUP BY stage ORDER BY count DESC""",
            park_params,
        ).fetchall()
    return jsonify([dict(row) for row in rows])


def order_summary(db: sqlite3.Connection, order_id: str, park: str) -> dict:
    """订单级汇总：每个订单确定一个大致发生时间和一个车牌号。

    - order_time：订单内最早的有效抓拍时间（OCR 时间），粗略代表订单发生时刻；
    - plate：整单只取一个车牌——识别车牌 plate_final 多数票（票数→置信度→
      字典序，保证结果唯一确定），整单没有识别车牌时回退摄像头 OSD 参考车牌
      的多数票；"未关联图片"桶是伪订单（混装未归单抓拍），不产出车牌。
    """
    if order_id == "未关联图片":
        row = db.execute(
            """SELECT MIN(capture_time) AS order_time FROM captures
               WHERE order_id=? AND coalesce(park,'')=? AND coalesce(capture_time,'')!=''""",
            (order_id, park)).fetchone()
        return {"order_time": row["order_time"] or "", "plate": ""}
    row = db.execute(
        """SELECT
             (SELECT MIN(capture_time) FROM captures
               WHERE order_id=? AND coalesce(park,'')=? AND coalesce(capture_time,'')!='')
               AS order_time,
             (SELECT plate_final FROM captures
               WHERE order_id=? AND coalesce(park,'')=? AND coalesce(plate_final,'')!=''
               GROUP BY plate_final
               ORDER BY COUNT(*) DESC, MAX(coalesce(plate_confidence,0)) DESC, plate_final
               LIMIT 1) AS plate_final,
             (SELECT osd_plate FROM captures
               WHERE order_id=? AND coalesce(park,'')=? AND coalesce(osd_plate,'')!=''
               GROUP BY osd_plate
               ORDER BY COUNT(*) DESC, osd_plate
               LIMIT 1) AS osd_pick""",
        (order_id, park, order_id, park, order_id, park),
    ).fetchone()
    plate = (row["plate_final"] or "") if row else ""
    if not plate and row:
        plate = normalize_plate(row["osd_pick"] or "")
    return {"order_time": (row["order_time"] or "") if row else "", "plate": plate}


@app.route("/api/orders")
def orders():
    """订单列表：订单号在园区内是聚合唯一键（不同园区可能撞号），
    同一订单跨日期文件夹只出现一行。

    rep_id 是该订单的一张已裁剪抓拍（优先），供列表缩略图使用；
    order_time / plate 是订单级汇总（大致发生时间 + 唯一车牌号）。
    """
    date = request.args.get("date", "").strip()
    stage = request.args.get("stage", "").strip()
    search = request.args.get("q", "").strip()
    park = park_param()
    limit = min(1000, positive_int(request.args.get("limit", "200"), 200))
    where, parameters = ["1=1"], []
    if date:
        where.append("c.date=?")
        parameters.append(date)
    if stage:
        where.append("c.id IN (SELECT id FROM captures WHERE stage=?)")
        parameters.append(stage)
    if search:
        where.append("c.order_id LIKE ?")
        parameters.append(f"%{search}%")
    if park:
        where.append("coalesce(c.park,'')=?")
        parameters.append(park)
    sql = f"""
        SELECT c.park, c.order_id, COUNT(*) AS count,
               COUNT(DISTINCT c.date) AS date_span,
               GROUP_CONCAT(DISTINCT c.date) AS dates,
               MIN(c.capture_time) AS first_seen, MAX(c.capture_time) AS last_seen,
               SUM(CASE WHEN coalesce(c.vehicle_crop_path,'')!='' THEN 1 ELSE 0 END) AS cropped,
               SUM(CASE WHEN c.capture_status='no_vehicle' THEN 1 ELSE 0 END) AS no_vehicle,
               MAX(c.vehicle_color) AS vehicle_color, MAX(c.vehicle_type) AS vehicle_type,
               MIN(CASE WHEN coalesce(c.vehicle_crop_path,'')!='' THEN c.id END) AS rep_id,
               MAX(oa.vehicle_color_conf) AS attr_color_conf,
               MAX(oa.vehicle_type_conf) AS attr_type_conf
        FROM captures c LEFT JOIN order_attributes oa
          ON oa.order_id=c.order_id AND coalesce(oa.park,'')=coalesce(c.park,'')
        WHERE {' AND '.join(where)}
        GROUP BY coalesce(c.park,''), c.order_id
        -- 订单列表排序：人脸/未检测到车辆占比高的沉底；有效车辆裁剪多、
        -- 属性置信度高的排前面（实施文档第 12 步的可读性要求）。
        ORDER BY
          SUM(CASE WHEN c.capture_status='no_vehicle' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) ASC,
          SUM(CASE WHEN coalesce(c.vehicle_crop_path,'')!='' THEN 1 ELSE 0 END) DESC,
          COALESCE(MAX(oa.vehicle_color_conf),0) + COALESCE(MAX(oa.vehicle_type_conf),0) DESC,
          c.order_id LIMIT ?
    """
    parameters.append(limit)
    with closing(get_db()) as db:
        rows = [dict(row) for row in db.execute(sql, parameters).fetchall()]
        for row in rows:
            attrs = db.execute(
                """SELECT vehicle_color, vehicle_color_conf, vehicle_type, vehicle_type_conf
                   FROM order_attributes WHERE order_id=? AND coalesce(park,'')=?""",
                (row["order_id"], row["park"] or ""),
            ).fetchone()
            if attrs:
                row["vehicle_color"] = attrs["vehicle_color"] or row["vehicle_color"]
                row["vehicle_type"] = attrs["vehicle_type"] or row["vehicle_type"]
                row["attr_conf"] = {"color": attrs["vehicle_color_conf"],
                                    "type": attrs["vehicle_type_conf"]}
            row.update(order_summary(db, row["order_id"], row["park"] or ""))
    return jsonify(rows)


@app.route("/api/order/<order_id>")
def order_detail(order_id: str):
    """订单内完整轨迹：跨日期文件夹自动聚合，按时间+阶段顺序展示。

    订单号在园区内唯一：不带 park 参数且该订单号只存在于一个园区时直接
    返回；撞号（存在于多个园区）时返回 400 并列出园区，调用方需指定 park。
    """
    park = park_param()
    with closing(get_db()) as db:
        park_rows = db.execute(
            """SELECT DISTINCT coalesce(park,'') FROM captures WHERE order_id=?""",
            (order_id,),
        ).fetchall()
        parks = [row[0] for row in park_rows]
        if not parks:
            return jsonify({"error": "订单不存在", "items": []}), 404
        if not park and len(parks) > 1:
            return jsonify({"error": f"订单号 {order_id} 在多个园区存在，请指定 park 参数",
                            "parks": parks}), 400
        if park and park not in parks:
            return jsonify({"error": f"订单不存在（园区 {park}）", "items": []}), 404
        park_clause, park_params = park_where(park)
        rows = db.execute(
            f"""SELECT {CAPTURE_FIELDS} FROM captures
                WHERE order_id=?{( ' AND ' + park_clause) if park_clause else ''}
                {TRACK_ORDER}""",
            [order_id, *park_params],
        ).fetchall()
        attrs = db.execute(
            """SELECT * FROM order_attributes WHERE order_id=? AND coalesce(park,'')=?""",
            (order_id, park if park else parks[0]),
        ).fetchone()
        summary = order_summary(db, order_id, park if park else parks[0])
        items = annotate_order_plates([dict(row) for row in rows], db)
    if not rows:
        return jsonify({"error": "订单不存在", "items": []}), 404
    return jsonify({"order_id": order_id, "park": park if park else parks[0],
                    "order_time": summary["order_time"], "plate": summary["plate"],
                    "attributes": dict(attrs) if attrs else None,
                    "items": items})


@app.route("/api/search")
def search():
    """综合查询：车牌（片段+形近归一）、车型、时段、订单号，均走 SQL 结构化字段。"""
    plate = normalize_plate(request.args.get("plate", ""))
    vehicle_type = request.args.get("vehicle_type", "").strip()
    start_hour = request.args.get("start_hour", "").strip()
    end_hour = request.args.get("end_hour", "").strip()
    order_id = request.args.get("order_id", "").strip()
    date = request.args.get("date", "").strip()
    park = park_param()
    limit = min(2000, positive_int(request.args.get("limit", "500"), 500))
    where, parameters = [], []
    if plate:
        where.append("(replace(upper(plate_final),' ','') LIKE ? "
                     "OR replace(upper(osd_plate),' ','') LIKE ?)")
        parameters.extend([f"%{plate}%", f"%{plate}%"])
    if vehicle_type:
        where.append("vehicle_type=?")
        parameters.append(vehicle_type)
    if start_hour != "":
        where.append("hour>=?")
        parameters.append(positive_int(start_hour))
    if end_hour != "":
        where.append("hour<=?")
        parameters.append(positive_int(end_hour, 23))
    if order_id:
        where.append("order_id LIKE ?")
        parameters.append(f"%{order_id}%")
    if date:
        where.append("date=?")
        parameters.append(date)
    park_clause, park_params = park_where(park)
    if park_clause:
        where.append(park_clause)
        parameters.extend(park_params)
    if not where:
        return jsonify({"error": "至少提供一个查询条件", "items": []}), 400
    parameters.append(limit)
    with closing(get_db()) as db:
        rows = db.execute(
            f"""SELECT {CAPTURE_FIELDS} FROM captures
                WHERE {' AND '.join(where)}
                ORDER BY date DESC, capture_time DESC, id DESC LIMIT ?""",
            parameters,
        ).fetchall()
    return jsonify([dict(row) for row in rows])


@app.route("/api/facets")
def facets():
    """轻量枚举值：车型、颜色（属性模型产出），供前端下拉框。"""
    park = park_param()
    where_sql, park_params = park_where(park)
    park_filter = f" AND {where_sql}" if where_sql else ""
    with closing(get_db()) as db:
        types = [row[0] for row in db.execute(
            f"""SELECT DISTINCT vehicle_type FROM captures
               WHERE coalesce(vehicle_type,'')!=''{park_filter} ORDER BY vehicle_type""",
            park_params)]
        colors = [row[0] for row in db.execute(
            f"""SELECT DISTINCT vehicle_color FROM captures
               WHERE coalesce(vehicle_color,'')!=''{park_filter} ORDER BY vehicle_color""",
            park_params)]
    return jsonify({"types": types, "colors": colors})


@app.route("/api/captures")
def captures():
    """按日期/阶段/订单浏览原始抓拍，便于检查分类和识别质量。"""
    date = request.args.get("date", "").strip()
    stage = request.args.get("stage", "").strip()
    order_id = request.args.get("order_id", "").strip()
    park = park_param()
    limit = min(500, positive_int(request.args.get("limit", "200"), 200))
    offset = positive_int(request.args.get("offset", "0"))
    where, parameters = ["1=1"], []
    if date:
        where.append("date=?")
        parameters.append(date)
    if stage:
        where.append("stage=?")
        parameters.append(stage)
    if order_id:
        where.append("order_id=?")
        parameters.append(order_id)
    park_clause, park_params = park_where(park)
    if park_clause:
        where.append(park_clause)
        parameters.extend(park_params)
    parameters.extend([limit, offset])
    with closing(get_db()) as db:
        rows = db.execute(
            f"SELECT {CAPTURE_FIELDS} FROM captures WHERE {' AND '.join(where)} "
            "ORDER BY date, capture_time, id LIMIT ? OFFSET ?", parameters
        ).fetchall()
    return jsonify([dict(row) for row in rows])


@app.route("/api/vehicles")
def vehicles():
    """车辆分组列表（左侧竖栏）：date 缺省时聚合全部日期。

    vehicle_id 聚类按园区隔离，因此每辆车只属于一个园区；可选 park 参数
    用于只看某个园区。
    """
    date = request.args.get("date", "").strip()
    stage = request.args.get("stage", "").strip()
    park = park_param()
    where = []
    parameters: list[object] = []
    if date:
        where.append("date=?")
        parameters.append(date)
    if stage:
        where.append("stage=?")
        parameters.append(stage)
    park_clause, park_params = park_where(park)
    if park_clause:
        where.append(park_clause)
        parameters.extend(park_params)
    if where:
        where.append("vehicle_id IS NOT NULL")
    else:
        where = ["vehicle_id IS NOT NULL"]
    where_sql = " AND ".join(where)
    sql = f"""
        WITH filtered AS (
          SELECT * FROM captures WHERE {where_sql}
        ), ranked AS (
          SELECT id, vehicle_id,
                 ROW_NUMBER() OVER (
                   PARTITION BY vehicle_id
                   ORDER BY coalesce(quality_score,0) DESC,
                            coalesce(plate_confidence,0) DESC, id
                 ) AS rank_no
          FROM filtered
        )
        SELECT f.vehicle_id, r.id AS rep_id, COUNT(*) AS count,
               MIN(coalesce(f.park,'')) AS park,
               GROUP_CONCAT(DISTINCT f.order_id) AS order_ids,
               (SELECT c2.plate_final FROM captures c2
                WHERE c2.vehicle_id=f.vehicle_id AND coalesce(c2.plate_final,'')!=''
                GROUP BY c2.plate_final ORDER BY COUNT(*) DESC, c2.plate_final LIMIT 1) AS plate,
               MAX(coalesce(f.plate_confidence,0)) AS plate_confidence,
               MAX(f.vehicle_color) AS color, MAX(f.vehicle_type) AS vehicle_type,
               COUNT(DISTINCT f.camera_key) AS camera_count,
               MIN(f.capture_time) AS first_seen, MAX(f.capture_time) AS last_seen
        FROM filtered f
        JOIN ranked r ON r.vehicle_id=f.vehicle_id AND r.rank_no=1
        GROUP BY f.vehicle_id, r.id
        ORDER BY count DESC, first_seen
    """
    with closing(get_db()) as db:
        rows = [dict(row) for row in db.execute(sql, parameters).fetchall()]
    return jsonify(rows)


def _plate_suspect(item: dict, main_plate: str) -> bool:
    """可疑照片判定：照片车牌（plate_final 为空时退回 OSD 参考车牌）与
    订单主车车牌去省份差 >=2（与 build_db 误读清洗同一判据）——多为排队
    车辆或人工误放。主牌缺失时无从判定，一律不算可疑。"""
    if not main_plate:
        return False
    if item.get("plate_final"):
        return build_db._plate_edit_distance(item["plate_final"], main_plate) >= 2
    osd = item.get("osd_plate") or ""
    return bool(osd) and build_db._plate_edit_distance(osd, main_plate) >= 2


def annotate_order_plates(items: list[dict], db: sqlite3.Connection) -> list[dict]:
    """给抓拍行补订单主车车牌（order_main_plate）与可疑标记（plate_mismatch）。"""
    cache: dict[tuple[str, str], str] = {}
    for item in items:
        oid = item.get("order_id") or ""
        if not oid or oid == "未关联图片":
            item["plate_mismatch"] = False
            continue
        key = (item.get("park") or "", oid)
        if key not in cache:
            cache[key] = order_summary(db, oid, key[0])["plate"]
        main_plate = cache[key]
        item["order_main_plate"] = main_plate
        item["plate_mismatch"] = _plate_suspect(item, main_plate)
    return items


@app.route("/api/track")
def track():
    vehicle_id = positive_int(request.args.get("vid", ""))
    date = request.args.get("date", "").strip()
    plate = normalize_plate(request.args.get("plate", ""))
    park = park_param()
    with closing(get_db()) as db:
        if vehicle_id:
            # 车辆身份按园区隔离：vehicle_id 本身不会跨园区。
            rows = db.execute(
                f"SELECT {CAPTURE_FIELDS} FROM captures WHERE vehicle_id=? {TRACK_ORDER}",
                (vehicle_id,),
            ).fetchall()
        elif plate:
            # 支持车牌片段；同时命中识别结果与摄像头OSD参考值。
            # 同牌车可能出现在多个园区（身份独立），可选 park 参数只看一个园区。
            where = ["coalesce(capture_status,'vehicle')!='no_vehicle'",
                     "(replace(upper(plate_final),' ','') LIKE ? "
                     "OR replace(upper(osd_plate),' ','') LIKE ?)"]
            parameters: list[object] = [f"%{plate}%", f"%{plate}%"]
            if date:
                where.append("date=?")
                parameters.append(date)
            park_clause, park_params = park_where(park)
            if park_clause:
                where.append(park_clause)
                parameters.extend(park_params)
            rows = db.execute(
                f"SELECT {CAPTURE_FIELDS} FROM captures WHERE {' AND '.join(where)} "
                f"{TRACK_ORDER}", parameters
            ).fetchall()
        else:
            return jsonify([])
        items = annotate_order_plates([dict(row) for row in rows], db)
    return jsonify(items)


@app.route("/api/track_by_img")
def track_by_img():
    """以指定抓拍的 FastReID 特征为 query，同园区内余弦检索 + 属性重排。

    园区隔离：候选只在抓拍所属园区内检索，不跨园区匹配。
    """
    capture_id = positive_int(request.args.get("id", ""))
    if not capture_id:
        return jsonify({"error": "缺少有效图片ID", "items": []}), 400
    with closing(get_db()) as db:
        query = db.execute(
            """SELECT reid_embedding, date, camera_key, vehicle_id, vehicle_color, vehicle_type,
                      coalesce(park,'') AS park,
                      coalesce(capture_status,'vehicle') AS capture_status,
                      coalesce(plate_final,'') AS plate_final
               FROM captures WHERE id=?""",
            (capture_id,),
        ).fetchone()
        if not query:
            return jsonify({"error": "图片不存在", "items": []}), 404
        if query["capture_status"] == "no_vehicle":
            rows = db.execute(
                f"""SELECT {CAPTURE_FIELDS} FROM captures
                    WHERE order_id=(SELECT order_id FROM captures WHERE id=?)
                      AND coalesce(park,'')=(SELECT coalesce(park,'') FROM captures WHERE id=?)
                      AND capture_status='no_vehicle' {TRACK_ORDER}""",
                (capture_id, capture_id),
            ).fetchall()
            return jsonify({"method": "未识别到车辆", "vehicle_id": None,
                            "items": [dict(row) for row in rows]})
        if query["vehicle_id"]:
            rows = db.execute(
                f"SELECT {CAPTURE_FIELDS} FROM captures WHERE vehicle_id=? {TRACK_ORDER}",
                (query["vehicle_id"],),
            ).fetchall()
            return jsonify({"method": "关联车辆", "vehicle_id": query["vehicle_id"],
                            "items": [dict(row) for row in rows]})
        if not query["reid_embedding"]:
            return jsonify({"error": "该图片尚未提取外观特征", "items": []}), 422

        query_feature = blob_to_feature(query["reid_embedding"])
        candidates = db.execute(
            f"""SELECT {CAPTURE_FIELDS}, reid_embedding AS feature FROM captures
                WHERE reid_embedding IS NOT NULL
                  AND coalesce(park,'')=?
                  AND coalesce(capture_status,'vehicle')!='no_vehicle'""",
            (query["park"],),
        ).fetchall()

    items = []
    for row in candidates:
        similarity = cosine(query_feature, blob_to_feature(row["feature"]))
        # 属性重排：车型冲突是“不同车”的强证据，直接剔除；颜色只是软证据。
        type_conflict = (query["vehicle_type"] and row["vehicle_type"]
                         and query["vehicle_type"] != row["vehicle_type"])
        if type_conflict:
            continue
        same_plate = query["plate_final"] and row["plate_final"] == query["plate_final"]
        same_color = (query["vehicle_color"] and row["vehicle_color"]
                      and query["vehicle_color"] == row["vehicle_color"])
        threshold = 0.84 if same_plate else 0.90
        if same_color and similarity >= threshold - 0.02:
            threshold -= 0.02  # 颜色一致放宽阈值
        if similarity >= threshold:
            item = {key: row[key] for key in row.keys() if key != "feature"}
            item["similarity"] = round(similarity, 3)
            items.append(item)
    items.sort(key=lambda item: item["similarity"], reverse=True)
    return jsonify({"method": "外观相似度", "items": items})


@app.route("/api/thumb/<int:capture_id>")
def thumbnail(capture_id: int):
    with closing(get_db()) as db:
        row = db.execute("SELECT filepath FROM captures WHERE id=?", (capture_id,)).fetchone()
    if not row:
        return "not found", 404
    path = resolve_capture_path(row["filepath"])
    if not path.is_file():
        return "not found", 404
    try:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((480, 320))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=86, optimize=True)
        buffer.seek(0)
        return send_file(buffer, mimetype="image/jpeg", max_age=3600)
    except (OSError, ValueError):
        return "invalid image", 422


@app.route("/api/image/<int:capture_id>")
def full_image(capture_id: int):
    with closing(get_db()) as db:
        row = db.execute("SELECT filepath FROM captures WHERE id=?", (capture_id,)).fetchone()
    if not row or not resolve_capture_path(row["filepath"]).is_file():
        return "not found", 404
    return send_file(resolve_capture_path(row["filepath"]), conditional=True, max_age=3600)


@app.route("/api/crop/<int:capture_id>")
def vehicle_crop(capture_id: int):
    """返回 PaddleDetection 生成的整车裁剪；未运行裁剪流水线时返回明确状态。"""
    with closing(get_db()) as db:
        row = db.execute(
            "SELECT vehicle_crop_path FROM captures WHERE id=?", (capture_id,)
        ).fetchone()
    if not row or not row["vehicle_crop_path"]:
        return jsonify({"error": "该抓拍尚未生成 PaddleDetection 车辆裁剪"}), 404
    path = resolve_capture_path(row["vehicle_crop_path"])
    if not path.is_file():
        return jsonify({"error": "车辆裁剪文件不存在，请重新运行裁剪流水线"}), 404
    return send_file(path, conditional=True, max_age=3600)


# ---------------------------------------------------------------------------
# 人工标注（判定规则的复核入口）
# ---------------------------------------------------------------------------

def ensure_annotation_tables(db: sqlite3.Connection) -> None:
    """创建独立的人工真值表；不改写算法产生的 vehicle_id。"""
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS annotation_identities (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          display_name TEXT NOT NULL DEFAULT '',
          confirmed_plate TEXT NOT NULL DEFAULT '',
          notes TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS capture_annotations (
          capture_id INTEGER PRIMARY KEY,
          identity_id INTEGER,
          usability TEXT NOT NULL DEFAULT 'valid'
            CHECK (usability IN ('valid','invalid','uncertain')),
          corrected_plate TEXT NOT NULL DEFAULT '',
          notes TEXT NOT NULL DEFAULT '',
          annotator TEXT NOT NULL DEFAULT '',
          updated_at TEXT NOT NULL,
          FOREIGN KEY (capture_id) REFERENCES captures(id),
          FOREIGN KEY (identity_id) REFERENCES annotation_identities(id)
        );
        CREATE TABLE IF NOT EXISTS pair_annotations (
          left_capture_id INTEGER NOT NULL,
          right_capture_id INTEGER NOT NULL,
          relation TEXT NOT NULL CHECK (relation IN ('same','different','uncertain')),
          notes TEXT NOT NULL DEFAULT '',
          annotator TEXT NOT NULL DEFAULT '',
          updated_at TEXT NOT NULL,
          PRIMARY KEY (left_capture_id, right_capture_id),
          CHECK (left_capture_id < right_capture_id),
          FOREIGN KEY (left_capture_id) REFERENCES captures(id),
          FOREIGN KEY (right_capture_id) REFERENCES captures(id)
        );
        CREATE TABLE IF NOT EXISTS annotation_audit (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          action TEXT NOT NULL,
          capture_id INTEGER,
          related_capture_id INTEGER,
          identity_id INTEGER,
          details TEXT NOT NULL DEFAULT '',
          annotator TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_capture_annotation_identity
          ON capture_annotations(identity_id);
        CREATE INDEX IF NOT EXISTS idx_capture_annotation_usability
          ON capture_annotations(usability);
        """
    )


def annotation_capture(db: sqlite3.Connection, capture_id: int) -> sqlite3.Row | None:
    return db.execute(
        f"""SELECT c.*, ca.identity_id, ca.usability,
                   ca.corrected_plate, ca.notes AS annotation_notes,
                   ca.annotator, ca.updated_at AS annotation_updated_at,
                   ai.display_name AS identity_name,
                   ai.confirmed_plate AS identity_plate,
                   (SELECT COUNT(*) FROM capture_annotations members
                    WHERE members.identity_id=ca.identity_id) AS identity_size
            FROM (SELECT {CAPTURE_FIELDS} FROM captures WHERE id=?) c
            LEFT JOIN capture_annotations ca ON ca.capture_id=c.id
            LEFT JOIN annotation_identities ai ON ai.id=ca.identity_id
            """,
        (capture_id,),
    ).fetchone()


def upsert_capture_annotation(
    db: sqlite3.Connection,
    capture_id: int,
    identity_id: int | None,
    usability: str,
    annotator: str,
    corrected_plate: str = "",
    notes: str = "",
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    db.execute(
        """INSERT INTO capture_annotations
             (capture_id, identity_id, usability, corrected_plate, notes, annotator, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(capture_id) DO UPDATE SET
             identity_id=excluded.identity_id,
             usability=excluded.usability,
             corrected_plate=CASE WHEN excluded.corrected_plate!=''
                                  THEN excluded.corrected_plate
                                  ELSE capture_annotations.corrected_plate END,
             notes=CASE WHEN excluded.notes!='' THEN excluded.notes ELSE capture_annotations.notes END,
             annotator=excluded.annotator, updated_at=excluded.updated_at""",
        (capture_id, identity_id, usability, normalize_plate(corrected_plate),
         notes.strip()[:500], annotator[:80], now),
    )


def parse_vehicle_id_list(value: object) -> list[int]:
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = re.split(r"[,，;；\s]+", str(value or ""))
    result: list[int] = []
    for item in values:
        try:
            vehicle_id = int(item)
        except (TypeError, ValueError):
            continue
        if vehicle_id > 0 and vehicle_id not in result:
            result.append(vehicle_id)
    return result


@app.route("/api/annotations/stats")
def annotation_stats():
    with closing(get_db()) as db:
        ensure_annotation_tables(db)
        row = db.execute(
            """SELECT
                 (SELECT COUNT(*) FROM captures
                  WHERE coalesce(capture_status,'vehicle')!='no_vehicle') AS total,
                 (SELECT COUNT(*) FROM capture_annotations) AS reviewed,
                 (SELECT COUNT(*) FROM capture_annotations
                  WHERE identity_id IS NOT NULL AND usability='valid') AS assigned,
                 (SELECT COUNT(*) FROM capture_annotations
                  WHERE usability='invalid') AS invalid,
                 (SELECT COUNT(*) FROM capture_annotations
                  WHERE usability='uncertain') AS uncertain,
                 (SELECT COUNT(DISTINCT identity_id) FROM capture_annotations
                  WHERE identity_id IS NOT NULL) AS identities,
                 (SELECT COUNT(*) FROM pair_annotations WHERE relation='same') AS same_pairs,
                 (SELECT COUNT(*) FROM pair_annotations WHERE relation='different') AS different_pairs"""
        ).fetchone()
        db.commit()
    result = dict(row)
    result["remaining"] = max(0, result["total"] - result["reviewed"])
    return jsonify(result)


@app.route("/api/annotations/queue")
def annotation_queue():
    date = request.args.get("date", "").strip()
    stage = request.args.get("stage", "").strip()
    state = request.args.get("state", "unlabeled").strip()
    park = park_param()
    limit = min(1000, positive_int(request.args.get("limit", "500"), 500))
    where = ["coalesce(c.capture_status,'vehicle')!='no_vehicle'"]
    parameters: list[object] = []
    if date:
        where.append("c.date=?")
        parameters.append(date)
    if stage:
        where.append("c.stage=?")
        parameters.append(stage)
    park_clause, park_params = park_where(park, column="c.park")
    if park_clause:
        where.append(park_clause)
        parameters.extend(park_params)
    if state == "unlabeled":
        where.append("ca.capture_id IS NULL")
    elif state == "assigned":
        where.append("ca.identity_id IS NOT NULL AND ca.usability='valid'")
    elif state in {"invalid", "uncertain"}:
        where.append("ca.usability=?")
        parameters.append(state)
    elif state == "reviewed":
        where.append("ca.capture_id IS NOT NULL")
    parameters.append(limit)
    with closing(get_db()) as db:
        ensure_annotation_tables(db)
        rows = db.execute(
            f"""SELECT c.id, c.capture_time, c.stage, c.order_id, c.filename,
                       c.plate_final, c.vehicle_color, c.vehicle_type,
                       c.quality_score, c.vehicle_id,
                       c.vehicle_crop_path, c.crop_source, c.reid_source,
                       ca.identity_id, ca.usability, ca.corrected_plate,
                       ai.display_name AS identity_name,
                       (SELECT COUNT(*) FROM capture_annotations members
                        WHERE members.identity_id=ca.identity_id) AS identity_size
                FROM captures c
                LEFT JOIN capture_annotations ca ON ca.capture_id=c.id
                LEFT JOIN annotation_identities ai ON ai.id=ca.identity_id
                WHERE {' AND '.join(where)}
                ORDER BY CASE WHEN ca.capture_id IS NULL THEN 0 ELSE 1 END,
                         c.capture_time, c.id LIMIT ?""",
            parameters,
        ).fetchall()
        db.commit()
    return jsonify([dict(row) for row in rows])


@app.route("/api/annotations/item/<int:capture_id>")
def annotation_item(capture_id: int):
    candidate_limit = min(60, positive_int(request.args.get("limit", "24"), 24))
    with closing(get_db()) as db:
        ensure_annotation_tables(db)
        target = annotation_capture(db, capture_id)
        if not target:
            return jsonify({"error": "抓拍不存在"}), 404
        target_feature_row = db.execute(
            "SELECT reid_embedding AS feature, coalesce(park,'') AS park "
            "FROM captures WHERE id=?", (capture_id,)
        ).fetchone()
        target_feature = (blob_to_feature(target_feature_row["feature"])
                          if target_feature_row and target_feature_row["feature"] else None)
        candidates: list[tuple[float, float, int]] = []
        if target_feature is not None:
            # 候选只在同园区内找：外观相似不代表可以跨园区关联。
            rows = db.execute(
                """SELECT id, reid_embedding AS feature, plate_final, vehicle_id FROM captures
                   WHERE id!=? AND reid_embedding IS NOT NULL
                     AND coalesce(park,'')=?
                     AND coalesce(capture_status,'vehicle')!='no_vehicle'""",
                (capture_id, target_feature_row["park"]),
            ).fetchall()
            target_plate = normalize_plate(target["plate_final"] or "")
            for row in rows:
                similarity = cosine(target_feature, blob_to_feature(row["feature"]))
                rank_score = similarity
                candidate_plate = normalize_plate(row["plate_final"] or "")
                if target_plate and candidate_plate == target_plate:
                    rank_score += 0.05
                if target["vehicle_id"] and row["vehicle_id"] == target["vehicle_id"]:
                    rank_score += 0.03
                candidates.append((rank_score, similarity, int(row["id"])))
        candidate_rows = []
        for rank_score, similarity, candidate_id in sorted(candidates, reverse=True)[:candidate_limit]:
            candidate = annotation_capture(db, candidate_id)
            if not candidate:
                continue
            left_id, right_id = sorted((capture_id, candidate_id))
            relation = db.execute(
                """SELECT relation FROM pair_annotations
                   WHERE left_capture_id=? AND right_capture_id=?""",
                (left_id, right_id),
            ).fetchone()
            item = dict(candidate)
            item["similarity"] = round(similarity, 4)
            item["pair_relation"] = relation["relation"] if relation else ""
            candidate_rows.append(item)
        db.commit()
    return jsonify({"item": dict(target), "candidates": candidate_rows})


@app.route("/api/annotations/capture", methods=["POST"])
def save_capture_annotation():
    payload = request.get_json(silent=True) or {}
    capture_id = positive_int(str(payload.get("capture_id", "")))
    action = str(payload.get("action", "update"))
    usability = str(payload.get("usability", "valid"))
    if not capture_id or usability not in {"valid", "invalid", "uncertain"}:
        return jsonify({"error": "抓拍编号或可用状态无效"}), 400
    annotator = str(payload.get("annotator", "")).strip()[:80]
    with closing(get_db()) as db:
        ensure_annotation_tables(db)
        capture = db.execute(
            "SELECT id FROM captures WHERE id=? AND coalesce(capture_status,'vehicle')!='no_vehicle'",
            (capture_id,),
        ).fetchone()
        if not capture:
            return jsonify({"error": "抓拍不存在或没有有效车辆"}), 404
        existing = db.execute(
            "SELECT identity_id FROM capture_annotations WHERE capture_id=?", (capture_id,)
        ).fetchone()
        identity_id = int(existing["identity_id"]) if existing and existing["identity_id"] else None
        now = datetime.now().isoformat(timespec="seconds")
        if action == "new_identity":
            cursor = db.execute(
                """INSERT INTO annotation_identities
                     (display_name, confirmed_plate, notes, created_at, updated_at)
                   VALUES ('', ?, '', ?, ?)""",
                (normalize_plate(str(payload.get("corrected_plate", ""))), now, now),
            )
            identity_id = int(cursor.lastrowid)
        elif action == "assign_identity":
            identity_id = positive_int(str(payload.get("identity_id", ""))) or None
            if identity_id and not db.execute(
                "SELECT 1 FROM annotation_identities WHERE id=?", (identity_id,)
            ).fetchone():
                return jsonify({"error": "目标真值车辆不存在"}), 404
        elif action == "clear_identity":
            identity_id = None
        upsert_capture_annotation(
            db, capture_id, identity_id, usability, annotator,
            str(payload.get("corrected_plate", "")), str(payload.get("notes", "")),
        )
        db.execute(
            """INSERT INTO annotation_audit
                 (action, capture_id, identity_id, details, annotator, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (action, capture_id, identity_id, usability, annotator, now),
        )
        db.commit()
        result = annotation_capture(db, capture_id)
    return jsonify({"ok": True, "item": dict(result)})


@app.route("/api/annotations/pair", methods=["POST"])
def save_pair_annotation():
    payload = request.get_json(silent=True) or {}
    first_id = positive_int(str(payload.get("capture_id", "")))
    second_id = positive_int(str(payload.get("candidate_id", "")))
    relation = str(payload.get("relation", ""))
    annotator = str(payload.get("annotator", "")).strip()[:80]
    if not first_id or not second_id or first_id == second_id:
        return jsonify({"error": "需要两张不同的有效抓拍"}), 400
    if relation not in {"same", "different", "uncertain"}:
        return jsonify({"error": "关系必须是同车、不同车或不确定"}), 400
    left_id, right_id = sorted((first_id, second_id))
    now = datetime.now().isoformat(timespec="seconds")
    with closing(get_db()) as db:
        ensure_annotation_tables(db)
        found = db.execute(
            """SELECT id FROM captures WHERE id IN (?, ?)
               AND coalesce(capture_status,'vehicle')!='no_vehicle'""",
            (left_id, right_id),
        ).fetchall()
        if len(found) != 2:
            return jsonify({"error": "候选抓拍不存在或没有有效车辆"}), 404
        annotations = {
            int(row["capture_id"]): row["identity_id"]
            for row in db.execute(
                "SELECT capture_id, identity_id FROM capture_annotations WHERE capture_id IN (?, ?)",
                (left_id, right_id),
            ).fetchall()
        }
        first_identity = annotations.get(first_id)
        second_identity = annotations.get(second_id)
        merged_identity = None
        message = ""
        if relation == "same":
            identity_ids = {int(value) for value in (first_identity, second_identity) if value}
            if len(identity_ids) > 1 and not bool(payload.get("confirm_merge")):
                return jsonify({"error": "两张抓拍已属于不同真值车辆，确认后才能合并",
                                "requires_confirmation": True,
                                "identity_ids": sorted(identity_ids)}), 409
            if not identity_ids:
                cursor = db.execute(
                    """INSERT INTO annotation_identities
                         (display_name, confirmed_plate, notes, created_at, updated_at)
                       VALUES ('', '', '', ?, ?)""", (now, now)
                )
                merged_identity = int(cursor.lastrowid)
            elif len(identity_ids) == 1:
                merged_identity = next(iter(identity_ids))
            else:
                counts = {
                    identity_id: db.execute(
                        "SELECT COUNT(*) FROM capture_annotations WHERE identity_id=?",
                        (identity_id,),
                    ).fetchone()[0]
                    for identity_id in identity_ids
                }
                merged_identity = sorted(identity_ids, key=lambda value: (-counts[value], value))[0]
                sources = [value for value in identity_ids if value != merged_identity]
                for source in sources:
                    db.execute(
                        "UPDATE capture_annotations SET identity_id=?, annotator=?, updated_at=? WHERE identity_id=?",
                        (merged_identity, annotator, now, source),
                    )
                message = "已合并两个真值车辆组"
            upsert_capture_annotation(db, first_id, merged_identity, "valid", annotator)
            upsert_capture_annotation(db, second_id, merged_identity, "valid", annotator)
        elif relation == "different" and first_identity and first_identity == second_identity:
            cursor = db.execute(
                """INSERT INTO annotation_identities
                     (display_name, confirmed_plate, notes, created_at, updated_at)
                   VALUES ('', '', '', ?, ?)""", (now, now)
            )
            new_identity = int(cursor.lastrowid)
            upsert_capture_annotation(db, second_id, new_identity, "valid", annotator)
            message = "候选已从原真值车辆组拆出"
        db.execute(
            """INSERT INTO pair_annotations
                 (left_capture_id, right_capture_id, relation, notes, annotator, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(left_capture_id, right_capture_id) DO UPDATE SET
                 relation=excluded.relation, notes=excluded.notes,
                 annotator=excluded.annotator, updated_at=excluded.updated_at""",
            (left_id, right_id, relation, str(payload.get("notes", ""))[:500], annotator, now),
        )
        db.execute(
            """INSERT INTO annotation_audit
                 (action, capture_id, related_capture_id, identity_id, details, annotator, created_at)
               VALUES ('pair', ?, ?, ?, ?, ?, ?)""",
            (first_id, second_id, merged_identity, relation, annotator, now),
        )
        db.commit()
        first = annotation_capture(db, first_id)
    return jsonify({"ok": True, "relation": relation, "message": message,
                    "item": dict(first)})


@app.route("/api/annotations/export.csv")
def export_annotations():
    output = io.StringIO()
    output.write("\ufeff")
    writer = csv.writer(output)
    writer.writerow(["image_path", "pid", "camid", "capture_id", "capture_time",
                     "plate", "quality_score", "annotator"])
    with closing(get_db()) as db:
        ensure_annotation_tables(db)
        rows = db.execute(
            """SELECT c.filepath, ca.identity_id, c.camera_key, c.id, c.capture_time,
                      coalesce(nullif(ca.corrected_plate,''), c.plate_final, '') AS plate,
                      c.quality_score, ca.annotator
               FROM capture_annotations ca JOIN captures c ON c.id=ca.capture_id
               WHERE ca.identity_id IS NOT NULL AND ca.usability='valid'
               ORDER BY ca.identity_id, c.date, c.capture_time, c.id"""
        ).fetchall()
        db.commit()
    writer.writerows(rows)
    return Response(output.getvalue(), mimetype="text/csv; charset=utf-8",
                    headers={"Content-Disposition":
                             "attachment; filename=reid_annotations.csv"})


@app.route("/api/vehicle-merge", methods=["POST"])
def merge_vehicles():
    """人工确认多个车辆编号属于同一辆车，并保留指定的主编号。"""
    payload = request.get_json(silent=True) or {}
    vehicle_ids = parse_vehicle_id_list(payload.get("vehicle_ids"))
    if len(vehicle_ids) < 2:
        return jsonify({"error": "至少输入两个不同的车辆编号，例如：184,185,186,203"}), 400
    target_id = positive_int(str(payload.get("target_vehicle_id", "")), vehicle_ids[0])
    if target_id not in vehicle_ids:
        return jsonify({"error": "保留的主车辆编号必须包含在待合并编号中"}), 400

    with closing(get_db()) as db:
        db.execute(
            """CREATE TABLE IF NOT EXISTS manual_vehicle_merges (
                 source_vehicle_id INTEGER PRIMARY KEY,
                 target_vehicle_id INTEGER NOT NULL,
                 created_at TEXT NOT NULL
               )"""
        )
        placeholders = ",".join("?" for _ in vehicle_ids)
        existing = db.execute(
            f"SELECT vehicle_id, COUNT(*) AS count FROM captures "
            f"WHERE vehicle_id IN ({placeholders}) GROUP BY vehicle_id",
            vehicle_ids,
        ).fetchall()
        found = {int(row["vehicle_id"]): int(row["count"]) for row in existing}
        missing = [vehicle_id for vehicle_id in vehicle_ids if vehicle_id not in found]
        if missing:
            return jsonify({"error": f"车辆编号不存在或已被合并：{', '.join(map(str, missing))}"}), 404
        # 园区隔离：车辆身份按园区独立，禁止跨园区人工合并。
        vehicle_parks = [row[0] for row in db.execute(
            f"SELECT DISTINCT coalesce(park,'') FROM captures "
            f"WHERE vehicle_id IN ({placeholders})", vehicle_ids)]
        if len(vehicle_parks) > 1:
            return jsonify({
                "error": f"待合并车辆分属不同园区（{'、'.join(vehicle_parks)}），"
                         "车辆身份按园区隔离，不允许跨园区合并",
                "parks": sorted(vehicle_parks)}), 400

        # 将旧映射中指向本次参与合并的车辆一并转到新主编号，避免形成映射链。
        mappings = {
            int(row["source_vehicle_id"]): int(row["target_vehicle_id"])
            for row in db.execute(
                "SELECT source_vehicle_id, target_vehicle_id FROM manual_vehicle_merges"
            ).fetchall()
        }
        for source_id in list(mappings):
            if mappings[source_id] in vehicle_ids:
                mappings[source_id] = target_id
        for vehicle_id in vehicle_ids:
            mappings.pop(vehicle_id, None)
        for vehicle_id in vehicle_ids:
            if vehicle_id != target_id:
                mappings[vehicle_id] = target_id

        db.execute("DELETE FROM manual_vehicle_merges")
        created_at = datetime.now().isoformat(timespec="seconds")
        db.executemany(
            "INSERT INTO manual_vehicle_merges "
            "(source_vehicle_id, target_vehicle_id, created_at) VALUES (?, ?, ?)",
            [(source_id, mapped_target, created_at) for source_id, mapped_target in mappings.items()],
        )
        source_ids = [vehicle_id for vehicle_id in vehicle_ids if vehicle_id != target_id]
        source_placeholders = ",".join("?" for _ in source_ids)
        # 可靠重放：记录本次被并入主车的所有来源抓拍。capture_id 永不变化，
        # associate_orders 重聚类后按这些锚定对重放人工合并，不受车辆编号重排影响。
        source_capture_ids = [row[0] for row in db.execute(
            f"SELECT id FROM captures WHERE vehicle_id IN ({source_placeholders})",
            source_ids)]
        db.execute(
            f"UPDATE captures SET vehicle_id=?, plate_match_reason='人工合并' "
            f"WHERE vehicle_id IN ({source_placeholders})",
            [target_id, *source_ids],
        )
        # 同时应用已有映射中被转向本次主编号的来源车辆。
        redirected = [source_id for source_id, mapped_target in mappings.items()
                      if mapped_target == target_id and source_id not in source_ids]
        if redirected:
            redirected_placeholders = ",".join("?" for _ in redirected)
            db.execute(
                f"UPDATE captures SET vehicle_id=?, plate_match_reason='人工合并' "
                f"WHERE vehicle_id IN ({redirected_placeholders})",
                [target_id, *redirected],
            )
        # 写入可靠重放记录：每个来源抓拍 -> 主车代表抓拍（组内最小 id）。
        anchor_row = db.execute(
            "SELECT id FROM captures WHERE vehicle_id=? ORDER BY id LIMIT 1",
            (target_id,)).fetchone()
        if anchor_row and source_capture_ids:
            db.execute(
                """CREATE TABLE IF NOT EXISTS manual_vehicle_merge_pairs (
                     capture_id INTEGER PRIMARY KEY,
                     anchor_capture_id INTEGER NOT NULL,
                     park TEXT NOT NULL DEFAULT '',
                     created_at TEXT NOT NULL)"""
            )
            pair_park = vehicle_parks[0] if vehicle_parks else ""
            pair_now = datetime.now().isoformat(timespec="seconds")
            db.executemany(
                """INSERT OR REPLACE INTO manual_vehicle_merge_pairs
                   (capture_id, anchor_capture_id, park, created_at)
                   VALUES (?, ?, ?, ?)""",
                [(cid, anchor_row["id"], pair_park, pair_now)
                 for cid in source_capture_ids])
        db.commit()
        capture_count = sum(found.values())
    return jsonify({"ok": True, "target_vehicle_id": target_id,
                    "merged_vehicle_ids": vehicle_ids,
                    "capture_count": capture_count})


# ---------------------------------------------------------------------------
# 对外开放接口（/api/open/ 前缀，只读，供第三方系统对接）
#
# 与内部页面接口（无鉴权、裸数组）刻意分开：
#   - 统一响应包装：列表 {"code":0,"message":"ok","total":N,"page":p,"page_size":s,"items":[...]}，
#     单对象 {"code":0,"message":"ok","data":{...}}，错误 {"code":<HTTP状态码>,"message":"..."}；
#   - 日期过滤：date（单日）、date_from/date_to（区间，闭区间）三种参数可单独或组合使用；
#   - 分页：page 从 1 开始，page_size 默认 50、上限 500；
#   - 鉴权：请求头 X-API-Token（图片类接口也支持 ?token= 查询参数，便于嵌入 <img>），
#     token 由环境变量 OPEN_API_TOKEN 配置；未配置时整组接口直接停用（fail closed）。
# ---------------------------------------------------------------------------

OPEN_API_TOKEN = os.environ.get("OPEN_API_TOKEN", "")


def _open_ok(payload: dict):
    body = {"code": 0, "message": "ok"}
    body.update(payload)
    return jsonify(body)


def _open_error(status: int, message: str):
    return jsonify({"code": status, "message": message}), status


def open_api(func):
    """token 校验装饰器；只挂载在 /api/open/ 路由上，不影响内部页面接口。"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not OPEN_API_TOKEN:
            return _open_error(503, "开放接口未启用：服务端未配置 OPEN_API_TOKEN 环境变量")
        provided = request.headers.get("X-API-Token", "") or request.args.get("token", "")
        if not hmac.compare_digest(provided.encode("utf-8"), OPEN_API_TOKEN.encode("utf-8")):
            return _open_error(401, "token 无效或缺失（请求头 X-API-Token）")
        return func(*args, **kwargs)
    return wrapper


def open_date_filters(prefix: str = ""):
    """date / date_from / date_to 查询参数 → SQL 条件；日期非法时返回错误信息。"""
    column = f"{prefix}.date" if prefix else "date"
    where, parameters = [], []
    for name in ("date", "date_from", "date_to"):
        value = request.args.get(name, "").strip()
        if not value:
            continue
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            return [], [], f"日期参数 {name} 格式错误，应为 YYYY-MM-DD：{value}"
        if name == "date":
            where.append(f"{column}=?")
            parameters.append(value)
        elif name == "date_from":
            where.append(f"{column}>=?")
            parameters.append(value)
        else:
            where.append(f"{column}<=?")
            parameters.append(value)
    return where, parameters, None


def open_pagination(default_size: int = 50, max_size: int = 500):
    page = max(1, positive_int(request.args.get("page", "1"), 1))
    page_size = min(max_size, max(1, positive_int(request.args.get("page_size", str(default_size)), default_size)))
    return page, page_size, (page - 1) * page_size


def open_park_filter() -> tuple[str, list]:
    """可选 park 查询参数 → SQL 条件；空=全部园区。"""
    return park_where(request.args.get("park", "").strip())


@app.route("/api/open/ping")
@open_api
def open_ping():
    """连通性与 token 自检；对接方接入前先用它验证 token 是否有效。"""
    return _open_ok({"service": "car-trace",
                     "time": datetime.now().isoformat(timespec="seconds")})


@app.route("/api/open/parks")
@open_api
def open_parks():
    """园区清单：对接方做园区筛选下拉或分园区同步时使用。

    park 为空字符串表示未标注园区（老库数据）。车辆身份（vehicle_id）按
    园区隔离，跨园区查询同一车牌会得到相互独立的轨迹段。
    """
    with closing(get_db()) as db:
        rows = db.execute(
            """SELECT coalesce(park,'') AS park, COUNT(*) AS capture_count,
                      COUNT(DISTINCT CASE WHEN coalesce(order_id,'')!='' THEN
                           coalesce(park,'') || '|' || order_id END) AS order_count,
                      COUNT(DISTINCT vehicle_id) AS vehicle_count,
                      MIN(date) AS first_date, MAX(date) AS last_date
               FROM captures GROUP BY coalesce(park,'') ORDER BY park"""
        ).fetchall()
    return _open_ok({"items": [dict(row) for row in rows]})


@app.route("/api/open/dates")
@open_api
def open_dates():
    """有数据的日期清单及每日抓拍/订单数，供对接方做日期下拉或同步计划。"""
    park_clause, park_params = open_park_filter()
    with closing(get_db()) as db:
        rows = db.execute(
            f"""SELECT date, COUNT(*) AS capture_count,
                      COUNT(DISTINCT coalesce(park,'') || '|' || order_id) AS order_count
               FROM captures WHERE date!=''{(' AND ' + park_clause) if park_clause else ''}
               GROUP BY date ORDER BY date""",
            park_params,
        ).fetchall()
    return _open_ok({"items": [dict(row) for row in rows]})


@app.route("/api/open/orders")
@open_api
def open_orders():
    """订单列表（对外）：订单号在园区内聚合，跨日期只出一行；
    date/date_from/date_to/park 过滤。"""
    where, parameters, error = open_date_filters("c")
    if error:
        return _open_error(400, error)
    park_clause, park_params = open_park_filter()
    if park_clause:
        where.append("coalesce(c.park,'')=?")
        parameters.extend(park_params)
    order_like = request.args.get("order_id", "").strip()
    if order_like:
        where.append("c.order_id LIKE ?")
        parameters.append(f"%{order_like}%")
    stage = request.args.get("stage", "").strip()
    if stage:
        where.append("c.stage=?")
        parameters.append(stage)
    where_sql = " AND ".join(where) if where else "1=1"
    page, page_size, offset = open_pagination()
    with closing(get_db()) as db:
        total = db.execute(
            f"""SELECT COUNT(*) FROM (
                  SELECT DISTINCT coalesce(c.park,''), c.order_id
                  FROM captures c WHERE {where_sql})""",
            parameters,
        ).fetchone()[0]
        rows = db.execute(
            f"""SELECT coalesce(c.park,'') AS park, c.order_id, COUNT(*) AS capture_count,
                       COUNT(DISTINCT c.date) AS date_span,
                       GROUP_CONCAT(DISTINCT c.date) AS dates,
                       MIN(c.capture_time) AS first_seen, MAX(c.capture_time) AS last_seen,
                       SUM(CASE WHEN coalesce(c.vehicle_crop_path,'')!='' THEN 1 ELSE 0 END) AS cropped,
                       SUM(CASE WHEN c.capture_status='no_vehicle' THEN 1 ELSE 0 END) AS no_vehicle,
                       MAX(c.vehicle_color) AS vehicle_color, MAX(c.vehicle_type) AS vehicle_type,
                       MIN(CASE WHEN coalesce(c.vehicle_crop_path,'')!='' THEN c.id END) AS rep_id,
                       MAX(oa.vehicle_color_conf) AS attr_color_conf,
                       MAX(oa.vehicle_type_conf) AS attr_type_conf
                FROM captures c LEFT JOIN order_attributes oa
                  ON oa.order_id=c.order_id AND coalesce(oa.park,'')=coalesce(c.park,'')
                WHERE {where_sql}
                GROUP BY coalesce(c.park,''), c.order_id
                ORDER BY c.order_id
                LIMIT ? OFFSET ?""",
            [*parameters, page_size, offset],
        ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            attrs = db.execute(
                """SELECT vehicle_color, vehicle_color_conf, vehicle_type, vehicle_type_conf
                   FROM order_attributes WHERE order_id=? AND coalesce(park,'')=?""",
                (row["order_id"], row["park"]),
            ).fetchone()
            if attrs:
                item["vehicle_color"] = attrs["vehicle_color"] or item["vehicle_color"]
                item["vehicle_type"] = attrs["vehicle_type"] or item["vehicle_type"]
                item["attr_conf"] = {"color": attrs["vehicle_color_conf"],
                                     "type": attrs["vehicle_type_conf"]}
            item.update(order_summary(db, item["order_id"], item["park"]))
            items.append(item)
    return _open_ok({"total": total, "page": page, "page_size": page_size, "items": items})


@app.route("/api/open/order/<order_id>")
@open_api
def open_order_detail(order_id: str):
    """单个订单完整轨迹（对外）：跨日期聚合，按时间+阶段排序。

    订单号在园区内唯一：不带 park 参数且订单号只在单一园区时直接返回；
    撞号时返回 400 并列出园区，需带 park 参数重试。
    """
    park = request.args.get("park", "").strip()
    with closing(get_db()) as db:
        parks = [row[0] for row in db.execute(
            "SELECT DISTINCT coalesce(park,'') FROM captures WHERE order_id=?",
            (order_id,))]
        if not parks:
            return _open_error(404, "订单不存在")
        if not park and len(parks) > 1:
            return _open_error(400, f"订单号 {order_id} 在多个园区存在，请指定 park 参数；"
                                    f"可选值：{'、'.join(parks)}")
        if park and park not in parks:
            return _open_error(404, f"订单不存在（园区 {park}）")
        park_clause, park_params = park_where(park)
        rows = db.execute(
            f"""SELECT {CAPTURE_FIELDS} FROM captures
                WHERE order_id=?{( ' AND ' + park_clause) if park_clause else ''}
                {TRACK_ORDER}""",
            [order_id, *park_params],
        ).fetchall()
        attrs = db.execute(
            """SELECT * FROM order_attributes WHERE order_id=? AND coalesce(park,'')=?""",
            (order_id, park if park else parks[0]),
        ).fetchone()
        summary = order_summary(db, order_id, park if park else parks[0])
    return _open_ok({"data": {"order_id": order_id, "park": park if park else parks[0],
                              "order_time": summary["order_time"], "plate": summary["plate"],
                              "attributes": dict(attrs) if attrs else None,
                              "items": [dict(row) for row in rows]}})


@app.route("/api/open/vehicles")
@open_api
def open_vehicles():
    """车辆列表（对外）：vehicle_id 聚合（按园区隔离），代表图取质量分最高的一帧；
    date/date_from/date_to/park 过滤。"""
    where, parameters, error = open_date_filters()
    if error:
        return _open_error(400, error)
    park_clause, park_params = open_park_filter()
    if park_clause:
        where.append(park_clause)
        parameters.extend(park_params)
    where.append("vehicle_id IS NOT NULL")
    where_sql = " AND ".join(where)
    page, page_size, offset = open_pagination()
    with closing(get_db()) as db:
        total = db.execute(
            f"SELECT COUNT(DISTINCT vehicle_id) FROM captures WHERE {where_sql}",
            parameters,
        ).fetchone()[0]
        rows = db.execute(
            f"""WITH filtered AS (SELECT * FROM captures WHERE {where_sql}), ranked AS (
                  SELECT id, vehicle_id, ROW_NUMBER() OVER (
                    PARTITION BY vehicle_id
                    ORDER BY coalesce(quality_score,0) DESC, coalesce(plate_confidence,0) DESC, id
                  ) AS rank_no
                  FROM filtered
                )
                SELECT f.vehicle_id, r.id AS rep_id, COUNT(*) AS capture_count,
                       MIN(coalesce(f.park,'')) AS park,
                       GROUP_CONCAT(DISTINCT f.order_id) AS order_ids,
                       (SELECT c2.plate_final FROM captures c2
                        WHERE c2.vehicle_id=f.vehicle_id AND coalesce(c2.plate_final,'')!=''
                        GROUP BY c2.plate_final ORDER BY COUNT(*) DESC, c2.plate_final LIMIT 1) AS plate,
                       MAX(coalesce(f.plate_confidence,0)) AS plate_confidence,
                       MAX(f.vehicle_color) AS color, MAX(f.vehicle_type) AS vehicle_type,
                       COUNT(DISTINCT f.camera_key) AS camera_count,
                       MIN(f.capture_time) AS first_seen, MAX(f.capture_time) AS last_seen
                FROM filtered f
                JOIN ranked r ON r.vehicle_id=f.vehicle_id AND r.rank_no=1
                GROUP BY f.vehicle_id, r.id
                ORDER BY capture_count DESC, first_seen
                LIMIT ? OFFSET ?""",
            [*parameters, page_size, offset],
        ).fetchall()
    return _open_ok({"total": total, "page": page, "page_size": page_size,
                     "items": [dict(row) for row in rows]})


@app.route("/api/open/vehicle/<int:vehicle_id>")
@open_api
def open_vehicle_detail(vehicle_id: int):
    """单车完整轨迹（对外）：按时间+阶段排序。"""
    with closing(get_db()) as db:
        rows = db.execute(
            f"SELECT {CAPTURE_FIELDS} FROM captures WHERE vehicle_id=? {TRACK_ORDER}",
            (vehicle_id,),
        ).fetchall()
    if not rows:
        return _open_error(404, "车辆不存在")
    return _open_ok({"data": {"vehicle_id": vehicle_id,
                              "items": [dict(row) for row in rows]}})


@app.route("/api/open/captures")
@open_api
def open_captures():
    """抓拍明细（对外）：日期区间 + 订单号（精确）+ 车牌片段 + 车型 + 时段，任意组合。"""
    where, parameters, error = open_date_filters()
    if error:
        return _open_error(400, error)
    order_id = request.args.get("order_id", "").strip()
    if order_id:
        where.append("order_id=?")
        parameters.append(order_id)
    park_clause, park_params = open_park_filter()
    if park_clause:
        where.append(park_clause)
        parameters.extend(park_params)
    plate = normalize_plate(request.args.get("plate", ""))
    if plate:
        where.append("(replace(upper(plate_final),' ','') LIKE ? "
                     "OR replace(upper(osd_plate),' ','') LIKE ?)")
        parameters.extend([f"%{plate}%", f"%{plate}%"])
    vehicle_type = request.args.get("vehicle_type", "").strip()
    if vehicle_type:
        where.append("vehicle_type=?")
        parameters.append(vehicle_type)
    start_hour = request.args.get("start_hour", "").strip()
    if start_hour != "":
        where.append("hour>=?")
        parameters.append(positive_int(start_hour))
    end_hour = request.args.get("end_hour", "").strip()
    if end_hour != "":
        where.append("hour<=?")
        parameters.append(positive_int(end_hour, 23))
    where_sql = " AND ".join(where) if where else "1=1"
    page, page_size, offset = open_pagination()
    with closing(get_db()) as db:
        total = db.execute(
            f"SELECT COUNT(*) FROM captures WHERE {where_sql}", parameters
        ).fetchone()[0]
        rows = db.execute(
            f"""SELECT {CAPTURE_FIELDS} FROM captures WHERE {where_sql}
                ORDER BY date, capture_time, id LIMIT ? OFFSET ?""",
            [*parameters, page_size, offset],
        ).fetchall()
    return _open_ok({"total": total, "page": page, "page_size": page_size,
                     "items": [dict(row) for row in rows]})


@app.route("/api/open/crop/<int:capture_id>")
@open_api
def open_crop(capture_id: int):
    """车辆裁剪图（对外）。支持 ?token= 查询参数鉴权，便于第三方系统直接嵌入 <img> 标签。"""
    with closing(get_db()) as db:
        row = db.execute(
            "SELECT vehicle_crop_path FROM captures WHERE id=?", (capture_id,)
        ).fetchone()
    if not row or not row["vehicle_crop_path"]:
        return _open_error(404, "该抓拍尚未生成车辆裁剪")
    path = resolve_capture_path(row["vehicle_crop_path"])
    if not path.is_file():
        return _open_error(404, "车辆裁剪文件不存在")
    return send_file(path, conditional=True, max_age=3600)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "5000")),
            debug=os.environ.get("FLASK_DEBUG") == "1")
