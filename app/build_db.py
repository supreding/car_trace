# -*- coding: utf-8 -*-
"""园区抓拍照片扫描入库（多园区目录结构）。

目录结构（默认 --root 指向项目 data/ 目录）：
  data/<园区名>/<月份照片根（如 2026年8月份抓拍照片）>/<日期文件夹>/<订单号>/<阶段_uuid>.jpg
- 一级子目录名即园区名，写入 captures.park；园区下可放多个月份照片根
  （名字含"20xx年"），兼容园区下直接放日期文件夹的老布局；
- 兼容 --root 直接指向某个月份照片根（老布局）：此时园区名取 --park 参数，
  缺省为空字符串（显示为"未标注园区"）；
- 年份取自月份照片根名（如 2026年8月份抓拍照片 -> 2026）；
- 订单号在园区内是聚合唯一键：同一订单号可能出现在两个日期文件夹（订单跨
  午夜，按出磅日期归档），必须全盘扫描并按 (park, order_id) 聚合。不同园区
  可能使用同一套系统导致订单号撞号，订单聚合键必须带园区；
- 阶段前缀实测四种：入磅、出磅、抓拍、质检；订单内顺序 入磅->质检->出磅，
  抓拍没有固定位置，用 OCR 时间戳插入；
- 左上角时间戳用 RapidOCR 识别，失败或与日期文件夹矛盾时记录 ocr_status
  并标记时间不确定，不伪造时间；
- 图片下方摄像头写的车辆形象（车牌/颜色/车型）只作参考字段 osd_*，不能
  直接采用；
- 园区隔离（实施要求）：同一辆车的车牌在多个园区出现是合法的，但车辆身份
  （vehicle_id）按园区独立聚类，绝不跨园区合并，避免把一个园区的车辆追踪
  到另一个园区。

常用命令：
  python build_db.py --sync-only            # 只扫描目录入库，不跑 OCR
  python build_db.py --process-missing      # 对未 OCR 的记录补充时间戳/OSD
  python build_db.py --retry-failed         # 重跑左上角OCR失败的帧（时刻带抢救通道）
  python build_db.py --associate-only       # 只重新聚合订单内车辆
  python build_db.py --assign-loose         # 试运行：未关联货车归类到订单
  python build_db.py --assign-loose --apply # 真正移动文件并写库（可 --undo-loose-assign 撤销）
  python build_db.py --undo-loose-assign    # 撤销全部归类；--capture-id 123 只撤销单张
  python build_db.py --dedup-gate           # 试运行：共用大门跨园区重复照片判重
  python build_db.py --dedup-gate --apply   # 落败副本移入隔离园区（可 --undo-dedup 撤销）
  python build_db.py --undo-dedup           # 撤销全部去重；--capture-id 123 只撤销单张
  python build_db.py --root data --park 农易行临泉辉煌库   # 只处理指定园区
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = APP_DIR.parent / "data"
DEFAULT_DB = APP_DIR / "granary.db"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

YEAR_RE = re.compile(r"(20\d{2})年")
DATE_DIR_RE = re.compile(r"^(\d{1,2})月(\d{1,2})日$")
# 从老库 filepath 推断园区名：...data\<园区名>\<月份照片根>\...
PARK_PATH_RE = re.compile(r"[\\/]data[\\/]+([^\\/]+)[\\/]")
STAGE_RE = re.compile(r"^(入磅|出磅|抓拍|质检|人脸)[_]?(.+)$")
# 新版NVR命名自带拍摄时间（全角冒号）：入磅2026年9月20日08：48：11_2268117
FILENAME_TIME_RE = re.compile(
    r"(\d{4})年(\d{1,2})月(\d{1,2})日(\d{1,2})[：:](\d{2})[：:](\d{2})")
DATE_RE = re.compile(r"(20\d{2})[-年/.](\d{1,2})[-月/.](\d{1,2})")
TIME_RE = re.compile(r"(2[0-3]|[01]?\d)[:：]([0-5]\d)[:：]([0-5]\d)")

# 订单内的通行顺序：入磅 -> 质检 -> 出磅；抓拍没有固定位置（-1），按时间插入；
# 人脸抓拍（司机身份）不是车辆图片，排最前并直接标记非车辆。
STAGE_RANK = {"入磅": 0, "质检": 1, "抓拍": -1, "人脸": -2, "出磅": 2}

# ---- 共用大门跨园区去重 ----
# 多个园区共用同一大门时，上游 NVR 会把同一张抓拍照片写进每个园区的目录树
# （文件名一致，新版命名末尾带唯一 id，如 出磅2026年11月5日08：15：59_2346200）。
# sync_files 的行键 (park, order_id, filename) 含园区，重复照片在每个园区各成
# 一行。dedup_gate_captures 判重后把落败副本移入"重复抓拍隔离-<原园区>"园区
# （每个原园区一个隔离园区，保证重扫键唯一、可在文件共享里按园区浏览）。
# 所有归类/聚类/统计/开放接口环节都必须排除该前缀的园区。
QUARANTINE_PARK_PREFIX = "重复抓拍隔离"
QUARANTINE_PARK_LIKE = QUARANTINE_PARK_PREFIX + "%"
# 本系统移动文件时的防覆盖改名后缀（assign 阶段产生，如 出磅..._reid1.jpg）：
# 比对文件名身份前先剥离，两个园区的拷贝才能对上。
_REID_SUFFIX_RE = re.compile(r"_reid\d+$")
# 新版NVR命名：时间片段之后的末尾唯一id（出磅2026年11月5日08：15：59_2346200 -> 2346200）
_FILENAME_UID_RE = re.compile(r"_([0-9]+)$")
# 老命名 <阶段>_<截断uuid>（如 入磅_467f1888-f8a）：无拍摄时间，uuid 作候选键
# （截断uuid可能撞号，判重必须内容确认）
_GATE_TOKEN_RE = re.compile(r"^(?:入磅|出磅|抓拍|质检|人脸)[_]?(.+)$")
_UUID_TOKEN_RE = re.compile(r"[0-9a-fA-F-]{6,}")


def to_stored_path(path: Path) -> str:
    """入库路径统一存为相对项目根的正斜杠形式（如 data/<园区>/2026年8月份抓拍照片/...），
    项目整体复制到其他机器（或换盘符）后无需改库；项目外目录退回绝对路径。"""
    absolute = Path(path).resolve()
    try:
        return absolute.relative_to(APP_DIR.parent).as_posix()
    except ValueError:
        return str(absolute)


def resolve_stored_path(stored: str) -> Path:
    """还原库里存的路径：绝对路径原样返回；相对路径依次尝试 app 目录
    （裁剪图 reid_crops/...）与项目根（原图 data/...），兼容三类历史写法。"""
    path = Path(stored)
    if path.is_absolute():
        return path
    for base in (APP_DIR, APP_DIR.parent):
        candidate = base / path
        if candidate.is_file():
            return candidate
    return APP_DIR / path


def project_file(path: str | Path) -> Path:
    """把库存相对路径锚定到项目根的绝对路径（文件系统操作专用）。

    归类移动文件时若用相对路径，mkdir/shutil 会相对当前工作目录解析，
    在 app/ 下误建 data/ 目录树——所有落盘操作必须走本函数转绝对路径。
    """
    path = Path(path)
    return path if path.is_absolute() else APP_DIR.parent / path

# 摄像头叠字与远距离图像识别常把省份汉字漏认/认错，或少认 1 位字符；
# 完整标准车牌走 PLATE_RE，缺省份或少 1 位（5~7 位主体且含 >=2 个数字）在
# allow_partial=True 时保留，供后续无视首字+形近+缺位匹配使用（实施文档 13.1）。
_PROVINCE_CHARS = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
PLATE_RE = re.compile(
    rf"[{_PROVINCE_CHARS}][A-Z][A-Z0-9挂学警港澳使领]{{5,6}}"
)
_PLATE_PARTIAL_PROV_RE = re.compile(
    rf"[{_PROVINCE_CHARS}][A-Z0-9][A-Z0-9挂学警港澳使领]{{4,6}}"
)
_PLATE_PARTIAL_NOPROV_RE = re.compile(
    r"(?<![A-Z0-9])[A-Z][A-Z0-9挂学警港澳使领]{4,6}(?![A-Z0-9])"
)
# 常见非车牌设备编号/车型型号前缀（避免 allow_partial 误把 DMC01 / CA4250 当车牌）
_NON_PLATE_NOISE_RE = re.compile(r"^(?:DMC|DMR|CAM|J6P|CA\d{4}|FAW)")


def normalize_plate(value: str | None, allow_partial: bool = False) -> str:
    if not value:
        return ""
    value = value.upper().replace(" ", "").replace("·", "").replace("-", "")
    value = re.sub(r"[^\u4e00-\u9fffA-Z0-9挂学警港澳使领]", "", value)
    match = PLATE_RE.search(value)
    if match:
        return match.group(0)
    if not allow_partial:
        return ""
    for regex in (_PLATE_PARTIAL_PROV_RE, _PLATE_PARTIAL_NOPROV_RE):
        for m in regex.finditer(value):
            cand = m.group(0)
            body = cand[1:] if ("\u4e00" <= cand[0] <= "\u9fff") else cand
            if len(body) < 5 or len(body) > 7:
                continue
            if sum(ch.isdigit() for ch in body) < 2:
                continue
            if _NON_PLATE_NOISE_RE.match(body):
                continue
            return cand
    return ""


# 车辆颜色 / 车型标准代码（与 PaddleClas PULC vehicle_attribute 模型输出严格对齐）
# captures.vehicle_color / vehicle_type 一律存英文标准码，避免与模型输出混写中文导致
# 聚类第 7 步车型拆分误踢帧、未关联货车归类（--loose-types truck）漏筛。
COLOR_CODES: dict[str, str] = {
    "yellow": "黄色", "orange": "橙色", "green": "绿色", "gray": "灰色",
    "red": "红色", "blue": "蓝色", "white": "白色", "golden": "金色",
    "brown": "棕色", "black": "黑色",
}
TYPE_CODES: dict[str, str] = {
    "sedan": "轿车", "suv": "SUV", "van": "厢式车", "hatchback": "两厢车",
    "mpv": "MPV", "pickup": "皮卡", "bus": "客车", "truck": "货车",
    "estate": "旅行车",
}
_TYPE_ALIASES: dict[str, str] = {
    "面包车": "van", "微面": "van", "小货车": "truck", "大货车": "truck",
    "重卡": "truck", "罐车": "truck", "挂车": "truck", "半挂车": "truck",
    "工程机械": "truck", "工程车": "truck", "小轿车": "sedan", "大巴": "bus",
}
_COLOR_ALIASES: dict[str, str] = {
    "灰": "gray", "白": "white", "黑": "black", "红": "red", "蓝": "blue",
    "黄": "yellow", "绿": "green", "棕": "brown", "橙": "orange", "金": "golden",
    "银": "gray", "银色": "gray", "银灰": "gray",
}


def normalize_color(value: str | None) -> str:
    """将颜色（英文码或中文名）归一为模型标准英文码，无法识别时返回空串。"""
    v = (value or "").strip()
    if not v:
        return ""
    low = v.lower()
    if low in COLOR_CODES:
        return low
    for code, cn in COLOR_CODES.items():
        if v == cn:
            return code
    return _COLOR_ALIASES.get(v, "")


def normalize_type(value: str | None) -> str:
    """将车型（英文码或中文名）归一为模型标准英文码，无法识别时返回空串。"""
    v = (value or "").strip()
    if not v:
        return ""
    low = v.lower()
    if low in TYPE_CODES:
        return low
    for code, cn in TYPE_CODES.items():
        if v == cn:
            return code
    return _TYPE_ALIASES.get(v, "")


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        -- 每车一条记录：一张抓拍图里的每辆检测到的车辆各占一行。
        -- instance_index=0 是主体车辆（与 captures 行同步写入），
        -- 1..n 是同帧其他车辆（无关车辆由此分开跟踪，不再错挂）。
        CREATE TABLE IF NOT EXISTS vehicle_instances (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          capture_id INTEGER NOT NULL,
          instance_index INTEGER NOT NULL,
          crop_path TEXT, vehicle_box TEXT, vehicle_det_score REAL,
          vehicle_color TEXT, vehicle_color_conf REAL,
          vehicle_type TEXT, vehicle_type_conf REAL,
          attr_model_version TEXT,
          lpr_plate TEXT, lpr_conf REAL,
          lpr_attempted_at TEXT,
          reid_embedding BLOB, reid_source TEXT, reid_model_version TEXT,
          vehicle_id INTEGER,
          UNIQUE(capture_id, instance_index)
        );
        CREATE INDEX IF NOT EXISTS idx_vi_capture ON vehicle_instances(capture_id);
        CREATE INDEX IF NOT EXISTS idx_vi_vehicle ON vehicle_instances(vehicle_id);
        """
    )
    # 老库补列：lpr_attempted_at 记录车牌识别"已尝试"时刻——模型读不出车牌的
    # 裁剪图也要落标记，否则车牌阶段每轮重复处理上万张无牌图（GPU 空转循环）。
    _vi_cols = {row[1] for row in conn.execute("PRAGMA table_info(vehicle_instances)")}
    if "lpr_attempted_at" not in _vi_cols:
        conn.execute("ALTER TABLE vehicle_instances ADD COLUMN lpr_attempted_at TEXT")
    conn.executescript(
        """
        -- 人工移动决定（聚类重跑后重新应用，人工意志优先于自动聚类）：
        -- target_plate=NULL 表示拆出为独立车辆；否则为移动到的目标车牌
        -- （去省份后的 6 位，与聚类锚点一致）。
        CREATE TABLE IF NOT EXISTS manual_vehicle_moves (
          capture_id INTEGER PRIMARY KEY,
          target_plate TEXT,
          scope TEXT NOT NULL DEFAULT 'capture',
          moved_at TEXT NOT NULL
        );
        """
    )
    conn.executescript(
        """
        -- 人工合并的可靠重放记录：capture 锚定到主车代表抓拍。
        -- vehicle_id 每次重聚类都会整体重排，故人工合并不按编号存储；
        -- associate_orders 末尾按本表重放（把 capture_id 所在组并回锚点所在组）。
        CREATE TABLE IF NOT EXISTS manual_vehicle_merge_pairs (
          capture_id INTEGER PRIMARY KEY,
          anchor_capture_id INTEGER NOT NULL,
          park TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL
        );
        """
    )
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS captures (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          date TEXT, park TEXT, order_id TEXT, stage TEXT, filename TEXT, filepath TEXT,
          resolution TEXT, camera_key TEXT, osd_camera TEXT,
          capture_time TEXT, hour INTEGER, ocr_status TEXT,
          osd_plate TEXT, osd_color TEXT, osd_type TEXT,
          lpr_plate TEXT, lpr_conf REAL,
          plate_final TEXT, plate_observed TEXT, plate_confidence REAL,
          recognition_source TEXT, plate_match_reason TEXT,
          vehicle_color TEXT, vehicle_color_conf REAL,
          vehicle_type TEXT, vehicle_type_conf REAL, attr_model_version TEXT,
          vehicle_crop_path TEXT, vehicle_box TEXT, crop_source TEXT,
          detection_count INTEGER,
          reid_embedding BLOB, reid_source TEXT, reid_model_version TEXT,
          quality_score REAL, processed_at TEXT,
          capture_status TEXT DEFAULT 'vehicle', status_reason TEXT,
          vehicle_id INTEGER, stage_rank INTEGER
        );
        CREATE TABLE IF NOT EXISTS manual_vehicle_merges (
          source_vehicle_id INTEGER PRIMARY KEY,
          target_vehicle_id INTEGER NOT NULL,
          created_at TEXT NOT NULL
        );
        -- 人工框选修正：本表中的抓拍不再被检测/复核阶段自动改框（人工意志
        -- 优先）。box 是人工确认的裁剪矩形 [l,t,r,b]（原图像素坐标）；
        -- source 区分手动画框（manual_box）与同帧实例切换（manual_promote）。
        CREATE TABLE IF NOT EXISTS manual_box_edits (
          capture_id INTEGER PRIMARY KEY,
          box TEXT NOT NULL,
          source TEXT NOT NULL DEFAULT 'manual_box',
          edited_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_order ON captures(order_id);
        CREATE INDEX IF NOT EXISTS idx_date ON captures(date);
        CREATE INDEX IF NOT EXISTS idx_stage ON captures(stage);
        CREATE INDEX IF NOT EXISTS idx_camera ON captures(camera_key);
        CREATE INDEX IF NOT EXISTS idx_plate ON captures(plate_final);
        CREATE INDEX IF NOT EXISTS idx_vid ON captures(vehicle_id);
        CREATE INDEX IF NOT EXISTS idx_time ON captures(capture_time);
        CREATE INDEX IF NOT EXISTS idx_crop ON captures(vehicle_crop_path);
        """
    )
    existing = {row[1] for row in conn.execute("PRAGMA table_info(captures)")}
    additions = {
        "park": "TEXT",
        "order_id": "TEXT", "stage": "TEXT", "resolution": "TEXT",
        "camera_key": "TEXT", "osd_camera": "TEXT",
        "capture_time": "TEXT", "hour": "INTEGER", "ocr_status": "TEXT",
        "vehicle_color_conf": "REAL", "vehicle_type_conf": "REAL",
        "attr_model_version": "TEXT", "detection_count": "INTEGER",
        "reid_embedding": "BLOB", "reid_source": "TEXT",
        "reid_model_version": "TEXT", "vehicle_crop_path": "TEXT",
        "vehicle_box": "TEXT", "crop_source": "TEXT",
        "vehicle_det_score": "REAL",
        "stage_rank": "INTEGER",
        "lpr_attempted_at": "TEXT",
    }
    for column, declaration in additions.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE captures ADD COLUMN {column} {declaration}")
    # park 列可能刚由上面的迁移补上，索引必须在加列之后创建。
    conn.execute("CREATE INDEX IF NOT EXISTS idx_park ON captures(park)")
    conn.executescript(
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
    ensure_label_schema(conn)
    _migrate_park_backfill(conn)
    ensure_order_attributes_schema(conn)
    conn.commit()


def ensure_label_schema(conn: sqlite3.Connection) -> None:
    """统一标注面板的表与列（幂等）：capture_annotations 人工修正列 +
    manual_order_anchors 订单时点锚点 + manual_device_registry 设备机位字典。
    server.ensure_label_tables 复用本函数，避免两处 DDL 漂移。
    """
    for column in ("corrected_color TEXT NOT NULL DEFAULT ''",
                   "corrected_type TEXT NOT NULL DEFAULT ''",
                   "corrected_status TEXT NOT NULL DEFAULT ''",
                   "corrected_stage TEXT NOT NULL DEFAULT ''",
                   "corrected_direction TEXT NOT NULL DEFAULT ''",
                   "corrected_time TEXT NOT NULL DEFAULT ''",
                   "no_vehicle_manual INTEGER NOT NULL DEFAULT 0"):
        try:
            conn.execute(f"ALTER TABLE capture_annotations ADD COLUMN {column}")
        except sqlite3.OperationalError:
            pass  # 列已存在
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS manual_order_anchors (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          park TEXT DEFAULT '',
          order_id TEXT NOT NULL,
          kind TEXT NOT NULL,
          anchor_time TEXT NOT NULL,
          annotator TEXT NOT NULL DEFAULT '',
          note TEXT NOT NULL DEFAULT '',
          created_at TEXT DEFAULT (datetime('localtime'))
        );
        CREATE INDEX IF NOT EXISTS idx_order_anchors_order
          ON manual_order_anchors(order_id);
        CREATE TABLE IF NOT EXISTS manual_device_registry (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          park TEXT DEFAULT '',
          device TEXT NOT NULL,
          location TEXT NOT NULL,
          direction TEXT DEFAULT '',
          annotator TEXT NOT NULL DEFAULT '',
          note TEXT NOT NULL DEFAULT '',
          created_at TEXT DEFAULT (datetime('localtime')),
          UNIQUE(park, device)
        );
        """
    )


def replay_manual(conn: sqlite3.Connection) -> dict:
    """全库重建后恢复人工成果：把人工表里的写回字段重新落到 captures。

    人工表是唯一事实源，captures 对应列视为缓存；本命令幂等可重复执行。
    """
    ensure_label_schema(conn)
    rows = conn.execute(
        """SELECT capture_id, corrected_stage, corrected_direction,
                  corrected_color, corrected_type, corrected_plate,
                  corrected_time, no_vehicle_manual
           FROM capture_annotations"""
    ).fetchall()
    applied = 0
    for row in rows:
        sets, params = [], []
        if row["corrected_stage"]:
            sets += ["stage=?", "stage_rank=?"]
            params += [row["corrected_stage"], STAGE_RANK.get(row["corrected_stage"], -1)]
        if row["corrected_direction"]:
            sets.append("osd_camera=?")
            params.append("大门抓拍出" if row["corrected_direction"] == "out" else "大门抓拍进")
        if row["corrected_color"]:
            norm_c = normalize_color(row["corrected_color"]) or row["corrected_color"]
            sets += ["vehicle_color=?", "vehicle_color_conf=1.0"]
            params.append(norm_c)
        if row["corrected_type"]:
            norm_t = normalize_type(row["corrected_type"]) or row["corrected_type"]
            sets += ["vehicle_type=?", "vehicle_type_conf=1.0"]
            params.append(norm_t)
        if row["corrected_plate"]:
            sets += ["plate_final=?", "plate_confidence=1.0", "recognition_source='人工标注'"]
            params.append(row["corrected_plate"])
        if row["corrected_time"]:
            day = conn.execute("SELECT date FROM captures WHERE id=?",
                               (row["capture_id"],)).fetchone()
            if day and day["date"]:
                sets += ["capture_time=?", "hour=?", "ocr_status='manual'"]
                params += [f"{day['date']} {row['corrected_time']}",
                           int(row["corrected_time"][:2])]
        if row["no_vehicle_manual"]:
            sets.append("capture_status=?")
            params.append("no_vehicle")
        if sets:
            conn.execute(f"UPDATE captures SET {', '.join(sets)} WHERE id=?",
                         [*params, row["capture_id"]])
            applied += 1
    conn.commit()
    return {"annotated_rows": len(rows), "applied": applied}


def ensure_order_attributes_schema(conn: sqlite3.Connection) -> None:
    """order_attributes 表：主键 (park, order_id)。

    不同园区可能使用同一套系统导致订单号撞号，订单属性必须按园区分开。
    老表（主键 order_id，无 park 列）自动重建，旧记录园区名从 captures 回填。
    server 每次建连都会执行 ensure_schema，因此老库在查询前即可完成迁移。
    """
    conn.execute(
        """CREATE TABLE IF NOT EXISTS order_attributes (
             park TEXT NOT NULL DEFAULT '',
             order_id TEXT NOT NULL,
             vehicle_color TEXT NOT NULL DEFAULT '',
             vehicle_color_conf REAL NOT NULL DEFAULT 0,
             vehicle_type TEXT NOT NULL DEFAULT '',
             vehicle_type_conf REAL NOT NULL DEFAULT 0,
             sample_count INTEGER NOT NULL DEFAULT 0,
             updated_at TEXT NOT NULL,
             PRIMARY KEY (park, order_id)
           )"""
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(order_attributes)")}
    if "park" in cols:
        return
    conn.execute("ALTER TABLE order_attributes RENAME TO order_attributes_legacy")
    conn.execute(
        """CREATE TABLE order_attributes (
             park TEXT NOT NULL DEFAULT '',
             order_id TEXT NOT NULL,
             vehicle_color TEXT NOT NULL DEFAULT '',
             vehicle_color_conf REAL NOT NULL DEFAULT 0,
             vehicle_type TEXT NOT NULL DEFAULT '',
             vehicle_type_conf REAL NOT NULL DEFAULT 0,
             sample_count INTEGER NOT NULL DEFAULT 0,
             updated_at TEXT NOT NULL,
             PRIMARY KEY (park, order_id)
           )"""
    )
    conn.execute(
        """INSERT INTO order_attributes
             (park, order_id, vehicle_color, vehicle_color_conf,
              vehicle_type, vehicle_type_conf, sample_count, updated_at)
           SELECT coalesce((SELECT c.park FROM captures c
                            WHERE c.order_id=l.order_id AND coalesce(c.park,'')!=''
                            LIMIT 1), ''),
                  l.order_id, l.vehicle_color, l.vehicle_color_conf,
                  l.vehicle_type, l.vehicle_type_conf, l.sample_count, l.updated_at
           FROM order_attributes_legacy l"""
    )
    conn.execute("DROP TABLE order_attributes_legacy")


def _migrate_park_backfill(conn: sqlite3.Connection) -> None:
    """老库一次性迁移：captures.park 建列后，从 filepath 里的 data/<园区>/ 段
    回填园区名。无法推断的记录（如老 granary 库的项目根目录）保持为空，
    展示为"未标注园区"。只执行一次（schema_migrations 记录），避免每次
    打开连接都全表扫描。"""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(name TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    done = conn.execute(
        "SELECT 1 FROM schema_migrations WHERE name='park_backfill'"
    ).fetchone()
    if done:
        return
    rows = conn.execute(
        "SELECT id, filepath FROM captures WHERE park IS NULL OR park=''"
    ).fetchall()
    updates = []
    for capture_id, filepath in rows:
        match = PARK_PATH_RE.search(str(filepath or ""))
        if match:
            updates.append((match.group(1), capture_id))
    if updates:
        conn.executemany("UPDATE captures SET park=? WHERE id=?", updates)
        print(f"园区字段回填：{len(updates)} 条记录从文件路径推断园区", flush=True)
    conn.execute(
        "INSERT OR REPLACE INTO schema_migrations (name, applied_at) VALUES (?, ?)",
        ("park_backfill", datetime.now().isoformat(timespec="seconds")),
    )


def parse_folder_date(root: Path, date_dir: str) -> str:
    year_match = YEAR_RE.search(root.name)
    year = int(year_match.group(1)) if year_match else datetime.now().year
    match = DATE_DIR_RE.match(date_dir)
    if not match:
        return ""
    return f"{year:04d}-{int(match[1]):02d}-{int(match[2]):02d}"


def iter_month_roots(park_dir: Path) -> list[Path]:
    """园区目录下的月份照片根：子目录名含年份（如 2026年8月份抓拍照片）。

    园区下直接放日期文件夹（老布局）时，园区目录本身就是照片根。
    """
    roots = [d for d in park_dir.iterdir()
             if d.is_dir() and YEAR_RE.search(d.name)]
    return sorted(roots, key=lambda p: p.name) if roots else [park_dir]


def iter_park_dirs(root: Path, only_park: str = "") -> Iterable[tuple[str, Path]]:
    """yield (园区名, 园区目录)。

    - root 是 data 根目录：一级子目录名即园区名；
    - root 本身就是照片根（子目录里有日期文件夹，老布局）：视为单一隐式
      园区，园区名取 only_park（--park 参数），缺省为空；
    - only_park 非空时只处理同名园区。
    """
    has_date_dirs = any(d.is_dir() and DATE_DIR_RE.match(d.name)
                        for d in root.iterdir())
    if has_date_dirs:
        yield only_park, root
        return
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if child.is_dir() and (not only_park or child.name == only_park):
            yield child.name, child


def iter_images(park_dir: Path, park: str = ""
                ) -> Iterable[tuple[Path, str, str, str, str, str]]:
    """按园区->月份根->日期->订单->文件返回 (路径, 园区, 日期, 订单号, 阶段, 文件名)。"""
    for month_root in iter_month_roots(park_dir):
        for date_dir in sorted(month_root.iterdir(), key=lambda p: p.name):
            if not date_dir.is_dir() or not DATE_DIR_RE.match(date_dir.name):
                continue
            date = parse_folder_date(month_root, date_dir.name)
            if not date:
                continue
            for order_dir in sorted(date_dir.iterdir(), key=lambda p: p.name):
                if not order_dir.is_dir():
                    continue
                order_id = order_dir.name.strip()
                for path in sorted(order_dir.iterdir(), key=lambda p: p.name):
                    if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
                        continue
                    stage_match = STAGE_RE.match(path.stem)
                    stage = stage_match.group(1) if stage_match else ""
                    yield path, park, date, order_id, stage, path.name


def sync_files(conn: sqlite3.Connection, root: Path,
               only_park: str = "") -> dict[str, int]:
    """全盘扫描照片目录；订单号在园区内是聚合唯一键（不同园区可能撞号），
    重复文件更新路径。"""
    rows_by_key: dict[tuple[str, str, str], list[int]] = {}
    for row in conn.execute("SELECT id, park, order_id, filename FROM captures"):
        rows_by_key.setdefault((row[1] or "", row[2], row[3]), []).append(row[0])

    added = updated = duplicates = 0
    for park_name, park_dir in iter_park_dirs(root, only_park):
        for path, park, date, order_id, stage, filename in iter_images(park_dir, park_name):
            with Image.open(path) as opened:
                width, height = opened.size
            resolution = f"{width}x{height}"
            camera_key = f"{stage}@{resolution}" if stage else f"未知@{resolution}"
            key = (park, order_id, filename)
            ids = rows_by_key.get(key, [])
            values = (to_stored_path(path), park, date, order_id, stage, resolution,
                      camera_key, STAGE_RANK.get(stage, -1))
            # 人脸抓拍是司机身份快照，不是车辆图片：入库但不参与检测/特征。
            if stage == "人脸":
                capture_status, status_reason = "no_vehicle", "人脸抓拍，非车辆图片"
            else:
                capture_status, status_reason = "vehicle", None
            if ids:
                conn.execute(
                    """UPDATE captures SET filepath=?, park=?, date=?, stage=?, resolution=?,
                              camera_key=?, stage_rank=? WHERE id=?""",
                    (to_stored_path(path), park, date, stage, resolution, camera_key,
                     STAGE_RANK.get(stage, -1), ids[0]),
                )
                updated += 1
                duplicates += max(0, len(ids) - 1)
            else:
                cursor = conn.execute(
                    """INSERT INTO captures
                       (filepath, park, date, order_id, stage, resolution,
                        camera_key, stage_rank, filename,
                        capture_status, status_reason)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (*values, filename, capture_status, status_reason),
                )
                rows_by_key[key] = [cursor.lastrowid]
                added += 1
    conn.commit()
    return {"total": added + updated, "added": added, "updated": updated,
            "duplicates": duplicates}


_ocr = None
_ocr_large = None


def get_ocr(large: bool = False):
    """RapidOCR 实例；large=True 使用更大检测边长（1280）且关闭 180° 方向分类，
    用于时间戳重试通道与抓拍底部纯黑 OSD 条带（NVR 叠字永远正立，开 cls 反会
    把超宽横条误翻 180° 导致整行乱码）。

    intra_op_num_threads 必须压到 2：默认 -1 会让进程池里每个 worker 的
    每个会话都开满全部核心，互相踩踏导致吞吐不升反降。
    """
    global _ocr, _ocr_large
    if large:
        if _ocr_large is None:
            from rapidocr_onnxruntime import RapidOCR

            _ocr_large = RapidOCR(det_limit_side_len=1280, det_limit_type="max",
                                  use_cls=False, intra_op_num_threads=2)
        return _ocr_large
    if _ocr is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "RapidOCR 未安装，无法识别时间戳。请先安装 rapidocr_onnxruntime"
            ) from exc
        # det_limit_type=max：检测网络不做全分辨率推理，速度提升约7倍。
        _ocr = RapidOCR(det_limit_side_len=736, det_limit_type="max",
                        intra_op_num_threads=2)
    return _ocr


def _find_black_footer_y(bgr: np.ndarray) -> int:
    """定位抓拍图底部纯黑 OSD 信息条的上边界像素行；不存在返回 -1。

    1920x1264 等抓拍图由 1920x1080 画面 + 底部 ~184px 纯黑叠字条拼接而成。
    旧逻辑从 y=0.72 裁剪会把上方 170px 明亮车头/路面一并送入检测器，导致
    DBNet 在 35% 以上的图片里直接漏检中间一行（车牌号码+车身颜色）。
    精确切出纯黑条带可彻底消除画面纹理干扰。
    """
    import cv2

    height = bgr.shape[0]
    y_start = int(height * 0.70)
    gray = cv2.cvtColor(bgr[y_start:, :], cv2.COLOR_BGR2GRAY)
    black_frac = (gray < 20).mean(axis=1)
    y_rel = gray.shape[0]
    for row_idx in range(gray.shape[0] - 1, -1, -1):
        if black_frac[row_idx] >= 0.45:
            y_rel = row_idx
        else:
            break
    y_abs = y_start + y_rel
    return y_abs if (height - y_abs) >= 60 else -1


def _ocr_osd_footer(bgr: np.ndarray) -> tuple[str, list[float]]:
    """精准识别抓拍图底部纯黑 OSD 条带（车牌/时间/颜色/车型/设备名）。

    1) 仅截取纯黑条带并四周补 20px 黑边，避免文字贴边被裁切；
    2) 关闭 cls（use_cls=False），避免超宽条带被误翻 180°；
    3) 若整条一次识别漏掉第 2 行（"车牌"未命中），对 x=18%..100% 区域补扫
       一次——车牌号码与车身颜色固定位于该区段，横纵比缩窄后召回率接近 100%。
    """
    import cv2

    y0 = _find_black_footer_y(bgr)
    if y0 < 0:
        return "", []
    footer = bgr[y0:, :]
    fw = footer.shape[1]
    scale = 0.5 if fw >= 1400 else 1.0
    ocr = get_ocr(large=True)

    def _run_strip(strip: np.ndarray, strip_scale: float) -> tuple[str, list[float]]:
        if strip_scale != 1.0:
            strip = cv2.resize(strip, None, fx=strip_scale, fy=strip_scale,
                               interpolation=cv2.INTER_AREA)
        padded = cv2.copyMakeBorder(
            strip, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        res, _ = ocr(padded)
        if not res:
            return "", []
        row_h = max(15, int(35 * strip_scale))
        res.sort(key=lambda it: (int(min(p[1] for p in it[0]) // row_h),
                                 min(p[0] for p in it[0])))
        return (
            re.sub(r"\s+", "", "".join(str(t) for _, t, _ in res)),
            [float(c) for _, _, c in res],
        )

    joined, confs = _run_strip(footer, scale)
    has_plate = any(
        normalize_plate(m[1])
        for pat in (r"车牌号码[:：]?([^\s:：]{6,12})", r"车牌号[:：]?([^\s:：]{6,12})")
        if (m := re.search(pat, joined))
    )
    if not has_plate and "无车牌" not in joined and fw >= 600:
        sub_joined, sub_confs = _run_strip(footer[:, int(fw * 0.18):], 1.0)
        if sub_joined:
            joined = sub_joined + joined
            confs = sub_confs + confs
    return joined, confs


def run_ocr_text(image: np.ndarray, large: bool = False) -> list[tuple[str, float]]:
    result, _ = get_ocr(large)(image)
    if not result:
        return []
    return [(str(text), float(conf)) for _, text, conf in result]


def _cc_text_mask(gray: np.ndarray) -> np.ndarray:
    """OSD 像素字字形掩膜：白色字芯（>=226）连通域过滤。

    叠字是固定尺寸的像素字：连通域高度落在字高范围内、宽度有限，据此
    滤掉大块背景（白色车身/天空/墙面）与细碎噪声；面积阈值放宽到 5
    是为了保留冒号的上下点（两点互不相连，各只有几个像素）。
    """
    import cv2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (gray >= 226).astype(np.uint8), connectivity=8)
    keep = np.zeros_like(gray, dtype=np.uint8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 8 <= h <= 90 and w <= 95 and area >= 5:
            keep[labels == i] = 1
    return keep


def _parse_time_candidates(joined: str) -> set[str]:
    """从 OCR 文本枚举合法时刻 HH:MM:SS。

    冒号常被读成 . - 空格或干脆丢字符：分隔符放宽为任意 0-2 个非数字；
    时分秒越界（如把 '07929' 读出 79 分）直接判非法，不生成候选。
    """
    candidates = set()
    for match in re.finditer(r"(\d{1,2})\D{0,2}(\d{2})\D{0,2}(\d{2})", joined):
        hh, mm, ss = int(match[1]), int(match[2]), int(match[3])
        if hh <= 23 and mm <= 59 and ss <= 59:
            candidates.add(f"{hh:02d}:{mm:02d}:{ss:02d}")
    return candidates


def _rescue_timestamp(rgb: np.ndarray, date_box: tuple) -> str:
    """时刻带抢救：以日期框为锚裁出右侧时刻带，多路预处理重试。

    入磅/出磅镜头的时刻常压在白车/栏杆/树枝等复杂背景上，整条时间带
    直接 OCR 检测不到文本框（日期部分在深色篷布上反而都能读出）。利用
    叠字位置相对日期框固定、字芯纯白的特点：裁出右侧等高时刻带后，
    字形掩膜二值图与原图多倍放大并行重试；不同预处理通道读出同一个
    合法时刻才采信——单路结果可能把背景纹理误读成数字，宁缺毋滥。
    """
    import cv2

    _, x0, y0, x1, y1 = date_box
    height, width = rgb.shape[:2]
    box_h = max(8.0, y1 - y0)
    tx0 = max(0, int(x1 + box_h * 0.15))
    tx1 = min(width, int(x1 + box_h * 8.6))
    ty0 = max(0, int(y0 - box_h * 0.35))
    ty1 = min(height, int(y1 + box_h * 0.35))
    if tx1 - tx0 < box_h * 2:
        return ""
    band = rgb[ty0:ty1, tx0:tx1]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    mask_img = np.where(_cc_text_mask(gray) > 0, 0, 255).astype(np.uint8)
    variants = [
        cv2.resize(band, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC),
        cv2.resize(band, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC),
        cv2.resize(mask_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC),
    ]
    band_w = band.shape[1]
    if band_w >= 120:
        # 宽带切左右两半分别识别：检测网络有最大边长限制，整条宽带会被
        # 压缩导致小字漏检，切半后有效分辨率翻倍（两端各留 20px 重叠防切字）。
        variants.append(cv2.resize(band[:, : band_w // 2 + 20], None, fx=2.5, fy=2.5,
                                   interpolation=cv2.INTER_CUBIC))
        variants.append(cv2.resize(band[:, band_w // 2 - 20:], None, fx=2.5, fy=2.5,
                                   interpolation=cv2.INTER_CUBIC))
    votes: dict[str, int] = {}
    for image in variants:
        texts = run_ocr_text(image, large=True)
        for cand in _parse_time_candidates("".join(t for t, _ in texts)):
            votes[cand] = votes.get(cand, 0) + 1
    if not votes:
        return ""
    best = max(votes, key=lambda k: votes[k])
    return best if votes[best] >= 2 else ""


def _parse_osd_camera_and_type(bottom_joined: str) -> tuple[str, str]:
    """解析抓拍图底部纯黑 OSD 条带的机位与车辆类型。

    现场物理规律（农易行临泉辉煌农事服务中心等标准 NVR 布局）：
    - 设备编号按拼音首字母命名：
        DMR* = 大门入相机（Da Men Ru）-> "大门抓拍进"
        DMC* = 大门出相机（Da Men Chu）-> "大门抓拍出"
        CK*  = 仓库抓拍（Cang Ku），位于卸货区 -> "仓库抓拍"（实锤「卸货」事件）
    - 注意：NVR 品牌算法对单张车的「(背向)」识别存在较多误报（正对 DMC 驶出的车头也常误标
      「(背向)」），因此这里绝不单凭「背向」字样强行翻转方向；真正的「过门背向连拍」由
      fix_opposite_gate_rear_directions() 基于“同车牌 1~60 秒内先后触发对向大门相机”这一
      物理时序规律精准识别并改标为「大门抓拍进(背向)」/「大门抓拍出(背向)」。
    - OCR 常把中文"抓拍"误读为"抑拍/批拍"、数字 0 误读为 O/Q（如 DMRO1/DMCO1/DMRQ1），
      优先按设备编号判定，比单纯匹配中文词稳得多。
    """
    text = bottom_joined or ""

    # 1) 车辆类型：先匹配明确中文车型，再结合黄牌与重卡品牌词兜底
    vehicle_type = ""
    for candidate in ("小型货车", "大型货车", "重型货车", "货车", "轿车", "客车",
                      "SUV", "皮卡", "面包车", "卡车", "挂车", "罐车", "越野车",
                      "小货车"):
        if candidate in text:
            vehicle_type = candidate
            break
    is_yellow_plate = bool(re.search(r"车牌号?码?[:：]?黄", text))
    is_truck_brand = bool(re.search(
        r"(卡车|渣土|罐车|重卡|挂车|牵引|集装箱|混凝土搅拌|解放|天龙|乘龙|陕汽|重汽|HOWO|江淮.*好运)",
        text, re.IGNORECASE,
    ))
    if (is_yellow_plate or is_truck_brand) and vehicle_type in ("", "SUV", "轿车"):
        vehicle_type = "货车"

    # 2) 机位与方向：设备编号（DMR/DMC/CK）优先，其次中文监控点名（兼容抑拍/批拍），最后 Camera xx
    raw_role = ""
    if re.search(r"设备编号[:：]?DMR[0O0-9Q]*", text, re.IGNORECASE) or re.search(r"DMR[0O0-9Q]{1,3}", text, re.IGNORECASE):
        raw_role = "gate_in"
    elif re.search(r"设备编号[:：]?DMC[0O0-9Q]*", text, re.IGNORECASE) or re.search(r"DMC[0O0-9Q]{1,3}", text, re.IGNORECASE):
        raw_role = "gate_out"
    elif (re.search(r"设备编号[:：]?CK[0O0-9Q]*", text, re.IGNORECASE)
          or re.search(r"CK[0O0-9Q]{1,3}", text, re.IGNORECASE)
          or "仓库" in text):
        raw_role = "warehouse"
    elif re.search(r"大门.{0,4}进", text):
        raw_role = "gate_in"
    elif re.search(r"大门.{0,4}出", text):
        raw_role = "gate_out"

    if raw_role == "gate_in":
        camera = "大门抓拍进"
    elif raw_role == "gate_out":
        camera = "大门抓拍出"
    elif raw_role == "warehouse":
        camera = "仓库抓拍"
    else:
        camera_match = re.search(r"(camera\s*\d{1,3})", text, re.IGNORECASE)
        camera = camera_match.group(1).replace(" ", "") if camera_match else ""
    return camera, vehicle_type


def fix_opposite_gate_rear_directions(
    conn: sqlite3.Connection,
    park_filter: str = "",
    max_gap_sec: int = 60,
) -> int:
    """基于「同车牌 1~60 秒先后触发对向大门相机」识别过门背向车尾抓拍并纠正方向。

    现场单通道双向大门同时装有 DMR（进）和 DMC（出）两个背靠背相机：
    - 车辆进门时：先触发 DMR（正脸进），5~30 秒后车尾驶过 DMC（背向抓拍）-> 第 2 张改标
      「大门抓拍进(背向)」，不再误判为出门；
    - 车辆出门时：先触发 DMC（正脸出），5~30 秒后车尾驶过 DMR（背向抓拍）-> 第 2 张改标
      「大门抓拍出(背向)」，不再误判为进门；
    - 单独只触发一次 DMC（如地磅直接驶出未触发 DMR），保持「大门抓拍出」不误翻。
    已由人工标注 corrected_direction 的抓拍一律跳过不覆盖。
    """
    where = ("stage='抓拍' AND coalesce(capture_time,'')!='' "
             "AND osd_camera IN ('大门抓拍进', '大门抓拍出') "
             "AND coalesce(park,'') NOT LIKE ?")
    params: list = [QUARANTINE_PARK_LIKE]
    if park_filter:
        where += " AND coalesce(park,'')=?"
        params.append(park_filter)

    # 查出已有人工方向覆盖的抓拍，避免改写人工结论
    manual_dir_ids: set[int] = set()
    has_ann = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='capture_annotations'"
    ).fetchone()
    if has_ann:
        manual_dir_ids = {
            int(r[0]) for r in conn.execute(
                "SELECT capture_id FROM capture_annotations WHERE coalesce(corrected_direction,'')!=''"
            ).fetchall()
        }

    rows = conn.execute(
        f"""SELECT id, coalesce(park,''), capture_time, osd_camera,
                   coalesce(plate_final,''), coalesce(osd_plate,''), coalesce(lpr_plate,'')
            FROM captures WHERE {where}
            ORDER BY coalesce(park,''), capture_time, id""",
        params,
    ).fetchall()

    # 按园区维护最近 max_gap_sec 秒内「未被翻转的原始正向大门事件」滑窗
    recent_by_park: dict[str, list[tuple[datetime, str, str]]] = {}
    updates: list[tuple[str, int]] = []
    for cid, park, ct, cam, pf, op, lp in rows:
        cid = int(cid)
        plate = pf or op or lp
        if not plate or len(_plate_body(plate)) < 5:
            continue
        try:
            ts = datetime.strptime(ct, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        window = recent_by_park.setdefault(park, [])
        # 清理超出 max_gap_sec 的历史项
        while window and (ts - window[0][0]).total_seconds() > max_gap_sec:
            window.pop(0)
        opposite = "大门抓拍出" if cam == "大门抓拍进" else "大门抓拍进"
        matched_prev = False
        for prev_ts, prev_cam, prev_plate in reversed(window):
            gap = (ts - prev_ts).total_seconds()
            if gap <= 0 or gap > max_gap_sec:
                continue
            if prev_cam == opposite and _plate_edit_distance(prev_plate, plate) <= 1.0:
                matched_prev = True
                break
        if matched_prev and cid not in manual_dir_ids:
            new_cam = f"{opposite}(背向)"
            updates.append((new_cam, cid))
        else:
            window.append((ts, cam, plate))

    # 规则 2（出磅后 0~10 分钟内 DMR 车尾漏拍兜底）：
    # 若 DMC 未抓拍到出门正脸，车辆刚完成「出磅」0~10 分钟内经过大门被 DMR 拍到背向车尾
    # （物理上不可能在 10 分钟内离场装粮再次入场），纠偏为「大门抓拍出(背向)」。
    # 车牌集合取整单所有已知车牌（避免出磅单帧被磅旁停放他车遮挡车牌时漏判）。
    flipped_ids = {cid for _, cid in updates}
    order_wout: dict[tuple[str, str], datetime] = {}
    order_plates: dict[tuple[str, str], set[str]] = {}
    for o_park, o_id, o_stage, o_ct, o_pf, o_lp in conn.execute(
        """SELECT coalesce(park,''), order_id, coalesce(stage,''),
                  coalesce(capture_time,''), coalesce(plate_final,''), coalesce(lpr_plate,'')
           FROM captures
           WHERE coalesce(order_id,'')!='' AND order_id!='未关联图片'
             AND coalesce(park,'') NOT LIKE ?""",
        (QUARANTINE_PARK_LIKE,),
    ).fetchall():
        if park_filter and o_park != park_filter:
            continue
        ok = (o_park, o_id)
        for p in (o_pf, o_lp):
            if p and len(_plate_body(p)) >= 5:
                order_plates.setdefault(ok, set()).add(p)
        if o_stage == "出磅" and o_ct:
            try:
                wts = datetime.strptime(o_ct, "%Y-%m-%d %H:%M:%S")
                if ok not in order_wout or wts > order_wout[ok]:
                    order_wout[ok] = wts
            except ValueError:
                pass
    weigh_outs_by_park: dict[str, list[tuple[datetime, set[str]]]] = {}
    for ok, wts in order_wout.items():
        ps = order_plates.get(ok)
        if ps:
            weigh_outs_by_park.setdefault(ok[0], []).append((wts, ps))

    for cid, park, ct, cam, pf, op, lp in rows:
        cid = int(cid)
        if cam != "大门抓拍进" or cid in flipped_ids or cid in manual_dir_ids:
            continue
        plate = pf or op or lp
        if not plate or len(_plate_body(plate)) < 5:
            continue
        try:
            ts = datetime.strptime(ct, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        for wts, ps in weigh_outs_by_park.get(park, ()):
            gap = (ts - wts).total_seconds()
            if 0 <= gap <= 600 and any(_plate_edit_distance(wp, plate) <= 1.0 for wp in ps):
                updates.append(("大门抓拍出(背向)", cid))
                flipped_ids.add(cid)
                break

    if updates:
        conn.executemany("UPDATE captures SET osd_camera=? WHERE id=?", updates)
        conn.commit()
    return len(updates)


def _refresh_footer_worker(payload: dict) -> dict:
    """进程池工作函数：快速重扫抓拍底部纯黑 OSD 条带，刷新机位/方向/车型/车牌。"""
    import cv2

    cid, filepath = payload["id"], payload["filepath"]
    has_plate = bool(payload.get("has_plate"))
    data = np.fromfile(str(filepath), dtype=np.uint8)
    rgb = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if rgb is None:
        return {"id": cid, "ok": False}
    y0 = _find_black_footer_y(rgb)
    if y0 < 0:
        return {"id": cid, "ok": False}
    footer = rgb[y0:, :]
    fw = footer.shape[1]
    scale = 0.5 if fw >= 1400 else 1.0
    strip = cv2.resize(footer, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else footer
    padded = cv2.copyMakeBorder(strip, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    ocr = get_ocr(large=False)
    res, _ = ocr(padded)
    if not res:
        return {"id": cid, "ok": False}
    row_h = max(15, int(35 * scale))
    res.sort(key=lambda it: (int(min(p[1] for p in it[0]) // row_h), min(p[0] for p in it[0])))
    bottom_joined = re.sub(r"\s+", "", "".join(str(t) for _, t, _ in res))
    camera, vehicle_type = _parse_osd_camera_and_type(bottom_joined)
    plate = ""
    for allow_partial in (False, True):
        for pattern in (r"车牌号码[:：]?([^\s:：]{5,12})", r"车牌号[:：]?([^\s:：]{5,12})"):
            for match in re.finditer(pattern, bottom_joined):
                plate = normalize_plate(match[1], allow_partial=allow_partial)
                if plate:
                    break
            if plate:
                break
        if plate:
            break
    if not camera and not plate and not has_plate and "无车牌" not in bottom_joined and fw >= 600:
        sub_joined, _ = _ocr_osd_footer(rgb)
        if sub_joined:
            bottom_joined = sub_joined + bottom_joined
            camera2, vt2 = _parse_osd_camera_and_type(bottom_joined)
            camera = camera or camera2
            vehicle_type = vehicle_type or vt2
            for pattern in (r"车牌号码[:：]?([^\s:：]{5,12})", r"车牌号[:：]?([^\s:：]{5,12})"):
                if m := re.search(pattern, bottom_joined):
                    plate = normalize_plate(m[1], allow_partial=True)
                    if plate:
                        break
    color = ""
    color_match = re.search(r"车身颜色[:：]?(.{0,12})", bottom_joined)
    if color_match:
        color_text = color_match.group(1)
        for color_char, candidate in (("白", "白色"), ("黑", "黑色"), ("灰", "灰色"),
                                      ("银", "银色"), ("红", "红色"), ("蓝", "蓝色"),
                                      ("黄", "黄色"), ("绿", "绿色"), ("棕", "棕色"),
                                      ("橙", "橙色")):
            if color_char in color_text:
                color = candidate
                break
    return {"id": cid, "ok": True, "osd_camera": camera, "osd_type": vehicle_type,
            "osd_color": color, "osd_plate": plate}


def refresh_osd_footers(conn: sqlite3.Connection, park_filter: str = "", limit: int = 0,
                        relevant_only: bool = False) -> dict[str, int]:
    """批量重刷 stage='抓拍' 的底部黑底 OSD 信息（设备编号 + 正/背向方向 + 仓库 + 货车类型）。

    relevant_only=True 时只刷新订单内抓拍 + 车牌与任意订单相近/无车牌/无设备名的未关联抓拍，
    跳过上千张无关社会过路车，速度提升数倍。
    """
    from concurrent.futures import ProcessPoolExecutor

    order_plate_bodies: set[str] = set()
    order_time_spans: list[tuple[str, str]] = []
    if relevant_only:
        for (pf,) in conn.execute(
            """SELECT DISTINCT plate_final FROM captures
               WHERE coalesce(order_id,'')!='' AND order_id!='未关联图片'
                 AND coalesce(plate_final,'')!=''"""
        ).fetchall():
            b = _plate_body(pf)
            if len(b) >= 5:
                order_plate_bodies.add(b)
        for tmin, tmax in conn.execute(
            """SELECT MIN(capture_time), MAX(capture_time) FROM captures
               WHERE coalesce(order_id,'')!='' AND order_id!='未关联图片'
                 AND coalesce(capture_time,'')!=''
               GROUP BY park, order_id"""
        ).fetchall():
            if tmin and tmax:
                order_time_spans.append((
                    (datetime.strptime(tmin, "%Y-%m-%d %H:%M:%S") - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                    (datetime.strptime(tmax, "%Y-%m-%d %H:%M:%S") + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
                ))

    where = "stage='抓拍' AND coalesce(park,'') NOT LIKE ?"
    params: list = [QUARANTINE_PARK_LIKE]
    if park_filter:
        where += " AND coalesce(park,'')=?"
        params.append(park_filter)
    sql = (f"SELECT id, filepath, coalesce(order_id,''), coalesce(osd_plate,''), "
           f"coalesce(plate_final,''), coalesce(osd_camera,''), coalesce(capture_time,'') "
           f"FROM captures WHERE {where} ORDER BY id")
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    payloads = []
    for cid, fp, oid, op, pf, cam, ct in rows:
        if relevant_only and oid == "未关联图片":
            eff_p = pf or op
            plate_hit = False
            if eff_p:
                pb = _plate_body(eff_p)
                plate_hit = (pb in order_plate_bodies or any(
                    _plate_edit_distance(eff_p, "皖" + ob) <= 1.0 for ob in order_plate_bodies
                ))
            time_hit = bool(ct) and any(s <= ct <= e for s, e in order_time_spans)
            if not plate_hit and not (not eff_p and time_hit):
                continue
        abs_p = resolve_stored_path(fp)
        if abs_p.is_file():
            payloads.append({"id": int(cid), "filepath": str(abs_p), "has_plate": bool(op or pf)})
    updated = flipped_cam = warehouse = 0
    if not payloads:
        return {"selected": 0, "updated": 0, "warehouse": 0, "rear_flipped": 0}
    print(f"[refresh-footer] 开始刷新 {len(payloads)} 张抓拍底部 OSD...", flush=True)
    conn.isolation_level = None
    with ProcessPoolExecutor(max_workers=min(4, (os.cpu_count() or 4)),
                             initializer=_ocr_worker_init) as pool:
        for res in pool.map(_refresh_footer_worker, payloads, chunksize=4):
            if not res.get("ok"):
                continue
            cid = res["id"]
            cam = res["osd_camera"]
            otype = res["osd_type"]
            ocolor = res["osd_color"]
            oplate = res["osd_plate"]
            if cam == "仓库抓拍":
                warehouse += 1
            elif "背向" in cam:
                flipped_cam += 1
            conn.execute(
                """UPDATE captures SET
                     osd_camera=CASE WHEN ?!='' THEN ? ELSE osd_camera END,
                     osd_type=CASE WHEN ?!='' THEN ? ELSE osd_type END,
                     osd_color=CASE WHEN ?!='' THEN ? ELSE osd_color END,
                     osd_plate=CASE WHEN ?!='' THEN ? ELSE osd_plate END,
                     plate_final=CASE WHEN ?!='' AND coalesce(plate_final,'')=''
                       AND NOT EXISTS (
                         SELECT 1 FROM capture_annotations ca
                         WHERE ca.capture_id=captures.id AND coalesce(ca.corrected_plate,'')!='')
                       THEN ? ELSE plate_final END
                   WHERE id=?""",
                (cam, cam, otype, otype, ocolor, ocolor, oplate, oplate, oplate, oplate, cid),
            )
            updated += 1
            if updated % 100 == 0:
                print(f"[refresh-footer] 已刷新 {updated}/{len(payloads)}（仓库 {warehouse}）", flush=True)
    flipped_cam = fix_opposite_gate_rear_directions(conn, park_filter=park_filter)
    if flipped_cam:
        print(f"[refresh-footer] 过门对向连拍背向纠偏：{flipped_cam} 张", flush=True)
    return {"selected": len(payloads), "updated": updated, "warehouse": warehouse, "rear_flipped": flipped_cam}


def ocr_capture(payload: dict) -> dict:
    """OCR 单张抓拍的时间戳与（抓拍镜头的）底部车辆描述。

    进程池工作函数：时间戳先走小图快速通道，失败再走宽区域大字号重试
    通道，仍失败则用日期框锚定时刻带走抢救通道；底部OSD只出现在抓拍
    镜头，其他阶段跳过。
    """
    import cv2

    filepath, stage = payload["filepath"], payload["stage"]
    # cv2.imread 不支持含中文的 Windows 路径，用 fromfile+imdecode 代替。
    data = np.fromfile(str(filepath), dtype=np.uint8)
    rgb = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"图片无法读取: {filepath}")
    height, width = rgb.shape[:2]

    def ocr_region(y0: float, y1: float, x1: float, scale: float,
                   large: bool = False) -> tuple[str, list[float], list[tuple]]:
        region = rgb[int(height * y0) : int(height * y1), 0 : int(width * x1)]
        if scale != 1.0:
            region = cv2.resize(region, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
        result, _ = get_ocr(large)(region)
        texts, confs, boxes = [], [], []
        if result:
            for box, text, conf in result:
                texts.append(str(text))
                confs.append(float(conf))
                xs = [p[0] / scale for p in box]
                ys = [p[1] / scale for p in box]
                boxes.append((str(text), min(xs), min(ys), max(xs), max(ys)))
        return (re.sub(r"\s+", "", " ".join(texts)), confs, boxes)

    def parse_datetime(joined: str) -> tuple[str, str]:
        date_part = time_part = ""
        if match := DATE_RE.search(joined):
            date_part = f"{int(match[1]):04d}-{int(match[2]):02d}-{int(match[3]):02d}"
        if match := TIME_RE.search(joined):
            time_part = f"{int(match[1]):02d}:{int(match[2]):02d}:{int(match[3]):02d}"
        return date_part, time_part

    # 时间戳是轨迹排序的基石。新版NVR命名自带拍摄时间（阶段+日期时刻+序号，
    # 如 入磅2026年9月20日08：48：11_2268117）：文件名时间是NVR写入的权威
    # 时间，直接采用，省去左上角两通道OCR（底部OSD仍照常识别车辆形象）。
    # 老命名（阶段_uuid）才走图像OCR：小图快速通道 -> 大字号重试 -> 时刻带抢救。
    name_time = FILENAME_TIME_RE.search(Path(filepath).stem)
    confidence_values: list[float] = []
    if name_time:
        capture_date = (f"{int(name_time[1]):04d}-{int(name_time[2]):02d}-"
                        f"{int(name_time[3]):02d}")
        hms = (f"{int(name_time[4]):02d}:{int(name_time[5]):02d}:"
               f"{int(name_time[6]):02d}")
    else:
        top_joined, top_confs, top_boxes = ocr_region(0.0, 0.20, 0.60, 0.6)
        capture_date, hms = parse_datetime(top_joined)
        date_box = next((b for b in top_boxes if DATE_RE.search(b[0])), None)
        confidence_values = list(top_confs)
        if not (capture_date and hms):
            top_joined, retry_confs, retry_boxes = ocr_region(0.0, 0.22, 0.80, 0.9, large=True)
            capture_date, hms = parse_datetime(top_joined)
            date_box = date_box or next((b for b in retry_boxes if DATE_RE.search(b[0])), None)
            confidence_values += list(retry_confs)
        if not hms and capture_date and date_box:
            rescued = _rescue_timestamp(rgb, date_box)
            if rescued:
                hms = rescued
    time_ok = bool(capture_date and hms)

    plate = color = vehicle_type = camera = ""
    bottom_date = bottom_hms = ""
    if stage == "抓拍":
        # 优先精确切出底部纯黑 OSD 条带识别（规避画面车头/路面纹理干扰，
        # 车牌召回率从 ~65% 提到 ~99%）；无纯黑底条的老相机图回退旧区域。
        bottom_joined, bottom_confs = _ocr_osd_footer(rgb)
        if not bottom_joined:
            bottom_joined, bottom_confs, _ = ocr_region(
                0.72, 1.0, 1.0, 0.75, large=True)
        confidence_values += list(bottom_confs)
        bottom_date, bottom_hms = parse_datetime(bottom_joined)
        for allow_partial in (False, True):
            for pattern in (r"车牌号码[:：]?([^\s:：]{5,12})", r"车牌号[:：]?([^\s:：]{5,12})"):
                for match in re.finditer(pattern, bottom_joined):
                    plate = normalize_plate(match[1], allow_partial=allow_partial)
                    if plate:
                        break
                if plate:
                    break
            if plate:
                break
        color_match = re.search(r"车身颜色[:：]?(.{0,12})", bottom_joined)
        if color_match:
            color_text = color_match.group(1)
            for color_char, candidate in (("白", "白色"), ("黑", "黑色"), ("灰", "灰色"),
                                          ("银", "银色"), ("红", "红色"), ("蓝", "蓝色"),
                                          ("黄", "黄色"), ("绿", "绿色"), ("棕", "棕色"),
                                          ("橙", "橙色")):
                if color_char in color_text:
                    color = candidate
                    break
        camera, vehicle_type = _parse_osd_camera_and_type(bottom_joined)

    # 顶部失败时才用底部OSD时间兜底（其日期可能被换行切断，写库阶段
    # 仍会按 folder/embedded 校验，不通过则标记 conflict）。
    if not (capture_date and hms) and bottom_date and bottom_hms:
        capture_date, hms = bottom_date, bottom_hms
        time_ok = bool(capture_date and hms)

    return {
        "id": payload["id"],
        "stage": payload["stage"],
        "date": capture_date,
        "hms": hms,
        "time_ok": time_ok,
        "time_source": "filename" if name_time else "ocr",
        "bottom_date": bottom_date,
        "bottom_hms": bottom_hms,
        "osd_plate": plate,
        "osd_color": color,
        "osd_type": vehicle_type,
        "osd_camera": camera,
        "ocr_conf": round(float(np.mean(confidence_values)) if confidence_values else 0.0, 4),
        "quality": _cv_quality(rgb),
    }


def _cv_quality(bgr: np.ndarray) -> float:
    """与 image_quality 相同的质量分，直接基于已解码的BGR图计算。"""
    import cv2

    scale = 480.0 / max(1, bgr.shape[1])
    resized = cv2.resize(bgr, (480, max(1, int(bgr.shape[0] * scale))))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)
    contrast = min(1.0, float(gray.std()) / 64.0)
    sharpness = min(1.0, (float(np.abs(np.diff(gray, axis=0)).mean()) +
                          float(np.abs(np.diff(gray, axis=1)).mean())) / 32.0)
    exposure = max(0.0, 1.0 - abs(float(gray.mean()) - 128.0) / 128.0)
    return round(100.0 * (0.45 * sharpness + 0.35 * contrast + 0.20 * exposure), 1)


def image_quality(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L").resize((480, 316)), dtype=np.float32)
    contrast = min(1.0, float(gray.std()) / 64.0)
    sharpness = min(1.0, (float(np.abs(np.diff(gray, axis=0)).mean()) +
                          float(np.abs(np.diff(gray, axis=1)).mean())) / 32.0)
    exposure = max(0.0, 1.0 - abs(float(gray.mean()) - 128.0) / 128.0)
    return round(100.0 * (0.45 * sharpness + 0.35 * contrast + 0.20 * exposure), 1)


def process_missing(conn: sqlite3.Connection, limit: int = 0, force: bool = False,
                    retry_failed: bool = False, stage: str = "",
                    retry_osd: bool = False) -> int:
    """补充 OCR：时间戳、底部OSD参考、质量分。失败时降级为阶段顺序，不伪造时间。

    OCR 在进程池中并行执行（每个 worker 独立加载 RapidOCR），主进程写库；
    结束后对仍无时间的帧做兄弟帧补时（fill_times_from_siblings）。
    retry_failed=True 只重跑 ocr_status='fallback' 的历史失败帧（启用时刻带
    抢救通道后再试一次），不动已成功的帧。
    retry_osd=True 只重跑 stage='抓拍' 且 osd_plate 为空的历史帧（用纯黑底部
    条带精准 OCR 补回遗漏的 OSD 车牌/颜色/方向）。
    stage 参数限定只处理指定环节（配合 force 做定向回填，如历史抓拍照片
    补中文设备名 osd_camera），空串=不限。
    """
    from concurrent.futures import ProcessPoolExecutor

    # 流水线/服务可能并发写库：逐条自动提交，避免长事务长期持有写锁。
    conn.isolation_level = None
    if force:
        where = "1=1"
    elif retry_osd:
        where = "stage='抓拍' AND coalesce(osd_plate,'')=''"
    elif retry_failed:
        where = "coalesce(ocr_status,'')='fallback'"
    else:
        where = "coalesce(ocr_status,'')=''"
    # 人工改过时间的帧（ocr_status='manual'）不被自动 OCR 覆盖
    where += " AND coalesce(ocr_status,'')!='manual'"
    if stage:
        where += " AND stage=?"
        rows = conn.execute(
            f"""SELECT id, filepath, stage FROM captures
                WHERE {where}
                ORDER BY date, order_id, stage_rank, filename"""
            + (" LIMIT ?" if limit else ""),
            (stage,) if not limit else (stage, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            f"""SELECT id, filepath, stage FROM captures
                WHERE {where}
                ORDER BY date, order_id, stage_rank, filename"""
            + (" LIMIT ?" if limit else ""),
            () if not limit else (limit,),
        ).fetchall()
    payloads = [{"id": int(row[0]), "filepath": str(resolve_stored_path(row[1])),
                 "stage": row[2]} for row in rows]
    started = time.time()
    processed = failed = fallback = conflict = 0
    with ProcessPoolExecutor(max_workers=min(6, (os.cpu_count() or 4)),
                             initializer=_ocr_worker_init) as pool:
        for osd in pool.map(ocr_capture, payloads, chunksize=4):
            capture_id = osd["id"]
            stage = str(osd.get("stage") or "")
            try:
                capture_time = ""
                ocr_status = "ok"
                if osd["time_ok"]:
                    capture_time = f"{osd['date']} {osd['hms']}"
                    row = conn.execute(
                        "SELECT date, order_id FROM captures WHERE id=?", (capture_id,)
                    ).fetchone()
                    folder_date, order_id = str(row[0]), str(row[1] or "")
                    # 归档文件夹日期不可作唯一校验依据：跨午夜订单按出磅日期
                    # 归档，图片实际拍摄日期可能是前一天（实施文档 3.1）。可信
                    # 判据 = 与文件夹日期一致 / 与订单内嵌日期一致（第7-10位
                    # MMdd，如 68002608090040 中的 0809）/ 底部OSD“抓拍时间”
                    # 的时刻部分互证（±5秒；其日期常被换行切断，不参与校验）。
                    embedded_date = ""
                    digits = "".join(ch for ch in order_id if ch.isdigit())
                    # 单号前缀不限定（老库 68 开头，农易行库 109 开头），
                    # 只要第 7-10 位是合法 MMDD 即作为内嵌日期证据。
                    if len(digits) >= 10:
                        mm, dd = digits[6:8], digits[8:10]
                        if mm.isdigit() and 1 <= int(mm) <= 12 and dd.isdigit() and 1 <= int(dd) <= 31:
                            embedded_date = f"{folder_date[:4]}-{mm}-{dd}"
                    date_trusted = (
                        str(osd["date"]) == folder_date
                        or (embedded_date and str(osd["date"]) == embedded_date)
                    )
                    bottom_hms = str(osd.get("bottom_hms") or "")
                    if osd.get("time_source") == "filename":
                        # 文件名时间是NVR写入的权威拍摄时间（确定性解析，非OCR），
                        # 与归档日期矛盾时（跨午夜按出磅日期归档）以文件名为准。
                        date_trusted = True
                    elif bottom_hms and str(osd["hms"]):
                        try:
                            t_top = datetime.strptime(str(osd["hms"]), "%H:%M:%S")
                            t_bottom = datetime.strptime(bottom_hms, "%H:%M:%S")
                            drift = abs((t_top - t_bottom).total_seconds())
                        except ValueError:
                            drift = 1e9
                        if drift <= 5:
                            date_trusted = True
                    if not date_trusted:
                        ocr_status = "conflict"
                        capture_time = ""
                        conflict += 1
                else:
                    ocr_status = "fallback"
                    fallback += 1
                hour = int(str(osd["hms"]).split(":")[0]) if osd["hms"] else -1
                osd_plate = str(osd["osd_plate"])
                status_reason = ""
                if stage == "人脸":
                    # 人脸快照不是车辆图片：保留入库时的非车辆标记。
                    status_reason = "人脸抓拍，非车辆图片"
                elif ocr_status != "ok":
                    status_reason = "时间戳OCR不可用，时间不确定（按阶段顺序推断）"
                # 车牌字段只在本次读到新 OSD 车牌且尚无 LPR 模型结果时回填：
                # 重跑/补扫 OSD 时不得覆盖已有 LPR 主证据（实施文档 13.1），若与
                # LPR 读数一致则升级为"OSD+车牌模型一致"；人工已标车牌更不得覆盖。
                conn.execute(
                    """UPDATE captures SET capture_time=?, hour=?, ocr_status=?, quality_score=?,
                              osd_plate=CASE WHEN ?!='' THEN ? ELSE osd_plate END,
                              osd_color=CASE WHEN ?!='' THEN ? ELSE osd_color END,
                              osd_type=CASE WHEN ?!='' THEN ? ELSE osd_type END,
                              osd_camera=CASE WHEN ?!='' THEN ? ELSE osd_camera END,
                              plate_final=CASE WHEN ?!='' AND coalesce(lpr_plate,'')=''
                                AND NOT EXISTS (
                                    SELECT 1 FROM capture_annotations ca
                                    WHERE ca.capture_id=captures.id
                                      AND coalesce(ca.corrected_plate,'')!='')
                                 THEN ? ELSE plate_final END,
                              plate_observed=CASE WHEN ?!='' AND coalesce(plate_observed,'')=''
                                 THEN ? ELSE plate_observed END,
                              plate_confidence=CASE WHEN ?!='' AND coalesce(lpr_plate,'')=''
                                 THEN ? ELSE plate_confidence END,
                              recognition_source=CASE
                                WHEN ?!='' AND coalesce(lpr_plate,'')!='' AND lpr_plate=?
                                  THEN 'OSD+车牌模型一致'
                                WHEN ?!='' AND coalesce(lpr_plate,'')='' THEN ?
                                ELSE recognition_source END,
                              capture_status=CASE
                                WHEN capture_status='no_vehicle' THEN capture_status ELSE 'vehicle' END,
                              status_reason=?
                       WHERE id=?""",
                    (capture_time, hour, ocr_status, osd["quality"],
                     osd_plate, osd_plate,
                     str(osd["osd_color"]), str(osd["osd_color"]),
                     str(osd["osd_type"]), str(osd["osd_type"]),
                     str(osd["osd_camera"]), str(osd["osd_camera"]),
                     osd_plate, osd_plate, osd_plate, osd_plate,
                     osd_plate, float(osd["ocr_conf"]),
                     osd_plate, osd_plate, osd_plate, "OSD文字",
                     status_reason, capture_id),
                )
                processed += 1
            except Exception as exc:
                failed += 1
                print(f"OCR写库失败 #{capture_id}: {exc}", flush=True)
            if processed % 100 == 0:
                conn.commit()
                print(f"OCR进度 {processed}/{len(payloads)}，用时 {time.time() - started:.0f}s",
                      flush=True)
    conn.commit()
    print(f"OCR完成：{processed}张（fallback {fallback}，conflict {conflict}，失败 {failed}）",
          flush=True)
    filled = fill_times_from_siblings(conn)
    if filled:
        print(f"兄弟帧补时：{filled}张（同订单同阶段连拍帧继承）", flush=True)
    rear_fixed = fix_opposite_gate_rear_directions(conn)
    if rear_fixed:
        print(f"过门对向连拍背向纠偏：{rear_fixed}张", flush=True)
    return processed


def fill_times_from_siblings(conn: sqlite3.Connection) -> int:
    """兄弟帧补时：左上角OCR失败的帧，从同订单同阶段连拍帧继承时间。

    实测同订单同阶段相邻帧 96% 间隔 <=10 秒（NVR 对同一事件连拍数张），
    失败帧中 95% 存在带可信时间（ocr_status='ok'）的兄弟帧。继承规则：
    兄弟时间聚成簇（簇内最大间隔 <=600 秒，覆盖实测 p99=481 秒），
    恰好一簇才继承其中位时间——存在多簇说明同阶段有多个批次，无法判断
    失败帧属于哪一批，宁可缺时间也不写错时间。
    conflict 帧不补：其自身读数与归档日期矛盾（常见于跨午夜订单），
    需要人工甄别，自动补时可能固化错误。
    """
    failed = conn.execute(
        """SELECT id, park, order_id, stage FROM captures
           WHERE coalesce(capture_time,'')='' AND ocr_status='fallback'
             AND coalesce(order_id,'')!='' AND stage!='人脸'
             AND coalesce(capture_status,'vehicle')!='no_vehicle'"""
    ).fetchall()
    if not failed:
        return 0
    by_group: dict[tuple[str, str, str], list[str]] = {}
    for row in conn.execute(
        """SELECT coalesce(park,''), order_id, stage, capture_time FROM captures
           WHERE ocr_status='ok' AND capture_time IS NOT NULL AND capture_time!=''
             AND stage!='人脸'
             AND coalesce(capture_status,'vehicle')!='no_vehicle'"""
    ).fetchall():
        by_group.setdefault((row[0], row[1], row[2]), []).append(row[3])

    def cluster(times: list[str]) -> list[list[datetime]]:
        parsed = sorted(datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in times)
        clusters: list[list[datetime]] = [[parsed[0]]]
        for current in parsed[1:]:
            if (current - clusters[-1][-1]).total_seconds() <= 600:
                clusters[-1].append(current)
            else:
                clusters.append([current])
        return clusters

    filled = 0
    conn.isolation_level = None
    for capture_id, park, order_id, stage in failed:
        times = by_group.get((park or "", order_id, stage))
        if not times:
            continue
        clusters = cluster(times)
        if len(clusters) != 1:
            continue
        median = clusters[0][len(clusters[0]) // 2]
        capture_time = median.strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            """UPDATE captures SET capture_time=?, hour=?, ocr_status='sibling',
                      status_reason='左上角时间OCR失败，按同订单同阶段连拍帧补时（帧间隔<=10分钟）'
               WHERE id=?""",
            (capture_time, median.hour, capture_id),
        )
        filled += 1
    conn.commit()
    return filled


def _ocr_worker_init() -> None:
    """进程池 worker 初始化：限制线程数并预加载 OCR 模型（get_ocr 复用）。"""
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    try:
        cv2 = __import__("cv2")
        cv2.setNumThreads(2)
    except Exception:
        pass
    get_ocr()
    get_ocr(large=True)


def blob_to_feature(blob: bytes) -> np.ndarray:
    feature = np.frombuffer(blob, dtype=np.float32)
    return feature / (np.linalg.norm(feature) + 1e-9)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size:
        return -1.0
    return float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-9))


# 车牌 OCR 易混淆字符对（实施文档 13.1，已在 test_nov.db + granary.db 标定零误并）：
# - 硬等价（成本 0.0）：O<->0、I<->1（国标 GA 36-2018 民用车牌不含 O/I）
# - 软形近（成本 0.5）：0<->D/Q、1<->L/T、2<->Z、5<->S、8<->B、6<->G、C<->G、
#   U<->V、E<->F、M<->N。两处形近差异累计 1.0 仍判同牌；非形近差异成本 1.0；
#   少/多 1 个字符（Levenshtein 插删）成本 1.0，同样匹配（<=1.0）。
_CONFUSABLE_PAIRS: set[frozenset[str]] = {
    frozenset(("0", "D")), frozenset(("0", "Q")), frozenset(("D", "Q")),
    frozenset(("1", "L")), frozenset(("1", "T")),
    frozenset(("2", "Z")),
    frozenset(("5", "S")),
    frozenset(("8", "B")),
    frozenset(("6", "G")), frozenset(("C", "G")),
    frozenset(("U", "V")),
    frozenset(("E", "F")),
    frozenset(("M", "N")),
}


def _plate_body(plate: str | None) -> str:
    """提取车牌有效比对主体：无视开头省份汉字，保留后续字母数字及挂/学等后缀。

    - '皖K5713A' -> 'K5713A'
    - 'K5713A'（首字缺失/未识别）-> 'K5713A'（不会盲切掉发牌机关字母）
    - 将民用车牌不存在的 O/I 统一归一为 0/1。
    """
    if not plate:
        return ""
    s = re.sub(r"[^\u4e00-\u9fffA-Z0-9挂学警港澳使领]", "", plate.strip().upper())
    if s and "\u4e00" <= s[0] <= "\u9fff":
        s = s[1:]
    return s.replace("O", "0").replace("I", "1")


def _plate_edit_distance(left: str, right: str) -> float:
    """车牌加权编辑距离（实施文档 13.1）：
    1) 首个省份汉字完全不参与比对（兼容首字错认或缺失）；
    2) O<->0、I<->1 成本 0.0；形近混淆字符对（0/D/Q、1/L/T、2/Z、5/S、
       8/B、6/G/C、U/V、E/F、M/N）替换成本 0.5；其他替换成本 1.0；
    3) 允许少/多 1 个字符（插/删成本 1.0，如 皖K5713 vs 皖K5713A、鲁WL887 vs 鲁WYL887）；
    4) 主体长度 <5 或长度差 >1 返回 99.0。
    判定口径：<= 1.0 视为同一车牌，> 1.0 视为不同车牌。
    """
    a, b = _plate_body(left), _plate_body(right)
    if not a or not b or min(len(a), len(b)) < 5 or abs(len(a) - len(b)) > 1:
        return 99.0
    if a == b:
        return 0.0
    m, n = len(a), len(b)
    dp = [float(j) for j in range(n + 1)]
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = float(i)
        ca = a[i - 1]
        for j in range(1, n + 1):
            cb = b[j - 1]
            temp = dp[j]
            if ca == cb:
                sub = 0.0
            elif frozenset((ca, cb)) in _CONFUSABLE_PAIRS:
                sub = 0.5
            else:
                sub = 1.0
            dp[j] = min(dp[j] + 1.0, dp[j - 1] + 1.0, prev + sub)
            prev = temp
    return dp[n]


def _confusable_fuzzy_key(text: str) -> str:
    """短片段检索用的粗粒度形近归一键（仅用于 Web 查询片段子串匹配）。"""
    table = str.maketrans({
        "O": "0", "D": "0", "Q": "0",
        "I": "1", "L": "1", "T": "1",
        "Z": "2", "S": "5", "B": "8",
        "G": "6", "V": "U", "F": "E", "N": "M",
    })
    return _plate_body(text).translate(table)


def plate_matches_query(candidate: str | None, query: str | None) -> bool:
    """车牌检索模糊匹配（供 server.py /api/search 与 /api/track 复用）：
    - 完整/近完整查询（去首字后 >=5 位）：走与归类完全一致的 _plate_edit_distance <= 1.0
      （无视省份首字 + 形近字符 + 少/多 1 位）；
    - 短片段查询（如 '5713'、'KM9'）：去首字后按形近归一做子串匹配；
    - 仅输入单字省份（如 '皖'）：按原文前缀/子串匹配。
    """
    if not candidate or not query:
        return False
    q_raw = re.sub(r"[^\u4e00-\u9fffA-Z0-9挂学警港澳使领]", "", query.strip().upper())
    c_raw = re.sub(r"[^\u4e00-\u9fffA-Z0-9挂学警港澳使领]", "", candidate.strip().upper())
    if not q_raw or not c_raw:
        return False
    qb = _plate_body(q_raw)
    cb = _plate_body(c_raw)
    if not qb:
        return q_raw in c_raw
    if not cb:
        return False
    if qb in cb:
        return True
    if len(qb) >= 5:
        return _plate_edit_distance(c_raw, q_raw) <= 1.0
    return _confusable_fuzzy_key(qb) in _confusable_fuzzy_key(cb)


def _loose_plate_match(loose: dict, order_consensus: str) -> tuple[str, float, str]:
    """双源车牌匹配（图像 LPR + 底部 OSD 互证互补）：
    返回 (状态, 距离, 来源说明)，状态为 'match' / 'conflict' / 'unplated'。

    规则：
    - 仅当图像 LPR、OSD、plate_final 全为空时返回 'unplated'；
    - 只要任一来源有车牌，而目标订单无共识车牌 -> 返回 'conflict'（防黑洞订单）；
    - LPR 与 OSD 同时命中（<=1.0）-> 'match'（双源一致，最强）；
    - LPR 命中（<=1.0）且（置信度 >=0.80 或无 OSD）-> 'match'（不被误判 OSD 耽误）；
    - OSD 命中（<=1.0）且（无 LPR 或 LPR 置信度 <0.85）-> 'match'（不被模糊低置信 LPR 耽误）；
    - 若高置信 LPR（>=0.85）与目标订单明确冲突（>1.0）-> 'conflict'（绝不被误判 OSD 强拉）；
    - 兼容单测仅传 loose['plate']（无 lpr_plate/osd_plate 键）的场景。
    """
    pf = loose.get("plate") or ""
    lpr = loose.get("lpr_plate") or ""
    lpr_conf = float(loose.get("lpr_conf") or 0.0)
    osd = loose.get("osd_plate") or ""
    guard = loose.get("guard_plate") or ""
    if not (pf or lpr or osd or guard):
        return "unplated", -1.0, ""
    if not order_consensus:
        return "conflict", 99.0, ""
    d_lpr = _plate_edit_distance(lpr, order_consensus) if lpr else 99.0
    d_osd = _plate_edit_distance(osd, order_consensus) if osd else 99.0
    d_pf = _plate_edit_distance(pf, order_consensus) if pf else 99.0
    d_guard = _plate_edit_distance(guard, order_consensus) if guard else 99.0
    lpr_is_full = bool(normalize_plate(lpr, allow_partial=False)) if lpr else False
    lpr_high_conflict = lpr_is_full and lpr_conf >= 0.85 and d_lpr > 1.0
    if d_lpr <= 1.0 and d_osd <= 1.0:
        return "match", min(d_lpr, d_osd), "双源车牌一致"
    if d_lpr <= 1.0 and (lpr_conf >= 0.80 or not osd):
        return "match", d_lpr, "图像车牌"
    if d_osd <= 1.0 and not lpr_high_conflict:
        return "match", d_osd, "OSD车牌"
    if d_pf <= 1.0 and not lpr_high_conflict:
        return "match", d_pf, "车牌"
    if d_guard <= 1.0 and not lpr_high_conflict:
        return "match", d_guard, "车牌"
    return "conflict", 99.0, ""


def _guard_plate(plate_final: str | None, lpr_plate: str | None = None,
                 lpr_conf: float | None = None,
                 osd_plate: str | None = None) -> str:
    """防串单车牌：优先采用 plate_final；若被历史清洗清空，回溯高置信 LPR
    (>=0.85) 或 OSD 车牌——避免真实车牌在归类环节被当成“无车牌”塞进异牌订单。"""
    if plate_final:
        return plate_final
    if lpr_plate and float(lpr_conf or 0.0) >= 0.85:
        return lpr_plate
    if osd_plate:
        return osd_plate
    return ""


def _order_plate_profile(plate_counts: Counter) -> tuple[str, set[str]]:
    """计算订单的共识车牌与已知车牌集合（去省份 + 形近/缺位距离 <=1.0 归簇计票）。

    返回 (consensus_plate, known_plates)：
    - 先按去省份且加权距离 <=1.0 归簇计票（如 鲁WYL887+皖WYL887、皖K5713A+皖B5713A、
      皖KM9U08+皖KM9V08 合并为同一簇），避免因省份/形近/单字误读落成无共识；
    - 仅当整单只有 1 个簇、或最大簇票数 >=2 且严格领先次大簇时才输出唯一共识，
      避免同单并存两辆不同车（如 2 vs 2）时强选或双向桥接；
    - known_plates 始终保留订单内全部非空车牌，供冲突否决使用。
    """
    known = {p for p, c in plate_counts.items() if p and c > 0}
    if not known:
        return "", set()
    # 优先频次更高、且带完整省份汉字的读数作簇代表
    ordered = sorted(
        known,
        key=lambda p: (-plate_counts[p], -len(p), p),
    )
    clusters: list[list[str]] = []
    for p in ordered:
        for cl in clusters:
            if _plate_edit_distance(p, cl[0]) <= 1.0:
                cl.append(p)
                break
        else:
            clusters.append([p])
    scored = sorted(
        ((sum(plate_counts[p] for p in cl), cl[0]) for cl in clusters),
        key=lambda item: (-item[0], item[1]),
    )
    top_count, top_plate = scored[0]
    if len(scored) == 1:
        return top_plate, known
    if top_count >= 2 and top_count > scored[1][0]:
        return top_plate, known
    return "", known


def associate_orders(conn: sqlite3.Connection) -> dict[str, int]:
    """车辆身份分组 v3：以车牌簇为锚的多证据归组。

    标定结论（granary.db，525 组）：VeRi 外观模型在本场无法独立判同——
    同车对均值相似度 0.869，但同型号同色异车对 p90 高达 0.805，任何全局
    阈值都会把 45%~73% 的异车对错误合并。因此车辆身份以车牌为锚：

    1. 基础组：每个订单一组；"未关联图片"桶内每张抓拍自成一组（允许
       通过车牌/外观挂进订单车簇）；
    2. 车牌簇（并查集，可传递桥接）：
       - 完全相同车牌（省份汉字无视）-> 同一车辆（跨天合并，主锚点）；
       - 去掉省份后逐字符差 <=1 -> 同一车辆（单帧 OCR 误差，不限同天，
         现场确认同一辆车多次过磅的读数差异就这一个字符）；
    3. 挂靠过的车牌不升级为锚点（链条无法延伸）；无牌订单/无法匹配
       的组保持独立成组——车牌确定不了就不并，宁缺毋滥。

    园区隔离：所有合并只允许发生在同一园区（captures.park）内的组之间，
    车牌索引/锚点均按园区分桶，union 时二次校验园区一致——同一辆车去
    多个园区是合法情况，但车辆身份按园区独立跟踪，绝不跨园区合并。

    车辆聚合只包含检测到车辆的图片：人脸快照与未检测到车辆的图片
    vehicle_id 保持为空，绝不进入车辆轨迹。

    人工合并/拆分走 /annotate 与 vehicle-merge 接口，不受此逻辑影响。
    """
    conn.execute("UPDATE captures SET vehicle_id=NULL")
    next_gid = int(conn.execute("SELECT coalesce(MAX(vehicle_id),0) FROM captures").fetchone()[0])

    # ---- 1) 基础组（车辆聚合只包含检测到车辆的图片；人脸/无车不参与）----
    groups: dict[int, dict] = {}
    orders = conn.execute(
        """SELECT park, order_id FROM captures
           WHERE coalesce(order_id,'')!='' AND order_id!='未关联图片'
             AND coalesce(capture_status,'vehicle')!='no_vehicle'
             AND coalesce(park,'') NOT LIKE ?
           GROUP BY park, order_id ORDER BY MIN(date), order_id""",
        (QUARANTINE_PARK_LIKE,),
    ).fetchall()
    for park, order_id in orders:
        next_gid += 1
        groups[next_gid] = {"kind": "order", "order_id": order_id, "park": park or ""}
        conn.execute(
            """UPDATE captures SET vehicle_id=?
               WHERE coalesce(park,'')=? AND order_id=?
                 AND coalesce(capture_status,'vehicle')!='no_vehicle'
                 AND (status_reason IS NULL OR (status_reason NOT LIKE '%待人工归组%'
                      AND status_reason NOT LIKE '%订单主车%'))""",
            (next_gid, park or "", order_id),
        )
    loose = conn.execute(
        """SELECT id, coalesce(park,'') FROM captures WHERE order_id='未关联图片'
           AND reid_embedding IS NOT NULL
           AND coalesce(capture_status,'vehicle')!='no_vehicle'
           AND coalesce(park,'') NOT LIKE ?""",
        (QUARANTINE_PARK_LIKE,),
    ).fetchall()
    for cid, park in loose:
        next_gid += 1
        groups[next_gid] = {"kind": "loose", "capture_id": cid, "park": park}
        conn.execute("UPDATE captures SET vehicle_id=? WHERE id=?", (next_gid, cid))
    # 同帧其他车辆实例：有车牌的作为独立基础组参与归并（无关车辆分开跟踪）
    inst_rows = conn.execute(
        """SELECT vi.id, vi.lpr_plate, vi.lpr_conf, c.date, c.park,
                  vi.vehicle_color, vi.vehicle_type, vi.reid_embedding
           FROM vehicle_instances vi JOIN captures c ON c.id=vi.capture_id
           WHERE vi.reid_embedding IS NOT NULL AND coalesce(vi.lpr_plate,'')!=''
             AND coalesce(c.park,'') NOT LIKE ?""",
        (QUARANTINE_PARK_LIKE,),
    ).fetchall()
    for iid, lpr_plate, lpr_conf, date, park, color, vtype, blob in inst_rows:
        next_gid += 1
        groups[next_gid] = {"kind": "instance", "capture_id": iid, "park": park or ""}
        conn.execute("UPDATE vehicle_instances SET vehicle_id=? WHERE id=?", (next_gid, iid))
        # 实例是单帧组：特征直接内置，走统一的组特征逻辑
        g = groups[next_gid]
        g["_prefetched"] = [(lpr_plate, lpr_conf, date, color, vtype, blob)]

    # 人工标注车牌集合（人工确认的车牌视为绝对高可信 1.0，不被第 6 步误读清洗）
    manual_plate_cids: set[int] = set()
    try:
        manual_plate_cids = {
            int(r[0]) for r in conn.execute(
                "SELECT capture_id FROM capture_annotations WHERE coalesce(corrected_plate,'')!=''"
            ).fetchall()
        }
    except sqlite3.OperationalError:
        pass
    # 人工标注同车/异车约束（pair_annotations，供聚类并查集防误并 + 第 9b 步重放）
    pair_same_list: list[tuple[int, int]] = []
    pair_diff_map: dict[int, set[int]] = defaultdict(set)
    try:
        for lid, rid, rel in conn.execute(
            "SELECT left_capture_id, right_capture_id, relation FROM pair_annotations "
            "WHERE relation IN ('same', 'different') ORDER BY updated_at"
        ).fetchall():
            lid, rid = int(lid), int(rid)
            if rel == "same":
                pair_same_list.append((lid, rid))
            elif rel == "different":
                pair_diff_map[lid].add(rid)
                pair_diff_map[rid].add(lid)
    except sqlite3.OperationalError:
        pass

    # 组特征：车牌（含来源置信）、日期、颜色/车型分布、组均值向量、成员 capture_id
    for gid, g in groups.items():
        if "_prefetched" in g:
            pref = g.pop("_prefetched")
            rows = [(0, p, lc, d, c, vt, b, "") for (p, lc, d, c, vt, b) in pref]
        elif g["kind"] == "order":
            rows = conn.execute(
                """SELECT id, coalesce(plate_final,'') AS plate, lpr_conf, date,
                          vehicle_color, vehicle_type, reid_embedding,
                          coalesce(recognition_source,'') AS src
                   FROM captures
                   WHERE vehicle_id=?
                     AND (reid_embedding IS NOT NULL OR recognition_source='人工标注'
                          OR coalesce(plate_final,'')!='')
                     AND coalesce(capture_status,'vehicle')!='no_vehicle'""",
                (gid,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, coalesce(plate_final,'') AS plate, lpr_conf, date,
                          vehicle_color, vehicle_type, reid_embedding,
                          coalesce(recognition_source,'') AS src
                   FROM captures WHERE vehicle_id=?""",
                (gid,),
            ).fetchall()
        g["plates"] = {}
        g["dates"] = set()
        g["colors"] = {}
        g["types"] = {}
        g["cids"] = set()
        embs = []
        plate_counts: dict[str, int] = {}
        plate_conf: dict[str, float] = {}
        n_frames = 0
        for cid, plate, lpr_conf, date, color, vtype, blob, src in rows:
            n_frames += 1
            if cid:
                g["cids"].add(int(cid))
            if plate:
                is_manual = (src == "人工标注") or (int(cid) in manual_plate_cids)
                conf = 1.0 if is_manual else float(lpr_conf or 0.9)
                plate_counts[plate] = plate_counts.get(plate, 0) + 1
                plate_conf[plate] = max(plate_conf.get(plate, 0.0), conf)
            if date:
                g["dates"].add(date)
            if color:
                g["colors"][color] = g["colors"].get(color, 0) + 1
            if vtype:
                g["types"][vtype] = g["types"].get(vtype, 0) + 1
            if blob:
                f = np.array(np.frombuffer(blob, np.float32), copy=True)
                norm = np.linalg.norm(f)
                if norm > 0:
                    embs.append(f / norm)
        # 组内车牌共识：订单内个别帧会幻觉出邻车/噪声车牌，或同框出现两辆车
        # （实测一单 4 帧车A + 2 帧车B）。若同时把两辆不同车（去省份距离>=2）
        # 都当共识，exact_union_pass 会借该订单把两条车队错误桥接成同一 vehicle_id。
        # 因此订单组只取唯一主牌簇（距离<=1归簇计票的严格胜出者）；平票不桥接。
        if g["kind"] == "order":
            dom_plate, _ = _order_plate_profile(Counter(plate_counts))
            if dom_plate:
                consensus = [p for p in plate_counts
                             if _plate_edit_distance(p, dom_plate) <= 1]
                cluster_total = sum(plate_counts[p] for p in consensus)
            else:
                consensus = []
                cluster_total = 0
        else:
            consensus = [p for p, c in plate_counts.items()
                         if c >= 2 or len(plate_counts) == 1]
            if not consensus and plate_counts:
                consensus = [max(plate_counts, key=lambda p: plate_conf[p])]
            cluster_total = sum(plate_counts[p] for p in consensus)
        # 可信车牌：主簇总计出现 >=2 次、或置信度 >=0.95。单帧低置信读数
        # 不能参与跨组合并（否则按 ≤1 字符规则会把整个车队链式吞进来）。
        g["plates"] = {p: plate_conf[p] for p in consensus}
        g["plate_trusted"] = {
            p: (plate_counts.get(p, 0) >= 2 or cluster_total >= 2
                or plate_conf[p] >= 0.95)
            for p in consensus
        }
        g["emb"] = None
        if embs:
            m = np.mean(embs[:12], axis=0)
            n = np.linalg.norm(m)
            g["emb"] = m / n if n > 0 else None

    # ---- 2) 车牌簇（并查集，可传递桥接）----
    parent = {gid: gid for gid in groups}

    def find(gid):
        while parent[gid] != gid:
            parent[gid] = parent[parent[gid]]
            gid = parent[gid]
        return gid

    merge_reasons: dict[int, str] = {}

    def union(a, b, reason):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        # 园区隔离防线：不同园区的组绝不合并（车牌/外观证据再强也不行），
        # 同一辆车去多个园区时身份按园区独立跟踪。
        if groups[ra].get("park", "") != groups[rb].get("park", ""):
            return False
        # 人工异车防线（pair_annotations relation='different'）：若两组内存在
        # 人工明确标记为"不同车辆"的抓拍对，禁止自动合并。
        if pair_diff_map:
            cids_a = groups[ra].get("cids") or ()
            cids_b = groups[rb].get("cids") or ()
            if cids_a and cids_b:
                for ca in cids_a:
                    if ca in pair_diff_map and (pair_diff_map[ca] & cids_b):
                        return False
        low, high = sorted((ra, rb))
        parent[high] = low
        ga, gb = groups[low], groups[high]
        for plate, conf in gb["plates"].items():
            ga["plates"][plate] = max(ga["plates"].get(plate, 0.0), conf)
        ga["dates"] |= gb["dates"]
        ga["cids"] |= gb["cids"]
        for k, v in gb["colors"].items():
            ga["colors"][k] = ga["colors"].get(k, 0) + v
        for k, v in gb["types"].items():
            ga["types"][k] = ga["types"].get(k, 0) + v
        merge_reasons[high] = reason
        return True

    # 两个索引（key 均带园区，跨园区同牌不合并）：
    # - exact_index（精确同牌合并）：可信车牌，或置信度 >=0.80 的合格读数
    #   （相同字符串本身就是强证据，0.9 左右的读数也该归到同一辆车）；
    # - near_index（1 字符相近合并）：仅可信车牌（count>=2 或 conf>=0.95），
    #   字符串不同，必须最严格的读数才允许桥接。
    exact_index: dict[tuple[str, str], list[int]] = {}
    near_index: dict[tuple[str, str], list[int]] = {}
    for gid, g in groups.items():
        park_key = g.get("park", "")
        for plate, trusted in g["plate_trusted"].items():
            body = _plate_body(plate)
            if len(body) < 5:
                continue
            key = (park_key, body)
            exact_index.setdefault(key, []).append(gid)
            if trusted:
                near_index.setdefault(key, []).append(gid)
            elif g["plates"].get(plate, 0) >= 0.80:
                exact_index.setdefault(key, [])
                exact_index[key].append(gid)

    def exact_union_pass() -> int:
        n = 0
        for gids in exact_index.values():
            roots_now = {find(g) for g in gids}
            if len(roots_now) > 1:
                first = sorted(roots_now)[0]
                for other in sorted(roots_now)[1:]:
                    if union(first, other, "相同车牌合并"):
                        n += 1
        return n

    exact_merges = exact_union_pass()

    # ---- 2) 锚定挂靠：只有"精确重复出现的车牌"才是车辆身份锚点 ----
    # 车牌簇内核 = 同一车牌（去省份）在多个订单组出现的精确聚类。
    # 其他可信车牌只有与某个锚点差 <=1 字符才挂靠到该簇；挂靠车牌
    # 不升级为锚点（链条无法延伸，杜绝 WE->W8->08 的传递漂移把
    # 车牌差距很大的车并进来）。
    # exact_index 的 key 带园区，因此锚点根与锚点车牌天然园区内闭环；
    # 挂靠匹配时再按园区过滤锚点，双保险。
    anchors: dict[int, set[str]] = defaultdict(set)  # root -> 锚点车牌(去省份)
    in_core = set()
    for (_, plate), gids in exact_index.items():
        distinct_roots = {find(g) for g in gids}
        if len(distinct_roots) > 1 or len(gids) > 1:
            root = find(gids[0])
            anchors[root].add(plate)
            in_core.update(gids)
            for g in gids:
                groups[g]["in_cluster"] = True
    # 合并后根可能变化，重算锚点归属
    anchors = defaultdict(set)
    for (_, plate), gids in exact_index.items():
        distinct_roots = {find(g) for g in gids}
        if len(distinct_roots) > 1 or len(gids) > 1:
            anchors[find(gids[0])].add(plate)
            in_core.update(gids)

    near_merges = 0
    for gid, g in groups.items():
        root = find(gid)
        if root in in_core:
            continue
        for plate, trusted in g["plate_trusted"].items():
            if not trusted:
                continue
            key = _plate_body(plate)
            if len(key) < 5:
                continue
            best_root, best_d = None, 99.0
            for root_anchor, anchor_plates in anchors.items():
                if root_anchor == find(gid):
                    continue
                if groups[root_anchor].get("park", "") != g.get("park", ""):
                    continue  # 只挂靠同园区的锚点
                d = min(_plate_edit_distance(key, a) for a in anchor_plates)
                if d < best_d:
                    best_root, best_d = root_anchor, d
            if best_root is not None and best_d <= 1.0:
                union(find(gid), best_root, "相近车牌挂靠")
                near_merges += 1
            break  # 每组只用一个代表性车牌判定，避免多牌轮流试挂

    # ---- 4) 写回（按根组重排 vehicle_id）----
    roots = sorted({find(gid) for gid in groups})
    remap = {root: idx for idx, root in enumerate(roots, start=1)}
    for gid, g in groups.items():
        root = find(gid)
        reason = merge_reasons.get(gid, "")
        if g["kind"] == "order":
            conn.execute(
                """UPDATE captures SET vehicle_id=?, plate_match_reason=CASE
                         WHEN ?!='' AND coalesce(plate_match_reason,'')='' THEN ?
                         WHEN plate_match_reason LIKE '人工合并%' THEN plate_match_reason
                         ELSE plate_match_reason END
                   WHERE coalesce(park,'')=? AND order_id=? AND coalesce(capture_status,'vehicle')!='no_vehicle'
                     AND (status_reason IS NULL OR (status_reason NOT LIKE '%待人工归组%'
                          AND status_reason NOT LIKE '%订单主车%'))""",
                (remap[root], reason, reason, g["park"], g["order_id"]),
            )
        elif g["kind"] == "instance":
            conn.execute(
                "UPDATE vehicle_instances SET vehicle_id=? WHERE id=?",
                (remap[root], g["capture_id"]),
            )
        else:
            conn.execute(
                """UPDATE captures SET vehicle_id=?, plate_match_reason=CASE
                         WHEN ?!='' THEN ? ELSE plate_match_reason END
                   WHERE id=?""",
                (remap[root], reason, reason, g["capture_id"]),
            )
    conn.commit()

    # ---- 5) 共识车牌远距拆分：组内某订单的共识车牌与组主车牌（去省份
    # 后）距离 >=2 时，该订单拆出（车辆分组只保留车牌基本一致的主车）----
    per_order: dict[tuple[int, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for vid, oid, plate in conn.execute(
        """SELECT vehicle_id, order_id, plate_final FROM captures
           WHERE vehicle_id IS NOT NULL AND coalesce(plate_final,'')!=''
             AND coalesce(capture_status,'vehicle')!='no_vehicle' AND order_id!='未关联图片'"""
    ):
        per_order[(vid, oid)][plate] += 1
    group_families: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    order_dom: dict[tuple[int, str], str] = {}
    for (vid, oid), counter in per_order.items():
        dom, _ = _order_plate_profile(Counter(counter))
        if not dom:
            dom = max(counter, key=lambda k: counter[k])
        order_dom[(vid, oid)] = dom
        group_families[vid][dom] += 1
    split = 0
    for vid, families in group_families.items():
        if len(families) < 2:
            continue
        main = max(families, key=lambda k: families[k])
        for (vid_, oid), plate_dom in order_dom.items():
            if vid_ != vid or plate_dom == main:
                continue
            if _plate_edit_distance(plate_dom, main) >= 2:
                conn.execute(
                    """UPDATE captures SET vehicle_id=NULL,
                              plate_match_reason='共识车牌与车辆主牌不符，已拆出待人工归组'
                       WHERE order_id=? AND vehicle_id=?""",
                    (oid, vid),
                )
                split += 1
    conn.commit()

    # ---- 6) 逐帧车牌清洗：与车辆主牌（去省份后）距离 >=2 的单帧读数
    # 判为误读——原始读数保留在 plate_observed 供审计，plate_final 清空，
    # 不再污染车牌检索与轨迹展示。
    # 两类读数不清洗（它们是真实存在的车牌，常见于同框第二辆车/过路车，
    # 清掉会让用户看到"识别成功却没车牌"）：
    # - 双重印证：图像模型与 OSD 独立读出同一字符串（OSD+车牌模型一致）；
    # - 组内重复出现：同一字符串在该车辆组 >=2 帧一致出现——噪声读数
    #   表现为随机字符串（一单 5 帧读出 5 个乱牌），一致的少数派读数是
    #   真实车辆，不是幻觉。----
    dominant: dict[int, str] = {}
    for vid in {v for v, _ in order_dom}:
        counter = Counter()
        for (v, _oid), plate in order_dom.items():
            if v == vid:
                counter[plate] += 1
        if counter:
            dominant[vid] = max(counter, key=lambda k: counter[k])
    cleaned = 0
    restored = 0
    member_rows = conn.execute(
        """SELECT vehicle_id, id, plate_final, coalesce(recognition_source,''),
                  coalesce(order_id,''), coalesce(lpr_conf, 0.0)
           FROM captures
           WHERE vehicle_id IS NOT NULL AND coalesce(plate_final,'')!=''"""
    ).fetchall()
    # 历史被清洗的行重新评估：旧规则清掉的真实车牌（双重印证 / 组内重复 /
    # 高置信模型读数 / 未关联图片）按新规则恢复（幂等可重复跑；恢复后
    # plate_final 还原，原文审计字段不变）。注意：旧库中部分被清洗行的
    # plate_match_reason 后来被归类/撤销覆写，因此对未关联图片或高置信 LPR
    # 即使 reason 已被覆写也要能按 lpr_plate/osd_plate/plate_observed 找回。
    cleaned_reason = "单帧车牌与车辆主牌差异过大，判为误读"
    restore_rows = conn.execute(
        """SELECT vehicle_id, id,
                  coalesce(lpr_plate,''), coalesce(plate_observed,''), coalesce(osd_plate,''),
                  coalesce(recognition_source,''),
                  coalesce(order_id,''), coalesce(lpr_conf, 0.0),
                  coalesce(plate_match_reason,'')
           FROM captures
           WHERE coalesce(plate_final,'')=''
             AND coalesce(capture_status,'vehicle')!='no_vehicle'
             AND (plate_match_reason=?
                  OR (order_id='未关联图片'
                      AND (coalesce(lpr_plate,'')!='' OR coalesce(osd_plate,'')!=''))
                  OR (vehicle_id IS NOT NULL
                      AND (coalesce(lpr_plate,'')!='' OR coalesce(osd_plate,'')!=''))
                  OR (coalesce(lpr_plate,'')!='' AND coalesce(lpr_conf, 0.0)>=0.95))
             AND coalesce(nullif(lpr_plate,''), nullif(plate_observed,''),
                          nullif(osd_plate,'')) IS NOT NULL
             AND NOT EXISTS (
                  SELECT 1 FROM capture_annotations ca
                  WHERE ca.capture_id=captures.id
                    AND coalesce(ca.corrected_plate,'')!='')""",
        (cleaned_reason,),
    ).fetchall()
    group_plate_counts: Counter = Counter()
    for vid, _cid, plate, _src, _oid, _conf in member_rows:
        group_plate_counts[(vid, _plate_body(plate))] += 1
    for vid, _cid, lp, obs, op, _src, _oid, _conf, _rsn in restore_rows:
        plate = lp or obs or op
        if vid is not None and plate:
            group_plate_counts[(vid, _plate_body(plate))] += 1
    for vid, cid, plate, src, oid, conf in member_rows:
        if oid == "未关联图片":
            continue  # 未关联散图绝不清洗车牌，防止后续归类盲视车牌冲突
        dom = dominant.get(vid)
        if not dom:
            continue
        if _plate_edit_distance(plate, dom) < 2:
            continue
        if (src in ("OSD+车牌模型一致", "人工标注", "人工复核")
                or cid in manual_plate_cids
                or group_plate_counts[(vid, _plate_body(plate))] >= 2
                or float(conf or 0.0) >= 0.95):
            continue  # 人工确认/双重印证/组内重复/高置信真实车牌，不清洗（交 6.5 拆出）
        conn.execute(
            """UPDATE captures
               SET plate_observed=coalesce(nullif(plate_observed,''), plate_final),
                   plate_final='',
                   plate_match_reason=?
               WHERE id=?""",
            (cleaned_reason, cid),
        )
        cleaned += 1
    for vid, cid, lp, obs, op, src, oid, conf, rsn in restore_rows:
        dom = dominant.get(vid) if vid is not None else None
        # 若任一原始来源（lpr / osd / observed）与当前所属车辆组主牌一致（<=1.0），
        # 优先恢复该一致来源（兼容大门重归类迁入正确订单后、或残缺LPR让位OSD后的恢复）。
        plate = ""
        if dom is not None:
            for cand in (lp, op, obs):
                if cand and _plate_edit_distance(cand, dom) <= 1.0:
                    plate = cand
                    break
        if not plate:
            lp_full = bool(normalize_plate(lp, allow_partial=False)) if lp else False
            op_full = bool(normalize_plate(op, allow_partial=False)) if op else False
            if op_full and not lp_full:
                plate = op
            else:
                plate = lp or obs or op
        # 未关联散图、与组主牌一致、双重印证、组内去省份重复、高置信模型读数
        # (>=0.95)，任一成立即恢复——真实车牌不该因历史清洗而缺席。
        if oid == "未关联图片":
            restore = True
        elif dom is not None and _plate_edit_distance(plate, dom) <= 1.0:
            restore = True
        elif (src == "OSD+车牌模型一致"
              or (vid is not None and group_plate_counts[(vid, _plate_body(plate))] >= 2)
              or float(conf or 0.0) >= 0.95):
            restore = True
        else:
            restore = False
        if restore and plate:
            new_reason = "" if rsn == cleaned_reason else rsn
            conn.execute(
                """UPDATE captures SET plate_final=?, plate_match_reason=?
                   WHERE id=?""",
                (plate, new_reason, cid),
            )
            restored += 1
    conn.commit()

    # ---- 6.5) 车牌一致性拆分：组内帧的车牌与组主牌（去省份后）距离 >=2，
    # 且读数双重印证（OSD+模型一致）/ 组内去省份重复出现（>=2 帧）/ 高置信
    # 模型读数（>=0.95）——这是同框或误选框的另一辆真实车辆，从主车轨迹拆出
    # 待人工归组。保留在订单时间线、车牌照常展示；单帧低置信孤立读数不拆
    # （误读噪声由第 6 步清洗处理）。每轮重算幂等（与车型拆分同机制）。
    plate_split = 0
    split_rows = conn.execute(
        """SELECT vehicle_id, id, plate_final, coalesce(recognition_source,''),
                  coalesce(lpr_conf, 0.0)
           FROM captures
           WHERE vehicle_id IS NOT NULL AND coalesce(plate_final,'')!=''"""
    ).fetchall()
    split_counts: Counter = Counter()
    for vid, _cid, plate, _src, _conf in split_rows:
        split_counts[(vid, _plate_body(plate))] += 1
    for vid, cid, plate, src, conf in split_rows:
        dom = dominant.get(vid)
        if dom is None or _plate_edit_distance(plate, dom) < 2:
            continue
        if (src in ("OSD+车牌模型一致", "人工标注", "人工复核")
                or cid in manual_plate_cids
                or split_counts[(vid, _plate_body(plate))] >= 2
                or float(conf or 0.0) >= 0.95):
            conn.execute(
                """UPDATE captures SET vehicle_id=NULL,
                          plate_match_reason='车牌与车辆主牌不符，已拆出待人工归组'
                   WHERE id=?""",
                (cid,),
            )
            plate_split += 1
    conn.commit()

    # ---- 7) 车型一致性拆分：车辆组内车型与主体车型不符的帧拆出轨迹 ----
    # 主体车型 = 组内非空车型的多数投票。颜色不参与判定（红色货车从车尾
    # 拍，货箱常被读成 white/truck）。拆出的帧保留在订单时间线里（沉底
    # 由排序保证），但不再属于该车辆轨迹，等待人工复核。
    type_split = 0
    for vid, dom_type in conn.execute(
        """SELECT c.vehicle_id, c.vehicle_type FROM captures c
           WHERE c.vehicle_id IS NOT NULL AND c.vehicle_type IS NOT NULL
             AND coalesce(c.capture_status,'vehicle')!='no_vehicle'
           GROUP BY c.vehicle_id, c.vehicle_type
           HAVING COUNT(*) = (
             SELECT MAX(cnt) FROM (
               SELECT COUNT(*) AS cnt FROM captures c2
               WHERE c2.vehicle_id = c.vehicle_id
                 AND coalesce(c2.capture_status,'vehicle')!='no_vehicle'
               GROUP BY c2.vehicle_type))"""
    ):
        n = conn.execute(
            """UPDATE captures SET vehicle_id=NULL,
                      status_reason='选框车型(' || coalesce(vehicle_type,'?') ||
                      ')与车辆主体车型(' || ? || ')不符，已拆出待人工复核'
               WHERE vehicle_id=? AND vehicle_type IS NOT NULL AND vehicle_type!=?
                 AND coalesce(capture_status,'vehicle')!='no_vehicle'""",
            (dom_type, vid, dom_type),
        ).rowcount
        type_split += n
    conn.commit()

    # ---- 8) 应用人工移动决定（人工意志优先于自动聚类）----
    # manual_vehicle_moves.target_plate=NULL -> 拆出为独立车辆；
    # 否则解析目标车牌（去省份）当前所在的车辆组并移入。目标组后续若因
    # 新证据合并/拆分，重跑分组时按车牌重新解析，人工决定持续生效。
    manual_moves = conn.execute(
        "SELECT capture_id, target_plate FROM manual_vehicle_moves ORDER BY moved_at"
    ).fetchall()
    manual_applied = 0
    for cid, target_plate in manual_moves:
        park_row = conn.execute(
            "SELECT coalesce(park,'') FROM captures WHERE id=?", (cid,)
        ).fetchone()
        park = park_row[0] if park_row else ""
        if target_plate:
            # 目标车牌只在同一园区内查找，人工移动不会把车移到别的园区。
            hit = conn.execute(
                """SELECT vehicle_id FROM captures
                   WHERE vehicle_id IS NOT NULL
                     AND coalesce(plate_final,'')!=''
                     AND coalesce(park,'')=?
                     AND substr(plate_final,2)=? LIMIT 1""",
                (park, target_plate),
            ).fetchone()
            new_vid = hit[0] if hit else None
        else:
            new_vid = None
        conn.execute("UPDATE captures SET vehicle_id=? WHERE id=?", (new_vid, cid))
        conn.execute(
            "UPDATE vehicle_instances SET vehicle_id=? WHERE capture_id=?",
            (new_vid, cid),
        )
        manual_applied += 1
    if manual_applied:
        conn.commit()

    # ---- 9) 重放人工合并决定（按抓拍锚定，不受车辆编号重排影响）----
    # 每条记录的含义：capture_id 应与 anchor_capture_id 同属一辆车。
    # 重聚类后若两者被拆开，把 capture_id 当前所在组整组并回锚点所在组。
    # 园区校验：锚点与抓拍必须在同一园区（写入时已保证，重放时二次校验）。
    conn.execute(
        """CREATE TABLE IF NOT EXISTS manual_vehicle_merge_pairs (
             capture_id INTEGER PRIMARY KEY,
             anchor_capture_id INTEGER NOT NULL,
             park TEXT NOT NULL DEFAULT '',
             created_at TEXT NOT NULL)"""
    )
    pairs = conn.execute(
        "SELECT capture_id, anchor_capture_id FROM manual_vehicle_merge_pairs "
        "ORDER BY capture_id").fetchall()
    merge_pairs_applied = 0
    for cid, aid in pairs:
        cap_row = conn.execute(
            """SELECT vehicle_id, coalesce(park,'') FROM captures WHERE id=?""",
            (cid,)).fetchone()
        anchor_row = conn.execute(
            """SELECT vehicle_id, coalesce(park,'') FROM captures WHERE id=?""",
            (aid,)).fetchone()
        if not cap_row or not anchor_row:
            continue  # 抓拍已被删除
        if cap_row[0] is None or anchor_row[0] is None:
            continue  # 无车帧/人脸不参与车辆分组
        if cap_row[0] == anchor_row[0]:
            continue  # 已在同一组（幂等）
        if cap_row[1] != anchor_row[1]:
            print(f"警告：人工合并对 capture {cid}/{aid} 分属不同园区，跳过重放")
            continue
        conn.execute("UPDATE captures SET vehicle_id=? WHERE vehicle_id=?",
                     (anchor_row[0], cap_row[0]))
        conn.execute(
            """UPDATE vehicle_instances SET vehicle_id=?
               WHERE capture_id IN (SELECT id FROM captures WHERE vehicle_id=?)""",
            (anchor_row[0], cap_row[0]))
        merge_pairs_applied += 1
    if merge_pairs_applied:
        conn.commit()
        print(f"人工合并重放：{merge_pairs_applied} 对")

    # ---- 9b) 重放 ReID 配对标注约束（pair_annotations）----
    # 同一工作台里标注的同车对（same）实时并入同一车辆组（同园区且不与异车对冲突）；
    # 异车对（different）若仍同属一组，将后一张拆出独立编号。
    pair_same_applied = 0
    pair_diff_split = 0
    for lid, rid in pair_same_list:
        lrow = conn.execute(
            "SELECT vehicle_id, coalesce(park,'') FROM captures WHERE id=?",
            (lid,)).fetchone()
        rrow = conn.execute(
            "SELECT vehicle_id, coalesce(park,'') FROM captures WHERE id=?",
            (rid,)).fetchone()
        if not lrow or not rrow:
            continue
        if lrow[0] is None or rrow[0] is None or lrow[0] == rrow[0]:
            continue
        if lrow[1] != rrow[1]:
            continue
        # 检查两组间是否存在显式异车约束，避免矛盾合并
        l_members = {r[0] for r in conn.execute(
            "SELECT id FROM captures WHERE vehicle_id=?", (lrow[0],))}
        r_members = {r[0] for r in conn.execute(
            "SELECT id FROM captures WHERE vehicle_id=?", (rrow[0],))}
        if any((m in pair_diff_map and (pair_diff_map[m] & r_members)) for m in l_members):
            continue
        keep_vid, drop_vid = sorted((int(lrow[0]), int(rrow[0])))
        conn.execute(
            "UPDATE captures SET vehicle_id=?, plate_match_reason='人工配对同车' WHERE vehicle_id=?",
            (keep_vid, drop_vid))
        conn.execute(
            """UPDATE vehicle_instances SET vehicle_id=?
               WHERE capture_id IN (SELECT id FROM captures WHERE vehicle_id=?)""",
            (keep_vid, keep_vid))
        pair_same_applied += 1
    if pair_diff_map:
        max_vid = int(conn.execute("SELECT coalesce(MAX(vehicle_id),0) FROM captures").fetchone()[0])
        seen_diff = set()
        for lid, rset in pair_diff_map.items():
            for rid in rset:
                edge = (min(lid, rid), max(lid, rid))
                if edge in seen_diff:
                    continue
                seen_diff.add(edge)
                lrow = conn.execute("SELECT vehicle_id FROM captures WHERE id=?", (edge[0],)).fetchone()
                rrow = conn.execute("SELECT vehicle_id FROM captures WHERE id=?", (edge[1],)).fetchone()
                if lrow and rrow and lrow[0] is not None and lrow[0] == rrow[0]:
                    max_vid += 1
                    conn.execute(
                        "UPDATE captures SET vehicle_id=?, plate_match_reason='人工标记不同车已拆出' WHERE id=?",
                        (max_vid, edge[1]))
                    conn.execute(
                        "UPDATE vehicle_instances SET vehicle_id=? WHERE capture_id=? AND instance_index=0",
                        (max_vid, edge[1]))
                    pair_diff_split += 1
    if pair_same_applied or pair_diff_split:
        conn.commit()

    linked = conn.execute(
        "SELECT COUNT(*) FROM captures WHERE vehicle_id IS NOT NULL"
    ).fetchone()[0]
    return {"orders": len(orders), "vehicle_groups": len(roots),
            "exact_plate_merges": exact_merges, "near_plate_merges": near_merges,
            "far_consensus_split": split, "frame_misread_cleaned": cleaned,
            "frame_misread_restored": restored,
            "plate_split": plate_split,
            "type_split": type_split, "manual_moves_applied": manual_applied,
            "pair_same_applied": pair_same_applied,
            "pair_diff_split": pair_diff_split,
            "linked_captures": linked}


def ensure_loose_assignments_schema(conn: sqlite3.Connection) -> None:
    """未关联货车归类日志：每行是一次"未关联图片 -> 订单"的文件移动决定。

    记录原路径/原日期等完整现场，支持 --undo-loose-assign 逐张或全部撤销；
    capture_id 为主键（同一张图只会被归类一次，撤销后可再次归类）。
    """
    conn.execute(
        """CREATE TABLE IF NOT EXISTS loose_assignments (
             capture_id INTEGER PRIMARY KEY,
             park TEXT NOT NULL DEFAULT '',
             from_order TEXT NOT NULL DEFAULT '',
             to_order TEXT NOT NULL DEFAULT '',
             date_old TEXT, date_new TEXT,
             filepath_old TEXT, filepath_new TEXT,
             filename_old TEXT, filename_new TEXT,
             similarity REAL, time_diff_seconds INTEGER,
             plate_note TEXT NOT NULL DEFAULT '',
             reason TEXT NOT NULL DEFAULT '',
             moved_at TEXT NOT NULL
           )"""
    )


def ensure_gate_dedup_log_schema(conn: sqlite3.Connection) -> None:
    """共用大门去重日志：每行是一次"重复照片 -> 隔离园区"的落败决定。

    多园区共用同一大门时，上游会把同一张抓拍照片写进多个园区目录；判重
    后赢家留在正确订单、落败副本移入 重复抓拍隔离-<原园区> 园区。记录
    完整现场支持 --undo-dedup 逐张或全部撤销；capture_id 为主键（同一张
    图只判一次落败，撤销后可再次参与判重）。
    """
    conn.execute(
        """CREATE TABLE IF NOT EXISTS gate_dedup_log (
             capture_id INTEGER PRIMARY KEY,
             dup_key TEXT NOT NULL DEFAULT '',
             confirm TEXT NOT NULL DEFAULT '',
             from_park TEXT NOT NULL DEFAULT '',
             from_order TEXT NOT NULL DEFAULT '',
             from_path TEXT NOT NULL DEFAULT '',
             filename_old TEXT NOT NULL DEFAULT '',
             to_park TEXT NOT NULL DEFAULT '',
             to_path TEXT NOT NULL DEFAULT '',
             filename_new TEXT NOT NULL DEFAULT '',
             winner_id INTEGER,
             winner_park TEXT NOT NULL DEFAULT '',
             winner_order TEXT NOT NULL DEFAULT '',
             reason TEXT NOT NULL DEFAULT '',
             deduped_at TEXT NOT NULL
           )"""
    )


def _refresh_order_attributes(conn: sqlite3.Connection,
                              keys: set[tuple[str, str]]) -> None:
    """局部刷新订单属性多数投票（与 reid_pipeline.aggregate_order_attributes
    同一套权重：颜色/车型按置信度求和取最大，避免 build_db 反向依赖流水线模块）。
    """
    now = datetime.now().isoformat(timespec="seconds")
    for park, order_id in keys:
        rows = conn.execute(
            """SELECT vehicle_color, vehicle_color_conf, vehicle_type, vehicle_type_conf
               FROM captures
               WHERE coalesce(park,'')=? AND order_id=?
                 AND coalesce(capture_status,'vehicle')!='no_vehicle'
                 AND (coalesce(vehicle_color,'')!='' OR coalesce(vehicle_type,'')!='')""",
            (park, order_id),
        ).fetchall()
        if not rows:
            continue
        colors: dict[str, float] = {}
        types: dict[str, float] = {}
        for color, color_conf, vtype, type_conf in rows:
            if color:
                colors[color] = colors.get(color, 0.0) + max(0.05, float(color_conf or 0))
            if vtype:
                types[vtype] = types.get(vtype, 0.0) + max(0.05, float(type_conf or 0))
        best_color = max(colors, key=lambda k: colors[k]) if colors else ""
        best_type = max(types, key=lambda k: types[k]) if types else ""
        conn.execute(
            """INSERT INTO order_attributes
                 (park, order_id, vehicle_color, vehicle_color_conf,
                  vehicle_type, vehicle_type_conf, sample_count, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(park, order_id) DO UPDATE SET
                 vehicle_color=excluded.vehicle_color,
                 vehicle_color_conf=excluded.vehicle_color_conf,
                 vehicle_type=excluded.vehicle_type,
                 vehicle_type_conf=excluded.vehicle_type_conf,
                 sample_count=excluded.sample_count,
                 updated_at=excluded.updated_at""",
            (park, order_id, best_color,
             round(colors[best_color] / len(rows), 4) if best_color else 0.0,
             best_type,
             round(types[best_type] / len(rows), 4) if best_type else 0.0,
             len(rows), now),
        )


def _match_loose_capture(loose: dict, orders: dict, window_seconds: int,
                         sim_threshold: float, sim_margin: float) -> tuple:
    """单张未关联抓拍的订单匹配。返回 (命中订单key或None, 证据dict, 跳过原因)。

    证据分两级：
    - 车牌锚定（最强）：抓拍有明确车牌（plate 或防串单 guard_plate）时，只允许
      匹配共识车牌去省份差 <=1 的订单，归入时间差最小者（同一辆车多次进场时，
      时间窗负责挑出"这一次"）；绝不允许带牌过路车退化到无牌外观通道塞进
      无共识/无车牌订单（黑洞订单防线）；
    - 外观+时间（仅抓拍确实无车牌时）：与订单主车特征最大余弦 >= 阈值，且领先
      次优订单 >= margin 时外观定胜负；领先不足（园区同时在场货车常达
      7-10 单，同型同色货车外观本来就难分）按时间差就近归属——照片夹在
      两个订单之间时选时间更近的一单；两单时间差彼此 <300 秒仍是真歧义，
      保持待人工。车型与订单主体车型矛盾者先被否决。
    标定（granary.db 隐藏车牌模拟，86 个可评估样本）：阈值0.55+领先0.03
    精度 92.9%、召回 75.6%；margin 原用于拦误挂搁置歧义样本，现歧义样本
    交给时间裁决（2026-09 现场要求：宁可按时间就近归属，不留成未关联）。
    """
    evidence = {"similarity": 0.0, "time_diff_seconds": -1, "plate_note": ""}
    candidates = []
    loose_dt = loose["dt"]
    has_any_plate = bool(
        (loose.get("plate") or "") or (loose.get("lpr_plate") or "")
        or (loose.get("osd_plate") or "") or (loose.get("guard_plate") or "")
    )
    # 车牌锚定与外观通道均以 window_seconds（默认 2 小时）为界，避免同车一日多趟
    # （相隔 3~5 小时）跨趟误挂到另一笔订单。
    eff_window = window_seconds
    cam = loose.get("osd_camera") or ""
    loose_dir = ("in" if "进" in cam else ("out" if "出" in cam else ("unload" if "仓库" in cam else "")))
    loose_emb = loose.get("emb")

    for key, order in orders.items():
        if order["park"] != loose["park"] or not order["times"]:
            continue
        anchor_times = order.get("anchor_times") or order["times"]
        time_diff = min(abs((loose_dt - t).total_seconds()) for t in anchor_times)
        if time_diff > eff_window:
            continue
        plate_distance = -1.0
        plate_source = ""
        target_plates = order.get("strong_plates") or (
            [order["consensus_plate"]] if order.get("consensus_plate") else []
        )
        if has_any_plate:
            best_pm = None
            for tp in target_plates:
                status, dist, src_note = _loose_plate_match(loose, tp)
                if status == "match":
                    if best_pm is None or dist < best_pm[0]:
                        best_pm = (dist, src_note)
            if best_pm is None:
                continue  # 车牌冲突或目标订单无强共识车牌：否决（防黑洞订单与异牌混入）
            plate_distance, plate_source = best_pm
        else:
            # 仅无车牌外观通道才用视觉车型否决；车牌已精确匹配时不因大门近景把货车误判
            # 为 suv/sedan 而错杀实锤大门图。
            if loose.get("vehicle_type") and order.get("main_type") and \
                    loose["vehicle_type"] != order["main_type"]:
                continue
        sim = (max((cosine(loose_emb, emb) for emb in order["embs"]), default=-1.0)
               if loose_emb is not None else 0.0)
        # 方向时序惩罚：大门进优先归入磅前，大门出优先归出磅后，仓库抓拍优先归入磅~出磅间
        dir_penalty = 0
        t_start = order.get("first_weigh_in") or min(order["times"])
        t_end = order.get("last_weigh_out") or max(order["times"])
        if loose_dir == "in":
            if loose_dt > t_end + timedelta(minutes=5):
                dir_penalty = 10800
            elif loose_dt > t_start + timedelta(minutes=15):
                dir_penalty = 1800
        elif loose_dir == "out":
            if loose_dt < t_start - timedelta(minutes=10):
                dir_penalty = 10800
            elif loose_dt < t_end - timedelta(minutes=10):
                dir_penalty = 1200
        elif loose_dir == "unload":
            if not (t_start - timedelta(minutes=15) <= loose_dt <= t_end + timedelta(minutes=15)):
                dir_penalty = 7200
        candidates.append({"key": key, "sim": sim, "time_diff": time_diff,
                           "dir_penalty": dir_penalty,
                           "plate_distance": plate_distance,
                           "plate_source": plate_source})
    if not candidates:
        return (None, evidence,
                "抓拍有车牌，时间窗内无匹配车牌的订单" if has_any_plate
                else "时间窗内无可兼容订单")

    plate_matched = [c for c in candidates if c["plate_distance"] >= 0]
    if plate_matched:
        plate_matched.sort(key=lambda c: (c["dir_penalty"], c["time_diff"], c["plate_distance"], -c["sim"]))
        best = plate_matched[0]
        # 方向与订单时段硬逆序（如出磅后进场、或入磅前出场）：说明属于另一趟行程，拒绝强行误挂。
        if best["dir_penalty"] >= 10800:
            return None, evidence, "方向与订单时段矛盾，待人工判断"
        # 两笔车牌一致且方向一致的订单时间差彼此接近（<300秒）说明无法确定是哪一次
        # 进场（快速折返），不强行归类
        if (len(plate_matched) > 1
                and plate_matched[1]["dir_penalty"] == best["dir_penalty"]
                and plate_matched[1]["time_diff"] - best["time_diff"] < 300):
            return None, evidence, "多笔车牌一致订单时间接近，待人工判断"
        d_str = f"{best['plate_distance']:g}"
        dir_tag = f"·{cam}" if cam else ""
        note = (f"车牌去省份差{d_str}（{best['plate_source']}{dir_tag}）"
                if best["plate_source"] else f"车牌去省份差{d_str}{dir_tag}")
        evidence.update(similarity=best["sim"],
                        time_diff_seconds=int(best["time_diff"]),
                        plate_note=note)
        return best["key"], evidence, ""

    # 无车牌锚定：外观相似度阈值 + 对次优的领先幅度。外观分不出胜负时
    # （领先不足——同型同色货车常态）按时间差就近裁决：照片夹在两个订单
    # 之间时自动归时间更近的一单，不再搁置；两单时间差彼此 <300 秒仍是
    # 真歧义（无法分辨是哪一次），保持待人工判断，避免硬币翻面。
    candidates.sort(key=lambda c: (-c["sim"], c["time_diff"]))
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    lead = best["sim"] - second["sim"] if second else 1.0
    if best["sim"] < sim_threshold:
        return None, evidence, "最高相似度低于阈值"
    if lead >= sim_margin:
        evidence.update(similarity=best["sim"],
                        time_diff_seconds=int(best["time_diff"]),
                        plate_note="无车牌，外观+时间匹配")
        return best["key"], evidence, ""
    by_time = sorted(candidates, key=lambda c: (c["time_diff"], -c["sim"]))
    nearest = by_time[0]
    runner = by_time[1] if len(by_time) > 1 else None
    if runner and runner["time_diff"] - nearest["time_diff"] < 300:
        return None, evidence, "候选订单时间差接近，待人工判断"
    evidence.update(similarity=nearest["sim"],
                    time_diff_seconds=int(nearest["time_diff"]),
                    plate_note="外观难分，按时间就近归属")
    return nearest["key"], evidence, ""


def assign_loose_orders(conn: sqlite3.Connection, apply_changes: bool = False,
                        types: tuple[str, ...] = ("truck",),
                        window_seconds: int = 7200,
                        sim_threshold: float = 0.55,
                        sim_margin: float = 0.03,
                        park_filter: str = "") -> dict:
    """把"未关联图片"里的抓拍归类到时间相近且同一辆车的订单。

    背景：NVR 靠自己的车牌识别做图片归档，读不到牌的抓拍全部落进
    "未关联图片"文件夹（granary.db 中 985 张，其中货车 606 张）。本函数
    用三类证据恢复归属：OCR时间差（是否同一次在场）、ReID 外观相似度
    （是否同一辆车）、车牌（读到则一票定音/否决）。

    归类 = 物理移动文件到订单文件夹 + 更新 captures.order_id（订单归属
    的唯一事实来源，与目录结构保持一致，sync_files 的聚合键才不会把图
    重新扫回"未关联图片"），并写 loose_assignments 日志支持撤销。
    默认只试运行打印报告，apply_changes=True 才真正移动/写库。

    车辆分组（vehicle_id）不在此重算：归类后重跑 associate_orders 即可
    把这些抓拍随订单并入车簇（它们此刻已是订单成员）。
    """
    ensure_loose_assignments_schema(conn)
    type_list = tuple(t for t in types if t)
    # 两类未关联抓拍可参与归单：
    # 1) 常规无牌货车（reid_embedding 非空 + 检出车辆 + vehicle_type 在 type_list 内）；
    # 2) 明确带有车牌（plate_final / osd_plate / lpr_plate）的抓拍——即便大门近景被视觉
    #    属性误判为 suv/sedan、或检测器漏检标了 no_vehicle（但黑底 OSD 已写明车牌），
    #    同样进入车牌精确锚定通道（只允许命中共识车牌一致的订单，社会过路车不会误入）。
    loose_rows = conn.execute(
        """SELECT id, coalesce(park,''), date, capture_time, plate_final,
                  reid_embedding, vehicle_type, filepath, filename,
                  lpr_plate, lpr_conf, osd_plate, coalesce(osd_camera,''),
                  coalesce(capture_status,'vehicle')
           FROM captures
           WHERE order_id='未关联图片'
             AND capture_time IS NOT NULL AND capture_time!=''
             AND coalesce(park,'') NOT LIKE ?
             AND (
               (reid_embedding IS NOT NULL
                AND coalesce(capture_status,'vehicle')!='no_vehicle'
                AND vehicle_type IS NOT NULL)
               OR coalesce(plate_final,'')!=''
               OR coalesce(osd_plate,'')!=''
               OR coalesce(lpr_plate,'')!=''
             )""",
        (QUARANTINE_PARK_LIKE,),
    ).fetchall()
    if park_filter:
        loose_rows = [row for row in loose_rows if row[1] == park_filter]
    filtered_loose = []
    for row in loose_rows:
        has_plate = bool((row[4] or "") or (row[9] or "") or (row[11] or ""))
        if has_plate or (not type_list or row[6] in type_list):
            filtered_loose.append(row)
    loose_rows = filtered_loose

    # 订单索引：只收有可靠时间的车辆帧（同时记录最早入磅/最晚出磅，供方向时序裁决）
    orders: dict[tuple[str, str], dict] = {}
    for (park, order_id, capture_time, stage, plate, blob, vtype, filepath,
         lpr_plate, lpr_conf, osd_plate) in conn.execute(
        """SELECT coalesce(park,''), order_id, capture_time, coalesce(stage,''),
                  plate_final, reid_embedding, vehicle_type, filepath,
                  lpr_plate, lpr_conf, osd_plate
           FROM captures
           WHERE coalesce(order_id,'')!='' AND order_id!='未关联图片'
             AND capture_time IS NOT NULL AND capture_time!=''
             AND coalesce(capture_status,'vehicle')!='no_vehicle'"""
    ).fetchall():
        order = orders.setdefault(
            (park, order_id),
            {"park": park, "times": [], "weigh_in_times": [], "weigh_out_times": [],
             "plates": Counter(), "embs": [],
             "types": Counter(), "dirs": {}, "nearest_anchor": {}})
        dt_val = datetime.strptime(capture_time, "%Y-%m-%d %H:%M:%S")
        order["times"].append(dt_val)
        if stage == "入磅":
            order["weigh_in_times"].append(dt_val)
        elif stage == "出磅":
            order["weigh_out_times"].append(dt_val)
        eff_plate = _guard_plate(plate, lpr_plate, lpr_conf, osd_plate)
        if eff_plate:
            order["plates"][eff_plate] += 1
        if blob:
            feature = np.frombuffer(blob, np.float32)
            norm = np.linalg.norm(feature)
            if norm > 0 and len(order["embs"]) < 12:
                order["embs"].append(feature / norm)
        if vtype:
            order["types"][vtype] += 1
        if filepath:
            parent = str(Path(filepath).parent)
            order["dirs"].setdefault(parent, capture_time[:10])
            order["nearest_anchor"][parent] = capture_time
    for order in orders.values():
        order["main_type"] = order["types"].most_common(1)[0][0] if order["types"] else ""
        order["first_weigh_in"] = min(order["weigh_in_times"]) if order["weigh_in_times"] else (min(order["times"]) if order["times"] else None)
        order["last_weigh_out"] = max(order["weigh_out_times"]) if order["weigh_out_times"] else (max(order["times"]) if order["times"] else None)
        # 时间锚优先使用订单权威过磅时刻（入磅/出磅），避免订单内已有误挂抓拍拉偏时间窗
        order["anchor_times"] = (
            order["weigh_in_times"] + order["weigh_out_times"]
            if (order["weigh_in_times"] or order["weigh_out_times"])
            else order["times"]
        )
        # 共识车牌：去省份+距离<=1 归簇计票，避免因省份/单字误读落成无共识
        order["consensus_plate"], order["known_plates"] = _order_plate_profile(
            order["plates"])
        if order["consensus_plate"]:
            order["strong_plates"] = [order["consensus_plate"]]
        else:
            # 磅旁停放他车导致 2 vs 2 平票时（如 皖K5713A×2 vs 皖KP8555×2），
            # 凡在订单内达到 >=2 票的车牌簇均允许参与精准车牌锚定（单帧噪点仍被屏蔽）。
            clusters: list[list[str]] = []
            for p in sorted(order["plates"], key=lambda x: (-order["plates"][x], -len(x), x)):
                placed = False
                for cl in clusters:
                    if any(_plate_edit_distance(p, m) <= 1.0 for m in cl):
                        cl.append(p)
                        placed = True
                        break
                if not placed:
                    clusters.append([p])
            order["strong_plates"] = [
                cl[0] for cl in clusters
                if sum(order["plates"][m] for m in cl) >= 2
            ]

    plan = []
    skips: dict[str, int] = {}
    for row in loose_rows:
        (capture_id, park, _date, capture_time, plate, blob,
         vehicle_type, filepath, filename,
         lpr_plate, lpr_conf, osd_plate, osd_camera, cap_status) = row
        has_plate = bool((plate or "") or (lpr_plate or "") or (osd_plate or ""))
        emb_vec = None
        if blob:
            feature = np.frombuffer(blob, np.float32)
            norm = np.linalg.norm(feature)
            if norm > 0:
                emb_vec = feature / norm
        if emb_vec is None and not has_plate:
            skips["无效特征"] = skips.get("无效特征", 0) + 1
            continue
        if not resolve_stored_path(filepath).is_file():
            # 原始照片已从盘上清理的园区（服务器只留库记录与裁剪图）：物理
            # 移动无从做起，跳过而不是报 558 个移动失败刷屏。
            skips["原始照片不在盘上（归档已清理）"] = \
                skips.get("原始照片不在盘上（归档已清理）", 0) + 1
            continue
        loose = {"id": capture_id, "park": park, "plate": plate or "",
                 "lpr_plate": lpr_plate or "", "lpr_conf": float(lpr_conf or 0.0),
                 "osd_plate": osd_plate or "", "osd_camera": osd_camera or "",
                 "capture_status": cap_status,
                 "guard_plate": _guard_plate(plate, lpr_plate, lpr_conf, osd_plate),
                 "vehicle_type": vehicle_type, "filepath": filepath,
                 "filename": filename, "date": _date,
                 "dt": datetime.strptime(capture_time, "%Y-%m-%d %H:%M:%S"),
                 "emb": emb_vec}
        key, evidence, skip_reason = _match_loose_capture(
            loose, orders, window_seconds, sim_threshold, sim_margin)
        if key is None:
            skips[skip_reason] = skips.get(skip_reason, 0) + 1
            continue
        plan.append((loose, key, evidence))

    moved = failed = 0
    touched_orders: set[tuple[str, str]] = set()
    now = datetime.now().isoformat(timespec="seconds")
    if apply_changes:
        conn.isolation_level = None
    for loose, key, evidence in plan:
        order = orders[key]
        target_dir, target_date = _pick_order_dir(order, loose, key)
        target_stored = f"{target_dir}/{loose['filename']}"
        target_abs = project_file(target_stored)
        final_name = loose["filename"]
        if target_abs.exists():
            stem, suffix = target_abs.stem, target_abs.suffix
            k = 1
            while target_abs.with_name(f"{stem}_reid{k}{suffix}").exists():
                k += 1
            final_name = f"{stem}_reid{k}{suffix}"
            target_abs = target_abs.with_name(final_name)
        reason = (f"未关联货车归类：车牌一致({evidence['plate_note']})，"
                  f"时间差{evidence['time_diff_seconds'] // 60}分钟"
                  if evidence["plate_note"].startswith("车牌")
                  else f"未关联货车归类：时间差{evidence['time_diff_seconds'] // 60}分钟，"
                       f"外观相似度{evidence['similarity']:.2f}（{evidence['plate_note']}）")
        print(f"  {'归类' if apply_changes else '试运行'} #{loose['id']} "
              f"{loose['date']} {loose['dt'].strftime('%H:%M:%S')} -> 订单{key[1]} "
              f"({reason.split('：')[1]})")
        if not apply_changes:
            continue
        try:
            target_abs.parent.mkdir(parents=True, exist_ok=True)
            source_abs = project_file(loose["filepath"])
            if not source_abs.is_file():
                source_abs = resolve_stored_path(loose["filepath"])
            shutil.move(str(source_abs), str(target_abs))
        except OSError as exc:
            failed += 1
            print(f"  移动失败 #{loose['id']}: {exc}")
            continue
        # 归入订单当前主车组：取订单内出现最多的 vehicle_id（拆分遗留的
        # NULL 帧不参与投票）；订单尚无分组则留空，等 associate_orders 重算。
        vehicle_id = conn.execute(
            """SELECT vehicle_id, COUNT(*) AS cnt FROM captures
               WHERE coalesce(park,'')=? AND order_id=? AND vehicle_id IS NOT NULL
               GROUP BY vehicle_id ORDER BY cnt DESC LIMIT 1""",
            (key[0], key[1]),
        ).fetchone()
        conn.execute(
            """UPDATE captures SET order_id=?, date=?, filepath=?, filename=?,
                      plate_match_reason=?, vehicle_id=?
               WHERE id=?""",
            (key[1], target_date, to_stored_path(target_abs), final_name,
             reason, vehicle_id[0] if vehicle_id else None, loose["id"]),
        )
        conn.execute(
            """INSERT OR REPLACE INTO loose_assignments
                 (capture_id, park, from_order, to_order, date_old, date_new,
                  filepath_old, filepath_new, filename_old, filename_new,
                  similarity, time_diff_seconds, plate_note, reason, moved_at)
               VALUES (?, ?, '未关联图片', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (loose["id"], key[0], key[1], loose["date"], target_date,
             loose["filepath"], to_stored_path(target_abs), loose["filename"],
             final_name, evidence["similarity"], evidence["time_diff_seconds"],
             evidence["plate_note"], reason, now),
        )
        moved += 1
        touched_orders.add(key)
    if apply_changes:
        conn.commit()
        if touched_orders:
            _refresh_order_attributes(conn, touched_orders)
            conn.commit()

    result = {
        "loose_trucks": len(loose_rows), "planned": len(plan),
        "moved": moved, "failed": failed,
        "plate_anchored": sum(1 for _, _, e in plan if e["plate_note"].startswith("车牌")),
        "sim_matched": sum(1 for _, _, e in plan if not e["plate_note"].startswith("车牌")),
        "skips": skips,
    }
    mode = "已归类" if apply_changes else "试运行（加 --apply 落盘）"
    print(f"未关联货车归类{mode}：货车{result['loose_trucks']}张，"
          f"命中{result['planned']}张（车牌锚定{result['plate_anchored']}，"
          f"外观+时间{result['sim_matched']}），"
          f"移动失败{result['failed']}张")
    for reason_text, count in sorted(skips.items(), key=lambda kv: -kv[1]):
        print(f"  跳过 {count}张：{reason_text}")
    return result


def assign_gate_orders(conn: sqlite3.Connection, apply_changes: bool = False,
                       types: tuple[str, ...] = (),
                       window_seconds: int = 7200,
                       sim_threshold: float = 0.55,
                       sim_margin: float = 0.03,
                       park_filter: str = "",
                       max_rounds: int = 3) -> dict:
    """大门抓拍（入磅/出磅）按时间重新归属订单（默认试运行，--apply 落盘）。

    背景：大门照片人工分不到——上游 NVR 靠自身车牌识别把读得到牌的照片
    归档进订单文件夹，两个订单时间靠近时容易放错订单。本步骤用与
    assign_loose_orders 相同的证据链重新评估每张大门照片：

    - 订单时间锚包含已归属的大门照片（入磅提前窗口、出磅推后窗口），而
      不只看人工分好的抓拍/质检照片——"把进出大门的算上"；
    - 车牌锚定最强（去省份差 <=1，时间差最小者胜，两单 <300 秒真歧义
      保留现归属）；现属订单任一成员车牌与照片车牌一致时视为锚定在
      现属订单，外观证据不能推翻（防同型货车外观误拉走按牌归档的照片）；
      外观决定胜负需领先 >= margin；外观难分时按时间差
      就近归属（照片夹在两个订单之间时选更近的一单）；最高相似度低于
      阈值（外观与所有候选都矛盾）保留现归属不硬分；
    - 评估某张照片时把它自己从现属订单的锚里剔除（否则它永远离自己
      0 分钟，永远不会被纠正）；
    - 命中订单与当前归属不同才动：原图在盘 -> 物理移动文件进目标订单
      文件夹；原图不在盘（服务器清理过照片的园区）-> 仅库内改 order_id
      （路径保持原样便于审计）。两种情况都写 loose_assignments 日志
      （from_order=原订单，--undo-loose-assign 可撤销），随后建议重跑
      associate_orders 重聚车辆；
    - apply 模式迭代到稳定（上限 max_rounds 轮，前一轮的移动会扩展订单
      时间窗影响下一轮裁决），且单次运行每张照片至多移动一次；
    - 收敛保证：已有 loose_assignments 日志的照片不再参评——每张大门
      照片一生只自动重归类一次，日常流水线增量处理新照片，重复运行
      不会来回翻动。需要重评某张时先 --undo-loose-assign 撤销其记录。
    """
    ensure_loose_assignments_schema(conn)

    # ---- 订单索引：members 保留每张照片（时间/阶段/车牌/特征/车型），评估时
    # 可剔除被评照片自身后重算时间锚与共识车牌 ----
    members: dict[tuple[str, str], list[dict]] = {}
    for (cid, park, order_id, capture_time, stage, plate, blob, vtype,
         lpr_plate, lpr_conf, osd_plate) in conn.execute(
        """SELECT id, coalesce(park,''), order_id, capture_time, coalesce(stage,''),
                  plate_final, reid_embedding, vehicle_type,
                  lpr_plate, lpr_conf, osd_plate
           FROM captures
           WHERE coalesce(order_id,'')!='' AND order_id!='未关联图片'
             AND capture_time IS NOT NULL AND capture_time!=''
             AND coalesce(capture_status,'vehicle')!='no_vehicle'"""
    ).fetchall():
        emb = None
        if blob:
            f = np.frombuffer(blob, np.float32)
            norm = float(np.linalg.norm(f))
            emb = f / norm if norm > 0 else None
        gplate = _guard_plate(plate, lpr_plate, lpr_conf, osd_plate)
        members.setdefault((park, order_id), []).append(
            {"cid": cid, "dt": datetime.strptime(capture_time, "%Y-%m-%d %H:%M:%S"),
             "stage": stage, "plate": plate or "", "guard_plate": gplate,
             "emb": emb, "vtype": vtype or ""})
    # 目标文件夹参照：订单已有照片所在目录 -> 日期（_pick_order_dir 用）
    dirs: dict[tuple[str, str], dict[str, str]] = {}
    nearest_anchor: dict[tuple[str, str], dict[str, str]] = {}
    for park, order_id, filepath, capture_time in conn.execute(
        """SELECT coalesce(park,''), order_id, filepath, coalesce(capture_time,'')
           FROM captures
           WHERE coalesce(order_id,'')!='' AND order_id!='未关联图片'
             AND filepath IS NOT NULL AND filepath!=''"""
    ):
        parent = str(Path(filepath).parent)
        dirs.setdefault((park, order_id), {}).setdefault(parent, (capture_time or "")[:10])
        # 只收完整时间戳（部分老记录 capture_time 只有日期，strptime 会炸）
        if capture_time and len(capture_time) == 19:
            nearest_anchor.setdefault((park, order_id), {})[parent] = capture_time

    def order_anchors(key: tuple[str, str], exclude_cid: int | None = None):
        rows = members.get(key, [])
        if exclude_cid is not None:
            rows = [m for m in rows if m["cid"] != exclude_cid]
        plates = Counter(m["guard_plate"] for m in rows if m["guard_plate"])
        consensus, _known = _order_plate_profile(plates)
        types_counter = Counter(m["vtype"] for m in rows if m["vtype"])
        main_type = types_counter.most_common(1)[0][0] if types_counter else ""
        times = [m["dt"] for m in rows]
        embs = [m["emb"] for m in rows if m["emb"] is not None][:12]
        return times, embs, consensus, main_type

    # 候选订单时间锚缓存：只有被评照片的现属订单需要剔除自身重算，其余订单
    # 的锚在本轮内不变（apply 模式每轮移动后失效重建）。
    anchor_cache: dict[tuple[str, str], tuple] = {}

    def evaluate(row, cur_order: str):
        """评估一张大门照片：返回 (命中订单key或None=保留现归属, 证据, 原因)"""
        (cid, park, _cur, _date, capture_time, stage, plate, blob,
         vtype, _filepath, _filename, lpr_plate, lpr_conf, osd_plate) = row
        f = np.frombuffer(blob, np.float32)
        norm = float(np.linalg.norm(f))
        if norm <= 0:
            return None, {"similarity": 0.0, "time_diff_seconds": -1,
                          "plate_note": ""}, "无效特征"
        emb = f / norm
        dt = datetime.strptime(capture_time, "%Y-%m-%d %H:%M:%S")
        evidence = {"similarity": 0.0, "time_diff_seconds": -1, "plate_note": ""}
        cur_key = (park, cur_order)
        plate = plate or ""
        guard_plate = _guard_plate(plate, lpr_plate, lpr_conf, osd_plate)
        cur_anchors = order_anchors(cur_key, exclude_cid=cid)
        cur_consensus = cur_anchors[2]

        # 同秒多机位过磅连拍保护：地磅系统单次过磅同时触发多个机位（间隔 <=5s），
        # 绝不能把同秒连拍中未拍到车牌的侧面/车尾镜头靠 ReID 外观单独扯去别的订单，
        # 也不能把与现订单无车牌冲突的同秒连拍扯回 1 小时前的旧订单。
        burst_siblings = [
            m for m in members.get(cur_key, [])
            if m["cid"] != cid and m["stage"] == stage
            and abs((m["dt"] - dt).total_seconds()) <= 5
        ]
        burst_plates = {m["guard_plate"] for m in burst_siblings if m["guard_plate"]}
        if burst_siblings:
            if not guard_plate:
                return None, evidence, "同秒过磅连拍保留现归属"
            if not cur_consensus or _plate_edit_distance(guard_plate, cur_consensus) <= 1:
                return None, evidence, "同秒过磅连拍保留现归属"

        # 现属订单车牌保护：照片车牌与现属订单任一成员车牌去省份差 <=1，
        # 即视为车牌锚定在现属订单（上游 NVR 正是按牌归档的）——外观证据
        # 不能把照片拉走，除非另一订单也车牌锚定且时间明显更近。
        cur_plate_anchored = bool(guard_plate) and any(
            m["guard_plate"] and _plate_edit_distance(guard_plate, m["guard_plate"]) <= 1
            for m in members.get(cur_key, []) if m["cid"] != cid)
        candidates = []
        for key in members:
            if key[0] != park:
                continue
            if key == cur_key:
                anchors = cur_anchors
            else:
                if key not in anchor_cache:
                    anchor_cache[key] = order_anchors(key)
                anchors = anchor_cache[key]
            times, embs, consensus, main_type = anchors
            if not times:
                continue  # 现属订单剔自身后无锚：无从辩护，视为未防守
            time_diff = min(abs((dt - t).total_seconds()) for t in times)
            # 时间窗只用来发现新候选；现属订单始终参评（否则一张订单里
            # 最晚的大门照片会因剔自身后被判超窗而被外观随手移走）。
            if time_diff > window_seconds and key != cur_key:
                continue
            if vtype and main_type and vtype != main_type and key != cur_key:
                continue  # 车型矛盾：不可能是该订单主车（现属订单不因此逐出）
            if key != cur_key:
                # 行程时序 sanity check：出磅不得早于目标订单最早的入磅，
                # 入磅不得晚于目标订单最晚的出磅（避免相邻两单的过磅倒挂）。
                target_rows = members.get(key, [])
                if stage == "出磅":
                    in_times = [m["dt"] for m in target_rows if m["stage"] == "入磅"]
                    if in_times and dt < min(in_times):
                        continue
                elif stage == "入磅":
                    out_times = [m["dt"] for m in target_rows if m["stage"] == "出磅"]
                    if out_times and dt > max(out_times):
                        continue
            plate_distance = -1.0
            if guard_plate:
                if key != cur_key:
                    # 照片有明确车牌：绝不移入无共识车牌或车牌冲突的异单
                    if not consensus:
                        continue
                    guard_dist = _plate_edit_distance(guard_plate, consensus)
                    if guard_dist > 1.0:
                        continue
                    plate_distance = (_plate_edit_distance(plate, consensus)
                                      if plate else guard_dist)
                    if plate_distance > 1.0:
                        continue
                elif consensus:
                    plate_distance = _plate_edit_distance(guard_plate, consensus)
                    if plate_distance > 1.0:
                        plate_distance = -1.0
            elif key != cur_key and consensus:
                # 照片无车牌但现订单/同秒连拍已有明确车牌：绝不跨牌移入异牌订单
                if cur_consensus and _plate_edit_distance(cur_consensus, consensus) > 1.0:
                    continue
                if burst_plates and all(
                        _plate_edit_distance(bp, consensus) > 1.0 for bp in burst_plates):
                    continue
            sim = max((cosine(emb, e) for e in embs), default=-1.0)
            candidates.append({"key": key, "sim": sim, "time_diff": time_diff,
                               "plate_distance": plate_distance})
        if not candidates:
            return None, evidence, "时间窗内无兼容订单（保留现归属）"
        if os.environ.get("GATE_DEBUG_ID") == str(cid):
            for c in sorted(candidates, key=lambda c: c["time_diff"]):
                print(f"    [debug #{cid}] cur={cur_key[1]} 候选{c['key'][1]} "
                      f"sim={c['sim']:.3f} td={int(c['time_diff'])}s plate_d={c['plate_distance']}")

        def keep(winner_or_none, note):
            if winner_or_none is None or winner_or_none == cur_key:
                return None, evidence, f"保留现归属（{note}）"  # 证据不足以移动
            evidence["similarity"] = round(next(
                c["sim"] for c in candidates if c["key"] == winner_or_none), 4)
            evidence["time_diff_seconds"] = int(next(
                c["time_diff"] for c in candidates if c["key"] == winner_or_none))
            evidence["plate_note"] = note
            return winner_or_none, evidence, ""

        plate_matched = [c for c in candidates if c["plate_distance"] >= 0]
        if cur_plate_anchored:
            # 现属订单有车牌证据：只有另一订单也车牌锚定且时间明显更近才动
            cur_td = next((c["time_diff"] for c in candidates if c["key"] == cur_key),
                          None)
            better = [c for c in plate_matched
                      if c["key"] != cur_key
                      and (cur_td is None or c["time_diff"] < cur_td)]
            if not better:
                return None, evidence, "车牌锚定现归属（保留）"
            better.sort(key=lambda c: (c["time_diff"], -c["sim"]))
            best_other = better[0]
            if cur_td is not None and cur_td - best_other["time_diff"] < 300:
                return None, evidence, "两笔车牌一致订单时间接近，保留现归属"
            return keep(best_other["key"],
                        f"车牌去省份差{best_other['plate_distance']}（原订单同牌更远）")
        if plate_matched:
            plate_matched.sort(key=lambda c: (c["time_diff"], -c["sim"]))
            best = plate_matched[0]
            runner = plate_matched[1] if len(plate_matched) > 1 else None
            if runner and runner["time_diff"] - best["time_diff"] < 300:
                return None, evidence, "多笔车牌一致订单时间接近，保留现归属"
            return keep(best["key"], f"车牌去省份差{best['plate_distance']}")
        if guard_plate:
            return None, evidence, "照片有车牌但无匹配目标订单（保留现归属）"
        candidates.sort(key=lambda c: (-c["sim"], c["time_diff"]))
        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        lead = best["sim"] - second["sim"] if second else 1.0
        if best["sim"] < sim_threshold:
            return None, evidence, "最高相似度低于阈值（保留现归属）"
        if lead >= sim_margin:
            return keep(best["key"], "外观+时间匹配")
        by_time = sorted(candidates, key=lambda c: (c["time_diff"], -c["sim"]))
        nearest = by_time[0]
        runner = by_time[1] if len(by_time) > 1 else None
        if runner and runner["time_diff"] - nearest["time_diff"] < 300:
            return None, evidence, "候选订单时间差接近，保留现归属"
        return keep(nearest["key"], "外观难分，按时间就近归属")

    gate_rows = conn.execute(
        """SELECT id, coalesce(park,''), order_id, date, capture_time,
                  coalesce(stage,''), plate_final, reid_embedding,
                  vehicle_type, filepath, filename,
                  lpr_plate, lpr_conf, osd_plate
           FROM captures
           WHERE stage IN ('入磅','出磅')
             AND coalesce(capture_status,'vehicle')!='no_vehicle'
             AND capture_time IS NOT NULL AND capture_time!=''
             AND reid_embedding IS NOT NULL
             AND coalesce(order_id,'')!='' AND order_id!='未关联图片'
             AND coalesce(park,'') NOT LIKE ?
             AND id NOT IN (SELECT capture_id FROM loose_assignments)""",
        (QUARANTINE_PARK_LIKE,),
    ).fetchall()
    if park_filter:
        gate_rows = [r for r in gate_rows if r[1] == park_filter]
    if types:
        gate_rows = [r for r in gate_rows if r[8] in types or not r[8]]

    now = datetime.now().isoformat(timespec="seconds")
    moved = planned = failed = 0
    db_only_moves = 0
    skip_counter: dict[str, int] = {}
    keep_count = 0
    touched_orders: set[tuple[str, str]] = set()
    moved_cids: set[int] = set()
    # 现归属状态（apply 多轮迭代时 DB 行里的 order_id/filepath 会过期）
    state = {row[0]: {"order": row[2], "date": row[3], "filepath": row[9],
                      "filename": row[10]} for row in gate_rows}
    for round_no in range(1, (max_rounds if apply_changes else 1) + 1):
        plan = []
        keep_count = 0
        anchor_cache.clear()
        for row in gate_rows:
            if row[0] in moved_cids:
                continue  # 单次运行每张照片至多移动一次：防止耦合交换的
                # 照片因成员集变化在轮间反复横跳（不收敛）
            winner_key, evidence, reason = evaluate(row, state[row[0]]["order"])
            if winner_key is None:
                keep_count += 1
                if round_no == 1:
                    skip_counter[reason] = skip_counter.get(reason, 0) + 1
                continue
            plan.append((row, winner_key, evidence))
        if not plan:
            break
        if not apply_changes:
            planned = len(plan)
            for row, key, evidence in plan:
                print(f"  试运行 #{row[0]} {state[row[0]]['date']} {row[4][11:19]} "
                      f"-> 订单{key[1]}（{evidence['plate_note']}，"
                      f"时间差{evidence['time_diff_seconds'] // 60}分钟，"
                      f"现归属{state[row[0]]['order']}）")
            break
        # 应用本轮：移动文件 + 写库 + 日志；同步内存索引供下一轮评估
        conn.isolation_level = None
        for row, key, evidence in plan:
            cid, park = row[0], row[1]
            st = state[cid]
            cur_order, date = st["order"], st["date"]
            filepath, filename = st["filepath"], st["filename"]
            order_ctx = {"dirs": dirs.get(key, {}),
                         "nearest_anchor": nearest_anchor.get(key, {})}
            loose_ctx = {"date": date, "dt": datetime.strptime(row[4], "%Y-%m-%d %H:%M:%S"),
                         "filepath": filepath}
            target_dir, target_date = _pick_order_dir(order_ctx, loose_ctx, key)
            target_stored = f"{target_dir}/{filename}"
            target_abs = project_file(target_stored)
            final_name = filename
            if target_abs.exists():
                stem, suffix = target_abs.stem, target_abs.suffix
                k = 1
                while target_abs.with_name(f"{stem}_reid{k}{suffix}").exists():
                    k += 1
                final_name = f"{stem}_reid{k}{suffix}"
                target_abs = target_abs.with_name(final_name)
            minutes = evidence["time_diff_seconds"] // 60
            reason_text = (f"大门照片重归类：{evidence['plate_note']}，"
                           f"时间差{minutes}分钟（原归属{cur_order}）")
            print(f"  归类 #{cid} {date} {row[4][11:19]} {cur_order} -> 订单{key[1]} "
                  f"({evidence['plate_note']}，时间差{minutes}分钟)")
            # 原图在盘 -> 物理移动；不在盘（服务器清理过的园区）-> 仅库内
            # 改归属（order_id 是聚合唯一事实来源，路径保持原样便于审计）
            source_abs = project_file(filepath)
            if not source_abs.is_file():
                source_abs = resolve_stored_path(filepath)
            db_only = not source_abs.is_file()
            if not db_only:
                try:
                    target_abs.parent.mkdir(parents=True, exist_ok=True)
                    if source_abs.resolve() != target_abs.resolve():
                        shutil.move(str(source_abs), str(target_abs))
                except OSError as exc:
                    failed += 1
                    print(f"  移动失败 #{cid}: {exc}")
                    continue
                new_fp, new_fn, new_date = to_stored_path(target_abs), final_name, target_date
            else:
                db_only_moves += 1
                new_fp, new_fn, new_date = filepath, filename, date
                reason_text += "（原图不在盘，仅库内改归属）"
            vehicle_id = conn.execute(
                """SELECT vehicle_id, COUNT(*) AS cnt FROM captures
                   WHERE coalesce(park,'')=? AND order_id=? AND vehicle_id IS NOT NULL
                   GROUP BY vehicle_id ORDER BY cnt DESC LIMIT 1""",
                (key[0], key[1]),
            ).fetchone()
            conn.execute(
                """UPDATE captures SET order_id=?, date=?, filepath=?, filename=?,
                          plate_match_reason=?, vehicle_id=? WHERE id=?""",
                (key[1], new_date, new_fp, new_fn,
                 reason_text, vehicle_id[0] if vehicle_id else None, cid),
            )
            conn.execute(
                """INSERT OR REPLACE INTO loose_assignments
                     (capture_id, park, from_order, to_order, date_old, date_new,
                      filepath_old, filepath_new, filename_old, filename_new,
                      similarity, time_diff_seconds, plate_note, reason, moved_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (cid, key[0], cur_order, key[1], date, new_date,
                 filepath, new_fp, filename, new_fn,
                 evidence["similarity"], evidence["time_diff_seconds"],
                 evidence["plate_note"], reason_text, now),
            )
            # 内存索引同步：照片从原订单挪到新订单（时间锚/文件夹参照随更新）
            old_members = members.get((park, cur_order), [])
            member = next((m for m in old_members if m["cid"] == cid), None)
            if member:
                old_members.remove(member)
                members.setdefault(key, []).append(member)
            if not db_only:
                old_parent = str(Path(filepath).parent)
                # dirs 收日期即可；nearest_anchor 必须存完整时间戳
                # （_pick_order_dir 会 strptime），target_date 可能只有日期
                dirs.setdefault(key, {}).setdefault(str(target_abs.parent), target_date)
                nearest_anchor.setdefault(key, {}).setdefault(str(target_abs.parent), row[4])
                dirs.get((park, cur_order), {}).pop(old_parent, None)
                nearest_anchor.get((park, cur_order), {}).pop(old_parent, None)
                st.update(order=key[1], date=target_date,
                          filepath=to_stored_path(target_abs), filename=final_name)
            else:
                st.update(order=key[1])
            touched_orders.add(key)
            touched_orders.add((park, cur_order))
            moved_cids.add(cid)
            moved += 1
    if apply_changes:
        if touched_orders:
            # 来源/目标订单的颜色车型投票都随成员变化刷新
            _refresh_order_attributes(conn, touched_orders)
        conn.commit()

    result = {
        "gate_captures": len(gate_rows), "reassigned": moved, "planned": planned,
        "failed": failed, "kept": keep_count, "skips": skip_counter,
        "orders_touched": len(touched_orders), "db_only": db_only_moves,
    }
    mode = "已重归类" if apply_changes else "试运行（加 --apply 落盘）"
    move_stat = (f"移动{moved}张（其中仅库内改归属{db_only_moves}张）" if apply_changes
                 else f"计划移动{planned}张")
    print(f"大门照片重归属{mode}：入磅/出磅 {result['gate_captures']}张，"
          f"{move_stat}，失败{failed}张，保留现归属{keep_count}张")
    for reason_text, count in sorted(skip_counter.items(), key=lambda kv: -kv[1]):
        print(f"  {count}张：{reason_text}")
    return result


def _pick_order_dir(order: dict, loose: dict, key: tuple[str, str]) -> tuple[str, str]:
    """选择归类目标文件夹：优先与抓拍同日期的订单文件夹，其次时间最近的
    订单抓拍所在文件夹；订单在该日期还没有文件夹时（跨天订单另一头）
    按月份照片根新建 <M月D日>/<订单号>。"""
    if loose["date"] in order["dirs"].values():
        for directory, directory_date in order["dirs"].items():
            if directory_date == loose["date"]:
                return directory, directory_date
    nearest_dir, nearest_dt, nearest_date = "", None, loose["date"]
    for directory, capture_time in order["nearest_anchor"].items():
        dt = datetime.strptime(capture_time, "%Y-%m-%d %H:%M:%S")
        if nearest_dt is None or abs((dt - loose["dt"]).total_seconds()) < \
                abs((nearest_dt - loose["dt"]).total_seconds()):
            nearest_dir, nearest_dt, nearest_date = directory, dt, capture_time[:10]
    if nearest_dir:
        return nearest_dir, nearest_date
    # 订单还没有任何已落盘文件夹可参照：按 loose 自身路径推月份照片根
    loose_dir = Path(loose["filepath"]).parent.parent  # .../月份根/日期/未关联图片
    month_root = loose_dir.parent
    y, m, d = (int(part) for part in loose["date"].split("-"))
    target = month_root / f"{m}月{d}日" / key[1]
    return str(target), loose["date"]


def undo_loose_assign(conn: sqlite3.Connection, capture_id: int | None = None) -> int:
    """撤销未关联归类：按 loose_assignments 日志把文件移回原处并还原字段。

    capture_id=None 撤销全部。恢复后建议重跑 associate_orders 重聚车辆。
    """
    ensure_loose_assignments_schema(conn)
    rows = conn.execute(
        """SELECT capture_id, park, from_order, date_old, filepath_old,
                  filename_old, filepath_new, filename_new
           FROM loose_assignments""" + (" WHERE capture_id=?" if capture_id else ""),
        (capture_id,) if capture_id else (),
    ).fetchall()
    conn.isolation_level = None
    restored = failed = 0
    for row in rows:
        (cid, park, from_order, date_old, filepath_old, filename_old,
         filepath_new, filename_new) = row
        current = conn.execute(
            "SELECT filepath, filename FROM captures WHERE id=?", (cid,)).fetchone()
        if not current:
            continue
        # 现文件可能记在 current.filepath 或日志 filepath_new（撤销后重归类的
        # 场景两者不同）；一律项目根锚定，避免 app/ 相对路径歧义。
        source = None
        for stored in (current[0], filepath_new):
            candidate = project_file(stored)
            if candidate.is_file():
                source = candidate
                break
        old_dir = project_file(Path(filepath_old).parent)
        target_old = old_dir / (filename_old or current[1])
        try:
            old_dir.mkdir(parents=True, exist_ok=True)
            if source and source.resolve() != target_old.resolve():
                shutil.move(str(source), str(target_old))
        except OSError as exc:
            failed += 1
            print(f"  撤销失败 #{cid}: {exc}")
            continue
        conn.execute(
            """UPDATE captures SET order_id=?, date=?, filepath=?, filename=?,
                      plate_match_reason='', vehicle_id=NULL
               WHERE id=?""",
            (from_order, date_old, filepath_old, filename_old or current[1], cid),
        )
        conn.execute("DELETE FROM loose_assignments WHERE capture_id=?", (cid,))
        restored += 1
    conn.commit()
    print(f"撤销归类：恢复{restored}张，失败{failed}张"
          + ("，建议重跑 associate_orders 重聚车辆" if restored else ""))
    return restored


# ---------------------------------------------------------------------------
# 共用大门跨园区去重：同一张抓拍照片被上游写进多个园区目录后，sync_files 按
# (park, order_id, filename) 各插一行、互不相识，车辆被多个园区重复计入。
# 本步骤把大门照片（入磅/出磅）按文件名身份分组判重：
# - 新版命名（末尾唯一id，如 ...08：15：59_2346200）：id 全等直接判同一事件
#   （上游保证跨园区文件名一致，不做内容校验——拷贝若被转码，内容校验反而漏真重复）；
#   id 不同即不同事件（同秒不同车），绝不合并——这正是唯一id的价值；
# - 旧命名（无id，存量绝大多数）：同 (阶段, 拍摄秒) 或同 uuid 只是"疑似"，
#   必须文件字节一致（size+MD5）才判同一事件——不同车不可能字节相同，
#   保证旧数据重跑零误杀；任一文件不在盘上无法确认时跳过不合并。
# 确认重复的组内按证据仲裁唯一赢家（真实订单 > 车牌锚定 > 行程时间线 >
# 外观 > 时间就近 > 编号），落败副本移入 重复抓拍隔离-<原园区> 园区，
# 全部落 gate_dedup_log 日志，--undo-dedup 可撤销。
# ---------------------------------------------------------------------------

def _file_md5(path: Path) -> str | None:
    """文件内容 MD5（流式读取）；IO 失败返回 None（按无法确认处理）。"""
    try:
        digest = hashlib.md5()
        with open(path, "rb") as fin:
            for chunk in iter(lambda: fin.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _parse_capture_time(value: str | None) -> datetime | None:
    """完整时间戳解析；date-only/空/异常一律 None（证据降级，不抛错）。"""
    if not value or len(value) < 19:
        return None
    try:
        return datetime.strptime(value[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _normalized_emb(blob) -> np.ndarray | None:
    if not blob:
        return None
    feature = np.frombuffer(blob, np.float32)
    norm = float(np.linalg.norm(feature))
    return feature / norm if norm > 0 else None


def _dedup_identity(stage: str, filename: str) -> tuple[str, str | None] | None:
    """解析大门照片文件名的去重身份：返回 (候选组键, 事件id或None)。

    候选组键把"可能是同一事件"的照片聚到一起；事件id 非 None 表示新版
    NVR 命名末尾带唯一 id（上游保证同一事件跨园区文件名一致），可直接
    判同一事件。文件名识别不了（无时间片段也非 <阶段>_<uuid> 形态）时
    返回 None，该照片跳过判重（保守，宁可漏判不可误判）。
    """
    stem = _REID_SUFFIX_RE.sub("", Path(filename).stem)
    time_match = FILENAME_TIME_RE.search(stem)
    if time_match:
        # 全角/半角冒号统一，两个园区写法差异不影响分组
        time_key = time_match.group(0).replace("：", ":")
        uid_match = _FILENAME_UID_RE.match(stem[time_match.end():])
        uid = uid_match.group(1) if uid_match else ""
        candidate = f"{stage}|T|{time_key}"
        return candidate, (f"{candidate}|{uid}" if uid else None)
    token_match = _GATE_TOKEN_RE.match(stem)
    if token_match and _UUID_TOKEN_RE.fullmatch(token_match.group(1)):
        return f"{stage}|U|{token_match.group(1).lower()}", None
    return None


def _dedup_evidence(item: dict, members_index: dict,
                    seq_max_seconds: float) -> dict:
    """候选副本对本订单的"主张强度"证据（口径对齐 assign_gate 的评估）。

    - real_order：归属真实订单（非未关联图片）；
    - plate_anchor：照片车牌与本订单任一其他成员车牌去省份差 <=1（上游
      NVR 正是按牌归档的）；
    - seq_rank：行程时间线，2=顺/1=无从判断/0=逆——出磅照片的订单内应有
      更早的入磅/质检成员（0~seq_max小时内）；入磅照片的订单内出现更早的
      出磅成员即为逆（这张入磅必属于下一次进场）。同牌车在两个园区都有
      合法订单时，这是分辨"这一次过磅属于哪园"的关键证据；
    - sim / time_diff：对订单其余成员的最大外观相似度与最小时间差。
    """
    real_order = bool(item["order_id"]) and item["order_id"] != "未关联图片"
    ev = {"real_order": real_order, "plate_anchor": False,
          "seq_rank": 1, "sim": -1.0, "time_diff": None}
    others = []
    if real_order:
        others = [m for m in members_index.get((item["park"], item["order_id"]), [])
                  if m["cid"] != item["cid"]]
    if item["plate"] and others:
        ev["plate_anchor"] = any(
            m["plate"] and _plate_edit_distance(item["plate"], m["plate"]) <= 1
            for m in others)
    dt = item["dt"]
    if others:
        dts = [m["dt"] for m in others if m["dt"]]
        if dts and dt:
            ev["time_diff"] = min(abs((dt - t).total_seconds()) for t in dts)
        if item["emb"] is not None:
            sims = [cosine(item["emb"], m["emb"]) for m in others
                    if m["emb"] is not None]
            if sims:
                ev["sim"] = max(sims)
    if dt and others:
        if item["stage"] == "出磅":
            ok = any(m["dt"] and m["stage"] in ("入磅", "质检")
                     and 0 <= (dt - m["dt"]).total_seconds() <= seq_max_seconds
                     for m in others)
            ev["seq_rank"] = 2 if ok else 0
        elif item["stage"] == "入磅":
            bad = any(m["dt"] and m["stage"] == "出磅" and m["dt"] < dt
                      for m in others)
            ev["seq_rank"] = 0 if bad else 2
    return ev


def _evidence_note(ev: dict) -> str:
    notes = []
    if ev["plate_anchor"]:
        notes.append("车牌锚定")
    if ev["seq_rank"] == 2:
        notes.append("行程顺")
    elif ev["seq_rank"] == 0:
        notes.append("行程逆")
    if ev["sim"] > 0:
        notes.append(f"外观{ev['sim']:.2f}")
    if ev["time_diff"] is not None:
        notes.append(f"时间差{int(ev['time_diff']) // 60}分钟")
    return "/".join(notes) or "证据持平"


def _dedup_better(a: tuple, b: tuple, sim_margin: float,
                  time_margin: float) -> bool:
    """仲裁比较：候选 a 是否比 b 更可能是"正确订单"的那一份。

    证据阶梯逐级裁决；外观/时间差距在噪声带内（margin）不作为决胜依据，
    最终回退到确定性键（证据值+编号），保证强制选边且结果可复现。
    """
    ea, eb = a[1], b[1]
    if ea["real_order"] != eb["real_order"]:
        return ea["real_order"]
    if ea["plate_anchor"] != eb["plate_anchor"]:
        return ea["plate_anchor"]
    if ea["seq_rank"] != eb["seq_rank"]:
        return ea["seq_rank"] > eb["seq_rank"]
    if abs(ea["sim"] - eb["sim"]) >= sim_margin:
        return ea["sim"] > eb["sim"]
    ta, tb = ea["time_diff"], eb["time_diff"]
    if ta is not None and tb is not None and abs(ta - tb) >= time_margin:
        return ta < tb
    ka = (0 if ea["real_order"] else 1, 0 if ea["plate_anchor"] else 1,
          -ea["seq_rank"], -ea["sim"],
          ta if ta is not None else float("inf"), a[0]["cid"])
    kb = (0 if eb["real_order"] else 1, 0 if eb["plate_anchor"] else 1,
          -eb["seq_rank"], -eb["sim"],
          tb if tb is not None else float("inf"), b[0]["cid"])
    return ka < kb


def _quarantine_dir(root: Path, park: str, order_id: str, date_str: str) -> Path:
    """落败副本的隔离目录（绝对路径）：
    <照片根>/重复抓拍隔离-<原园区>/<月份根>/<M月D日>/<订单号>。

    必须与 sync_files 的扫描结构一致（园区 -> 月份根 -> 日期 -> 订单，
    月份根名内嵌年份供 parse_folder_date 取年），全盘重扫才能按
    (隔离园区, order_id, filename) 命中已改归属的行，幂等不复活。目录
    建在扫描根下（与 --root 一致，默认项目 data/），每个原园区一个隔离
    园区，隔离区内订单号不撞号、原归属一目了然。
    """
    park_dir = root / f"{QUARANTINE_PARK_PREFIX}-{park}"
    if date_str and len(date_str) >= 10:
        try:
            y, m, d = (int(part) for part in date_str[:10].split("-"))
        except ValueError:
            y = m = d = 0
        if y and m and d:
            return park_dir / f"{y}年{m}月份抓拍照片" / f"{m}月{d}日" / \
                (order_id or "未关联图片")
    return park_dir / "未标日期" / (order_id or "未关联图片")


def dedup_gate_captures(conn: sqlite3.Connection, apply_changes: bool = False,
                        sim_margin: float = 0.03,
                        time_margin: float = 300.0,
                        seq_max_hours: float = 12.0,
                        root: Path = DEFAULT_ROOT) -> dict:
    """共用大门照片跨园区去重（默认试运行，--apply 落盘）。

    每个确认重复的组仲裁出唯一赢家留在其现属订单，落败副本移入扫描根下
    的 重复抓拍隔离-<原园区> 园区（文件不在盘上的仅库内改 park）。两种
    情形不强制选边、原样保留待人工：整组都挂在"未关联图片"（没有"正确
    订单"可归属）；整组都属同一园区（同园区重复说明是上游把同一张过磅
    照片写进了两个订单，自动选边会把照片从订单里拉走，伤害订单归属）。
    阈值可环境变量覆盖：GATE_DEDUP_SIM_MARGIN、GATE_DEDUP_TIME_MARGIN、
    GATE_DEDUP_SEQ_MAX_HOURS。
    """
    ensure_gate_dedup_log_schema(conn)
    sim_margin = float(os.environ.get("GATE_DEDUP_SIM_MARGIN", sim_margin))
    time_margin = float(os.environ.get("GATE_DEDUP_TIME_MARGIN", time_margin))
    seq_max_seconds = float(os.environ.get(
        "GATE_DEDUP_SEQ_MAX_HOURS", seq_max_hours)) * 3600

    rows = conn.execute(
        """SELECT id, coalesce(park,''), order_id, date, capture_time, plate_final,
                  reid_embedding, vehicle_type, filepath, filename, stage
           FROM captures
           WHERE stage IN ('入磅','出磅')
             AND coalesce(capture_status,'vehicle')!='no_vehicle'
             AND coalesce(park,'') NOT LIKE ?
             AND filename IS NOT NULL AND filename!=''
             AND id NOT IN (SELECT capture_id FROM gate_dedup_log)""",
        (QUARANTINE_PARK_LIKE,),
    ).fetchall()

    # 证据索引：订单成员（含阶段，供行程时间线判定）；隔离区订单不参与，
    # 落败副本也不该成为任何一方的证据。
    members_index: dict[tuple[str, str], list[dict]] = {}
    for mid, mpark, morder, mtime, mplate, mblob, mvtype, mstage in conn.execute(
        """SELECT id, coalesce(park,''), order_id, capture_time, plate_final,
                  reid_embedding, vehicle_type, stage
           FROM captures
           WHERE coalesce(order_id,'')!='' AND order_id!='未关联图片'
             AND coalesce(capture_status,'vehicle')!='no_vehicle'
             AND coalesce(park,'') NOT LIKE ?""",
        (QUARANTINE_PARK_LIKE,),
    ):
        members_index.setdefault((mpark, morder), []).append(
            {"cid": mid, "dt": _parse_capture_time(mtime), "plate": mplate or "",
             "emb": _normalized_emb(mblob), "vtype": mvtype or "",
             "stage": mstage or ""})

    # ---- 按候选组键分组 ----
    candidates: dict[str, list[dict]] = defaultdict(list)
    unparseable = 0
    for row in rows:
        identity = _dedup_identity(row[10], row[8])
        if identity is None:
            unparseable += 1
            continue
        candidate_key, event_id = identity
        candidates[candidate_key].append(
            {"cid": row[0], "park": row[1], "order_id": row[2], "date": row[3],
             "capture_time": row[4], "plate": row[5] or "", "blob": row[6],
             "vtype": row[7], "filepath": row[8], "filename": row[9],
             "stage": row[10], "event_id": event_id, "md5": None, "dt": None,
             "emb": None})

    def _ensure_md5(item: dict) -> None:
        if item["md5"] is not None:
            return
        source = project_file(item["filepath"])
        if not source.is_file():
            source = resolve_stored_path(item["filepath"])
        item["md5"] = _file_md5(source) if source.is_file() else None

    def _cluster(items: list[dict]) -> list[list[dict]]:
        """候选组内切事件簇：唯一id一致直接并；否则字节一致才并；
        双方都有唯一id但不一致时绝不并（id 优先于内容）。"""
        n = len(items)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                ui, uj = items[i]["event_id"], items[j]["event_id"]
                if ui and uj:
                    same = ui == uj
                else:
                    _ensure_md5(items[i])
                    _ensure_md5(items[j])
                    mi, mj = items[i]["md5"], items[j]["md5"]
                    same = bool(mi and mj and mi == mj)
                if same:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)
        clusters: dict[int, list[dict]] = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(items[i])
        return [c for c in clusters.values() if len(c) >= 2]

    dup_groups = needs_review = skip_md5_mismatch = skip_no_file = 0
    same_park_review = 0
    plan: list[tuple[dict, dict, str, str, str]] = []  # (落败副本, 赢家, confirm, 理由, 组键)
    for candidate_key in sorted(candidates):
        items = candidates[candidate_key]
        if len(items) < 2:
            continue
        clusters = _cluster(items)
        if not clusters:
            if any(not it["event_id"] for it in items):
                if any(it["md5"] is None for it in items):
                    skip_no_file += 1
                else:
                    skip_md5_mismatch += 1
            continue
        for cluster in clusters:
            dup_groups += 1
            scored = []
            for item in cluster:
                item["dt"] = _parse_capture_time(item["capture_time"])
                item["emb"] = _normalized_emb(item["blob"])
                scored.append((item, _dedup_evidence(item, members_index,
                                                     seq_max_seconds)))
            if not any(ev["real_order"] for _, ev in scored):
                needs_review += 1  # 全是未关联图片：无"正确订单"可归属，保留两边
                continue
            if len({item["park"] for item, _ev in scored}) == 1:
                # 同园区内出现同文件名副本（上游把同一张过磅照片写进同园区
                # 两个订单）：强制选边会把照片从订单里拉走，直接伤害订单
                # 归属与归类覆盖——保留两边待人工，绝不自动隔离。
                same_park_review += 1
                continue
            scored.sort(key=lambda pair: pair[0]["cid"])
            winner = scored[0]
            for pair in scored[1:]:
                if _dedup_better(pair, winner, sim_margin, time_margin):
                    winner = pair
            for loser, _ev in scored:
                if loser is winner[0]:
                    continue
                if loser["event_id"] and loser["event_id"] == winner[0]["event_id"]:
                    confirm = "filename_id"
                else:
                    confirm = "content_md5"
                reason = (f"共用大门重复照片：与 #{winner[0]['cid']}"
                          f"（{winner[0]['park']}/订单{winner[0]['order_id'] or '未关联'}）"
                          f"为同一事件（{'文件名唯一id一致' if confirm == 'filename_id' else '文件内容MD5一致'}），"
                          f"本单证据[{_evidence_note(_ev)}]弱于对方[{_evidence_note(winner[1])}]")
                plan.append((loser, winner[0], confirm, reason, candidate_key))

    moved = failed = db_only_moves = 0
    touched_orders: set[tuple[str, str]] = set()
    now = datetime.now().isoformat(timespec="seconds")
    if apply_changes:
        conn.isolation_level = None
        for loser, winner_item, confirm, reason, dup_key in plan:
            to_park = f"{QUARANTINE_PARK_PREFIX}-{loser['park']}"
            target_dir_abs = _quarantine_dir(root, loser["park"], loser["order_id"],
                                             loser["date"])
            source_abs = project_file(loser["filepath"])
            if not source_abs.is_file():
                source_abs = resolve_stored_path(loser["filepath"])
            db_only = not source_abs.is_file()
            new_fp, new_fn = loser["filepath"], loser["filename"]
            if not db_only:
                target_abs = target_dir_abs / loser["filename"]
                final_name = loser["filename"]
                try:
                    target_dir_abs.mkdir(parents=True, exist_ok=True)
                    if target_abs.exists():  # 理论不可达（同键早被 sync 合并），兜底防覆盖
                        stem, suffix = target_abs.stem, target_abs.suffix
                        k = 1
                        while target_abs.with_name(f"{stem}_dedup{k}{suffix}").exists():
                            k += 1
                        final_name = f"{stem}_dedup{k}{suffix}"
                        target_abs = target_abs.with_name(final_name)
                    if source_abs.resolve() != target_abs.resolve():
                        shutil.move(str(source_abs), str(target_abs))
                    new_fp, new_fn = to_stored_path(target_abs), final_name
                except OSError as exc:
                    failed += 1
                    print(f"  隔离失败 #{loser['cid']}: {exc}")
                    continue
            else:
                db_only_moves += 1
                reason += "（原图不在盘，仅库内改园区）"
            conn.execute(
                """UPDATE captures SET park=?, filepath=?, filename=?, vehicle_id=NULL
                   WHERE id=?""",
                (to_park, new_fp, new_fn, loser["cid"]),
            )
            conn.execute(
                """INSERT OR REPLACE INTO gate_dedup_log
                     (capture_id, dup_key, confirm, from_park, from_order,
                      from_path, filename_old, to_park, to_path, filename_new,
                      winner_id, winner_park, winner_order, reason, deduped_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (loser["cid"], dup_key, confirm, loser["park"], loser["order_id"],
                 loser["filepath"], loser["filename"], to_park, new_fp, new_fn,
                 winner_item["cid"], winner_item["park"], winner_item["order_id"],
                 reason, now),
            )
            if loser["order_id"] and loser["order_id"] != "未关联图片":
                touched_orders.add((loser["park"], loser["order_id"]))
            moved += 1
            print(f"  隔离 #{loser['cid']} {loser['park']}/订单{loser['order_id'] or '未关联'} "
                  f"-> {to_park}（赢家 #{winner_item['cid']}）")
        if touched_orders:
            _refresh_order_attributes(conn, touched_orders)
        conn.commit()
    else:
        for loser, winner_item, confirm, reason, _key in plan[:50]:
            print(f"  试运行 #{loser['cid']} {loser['park']}/订单{loser['order_id'] or '未关联'} "
                  f"-> 隔离（赢家 #{winner_item['cid']} "
                  f"{winner_item['park']}，{confirm}）")
        if len(plan) > 50:
            print(f"  …（其余 {len(plan) - 50} 条略）")

    result = {
        "gate_captures": len(rows), "duplicates": len(plan),
        "dup_groups": dup_groups, "moved": moved, "db_only": db_only_moves,
        "failed": failed, "needs_review_groups": needs_review,
        "same_park_review_groups": same_park_review,
        "skipped_md5_mismatch": skip_md5_mismatch,
        "skipped_no_file": skip_no_file, "unparseable": unparseable,
    }
    mode = "已隔离" if apply_changes else "试运行（加 --apply 落盘）"
    stat = (f"隔离{moved}张（其中仅库内改园区{db_only_moves}张），失败{failed}张"
            if apply_changes else f"计划隔离{len(plan)}张")
    print(f"共用大门照片去重{mode}：入磅/出磅 {len(rows)}张，确认重复组{dup_groups}个，"
          f"{stat}；全未关联待人工{needs_review}组，同园区重复保留待人工{same_park_review}组，"
          f"内容不一致跳过{skip_md5_mismatch}组，"
          f"文件不在盘跳过{skip_no_file}组，文件名不可解析{unparseable}张")
    return result


def undo_gate_dedup(conn: sqlite3.Connection, capture_id: int | None = None) -> int:
    """撤销共用大门去重：按 gate_dedup_log 把文件移回原园区并还原字段。

    capture_id=None 撤销全部。恢复后建议重跑 associate_orders 重聚车辆。
    """
    ensure_gate_dedup_log_schema(conn)
    rows = conn.execute(
        """SELECT capture_id, from_park, from_path, filename_old,
                  to_path, filename_new
           FROM gate_dedup_log""" + (" WHERE capture_id=?" if capture_id else ""),
        (capture_id,) if capture_id else (),
    ).fetchall()
    conn.isolation_level = None
    restored = failed = 0
    for row in rows:
        cid, from_park, from_path, filename_old, to_path, filename_new = row
        current = conn.execute(
            "SELECT filepath, filename FROM captures WHERE id=?", (cid,)).fetchone()
        if not current:
            continue
        source = None
        for stored in (current[0], to_path):
            candidate = project_file(stored)
            if candidate.is_file():
                source = candidate
                break
        old_dir = project_file(Path(from_path).parent)
        target_old = old_dir / (filename_old or current[1])
        try:
            old_dir.mkdir(parents=True, exist_ok=True)
            if source and source.resolve() != target_old.resolve():
                shutil.move(str(source), str(target_old))
        except OSError as exc:
            failed += 1
            print(f"  撤销失败 #{cid}: {exc}")
            continue
        conn.execute(
            """UPDATE captures SET park=?, filepath=?, filename=?, vehicle_id=NULL
               WHERE id=?""",
            (from_park, from_path, filename_old or current[1], cid),
        )
        conn.execute("DELETE FROM gate_dedup_log WHERE capture_id=?", (cid,))
        restored += 1
    conn.commit()
    print(f"撤销大门去重：恢复{restored}张，失败{failed}张"
          + ("，建议重跑 associate_orders 重聚车辆" if restored else ""))
    return restored


def export_training_datasets(
    conn: sqlite3.Connection,
    db_path: Path,
    out_dir: Path,
    *,
    kind: str = "all",
    include_verified_plates: bool = True,
) -> dict[str, Any]:
    """将人工标注表物化为标准训练数据集（防同日同车泄漏 + manifest 版本化）。

    支持 kind:
      - reid:      VeRi/FastReID 格式（image_train / image_query / image_test + pairs.json）
      - detection: COCO JSON（manual_box_edits，供 PaddleDetection 微调）
      - attr:      PaddleClas PULC PPLCNet 19维属性标注（颜色10+车型9）
      - lpr:       PP-OCR 车牌识别清单（rec_gt_train.txt / rec_gt_val.txt）
      - all:       一次性导出以上全部四套
    """
    ensure_schema(conn)
    ensure_label_schema(conn)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    db_parent = Path(db_path).resolve().parent
    kinds = ["reid", "detection", "attr", "lpr"] if kind == "all" else [kind]
    summary: dict[str, Any] = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "db": str(Path(db_path).resolve()),
        "out_dir": str(out_dir),
        "kinds": {},
    }

    def _crop_abs(stored: str | None) -> Path | None:
        if not stored:
            return None
        p = Path(stored)
        p = p if p.is_absolute() else (db_parent / p)
        return p if p.is_file() else None

    if "reid" in kinds:
        import finetune_reid
        reid_dir = out_dir / "reid"
        for sub in ("image_train", "image_query", "image_test"):
            (reid_dir / sub).mkdir(parents=True, exist_ok=True)
        ds = finetune_reid.collect_reid_dataset(
            Path(db_path), include_verified_plates=include_verified_plates
        )
        cam_ids: dict[str, int] = {}

        def _cam_idx(cam_key: str) -> int:
            if cam_key not in cam_ids:
                cam_ids[cam_key] = len(cam_ids) + 1
            return min(999, cam_ids[cam_key])

        train_n = query_n = test_n = 0
        for s in ds["train_samples"]:
            cid_idx = _cam_idx(s.get("camera_key") or "cam")
            dst = reid_dir / "image_train" / f"{s['pid'] + 1:04d}_c{cid_idx:03d}_{s['capture_id']:08d}.jpg"
            shutil.copy2(s["crop_path"], dst)
            train_n += 1
        seen_q_pid: set[int] = set()
        for s in ds["val_samples"]:
            cid_idx = _cam_idx(s.get("camera_key") or "cam")
            fname = f"{s['pid'] + 1:04d}_c{cid_idx:03d}_{s['capture_id']:08d}.jpg"
            if s["pid"] not in seen_q_pid:
                seen_q_pid.add(s["pid"])
                shutil.copy2(s["crop_path"], reid_dir / "image_query" / fname)
                query_n += 1
            else:
                shutil.copy2(s["crop_path"], reid_dir / "image_test" / fname)
                test_n += 1
        (reid_dir / "pairs.json").write_text(
            json.dumps(
                {
                    "train_same": ds["train_same_pairs"],
                    "train_diff": ds["train_diff_pairs"],
                    "val_manual": ds["val_manual_pairs"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        summary["kinds"]["reid"] = {
            **ds["stats"],
            "exported_train": train_n,
            "exported_query": query_n,
            "exported_gallery": test_n,
            "dir": str(reid_dir),
        }

    if "detection" in kinds:
        det_dir = out_dir / "detection"
        det_dir.mkdir(parents=True, exist_ok=True)
        rows = conn.execute(
            """SELECT m.capture_id, m.box, c.filepath, coalesce(c.date,'') AS date
               FROM manual_box_edits m JOIN captures c ON c.id=m.capture_id
               ORDER BY c.date, m.capture_id"""
        ).fetchall()
        images, annotations = [], []
        ann_id = 1
        for r in rows:
            src = resolve_stored_path(r[2])
            if not src.is_file():
                continue
            try:
                box = json.loads(r[1])
                x1, y1, x2, y2 = [float(v) for v in box]
                if x2 - x1 < 24 or y2 - y1 < 24:
                    continue
                with Image.open(src) as im:
                    w, h = im.size
            except Exception:
                continue
            img_id = int(r[0])
            images.append({"id": img_id, "file_name": str(src), "width": w, "height": h, "date": r[3]})
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "bbox": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
                "area": round((x2 - x1) * (y2 - y1), 1), "iscrowd": 0,
            })
            ann_id += 1
        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "vehicle"}],
        }
        (det_dir / "annotations_coco.json").write_text(
            json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary["kinds"]["detection"] = {
            "images": len(images),
            "annotations": len(annotations),
            "ready": len(images) >= 200,
            "note": "" if len(images) >= 200 else "人工框选样本 <200 张，建议继续积累后再微调检测模型",
            "file": str(det_dir / "annotations_coco.json"),
        }

    if "attr" in kinds:
        attr_dir = out_dir / "attr"
        attr_dir.mkdir(parents=True, exist_ok=True)
        color_order = ("yellow", "orange", "green", "gray", "red", "blue",
                       "white", "golden", "brown", "black")
        type_order = ("sedan", "suv", "van", "hatchback", "mpv", "pickup",
                      "bus", "truck", "estate")
        rows = conn.execute(
            """SELECT c.id, c.vehicle_crop_path, coalesce(c.date,'') AS date,
                      coalesce(ca.corrected_color,'') AS mc, coalesce(ca.corrected_type,'') AS mt,
                      coalesce(c.vehicle_color,'') AS ac, coalesce(c.vehicle_type,'') AS at
               FROM capture_annotations ca JOIN captures c ON c.id=ca.capture_id
               WHERE coalesce(ca.usability,'valid')!='invalid'
                 AND coalesce(ca.no_vehicle_manual,0)=0
                 AND (coalesce(ca.corrected_color,'')!='' OR coalesce(ca.corrected_type,'')!='')
               ORDER BY c.date, c.id"""
        ).fetchall()
        lines: list[tuple[str, str]] = []
        for r in rows:
            crop = _crop_abs(r[1])
            if not crop:
                continue
            cc = normalize_color(r[3]) or normalize_color(r[5])
            tt = normalize_type(r[4]) or normalize_type(r[6])
            vec = [1 if c == cc else 0 for c in color_order] + [1 if t == tt else 0 for t in type_order]
            lines.append((r[2], f"{crop}\t{','.join(map(str, vec))}"))
        split_at = max(1, int(len(lines) * 0.8)) if len(lines) > 1 else len(lines)
        (attr_dir / "train_list.txt").write_text("\n".join(x[1] for x in lines[:split_at]), encoding="utf-8")
        (attr_dir / "val_list.txt").write_text("\n".join(x[1] for x in lines[split_at:]), encoding="utf-8")
        summary["kinds"]["attr"] = {
            "total": len(lines),
            "train": split_at if lines else 0,
            "val": max(0, len(lines) - split_at),
            "ready": len(lines) >= 300,
            "note": "" if len(lines) >= 300 else "属性样本 <300 张，建议继续积累后再微调属性模型",
            "dir": str(attr_dir),
        }

    if "lpr" in kinds:
        lpr_dir = out_dir / "lpr"
        lpr_dir.mkdir(parents=True, exist_ok=True)
        rows = conn.execute(
            """SELECT c.id, c.vehicle_crop_path, coalesce(c.date,'') AS date, ca.corrected_plate
               FROM capture_annotations ca JOIN captures c ON c.id=ca.capture_id
               WHERE coalesce(ca.usability,'valid')='valid'
                 AND coalesce(ca.no_vehicle_manual,0)=0
                 AND coalesce(ca.corrected_plate,'')!=''
               ORDER BY c.date, c.id"""
        ).fetchall()
        lines = []
        for r in rows:
            crop = _crop_abs(r[1])
            if not crop:
                continue
            lines.append((r[2], f"{crop}\t{r[3]}"))
        split_at = max(1, int(len(lines) * 0.8)) if len(lines) > 1 else len(lines)
        (lpr_dir / "rec_gt_train.txt").write_text("\n".join(x[1] for x in lines[:split_at]), encoding="utf-8")
        (lpr_dir / "rec_gt_val.txt").write_text("\n".join(x[1] for x in lines[split_at:]), encoding="utf-8")
        summary["kinds"]["lpr"] = {
            "total": len(lines),
            "train": split_at if lines else 0,
            "val": max(0, len(lines) - split_at),
            "ready": len(lines) >= 300,
            "note": "" if len(lines) >= 300 else "车牌样本 <300 张，建议继续积累后再微调 LPR 模型",
            "dir": str(lpr_dir),
        }

    (out_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def print_stats(conn: sqlite3.Connection) -> None:
    total = conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0]
    orders = conn.execute(
        """SELECT COUNT(*) FROM (
             SELECT DISTINCT park, order_id FROM captures
             WHERE coalesce(order_id,'')!='')"""
    ).fetchone()[0]
    stages = conn.execute(
        """SELECT stage, COUNT(*) FROM captures
           WHERE coalesce(stage,'')!='' GROUP BY stage ORDER BY stage"""
    ).fetchall()
    staged = "，".join(f"{stage}{count}" for stage, count in stages)
    cross_date = conn.execute(
        """SELECT COUNT(*) FROM (
             SELECT park, order_id FROM captures
             WHERE coalesce(order_id,'')!='' AND coalesce(date,'')!=''
             GROUP BY park, order_id HAVING COUNT(DISTINCT date)>1)"""
    ).fetchone()[0]
    print(f"数据统计：{total}张，{orders}个订单，跨日期订单{cross_date}个；{staged}")
    park_rows = conn.execute(
        """SELECT coalesce(park,'') AS park, COUNT(*) AS captures,
                  COUNT(DISTINCT order_id) AS orders
           FROM captures GROUP BY coalesce(park,'') ORDER BY park"""
    ).fetchall()
    for park, count, park_orders in park_rows:
        label = park or "（未标注园区）"
        print(f"  园区[{label}]：{count}张，{park_orders}个订单")


def main() -> None:
    parser = argparse.ArgumentParser(description="园区抓拍照片扫描入库")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="照片根目录：data 根（一级子目录=园区，默认）"
                             "或某个月份照片根（老布局，园区名取 --park）")
    parser.add_argument("--park", default="",
                        help="只处理指定园区（名字为 data 下的一级目录名）；"
                             "老布局照片根时作为其园区名，缺省为未标注；"
                             "与 --assign-loose 联用时限定归类的园区")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--sync-only", action="store_true", help="只扫描目录，不跑OCR")
    parser.add_argument("--process-missing", action="store_true", help="补充OCR时间戳/OSD")
    parser.add_argument("--force-process", action="store_true", help="强制重跑OCR")
    parser.add_argument("--stage", default="",
                        help="配合 --force-process/--process-missing 只处理指定环节"
                             "（如 --stage 抓拍 定向回填中文设备名）")
    parser.add_argument("--retry-failed", action="store_true",
                        help="只重跑 ocr_status=fallback 的帧（启用时刻带抢救通道）")
    parser.add_argument("--retry-osd", action="store_true",
                        help="只重跑抓拍阶段 osd_plate 为空的帧（纯黑底部条带精准补扫车牌）")
    parser.add_argument("--refresh-osd-footer", action="store_true",
                        help="按新规则（设备编号 DMR/DMC/CK + 背向翻转 + 黄牌货车）快速重刷抓拍底部 OSD")
    parser.add_argument("--relevant-only", action="store_true",
                        help="配合 --refresh-osd-footer 只刷新订单内及与订单相关的未关联抓拍（跳过社会过路车）")
    parser.add_argument("--replay-manual", action="store_true",
                        help="全库重建后从人工表恢复写回字段（环节/方向/颜色/车型/"
                             "车牌/时间/误检标记），幂等可重复执行")
    parser.add_argument("--export-training", action="store_true",
                        help="从人工标注表导出标准训练数据集（COCO/PPLCNet/PP-OCR/FastReID）")
    parser.add_argument("--kind", default="all",
                        choices=["all", "reid", "detection", "attr", "lpr"],
                        help="配合 --export-training 指定导出种类（默认 all）")
    parser.add_argument("--out", type=Path, default=APP_DIR.parent / "datasets" / "latest",
                        help="配合 --export-training 指定输出目录")
    parser.add_argument("--manual-only", action="store_true",
                        help="配合 --export-training 仅导出纯人工标注的 ReID 样本")
    parser.add_argument("--associate-only", action="store_true", help="只重新聚合订单内车辆")
    parser.add_argument("--assign-loose", action="store_true",
                        help="未关联货车归类到订单（默认试运行，加 --apply 落盘）")
    parser.add_argument("--assign-gate", action="store_true",
                        help="入磅/出磅大门照片按时间重新归属订单（默认试运行，"
                             "加 --apply 落盘；订单时间锚包含大门照片，夹在两个"
                             "订单之间时归时间更近的一单）")
    parser.add_argument("--apply", action="store_true",
                        help="与 --assign-loose / --assign-gate / --dedup-gate 联用："
                             "真正移动文件并写库")
    parser.add_argument("--undo-loose-assign", action="store_true",
                        help="撤销未关联归类（全部，或配合 --capture-id 撤销单张）")
    parser.add_argument("--dedup-gate", action="store_true",
                        help="共用大门跨园区重复照片判重（默认试运行，加 --apply 落盘；"
                             "落败副本移入 重复抓拍隔离-<原园区> 园区）")
    parser.add_argument("--undo-dedup", action="store_true",
                        help="撤销大门照片去重（全部，或配合 --capture-id 撤销单张）")
    parser.add_argument("--capture-id", type=int, default=0,
                        help="--undo-loose-assign 的单张撤销目标")
    parser.add_argument("--loose-types", default="truck",
                        help="参与归类的车型，逗号分隔（默认 truck；空串=不限）")
    parser.add_argument("--loose-window-min", type=int, default=120,
                        help="归类时间窗（分钟）：与订单抓拍时刻的最小距离上限")
    parser.add_argument("--loose-sim", type=float, default=0.55,
                        help="无车牌归类的外观相似度阈值")
    parser.add_argument("--loose-margin", type=float, default=0.03,
                        help="无车牌归类对次优订单的相似度领先幅度下限")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    ensure_schema(conn)
    if args.replay_manual:
        print(f"人工标注回放完成：{replay_manual(conn)}")
        return
    if args.refresh_osd_footer:
        res = refresh_osd_footers(conn, park_filter=args.park, limit=args.limit,
                                  relevant_only=args.relevant_only)
        print(f"抓拍黑底 OSD 刷新完成：{res}")
        return
    if args.export_training:
        res = export_training_datasets(
            conn, args.db, args.out, kind=args.kind,
            include_verified_plates=not args.manual_only,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    standalone = (args.assign_loose or args.assign_gate or args.undo_loose_assign
                  or args.dedup_gate or args.undo_dedup or args.retry_osd)
    # 默认流程始终全盘同步：每日流水线靠它发现新园区/新照片（库非空也必须
    # 扫描，否则新数据永远进不了库——金冠库上线时踩过这个坑）。sync_files
    # 对已有文件只做路径刷新，不会重复入库。
    if not args.associate_only and not standalone:
        if not args.root.is_dir():
            parser.error(f"照片目录不存在: {args.root}")
        summary = sync_files(conn, args.root, args.park)
        print(f"目录同步完成：{summary}")
    if args.retry_failed:
        process_missing(conn, args.limit, retry_failed=True)
    if args.retry_osd:
        process_missing(conn, args.limit, retry_osd=True)
    if args.process_missing or args.force_process or not (
        args.sync_only or args.associate_only or standalone or args.retry_failed
    ):
        process_missing(conn, args.limit, args.force_process, stage=args.stage)
    if args.undo_loose_assign:
        undo_loose_assign(conn, args.capture_id or None)
    elif args.assign_loose:
        assign_loose_orders(
            conn, apply_changes=args.apply,
            types=tuple(t.strip() for t in args.loose_types.split(",") if t.strip()),
            window_seconds=args.loose_window_min * 60,
            sim_threshold=args.loose_sim, sim_margin=args.loose_margin,
            park_filter=args.park)
    elif args.assign_gate:
        assign_gate_orders(
            conn, apply_changes=args.apply,
            types=tuple(t.strip() for t in args.loose_types.split(",") if t.strip()),
            window_seconds=args.loose_window_min * 60,
            sim_threshold=args.loose_sim, sim_margin=args.loose_margin,
            park_filter=args.park)
    elif args.dedup_gate:
        dedup_gate_captures(conn, apply_changes=args.apply, root=args.root)
    elif args.undo_dedup:
        undo_gate_dedup(conn, args.capture_id or None)
    elif not args.sync_only:
        print(f"订单内车辆聚合：{associate_orders(conn)}")
    if not standalone:
        print_stats(conn)
    conn.close()


if __name__ == "__main__":
    main()
