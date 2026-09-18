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
  python build_db.py --associate-only       # 只重新聚合订单内车辆
  python build_db.py --root data --park 农易行临泉辉煌库   # 只处理指定园区
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime
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
STAGE_RE = re.compile(r"^(入磅|出磅|抓拍|质检|人脸)_(.+)$")
DATE_RE = re.compile(r"(20\d{2})[-年/.](\d{1,2})[-月/.](\d{1,2})")
TIME_RE = re.compile(r"(2[0-3]|[01]?\d)[:：]([0-5]\d)[:：]([0-5]\d)")

# 订单内的通行顺序：入磅 -> 质检 -> 出磅；抓拍没有固定位置（-1），按时间插入；
# 人脸抓拍（司机身份）不是车辆图片，排最前并直接标记非车辆。
STAGE_RANK = {"入磅": 0, "质检": 1, "抓拍": -1, "人脸": -2, "出磅": 2}


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

# 摄像头叠字常把省份/字母认错，车牌只做归一化保存，匹配规则见实施文档 13.1。
PLATE_RE = re.compile(
    r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]"
    r"[A-Z][A-Z0-9挂学警港澳使领]{5,6}"
)


def normalize_plate(value: str | None) -> str:
    if not value:
        return ""
    value = value.upper().replace(" ", "").replace("·", "").replace("-", "")
    value = re.sub(r"[^\u4e00-\u9fffA-Z0-9挂学警港澳使领]", "", value)
    match = PLATE_RE.search(value)
    return match.group(0) if match else ""


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
          reid_embedding BLOB, reid_source TEXT, reid_model_version TEXT,
          vehicle_id INTEGER,
          UNIQUE(capture_id, instance_index)
        );
        CREATE INDEX IF NOT EXISTS idx_vi_capture ON vehicle_instances(capture_id);
        CREATE INDEX IF NOT EXISTS idx_vi_vehicle ON vehicle_instances(vehicle_id);
        """
    )
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
    _migrate_park_backfill(conn)
    ensure_order_attributes_schema(conn)
    conn.commit()


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
    """RapidOCR 实例；large=True 使用更大检测边长，用于时间戳重试通道。

    intra_op_num_threads 必须压到 2：默认 -1 会让进程池里每个 worker 的
    每个会话都开满全部核心，互相踩踏导致吞吐不升反降。
    """
    global _ocr, _ocr_large
    if large and _ocr_large is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr_large = RapidOCR(det_limit_side_len=1280, det_limit_type="max",
                              intra_op_num_threads=2)
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


def run_ocr_text(image: np.ndarray, large: bool = False) -> list[tuple[str, float]]:
    result, _ = get_ocr(large)(image)
    if not result:
        return []
    return [(str(text), float(conf)) for _, text, conf in result]


def ocr_capture(payload: dict) -> dict:
    """OCR 单张抓拍的时间戳与（抓拍镜头的）底部车辆描述。

    进程池工作函数：时间戳先走小图快速通道，失败再走宽区域大字号重试通道；
    底部OSD只出现在抓拍镜头，其他阶段跳过。
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
                   large: bool = False) -> tuple[str, list[float]]:
        region = rgb[int(height * y0) : int(height * y1), 0 : int(width * x1)]
        if scale != 1.0:
            region = cv2.resize(region, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
        texts = run_ocr_text(region, large=large)
        return (re.sub(r"\s+", "", " ".join(text for text, _ in texts)),
                [conf for _, conf in texts])

    def parse_datetime(joined: str) -> tuple[str, str]:
        date_part = time_part = ""
        if match := DATE_RE.search(joined):
            date_part = f"{int(match[1]):04d}-{int(match[2]):02d}-{int(match[3]):02d}"
        if match := TIME_RE.search(joined):
            time_part = f"{int(match[1]):02d}:{int(match[2]):02d}:{int(match[3]):02d}"
        return date_part, time_part

    # 时间戳是轨迹排序的基石：先小图快速识别，失败再走大字号重试通道。
    top_joined, top_confs = ocr_region(0.0, 0.20, 0.60, 0.6)
    capture_date, hms = parse_datetime(top_joined)
    confidence_values = list(top_confs)
    if not (capture_date and hms):
        top_joined, retry_confs = ocr_region(0.0, 0.22, 0.80, 0.9, large=True)
        capture_date, hms = parse_datetime(top_joined)
        confidence_values += list(retry_confs)
    time_ok = bool(capture_date and hms)

    plate = color = vehicle_type = camera = ""
    bottom_date = bottom_hms = ""
    if stage == "抓拍":
        # 底部OSD字号大但条带很宽：0.75缩放+更大检测边长，车牌召回率
        # 从约10%提升到90%+（0.5缩放时小字严重漏检）。
        bottom_joined, bottom_confs = ocr_region(0.72, 1.0, 1.0, 0.75, large=True)
        confidence_values += list(bottom_confs)
        bottom_date, bottom_hms = parse_datetime(bottom_joined)
        for pattern in (r"车牌号码[:：]?([^\s:：]{6,12})", r"车牌号[:：]?([^\s:：]{6,12})"):
            if match := re.search(pattern, bottom_joined):
                plate = normalize_plate(match[1])
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
        for candidate in ("小型货车", "大型货车", "重型货车", "货车", "轿车", "客车",
                          "SUV", "皮卡", "面包车", "卡车", "挂车", "罐车", "越野车",
                          "小货车"):
            if candidate in bottom_joined:
                vehicle_type = candidate
                break
        camera_match = re.search(r"(CAMERA\s*\d{1,3})", bottom_joined)
        camera = camera_match.group(1).replace(" ", "") if camera_match else ""

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


def process_missing(conn: sqlite3.Connection, limit: int = 0, force: bool = False) -> int:
    """补充 OCR：时间戳、底部OSD参考、质量分。失败时降级为阶段顺序，不伪造时间。

    OCR 在进程池中并行执行（每个 worker 独立加载 RapidOCR），主进程写库。
    """
    from concurrent.futures import ProcessPoolExecutor

    # 流水线/服务可能并发写库：逐条自动提交，避免长事务长期持有写锁。
    conn.isolation_level = None
    rows = conn.execute(
        f"""SELECT id, filepath, stage FROM captures
            WHERE {'1=1' if force else "coalesce(ocr_status,'')=''"}
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
                    if bottom_hms and str(osd["hms"]):
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
                conn.execute(
                    """UPDATE captures SET capture_time=?, hour=?, ocr_status=?, quality_score=?,
                              osd_plate=?, osd_color=?, osd_type=?, osd_camera=?,
                              plate_final=?, plate_observed=?, plate_confidence=?,
                              recognition_source=?, capture_status=CASE
                                WHEN capture_status='no_vehicle' THEN capture_status ELSE 'vehicle' END,
                              status_reason=?
                       WHERE id=?""",
                    (capture_time, hour, ocr_status, osd["quality"],
                     osd_plate, str(osd["osd_color"]), str(osd["osd_type"]),
                     str(osd["osd_camera"]),
                     osd_plate, osd_plate, float(osd["ocr_conf"]),
                     "OSD文字" if osd_plate else "", status_reason, capture_id),
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
    return processed


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


def _plate_edit_distance(left: str, right: str) -> int:
    """车牌相似距离：车牌前省份汉字无视（实施文档 13.1 + 现场确认），
    从发牌机关字母起逐字符比较；位数不同返回 99。"""
    if len(left) != len(right):
        return 99
    tail_left, tail_right = left[1:], right[1:]
    return sum(1 for x, y in zip(tail_left, tail_right) if x != y)


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
           GROUP BY park, order_id ORDER BY MIN(date), order_id"""
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
           AND coalesce(capture_status,'vehicle')!='no_vehicle'"""
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
           WHERE vi.reid_embedding IS NOT NULL AND coalesce(vi.lpr_plate,'')!=''"""
    ).fetchall()
    for iid, lpr_plate, lpr_conf, date, park, color, vtype, blob in inst_rows:
        next_gid += 1
        groups[next_gid] = {"kind": "instance", "capture_id": iid, "park": park or ""}
        conn.execute("UPDATE vehicle_instances SET vehicle_id=? WHERE id=?", (next_gid, iid))
        # 实例是单帧组：特征直接内置，走统一的组特征逻辑
        g = groups[next_gid]
        g["_prefetched"] = [(lpr_plate, lpr_conf, date, color, vtype, blob)]

    # 组特征：车牌（含来源置信）、日期、颜色/车型分布、组均值向量
    for gid, g in groups.items():
        if "_prefetched" in g:
            rows = g.pop("_prefetched")
        elif g["kind"] == "order":
            rows = conn.execute(
                """SELECT coalesce(plate_final,'') AS plate, lpr_conf, date,
                          vehicle_color, vehicle_type, reid_embedding
                   FROM captures WHERE vehicle_id=? AND reid_embedding IS NOT NULL
                   AND coalesce(capture_status,'vehicle')!='no_vehicle'""",
                (gid,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT coalesce(plate_final,'') AS plate, lpr_conf, date,
                          vehicle_color, vehicle_type, reid_embedding
                   FROM captures WHERE vehicle_id=?""",
                (gid,),
            ).fetchall()
        g["plates"] = {}
        g["dates"] = set()
        g["colors"] = {}
        g["types"] = {}
        embs = []
        plate_counts: dict[str, int] = {}
        plate_conf: dict[str, float] = {}
        n_frames = 0
        for plate, lpr_conf, date, color, vtype, blob in rows:
            n_frames += 1
            if plate:
                conf = float(lpr_conf or 0.9)
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
        # 组内车牌共识：订单内个别帧会幻觉出邻车/噪声车牌（实测一单 5 帧
        # 读出 5 个不同车牌），聚类只用共识车牌，否则噪声车牌会成为跨组
        # 合并的桥。共识 = 出现 >=2 次的车牌；整单只出现一次时取置信度
        # 最高的那个（单帧订单即其本身）。
        consensus = [p for p, c in plate_counts.items()
                     if c >= 2 or len(plate_counts) == 1]
        if not consensus and plate_counts:
            consensus = [max(plate_counts, key=lambda p: plate_conf[p])]
        # 可信车牌：共识中出现 >=2 次、或置信度 >=0.95。单帧低置信读数
        # 不能参与跨组合并（否则按 ≤1 字符规则会把整个车队链式吞进来）。
        g["plates"] = {p: plate_conf[p] for p in consensus}
        g["plate_trusted"] = {
            p: (plate_counts.get(p, 0) >= 2 or plate_conf[p] >= 0.95)
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
        low, high = sorted((ra, rb))
        parent[high] = low
        ga, gb = groups[low], groups[high]
        for plate, conf in gb["plates"].items():
            ga["plates"][plate] = max(ga["plates"].get(plate, 0.0), conf)
        ga["dates"] |= gb["dates"]
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
            key = (park_key, plate[1:])
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
            key = plate[1:]
            if len(key) < 5:
                continue
            best_root, best_d = None, 99
            for root_anchor, anchor_plates in anchors.items():
                if root_anchor == find(gid):
                    continue
                if groups[root_anchor].get("park", "") != g.get("park", ""):
                    continue  # 只挂靠同园区的锚点
                d = min((sum(1 for x, y in zip(key, a) if x != y)
                         if len(key) == len(a) else 99) for a in anchor_plates)
                if d < best_d:
                    best_root, best_d = root_anchor, d
            if best_root is not None and best_d <= 1:
                union(find(gid), best_root, "相近车牌挂靠")
                anchors[find(best_root)].add(key)
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
    # 不再污染车牌检索与轨迹展示。----
    dominant: dict[int, str] = {}
    for vid in {v for v, _ in order_dom}:
        counter = Counter()
        for (v, _oid), plate in order_dom.items():
            if v == vid:
                counter[plate] += 1
        if counter:
            dominant[vid] = max(counter, key=lambda k: counter[k])
    cleaned = 0
    for vid, cid, plate in conn.execute(
        """SELECT vehicle_id, id, plate_final FROM captures
           WHERE vehicle_id IS NOT NULL AND coalesce(plate_final,'')!=''"""
    ).fetchall():
        dom = dominant.get(vid)
        if not dom:
            continue
        if _plate_edit_distance(plate, dom) >= 2:
            conn.execute(
                """UPDATE captures
                   SET plate_observed=coalesce(nullif(plate_observed,''), plate_final),
                       plate_final='',
                       plate_match_reason='单帧车牌与车辆主牌差异过大，判为误读'
                   WHERE id=?""",
                (cid,),
            )
            cleaned += 1
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

    linked = conn.execute(
        "SELECT COUNT(*) FROM captures WHERE vehicle_id IS NOT NULL"
    ).fetchone()[0]
    return {"orders": len(orders), "vehicle_groups": len(roots),
            "exact_plate_merges": exact_merges, "near_plate_merges": near_merges,
            "far_consensus_split": split, "frame_misread_cleaned": cleaned,
            "type_split": type_split, "manual_moves_applied": manual_applied,
            "linked_captures": linked}


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
                             "老布局照片根时作为其园区名，缺省为未标注")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--sync-only", action="store_true", help="只扫描目录，不跑OCR")
    parser.add_argument("--process-missing", action="store_true", help="补充OCR时间戳/OSD")
    parser.add_argument("--force-process", action="store_true", help="强制重跑OCR")
    parser.add_argument("--associate-only", action="store_true", help="只重新聚合订单内车辆")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    ensure_schema(conn)
    capture_count = conn.execute("SELECT COUNT(*) FROM captures").fetchone()[0]
    if not args.associate_only and (args.sync_only or capture_count == 0):
        if not args.root.is_dir():
            parser.error(f"照片目录不存在: {args.root}")
        summary = sync_files(conn, args.root, args.park)
        print(f"目录同步完成：{summary}")
    if args.process_missing or args.force_process or not (
        args.sync_only or args.associate_only
    ):
        process_missing(conn, args.limit, args.force_process)
    if not args.sync_only:
        print(f"订单内车辆聚合：{associate_orders(conn)}")
    print_stats(conn)
    conn.close()


if __name__ == "__main__":
    main()
