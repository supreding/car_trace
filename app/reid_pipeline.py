# -*- coding: utf-8 -*-
"""PaddleDetection 检测/裁剪 + 车辆属性 + FastReID 特征提取流水线。

训练/批处理环境使用此模块，Flask 只读取生成后的 crop 和 embedding。
两个上游项目均采用可选导入，未安装或未配置权重时会给出明确错误，不会静默
退回旧的通用分类特征。

输入是 build_db.py 建好的 granary.db；输出：
- reid_crops/{capture_id:08d}.jpg 车辆裁剪图
- captures 表的 vehicle_crop_path/vehicle_box/crop_source/detection_count
- captures 表的 vehicle_color(+conf)/vehicle_type(+conf)/attr_model_version
- captures 表的 reid_embedding/reid_source/reid_model_version
- order_attributes 表：订单级多数投票属性（用于展示降噪，不覆盖单图证据）
"""
from __future__ import annotations

import argparse
import collections
import collections.abc
import importlib.util
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from collections import Counter, defaultdict

import build_db

# FastReID 1.3 仍使用 Python 3.7 时代的 collections.Mapping 写法。
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, "Sequence"):
    collections.Sequence = collections.abc.Sequence


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PaddleVehicleCropper:
    """调用 PaddleDetection 导出的检测模型，返回整车框。"""

    def __init__(self, model_dir: Path, device: str = "GPU", threshold: float = 0.35):
        self.model_dir = Path(model_dir).resolve()
        if not (self.model_dir / "infer_cfg.yml").is_file():
            raise FileNotFoundError(f"PaddleDetection 模型目录缺少 infer_cfg.yml：{self.model_dir}")
        self.device = device.upper()
        self.threshold = threshold
        deploy_python = self.model_dir / "_paddledetection_deploy_python"
        # 支持通过环境变量指定 PaddleDetection 源码目录。
        source_root = os.environ.get("PADDLE_DETECTION_ROOT", "").strip()
        if source_root:
            deploy_python = Path(source_root).resolve() / "deploy" / "python"
        if not (deploy_python / "infer.py").is_file():
            raise FileNotFoundError(
                "找不到 PaddleDetection deploy/python/infer.py。"
                "请设置 PADDLE_DETECTION_ROOT 指向 PaddleDetection 源码目录。"
            )
        module = load_module_from_path("paddledetection_deploy_infer", deploy_python / "infer.py")
        # PP-Vehicle 2022 权重是 legacy model.pdmodel；Paddle 3.x 的部署脚本
        # 会误走 PIR 的 model.json 分支。部署 API 仍兼容旧格式，显式选择 legacy。
        import paddle
        if paddle.__version__.split(".")[0].isdigit() and int(paddle.__version__.split(".")[0]) >= 3:
            module.paddle.__version__ = "2.6.0"
        self.detector = module.Detector(
            model_dir=str(self.model_dir), device=self.device, batch_size=1,
            threshold=threshold,
        )
        labels = getattr(self.detector.pred_config, "labels", []) or []
        self.labels = [str(label).lower() for label in labels]

    def detect(self, image: Image.Image) -> list[dict[str, Any]]:
        import cv2

        rgb = np.asarray(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        result = self.detector.predict_image([bgr], visual=False)
        boxes = result.get("boxes") if isinstance(result, dict) else None
        if boxes is None:
            return []
        count = int(result.get("boxes_num", [len(boxes)])[0])
        found: list[dict[str, Any]] = []
        for row in np.asarray(boxes)[:count]:
            if len(row) < 6:
                continue
            class_id, score, x1, y1, x2, y2 = row[:6]
            score = float(score)
            if score < self.threshold:
                continue
            class_id = int(class_id)
            label = self.labels[class_id] if class_id < len(self.labels) else str(class_id)
            found.append({"class_id": class_id, "label": label, "score": score,
                          "box": [float(x1), float(y1), float(x2), float(y2)]})
        return sorted(found, key=lambda item: item["score"], reverse=True)

    @staticmethod
    def choose_main(detections: list[dict[str, Any]], width: int, height: int) -> dict[str, Any]:
        """主体车辆选择：置信度 × 面积 × 中心度 加权。

        过磅场景主体车辆位于画面中心且占比较大。单纯取最高分会选到画面
        边缘的部分/无关车辆（实测置信度 0.8 以下时尤甚），加入面积与
        中心距离两项后，边缘小框几乎不可能胜出。
        """
        import math

        best, best_score = detections[0], -1.0
        cx, cy = width / 2.0, height / 2.0
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            bx, by = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dist = math.hypot((bx - cx) / width, (by - cy) / height)
            area = max(0.0, (x2 - x1) * (y2 - y1)) / float(width * height)
            s = det["score"] * (1.0 - min(dist, 0.7)) * (area ** 0.25)
            if s > best_score:
                best, best_score = det, s
        return best

    @staticmethod
    def crop(image: Image.Image, detection: dict[str, Any], margin: float = 0.08) -> tuple[Image.Image, list[int]]:
        x1, y1, x2, y2 = detection["box"]
        width, height = image.size
        box_width, box_height = max(1.0, x2 - x1), max(1.0, y2 - y1)
        left = max(0, int(x1 - box_width * margin))
        top = max(0, int(y1 - box_height * margin))
        right = min(width, int(x2 + box_width * margin))
        bottom = min(height, int(y2 + box_height * margin))
        return image.crop((left, top, right, bottom)), [left, top, right, bottom]


class VehicleAttrClassifier:
    """PP-Vehicle 车辆属性模型（PULC PPLCNet，来自 PaddleClas）。

    预处理必须与 PaddleClas 官方训练配置一致
    （ppcls/configs/PULC/vehicle_attribute/PPLCNet_x1_0.yaml）：
    - DecodeImage to_rgb=True：输入是 RGB 通道顺序。按 deploy 流程的 BGR
      惯例喂图会让红/蓝完全互换（红色货车被识别成 blue，实测复现）；
    - ResizeImage size=[256, 192]，即宽 256、高 192（对应导出模型
      infer_cfg.yml 的 target_size [192, 256] = [h, w]，deploy 的 Resize
      按 (w, h) 展开）。宽高写反会让特征进一步退化；
    - NormalizeImage：先乘 1/255，再减 mean [0.485,0.456,0.406]、除
      std [0.229,0.224,0.225]，HWC 转 CHW。
    输出 10 种颜色 + 9 种车型及其置信度，颜色/车型各自 argmax 过阈值。
    """

    COLOR_LIST = ("yellow", "orange", "green", "gray", "red", "blue",
                  "white", "golden", "brown", "black")
    TYPE_LIST = ("sedan", "suv", "van", "hatchback", "mpv", "pickup",
                 "bus", "truck", "estate")

    def __init__(self, model_dir: Path, device: str = "GPU",
                 color_threshold: float = 0.5, type_threshold: float = 0.5):
        self.model_dir = Path(model_dir).resolve()
        if not (self.model_dir / "infer_cfg.yml").is_file():
            raise FileNotFoundError(f"车辆属性模型目录缺少 infer_cfg.yml：{self.model_dir}")
        self.color_threshold = color_threshold
        self.type_threshold = type_threshold
        self.model_version = f"paddledet:{self.model_dir.name}"
        import paddle
        from paddle.inference import Config, create_predictor

        config = Config(
            str(self.model_dir / "model.pdmodel"),
            str(self.model_dir / "model.pdiparams"),
        )
        if device.upper() == "GPU":
            config.enable_use_gpu(200, 0)
        else:
            config.disable_gpu()
            config.enable_mkldnn()
        if hasattr(config, "enable_new_ir"):
            config.enable_new_ir(False)  # legacy model.pdmodel
        self.predictor = create_predictor(config)
        input_names = self.predictor.get_input_names()
        self.input_name = input_names[0]
        self.output_name = self.predictor.get_output_names()[0]

    def predict(self, image: Image.Image) -> dict[str, Any]:
        import cv2

        # PULC 训练用 to_rgb=True：保持 RGB 顺序，不要转 BGR。
        rgb = np.asarray(image.convert("RGB"))
        # 官方 Resize（keep_ratio=false）：target_size [h, w] = [192, 256]。
        resized = cv2.resize(rgb, (256, 192), interpolation=cv2.INTER_LINEAR)
        scaled = resized.astype("float32") / 255.0
        scaled -= np.array([0.485, 0.456, 0.406], dtype="float32")
        scaled /= np.array([0.229, 0.224, 0.225], dtype="float32")
        chw = scaled.transpose(2, 0, 1)[np.newaxis, ...]

        handle = self.predictor.get_input_handle(self.input_name)
        handle.copy_from_cpu(chw)
        self.predictor.run()
        scores = self.predictor.get_output_handle(self.output_name).copy_to_cpu()
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        color_idx = int(np.argmax(scores[:10]))
        type_idx = int(np.argmax(scores[10:]))
        color = self.COLOR_LIST[color_idx] if scores[color_idx] >= self.color_threshold else ""
        vehicle_type = (self.TYPE_LIST[type_idx]
                        if scores[type_idx + 10] >= self.type_threshold else "")
        return {
            "color": color,
            "color_conf": round(float(scores[color_idx]), 4),
            "type": vehicle_type,
            "type_conf": round(float(scores[type_idx + 10]), 4),
        }


class FastReIDExtractor:
    """使用 FastReID DefaultPredictor 生成归一化 embedding。"""

    def __init__(self, config_file: Path, weights: Path, device: str = "cuda:0"):
        self.config_file = Path(config_file).resolve()
        self.weights = Path(weights).resolve()
        if not self.config_file.is_file():
            raise FileNotFoundError(f"FastReID 配置不存在：{self.config_file}")
        if not self.weights.is_file():
            raise FileNotFoundError(f"FastReID 权重不存在：{self.weights}")
        if importlib.util.find_spec("fastreid") is None:
            raise RuntimeError(
                "未安装 fastreid。请将 PYTHONPATH 指向 third_party/fast-reid 后运行。"
            )
        from fastreid.config import get_cfg
        from fastreid.engine import DefaultPredictor

        cfg = get_cfg()
        cfg.merge_from_file(str(self.config_file))
        cfg.MODEL.WEIGHTS = str(self.weights)
        cfg.MODEL.DEVICE = device
        cfg.freeze()
        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
        self.model_version = f"fastreid:{self.weights.name}"

    def extract(self, image: Image.Image) -> np.ndarray:
        import cv2
        import torch

        # FastReID 的 DefaultPredictor 文档示例仍写 BGR ndarray，但当前实现实际
        # 接收已经 resize 的 NCHW torch.Tensor；这里与官方 demo/predictor.py
        # 保持一致，并转换为模型训练使用的 RGB 顺序。
        rgb = np.asarray(image.convert("RGB"))
        test_height, test_width = self.cfg.INPUT.SIZE_TEST
        resized = cv2.resize(rgb, (test_width, test_height), interpolation=cv2.INTER_CUBIC)
        tensor = torch.as_tensor(
            resized.astype("float32").transpose(2, 0, 1)
        ).unsqueeze(0)
        output = self.predictor(tensor)
        if isinstance(output, dict):
            if "features" in output:
                output = output["features"]
            elif "embedding" in output:
                output = output["embedding"]
            else:
                output = next(iter(output.values()))
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        feature = np.asarray(output, dtype=np.float32).reshape(-1)
        if not feature.size:
            raise RuntimeError("FastReID 没有返回 embedding")
        return feature / (np.linalg.norm(feature) + 1e-9)


def aggregate_order_attributes(db: sqlite3.Connection) -> int:
    """订单级属性多数投票：降噪用，写入 order_attributes，不覆盖单图证据。

    按 (park, order_id) 分组：不同园区可能使用同一套系统导致订单号撞号，
    投票只在园区内进行。表结构（含老表迁移）由 build_db.ensure_order_attributes_schema
    统一负责。
    """
    build_db.ensure_order_attributes_schema(db)
    rows = db.execute(
        """SELECT coalesce(park,'') AS park, order_id, vehicle_color, vehicle_color_conf,
                  vehicle_type, vehicle_type_conf
           FROM captures
           WHERE coalesce(order_id,'')!='' AND coalesce(capture_status,'vehicle')!='no_vehicle'
             AND (coalesce(vehicle_color,'')!='' OR coalesce(vehicle_type,'')!='')"""
    ).fetchall()
    by_order: dict[tuple[str, str], list[tuple]] = {}
    for row in rows:
        by_order.setdefault((row[0], row[1]), []).append(row[2:])
    now = datetime.now().isoformat(timespec="seconds")
    updated = 0
    for (park, order_id), items in by_order.items():
        colors: dict[str, float] = {}
        types: dict[str, float] = {}
        for color, color_conf, vtype, type_conf in items:
            if color:
                colors[color] = colors.get(color, 0.0) + max(0.05, float(color_conf or 0))
            if vtype:
                types[vtype] = types.get(vtype, 0.0) + max(0.05, float(type_conf or 0))
        best_color = max(colors, key=lambda k: colors[k]) if colors else ""
        best_type = max(types, key=lambda k: types[k]) if types else ""
        color_conf = round(colors[best_color] / len(items), 4) if best_color else 0.0
        type_conf = round(types[best_type] / len(items), 4) if best_type else 0.0
        db.execute(
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
            (park, order_id, best_color, color_conf, best_type, type_conf,
             len(items), now),
        )
        updated += 1
    db.commit()
    return updated


def detect_pass(
    db_path: Path,
    output_dir: Path,
    detector: "PaddleVehicleCropper",
    attr: "VehicleAttrClassifier | None",
    limit: int = 0,
    force: bool = False,
) -> dict[str, int]:
    """阶段一（PaddleDetection）：检测、裁剪、车辆属性。

    与 FastReID 分进程运行：GPU 版 torch 与 paddlepaddle-gpu 各自捆绑不同
    小版本的 cuDNN DLL，同进程加载必然冲突（WinError 127），拆开后各自完整
    可用。本阶段不 import torch。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # OCR/服务可能并发访问同一数据库：自动提交 + 放宽锁等待，避免长事务互顶。
    db = sqlite3.connect(db_path, timeout=60)
    db.isolation_level = None
    db.row_factory = sqlite3.Row
    # 新增列（如 vehicle_det_score）由 schema 迁移保证，不依赖手工执行 build_db。
    build_db.ensure_schema(db)
    if force:
        db.execute("DELETE FROM vehicle_instances")
    # 断点续跑：有 instance 记录的图片视为已处理（每车一条记录）；
    # 已被属性复核拆出/标记的帧也不再被检测重跑覆盖（复核结论优先）。
    where = ("coalesce(capture_status,'vehicle')!='no_vehicle' "
             "AND (status_reason IS NULL OR status_reason NOT LIKE '%与订单主车不符%') "
             "AND id NOT IN (SELECT DISTINCT capture_id FROM vehicle_instances)")
    if not force:
        # 已有裁剪图的帧是早期版本的检测产物，不可能再产生实例记录；
        # 不排除会永久挂起在待处理队列里（每轮 GPU 都被重选又跳过）。
        where += " AND vehicle_crop_path IS NULL"
    sql = (f"SELECT id, filepath FROM captures WHERE {where} "
           "ORDER BY date, order_id, stage_rank, filename")
    params: tuple[Any, ...] = ()
    if limit:
        sql += " LIMIT ?"
        params = (limit,)
    rows = db.execute(sql, params).fetchall()
    processed = no_vehicle = skipped = 0
    print(
        f"[detect] 开始处理 {len(rows)} 张抓拍：PaddleDetection 检测/裁剪 -> 车辆属性；"
        "未检测到车辆的图片会标记 no_vehicle 并跳过。",
        flush=True,
    )
    for row in rows:
        capture_id = int(row["id"])
        source = build_db.resolve_stored_path(row["filepath"])
        if not source.is_file():
            # 原始照片缺失（归档清理/复制遗漏）是永久状态：标记后不再进入
            # 待处理队列，否则该帧每轮 GPU 都被选中又跳过，待处理数永不清零。
            db.execute(
                """UPDATE captures SET capture_status='no_vehicle',
                          status_reason='原始照片缺失，无法检测'
                   WHERE id=?""",
                (capture_id,),
            )
            skipped += 1
            continue
        try:
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            detections = detector.detect(image)
            db.execute("UPDATE captures SET detection_count=? WHERE id=?",
                       (len(detections), capture_id))
            if not detections:
                db.execute(
                    """UPDATE captures SET capture_status='no_vehicle',
                              status_reason='PaddleDetection 未检测到车辆',
                              vehicle_crop_path=NULL, reid_embedding=NULL,
                              reid_source=NULL, crop_source=NULL,
                              vehicle_det_score=NULL
                       WHERE id=?""",
                    (capture_id,),
                )
                no_vehicle += 1
                continue
            chosen = detector.choose_main(detections, image.size[0], image.size[1])
            width, height = image.size
            crop, box = detector.crop(image, chosen)
            crop_path = output_dir / f"{capture_id:08d}.jpg"
            crop.save(crop_path, "JPEG", quality=94, optimize=True)
            attr_values: list[Any] = [None, None, None, None, None]
            if attr is not None:
                attr_result = attr.predict(crop)
                attr_values = [
                    attr_result["color"] or None, attr_result["color_conf"],
                    attr_result["type"] or None, attr_result["type_conf"],
                    attr.model_version,
                ]
            # 记录相对于数据库目录的路径，便于把 app 目录整体复制到甲方环境。
            stored_crop_path = Path(
                os.path.relpath(crop_path.resolve(), Path(db_path).resolve().parent)
            ).as_posix()
            db.execute(
                """UPDATE captures SET vehicle_crop_path=?, vehicle_box=?, crop_source=?,
                           vehicle_det_score=?, detection_count=detection_count,
                           vehicle_color=coalesce(?, vehicle_color),
                           vehicle_color_conf=coalesce(?, vehicle_color_conf),
                           vehicle_type=coalesce(?, vehicle_type),
                           vehicle_type_conf=coalesce(?, vehicle_type_conf),
                           attr_model_version=coalesce(?, attr_model_version),
                           capture_status='vehicle',
                           processed_at=?
                   WHERE id=?""",
                (stored_crop_path, json.dumps(box), "PaddleDetection",
                 float(chosen["score"]),
                 *attr_values,
                 datetime.now().isoformat(timespec="seconds"), capture_id),
            )
            # ---- 每车一条记录：同帧其余检测目标各自建档（无关车辆由此分开）。
            # 主体车辆为 instance_index=0，其余按分数取至多 5 个
            # （分数 >=0.5 且面积 >=2%，过滤边缘碎片框）。
            main_box = chosen["box"]
            instance_index = 0
            db.execute(
                """INSERT INTO vehicle_instances
                     (capture_id, instance_index, crop_path, vehicle_box, vehicle_det_score,
                      vehicle_color, vehicle_color_conf, vehicle_type, vehicle_type_conf,
                      attr_model_version)
                   VALUES (?, 0, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (capture_id, stored_crop_path, json.dumps(box), float(chosen["score"]),
                 *attr_values),
            )
            for det in detections:
                if det is chosen:
                    continue
                if det["score"] < 0.5:
                    continue
                x1, y1, x2, y2 = det["box"]
                if (x2 - x1) * (y2 - y1) < width * height * 0.02:
                    continue
                if instance_index >= 5:
                    break
                instance_index += 1
                inst_crop, inst_box = detector.crop(image, det)
                inst_path = output_dir / f"{capture_id}_{instance_index:02d}.jpg"
                inst_crop.save(inst_path, "JPEG", quality=94, optimize=True)
                inst_stored = Path(
                    os.path.relpath(inst_path.resolve(), Path(db_path).resolve().parent)
                ).as_posix()
                inst_attr: list[Any] = [None, None, None, None, None]
                if attr is not None:
                    inst_result = attr.predict(inst_crop)
                    inst_attr = [
                        inst_result["color"] or None, inst_result["color_conf"],
                        inst_result["type"] or None, inst_result["type_conf"],
                        attr.model_version,
                    ]
                db.execute(
                    """INSERT INTO vehicle_instances
                         (capture_id, instance_index, crop_path, vehicle_box, vehicle_det_score,
                          vehicle_color, vehicle_color_conf, vehicle_type, vehicle_type_conf,
                          attr_model_version)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (capture_id, instance_index, inst_stored, json.dumps(inst_box),
                     float(det["score"]), *inst_attr),
                )
            processed += 1
            if processed % 50 == 0:
                db.commit()
                print(f"[detect] 已处理 {processed}/{len(rows)}", flush=True)
        except Exception as exc:  # 单张失败不影响断点续跑
            skipped += 1
            print(f"[detect] 跳过抓拍 #{capture_id}: {exc}", flush=True)
    db.commit()
    db.close()
    return {"selected": len(rows), "cropped": processed,
            "no_vehicle": no_vehicle, "skipped": skipped}


def attr_pass(
    db_path: Path,
    attr: "VehicleAttrClassifier",
    limit: int = 0,
    force: bool = False,
) -> dict[str, int]:
    """仅重算车辆属性：读取已生成的裁剪图，不重新检测。

    预处理规则变更（如 RGB/宽高修正）后用它快速刷新 4091 张已有裁剪，
    避免重跑耗时的 PaddleDetection 阶段；完成后重建订单级属性投票。
    """
    db = sqlite3.connect(db_path, timeout=60)
    db.isolation_level = None
    db.row_factory = sqlite3.Row
    where = "vehicle_crop_path IS NOT NULL"
    if not force:
        where += " AND attr_model_version IS NULL"
    sql = f"SELECT id, vehicle_crop_path FROM captures WHERE {where} ORDER BY id"
    params: tuple[Any, ...] = ()
    if limit:
        sql += " LIMIT ?"
        params = (limit,)
    rows = db.execute(sql, params).fetchall()
    processed = skipped = 0
    print(f"[attr] 开始处理 {len(rows)} 张裁剪：车辆属性（颜色/车型+置信度）。", flush=True)
    for row in rows:
        capture_id = int(row["id"])
        crop_path = resolve_crop_path(db_path, row["vehicle_crop_path"])
        if not crop_path.is_file():
            skipped += 1
            continue
        try:
            with Image.open(crop_path) as opened:
                crop = ImageOps.exif_transpose(opened).convert("RGB")
            result = attr.predict(crop)
            db.execute(
                """UPDATE captures SET vehicle_color=?, vehicle_color_conf=?,
                          vehicle_type=?, vehicle_type_conf=?, attr_model_version=?
                   WHERE id=?""",
                (result["color"] or None, result["color_conf"],
                 result["type"] or None, result["type_conf"],
                 attr.model_version, capture_id),
            )
            processed += 1
            if processed % 200 == 0:
                print(f"[attr] 已处理 {processed}/{len(rows)}", flush=True)
        except Exception as exc:
            skipped += 1
            print(f"[attr] 跳过抓拍 #{capture_id}: {exc}", flush=True)
    order_count = aggregate_order_attributes(db)
    db.close()
    print(f"订单级属性投票完成：{order_count} 个订单", flush=True)
    return {"selected": len(rows), "processed": processed, "skipped": skipped}


class PlateLPR:
    """PP-Vehicle 官方车牌识别（ch_PP-OCRv3_det + ch_PP-OCRv3_rec）。

    复用 third_party/PaddleDetection/deploy/pipeline/ppvehicle 的官方实现
    （vehicle_plate.PlateDetector/TextRecognizer、DB 后处理、透视裁剪），
    仅去掉命令行参数外壳。输入是第 8 步的车辆裁剪图（BGR ndarray），
    输出中文车牌文本与识别置信度（不做官方的省份转拼音前缀，便于入库比对）。
    """

    def __init__(self, det_model_dir: Path, rec_model_dir: Path,
                 word_dict_path: Path, device: str = "GPU"):
        self.det_model_dir = Path(det_model_dir).resolve()
        self.rec_model_dir = Path(rec_model_dir).resolve()
        for path in (self.det_model_dir, self.rec_model_dir):
            if not (path / "inference.pdmodel").is_file():
                raise FileNotFoundError(f"车牌模型缺少 inference.pdmodel：{path}")
        if not Path(word_dict_path).is_file():
            raise FileNotFoundError(f"车牌识别字典不存在：{word_dict_path}")
        self.model_version = "paddledet:ch_PP-OCRv3_plate"

        plate_root = Path(word_dict_path).resolve().parent
        deploy_root = plate_root.parent.parent  # deploy/pipeline/ppvehicle -> deploy
        import sys
        if str(deploy_root) not in sys.path:
            sys.path.insert(0, str(deploy_root))
        from pipeline.ppvehicle import vehicle_plate as vp

        args = argparse.Namespace(
            device=device.upper(), run_mode="paddle", trt_min_shape=1,
            trt_max_shape=1280, trt_opt_shape=640, trt_calib_mode=False,
            cpu_threads=1, enable_mkldnn=False,
        )
        cfg = {
            "det_model_dir": str(self.det_model_dir),
            "rec_model_dir": str(self.rec_model_dir),
            # 车牌裁剪图普遍较小（车牌仅20~40像素），官方默认 736/min 放大不足
            # 会大面积漏检；提到 1280 后小图放大更充分（limit_type=min 不缩大图）。
            "det_limit_side_len": 1280,
            "det_limit_type": "min",
            "rec_image_shape": [3, 48, 320],
            "rec_batch_num": 6,
            "word_dict_path": str(word_dict_path),
        }
        self.detector = vp.PlateDetector(args, cfg)
        self.recognizer = vp.TextRecognizer(args, cfg, use_gpu=device.upper() == "GPU")
        self._get_rotate_crop_image = vp.get_rotate_crop_image
        # 官方 2022 代码仍使用 np.int 别名（NumPy 1.24 起移除）。
        if not hasattr(np, "int"):
            np.int = int

    def recognize(self, bgr: "np.ndarray") -> dict[str, Any]:
        """识别一张车辆裁剪图中的车牌，返回最佳候选。"""
        boxes_batch, _ = self.detector.predict_image([bgr])
        boxes = boxes_batch[0] if boxes_batch else []
        candidates: list[tuple[str, float]] = []
        for box in boxes:
            try:
                crop = self._get_rotate_crop_image(bgr, box)
                rec_res, _ = self.recognizer.predict_text([crop])
                for text, conf in rec_res:
                    text, conf = str(text), float(conf)
                    if 3 <= len(text) <= 9 and conf > 0:
                        candidates.append((text, conf))
            except Exception:
                continue
        if not candidates:
            return {"plate": "", "conf": 0.0, "n_boxes": len(boxes)}
        # 优先 7 位标准车牌长度，其次置信度。
        candidates.sort(key=lambda t: (len(t[0]) >= 7, t[1]), reverse=True)
        text, conf = candidates[0]
        return {"plate": text, "conf": round(conf, 4), "n_boxes": len(boxes)}


def plate_pass(
    db_path: Path,
    lpr: PlateLPR,
    limit: int = 0,
    force: bool = False,
) -> dict[str, int]:
    """车牌识别阶段：对车辆裁剪图跑 PP-OCR det+rec，并与 OSD 参考值融合。

    融合规则（实施文档 13.1：模型结果是主证据，摄像头叠字只作参考）：
    - 车牌模型结果为主：plate_final=lpr_plate、recognition_source='车牌模型'；
    - 与 OSD 一致时置信度加成，来源记 'OSD+车牌模型一致'；
    - 模型未识别时保留原 OSD 参考值（若已有）。
    """
    import cv2

    db = sqlite3.connect(db_path, timeout=60)
    db.isolation_level = None
    db.row_factory = sqlite3.Row
    where = "vehicle_crop_path IS NOT NULL"
    if not force:
        # lpr_attempted_at：识别"已尝试"标记。模型读不出车牌的裁剪图也要落
        # 标记，否则这些行永远满足 lpr_plate IS NULL，GPU 流水线每轮重复
        # 识别上万张无牌图（服务器实测每 30 分钟空转一轮的根源）。
        where += " AND lpr_attempted_at IS NULL"
    sql = (f"SELECT id, vehicle_crop_path, coalesce(osd_plate,'') AS osd "
           f"FROM captures WHERE {where} ORDER BY id")
    params: tuple[Any, ...] = ()
    if limit:
        sql += " LIMIT ?"
        params = (limit,)
    rows = db.execute(sql, params).fetchall()
    processed = skipped = 0
    print(f"[plate] 开始处理 {len(rows)} 张裁剪：PP-OCR 车牌检测+识别。", flush=True)
    for row in rows:
        capture_id = int(row["id"])
        crop_path = resolve_crop_path(db_path, row["vehicle_crop_path"])
        if not crop_path.is_file():
            # 裁剪文件缺失是永久状态（归档清理/复制遗漏），落标记防止每轮重选。
            db.execute("UPDATE captures SET lpr_attempted_at=? WHERE id=?",
                       (datetime.now().isoformat(timespec="seconds"), capture_id))
            skipped += 1
            continue
        try:
            data = np.fromfile(str(crop_path), dtype=np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                db.execute("UPDATE captures SET lpr_attempted_at=? WHERE id=?",
                           (datetime.now().isoformat(timespec="seconds"), capture_id))
                skipped += 1
                continue
            result = lpr.recognize(bgr)
            lpr_plate = build_db.normalize_plate(result["plate"])
            lpr_conf = result["conf"]
            # 质量闸：小图远距离车牌容易出碎片（如“皖25”）或低置信度噪声，
            # 归一化后不足 5 位或置信度过低的不作为模型结果。
            if len(lpr_plate) < 5 or lpr_conf < 0.55:
                lpr_plate, lpr_conf = "", 0.0
            osd_plate = str(row["osd"] or "")
            if lpr_plate:
                if osd_plate and osd_plate == lpr_plate:
                    source = "OSD+车牌模型一致"
                    lpr_conf = min(0.99, lpr_conf + 0.05)
                else:
                    source = "车牌模型"
                plate_final = lpr_plate
            else:
                plate_final = osd_plate
                source = "OSD文字" if osd_plate else ""
            db.execute(
                """UPDATE captures SET lpr_plate=?, lpr_conf=?,
                          plate_final=?, plate_confidence=?,
                          lpr_attempted_at=?,
                          recognition_source=CASE
                            WHEN ?!='' THEN ? ELSE recognition_source END
                   WHERE id=?""",
                (lpr_plate or None, lpr_conf, plate_final,
                 lpr_conf if lpr_plate else None,
                 datetime.now().isoformat(timespec="seconds"),
                 source, source, capture_id),
            )
            processed += 1
            if processed % 200 == 0:
                print(f"[plate] 已处理 {processed}/{len(rows)}", flush=True)
        except Exception as exc:
            skipped += 1
            print(f"[plate] 跳过抓拍 #{capture_id}: {exc}", flush=True)
    db.close()
    return {"selected": len(rows), "processed": processed, "skipped": skipped}


def correct_pass(
    db_path: Path,
    detector: "PaddleVehicleCropper",
    attr: "VehicleAttrClassifier",
    limit: int = 0,
) -> dict[str, int]:
    """属性一致性复核：用车辆组级颜色/车型共识回检可疑选框。

    共识取自车辆组（跨订单）内全部有效帧的多数投票——单订单投票会被
    混入帧稀释（10 帧里 5 红 3 黑 2 白 -> 置信度 0.4 被门槛挡住），而
    车辆组跨 12 个订单的投票（红 90/114）可靠得多。

    触发条件（组共识样本 >=2 且多数占比 >=0.5）：
    - 选框颜色与共识颜色冲突（如红色货车订单某帧选到蓝色三轮车）；
    - 或车型识别置信度 < 0.7（车型半信半疑说明选框可疑）。
    处理：以更低阈值（0.2）重检原图，对每个候选框跑属性模型，选与共识
    最匹配（颜色一致优先、其次车型一致）的框替换原裁剪；帧内确实没有
    主车、且当前选框车型也与共识不符的，拆出车辆轨迹待人工复核。
    替换后重置 lpr_plate 与 reid_embedding，由后续 plate/embed 阶段
    按断点续跑机制增量补算。
    """
    import cv2

    db = sqlite3.connect(db_path, timeout=60)
    db.isolation_level = None
    db.row_factory = sqlite3.Row
    build_db.ensure_schema(db)

    # 组级共识：车辆组内全部有效帧的多数颜色/车型（多数占比 >=0.5）
    consensuses: dict[int, dict] = {}
    votes = defaultdict(lambda: {"colors": Counter(), "types": Counter(), "n": 0})
    for vid, color, vtype in db.execute(
        """SELECT vehicle_id, vehicle_color, vehicle_type FROM captures
           WHERE vehicle_id IS NOT NULL
             AND coalesce(capture_status,'vehicle')!='no_vehicle'"""
    ):
        votes[vid]["n"] += 1
        if color:
            votes[vid]["colors"][color] += 1
        if vtype:
            votes[vid]["types"][vtype] += 1
    for vid, v in votes.items():
        if v["n"] < 2 or not v["types"]:
            continue
        dom_type, type_n = max(v["types"].items(), key=lambda kv: kv[1])
        dom_color = (max(v["colors"].items(), key=lambda kv: kv[1])[0]
                     if v["colors"] else "")
        if type_n / v["n"] < 0.5:
            continue
        consensuses[vid] = {"vehicle_type": dom_type,
                            "vehicle_type_conf": type_n / v["n"],
                            "vehicle_color": dom_color,
                            "sample_count": v["n"]}

    # 可疑帧：所属车辆组有共识，且（颜色与共识冲突 或 车型置信度 <0.7 且
    # 车型与共识不符）。注意不依赖 vehicle_id——被拆出的行（vehicle_id
    # 为空）同样要能被复核修正；组共识通过订单兄弟行回溯。
    # 订单->车辆组映射按 (park, order_id) 建键：不同园区可能撞号。
    order_group: dict[tuple[str, str], int] = {}
    for park, oid, vid in db.execute(
        """SELECT DISTINCT coalesce(park,'') AS park, order_id, vehicle_id FROM captures
           WHERE vehicle_id IS NOT NULL AND coalesce(order_id,'')!=''"""
    ):
        order_group.setdefault((park, oid), vid)
    suspicious = []
    for row in db.execute(
        """SELECT id, filepath, coalesce(park,'') AS park, order_id, vehicle_crop_path,
                  vehicle_color, vehicle_type, vehicle_type_conf, vehicle_id, status_reason
           FROM captures
           WHERE vehicle_crop_path IS NOT NULL
             AND coalesce(capture_status,'vehicle')!='no_vehicle'
             AND coalesce(order_id,'')!='' AND order_id!='未关联图片'
           ORDER BY id"""
    ).fetchall():
        vid = order_group.get((row["park"], row["order_id"]))
        consensus = consensuses.get(vid) if vid else None
        if not consensus:
            continue
        color_conflict = (row["vehicle_color"] and consensus["vehicle_color"]
                          and row["vehicle_color"] != consensus["vehicle_color"])
        type_suspect = (row["vehicle_type_conf"] is not None
                        and row["vehicle_type_conf"] < 0.7)
        if color_conflict or type_suspect:
            suspicious.append(row)
    rows = suspicious[::] if not limit else suspicious[:limit]

    processed = replaced = flagged = unchanged = skipped = 0
    print(f"[correct] 属性复核 {len(rows)} 张可疑选框（重检阈值 0.2）。", flush=True)
    for row in rows:
        capture_id = int(row["id"])
        consensus = consensuses.get(order_group.get((row["park"], row["order_id"])))
        if not consensus:
            skipped += 1
            continue
        try:
            source = build_db.resolve_stored_path(row["filepath"])
            if not source.is_file():
                skipped += 1
                continue
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            width, height = image.size
            detections = detector.detect(image)
            best, best_score = None, -1.0
            for det in detections[:6]:
                x1, y1, x2, y2 = det["box"]
                if (x2 - x1) * (y2 - y1) < width * height * 0.02:
                    continue  # 过小的碎片框不参与
                crop_c, _ = detector.crop(image, det)
                a = attr.predict(crop_c)
                match = 0.0
                if consensus["vehicle_color"] and a["color"] == consensus["vehicle_color"]:
                    match += 2.0
                if consensus["vehicle_type"] and a["type"] == consensus["vehicle_type"]:
                    match += 1.0
                match += a["type_conf"] * 0.25
                if match > best_score:
                    best_score, best = match, (det, a, crop_c)
            if best is None:
                skipped += 1
                continue
            det, a, crop_c = best
            color_match = bool(consensus["vehicle_color"] and a["color"] == consensus["vehicle_color"])
            type_match = bool(consensus["vehicle_type"] and a["type"] == consensus["vehicle_type"])
            current_box = json.loads(
                db.execute("SELECT vehicle_box FROM captures WHERE id=?", (capture_id,)).fetchone()[0]
            )
            box_changed = max(abs(x - y) for x, y in zip(current_box, det["box"])) > 15
            # 拆出判定看车型而非颜色：红色货车从车尾拍，货箱常被读成
            # white/truck——车型与共识一致（truck）就保留在轨迹里；
            # 车型不符（如 suv/sedan）且找不到更优候选的，拆出待人工复核。
            current_type_matches = (row["vehicle_type"] is not None
                                    and consensus["vehicle_type"] == row["vehicle_type"])
            if color_match and box_changed:
                # 更换选框：写新裁剪与属性，重置下游（车牌/特征）等待补算。
                crop_path = resolve_crop_path(db_path, row["vehicle_crop_path"])
                crop_c.save(crop_path, "JPEG", quality=94, optimize=True)
                sib = db.execute(
                    """SELECT vehicle_id FROM captures
                       WHERE coalesce(park,'')=? AND order_id=? AND vehicle_id IS NOT NULL
                       LIMIT 1""",
                    (row["park"], row["order_id"]),
                ).fetchone()
                db.execute(
                    """UPDATE captures SET vehicle_box=?, vehicle_det_score=?,
                              vehicle_color=?, vehicle_color_conf=?,
                              vehicle_type=?, vehicle_type_conf=?,
                              lpr_plate=NULL, lpr_conf=NULL, plate_final=NULL,
                              plate_confidence=NULL, recognition_source='',
                              reid_embedding=NULL, reid_source=NULL,
                              vehicle_id=?,
                          status_reason=CASE WHEN status_reason LIKE '%订单主车%'
                                             THEN '' ELSE status_reason END
                       WHERE id=?""",
                    (json.dumps(det["box"]), float(det["score"]),
                     a["color"] or None, a["color_conf"],
                     a["type"] or None, a["type_conf"],
                     sib["vehicle_id"] if sib else None, capture_id),
                )
                replaced += 1
            elif color_match and not box_changed:
                unchanged += 1  # 当前框已与共识一致，仅车型置信度低，无需换框
            elif current_type_matches and not (a["type"] and type_match and box_changed):
                unchanged += 1  # 当前框车型与共识一致（如白色货箱车尾帧），保留
                # 此前被误拆/误标的帧：恢复订单车辆的归属并清除标记
                if row["vehicle_id"] is None or (row["status_reason"] or "").find("订单主车") >= 0:
                    sib = db.execute(
                        """SELECT vehicle_id FROM captures
                           WHERE coalesce(park,'')=? AND order_id=? AND vehicle_id IS NOT NULL
                           LIMIT 1""",
                        (row["park"], row["order_id"]),
                    ).fetchone()
                    db.execute(
                        """UPDATE captures SET vehicle_id=?, status_reason=''
                           WHERE id=?""",
                        (sib["vehicle_id"] if sib else None, capture_id),
                    )
            else:
                # 帧内没有订单主车（选到的是无关车辆）：拆出车辆轨迹，待人工复核。
                db.execute(
                    """UPDATE captures SET vehicle_id=NULL,
                              status_reason='帧内未检测到订单主车（画面为其他车辆），已拆出待人工复核'
                       WHERE id=?""",
                    (capture_id,),
                )
                flagged += 1
            processed += 1
            if processed % 100 == 0:
                print(f"[correct] 已处理 {processed}/{len(rows)}", flush=True)
        except Exception as exc:
            skipped += 1
            print(f"[correct] 跳过抓拍 #{capture_id}: {exc}", flush=True)
    aggregate_order_attributes(db)
    db.close()
    return {"selected": len(rows), "processed": processed, "replaced": replaced,
            "flagged": flagged, "unchanged": unchanged, "skipped": skipped}


def instance_plate_pass(db_path: Path, lpr: PlateLPR, limit: int = 0) -> dict[str, int]:
    """同帧其他车辆实例的车牌识别（vehicle_instances 表）。"""
    import cv2

    db = sqlite3.connect(db_path, timeout=60)
    db.isolation_level = None
    rows = db.execute(
        """SELECT id, crop_path FROM vehicle_instances
           WHERE (lpr_plate IS NULL OR lpr_plate='') AND lpr_attempted_at IS NULL
           ORDER BY id"""
        + (" LIMIT ?" if limit else ""),
        () if not limit else (limit,),
    ).fetchall()
    processed = skipped = 0
    if rows:
        print(f"[plate-inst] 同帧车辆实例车牌识别 {len(rows)} 条。", flush=True)
    for iid, crop_rel in rows:
        crop_path = resolve_crop_path(db_path, crop_rel)
        if not crop_path.is_file():
            db.execute("UPDATE vehicle_instances SET lpr_attempted_at=? WHERE id=?",
                       (datetime.now().isoformat(timespec="seconds"), iid))
            skipped += 1
            continue
        try:
            data = np.fromfile(str(crop_path), dtype=np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                db.execute("UPDATE vehicle_instances SET lpr_attempted_at=? WHERE id=?",
                           (datetime.now().isoformat(timespec="seconds"), iid))
                skipped += 1
                continue
            result = lpr.recognize(bgr)
            plate = build_db.normalize_plate(result["plate"])
            conf = result["conf"]
            if len(plate) < 5 or conf < 0.55:
                plate, conf = "", 0.0
            db.execute(
                "UPDATE vehicle_instances SET lpr_plate=?, lpr_conf=?, lpr_attempted_at=? WHERE id=?",
                (plate or None, conf, datetime.now().isoformat(timespec="seconds"), iid),
            )
            processed += 1
        except Exception as exc:
            skipped += 1
            print(f"[plate-inst] 跳过实例 #{iid}: {exc}", flush=True)
    db.close()
    return {"selected": len(rows), "processed": processed, "skipped": skipped}


def embed_pass(
    db_path: Path,
    extractor: "FastReIDExtractor",
    limit: int = 0,
    force: bool = False,
) -> dict[str, int]:
    """阶段二（FastReID）：对已有裁剪图提取 embedding。本阶段不 import paddle。"""
    db = sqlite3.connect(db_path, timeout=60)
    db.isolation_level = None
    db.row_factory = sqlite3.Row
    where = ("coalesce(capture_status,'vehicle')!='no_vehicle' "
             "AND vehicle_crop_path IS NOT NULL")
    where += " AND reid_embedding IS NULL" if not force else ""
    sql = f"SELECT id, vehicle_crop_path FROM captures WHERE {where} ORDER BY id"
    params: tuple[Any, ...] = ()
    if limit:
        sql += " LIMIT ?"
        params = (limit,)
    rows = db.execute(sql, params).fetchall()
    processed = skipped = 0
    print(f"[embed] 开始处理 {len(rows)} 张裁剪：FastReID 提取归一化特征。", flush=True)
    for row in rows:
        capture_id = int(row["id"])
        crop_path = resolve_crop_path(db_path, row["vehicle_crop_path"])
        if not crop_path.is_file():
            skipped += 1
            continue
        try:
            with Image.open(crop_path) as opened:
                crop = ImageOps.exif_transpose(opened).convert("RGB")
            feature = extractor.extract(crop)
            db.execute(
                """UPDATE captures SET reid_embedding=?, reid_source=?, reid_model_version=?,
                          processed_at=?
                   WHERE id=?""",
                (feature.astype(np.float32).tobytes(), "FastReID", extractor.model_version,
                 datetime.now().isoformat(timespec="seconds"), capture_id),
            )
            processed += 1
            if processed % 100 == 0:
                db.commit()
                print(f"[embed] 已处理 {processed}/{len(rows)}", flush=True)
        except Exception as exc:
            skipped += 1
            print(f"[embed] 跳过抓拍 #{capture_id}: {exc}", flush=True)

    # ---- 同帧其他车辆实例的特征（每车一条记录）----
    inst_rows = db.execute(
        """SELECT id, crop_path FROM vehicle_instances
           WHERE reid_embedding IS NULL ORDER BY id"""
    ).fetchall()
    inst_processed = 0
    if inst_rows:
        print(f"[embed-inst] 同帧车辆实例特征提取 {len(inst_rows)} 条。", flush=True)
    for iid, crop_rel in inst_rows:
        crop_path = resolve_crop_path(db_path, crop_rel)
        if not crop_path.is_file():
            continue
        try:
            with Image.open(crop_path) as opened:
                crop = ImageOps.exif_transpose(opened).convert("RGB")
            feature = extractor.extract(crop)
            db.execute(
                """UPDATE vehicle_instances
                   SET reid_embedding=?, reid_source=?, reid_model_version=?
                   WHERE id=?""",
                (feature.astype(np.float32).tobytes(), "FastReID",
                 extractor.model_version, iid),
            )
            inst_processed += 1
        except Exception as exc:
            print(f"[embed-inst] 跳过实例 #{iid}: {exc}", flush=True)
    db.commit()
    order_count = aggregate_order_attributes(db)
    db.close()
    print(f"订单级属性投票完成：{order_count} 个订单", flush=True)
    return {"selected": len(rows), "processed": processed,
            "instances": inst_processed, "skipped": skipped}


def resolve_crop_path(db_path: Path, stored: str) -> Path:
    path = Path(stored)
    if path.is_absolute():
        return path
    return (Path(db_path).resolve().parent / path)


def main() -> None:
    parser = argparse.ArgumentParser(description="PaddleDetection检测裁剪+车辆属性+FastReID特征")
    parser.add_argument("--db", type=Path, default=Path(__file__).with_name("granary.db"))
    parser.add_argument("--detector-model", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "models/paddledetection/mot_ppyoloe_l_36e_ppvehicle")
    parser.add_argument("--attr-model", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "models/paddledetection/vehicle_attribute")
    parser.add_argument("--fastreid-config", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "third_party/fast-reid/configs/VeRi/sbs_R50-ibn.yml")
    parser.add_argument("--fastreid-weights", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "models/fastreid/veri_sbs_R50-ibn.pth")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).with_name("reid_crops"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--paddle-device", default="GPU", choices=["CPU", "GPU"])
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-attr", action="store_true", help="跳过车辆属性模型")
    parser.add_argument("--stage",
                        choices=["detect", "attr", "plate", "correct", "embed", "all"],
                        default="all",
                        help="detect=PaddleDetection阶段；attr=仅重算车辆属性；"
                             "plate=车牌识别；correct=属性一致性复核；embed=FastReID阶段；"
                             "all=顺序执行各独立子进程")
    parser.add_argument("--plate-det-model", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "models/paddledetection/ch_PP-OCRv3_det_infer")
    parser.add_argument("--plate-rec-model", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "models/paddledetection/ch_PP-OCRv3_rec_infer")
    args = parser.parse_args()

    if args.stage == "all":
        # GPU 版 torch 与 paddlepaddle-gpu 的 cuDNN DLL 不能同进程共存，
        # 各阶段分别以独立子进程运行（见 detect_pass 文档字符串）。
        import subprocess
        import sys

        forward = [arg for arg in sys.argv[1:] if not arg.startswith("--stage")]
        stages = ["detect", "plate", "embed"]
        if not any(a == "--no-attr" for a in forward):
            stages.insert(1, "correct")  # 属性复核依赖 detect 产出的共识
        for stage in stages:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--stage", stage, *forward],
            )
            if result.returncode != 0:
                raise SystemExit(f"阶段 {stage} 执行失败（退出码 {result.returncode}）")
        return

    if args.stage == "detect":
        cropper = PaddleVehicleCropper(args.detector_model, args.paddle_device, args.threshold)
        attr = None if args.no_attr else VehicleAttrClassifier(args.attr_model, args.paddle_device)
        result = detect_pass(args.db, args.output_dir, cropper, attr, args.limit, args.force)
    elif args.stage == "attr":
        attr = VehicleAttrClassifier(args.attr_model, args.paddle_device)
        result = attr_pass(args.db, attr, args.limit, args.force)
    elif args.stage == "plate":
        word_dict = (Path(__file__).resolve().parent.parent
                     / "third_party/PaddleDetection/deploy/pipeline/ppvehicle/rec_word_dict.txt")
        lpr = PlateLPR(args.plate_det_model, args.plate_rec_model, word_dict, args.paddle_device)
        result = plate_pass(args.db, lpr, args.limit, args.force)
        inst_result = instance_plate_pass(args.db, lpr, args.limit)
        result["instances"] = inst_result
    elif args.stage == "correct":
        # 复核用更低阈值（0.2）重检原图，让被 0.35 挡掉的主体车辆有机会出现。
        detector = PaddleVehicleCropper(args.detector_model, args.paddle_device, 0.2)
        attr = VehicleAttrClassifier(args.attr_model, args.paddle_device)
        result = correct_pass(args.db, detector, attr, args.limit)
    else:
        extractor = FastReIDExtractor(args.fastreid_config, args.fastreid_weights, args.device)
        result = embed_pass(args.db, extractor, args.limit, args.force)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
